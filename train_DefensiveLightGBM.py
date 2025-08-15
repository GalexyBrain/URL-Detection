#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train multiple *adversarial detectors* on features_adversarial_defense_dataset.csv
and keep ONLY the best model (by validation AUROC, pos class = is_adv==1).

Key fixes vs prior version
--------------------------
- TARGET = is_adv (1=adversarial, 0=natural)
- Stratified, group-aware split by parent_idx so each split has both classes
  and row volumes stay roughly in-ratio even with many ADV duplicates.
- Robust metrics + calibrator (skip when single-class edge cases occur).
"""

from __future__ import annotations
from pathlib import Path
import json, math, warnings, os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# LightGBM
import lightgbm as lgb

# Try XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# sklearn bits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# --------------------- CONFIG ---------------------
CFG = {
    "data_csv": "features_adversarial_defense_dataset.csv",

    # Where to save (mirror multi-model layout)
    "models_dir": "models",
    "results_dir": "results_defense_features",
    "model_dir_name": "Defense-LGBM",     # keep path stable for downstream code

    "random_state": 42,

    # Target split ratios (applied per-group *stratified* by has-ADV)
    "train_ratio": 0.70,
    "val_ratio":   0.10,
    "test_ratio":  0.20,

    # LightGBM
    "lgbm": {
        "num_boost_round": 4000,
        "early_stopping_rounds": 200,
        "learning_rate": 0.03,
        "num_leaves": 128,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbose_eval": 50,
    },

    # XGBoost (sklearn API)
    "xgb": {
        "n_estimators": 10000,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "early_stopping_rounds": 200,
        "verbose_eval": 50,
    },

    # Logistic Regression (saga)
    "logreg": {
        "C": 1.0,
        "max_iter": 4000,
        "n_jobs": -1,
        "class_weight": "balanced",
        "solver": "saga",
        "penalty": "l2",
    },

    # eval/pred chunking (keep memory calm on huge sets)
    "pred_chunk_rows": 100_000,
}

# Non-feature columns present in defense dataset
NON_FEATURE_COLS = {
    "orig_label", "is_adv", "attack_type", "source_model", "parent_idx"
}

# --------------------- HELPERS ---------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def detect_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    assert feats, "No numeric feature columns detected."
    return feats

def _safe_ap(y_true, y_score) -> float:
    try:
        if len(np.unique(y_true)) <= 1:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")

def metrics_block(y_true, scores, thr=0.5):
    """scores: proba for class-1 (adv) or a margin; ROC/AP use raw scores."""
    scores = np.asarray(scores)
    probs_like = scores if (0.0 <= scores.min() <= scores.max() <= 1.0) else 1.0 / (1.0 + np.exp(-scores))
    y_pred = (probs_like >= thr).astype(np.int32)
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "avg_precision": _safe_ap(y_true, scores),
    }

def _sigmoidify(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    return scores if (0.0 <= scores.min() <= scores.max() <= 1.0) else 1.0 / (1.0 + np.exp(-scores))

def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    if len(y_true) == 0: return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, labels=[0,1])
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def plot_roc(y_true, y_score, out_path: Path, title: str):
    if len(y_true) == 0 or len(np.unique(y_true)) < 2: return
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title(title)
        fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    except Exception:
        pass

def chunked_predict_scores(est, X: np.ndarray, chunk: int, desc: str):
    """Return class-1 scores (p_adv or margin)."""
    N = X.shape[0]
    out = np.empty((N,), dtype=np.float32)

    if isinstance(est, lgb.Booster):
        num_iter = est.best_iteration if getattr(est, "best_iteration", None) else est.current_iteration()
        for s in tqdm(range(0, N, chunk), desc=desc, unit="rows", leave=False, dynamic_ncols=True):
            e = min(N, s + chunk)
            out[s:e] = est.predict(X[s:e], num_iteration=num_iter)
    else:
        has_proba = hasattr(est, "predict_proba")
        has_dec = hasattr(est, "decision_function")
        for s in tqdm(range(0, N, chunk), desc=desc, unit="rows", leave=False, dynamic_ncols=True):
            e = min(N, s + chunk)
            if has_proba:
                p = est.predict_proba(X[s:e])
                if isinstance(p, list): p = p[0]
                if getattr(p, "ndim", 1) == 2 and p.shape[1] == 2:
                    out[s:e] = p[:, 1]
                else:
                    out[s:e] = np.asarray(p).ravel()
            elif has_dec:
                out[s:e] = np.asarray(est.decision_function(X[s:e])).ravel()
            else:
                out[s:e] = np.asarray(est.predict(X[s:e])).astype(float).ravel()

    return out

def write_reports(result_dir: Path, tag: str, y, scores, feature_importances=None, feature_cols=None, model_name=None):
    probs_like = _sigmoidify(scores)
    y_pred = (probs_like >= 0.5).astype(np.int32)

    mb = metrics_block(y, scores)
    save_json(mb, result_dir / f"metrics_{tag}.json")
    (result_dir / f"classification_report_{tag}.txt").write_text(
        classification_report(y, y_pred, labels=[0,1], digits=4, zero_division=0), encoding="utf-8"
    )
    plot_confusion(y, y_pred, result_dir / f"confusion_matrix_{tag}.png", f"Confusion Matrix ({tag}, pos=adv)")
    plot_roc(y, scores, result_dir / f"roc_curve_{tag}.png", f"ROC Curve ({tag}, pos=adv)")

    if feature_importances is not None and feature_cols is not None:
        imp_df = pd.DataFrame({"feature": feature_cols, "importance": feature_importances}).sort_values("importance", ascending=False)
        imp_df.to_csv(result_dir / "feature_importance.csv", index=False)
        topk = min(30, len(feature_cols))
        fig, ax = plt.subplots(figsize=(8, max(6, topk * 0.25)))
        ax.barh(imp_df.head(topk)["feature"][::-1], imp_df.head(topk)["importance"][::-1])
        ttl = f"Top Feature Importances ({model_name})" if model_name else "Top Feature Importances"
        ax.set_title(ttl)
        fig.tight_layout(); fig.savefig(result_dir / "feature_importance_top30.png", dpi=200); plt.close(fig)

    return mb

# --------------------- STRATIFIED GROUP SPLIT ---------------------
def stratified_group_split(groups: np.ndarray, y: np.ndarray, *, train_ratio: float, val_ratio: float, test_ratio: float, random_state: int):
    """
    Split by unique groups (parent_idx) but STRATIFY by group label:
      group_label = 1 if any row in the group has is_adv==1, else 0.
    Ensures each split has both classes (when possible) and keeps volumes sane.
    """
    assert math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6)
    df = pd.DataFrame({"g": groups, "y": y})
    grp = df.groupby("g")["y"].max()  # 1 if any ADV in the group
    pos_groups = grp.index[grp.values == 1].to_numpy()
    neg_groups = grp.index[grp.values == 0].to_numpy()

    rng = np.random.RandomState(random_state)

    def _alloc(gs):
        gs = gs.copy()
        rng.shuffle(gs)
        n = len(gs)
        n_tr = int(round(train_ratio * n))
        n_val = int(round(val_ratio * n))
        n_tr = min(n_tr, n)  # guard
        n_val = min(n_val, n - n_tr)
        n_te = n - n_tr - n_val
        return set(gs[:n_tr]), set(gs[n_tr:n_tr+n_val]), set(gs[n_tr+n_val:])

    # Try a few times to guarantee both classes in every split
    for attempt in range(32):
        tr_pos, val_pos, te_pos = _alloc(pos_groups)
        tr_neg, val_neg, te_neg = _alloc(neg_groups)

        tr_g = tr_pos | tr_neg
        val_g = val_pos | val_neg
        te_g  = te_pos | te_neg

        tr_mask = np.isin(groups, list(tr_g))
        val_mask = np.isin(groups, list(val_g))
        te_mask  = np.isin(groups, list(te_g))

        tr_idx = np.flatnonzero(tr_mask)
        val_idx = np.flatnonzero(val_mask)
        te_idx  = np.flatnonzero(te_mask)

        # Check class presence
        ok = True
        for split_name, idx in [("TRAIN", tr_idx), ("VAL", val_idx), ("TEST", te_idx)]:
            uy = np.unique(y[idx])
            if uy.size < 2:
                ok = False
                break
        if ok:
            return np.sort(tr_idx), np.sort(val_idx), np.sort(te_idx)

    # Fallback (should be rare): just return the last attempt
    return np.sort(tr_idx), np.sort(val_idx), np.sort(te_idx)

# --------------------- TRAINERS ---------------------
def train_lgbm(X_tr, y_tr, X_val, y_val, feature_cols, rnd):
    p = CFG["lgbm"]
    pos = max(1, int(y_tr.sum()))
    neg = max(1, int(len(y_tr) - pos))
    spw = neg / pos

    dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_cols, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": p["learning_rate"],
        "num_leaves": p["num_leaves"],
        "min_data_in_leaf": p["min_data_in_leaf"],
        "feature_fraction": p["feature_fraction"],
        "bagging_fraction": p["bagging_fraction"],
        "bagging_freq": p["bagging_freq"],
        "verbosity": -1,
        "scale_pos_weight": spw,
        "deterministic": True,
        "seed": rnd,
        "feature_fraction_seed": rnd + 1,
        "bagging_seed": rnd + 2,
        "data_random_seed": rnd + 3,
    }
    cbs = [
        lgb.early_stopping(p["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(p["verbose_eval"]),
    ]
    used_device = "cpu"
    try:
        booster = lgb.train({**params, "device_type": "gpu"}, dtrain,
                            num_boost_round=p["num_boost_round"],
                            valid_sets=[dtrain, dval], valid_names=["train","val"], callbacks=cbs)
        used_device = "gpu"
    except Exception as e1:
        try:
            booster = lgb.train({**params, "device": "gpu"}, dtrain,
                                num_boost_round=p["num_boost_round"],
                                valid_sets=[dtrain, dval], valid_names=["train","val"], callbacks=cbs)
            used_device = "gpu"
        except Exception as e2:
            print(f"[WARN] LGBM GPU unavailable ({e1}); legacy key failed ({e2}). Using CPU.")
            booster = lgb.train({**params, "device_type": "cpu"}, dtrain,
                                num_boost_round=p["num_boost_round"],
                                valid_sets=[dtrain, dval], valid_names=["train","val"], callbacks=cbs)
            used_device = "cpu"

    best_iter = booster.best_iteration if getattr(booster, "best_iteration", None) else booster.current_iteration()
    meta = {"name": "LightGBM", "device": used_device, "best_iteration": int(best_iter), "params": params}
    return booster, meta

def train_xgb(X_tr, y_tr, X_val, y_val, rnd):
    if not HAS_XGB:
        raise RuntimeError("XGBoost not installed")
    p = CFG["xgb"]
    pos = max(1, int(y_tr.sum()))
    neg = max(1, int(len(y_tr) - pos))
    spw = neg / pos

    # default to CPU hist; try GPU if explicitly requested via env
    tree_method = "gpu_hist" if os.environ.get("XGB_USE_GPU", "0") == "1" else "hist"
    clf = xgb.XGBClassifier(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["reg_lambda"],
        reg_alpha=p["reg_alpha"],
        objective="binary:logistic",
        tree_method=tree_method,
        random_state=rnd,
        n_jobs=-1,
        eval_metric=["auc","logloss"],
        scale_pos_weight=spw,
        verbosity=0,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
        early_stopping_rounds=p["early_stopping_rounds"]
    )
    meta = {"name": "XGBoost", "device": tree_method, "best_iteration": int(getattr(clf, "best_iteration_", clf.get_booster().best_ntree_limit)), "params": clf.get_params()}
    return clf, meta

def train_logreg(X_tr, y_tr, rnd):
    p = CFG["logreg"]
    clf = LogisticRegression(
        C=p["C"], max_iter=p["max_iter"], n_jobs=p["n_jobs"], solver=p["solver"],
        class_weight=p["class_weight"], penalty=p["penalty"], random_state=rnd
    )
    clf.fit(X_tr, y_tr)
    meta = {"name": "LogisticRegression", "device": "cpu", "params": clf.get_params()}
    return clf, meta

# --------------------- MAIN ---------------------
def main():
    # Folders aligned with your multi-model layout
    models_root  = ensure_dir(Path(CFG["models_dir"]))
    results_root = ensure_dir(Path(CFG["results_dir"]))
    global_dir   = ensure_dir(models_root / "_global")
    model_dir    = ensure_dir(models_root / CFG["model_dir_name"])
    result_dir   = ensure_dir(results_root / CFG["model_dir_name"])

    # ---------- Load CSV with compact dtypes ----------
    df = pd.read_csv(
        CFG["data_csv"],
        dtype={
            "orig_label": "int8",
            "is_adv": "int8",
            "attack_type": "category",
            "source_model": "category",
            "parent_idx": "int32",
        }
    )

    # Downcast numeric features + NaN/Inf hygiene
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.isna().any():
            med = float(np.nanmedian(s.values)) if not np.isnan(np.nanmedian(s.values)) else 0.0
            df[c] = s.fillna(med).astype("float32")
        else:
            df[c] = s.astype("float32")

    # Detect numeric features from this CSV
    feature_cols = detect_features(df)

    # Reuse GLOBAL schema order if it exists; append any extras at the end (rare)
    global_schema = global_dir / "feature_columns.json"
    if global_schema.exists():
        try:
            saved = json.loads(global_schema.read_text(encoding="utf-8"))
            global_feats = saved["feature_columns"] if isinstance(saved, dict) else list(saved)
            ordered = [c for c in global_feats if c in feature_cols]      # keep order
            extras  = [c for c in feature_cols if c not in ordered]       # append any new
            feature_cols = ordered + extras
        except Exception:
            pass

    print(f"[INFO] Using {len(feature_cols)} feature columns.")
    save_json({"defense_feature_columns": feature_cols}, model_dir / "feature_columns.json")
    if not global_schema.exists():
        save_json({"feature_columns": feature_cols}, global_schema)

    # ---------------- TARGET = is_adv (1=adversarial, 0=natural) ----------------
    if "is_adv" not in df.columns:
        raise AssertionError("Column 'is_adv' is missing from the defense dataset.")
    y = df["is_adv"].astype("int8").to_numpy()
    groups = df["parent_idx"].to_numpy()
    X = df[feature_cols].to_numpy(dtype=np.float32)

    # ---------- STRATIFIED GROUP SPLIT ----------
    tr_idx, val_idx, te_idx = stratified_group_split(
        groups, y,
        train_ratio=CFG["train_ratio"], val_ratio=CFG["val_ratio"], test_ratio=CFG["test_ratio"],
        random_state=CFG["random_state"]
    )
    save_json(
        {"train_idx": tr_idx.tolist(), "val_idx": val_idx.tolist(), "test_idx": te_idx.tolist()},
        model_dir / "defense_split_indices.json"
    )

    def _stats(name, idx):
        ys = y[idx]
        n = ys.size
        pos = int(ys.sum())
        neg = int(n - pos)
        print(f"[SPLIT] {name:5s}: rows={n:,}  pos={pos:,}  neg={neg:,}  pos_rate={pos/max(1,n):.3f}")

    print(f"[INFO] Defense split sizes (rows): train={len(tr_idx):,}  val={len(val_idx):,}  test={len(te_idx):,}")
    _stats("TRAIN", tr_idx)
    _stats("VAL",   val_idx)
    _stats("TEST",  te_idx)

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te,  y_te  = X[te_idx],  y[te_idx]

    # ----------------- Train candidates -----------------
    rnd = int(CFG["random_state"])
    candidates = []

    # LightGBM
    try:
        lgbm_model, lgbm_meta = train_lgbm(X_tr, y_tr, X_val, y_val, feature_cols, rnd)
        cand_scores = chunked_predict_scores(lgbm_model, X_val, CFG["pred_chunk_rows"], "VAL (LGBM)")
        cand_metrics = metrics_block(y_val, cand_scores)
        candidates.append({"name": "LightGBM", "model": lgbm_model, "meta": lgbm_meta, "val_metrics": cand_metrics})
        print(f"[VAL] LightGBM AUROC={cand_metrics['auroc']:.4f}  AP={cand_metrics['avg_precision']:.4f}  F1={cand_metrics['f1']:.4f}")
    except Exception as e:
        print(f"[WARN] LightGBM training failed: {e}")

    # XGBoost
    if HAS_XGB:
        try:
            xgb_model, xgb_meta = train_xgb(X_tr, y_tr, X_val, y_val, rnd)
            cand_scores = chunked_predict_scores(xgb_model, X_val, CFG["pred_chunk_rows"], "VAL (XGB)")
            cand_metrics = metrics_block(y_val, cand_scores)
            candidates.append({"name": "XGBoost", "model": xgb_model, "meta": xgb_meta, "val_metrics": cand_metrics})
            print(f"[VAL] XGBoost  AUROC={cand_metrics['auroc']:.4f}  AP={cand_metrics['avg_precision']:.4f}  F1={cand_metrics['f1']:.4f}")
        except Exception as e:
            print(f"[WARN] XGBoost training failed: {e}")
    else:
        print("[INFO] XGBoost not installed; skipping.")

    # Logistic Regression (saga)
    try:
        lr_model, lr_meta = train_logreg(X_tr, y_tr, rnd)
        cand_scores = chunked_predict_scores(lr_model, X_val, CFG["pred_chunk_rows"], "VAL (LogReg)")
        cand_metrics = metrics_block(y_val, cand_scores)
        candidates.append({"name": "LogisticRegression", "model": lr_model, "meta": lr_meta, "val_metrics": cand_metrics})
        print(f"[VAL] LogReg   AUROC={cand_metrics['auroc']:.4f}  AP={cand_metrics['avg_precision']:.4f}  F1={cand_metrics['f1']:.4f}")
    except Exception as e:
        print(f"[WARN] LogisticRegression training failed: {e}")

    assert candidates, "No candidate models were trained successfully."

    # ----------------- Pick the best by VAL AUROC (AP then F1 tie-breakers) -----------------
    def _key(c):
        m = c["val_metrics"]
        return (m["auroc"], m["avg_precision"], m["f1"])
    best = max(candidates, key=_key)

    # Save candidate summary (VAL)
    cand_summary = [
        {"name": c["name"], **c["val_metrics"], "meta": {"device": c["meta"].get("device",""), "best_iteration": c["meta"].get("best_iteration", None)}}
        for c in candidates
    ]
    ensure_dir(result_dir)
    save_json(cand_summary, result_dir / "candidate_summary.json")
    print(f"\n[SELECT] Best adversarial detector by VAL: {best['name']} (AUROC={best['val_metrics']['auroc']:.4f}, AP={best['val_metrics']['avg_precision']:.4f})")

    winner = best["model"]; winner_name = best["name"]; winner_meta = best["meta"]

    # ----------------- Evaluate winner on VAL & TEST -----------------
    save_json({"winner": winner_name, "meta": winner_meta}, result_dir / "params_used.json")

    val_scores = chunked_predict_scores(winner, X_val, CFG["pred_chunk_rows"], f"Predict VAL ({winner_name})")
    write_reports(result_dir, "val", y_val, val_scores)

    test_scores = chunked_predict_scores(winner, X_te, CFG["pred_chunk_rows"], f"Predict TEST ({winner_name})")
    mb_test = write_reports(result_dir, "test", y_te, test_scores)

    # Per-subgroup on TEST (NAT vs ADV) == simply class split
    for flag, name in [(0, "NAT"), (1, "ADV")]:
        mask = (y_te == flag)
        if mask.sum() == 0:
            continue
        try:
            au = roc_auc_score(y_te[mask], test_scores[mask]) if len(np.unique(y_te[mask])) > 1 else float("nan")
        except Exception:
            au = float("nan")
        ap = _safe_ap(y_te[mask], test_scores[mask])
        print(f"[TEST subgroup {name}] n={mask.sum():,}  AUROC={au:.4f}  AP={ap:.4f}")

    # ----------------- Fit & save isotonic calibrator on VAL (winner) -----------------
    try:
        if len(np.unique(y_val)) == 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            val_probs_like = _sigmoidify(val_scores)
            iso.fit(val_probs_like, y_val.astype(int))
            joblib.dump(iso, model_dir / "calibrator.joblib")
            print(f"[OK] Saved isotonic calibrator -> {model_dir / 'calibrator.joblib'}")
        else:
            print("[WARN] Skipping calibration: VAL split is single-class.")
    except Exception as e:
        print(f"[WARN] Skipped calibration: {e}")

    # ----------------- Save ONLY the winner model & artifacts -----------------
    save_json({"defense_feature_columns": feature_cols}, model_dir / "feature_columns.json")
    save_json({"train_idx": tr_idx.tolist(), "val_idx": val_idx.tolist(), "test_idx": te_idx.tolist()},
              model_dir / "defense_split_indices.json")

    joblib.dump(winner, model_dir / "model.joblib")

    try:
        if isinstance(winner, lgb.Booster):
            winner.save_model(str(model_dir / "model.txt"))
        else:
            (model_dir / "model.txt").write_text(
                f"{winner_name} parameters:\n{json.dumps(winner_meta.get('params', {}), indent=2)}\n",
                encoding="utf-8"
            )
    except Exception as e:
        print(f"[WARN] Could not save model.txt: {e}")

    best_info = {
        "winner_name": winner_name,
        "val_metrics": best["val_metrics"],
        "test_metrics": mb_test,
        "meta": winner_meta,
        "target": "is_adv (1=adversarial, 0=natural)"
    }
    save_json(best_info, model_dir / "best_model_info.json")
    print(f"[OK] Saved WINNER adversarial detector -> {model_dir}  ({winner_name})")
    print(f"[DONE] Winner evaluated and artifacts saved. ({winner_name})")

if __name__ == "__main__":
    main()

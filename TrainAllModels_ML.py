"""
Multi-model trainer with diagnostics (Python 3.13+), bug-fixed.

Key fixes:
- Fit StandardScaler on TRAIN only (no leakage).
- Predictions: use 0.5 cutoff only for probabilities; use 0.0 for decision_function margins.
- Brier score = MSE(proba, y); also report cross-entropy as log_loss_ce.
- Robust KS implementation.
"""

from pathlib import Path
from datetime import datetime, timezone
import json, warnings, math
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    log_loss, PrecisionRecallDisplay, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import IsolationForest

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CONFIG = {
    "data_csv": "features_extracted.csv",
    "label_col": "label",
    "models_dir": "models",
    "results_dir": "results",
    "random_state": 42,
    "train_ratio": 0.70,
    "val_ratio":   0.10,
    "test_ratio":  0.20,
    "drift_bins": 10,
    "ece_bins": 15,
    "isoforest_contamination": 0.02
}

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")

def prob_vector(est, X):
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        if isinstance(p, list): p = p[0]
        if getattr(p, "ndim", 1) == 2 and p.shape[1] == 2:
            return p[:, 1]
    return None

def score_vector(est, X):
    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        return np.asarray(s).ravel()
    return None

def unwrap_calibrated(est):
    if isinstance(est, CalibratedClassifierCV) and hasattr(est, "calibrated_classifiers_") and est.calibrated_classifiers_:
        inner = est.calibrated_classifiers_[0]
        if hasattr(inner, "estimator") and inner.estimator is not None: return inner.estimator
        if hasattr(inner, "base_estimator") and inner.base_estimator is not None: return inner.base_estimator
    return est

def feature_importance(estimator, feature_names):
    src = unwrap_calibrated(estimator)
    if hasattr(src, "feature_importances_"):
        return feature_names, np.asarray(src.feature_importances_, dtype=float)
    if hasattr(src, "coef_"):
        coefs = np.asarray(src.coef_, dtype=float)
        vals = np.abs(coefs[0]) if (coefs.ndim == 2 and coefs.shape[0] == 1) else np.mean(np.abs(coefs), axis=0)
        return feature_names, vals
    return None

# ---- Calibration (ECE) ----
def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).clip(1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; total = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & ((y_prob < hi) if i < n_bins-1 else (y_prob <= hi))
        if not np.any(mask): continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask)/total) * abs(acc - conf)
    return float(ece)

# ---- Threshold sweep (for probabilities) ----
def sweep_thresholds(y_true, y_prob, thresholds=None):
    if thresholds is None: thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bal  = balanced_accuracy_score(y_true, y_pred)
        j    = rec + spec - 1.0
        rows.append({"threshold": float(t), "precision": float(prec), "recall": float(rec),
                     "f1": float(f1), "specificity": float(spec), "balanced_accuracy": float(bal),
                     "youden_j": float(j)})
    df = pd.DataFrame(rows)
    t_f1 = float(df.loc[df["f1"].idxmax(), "threshold"]) if len(df) else 0.5
    t_j  = float(df.loc[df["youden_j"].idxmax(), "threshold"]) if len(df) else 0.5
    return df, t_f1, t_j

# ---- Data quality ----
def data_quality_report(df: pd.DataFrame, label_col: str):
    num_rows = int(df.shape[0])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols + [label_col]]
    miss = df.isna().sum().to_dict()
    infs = {c: int(np.isinf(df[c]).sum()) for c in numeric_cols}
    dup_rows = int(df.duplicated().sum())
    const_cols = [c for c in numeric_cols if df[c].nunique(dropna=False) <= 1]
    near_const = [c for c in numeric_cols if df[c].value_counts(normalize=True, dropna=False).iloc[0] >= 0.99]
    return {
        "rows": num_rows,
        "non_numeric_excluded": non_numeric,
        "missing_counts": miss,
        "infinite_counts": infs,
        "duplicate_rows": dup_rows,
        "constant_numeric_columns": const_cols,
        "near_constant_numeric_columns_(>=99%_same_value)": near_const
    }

# ---- Drift: robust KS & PSI on RAW features ----
def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    v = np.unique(np.concatenate([a, b]))
    # CDFs at bin edges
    cdfa = np.searchsorted(np.sort(a), v, side="right") / a.size
    cdfb = np.searchsorted(np.sort(b), v, side="right") / b.size
    return float(np.max(np.abs(cdfa - cdfb)))

def psi(train_col: np.ndarray, test_col: np.ndarray, bins=10) -> float:
    eps = 1e-8
    qs = np.linspace(0, 1, bins+1)
    cuts = np.unique(np.quantile(train_col, qs))
    if cuts.size < 2: return 0.0
    def frac(x):
        idx = np.searchsorted(cuts, x, side="right") - 1
        idx = np.clip(idx, 0, cuts.size-2)
        counts = np.bincount(idx, minlength=cuts.size-1).astype(float)
        return counts / max(1, x.size)
    p = np.clip(frac(train_col), eps, None)
    q = np.clip(frac(test_col),  eps, None)
    return float(np.sum((q - p) * np.log(q / p)))

def drift_report(Xtr_raw, Xte_raw, feat_names, bins=10):
    rows = []
    for j, name in enumerate(feat_names):
        col_tr = Xtr_raw[:, j]; col_te = Xte_raw[:, j]
        rows.append({
            "feature": name,
            "ks": ks_statistic(col_tr, col_te),
            "psi": psi(col_tr, col_te, bins=bins),
            "train_mean": float(np.mean(col_tr)),
            "test_mean": float(np.mean(col_te))
        })
    return pd.DataFrame(rows).sort_values(["psi", "ks"], ascending=False)

# ---- Metrics pack ----
def rich_metrics(y_true, y_score, y_pred, proba=None):
    out = {}
    out["samples"] = int(len(y_true))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    try: out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception: out["roc_auc"] = None
    try: out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception: out["pr_auc"] = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out["specificity"] = float(tn / (tn + fp) if (tn + fp) else 0.0)
    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    if proba is not None:
        out["brier"] = float(np.mean((proba - y_true)**2))        # TRUE Brier
        out["log_loss_ce"] = float(log_loss(y_true, proba, labels=[0,1]))  # Cross-entropy
        out["ece"] = float(expected_calibration_error(y_true, proba, n_bins=CONFIG["ece_bins"]))
    else:
        out["brier"] = None; out["log_loss_ce"] = None; out["ece"] = None
    return out

def build_models(rs: int):
    models = {
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=rs,
                                                  max_iter=2000, class_weight="balanced"),
        "Calibrated LinearSVC": CalibratedClassifierCV(
            estimator=LinearSVC(dual="auto", random_state=rs, max_iter=5000), cv=3),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=rs, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1,
                                                class_weight="balanced", max_depth=None),
        "AdaBoost": AdaBoostClassifier(random_state=rs, n_estimators=400, learning_rate=0.5),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            eval_metric="logloss", random_state=rs, n_jobs=-1,
            n_estimators=4000, learning_rate=0.02,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", early_stopping_rounds=200
        )
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            random_state=rs, n_jobs=-1, objective="binary",
            n_estimators=3000, learning_rate=0.02,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8
        )
    return models

def main():
    cfg = CONFIG
    data_csv = Path(cfg["data_csv"])
    models_root = ensure_dir(Path(cfg["models_dir"]))
    results_root = ensure_dir(Path(cfg["results_dir"]))
    global_dir = ensure_dir(models_root / "_global")
    global_res = ensure_dir(results_root / "_global")
    rs = int(cfg["random_state"])

    print(f"[INFO] Loading {data_csv} ...")
    df = pd.read_csv(data_csv)
    assert cfg["label_col"] in df.columns, f"Missing label column: {cfg['label_col']}"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cfg["label_col"] not in numeric_cols:
        raise ValueError(f"Label column '{cfg['label_col']}' must be numeric (0/1).")
    feature_cols = [c for c in numeric_cols if c != cfg["label_col"]]
    dropped = [c for c in df.columns if c not in feature_cols + [cfg["label_col"]]]
    if dropped: print(f"[INFO] Excluding non-numeric columns from X: {dropped}")

    dq = data_quality_report(df[feature_cols + [cfg["label_col"]]], cfg["label_col"])
    save_json(dq, global_res / "data_quality.json")

    X_raw_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df[cfg["label_col"]].to_numpy(dtype=int)
    assert set(np.unique(y_all)) <= {0,1}, "Binary labels {0,1} expected."

    # --- Split on RAW first (avoid leakage) ---
    X_train_raw, X_tmp_raw, y_train, y_tmp = train_test_split(
        X_raw_all, y_all, test_size=(1.0 - cfg["train_ratio"]), random_state=rs, stratify=y_all)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_tmp_raw, y_tmp, test_size=(cfg["test_ratio"]/(cfg["val_ratio"]+cfg["test_ratio"])),
        random_state=rs, stratify=y_tmp)

    # --- Fit scaler on TRAIN only ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val   = scaler.transform(X_val_raw)
    X_test  = scaler.transform(X_test_raw)

    joblib.dump(scaler, global_dir / "scaler.joblib")
    save_json({"feature_columns": feature_cols}, global_dir / "feature_columns.json")
    print(f"[INFO] Saved global scaler & feature columns.")
    print(f"[INFO] Shapes -> train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")

    # Class balance
    def counts(v):
        u, c = np.unique(v, return_counts=True); d = {int(k): int(vv) for k, vv in zip(u, c)}
        return {"counts": d, "ratio_pos": float(d.get(1,0)/max(1,sum(d.values())))}
    label_stats = {"train": counts(y_train), "val": counts(y_val), "test": counts(y_test)}
    save_json(label_stats, global_res / "label_stats.json")

    # Drift on RAW distributions
    drift_df = drift_report(X_train_raw, X_test_raw, np.array(feature_cols), bins=cfg["drift_bins"])
    drift_df.to_csv(global_res / "drift_report.csv", index=False)
    save_json({
        "top_by_psi": drift_df.head(20).to_dict(orient="records"),
        "psi_guidance": "PSI <0.1 stable; 0.1–0.25 moderate; >0.25 major shift",
        "ks_guidance": "KS <0.1 small; 0.1–0.2 moderate; >0.2 large"
    }, global_res / "drift_report.json")

    # Outliers on scaled space (fit on train)
    iso = IsolationForest(n_estimators=200, random_state=rs,
                          contamination=cfg["isoforest_contamination"])
    iso.fit(X_train)
    def outlier_frac(Xp): return float(np.mean(iso.predict(Xp) == -1))
    save_json({
        "contamination_cfg": cfg["isoforest_contamination"],
        "train_flagged_fraction": outlier_frac(X_train),
        "val_flagged_fraction": outlier_frac(X_val),
        "test_flagged_fraction": outlier_frac(X_test)
    }, global_res / "outliers.json")

    cls_w = compute_class_weight("balanced", classes=np.array([0,1]), y=y_train)
    save_json({"class_weights_(0,1)": [float(cls_w[0]), float(cls_w[1])]}, global_res / "class_weights.json")
    print(f"[INFO] Class weights (0,1): {cls_w}")

    models = build_models(rs)

    for name, model in models.items():
        print(f"\n===== Training: {name} =====")
        mname = safe_name(name)
        mdir = ensure_dir(models_root / mname)
        rdir = ensure_dir(results_root / mname)

        if (lgb is not None) and isinstance(model, lgb.LGBMClassifier):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        elif (xgb is not None) and isinstance(model, xgb.XGBClassifier):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        joblib.dump(model, mdir / "model.joblib")

        # ---- pack metrics per split ----
        def pack(split_name, Xs, ys):
            proba = prob_vector(model, Xs)
            score = score_vector(model, Xs)
            if proba is not None:
                y_pred = (proba >= 0.5).astype(int)
                used_score = proba
            elif score is not None:
                y_pred = (score >= 0.0).astype(int)  # margin threshold
                used_score = score
            else:
                y_pred = model.predict(Xs)
                used_score = y_pred
            m = rich_metrics(ys, used_score, y_pred, proba)
            m["split"] = split_name
            m["timestamp"] = datetime.now(timezone.utc).isoformat()
            return m, used_score, proba, y_pred

        m_train, s_train, p_train, yhat_train = pack("train", X_train, y_train)
        m_val,   s_val,   p_val,   yhat_val   = pack("val",   X_val,   y_val)
        m_test,  s_test,  p_test,  yhat_test  = pack("test",  X_test,  y_test)

        save_json(m_train, rdir / "metrics_train.json")
        save_json(m_val,   rdir / "metrics_val.json")
        save_json(m_test,  rdir / "metrics_test.json")

        # ---- Threshold selection on VAL (probability models only) ----
        if p_val is not None:
            sweep_df, t_f1, t_j = sweep_thresholds(y_val, p_val)
            sweep_df.to_csv(rdir / "thresholds_val.csv", index=False)
            chosen_t = float(t_f1)
            yhat_opt = (p_test >= chosen_t).astype(int) if p_test is not None else yhat_test
            m_test_opt = rich_metrics(y_test, (p_test if p_test is not None else s_test), yhat_opt, p_test)
            m_test_opt["threshold_used"] = chosen_t
            save_json(m_test_opt, rdir / "metrics_test_at_opt_threshold.json")
        else:
            save_text("No predict_proba => threshold sweep skipped.", rdir / "thresholds_val.txt")

        # Confusion (numbers + plot)
        tn, fp, fn, tp = confusion_matrix(y_test, yhat_test, labels=[0,1]).ravel()
        save_json({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, rdir / "confusion_matrix.json")
        fig, ax = plt.subplots(figsize=(5.5,5))
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(y_test, yhat_test, ax=ax)
        ax.set_title(f"Confusion Matrix: {name}")
        fig.tight_layout(); fig.savefig(rdir / "confusion_matrix.png", dpi=200); plt.close(fig)

        try:
            fig, ax = plt.subplots(figsize=(6,5))
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
            ax.set_title(f"ROC: {name}"); fig.tight_layout()
            fig.savefig(rdir / "roc_curve.png", dpi=200); plt.close(fig)
        except Exception:
            pass

        try:
            fig, ax = plt.subplots(figsize=(6,5))
            PrecisionRecallDisplay.from_predictions(y_test, (p_test if p_test is not None else s_test), ax=ax)
            ax.set_title(f"Precision-Recall: {name}"); fig.tight_layout()
            fig.savefig(rdir / "pr_curve.png", dpi=200); plt.close(fig)
        except Exception:
            pass

        if p_test is not None:
            try:
                fig, ax = plt.subplots(figsize=(6,5))
                CalibrationDisplay.from_predictions(y_test, p_test, n_bins=CONFIG["ece_bins"], ax=ax)
                ax.set_title(f"Calibration: {name}"); fig.tight_layout()
                fig.savefig(rdir / "calibration.png", dpi=200); plt.close(fig)
                save_text(
                    f"Brier: {np.mean((p_test - y_test)**2):.6f}\n"
                    f"ECE(bins={CONFIG['ece_bins']}): {expected_calibration_error(y_test, p_test, CONFIG['ece_bins']):.6f}\n"
                    f"LogLoss (cross-entropy): {log_loss(y_test, p_test, labels=[0,1]):.6f}\n",
                    rdir / "calibration_stats.txt"
                )
            except Exception:
                pass

        fi = feature_importance(model, np.array(feature_cols))
        if fi is not None:
            fnames, fvals = fi
            df_fi = pd.DataFrame({"feature": fnames, "importance": fvals}).sort_values("importance", ascending=False)
            df_fi.to_csv(rdir / "feature_importance.csv", index=False)
            fig_h = max(4.0, 0.36 * len(df_fi))
            fig_w = max(7.0, 0.28 * len(df_fi) if len(df_fi) > 35 else 7.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.barh(df_fi["feature"][::-1], df_fi["importance"][::-1])
            ax.set_title(f"Feature Importances: {name}")
            ax.set_xlabel("Importance")
            fig.tight_layout(); fig.savefig(rdir / "feature_importance.png", dpi=200); plt.close(fig)

        gaps = {
            "acc_gap_train_val": float(m_train["accuracy"] - m_val["accuracy"]),
            "acc_gap_train_test": float(m_train["accuracy"] - m_test["accuracy"]),
            "auc_gap_train_val": (None if (m_train["roc_auc"] is None or m_val["roc_auc"] is None)
                                  else float(m_train["roc_auc"] - m_val["roc_auc"])),
            "auc_gap_train_test": (None if (m_train["roc_auc"] is None or m_test["roc_auc"] is None)
                                   else float(m_train["roc_auc"] - m_test["roc_auc"]))
        }
        save_json(gaps, rdir / "generalization_gap.json")

        print(f"[OK] Saved model -> {mdir}")
        print(f"[OK] Saved results -> {rdir}")

    print("\n[DONE] All models trained and diagnostics saved.")

if __name__ == "__main__":
    main()

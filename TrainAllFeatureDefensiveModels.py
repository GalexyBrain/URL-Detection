#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiclass (0=benign, 1=malicious, 2=ADV) trainer for tabular data — OOM-safe.

Fixes in this version:
- LightGBM: automatic GPU(OpenCL) -> CPU fallback if no OpenCL device is present.
- DL-MLP: ReduceLROnPlateau on val macro-F1, default epochs=20, per-epoch LR logging.
- CLI: --only-dl and --only-ml to run specific model groups.
- LinearSVC key fixed to ensure prefit + validation calibration path is used.

Artifacts:
  models_base3/<ModelName>/{model.joblib|model.pt}
  results_base3/<ModelName>/{metrics.json, classification_report.txt,
                             confusion_matrix.png, roc_curve.png,
                             feature_importance.csv/.png (when supported),
                             train_log.txt}
  models_base3/_global/{scaler.joblib, feature_columns.json, split_info.json}
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, warnings, os, time, gc, traceback

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Optional GBMs
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------- CLI / CONFIG ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="features_base3_strict_cap.csv", help="Path to 3-class dataset CSV")
    p.add_argument("--label-col", default="label", help="Label column name (expects {0,1,2})")
    p.add_argument("--models-dir", default="models_base3", help="Root to save models")
    p.add_argument("--results-dir", default="results_base3", help="Root to save results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.20)

    # System / parallelism
    p.add_argument("--jobs", type=int, default=max(1, min((os.cpu_count() or 8) - 1, 8)),
                   help="Max parallel jobs/threads for sklearn where applicable")

    # OOM-safe caps (None = use full train)
    p.add_argument("--rf-max-samples", type=int, default=200_000,
                   help="Max training rows for RandomForest (stratified).")
    p.add_argument("--ada-max-samples", type=int, default=300_000,
                   help="Max training rows for AdaBoost (stratified).")
    p.add_argument("--svc-calib-max", type=int, default=300_000,
                   help="Max rows from validation used for LinearSVC calibration.")
    p.add_argument("--fallback-samples", type=int, default=150_000,
                   help="If a model OOMs, retry with this many rows (stratified).")

    # DL knobs
    p.add_argument("--epochs", type=int, default=20)  # bumped from 12 -> 20
    p.add_argument("--batch-gpu", type=int, default=8192)
    p.add_argument("--batch-cpu", type=int, default=8192)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)

    # What to run
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-dl", action="store_true", help="Run only deep learning models")
    g.add_argument("--only-ml", action="store_true", help="Run only classic ML models")

    return p.parse_args()


# ---------------- FS / logging helpers ----------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True); return path

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(model_dir: Path, *msg):
    s = f"[{now_utc()}] " + " ".join(str(m) for m in msg)
    tqdm.write(s)
    with (model_dir / "train_log.txt").open("a", encoding="utf-8") as f:
        f.write(s + "\n")

def mem_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------- Plotting ----------------
def plot_confusion(y_true, y_pred, out_path: Path, title: str, classes=(0,1,2)):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=list(classes), ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_roc_multiclass(y_true, prob_mat, out_path: Path, title: str, classes=(0,1,2)):
    y_bin = label_binarize(y_true, classes=list(classes))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plotted = False
    for i, cls in enumerate(classes):
        if prob_mat is None:
            continue
        try:
            RocCurveDisplay.from_predictions(y_bin[:, i], prob_mat[:, i], name=f"class {cls}", ax=ax)
            plotted = True
        except Exception:
            pass
    if plotted:
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------- Model utils ----------------
def score_matrix(estimator, X) -> np.ndarray | None:
    try:
        if hasattr(estimator, "predict_proba"):
            p = estimator.predict_proba(X)
            if isinstance(p, list):
                p = p[0]
            if getattr(p, "ndim", 1) == 2:
                return np.asarray(p)
    except Exception:
        pass
    try:
        if hasattr(estimator, "decision_function"):
            z = np.asarray(estimator.decision_function(X))
            if z.ndim == 2:
                e = np.exp(z - z.max(axis=1, keepdims=True))
                p = e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
                return p
    except Exception:
        pass
    return None

def unwrap_calibrated(estimator):
    if isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
            inner = estimator.calibrated_classifiers_[0]
            base = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
            if base is not None:
                return base
    return estimator

def feature_importance(estimator, feature_names):
    src = unwrap_calibrated(estimator)
    if hasattr(src, "feature_importances_"):
        vals = np.asarray(src.feature_importances_, dtype=float)
        return feature_names, vals, None
    if hasattr(src, "coef_"):
        coefs = np.asarray(src.coef_, dtype=float)  # (C,F) or (1,F)
        if coefs.ndim == 1:
            absmean = np.abs(coefs)
            return feature_names, absmean, {"per_class": coefs}
        absmean = np.mean(np.abs(coefs), axis=0)
        return feature_names, absmean, {"per_class": coefs}
    return None


# ---------------- Subsampling helpers ----------------
def stratified_subsample(X, y, cap: int, seed: int):
    if cap is None or cap <= 0 or cap >= len(y):
        return X, y, None
    rng = np.random.default_rng(seed)
    idx_by_c = [np.where(y == c)[0] for c in np.unique(y)]
    per_c = max(1, cap // len(idx_by_c))
    pick = []
    for arr in idx_by_c:
        take = min(per_c, len(arr))
        pick.append(rng.choice(arr, size=take, replace=False))
    sel = np.concatenate(pick, axis=0)
    rng.shuffle(sel)
    return X[sel], y[sel], int(sel.size)


# ---------------- Classic ML models ----------------
def build_models(rs: int, n_classes: int, n_jobs: int):
    models = {
        "Logistic_Regression": LogisticRegression(
            solver="lbfgs", multi_class="multinomial", random_state=rs, max_iter=2000
        ),
        # keep key EXACT to trigger calibrated path below
        "LinearSVC_prefit+Calib": LinearSVC(dual="auto", random_state=rs, max_iter=6000),
        "Gaussian_Naive_Bayes": GaussianNB(),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=18, random_state=rs, min_samples_leaf=2
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, random_state=rs, n_jobs=n_jobs,
            bootstrap=True, max_samples=0.2,
            max_features="sqrt", min_samples_leaf=2
        ),
        "AdaBoost": AdaBoostClassifier(
            random_state=rs, n_estimators=300, learning_rate=0.3
        ),
    }
    if xgb is not None:
        tm = "gpu_hist" if (hasattr(xgb, "config_context") and torch.cuda.is_available()) else "hist"
        models["XGBoost"] = xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            eval_metric="mlogloss", random_state=rs, n_jobs=n_jobs,
            n_estimators=10000, learning_rate=0.03,
            max_depth=8, subsample=0.8, colsample_bytree=0.8,
            tree_method=tm, early_stopping_rounds=200,
        )
    if lgb is not None:
        device = "gpu" if torch.cuda.is_available() else "cpu"
        models["LightGBM"] = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes, random_state=rs, n_jobs=n_jobs,
            n_estimators=10000, learning_rate=0.03,
            num_leaves=127, subsample=0.8, colsample_bytree=0.8,
            device_type=device
        )
    return models


# ---------------- Tabular DL models ----------------
class MLPNetMulti(nn.Module):
    def __init__(self, in_dim: int, out_classes=3, hidden=(256,128,64), p_drop=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(p_drop)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_classes)
    def forward(self, x):
        return self.head(self.backbone(x))

class FTTransformer(nn.Module):
    def __init__(self, n_features: int, out_classes=3, d_model=64, n_heads=8, n_layers=3, p_drop=0.1):
        super().__init__()
        self.n_features = n_features
        self.value_emb = nn.Linear(1, d_model, bias=False)
        self.feat_emb = nn.Embedding(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=p_drop, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model, out_classes))
    def forward(self, x):
        B, F = x.shape
        val_tok = self.value_emb(x.view(B*F,1)).view(B, F, -1)
        idx = torch.arange(F, device=x.device).view(1, F)
        feat_tok = self.feat_emb(idx).expand(B, -1, -1)
        z = self.encoder(val_tok + feat_tok)
        z = self.norm(z).mean(dim=1)
        return self.head(z)

@torch.no_grad()
def infer_probs(model, loader, device):
    model.eval()
    outs, labs = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        logits = model(xb)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        outs.append(probs);  labs.append(yb.numpy())
    return np.concatenate(outs), np.concatenate(labs)

def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])

def train_dl_model(name, model, train_loader, val_loader, test_loader, device, out_mdir: Path, out_rdir: Path,
                   epochs=20, patience=5, min_delta=1e-4,
                   use_plateau: bool=False, plateau_factor: float=0.5, plateau_patience: int=1, min_lr: float=1e-5):
    start = time.time()
    ensure_dir(out_mdir); ensure_dir(out_rdir)
    log(out_mdir, f"DL START name={name} epochs={epochs} patience={patience} device={device}")

    # Class weights for CE
    y_tr = train_loader.dataset.tensors[1].cpu().numpy()
    classes = np.unique(y_tr)
    cls_w = compute_class_weight("balanced", classes=classes, y=y_tr)
    weight = torch.tensor(cls_w, dtype=torch.float32, device=device)
    log(out_mdir, f"class_weight={cls_w.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = None
    if use_plateau:
        # NOTE: some torch builds don't accept 'verbose' kw; omit for maximum compatibility
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=plateau_factor, patience=plateau_patience,
            threshold=min_delta, min_lr=min_lr
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_score, noimp, best_state = -np.inf, 0, None
    epoch_bar = tqdm(range(1, epochs+1), desc=f"{name} epochs", leave=False)
    for ep in epoch_bar:
        model.train()
        batch_bar = tqdm(train_loader, desc=f"{name} train", leave=False)
        running = 0.0
        for xb, yb in batch_bar:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * xb.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr(opt):.2e}")

        # Validation
        val_probs, val_y = infer_probs(model, val_loader, device)
        val_pred = val_probs.argmax(axis=1)
        val_f1 = f1_score(val_y, val_pred, average="macro")
        if scheduler is not None:
            scheduler.step(val_f1)
        epoch_bar.set_postfix(val_f1=f"{val_f1:.4f}", lr=f"{current_lr(opt):.2e}")
        log(out_mdir, f"epoch={ep} lr={current_lr(opt):.6f} train_loss={(running/len(train_loader.dataset)):.6f} val_macro_f1={val_f1:.6f}")

        improved = (val_f1 > best_score + min_delta)
        if improved:
            best_score = val_f1; noimp = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= patience:
                tqdm.write(f"[{name}] Early stopping at epoch {ep}. Best val macro-F1={best_score:.4f}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, out_mdir / "model.pt")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # Test
    test_probs, test_y = infer_probs(model, test_loader, device)
    test_pred = test_probs.argmax(axis=1)

    # Metrics
    acc = accuracy_score(test_y, test_pred)
    prec = precision_score(test_y, test_pred, average="macro", zero_division=0)
    rec = recall_score(test_y, test_pred, average="macro")
    f1 = f1_score(test_y, test_pred, average="macro")
    try:
        auc_ovr = roc_auc_score(test_y, test_probs, multi_class="ovr")
    except Exception:
        auc_ovr = None

    metrics = {
        "model": name,
        "samples_evaluated": int(len(test_y)),
        "accuracy": round(float(acc), 6),
        "precision_macro": round(float(prec), 6),
        "recall_macro": round(float(rec), 6),
        "f1_macro": round(float(f1), 6),
        "auc_ovr": (round(float(auc_ovr), 6) if auc_ovr is not None else None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_time_sec": round(time.time() - start, 2),
    }
    save_json(metrics, out_rdir / "metrics.json")
    save_text(classification_report(test_y, test_pred, digits=4), out_rdir / "classification_report.txt")
    plot_confusion(test_y, test_pred, out_rdir / "confusion_matrix.png", f"Confusion Matrix: {name}", classes=(0,1,2))
    plot_roc_multiclass(test_y, test_probs, out_rdir / "roc_curve.png", f"ROC (OvR): {name}", classes=(0,1,2))

    log(out_mdir, f"DL DONE metrics={metrics}")
    tqdm.write(f"[OK] {name}: model -> {out_mdir} | results -> {out_rdir}")


# ---------------- Main ----------------
def main():
    args = parse_args()

    # Tame joblib/loky parallelism
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(Path.cwd() / ".joblib_tmp"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, args.jobs)))
    ensure_dir(Path(os.environ["JOBLIB_TEMP_FOLDER"]))

    data_csv = Path(args.data)
    assert data_csv.exists(), f"Data CSV not found: {data_csv}"

    models_root = ensure_dir(Path(args.models_dir))
    results_root = ensure_dir(Path(args.results_dir))
    global_dir = ensure_dir(models_root / "_global")
    rs = int(args.seed)

    # Load data
    print(f"[INFO] Loading {data_csv} ...")
    df = pd.read_csv(data_csv)
    assert args.label_col in df.columns, f"Missing label column: {args.label_col}"

    # Numeric-only features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    assert args.label_col in numeric_cols, f"Label column '{args.label_col}' must be numeric."
    feature_cols = [c for c in numeric_cols if c != args.label_col]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[args.label_col].to_numpy(dtype=np.int64)
    uniq = set(np.unique(y).tolist())
    assert uniq <= {0,1,2}, f"Expected labels in {{0,1,2}}, got: {sorted(uniq)}"
    n_classes = 3

    # Global scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, global_dir / "scaler.joblib")
    save_json({"feature_columns": feature_cols}, global_dir / "feature_columns.json")

    # Shared split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1.0 - args.train_ratio), random_state=rs, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(args.test_ratio / (args.val_ratio + args.test_ratio)),
        random_state=rs, stratify=y_tmp
    )
    save_json(
        {"shapes": {
            "train": [int(X_train.shape[0]), int(X_train.shape[1])],
            "val":   [int(X_val.shape[0]),   int(X_val.shape[1])],
            "test":  [int(X_test.shape[0]),  int(X_test.shape[1])] },
         "timestamp": now_utc()},
        global_dir / "split_info.json"
    )
    print(f"[INFO] Shapes -> train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"[INFO] Class weights (train): {compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)}")

    results_summary = []

    # ---------------- Classic ML ----------------
    if not args.only_dl:
        print("\n[Stage] Training classic ML models...")
        models = build_models(rs, n_classes, args.jobs)

        for name, model in tqdm(models.items(), desc="ML models"):
            mname = safe_name(name)
            mdir = ensure_dir(models_root / mname)
            rdir = ensure_dir(results_root / mname)
            (mdir / "train_log.txt").write_text("", encoding="utf-8")  # fresh log

            start = time.time()
            try:
                params_snapshot = getattr(model, "get_params", lambda: {})()
            except Exception:
                params_snapshot = {}
            log(mdir, f"START model={name} n_train={len(y_train)} n_val={len(y_val)} n_test={len(y_test)}")
            log(mdir, f"params={params_snapshot}")

            try:
                if name == "LinearSVC_prefit+Calib":
                    # Train LinearSVC on full train; calibrate on (stratified) validation subset
                    base = model
                    log(mdir, "Fitting LinearSVC on full train...")
                    base.fit(X_train, y_train)

                    Xc, yc, picked = stratified_subsample(X_val, y_val, args.svc_calib_max, seed=rs)
                    if picked:
                        log(mdir, f"Calibrating on stratified val subset size={picked}")
                    else:
                        log(mdir, f"Calibrating on full validation size={len(y_val)}")

                    calib = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
                    calib.fit(Xc, yc)
                    fitted = calib

                elif name == "Random_Forest":
                    log(mdir, "Fitting RandomForest (per-tree subsample via max_samples=0.2)...")
                    fitted = model.fit(X_train, y_train)

                elif name == "AdaBoost":
                    Xs, ys, picked = stratified_subsample(X_train, y_train, args.ada_max_samples, seed=rs)
                    if picked:
                        log(mdir, f"AdaBoost using stratified subset size={picked}")
                    else:
                        log(mdir, f"AdaBoost using full train size={len(y_train)}")
                    fitted = model.fit(Xs, ys)

                elif (xgb is not None) and isinstance(model, xgb.XGBClassifier):
                    log(mdir, f"Fitting XGBoost (tree_method={model.get_xgb_params().get('tree_method')})...")
                    fitted = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    best_it = getattr(model, "best_iteration", None)
                    if best_it is not None:
                        log(mdir, f"XGB best_iteration={best_it}")

                elif (lgb is not None) and isinstance(model, lgb.LGBMClassifier):
                    dev = model.get_params().get("device_type", "cpu")
                    log(mdir, f"Fitting LightGBM (device_type={dev}; early stopping)...")
                    try:
                        fitted = model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            eval_metric="multi_logloss",
                            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
                        )
                    except Exception as e:
                        msg = str(e)
                        if "OpenCL" in msg or "No OpenCL device" in msg:
                            log(mdir, "LightGBM GPU/OpenCL not available → retrying on CPU.")
                            params = model.get_params()
                            params["device_type"] = "cpu"
                            model = lgb.LGBMClassifier(**params)
                            fitted = model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                eval_metric="multi_logloss",
                                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
                            )
                        else:
                            raise
                    best_it = getattr(model, "best_iteration_", None)
                    if best_it is not None:
                        log(mdir, f"LGB best_iteration_={best_it}")

                else:
                    log(mdir, "Fitting estimator...")
                    fitted = model.fit(X_train, y_train)

            except MemoryError as e:
                log(mdir, f"OOM encountered: {e}. Retrying with stratified fallback...")
                Xs, ys, picked = stratified_subsample(X_train, y_train, args.fallback_samples, seed=rs)
                log(mdir, f"Fallback subset size={picked}")
                fitted = model.fit(Xs, ys)
            except Exception as e:
                tb = traceback.format_exc()
                log(mdir, f"ERROR during fit: {e}\n{tb}")
                results_summary.append({"model": name, "error": str(e)})
                continue

            # Save model
            try:
                joblib.dump(fitted, mdir / "model.joblib")
            except Exception as e:
                log(mdir, f"WARNING: could not save model via joblib: {e}")

            # Predict & metrics
            try:
                y_pred = fitted.predict(X_test)
            except Exception as e:
                tb = traceback.format_exc()
                log(mdir, f"ERROR during predict: {e}\n{tb}")
                results_summary.append({"model": name, "error": f"predict: {e}"})
                continue

            prob_mat = score_matrix(fitted, X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="macro")
            f1   = f1_score(y_test, y_pred, average="macro")
            try:
                auc_ovr = roc_auc_score(y_test, prob_mat, multi_class="ovr") if prob_mat is not None else None
            except Exception:
                auc_ovr = None

            metrics = {
                "model": name,
                "samples_evaluated": int(X_test.shape[0]),
                "accuracy": round(float(acc), 6),
                "precision_macro": round(float(prec), 6),
                "recall_macro": round(float(rec), 6),
                "f1_macro": round(float(f1), 6),
                "auc_ovr": (round(float(auc_ovr), 6) if auc_ovr is not None else None),
                "timestamp": now_utc(),
                "train_time_sec": round(time.time() - start, 2),
            }
            save_json(metrics, rdir / "metrics.json")
            save_text(classification_report(y_test, y_pred, digits=4), rdir / "classification_report.txt")
            plot_confusion(y_test, y_pred, rdir / "confusion_matrix.png", f"Confusion Matrix: {name}", classes=(0,1,2))
            if prob_mat is not None:
                plot_roc_multiclass(y_test, prob_mat, rdir / "roc_curve.png", f"ROC (OvR): {name}", classes=(0,1,2))

            # Feature importances
            fi = feature_importance(fitted, np.array(feature_cols))
            if fi is not None:
                fnames, fvals, extra = fi
                df_fi = pd.DataFrame({"feature": fnames, "importance": fvals}).sort_values("importance", ascending=False)
                df_fi.to_csv(rdir / "feature_importance.csv", index=False)

                top = min(40, len(df_fi))
                fig_h = max(4.0, 0.32 * top)
                fig, ax = plt.subplots(figsize=(8, fig_h))
                ax.barh(df_fi["feature"].head(top)[::-1], df_fi["importance"].head(top)[::-1])
                ax.set_title(f"Feature Importances: {name} (top {top})")
                ax.set_xlabel("Importance")
                fig.tight_layout()
                fig.savefig(rdir / "feature_importance.png", dpi=200)
                plt.close(fig)

                if extra and "per_class" in extra:
                    coef = np.asarray(extra["per_class"], dtype=float)  # (C,F)
                    df_coef = pd.DataFrame(coef, columns=fnames)
                    df_coef.index = [f"class_{c}" for c in range(coef.shape[0])]
                    df_coef.to_csv(rdir / "linear_coefficients.csv")

            log(mdir, f"DONE metrics={metrics}")
            results_summary.append({"model": name, **metrics})
            mem_gc()

    # ---------------- Tabular DL ----------------
    if not args.only_ml:
        print("\n[Stage] Training tabular DL models...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        bs = args.batch_gpu if device.type == "cuda" else args.batch_cpu
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                  batch_size=bs, shuffle=True,  num_workers=4, pin_memory=(device.type=="cuda"))
        val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                                  batch_size=bs, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))
        test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test)),
                                  batch_size=bs, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))

        # DL-MLP with ReduceLROnPlateau
        name = "DL-MLP"
        mdl = MLPNetMulti(in_dim=X.shape[1], out_classes=n_classes, hidden=(256,128,64), p_drop=0.2).to(device)
        train_dl_model(
            name, mdl, train_loader, val_loader, test_loader, device,
            ensure_dir(models_root / safe_name(name)), ensure_dir(results_root / safe_name(name)),
            epochs=args.epochs, patience=args.patience, min_delta=args.min_delta,
            use_plateau=True, plateau_factor=0.5, plateau_patience=1, min_lr=1e-5
        )
        mem_gc()

        # DL-FTTransformer
        name = "DL-FTTransformer"
        mdl = FTTransformer(n_features=X.shape[1], out_classes=n_classes, d_model=64, n_heads=8, n_layers=3, p_drop=0.1).to(device)
        train_dl_model(
            name, mdl, train_loader, val_loader, test_loader, device,
            ensure_dir(models_root / safe_name(name)), ensure_dir(results_root / safe_name(name)),
            epochs=args.epochs, patience=args.patience, min_delta=args.min_delta,
            use_plateau=False
        )
        mem_gc()

    # ---------------- Summary table ----------------
    print("\n[SUMMARY]")
    for row in results_summary:
        if "error" in row:
            print(f"{row['model']:<24} ERROR: {row['error']}")
        else:
            print(f"{row['model']:<24} acc={row['accuracy']:.4f} f1M={row['f1_macro']:.4f} aucOvR={row['auc_ovr']} time={row['train_time_sec']}s")

    print("\n[DONE] All models trained. Artifacts in:")
    print(f"  Models : {models_root}")
    print(f"  Results: {results_root}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()

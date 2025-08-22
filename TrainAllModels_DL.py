"""
All-in-one Deep Learning trainer (Python 3.13+), outputs aligned with ML trainer.

Adds to the original DL script:
- Train-only scaler (no leakage) and shared 70/10/20 split indices
- Global diagnostics in results/_global: data_quality.json, drift_report.csv/json,
  outliers.json, label_stats.json, class_weights.json
- Per-model diagnostics in results/<Model>:
  metrics_train.json, metrics_val.json, metrics_test.json,
  thresholds_val.csv, metrics_test_at_opt_threshold.json,
  confusion_matrix.json/.png, roc_curve.png, pr_curve.png,
  calibration.png + calibration_stats.txt,
  legacy artifacts: metrics.json, classification_report.txt
- Generalization gaps (over/underfitting hints)
- Tabular models only: permutation-based feature_importance.csv/.png

Models trained:
  Tabular: DL-MLP, DL-FTTransformer
  Text (URL bytes): DL-CharCNN, DL-CharTransformer
"""

from pathlib import Path
from datetime import datetime, timezone
import os, gc, json, math, warnings
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
    matthews_corrcoef, cohen_kappa_score, confusion_matrix, log_loss,
    classification_report, PrecisionRecallDisplay, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import IsolationForest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

# --- CUDA hygiene
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Config ----------------
CONFIG = {
    "dataset_csv": "features_extracted.csv",
    "label_col": "label",
    "url_col": "url",

    "models_dir": "models",
    "results_dir": "results",
    "random_state": 42,

    "train_ratio": 0.70,
    "val_ratio":   0.10,
    "test_ratio":  0.20,

    # Training
    "epochs_tabular": 12,
    "epochs_text": 6,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "patience": 5,         # tabular early stopping
    "patience_text": 4,    # text early stopping
    "min_delta_auc": 1e-4,

    # Batch sizes
    "batch_size_gpu_tabular": 8192,
    "batch_size_cpu_tabular": 8192,
    "batch_size_gpu_text_cnn": 4096,
    "batch_size_gpu_text_transformer": 1024,
    "batch_size_cpu_text": 8192,
    "grad_accum_tabular": 1,
    "grad_accum_text": 4,

    # Diagnostics
    "drift_bins": 10,
    "ece_bins": 15,
    "isoforest_contamination": 0.02,

    # URL tokenizer
    "max_len_cap": 256,
    "use_cls_token": True,

    # Permutation importance (tabular)
    "perm_importance_repeats": 3
}

# -------------- Small utils --------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------- Metrics & diagnostics shared with ML script --------------
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

def rich_metrics(y_true, y_score, y_pred, proba=None, ece_bins=15):
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
        out["brier"] = float(np.mean((proba - y_true)**2))
        out["log_loss_ce"] = float(log_loss(y_true, proba, labels=[0,1]))
        out["ece"] = float(expected_calibration_error(y_true, proba, n_bins=ece_bins))
    else:
        out["brier"] = None; out["log_loss_ce"] = None; out["ece"] = None
    return out

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

def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    v = np.unique(np.concatenate([a, b]))
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

# -------------- URL tokenization (bytes) --------------
PAD_ID = 256
CLS_ID = 257
VOCAB_SIZE = 258

def map_label(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().lower()
    return 1 if s in {"1","true","malicious","phishing","malware","bad"} else 0

def pick_max_len(urls, cap=256, use_cls=True):
    lens = [len(u.encode("utf-8","ignore")) for u in urls]
    p99 = int(np.percentile(lens, 99)) if lens else 32
    return int(np.clip(p99 + (1 if use_cls else 0) + 1, 32, cap))

def encode_url(u: str, max_len: int, use_cls=True):
    b = u.encode("utf-8", "ignore")
    if use_cls:
        b = b[:max_len-1]
        ids = [CLS_ID] + [x for x in b]
    else:
        b = b[:max_len]
        ids = [x for x in b]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    ids = [i if i >= 256 else max(0, min(255, i)) for i in ids]
    return np.array(ids, dtype=np.int64)

class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len, use_cls=True):
        self.X = np.stack([encode_url(u, max_len, use_cls) for u in urls])
        self.y = np.array([map_label(y) for y in labels], dtype=np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------- DL models --------------
class MLPNet(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128, 64), p_drop=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(p_drop)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
    def forward(self, x): return self.head(self.backbone(x)).squeeze(-1)

class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_model=64, n_heads=8, n_layers=3, p_drop=0.1):
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
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model, 1))
    def forward(self, x):  # (B,F)
        B, F = x.shape
        val_tok = self.value_emb(x.view(B*F,1)).view(B, F, -1)
        idx = torch.arange(F, device=x.device).view(1, F)
        feat_tok = self.feat_emb(idx).expand(B, -1, -1)
        z = self.encoder(val_tok + feat_tok)
        z = self.norm(z).mean(dim=1)
        return self.head(z).squeeze(-1)

class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, p_drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.conv1 = nn.Conv1d(d_model, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn1, self.bn2 = nn.BatchNorm1d(128), nn.BatchNorm1d(256)
        self.act, self.drop = nn.GELU(), nn.Dropout(p_drop)
        self.head = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Dropout(p_drop), nn.Linear(128,1))
    def forward(self, x_ids):
        x = self.emb(x_ids).transpose(1,2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = torch.amax(x, dim=2)
        x = self.drop(x)
        return self.head(x).squeeze(-1)

class CharTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=96, n_heads=6, n_layers=4, max_len=256, p_drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=p_drop, batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model,d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model,1))
    def forward(self, x_ids):
        B, L = x_ids.shape
        pos = torch.arange(L, device=x_ids.device).unsqueeze(0).expand(B, L)
        z = self.enc(self.emb(x_ids) + self.pos(pos))
        z = self.norm(z)
        pooled = z[:,0,:] if (x_ids[:,0]==CLS_ID).all() else z.mean(dim=1)
        return self.head(pooled).squeeze(-1)

# -------------- Inference helpers --------------
@torch.no_grad()
def infer_proba(model, loader, device, is_text: bool):
    model.eval()
    sig = nn.Sigmoid()
    outs, labs = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).long() if is_text else xb.to(device, non_blocking=True).float()
        logits = model(xb)
        probs = sig(logits).detach().cpu().numpy()
        outs.append(probs); labs.append(yb.numpy())
    return np.concatenate(outs), np.concatenate(labs)

@torch.no_grad()
def infer_proba_array(model, X_np, device):
    # convenience for permutation importance (tabular only)
    bs = 65536
    sig = nn.Sigmoid()
    model.eval()
    out = []
    for i in range(0, len(X_np), bs):
        xb = torch.from_numpy(X_np[i:i+bs]).to(device).float()
        out.append(sig(model(xb)).detach().cpu().numpy())
    return np.concatenate(out)

# -------------- Feature importance (tabular, permutation on ROC-AUC) --------------
def permutation_importance_tabular(model, X_test, y_test, device, repeats=3):
    base = roc_auc_score(y_test, infer_proba_array(model, X_test, device))
    rng = np.random.default_rng(123)
    importances = np.zeros(X_test.shape[1], dtype=float)
    for j in range(X_test.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X_test.copy()
            rng.shuffle(Xp[:, j])
            prob = infer_proba_array(model, Xp, device)
            try:
                auc = roc_auc_score(y_test, prob)
            except Exception:
                auc = np.nan
            drops.append(base - auc if not np.isnan(auc) else 0.0)
        importances[j] = float(np.mean(drops))
    return importances

# -------------- Training loop --------------
def train_model(name, model, train_loader, val_loader, device, cfg, is_text=False, patience=5, epochs=20, grad_accum=1):
    # class imbalance → pos_weight
    if isinstance(train_loader.dataset, TensorDataset):
        y_tr = train_loader.dataset.tensors[1].cpu().numpy()
    else:
        y_tr = train_loader.dataset.y
    pos = int((y_tr == 1).sum()); neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(1, pos)], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc, no_improve, best_state = -1.0, 0, None
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)
        step = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).long() if is_text else xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(xb)
                raw_loss = criterion(logits, yb)
                loss = raw_loss / max(1, grad_accum)

            scaler.scale(loss).backward()
            step += 1
            if step % max(1, grad_accum) == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += raw_loss.item() * xb.size(0)

        if step % max(1, grad_accum) != 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        # Early stopping on VAL AUC
        with torch.no_grad():
            # reuse train_loader.dataset shapes for val set size
            val_probs, val_y = infer_proba(model, val_loader, device, is_text)
            try: val_auc = roc_auc_score(val_y, val_probs)
            except Exception: val_auc = float("nan")

        print(f"[{name}] Epoch {epoch:02d}/{epochs}  loss={running/len(train_loader.dataset):.6f}  val_auc={val_auc:.6f}")

        improved = (not math.isnan(val_auc)) and (val_auc > best_auc + cfg["min_delta_auc"])
        if improved:
            best_auc = val_auc; no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{name}] Early stopping. Best val AUC={best_auc:.6f}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return model, best_state

def write_legacy_and_plots(name, y_true, y_pred, y_score, rdir: Path, ece_bins: int):
    # legacy metrics.json + classification_report.txt + plots (confusion, ROC)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_score)
    except Exception: auc = None
    legacy = {
        "model": name,
        "samples_evaluated": int(len(y_true)),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1_score": round(float(f1), 6),
        "auc_roc": (round(float(auc), 6) if auc is not None else None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(legacy, rdir / "metrics.json")

    report_txt = classification_report(y_true, y_pred, digits=4)
    header = f"Performance Analysis for: {name}\n" + "=" * (24 + len(name)) + "\n"
    body = (
        f"Samples (evaluated):\t{len(y_true)}\n"
        f"Accuracy:\t\t{legacy['accuracy']}\n"
        f"Precision:\t\t{legacy['precision']}\n"
        f"Recall:\t\t\t{legacy['recall']}\n"
        f"F1-Score:\t\t{legacy['f1_score']}\n"
        f"AUC-ROC Score:\t\t{legacy['auc_roc']}\n"
        + "=" * (24 + len(name)) + "\n\n"
        "Detailed Classification Report\n"
        "------------------------------\n" + report_txt
    )
    save_text(body, rdir / "classification_report.txt")

    # confusion JSON + PNG
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    save_json({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}, rdir / "confusion_matrix.json")
    fig, ax = plt.subplots(figsize=(5.5,5))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix: {name}")
    fig.tight_layout(); fig.savefig(rdir / "confusion_matrix.png", dpi=200); plt.close(fig)

    # ROC
    try:
        fig, ax = plt.subplots(figsize=(6,5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title(f"ROC: {name}"); fig.tight_layout()
        fig.savefig(rdir / "roc_curve.png", dpi=200); plt.close(fig)
    except Exception:
        pass

    # PR
    try:
        fig, ax = plt.subplots(figsize=(6,5))
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title(f"Precision-Recall: {name}"); fig.tight_layout()
        fig.savefig(rdir / "pr_curve.png", dpi=200); plt.close(fig)
    except Exception:
        pass

    # Calibration plot + stats
    try:
        fig, ax = plt.subplots(figsize=(6,5))
        CalibrationDisplay.from_predictions(y_true, y_score, n_bins=ece_bins, ax=ax)
        ax.set_title(f"Calibration: {name}"); fig.tight_layout()
        fig.savefig(rdir / "calibration.png", dpi=200); plt.close(fig)
        save_text(
            f"Brier: {np.mean((y_score - y_true)**2):.6f}\n"
            f"ECE(bins={ece_bins}): {expected_calibration_error(y_true, y_score, ece_bins):.6f}\n"
            f"LogLoss (cross-entropy): {log_loss(y_true, y_score, labels=[0,1]):.6f}\n",
            rdir / "calibration_stats.txt"
        )
    except Exception:
        pass

# -------------- Main --------------
def main():
    cfg = CONFIG
    rs = int(cfg["random_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    models_root = ensure_dir(Path(cfg["models_dir"]))
    results_root = ensure_dir(Path(cfg["results_dir"]))
    global_models = ensure_dir(models_root / "_global")
    global_res = ensure_dir(results_root / "_global")

    # ===== Load dataset =====
    csv_path = Path(cfg["dataset_csv"]); assert csv_path.exists(), f"{csv_path} not found."
    df = pd.read_csv(csv_path)
    assert cfg["label_col"] in df.columns, f"Missing label column: {cfg['label_col']}"
    assert cfg["url_col"] in df.columns, f"Missing URL column '{cfg['url_col']}'"

    # Label mapping
    df[cfg["label_col"]] = df[cfg["label_col"]].apply(map_label).astype(np.int64)
    y_all = df[cfg["label_col"]].to_numpy(dtype=np.int64)
    assert set(np.unique(y_all)) <= {0,1}, "Labels must map to {0,1}"

    # ===== Shared stratified split of indices =====
    idx_all = np.arange(len(df))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx_all, y_all, test_size=(1.0 - cfg["train_ratio"]), random_state=rs, stratify=y_all
    )
    idx_val, idx_te, y_val, y_te = train_test_split(
        idx_tmp, y_tmp,
        test_size=(cfg["test_ratio"]/(cfg["val_ratio"]+cfg["test_ratio"])),
        random_state=rs, stratify=y_tmp
    )
    save_json({"train_idx": idx_tr.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_te.tolist()},
              global_models / "split_indices.json")
    print(f"[SPLIT] Sizes -> train {len(idx_tr)}, val {len(idx_val)}, test {len(idx_te)}")

    # ===== Global diagnostics (like Script 1) =====
    # Data quality (numeric + label)
    num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols_all if c != cfg["label_col"]]  # numeric-only features
    save_json(data_quality_report(df[feature_cols + [cfg["label_col"]]], cfg["label_col"]), global_res / "data_quality.json")

    # Raw arrays for drift (before scaling)
    X_raw_all = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    X_tr_raw, X_val_raw, X_te_raw = X_raw_all[idx_tr], X_raw_all[idx_val], X_raw_all[idx_te]

    # Fit scaler on TRAIN only (like Script 1)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_te  = scaler.transform(X_te_raw)
    joblib.dump(scaler, global_models / "scaler.joblib")
    save_json({"feature_columns": feature_cols}, global_models / "feature_columns.json")
    print("[Global] Saved scaler & feature_columns.")

    # Label stats
    def counts(v): 
        u, c = np.unique(v, return_counts=True); d = {int(k): int(vv) for k, vv in zip(u, c)}; 
        return {"counts": d, "ratio_pos": float(d.get(1,0)/max(1,sum(d.values())))}
    save_json({"train": counts(y_tr), "val": counts(y_val), "test": counts(y_te)}, global_res / "label_stats.json")

    # Drift on RAW distributions
    ddf = drift_report(X_tr_raw, X_te_raw, np.array(feature_cols), bins=cfg["drift_bins"])
    ddf.to_csv(global_res / "drift_report.csv", index=False)
    save_json({
        "top_by_psi": ddf.head(20).to_dict(orient="records"),
        "psi_guidance": "PSI <0.1 stable; 0.1–0.25 moderate; >0.25 major shift",
        "ks_guidance":  "KS <0.1 small; 0.1–0.2 moderate; >0.2 large"
    }, global_res / "drift_report.json")

    # Outlier scan (fit on TRAIN scaled)
    iso = IsolationForest(n_estimators=200, random_state=rs, contamination=cfg["isoforest_contamination"])
    iso.fit(X_tr)
    def outlier_frac(Xp): return float(np.mean(iso.predict(Xp) == -1))
    save_json({
        "contamination_cfg": cfg["isoforest_contamination"],
        "train_flagged_fraction": outlier_frac(X_tr),
        "val_flagged_fraction": outlier_frac(X_val),
        "test_flagged_fraction": outlier_frac(X_te)
    }, global_res / "outliers.json")

    # Class weights
    cls_w = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr)
    save_json({"class_weights_(0,1)": [float(cls_w[0]), float(cls_w[1])]}, global_res / "class_weights.json")

    # ===== Dataloaders =====
    y_tr_np, y_val_np, y_te_np = y_tr, y_val, y_te
    device_is_cuda = (device.type == "cuda")
    bs_tab = cfg["batch_size_gpu_tabular"] if device_is_cuda else cfg["batch_size_cpu_tabular"]

    def TL(X, y): 
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), 
                          batch_size=bs_tab, shuffle=False, num_workers=4, pin_memory=device_is_cuda)

    tr_loader_tab = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_np)),
                               batch_size=bs_tab, shuffle=True, num_workers=4, pin_memory=device_is_cuda)
    val_loader_tab = TL(X_val, y_val_np)
    te_loader_tab  = TL(X_te,  y_te_np)

    # ===== TABULAR: DL-MLP =====
    def run_tabular_model(name, model):
        mname = safe_name(name)
        mdir = ensure_dir(models_root / mname)
        rdir = ensure_dir(results_root / mname)

        model, state = train_model(name, model.to(device), tr_loader_tab, val_loader_tab, device, cfg,
                                   is_text=False, patience=cfg["patience"], epochs=cfg["epochs_tabular"],
                                   grad_accum=cfg["grad_accum_tabular"])
        torch.save(state, mdir / "model.pt")

        # Per-split metrics
        p_tr, y_tru = infer_proba(model, tr_loader_tab, device, is_text=False)
        p_va, y_vau = infer_proba(model, val_loader_tab, device, is_text=False)
        p_te, y_teu = infer_proba(model, te_loader_tab,  device, is_text=False)
        yhat_tr = (p_tr >= 0.5).astype(int); yhat_va = (p_va >= 0.5).astype(int); yhat_te = (p_te >= 0.5).astype(int)

        m_train = rich_metrics(y_tru, p_tr, yhat_tr, p_tr, ece_bins=cfg["ece_bins"])
        m_val   = rich_metrics(y_vau, p_va, yhat_va, p_va, ece_bins=cfg["ece_bins"])
        m_test  = rich_metrics(y_teu, p_te, yhat_te, p_te, ece_bins=cfg["ece_bins"])
        m_train["split"] = "train"; m_val["split"] = "val"; m_test["split"] = "test"
        for d, fn in [(m_train,"metrics_train.json"), (m_val,"metrics_val.json"), (m_test,"metrics_test.json")]:
            save_json(d, rdir / fn)

        # Threshold sweep on VAL -> apply to TEST
        sweep_df, t_f1, t_j = sweep_thresholds(y_vau, p_va)
        sweep_df.to_csv(rdir / "thresholds_val.csv", index=False)
        t_star = float(t_f1)
        yhat_opt = (p_te >= t_star).astype(int)
        m_test_opt = rich_metrics(y_teu, p_te, yhat_opt, p_te, ece_bins=cfg["ece_bins"])
        m_test_opt["threshold_used"] = t_star
        save_json(m_test_opt, rdir / "metrics_test_at_opt_threshold.json")

        # Legacy files + plots
        write_legacy_and_plots(name, y_teu, yhat_te, p_te, rdir, cfg["ece_bins"])

        # Generalization gaps
        gaps = {
            "acc_gap_train_val": float(m_train["accuracy"] - m_val["accuracy"]),
            "acc_gap_train_test": float(m_train["accuracy"] - m_test["accuracy"]),
            "auc_gap_train_val": float(m_train["roc_auc"] - m_val["roc_auc"]) if (m_train["roc_auc"] is not None and m_val["roc_auc"] is not None) else None,
            "auc_gap_train_test": float(m_train["roc_auc"] - m_test["roc_auc"]) if (m_train["roc_auc"] is not None and m_test["roc_auc"] is not None) else None
        }
        save_json(gaps, rdir / "generalization_gap.json")

        # Permutation feature importance (tabular only)
        try:
            imps = permutation_importance_tabular(model, X_te, y_teu, device,
                                                  repeats=cfg["perm_importance_repeats"])
            df_fi = pd.DataFrame({"feature": feature_cols, "importance": imps}).sort_values("importance", ascending=False)
            df_fi.to_csv(rdir / "feature_importance.csv", index=False)
            fig_h = max(4.0, 0.36 * len(df_fi)); fig_w = max(7.0, 0.28 * len(df_fi) if len(df_fi) > 35 else 7.0)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.barh(df_fi["feature"][::-1], df_fi["importance"][::-1])
            ax.set_title(f"Feature Importances (perm AUC drop): {name}")
            ax.set_xlabel("Importance (ΔAUC)")
            fig.tight_layout(); fig.savefig(rdir / "feature_importance.png", dpi=200); plt.close(fig)
        except Exception:
            pass

        # Test preds with URLs (handy for audits)
        pd.DataFrame({"url": df.loc[idx_te, cfg["url_col"]].values,
                      "y_true": y_teu, "y_pred": yhat_te, "y_score": p_te}).to_csv(rdir / "test_with_preds.csv", index=False)

        print(f"[OK] Saved model -> {mdir}")
        print(f"[OK] Saved results -> {rdir}")

    # Run tabular models
    run_tabular_model("DL-MLP", MLPNet(in_dim=X_tr.shape[1], hidden=(256,128,64), p_drop=0.2))
    cleanup_cuda()
    run_tabular_model("DL-FTTransformer", FTTransformer(n_features=X_tr.shape[1], d_model=64, n_heads=8, n_layers=3, p_drop=0.1))
    cleanup_cuda()

    # ===== TEXT (URL bytes) =====
    print("\n[Text] Preparing URL byte-token inputs (shared indices)...")
    urls = df.loc[:, cfg["url_col"]].astype(str).tolist()
    max_len = pick_max_len(urls, cap=cfg["max_len_cap"], use_cls=cfg["use_cls_token"])
    save_json({"tokenizer": "byte", "vocab_size": VOCAB_SIZE, "pad_id": PAD_ID, "cls_id": CLS_ID,
               "use_cls_token": bool(cfg["use_cls_token"]), "max_len": int(max_len)}, global_models / "url_tokenizer.json")
    print(f"[Text] max_len={max_len}")

    ds_train = URLDataset(df.loc[idx_tr, cfg["url_col"]].tolist(), y_tr_np, max_len, cfg["use_cls_token"])
    ds_val   = URLDataset(df.loc[idx_val, cfg["url_col"]].tolist(), y_val_np, max_len, cfg["use_cls_token"])
    ds_test  = URLDataset(df.loc[idx_te, cfg["url_col"]].tolist(), y_te_np,  max_len, cfg["use_cls_token"])

    def TL_text(ds, bs):
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=device_is_cuda)

    # Helper to run each text model with identical outputs
    def run_text_model(name, model, bs):
        mname = safe_name(name)
        mdir = ensure_dir(models_root / mname)
        rdir = ensure_dir(results_root / mname)

        tr_loader = DataLoader(ds_train, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=device_is_cuda)
        va_loader = TL_text(ds_val, bs); te_loader = TL_text(ds_test, bs)

        model, state = train_model(name, model.to(device), tr_loader, va_loader, device, cfg,
                                   is_text=True, patience=cfg["patience_text"], epochs=cfg["epochs_text"],
                                   grad_accum=cfg["grad_accum_text"])
        torch.save(state, mdir / "model.pt")

        p_tr, y_tru = infer_proba(model, tr_loader, device, is_text=True)
        p_va, y_vau = infer_proba(model, va_loader, device,  is_text=True)
        p_te, y_teu = infer_proba(model, te_loader, device,  is_text=True)
        yhat_tr = (p_tr >= 0.5).astype(int); yhat_va = (p_va >= 0.5).astype(int); yhat_te = (p_te >= 0.5).astype(int)

        m_train = rich_metrics(y_tru, p_tr, yhat_tr, p_tr, ece_bins=cfg["ece_bins"])
        m_val   = rich_metrics(y_vau, p_va, yhat_va, p_va, ece_bins=cfg["ece_bins"])
        m_test  = rich_metrics(y_teu, p_te, yhat_te, p_te, ece_bins=cfg["ece_bins"])
        for d, fn in [(m_train,"metrics_train.json"), (m_val,"metrics_val.json"), (m_test,"metrics_test.json")]:
            save_json(d, rdir / fn)

        sweep_df, t_f1, t_j = sweep_thresholds(y_vau, p_va)
        sweep_df.to_csv(rdir / "thresholds_val.csv", index=False)
        t_star = float(t_f1)
        yhat_opt = (p_te >= t_star).astype(int)
        m_test_opt = rich_metrics(y_teu, p_te, yhat_opt, p_te, ece_bins=cfg["ece_bins"])
        m_test_opt["threshold_used"] = t_star
        save_json(m_test_opt, rdir / "metrics_test_at_opt_threshold.json")

        write_legacy_and_plots(name, y_teu, yhat_te, p_te, rdir, cfg["ece_bins"])

        gaps = {
            "acc_gap_train_val": float(m_train["accuracy"] - m_val["accuracy"]),
            "acc_gap_train_test": float(m_train["accuracy"] - m_test["accuracy"]),
            "auc_gap_train_val": float(m_train["roc_auc"] - m_val["roc_auc"]) if (m_train["roc_auc"] is not None and m_val["roc_auc"] is not None) else None,
            "auc_gap_train_test": float(m_train["roc_auc"] - m_test["roc_auc"]) if (m_train["roc_auc"] is not None and m_test["roc_auc"] is not None) else None
        }
        save_json(gaps, rdir / "generalization_gap.json")

        pd.DataFrame({"url": df.loc[idx_te, cfg["url_col"]].values,
                      "y_true": y_teu, "y_pred": yhat_te, "y_score": p_te}).to_csv(rdir / "test_with_preds.csv", index=False)

        print(f"[OK] Saved model -> {mdir}")
        print(f"[OK] Saved results -> {rdir}")

    # Run text models
    bs_cnn = cfg["batch_size_gpu_text_cnn"] if device_is_cuda else cfg["batch_size_cpu_text"]
    run_text_model("DL-CharCNN", CharCNN(vocab_size=VOCAB_SIZE, d_model=64, p_drop=0.2), bs_cnn)
    cleanup_cuda()
    bs_tr  = cfg["batch_size_gpu_text_transformer"] if device_is_cuda else cfg["batch_size_cpu_text"]
    run_text_model("DL-CharTransformer", CharTransformer(vocab_size=VOCAB_SIZE, d_model=96, n_heads=6, n_layers=4, max_len=pick_max_len(urls, cfg["max_len_cap"], cfg["use_cls_token"]), p_drop=0.1), bs_tr)
    cleanup_cuda()

    print("\n[DONE] All DL models trained with outputs.")

if __name__ == "__main__":
    main()

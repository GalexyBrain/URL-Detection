"""
Adversarial evaluation for your trained models (Python 3.13+).

- Tabular white-box FGSM/PGD: DL-MLP, DL-FTTransformer, Logistic_Regression, Calibrated_LinearSVC
- Tabular black-box via transfer (surrogate + fine-tune): Decision_Tree, Random_Forest,
  LightGBM, XGBoost, AdaBoost, Gaussian_Naive_Bayes
- URL text models: DL-CharCNN, DL-CharTransformer using HotFlip-style FGSM/PGD in token space

Artifacts (per model and per attack):
  results/<ModelName>/adv_{NAT|FGSM|PGD}/
      metrics.json, classification_report.txt, confusion_matrix.png, roc_curve.png, attack_config.json

Global artifacts used:
  models/_global/scaler.joblib
  models/_global/feature_columns.json
  models/_global/url_tokenizer.json
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, gc, math, json, warnings
import numpy as np
import pandas as pd
import joblib

# cut fragmentation *before* CUDA alloc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import CalibratedClassifierCV

# sklearn models you'll load
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Optional libs (teachers were trained with these already)
try:
    import lightgbm as lgb  # noqa: F401
except Exception:
    lgb = None
try:
    import xgboost as xgb  # noqa: F401
except Exception:
    xgb = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CFG = {
    "models_dir": "models",
    "results_dir": "results",

    # Data files
    "tabular_csv": "features_extracted.csv",      # numeric features + label
    "url_csv":     "merged_url_dataset.csv",      # url,label (strings)

    # Columns
    "label_col": "label",
    "url_col": "url",

    # Split for attack evaluation subset (we only need val/test)
    "train_ratio": 0.70,
    "val_ratio":   0.10,
    "test_ratio":  0.20,

    "random_state": 42,

    # Limit samples to keep runtime sane
    "max_attack_samples_tab": 50000,
    "max_attack_samples_text": 30000,

    # Batch sizes (auto backoff on OOM)
    "tab_eval_bs":   8192,
    "tab_attack_bs": 4096,
    "text_eval_bs":  2048,
    "text_attack_bs":1024,

    # Device
    "use_gpu": True,

    # Dataloader workers
    "num_workers": 2,
}

# Attack knobs
ATTACKCFG = {
    # Tabular (z-space)
    "eps_tab_fgsm": 0.5,
    "eps_tab_pgd": 0.5,
    "alpha_tab_pgd": 0.05,
    "steps_tab_pgd": 10,

    # Linear white-box
    "eps_lin_fgsm": 0.5,
    "eps_lin_pgd": 0.5,
    "alpha_lin_pgd": 0.05,
    "steps_lin_pgd": 10,

    # URL HotFlip
    "char_topk_per_step": 1,     # flip k positions per step per sample
    "char_steps_fgsm": 1,        # 1 step => FGSM-like
    "char_steps_pgd": 5,         # multi-step PGD
    "freeze_cls_pos0": True,     # keep CLS token fixed
}

# -------------------------------------------------------------------
# Utility: IO / plotting / metrics
# -------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(txt, p: Path):
    p.write_text(txt, encoding="utf-8")

def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_roc(y_true, y_score, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def metrics_to_files(name, y_true, y_pred, y_score, out_dir: Path, tag: str):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None
    metrics = {
        "model": name, "attack": tag,
        "samples_evaluated": int(len(y_true)),
        "accuracy": round(float(acc), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1_score": round(float(f1), 6),
        "auc_roc": (round(float(auc), 6) if auc is not None else None),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(metrics, out_dir / "metrics.json")
    report = classification_report(y_true, y_pred, digits=4)
    hdr = f"Performance Analysis for: {name} ({tag})\n" + "=" * (28 + len(name)) + "\n"
    body = (
        f"Samples (evaluated):\t{len(y_true)}\n"
        f"Accuracy:\t\t{metrics['accuracy']}\n"
        f"Precision:\t\t{metrics['precision']}\n"
        f"Recall:\t\t\t{metrics['recall']}\n"
        f"F1-Score:\t\t{metrics['f1_score']}\n"
        f"AUC-ROC Score:\t\t{metrics['auc_roc']}\n"
        + "=" * (28 + len(name)) + "\n\n"
        "Detailed Classification Report\n"
        "------------------------------\n" + report
    )
    save_text(body, out_dir / "classification_report.txt")
    try:
        plot_confusion(y_true, y_pred, out_dir / "confusion_matrix.png", f"Confusion Matrix: {name} [{tag}]")
        plot_roc(y_true, y_score, out_dir / "roc_curve.png", f"ROC Curve: {name} [{tag}]")
    except Exception:
        pass

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------------------------------------------------------
# Data loading / label mapping
# -------------------------------------------------------------------
def map_label(v):
    if isinstance(v, (int, np.integer)): return int(v)
    s = str(v).strip().lower()
    return 1 if s in {"1","true","malicious","phishing","malware","bad"} else 0

def load_tabular_scaled(cfg: dict):
    models_root = Path(cfg["models_dir"])
    global_dir = models_root / "_global"

    df = pd.read_csv(cfg["tabular_csv"])
    assert cfg["label_col"] in df.columns, "label column missing"
    y = df[cfg["label_col"]].apply(map_label).astype(np.int64).to_numpy()

    # Drop any url column and keep numeric only
    drop = []
    if cfg["url_col"] in df.columns: drop.append(cfg["url_col"])
    X_df = df.drop(columns=drop + [cfg["label_col"]], errors="ignore").apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    if X_df.isna().any().any():
        X_df = X_df.fillna(X_df.median(numeric_only=True))
    feature_cols = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=np.float32)

    scaler_path = global_dir / "scaler.joblib"
    assert scaler_path.exists(), "Missing global scaler: models/_global/scaler.joblib"
    scaler: StandardScaler = joblib.load(scaler_path)
    Xz = scaler.transform(X).astype(np.float32)

    # z-bounds from data (for projection validity)
    z_min = Xz.min(axis=0)
    z_max = Xz.max(axis=0)

    return Xz, y, feature_cols, (z_min, z_max)

# URL tokenizer & dataset
PAD_ID = 256
CLS_ID = 257
VOCAB_SIZE = 258

def pick_max_len_from_global(cfg: dict):
    tok_path = Path(cfg["models_dir"]) / "_global" / "url_tokenizer.json"
    assert tok_path.exists(), "Missing models/_global/url_tokenizer.json"
    info = json.loads(tok_path.read_text(encoding="utf-8"))
    use_cls = bool(info.get("use_cls_token", True))
    max_len = int(info.get("max_len", 256))
    return max_len, use_cls

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
    # clamp to [0..255] for bytes, keep PAD/CLS as is
    ids = [i if i >= 256 else max(0, min(255, i)) for i in ids]
    return np.array(ids, dtype=np.int64)

class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len, use_cls=True):
        self.X = np.stack([encode_url(u, max_len, use_cls) for u in urls])
        self.y = np.array([map_label(y) for y in labels], dtype=np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def load_url_dataset(cfg: dict, subset_max: int | None):
    df = pd.read_csv(cfg["url_csv"])
    assert "url" in df.columns and "label" in df.columns
    max_len, use_cls = pick_max_len_from_global(cfg)
    urls = df["url"].astype(str).tolist()
    y = [map_label(v) for v in df["label"].tolist()]
    U_tr, U_tmp, y_tr, y_tmp = train_test_split(
        urls, y, test_size=(1.0-cfg["train_ratio"]),
        random_state=cfg["random_state"], stratify=y
    )
    U_val, U_te, y_val, y_te = train_test_split(
        U_tmp, y_tmp,
        test_size=(cfg["test_ratio"]/(cfg["val_ratio"]+cfg["test_ratio"])),
        random_state=cfg["random_state"], stratify=y_tmp
    )
    # Use test set for attack eval; limit size
    X_ids = URLDataset(U_te, y_te, max_len, use_cls)
    n = len(X_ids)
    idx = np.arange(n)
    if subset_max and n > subset_max:
        rng = np.random.default_rng(cfg["random_state"])
        idx = rng.choice(idx, size=subset_max, replace=False)
    X_sub = X_ids.X[idx]
    y_sub = X_ids.y[idx]
    return X_sub, y_sub, max_len, use_cls

# -------------------------------------------------------------------
# DL model definitions (must match training)
# -------------------------------------------------------------------
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
    def forward(self, x):  # x: (B,F)
        return self.head(self.backbone(x)).squeeze(-1)

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
    def forward(self, x):  # x:(B,F)
        B, F = x.shape
        val_tok = self.value_emb(x.view(B*F,1)).view(B, F, -1)
        idx = torch.arange(F, device=x.device).view(1, F)
        feat_tok = self.feat_emb(idx).expand(B, -1, -1)
        z = self.encoder(val_tok + feat_tok)
        z = self.norm(z).mean(dim=1)
        return self.head(z).squeeze(-1)

# Char models
class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, p_drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.conv1 = nn.Conv1d(d_model, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn1, self.bn2 = nn.BatchNorm1d(128), nn.BatchNorm1d(256)
        self.act, self.drop = nn.GELU(), nn.Dropout(p_drop)
        self.head = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Dropout(p_drop), nn.Linear(128,1))
    def forward(self, ids):  # ids: (B,L)
        x = self.emb(ids).transpose(1,2)         # (B,D,L)
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
    def forward(self, ids):  # ids: (B,L)
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        z = self.emb(ids) + self.pos(pos)
        z = self.enc(z)
        z = self.norm(z)
        pooled = z[:,0,:] if (ids[:,0]==CLS_ID).all() else z.mean(dim=1)
        return self.head(pooled).squeeze(-1)

# -------------------------------------------------------------------
# Helpers: linear model unwrapping & logits
# -------------------------------------------------------------------
def unwrap_calibrated(estimator):
    if isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
            inner = estimator.calibrated_classifiers_[0]
            if hasattr(inner, "estimator") and inner.estimator is not None:
                return inner.estimator
            if hasattr(inner, "base_estimator") and inner.base_estimator is not None:
                return inner.base_estimator
    return estimator

def linear_wb(model):
    base = unwrap_calibrated(model)
    assert hasattr(base, "coef_") and hasattr(base, "intercept_"), "Not a linear model with coef_/intercept_"
    w = np.array(base.coef_, dtype=np.float32).reshape(-1)
    b = float(np.array(base.intercept_, dtype=np.float32).reshape(1)[0])
    return w, b

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

# -------------------------------------------------------------------
# Eval (batched, auto OOM backoff)
# -------------------------------------------------------------------
@torch.no_grad()
def eval_torch_batched(model, Xz, y, device, batch_size):
    N = Xz.shape[0]
    probs_list = []
    bs = max(1, int(batch_size))
    i = 0
    while i < N:
        j = min(N, i + bs)
        try:
            with torch.inference_mode():
                xb = torch.from_numpy(Xz[i:j]).to(device).float()
                logits = model(xb).float()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_list.append(probs)
            i = j
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and bs > 1 and torch.cuda.is_available():
                cleanup_cuda()
                bs = max(1, bs // 2)
                print(f"[WARN] OOM during eval; retrying with batch_size={bs}")
                continue
            raise
    probs = np.concatenate(probs_list, axis=0)
    y_pred = (probs >= 0.5).astype(int)
    return y_pred, probs

@torch.no_grad()
def eval_url_torch_batched(model, ids_np, y, device, batch_size):
    N = ids_np.shape[0]
    probs_list = []
    bs = max(1, int(batch_size))
    i = 0
    while i < N:
        j = min(N, i + bs)
        try:
            with torch.inference_mode():
                ids = torch.from_numpy(ids_np[i:j]).to(device).long()
                logits = model(ids).float()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_list.append(probs)
            i = j
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and bs > 1 and torch.cuda.is_available():
                cleanup_cuda()
                bs = max(1, bs // 2)
                print(f"[WARN] OOM during url eval; retrying with batch_size={bs}")
                continue
            raise
    probs = np.concatenate(probs_list, axis=0)
    y_pred = (probs >= 0.5).astype(int)
    return y_pred, probs

def eval_sklearn(model, X, y):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:,1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        y_score = s if s.ndim==1 else s[:,0]
    else:
        y_score = y_pred
    return y_pred.astype(int), y_score.astype(np.float32)

# -------------------------------------------------------------------
# Attacks: Tabular PyTorch (batched)
# -------------------------------------------------------------------
def attack_tabular_torch_batched(model, Xz, y, eps, alpha, steps, device, z_min, z_max, batch_size):
    N, F = Xz.shape
    adv_list = []
    bs = max(1, int(batch_size))
    steps_eff = 1 if steps == 0 else steps
    alpha_eff = eps if steps == 0 else alpha

    zmin_t = torch.from_numpy(z_min).to(device)
    zmax_t = torch.from_numpy(z_max).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    ptr = 0
    while ptr < N:
        end = min(N, ptr + bs)
        try:
            X = torch.from_numpy(Xz[ptr:end]).to(device).float()
            Y = torch.from_numpy(y[ptr:end]).to(device).float()
            X0 = X.clone().detach()
            X_adv = X.clone().detach()

            for t in range(steps_eff):
                X_adv.requires_grad_(True)
                logits = model(X_adv).float()
                loss = loss_fn(logits, Y)
                model.zero_grad(set_to_none=True)
                if X_adv.grad is not None: X_adv.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    grad_sign = X_adv.grad.sign()
                    X_adv = X_adv + alpha_eff * grad_sign
                    X_adv = torch.clamp(X_adv, X0 - eps, X0 + eps)
                    X_adv = torch.max(torch.min(X_adv, zmax_t), zmin_t)
                    X_adv = X_adv.detach()
            adv_list.append(X_adv.detach().cpu().numpy())
            ptr = end
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and bs > 1 and torch.cuda.is_available():
                cleanup_cuda()
                bs = max(1, bs // 2)
                print(f"[WARN] OOM during tab attack; retrying with batch_size={bs}")
                continue
            raise
    return np.concatenate(adv_list, axis=0)

# -------------------------------------------------------------------
# Attacks: Linear sklearn (white-box)
# -------------------------------------------------------------------
def linear_wb_attack(model, Xz, y, eps, alpha, steps):
    w, b = linear_wb(model)
    is_logreg = isinstance(unwrap_calibrated(model), LogisticRegression)
    X0 = Xz.copy()
    X_adv = Xz.copy()
    steps_eff = 1 if steps == 0 else steps
    alpha_eff = eps if steps == 0 else alpha
    y = y.astype(np.float32)
    for _ in range(steps_eff):
        margin = X_adv @ w + b
        if is_logreg:
            p = 1.0 / (1.0 + np.exp(-margin))
            grad = (p - y)[:, None] * w[None, :]
        else:
            yy = (2*y-1).astype(np.float32)
            grad = (-yy)[:, None] * w[None, :]
        X_adv = X_adv + alpha_eff * np.sign(grad)
        X_adv = np.clip(X_adv, X0 - eps, X0 + eps)
    return X_adv

# -------------------------------------------------------------------
# Attacks: URL HotFlip (CharCNN / CharTransformer) batched
# -------------------------------------------------------------------
def hotflip_step(ids, y, model, emb_module, freeze_pos0=True, topk=1, device="cuda"):
    # We need grads, so make sure grad is enabled
    model.train()
    ids = ids.clone().to(device).long()
    y_t = y.to(device).float()

    loss_fn = nn.BCEWithLogitsLoss()
    model.zero_grad(set_to_none=True)

    if isinstance(model, CharCNN):
        # Embed with grad + retain grad on non-leaf
        E = model.emb(ids)
        E.requires_grad_(True)
        E.retain_grad()

        x = E.transpose(1, 2)                       # (B,D,L)
        x = model.act(model.bn1(model.conv1(x)))
        x = model.act(model.bn2(model.conv2(x)))
        x = torch.amax(x, dim=2)
        x = model.drop(x)
        logits = model.head(x).squeeze(-1)

    elif isinstance(model, CharTransformer):
        B, L = ids.shape
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        E = model.emb(ids)
        E.requires_grad_(True)
        E.retain_grad()

        z = E + model.pos(pos_idx)
        z = model.enc(z)
        z = model.norm(z)
        pooled = z[:, 0, :] if (ids[:, 0] == CLS_ID).all() else z.mean(dim=1)
        logits = model.head(pooled).squeeze(-1)

    else:
        raise RuntimeError("Unsupported URL model")

    loss = loss_fn(logits, y_t)
    loss.backward()                                  # populates E.grad (because we retained it)
    G = E.grad.detach()                              # (B,L,D)

    # HotFlip selection
    W = emb_module.weight.detach()                   # (V,D)
    B, L, D = G.shape
    ids_new = ids.clone()

    pos_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    if freeze_pos0:
        pos_mask[:, 0] = False
    pos_mask &= (ids != PAD_ID)

    curW = W[ids]                                     # (B,L,D)
    # score(v) = <W_v - W_cur, grad>
    scores = torch.einsum("vd,bld->blv", W, G) - torch.einsum("bld,bld->bl", curW, G)[..., None]

    # forbid staying the same / PAD token
    scores.scatter_(2, ids.unsqueeze(-1), float("-inf"))
    scores[:, :, PAD_ID] = float("-inf")

    gains, cand = scores.max(dim=2)                   # (B,L)
    gains = gains.masked_fill(~pos_mask, float("-inf"))
    k = min(topk, L)
    top_gain, top_pos = torch.topk(gains, k=k, dim=1)

    for b in range(B):
        for j in range(k):
            if top_gain[b, j].item() > 0:
                p = int(top_pos[b, j].item())
                new_id = int(cand[b, p].item())
                ids_new[b, p] = new_id

    return ids_new.detach()

def attack_url_hotflip_batched(model, ids_np, y_np, steps, topk, device, freeze_pos0=True, batch_size=1024):
    """Multi-step HotFlip (FGSM when steps=1) with OOM-safe batching."""
    N = ids_np.shape[0]
    bs = max(1, int(batch_size))
    out = np.empty_like(ids_np)

    i = 0
    while i < N:
        j = min(N, i + bs)
        try:
            ids = torch.from_numpy(ids_np[i:j]).long().to(device)
            y   = torch.from_numpy(y_np[i:j]).long().to(device)
            ids_adv = ids.clone()
            for _ in range(max(1, int(steps))):
                ids_adv = hotflip_step(
                    ids_adv, y, model, model.emb,
                    freeze_pos0=freeze_pos0, topk=topk, device=device
                )
            out[i:j] = ids_adv.detach().cpu().numpy()
            i = j
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and bs > 1 and torch.cuda.is_available():
                cleanup_cuda()
                bs = max(1, bs // 2)
                print(f"[WARN] OOM during HotFlip; retrying with batch_size={bs}")
                continue
            raise

    # set eval mode back (just in case caller assumes it)
    model.eval()
    return out

# -------------------------------------------------------------------
# Surrogate for black-box teachers (tabular)
# -------------------------------------------------------------------
class SurrogateMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256,128,64), p_drop=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
            prev = h
        self.f = nn.Sequential(*layers, nn.Linear(prev, 1))
    def forward(self, x): return self.f(x).squeeze(-1)

def distill_surrogate(global_surrogate, teacher, Xz, y, device, tag, epochs=6, lr=3e-4, wd=1e-5):
    if hasattr(teacher, "predict_proba"):
        soft = teacher.predict_proba(Xz)[:,1].astype(np.float32)
    elif hasattr(teacher, "decision_function"):
        df = teacher.decision_function(Xz).astype(np.float32)
        soft = 1.0 / (1.0 + np.exp(-df))
    else:
        soft = teacher.predict(Xz).astype(np.float32)

    model = SurrogateMLP(in_dim=Xz.shape[1])
    model.load_state_dict(global_surrogate.state_dict())
    model.to(device)
    model.train()

    ds = TensorDataset(torch.from_numpy(Xz).float(), torch.from_numpy(soft).float())
    dl = DataLoader(ds, batch_size=8192, shuffle=True, num_workers=CFG["num_workers"], pin_memory=(device.type=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossbce = nn.BCEWithLogitsLoss()

    best_state = None; best_loss = float("inf"); patience=2; noimp=0
    for ep in range(1, epochs+1):
        run=0.0; n=0
        for xb, sb in dl:
            xb = xb.to(device); sb=sb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = lossbce(logits, sb)
            loss.backward(); opt.step()
            run += loss.item() * xb.size(0); n+=xb.size(0)
        avg = run/max(1,n)
        print(f"[Surrogate {tag}] epoch {ep}/{epochs} distill loss={avg:.6f}")
        if avg < best_loss - 1e-5:
            best_loss = avg; best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}; noimp=0
        else:
            noimp += 1
            if noimp >= patience: break
    if best_state is None:
        best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k,v in best_state.items()})
    model.eval()
    return model

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    models_root = Path(CFG["models_dir"])
    results_root = Path(CFG["results_dir"])
    ensure_dir(results_root)

    # ---------------- Tabular data ----------------
    Xz_all, y_all, feat_cols, (z_min, z_max) = load_tabular_scaled(CFG)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        Xz_all, y_all, test_size=(1.0-CFG["train_ratio"]),
        random_state=CFG["random_state"], stratify=y_all
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=(CFG["test_ratio"]/(CFG["val_ratio"]+CFG["test_ratio"])),
        random_state=CFG["random_state"], stratify=y_tmp
    )
    if CFG["max_attack_samples_tab"] and X_te.shape[0] > CFG["max_attack_samples_tab"]:
        rng = np.random.default_rng(CFG["random_state"])
        idx = rng.choice(np.arange(X_te.shape[0]), size=CFG["max_attack_samples_tab"], replace=False)
        X_te_sub, y_te_sub = X_te[idx], y_te[idx]
    else:
        X_te_sub, y_te_sub = X_te, y_te

    # ---------------- URL text data ----------------
    if Path(CFG["url_csv"]).exists():
        Xids_te, y_text_te, max_len, use_cls = load_url_dataset(CFG, subset_max=CFG["max_attack_samples_text"])
    else:
        Xids_te, y_text_te, max_len, use_cls = None, None, None, None

    # ---------------- Surrogate global (for black-box teachers) ----------------
    print("[INFO] Preparing global surrogate for tabular transfer attacks...")
    global_surr = SurrogateMLP(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(global_surr.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_bce = nn.BCEWithLogitsLoss()
    ds_g = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr.astype(np.float32)))
    dl_g = DataLoader(ds_g, batch_size=8192, shuffle=True, num_workers=CFG["num_workers"], pin_memory=(device.type=="cuda"))
    global_surr.train()
    best = None; best_loss=float("inf")
    for ep in range(1,6):
        run=0.0; n=0
        for xb, yb in dl_g:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = global_surr(xb)
            loss = loss_bce(logits, yb)
            loss.backward(); opt.step()
            run += loss.item()*xb.size(0); n+=xb.size(0)
        avg=run/max(1,n)
        print(f"[GlobalSurrogate] epoch {ep}/5 loss={avg:.6f}")
        if avg<best_loss: best_loss=avg; best={k:v.detach().cpu() for k,v in global_surr.state_dict().items()}
    global_surr.load_state_dict({k:v.to(device) for k,v in best.items()})
    global_surr.eval()
    cleanup_cuda()

    # ---------------- Model registry ----------------
    model_list = [
        "DL-MLP",
        "DL-FTTransformer",
        "DL-CharCNN",
        "DL-CharTransformer",
        "Logistic_Regression",
        "Calibrated_LinearSVC",
        "Decision_Tree",
        "Random_Forest",
        "LightGBM",
        "XGBoost",
        "AdaBoost",
        "Gaussian_Naive_Bayes",
    ]

    for name in model_list:
        mdir = models_root / name
        if not mdir.exists():
            print(f"[WARN] Skipping {name}: not found at {mdir}")
            continue

        print(f"\n==== Attacking: {name} ====")
        out_base = ensure_dir(results_root / name)

        # ---------- load model ----------
        is_tab_dl = name in {"DL-MLP","DL-FTTransformer"}
        is_url_dl = name in {"DL-CharCNN","DL-CharTransformer"}
        is_linear = name in {"Logistic_Regression","Calibrated_LinearSVC"}
        is_blackbox = name in {"Decision_Tree","Random_Forest","LightGBM","XGBoost","AdaBoost","Gaussian_Naive_Bayes"}

        if is_tab_dl:
            if name == "DL-MLP":
                mdl = MLPNet(in_dim=X_tr.shape[1]).to(device)
            else:
                mdl = FTTransformer(n_features=X_tr.shape[1], d_model=64, n_heads=8, n_layers=3, p_drop=0.1).to(device)
            state = torch.load(mdir / "model.pt", map_location=device)
            mdl.load_state_dict(state); mdl.eval()

        elif is_url_dl:
            assert Xids_te is not None, "URL dataset not available for URL models"
            if name == "DL-CharCNN":
                mdl = CharCNN(vocab_size=VOCAB_SIZE, d_model=64, p_drop=0.2).to(device)
            else:
                mdl = CharTransformer(vocab_size=VOCAB_SIZE, d_model=96, n_heads=6, n_layers=4, max_len=max_len, p_drop=0.1).to(device)
            state = torch.load(mdir / "model.pt", map_location=device)
            mdl.load_state_dict(state); mdl.eval()

        else:
            mdl = joblib.load(mdir / "model.joblib")

        # ---------- baseline on test ----------
        if is_tab_dl:
            y_pred, y_score = eval_torch_batched(mdl, X_te_sub, y_te_sub, device, CFG["tab_eval_bs"])
        elif is_url_dl:
            y_pred, y_score = eval_url_torch_batched(mdl, Xids_te, y_text_te, device, CFG["text_eval_bs"])
        else:
            y_pred, y_score = eval_sklearn(mdl, X_te_sub, y_te_sub)
        nat_dir = ensure_dir(out_base / "adv_NAT")
        metrics_to_files(name, (y_text_te if is_url_dl else y_te_sub), y_pred, y_score, nat_dir, "NAT")

        # ---------- FGSM ----------
        fgsm_dir = ensure_dir(out_base / "adv_FGSM")
        if is_tab_dl:
            X_adv = attack_tabular_torch_batched(
                mdl, X_te_sub, y_te_sub,
                eps=ATTACKCFG["eps_tab_fgsm"], alpha=ATTACKCFG["eps_tab_fgsm"], steps=0,
                device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
            )
            y_pred_a, y_score_a = eval_torch_batched(mdl, X_adv, y_te_sub, device, CFG["tab_eval_bs"])

        elif is_linear:
            X_adv = linear_wb_attack(
                mdl, X_te_sub, y_te_sub,
                eps=ATTACKCFG["eps_lin_fgsm"], alpha=ATTACKCFG["eps_lin_fgsm"], steps=0
            )
            y_pred_a, y_score_a = eval_sklearn(mdl, X_adv, y_te_sub)

        elif is_blackbox:
            surr = distill_surrogate(global_surr, mdl, X_tr, y_tr, device, tag=name, epochs=6)
            X_adv = attack_tabular_torch_batched(
                surr, X_te_sub, y_te_sub,
                eps=ATTACKCFG["eps_tab_fgsm"], alpha=ATTACKCFG["eps_tab_fgsm"], steps=0,
                device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
            )
            y_pred_a, y_score_a = eval_sklearn(mdl, X_adv, y_te_sub)

        elif is_url_dl:
            ids_adv = attack_url_hotflip_batched(
                mdl, Xids_te, y_text_te,
                steps=ATTACKCFG["char_steps_fgsm"], topk=ATTACKCFG["char_topk_per_step"],
                device=device, freeze_pos0=ATTACKCFG["freeze_cls_pos0"],
                batch_size=CFG["text_attack_bs"]
            )
            y_pred_a, y_score_a = eval_url_torch_batched(mdl, ids_adv, y_text_te, device, CFG["text_eval_bs"])

        metrics_to_files(name, (y_text_te if is_url_dl else y_te_sub), y_pred_a, y_score_a, fgsm_dir, "FGSM")
        save_json({"attack":"FGSM","cfg":ATTACKCFG}, fgsm_dir / "attack_config.json")
        cleanup_cuda()

        # ---------- PGD ----------
        pgd_dir = ensure_dir(out_base / "adv_PGD")
        if is_tab_dl:
            X_adv = attack_tabular_torch_batched(
                mdl, X_te_sub, y_te_sub,
                eps=ATTACKCFG["eps_tab_pgd"], alpha=ATTACKCFG["alpha_tab_pgd"], steps=ATTACKCFG["steps_tab_pgd"],
                device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
            )
            y_pred_a, y_score_a = eval_torch_batched(mdl, X_adv, y_te_sub, device, CFG["tab_eval_bs"])

        elif is_linear:
            X_adv = linear_wb_attack(
                mdl, X_te_sub, y_te_sub,
                eps=ATTACKCFG["eps_lin_pgd"], alpha=ATTACKCFG["alpha_lin_pgd"], steps=ATTACKCFG["steps_lin_pgd"]
            )
            y_pred_a, y_score_a = eval_sklearn(mdl, X_adv, y_te_sub)

        elif is_blackbox:
            surr = distill_surrogate(global_surr, mdl, X_tr, y_tr, device, tag=name, epochs=6)
            X_adv = attack_tabular_torch_batched(
                surr, X_te_sub, y_te_sub,
                eps=ATTACKCFG["eps_tab_pgd"], alpha=ATTACKCFG["alpha_tab_pgd"], steps=ATTACKCFG["steps_tab_pgd"],
                device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
            )
            y_pred_a, y_score_a = eval_sklearn(mdl, X_adv, y_te_sub)

        elif is_url_dl:
            ids_adv = attack_url_hotflip_batched(
                mdl, Xids_te, y_text_te,
                steps=ATTACKCFG["char_steps_pgd"], topk=ATTACKCFG["char_topk_per_step"],
                device=device, freeze_pos0=ATTACKCFG["freeze_cls_pos0"],
                batch_size=CFG["text_attack_bs"]
            )
            y_pred_a, y_score_a = eval_url_torch_batched(mdl, ids_adv, y_text_te, device, CFG["text_eval_bs"])

        metrics_to_files(name, (y_text_te if is_url_dl else y_te_sub), y_pred_a, y_score_a, pgd_dir, "PGD")
        save_json({"attack":"PGD","cfg":ATTACKCFG}, pgd_dir / "attack_config.json")
        cleanup_cuda()

    print("\n[DONE] Adversarial evaluation complete.")


if __name__ == "__main__":
    main()

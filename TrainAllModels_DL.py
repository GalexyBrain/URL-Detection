"""
All-in-one Deep Learning trainer (Python 3.13+).
- Tabular DL: DL-MLP, DL-FTTransformer (from features_extracted.csv)
- URL Text DL: DL-CharCNN, DL-CharTransformer (from merged_url_dataset.csv)
- Uses GPU if available + mixed precision.
- Early stopping on validation AUC.
- Saves artifacts compatible with your ML pipeline:
    models/<ModelName>/model.pt
    results/<ModelName>/{metrics.json, classification_report.txt, confusion_matrix.png, roc_curve.png}
- Global artifacts:
    models/_global/scaler.joblib
    models/_global/feature_columns.json
    models/_global/url_tokenizer.json
"""

from pathlib import Path
from datetime import datetime, timezone
import os, gc, json, math, warnings

# Less CUDA fragmentation; must be set before first CUDA alloc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Speed wins on Ampere+ without accuracy loss for this task
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ---------------- Config ----------------
CONFIG = {
    # ---- Tabular (numeric features) ----
    "tabular_csv": "features_extracted.csv",  # numeric engineered features + label (0/1)
    "label_col": "label",
    "url_col": "url",  # ignored by tabular models if present

    # ---- URL text (raw url,label) ----
    "url_csv": "merged_url_dataset.csv",      # raw URL + label (benign/malicious)

    # ---- I/O ----
    "models_dir": "models",
    "results_dir": "results",
    "random_state": 42,

    # ---- Split: 70/10/20 ----
    "train_ratio": 0.70,
    "val_ratio":   0.10,
    "test_ratio":  0.20,

    # ---- Training ----
    "epochs_tabular": 10,
    "epochs_text": 5,

    # Tabular batch sizes
    "batch_size_gpu_tabular": 8192,
    "batch_size_cpu_tabular": 8192,

    # Text batch sizes (per model; tuned for 4GB GPUs)
    "batch_size_gpu_text_cnn": 4096,
    "batch_size_gpu_text_transformer": 1024,
    "batch_size_cpu_text": 8192,

    # Gradient accumulation (simulates larger effective batch without OOM)
    "grad_accum_tabular": 1,
    "grad_accum_text": 4,

    "lr": 3e-4,
    "weight_decay": 1e-5,
    "patience": 5,       # tabular early stopping
    "patience_text": 4,  # text early stopping
    "min_delta_auc": 1e-4,

    # ---- URL tokenizer ----
    "max_len_cap": 256,
    "use_cls_token": True,
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------- Utils --------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

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

def metrics_to_files(name, y_true, y_pred, y_score, out_dir: Path):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None
    metrics = {
        "model": name,
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
    header = f"Performance Analysis for: {name}\n" + "=" * (24 + len(name)) + "\n"
    body = (
        f"Samples (evaluated):\t{len(y_true)}\n"
        f"Accuracy:\t\t{metrics['accuracy']}\n"
        f"Precision:\t\t{metrics['precision']}\n"
        f"Recall:\t\t\t{metrics['recall']}\n"
        f"F1-Score:\t\t{metrics['f1_score']}\n"
        f"AUC-ROC Score:\t\t{metrics['auc_roc']}\n"
        + "=" * (24 + len(name)) + "\n\n"
        "Detailed Classification Report\n"
        "------------------------------\n" + report
    )
    save_text(body, out_dir / "classification_report.txt")
    plot_confusion(y_true, y_pred, out_dir / "confusion_matrix.png", f"Confusion Matrix: {name}")
    try:
        plot_roc(y_true, y_score, out_dir / "roc_curve.png", f"ROC Curve: {name}")
    except Exception:
        pass

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------- Tabular DL Models --------------
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

    def forward(self, x):
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

    def forward(self, x):  # (B,F)
        B, F = x.shape
        val_tok = self.value_emb(x.view(B*F,1)).view(B, F, -1)
        idx = torch.arange(F, device=x.device).view(1, F)
        feat_tok = self.feat_emb(idx).expand(B, -1, -1)
        z = self.encoder(val_tok + feat_tok)
        z = self.norm(z).mean(dim=1)
        return self.head(z).squeeze(-1)

# -------------- URL Text DL Models --------------
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
    p99 = int(np.percentile(lens, 99))
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
        x = self.emb(x_ids).transpose(1,2)     # (B,D,L)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = torch.amax(x, dim=2)               # global max pool
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

# -------------- Training Loops --------------
@torch.no_grad()
def infer_proba_scalar(model, loader, device, is_text: bool):
    model.eval()
    sig = nn.Sigmoid()
    outs, labs = [], []
    for xb, yb in loader:
        if is_text:
            xb = xb.to(device, non_blocking=True).long()
        else:
            xb = xb.to(device, non_blocking=True).float()
        logits = model(xb)
        probs = sig(logits).detach().cpu().numpy()
        outs.append(probs);  labs.append(yb.numpy())
    return np.concatenate(outs), np.concatenate(labs)

def train_model(name, model, train_loader, val_loader, test_loader, device, out_model_dir, out_res_dir, cfg, is_text=False, patience=5, epochs=20, grad_accum=1):
    # class imbalance -> pos_weight
    if isinstance(train_loader.dataset, TensorDataset):
        y_tr = train_loader.dataset.tensors[1].cpu().numpy()
    else:  # URLDataset
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
            if is_text:
                xb = xb.to(device, non_blocking=True).long()
            else:
                xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(xb)
                raw_loss = criterion(logits, yb)
                loss = raw_loss / max(1, grad_accum)

            scaler.scale(loss).backward()
            step += 1
            if step % max(1, grad_accum) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running += raw_loss.item() * xb.size(0)

        # flush leftover grads if any
        if step % max(1, grad_accum) != 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

        # Validation AUC for early stopping
        val_probs, val_y = infer_proba_scalar(model, val_loader, device, is_text)
        try:
            val_auc = roc_auc_score(val_y, val_probs)
        except Exception:
            val_auc = float("nan")

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
    ensure_dir(out_model_dir)
    torch.save(best_state, Path(out_model_dir) / "model.pt")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # Test
    test_probs, test_y = infer_proba_scalar(model, test_loader, device, is_text)
    test_pred = (test_probs >= 0.5).astype(np.int32)
    ensure_dir(out_res_dir)
    metrics_to_files(name, test_y, test_pred, test_probs, Path(out_res_dir))
    print(f"[OK] Saved model -> {out_model_dir}")
    print(f"[OK] Saved results -> {out_res_dir}")

# -------------- Main --------------
def main():
    cfg = CONFIG
    rs = int(cfg["random_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    models_root = ensure_dir(Path(cfg["models_dir"]))
    results_root = ensure_dir(Path(cfg["results_dir"]))
    global_dir = ensure_dir(models_root / "_global")

    # ===================== Tabular DL =====================
    tab_csv = Path(cfg["tabular_csv"])
    if tab_csv.exists():
        print(f"\n[Tabular] Loading {tab_csv} ...")
        df_tab = pd.read_csv(tab_csv)

        # Map label to {0,1} robustly (handles 'benign'/'malicious' etc.)
        assert cfg["label_col"] in df_tab.columns, f"Missing label column: {cfg['label_col']}"
        df_tab[cfg["label_col"]] = df_tab[cfg["label_col"]].apply(map_label).astype(np.int64)

        # Ignore raw URL column for tabular features
        drop_cols = []
        if cfg["url_col"] in df_tab.columns:
            print(f"[Tabular] Found '{cfg['url_col']}' column; ignoring for tabular models.")
            drop_cols.append(cfg["url_col"])
        df_feat = df_tab.drop(columns=drop_cols + [cfg["label_col"]], errors="ignore")

        # Keep numeric-only features; coerce others; clean NaN/inf
        df_feat = df_feat.apply(pd.to_numeric, errors="coerce")
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        if df_feat.isna().any().any():
            df_feat = df_feat.fillna(df_feat.median(numeric_only=True))

        feature_cols = df_feat.columns.tolist()
        X_all = df_feat.to_numpy(dtype=np.float32)
        y_all = df_tab[cfg["label_col"]].to_numpy(dtype=np.int64)
        assert set(np.unique(y_all)) <= {0,1}, "Tabular labels must be {0,1}"

        # global scaler: load if present, else fit+save (consistent with ML script)
        scaler_path = global_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print("[Tabular] Loaded existing global scaler.")
            X_all = scaler.transform(X_all)
        else:
            scaler = StandardScaler()
            X_all = scaler.fit_transform(X_all)
            joblib.dump(scaler, scaler_path)
            print("[Tabular] Fitted and saved new global scaler.")
        save_json({"feature_columns": feature_cols}, global_dir / "feature_columns.json")

        # split
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_all, y_all, test_size=(1.0-cfg["train_ratio"]), random_state=rs, stratify=y_all)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=(cfg["test_ratio"]/(cfg["val_ratio"]+cfg["test_ratio"])), random_state=rs, stratify=y_tmp)
        print(f"[Tabular] Shapes -> train {X_tr.shape}, val {X_val.shape}, test {X_te.shape}")
        cls_w = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr)
        print(f"[Tabular] Class weights (0,1): {cls_w}")

        bs_tab = cfg["batch_size_gpu_tabular"] if device.type=="cuda" else cfg["batch_size_cpu_tabular"]
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=bs_tab, shuffle=True, num_workers=4, pin_memory=(device.type=="cuda"))
        val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=bs_tab, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))
        test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=bs_tab, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))

        # DL-MLP
        name = "DL-MLP"
        mdl = MLPNet(in_dim=X_all.shape[1], hidden=(256,128,64), p_drop=0.2).to(device)
        train_model(name, mdl, train_loader, val_loader, test_loader, device,
                    models_root / safe_name(name), results_root / safe_name(name),
                    cfg, is_text=False, patience=cfg["patience"], epochs=cfg["epochs_tabular"],
                    grad_accum=cfg["grad_accum_tabular"])
        del mdl, train_loader, val_loader, test_loader; cleanup_cuda()

        # DL-FTTransformer
        name = "DL-FTTransformer"
        bs_tab = cfg["batch_size_gpu_tabular"] if device.type=="cuda" else cfg["batch_size_cpu_tabular"]
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=bs_tab, shuffle=True, num_workers=4, pin_memory=(device.type=="cuda"))
        val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=bs_tab, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))
        test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=bs_tab, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))
        mdl = FTTransformer(n_features=X_all.shape[1], d_model=64, n_heads=8, n_layers=3, p_drop=0.1).to(device)
        train_model(name, mdl, train_loader, val_loader, test_loader, device,
                    models_root / safe_name(name), results_root / safe_name(name),
                    cfg, is_text=False, patience=cfg["patience"], epochs=cfg["epochs_tabular"],
                    grad_accum=cfg["grad_accum_tabular"])
        del mdl, train_loader, val_loader, test_loader; cleanup_cuda()
    else:
        print(f"[Tabular] Skipped: {tab_csv} not found.")

    # ===================== URL Text DL =====================
    url_csv = Path(cfg["url_csv"])
    if url_csv.exists():
        print(f"\n[Text] Loading {url_csv} ...")
        df_url = pd.read_csv(url_csv)
        assert "url" in df_url.columns and "label" in df_url.columns, "URL CSV must have columns: url,label"

        urls = df_url["url"].astype(str).tolist()
        labels = [map_label(v) for v in df_url["label"].tolist()]
        y_all = np.array(labels, dtype=np.int64)
        assert set(np.unique(y_all)) <= {0,1}, "URL labels must map to {0,1}"

        max_len = pick_max_len(urls, cap=cfg["max_len_cap"], use_cls=cfg["use_cls_token"])
        print(f"[Text] Using max_len={max_len} (cap={cfg['max_len_cap']})")
        save_json({
            "tokenizer": "byte", "vocab_size": VOCAB_SIZE,
            "pad_id": PAD_ID, "cls_id": CLS_ID,
            "use_cls_token": bool(cfg["use_cls_token"]),
            "max_len": int(max_len)
        }, global_dir / "url_tokenizer.json")

        # split
        U_tr, U_tmp, y_tr, y_tmp = train_test_split(urls, y_all, test_size=(1.0-cfg["train_ratio"]), random_state=rs, stratify=y_all)
        U_val, U_te, y_val, y_te = train_test_split(U_tmp, y_tmp, test_size=(cfg["test_ratio"]/(cfg["val_ratio"]+cfg["test_ratio"])), random_state=rs, stratify=y_tmp)
        print(f"[Text] Sizes -> train {len(U_tr)}, val {len(U_val)}, test {len(U_te)}")
        cls_w = compute_class_weight("balanced", classes=np.array([0,1]), y=y_tr)
        print(f"[Text] Class weights (0,1): {cls_w}")

        # Build shared datasets once (so we can vary batch size per model)
        ds_train = URLDataset(U_tr, y_tr, max_len, cfg["use_cls_token"])
        ds_val   = URLDataset(U_val, y_val, max_len, cfg["use_cls_token"])
        ds_test  = URLDataset(U_te, y_te, max_len, cfg["use_cls_token"])

        # -------- DL-CharCNN --------
        name = "DL-CharCNN"
        bs_txt = cfg["batch_size_gpu_text_cnn"] if device.type=="cuda" else cfg["batch_size_cpu_text"]
        train_loader = DataLoader(ds_train, batch_size=bs_txt, shuffle=True,  num_workers=4, pin_memory=(device.type=="cuda"))
        val_loader   = DataLoader(ds_val,   batch_size=bs_txt, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))
        test_loader  = DataLoader(ds_test,  batch_size=bs_txt, shuffle=False, num_workers=4, pin_memory=(device.type=="cuda"))
        mdl = CharCNN(vocab_size=VOCAB_SIZE, d_model=64, p_drop=0.2).to(device)
        train_model(name, mdl, train_loader, val_loader, test_loader, device,
                    models_root / safe_name(name), results_root / safe_name(name),
                    cfg, is_text=True, patience=cfg["patience_text"], epochs=cfg["epochs_text"],
                    grad_accum=cfg["grad_accum_text"])
        del mdl, train_loader, val_loader, test_loader; cleanup_cuda()

        # -------- DL-CharTransformer --------
        name = "DL-CharTransformer"
        bs_txt = cfg["batch_size_gpu_text_transformer"] if device.type=="cuda" else cfg["batch_size_cpu_text"]
        train_loader = DataLoader(ds_train, batch_size=bs_txt, shuffle=True,  num_workers=2, pin_memory=(device.type=="cuda"))
        val_loader   = DataLoader(ds_val,   batch_size=bs_txt, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
        test_loader  = DataLoader(ds_test,  batch_size=bs_txt, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
        mdl = CharTransformer(vocab_size=VOCAB_SIZE, d_model=96, n_heads=6, n_layers=4, max_len=max_len, p_drop=0.1).to(device)
        train_model(name, mdl, train_loader, val_loader, test_loader, device,
                    models_root / safe_name(name), results_root / safe_name(name),
                    cfg, is_text=True, patience=cfg["patience_text"], epochs=cfg["epochs_text"],
                    grad_accum=cfg["grad_accum_text"])
        del mdl, train_loader, val_loader, test_loader; cleanup_cuda()
    else:
        print(f"[Text] Skipped: {url_csv} not found.")

    print("\n[DONE] All DL models trained and artifacts saved.")

if __name__ == "__main__":
    main()

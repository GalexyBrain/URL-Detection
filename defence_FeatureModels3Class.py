#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-class (0=benign, 1=malicious, 2=ADV) evaluation.

- Binary BASE models are used *only* to craft ADV examples (FGSM/PGD).
- LightGBM defense gate provides p_adv to choose a lenient tau_3c.
- Evaluate the 3-class model in two scenarios:
    (A) No defense (full mixed NAT+ADV)                -> metrics_before_defense.json
    (B) After defense with lenient tau_3c (accepted)   -> metrics_after_defense.json

Adds:
- Robust OOM handling for DL Transformer/MLP: adaptive CUDA batch backoff → CPU fallback.
- NAT-only (0/1) metrics + confusion (before/after defense) with correct totals even if preds==2.
- Robust class-order handling for sklearn 3-class models.
- Confusion matrices + accuracy_from_confusion sanity checks.
- Uses the 3-class counterpart of the ATTACK base model (per-base pairing only).

Artifacts:
  results_defence_features_3class/<BaseUsedForAttack>/(mixed_FGSM|mixed_PGD)/
    - metrics_before_defense.json
    - metrics_after_defense.json
    - confusion_no_def.png
    - confusion_after_def.png
    - confusion_nat_no_def.png
    - confusion_nat_after_def.png
    - roc_ovr_no_def.png (if probs available)
    - roc_ovr_after_def.png (if probs available)
    - classification_report_no_def.txt
    - classification_report_after_def.txt
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, gc, warnings, argparse, hashlib, math, traceback
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Threads / CUDA alloc
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Config --------------------
CFG = {
    # DATA / MODELS
    "models_dir": "models",                         # base (binary) models live here
    "defense_dir": "models/Defense-LGBM",          # gate
    "dataset_csv": "features_extracted.csv",       # same dataset used by base models (label in {0,1})
    "label_col": "label",

    # RESULTS
    "results_dir": "results_defence_features_3class",

    # ATTACK budget in z-space of base scaler
    "eps_z": 0.40,
    "alpha_z": 0.10,
    "steps_fgsm": 1,
    "steps_pgd": 5,

    # surrogate for non-linear sklearn base attacks
    "surrogate_hidden": 256,
    "surrogate_epochs_global": 5,
    "surrogate_epochs_finetune": 6,
    "surrogate_batch": 4096,
    "surrogate_sample_limit": 150_000,

    # batching
    "eval_bs": 100_000,     # sklearn scoring & defense
    "lin_attack_bs": 32_768,
    "bb_attack_bs": 8_192,
    "torch_eval_bs": 8192,   # torch eval (auto-backoff + CPU fallback on OOM)
    "torch_attack_bs": 4096, # torch attack (auto-backoff + CPU fallback on OOM)

    # device
    "use_gpu": True,

    # stream composition
    "NAT_FRAC": 0.5,
    "STREAM_SIZE": 1_000_000,

    # 3-class pipeline (ADV-aware)
    "models_base3_dir": "models_base3",
    "tau_quantile_3c": 0.995,    # aim ~99.5% NAT pass on NAT dist
    "tau_abs_3c": None,          # if set, overrides quantile
    "tau_target_accept_nat": None,# if set, pick τ so >= this fraction NAT pass
    "classifier3c": "auto",      # override; otherwise pair with base
    "threeclass_eval": True,

    # verbosity / speed
    "use_progress": True,
    "use_plots": True,
    "use_amp": False,
    "cache_adv": False,

    # LightGBM booster fast predict
    "FAST_PREDICT": False,
    "EARLY_STOP_MARGIN": 2.0,

    # RNG
    "seed": 42,
}

LABELS3 = [0, 1, 2]
LABELS_NAT = [0, 1]  # for per-class metrics averaging
NAT_CM_LABELS = [0, 1, 2]  # for confusion so preds==2 are counted

# -------------------- Helpers --------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(txt, p: Path):
    p.write_text(txt, encoding="utf-8")

def _file_md5(path: Path) -> str:
    if not path.exists(): return "missing"
    import hashlib as _hashlib
    md5 = _hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            md5.update(chunk)
    return md5.hexdigest()

def plot_confusion_any(y_true, y_pred, out_path: Path, title: str, labels):
    if len(y_true) == 0: return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax)
    ax.set_title(title); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def plot_confusion_3c(y_true, y_pred, out_path: Path, title: str):
    plot_confusion_any(y_true, y_pred, out_path, title, LABELS3)

def plot_confusion_nat(y_true, y_pred, out_path: Path, title: str):
    # use [0,1,2] so mispreds to 2 are included in totals
    plot_confusion_any(y_true, y_pred, out_path, title, NAT_CM_LABELS)

def plot_roc_multiclass(y_true, prob_mat, out_path: Path, title: str):
    try:
        y_true = np.asarray(y_true)
        if prob_mat is None or len(y_true) == 0: return
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ok = False
        for cls in LABELS3:
            y_bin = (y_true == cls).astype(int)
            try:
                RocCurveDisplay.from_predictions(y_bin, prob_mat[:, cls], name=f"class {cls}", ax=ax)
                ok = True
            except Exception:
                pass
        if ok:
            ax.set_title(title); fig.tight_layout(); fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

def cm_as_dict(y_true, y_pred, labels):
    """
    Return confusion + accuracy_from_confusion using len(y_true) as denominator.
    IMPORTANT: pass labels that include any possible predicted class you care about.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(y_true.shape[0])
    acc_from_cm = float(np.trace(cm) / max(1, total))
    return {"labels": list(map(int, labels)), "matrix": cm.tolist(), "total": total, "accuracy_from_confusion": acc_from_cm}

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------- Load globals --------------------
def load_globals(models_root: Path):
    gdir = models_root / "_global"
    scaler: StandardScaler = joblib.load(gdir / "scaler.joblib")
    feat_info = json.loads((gdir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_cols = feat_info["feature_columns"] if isinstance(feat_info, dict) else feat_info
    return scaler, list(feature_cols)

def load_dataset(csv_path: str, feature_cols: list[str], label_col: str):
    df = pd.read_csv(csv_path)
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column '{c}' missing in {csv_path}")
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_base = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    return X_raw, y_base

# -------------------- Base (binary) models --------------------
def unwrap_calibrated(estimator):
    if isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
            inner = estimator.calibrated_classifiers_[0]
            base = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
            if base is not None: return base
    return estimator

def linear_params(estimator):
    est = unwrap_calibrated(estimator)
    if hasattr(est, "coef_") and hasattr(est, "intercept_"):
        w = np.array(est.coef_, dtype=np.float32).reshape(1, -1)
        b = np.array(getattr(est, "intercept_", np.zeros(w.shape[1], dtype=np.float32)), dtype=np.float32).reshape(-1)
        w = (w[0] if w.shape[0] == 1 else np.mean(w, axis=0)).astype(np.float32, copy=False)
        b = np.float32(b[0] if b.size else 0.0)
        return w, b
    raise RuntimeError("Linear parameters not found for white-box attack.")

def attack_linear_fgsm_pgd(estimator, Xz, y, steps, eps, alpha, loss_kind="logistic", batch=32_768,
                           desc="lin/attack", use_progress=True):
    w, b = linear_params(estimator)
    sign_w = np.sign(w).astype(np.float32)
    X_adv = Xz.copy().astype(np.float32, copy=False); X0 = Xz.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    if loss_kind == "hinge": ypm = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    N = Xz.shape[0]; steps = max(1, int(steps)); eps = np.float32(eps); alpha = np.float32(alpha)
    s = 0; bs = int(batch); total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while s < N:
        e = min(N, s + bs)
        Xa = X_adv[s:e]; Xb = X0[s:e]
        for _ in range(steps):
            z = Xa @ w + b
            if loss_kind == "logistic":
                p = 1.0 / (1.0 + np.exp(-z))
                residual = p - y[s:e]
                step = (np.sign(residual)[:, None] * sign_w[None, :])
            else:
                ypm_se = ypm[s:e]
                m = ypm_se * z
                mask = (m < 1.0).astype(np.float32)
                step = mask[:, None] * ((-ypm_se)[:, None] * sign_w[None, :])
            Xa = Xa + alpha * step
            Xa = np.clip(Xa, Xb - eps, Xb + eps)
        X_adv[s:e] = Xa; s = e; pbar.update(1)
    pbar.close()
    return X_adv

class MLPNetBin(nn.Module):
    def __init__(self, in_dim: int, hidden=(256,128,64), p_drop=0.2):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(p_drop)]; prev=h
        self.backbone = nn.Sequential(*layers); self.head = nn.Linear(prev, 1)
    def forward(self, x): return self.head(self.backbone(x)).squeeze(-1)

class FTTransformerBin(nn.Module):
    def __init__(self, n_features: int, d_model=64, n_heads=8, n_layers=3, p_drop=0.1):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model, bias=False)
        self.feat_emb  = nn.Embedding(n_features, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                                         dropout=p_drop, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model, 1))
    def forward(self, x):
        B,F = x.shape
        vt = self.value_emb(x.unsqueeze(-1)).reshape(B,F,-1)
        idx = torch.arange(F, device=x.device, dtype=torch.long)
        ft = self.feat_emb(idx)[None,:,:].expand(B,-1,-1)
        z = self.encoder(vt+ft); z = self.norm(z).mean(dim=1)
        return self.head(z).squeeze(-1)

@torch.no_grad()
def eval_torch_bin(model, Xz, device, batch_size, desc, use_progress, use_amp):
    N = Xz.shape[0]; out=[]; i=0
    bs = max(1, int(batch_size))
    local_device = device
    model.eval()
    while True:
        try:
            pbar = tqdm(total=math.ceil(N/bs), desc=desc, disable=not use_progress)
            while i < N:
                j = min(N, i+bs)
                xb = torch.from_numpy(Xz[i:j]).to(local_device).float()
                with torch.cuda.amp.autocast(enabled=(use_amp and local_device.type=="cuda")):
                    logits = model(xb).float()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                del xb, logits
                out.append(probs); i=j; pbar.update(1)
            pbar.close()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and local_device.type == "cuda":
                cleanup_cuda()
                if bs > 1:
                    bs = max(1, bs // 2)
                    tqdm.write(f"[WARN] OOM in {desc}; retry with batch_size={bs}")
                    continue
                else:
                    tqdm.write(f"[WARN] OOM in {desc} at batch=1 → falling back to CPU.")
                    model.to(torch.device("cpu")); local_device = torch.device("cpu")
                    bs = max(256, bs)  # can go a bit larger on CPU
                    continue
            raise
    return np.concatenate(out, axis=0)

def attack_tabular_torch_batched(model, Xz, y, eps, alpha, steps, device, z_min, z_max, batch_size,
                                 desc="torch/attack", use_progress=True, use_amp=False):
    N, F = Xz.shape
    adv_list = []; bs = max(1, int(batch_size))
    steps_eff = 1 if steps == 0 else steps
    alpha_eff = eps if steps == 0 else alpha
    local_device = device
    zmin_t = torch.from_numpy(z_min).to(local_device)
    zmax_t = torch.from_numpy(z_max).to(local_device)
    loss_fn = nn.BCEWithLogitsLoss()
    ptr = 0
    while ptr < N:
        end = min(N, ptr + bs)
        try:
            X = torch.from_numpy(Xz[ptr:end]).to(local_device).float()
            Y = torch.from_numpy(y[ptr:end]).to(local_device).float()
            X0 = X.clone().detach()
            X_adv = X.clone().detach()
            for _ in range(steps_eff):
                X_adv.requires_grad_(True)
                with torch.cuda.amp.autocast(enabled=(use_amp and local_device.type=="cuda")):
                    logits = model(X_adv).float()
                    loss = loss_fn(logits, Y)
                model.zero_grad(set_to_none=True)
                if X_adv.grad is not None: X_adv.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    X_adv = X_adv + alpha_eff * X_adv.grad.sign()
                    X_adv = torch.clamp(X_adv, X0 - eps, X0 + eps)
                    X_adv = torch.max(torch.min(X_adv, zmax_t), zmin_t).detach()
            adv_list.append(X_adv.detach().cpu().numpy()); ptr = end
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                cleanup_cuda()
                if local_device.type == "cuda":
                    if bs > 1:
                        bs = max(1, bs // 2)
                        tqdm.write(f"[WARN] OOM during {desc}; retry with batch_size={bs}")
                        continue
                    else:
                        tqdm.write(f"[WARN] OOM during {desc} at batch=1 → falling back to CPU.")
                        model.to(torch.device("cpu"))
                        local_device = torch.device("cpu")
                        # re-create bounds on CPU
                        zmin_t = torch.from_numpy(z_min)
                        zmax_t = torch.from_numpy(z_max)
                        bs = max(128, bs)
                        continue
                else:
                    raise
            else:
                raise
    return np.concatenate(adv_list, axis=0)

# -------------------- Defense gate --------------------
def _booster_from_defense(defense):
    try:
        import lightgbm as lgb  # noqa
    except Exception:
        return None
    if isinstance(defense, lgb.Booster):
        return defense
    b = getattr(defense, "booster_", None)
    if isinstance(b, lgb.Booster):
        return b
    return None

def _predict_with_defense(def_model, X, fast_predict=False, early_stop_margin=2.0):
    booster = _booster_from_defense(def_model)
    if booster is not None:
        kw = dict(raw_score=False)
        if fast_predict:
            kw.update(dict(pred_early_stop=True, pred_early_stop_margin=early_stop_margin))
        p = booster.predict(X, **kw)  # (N,) prob of class-1
        return np.asarray(p)
    if hasattr(def_model, "predict_proba"):
        p = def_model.predict_proba(X)
        if isinstance(p, list):
            p = p[0]
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, -1]
        return p.reshape(-1)
    if hasattr(def_model, "predict"):
        try:
            p = def_model.predict(X, raw_score=False)
            return np.asarray(p).reshape(-1)
        except Exception:
            pass
        return np.asarray(def_model.predict(X)).reshape(-1)
    if hasattr(def_model, "decision_function"):
        z = np.asarray(def_model.decision_function(X)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))
    raise RuntimeError("Defense model does not support predict/predict_proba/decision_function.")

def defense_p_adv(def_model, X_raw, batch=100_000, calibrator=None, desc="defense/p_adv",
                  use_progress=True, fast_predict=False, early_stop_margin=2.0):
    N = X_raw.shape[0]; probs = np.empty(N, dtype=np.float32)
    bs = max(10_000, int(batch)); i = 0; total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while i < N:
        j = min(N, i + bs)
        xb = np.ascontiguousarray(X_raw[i:j], dtype=np.float32)
        try:
            p1 = _predict_with_defense(def_model, xb, fast_predict=fast_predict, early_stop_margin=early_stop_margin)
        except Exception as ex:
            if "MemoryError" in str(ex) or bs > 10_000:
                bs = max(10_000, bs // 2); cleanup_cuda()
                total = math.ceil((N - i) / bs); pbar.total = pbar.n + total; pbar.refresh(); continue
            pbar.close(); raise
        p1 = np.asarray(p1)
        if p1.ndim == 2 and p1.shape[1] == 2: p1 = p1[:,1]
        if p1.min() < 0 or p1.max() > 1: p1 = 1.0 / (1.0 + np.exp(-p1))
        if calibrator is not None:
            try: p1 = calibrator.predict(p1)
            except Exception: pass
        probs[i:j] = p1.astype(np.float32, copy=False); i = j; pbar.update(1)
    pbar.close(); return probs

def defense_p_adv_from_z(def_model, scaler: StandardScaler, Xz, batch=100_000, calibrator=None,
                         desc="defense/p_adv(z->raw)", use_progress=True,
                         fast_predict=False, early_stop_margin=2.0):
    N = Xz.shape[0]; out = np.empty(N, dtype=np.float32)
    bs = max(10_000, int(batch)); i = 0; total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while i < N:
        j = min(N, i + bs)
        xb_raw = scaler.inverse_transform(Xz[i:j]).astype(np.float32, copy=False)
        xb_raw = np.ascontiguousarray(xb_raw)
        p1 = _predict_with_defense(def_model, xb_raw, fast_predict=fast_predict, early_stop_margin=early_stop_margin)
        p1 = np.asarray(p1)
        if p1.ndim == 2 and p1.shape[1] == 2: p1 = p1[:,1]
        if p1.min() < 0 or p1.max() > 1: p1 = 1.0 / (1.0 + np.exp(-p1))
        if calibrator is not None:
            try: p1 = calibrator.predict(p1)
            except Exception: pass
        out[i:j] = p1.astype(np.float32, copy=False); i = j; pbar.update(1)
    pbar.close(); return out

# τ helpers
def choose_tau_from_nat_full(p_adv_nat_full: np.ndarray, q: float, tau_abs=None, target_accept=None) -> float:
    if tau_abs is not None:
        return float(np.clip(tau_abs, 0.0, 1.0))
    q = float(np.clip(q, 0.50, 0.9999))
    tau = float(np.quantile(p_adv_nat_full, q)) if len(p_adv_nat_full) else 0.5
    if target_accept is not None and len(p_adv_nat_full):
        ta = float(np.clip(target_accept, 0.5, 0.9999))
        tau = float(np.quantile(p_adv_nat_full, ta))
    return tau

# -------------------- 3-class model helpers --------------------
def _load_base3_globals(root: Path):
    gdir = root / "_global"
    scaler3c: StandardScaler = joblib.load(gdir / "scaler.joblib")
    info = json.loads((gdir / "feature_columns.json").read_text(encoding="utf-8"))
    feat3c = info["feature_columns"] if isinstance(info, dict) else info
    return scaler3c, list(feat3c)

def _align_X_for_3c(X_raw_base: np.ndarray, feat_base: list[str], feat3c: list[str]) -> np.ndarray:
    set_b, set_c = set(feat_base), set(feat3c)
    if set_b != set_c:
        missing_b = sorted(set_c - set_b)
        missing_c = sorted(set_b - set_c)
        raise ValueError(f"Feature mismatch base vs 3-class. Missing in base: {missing_b[:6]}..., missing in 3c: {missing_c[:6]}...")
    pos = [feat_base.index(c) for c in feat3c]  # reorder to 3c order
    return X_raw_base[:, pos]

class MLPNetMulti(nn.Module):
    def __init__(self, in_dim: int, out_classes=3, hidden=(256,128,64), p_drop=0.2):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(p_drop)]; prev=h
        self.backbone = nn.Sequential(*layers); self.head = nn.Linear(prev, out_classes)
    def forward(self, x): return self.head(self.backbone(x))

class FTTransformerMulti(nn.Module):
    def __init__(self, n_features: int, out_classes=3, d_model=64, n_heads=8, n_layers=3, p_drop=0.1):
        super().__init__()
        self.value_emb = nn.Linear(1, d_model, bias=False)
        self.feat_emb  = nn.Embedding(n_features, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                                         dropout=p_drop, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model, out_classes))
    def forward(self, x):
        B,F = x.shape
        vt = self.value_emb(x.unsqueeze(-1)).reshape(B,F,-1)
        idx = torch.arange(F, device=x.device, dtype=torch.long)
        ft = self.feat_emb(idx)[None,:,:].expand(B,-1,-1)
        z = self.encoder(vt+ft); z = self.norm(z).mean(dim=1)
        return self.head(z)

def _torch_softmax_probs(model, Xz, device, batch_size, desc, use_progress, use_amp):
    """
    Robust inference with:
      - adaptive CUDA OOM backoff (halve batch until 1)
      - CPU fallback if still OOM at batch=1
    """
    N = Xz.shape[0]; out=[]; i=0
    bs = max(1, int(batch_size))
    local_device = device
    model.eval()
    while True:
        try:
            pbar = tqdm(total=math.ceil(N/bs), desc=desc, disable=not use_progress)
            with torch.no_grad():
                while i < N:
                    j = min(N, i+bs)
                    xb = torch.from_numpy(Xz[i:j]).to(local_device).float()
                    with torch.cuda.amp.autocast(enabled=(use_amp and local_device.type=="cuda")):
                        logits = model(xb).float()
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                    del xb, logits
                    out.append(probs); i=j; pbar.update(1)
            pbar.close()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and local_device.type == "cuda":
                cleanup_cuda()
                if bs > 1:
                    bs = max(1, bs // 2)
                    tqdm.write(f"[WARN] OOM in {desc}; retry with batch_size={bs}")
                    continue
                else:
                    tqdm.write(f"[WARN] OOM in {desc} at batch=1 → falling back to CPU.")
                    model.to(torch.device("cpu"))
                    local_device = torch.device("cpu")
                    bs = max(256, bs)  # CPU can often go a bit larger
                    continue
            raise
    return np.concatenate(out, axis=0)

def _reorder_probs_to_labels(p: np.ndarray, estimator, labels=(0,1,2)):
    try:
        classes = getattr(estimator, "classes_", None)
        if classes is None: return p
        classes = list(map(int, list(classes)))
        want = list(labels)
        colmap = {c:i for i,c in enumerate(classes)}
        idx = [colmap.get(c, None) for c in want]
        out = np.zeros((p.shape[0], len(want)), dtype=np.float32)
        for j, src in enumerate(idx):
            if src is not None:
                out[:, j] = p[:, src]
            else:
                out[:, j] = 0.0
        return out
    except Exception:
        return p

def _chunked_probs_multi(est, X, chunk, desc, use_progress):
    N = X.shape[0]; bs = max(4096, int(chunk)); i=0; outs=[]
    pbar = tqdm(total=math.ceil(N/bs), desc=desc, disable=not use_progress)
    while i < N:
        j = min(N, i+bs); xb = X[i:j]
        if hasattr(est, "predict_proba"):
            p = np.asarray(est.predict_proba(xb))
            if isinstance(p, list): p = p[0]
        else:
            z = np.asarray(est.decision_function(xb))
            e = np.exp(z - z.max(axis=1, keepdims=True))
            p = e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
        p = p.astype(np.float32, copy=False)
        p = _reorder_probs_to_labels(p, est, labels=LABELS3)
        outs.append(p); i=j; pbar.update(1)
    pbar.close(); return np.concatenate(outs, axis=0)

# -------------------- Utility --------------------
def pick_stream_indices(N: int, stream_size, nat_frac: float, rng: np.random.RandomState):
    if stream_size is None:
        adv_idx = np.arange(N)
        nat_count = int(np.floor(N * nat_frac))
        nat_idx = rng.choice(N, size=nat_count, replace=False)
    else:
        total = int(max(1, stream_size))
        nat_count = int(np.clip(int(total * nat_frac), 1, total-1))
        adv_count = total - nat_count
        adv_idx = rng.choice(N, size=adv_count, replace=False)
        nat_idx = rng.choice(N, size=nat_count, replace=False)
    return adv_idx, nat_idx

def adv_cache_path(root: Path, base_name: str, which: str, adv_idx: np.ndarray, steps: int, eps: float, alpha: float):
    h = hashlib.md5(adv_idx.tobytes()).hexdigest()[:12]
    fname = f"adv_{base_name}_{which}_N{adv_idx.size}_steps{steps}_eps{eps:.3f}_alpha{alpha:.3f}_{h}.npz"
    return root / "_cache" / fname

# -------------------- 3c pairing & loader --------------------
# STRICT one-to-one pairing: every base uses its 3-class counterpart with the SAME folder name
BASE_TO_3C = {
    "DL-MLP": "DL-MLP",
    "DL-FTTransformer": "DL-FTTransformer",
    "Logistic_Regression": "Logistic_Regression",
    "Calibrated_LinearSVC": "Calibrated_LinearSVC",  # <-- fixed to match your folder name
    "Gaussian_Naive_Bayes": "Gaussian_Naive_Bayes",
    "Decision_Tree": "Decision_Tree",
    "Random_Forest": "Random_Forest",
    "AdaBoost": "AdaBoost",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
}

def _find_model_file(dirpath: Path):
    pt = dirpath / "model.pt"
    if pt.exists(): return "torch", pt
    for nm in ["model.joblib", "model.pkl"]:
        p = dirpath / nm
        if p.exists(): return "sk", p
    return None, None

def load_threeclass_for_name(base3_root: Path, name3: str, feat3c_len: int, device):
    kind, path = _find_model_file(base3_root / name3)
    assert kind is not None, f"3-class model not found for '{name3}' under {base3_root}"
    if kind == "torch":
        if name3 == "DL-MLP":
            model3 = MLPNetMulti(in_dim=feat3c_len, out_classes=3).to(device)
        elif name3 == "DL-FTTransformer":
            model3 = FTTransformerMulti(n_features=feat3c_len, out_classes=3, d_model=64, n_heads=8, n_layers=3, p_drop=0.1).to(device)
        else:
            raise RuntimeError(f"Unexpected torch 3-class model folder '{name3}'")
        state = torch.load(path, map_location=device)
        model3.load_state_dict(state); model3.eval()
        return ("torch", name3, model3)
    else:
        return ("sk", name3, joblib.load(path))

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="3-class evaluation only (binary base used only to craft ADV)")
    p.add_argument("--model", default="all",
                   choices=[
                       "all",
                       "DL-MLP","DL-FTTransformer",
                       "Logistic_Regression","Calibrated_LinearSVC","Gaussian_Naive_Bayes",
                       "Decision_Tree","Random_Forest","AdaBoost","XGBoost","LightGBM"
                   ])
    p.add_argument("--nat-frac", type=float, default=None, help="NAT fraction in mixed stream (default 0.5)")
    p.add_argument("--stream-size", type=int, default=None, help="Total mixed-stream size; None=use all ADV (N)")
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--pgd-steps", type=int, default=None)
    p.add_argument("--fgsm-steps", type=int, default=None)
    p.add_argument("--eval-bs", type=int, default=None)
    p.add_argument("--lin-bs", type=int, default=None)
    p.add_argument("--bb-bs", type=int, default=None)
    p.add_argument("--torch-eval-bs", type=int, default=None)
    p.add_argument("--torch-attack-bs", type=int, default=None)
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--amp", action="store_true", help="Use AMP (float16/bfloat16) for torch eval/attacks")
    p.add_argument("--cache-adv", action="store_true", help="Cache crafted ADV subsets on disk")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--fast-predict", action="store_true", help="LightGBM booster fast predict with pred_early_stop")
    p.add_argument("--seed", type=int, default=None)

    # 3-class τ switches
    p.add_argument("--tau-q-3c", type=float, default=None, help="Quantile q for NAT p_adv (default 0.995)")
    p.add_argument("--tau-abs-3c", type=float, default=None, help="Absolute τ (overrides quantile)")
    p.add_argument("--tau-target-accept", type=float, default=None, help="Pick τ so ≥ this fraction of NAT pass (uses cached NAT dist)")

    # 3-class model override (optional)
    p.add_argument("--classifier3c", type=str, default=None, help="Name under models_base3 to use for ALL bases (otherwise per-base pairing)")
    return p.parse_args()

# -------------------- Main --------------------
def main():
    args = parse_args()
    if args.nat_frac is not None:  CFG["NAT_FRAC"] = float(args.nat_frac)
    if args.stream_size is not None: CFG["STREAM_SIZE"] = int(args.stream_size)
    if args.eps is not None:       CFG["eps_z"] = float(args.eps)
    if args.alpha is not None:     CFG["alpha_z"] = float(args.alpha)
    if args.pgd_steps is not None: CFG["steps_pgd"] = int(args.pgd_steps)
    if args.fgsm_steps is not None: CFG["steps_fgsm"] = int(args.fgsm_steps)
    if args.eval_bs is not None:   CFG["eval_bs"] = int(args.eval_bs)
    if args.lin_bs is not None:    CFG["lin_attack_bs"] = int(args.lin_bs)
    if args.bb_bs is not None:     CFG["bb_attack_bs"] = int(args.bb_bs)
    if args.torch_eval_bs is not None:   CFG["torch_eval_bs"] = int(args.torch_eval_bs)
    if args.torch_attack_bs is not None: CFG["torch_attack_bs"] = int(args.torch_attack_bs)
    if args.no_plots:              CFG["use_plots"] = False
    if args.no_progress:           CFG["use_progress"] = False
    if args.amp:                   CFG["use_amp"] = True
    if args.cache_adv:             CFG["cache_adv"] = True
    if args.cpu:                   CFG["use_gpu"] = False
    if args.fast_predict:          CFG["FAST_PREDICT"] = True
    if args.seed is not None:      CFG["seed"] = int(args.seed)
    if args.tau_q_3c is not None:  CFG["tau_quantile_3c"] = float(args.tau_q_3c)
    if args.tau_abs_3c is not None: CFG["tau_abs_3c"] = float(args.tau_abs_3c)
    if args.tau_target_accept is not None: CFG["tau_target_accept_nat"] = float(args.tau_target_accept)
    if args.classifier3c is not None: CFG["classifier3c"] = args.classifier3c

    rng = np.random.RandomState(CFG["seed"])
    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load base globals & dataset (binary labels used only for NAT 0/1 label)
    models_root = Path(CFG["models_dir"])
    results_root = ensure_dir(Path(CFG["results_dir"]))
    defense_root = Path(CFG["defense_dir"])

    scaler_base, feat_base = load_globals(models_root)
    X_raw, y_base = load_dataset(CFG["dataset_csv"], feat_base, CFG["label_col"])
    Xz = scaler_base.transform(X_raw).astype(np.float32, copy=False)
    N = Xz.shape[0]; z_min = Xz.min(axis=0); z_max = Xz.max(axis=0)
    print(f"[INFO] Loaded dataset: N={N:,}, D={Xz.shape[1]}")

    # Load defense & calibrator
    def_path = defense_root / "model.joblib"
    assert def_path.exists(), "Missing Defense-LGBM at models/Defense-LGBM/model.joblib"
    defense = joblib.load(def_path)
    calibrator = None
    cal_path = defense_root / "calibrator.joblib"
    if cal_path.exists():
        try: calibrator = joblib.load(cal_path); print("[INFO] Using isotonic calibrator for defense probabilities.")
        except Exception: calibrator = None

    # Precompute NAT p_adv over full dataset (cached) for tau_3c choice
    common_out = ensure_dir(results_root / "_common")
    finger = f"{_file_md5(def_path)}_{_file_md5(cal_path)}_{_file_md5(models_root / '_global' / 'scaler.joblib')}_{N}_{len(feat_base)}"
    nat_cache = common_out / f"p_adv_nat_full_{finger}.npy"
    if nat_cache.exists():
        p_adv_nat_full = np.load(nat_cache).astype(np.float32, copy=False)
        print(f"[CACHE] Loaded NAT p_adv from {nat_cache.name}")
    else:
        p_adv_nat_full = defense_p_adv(
            defense, X_raw, batch=CFG["eval_bs"], calibrator=calibrator,
            desc="defense NAT p_adv", use_progress=CFG["use_progress"],
            fast_predict=CFG["FAST_PREDICT"], early_stop_margin=CFG["EARLY_STOP_MARGIN"]
        )
        np.save(nat_cache, p_adv_nat_full); print(f"[CACHE] Saved NAT p_adv to {nat_cache.name}")

    tau3c = choose_tau_from_nat_full(
        p_adv_nat_full,
        q=CFG["tau_quantile_3c"],
        tau_abs=CFG["tau_abs_3c"],
        target_accept=CFG["tau_target_accept_nat"]
    )
    print(f"[INFO] τ_3c = {tau3c:.6f} (q={CFG['tau_quantile_3c']}, abs={CFG['tau_abs_3c']}, target_accept={CFG['tau_target_accept_nat']})")

    # Load 3-class globals
    base3_root = Path(CFG["models_base3_dir"])
    scaler3c, feat3c = _load_base3_globals(base3_root)
    X_raw_reordered_for_3c = _align_X_for_3c(X_raw, feat_base, feat3c)

    # Global surrogate (pretrained) for non-linear sklearn base attacks
    print("[INFO] Training global surrogate (pretrain) for black-box sklearn attacks...")
    from torch.utils.data import DataLoader, TensorDataset
    class SurrogateMLP(nn.Module):
        def __init__(self, D, H=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(D, H), nn.ReLU(),
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(D, 1) if False else nn.Linear(H, 1)
            )
        def forward(self, x): return self.net(x).squeeze(-1)
    global_surr = SurrogateMLP(D=Xz.shape[1], H=CFG["surrogate_hidden"]).to(device)
    opt = torch.optim.AdamW(global_surr.parameters(), lr=3e-4, weight_decay=1e-5)
    ds = TensorDataset(torch.from_numpy(Xz).float(), torch.from_numpy(y_base.astype(np.float32)))
    dl = DataLoader(ds, batch_size=CFG["surrogate_batch"], shuffle=True)
    global_surr.train(); best=None; best_loss=float("inf")
    for ep in tqdm(range(1, CFG["surrogate_epochs_global"]+1), desc="surrogate/pretrain", disable=not CFG["use_progress"]):
        run=0.0; n=0
        for xb, yb in dl:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(global_surr(xb), yb)
            loss.backward(); opt.step()
            run += loss.item()*xb.size(0); n+=xb.size(0)
        avg=run/max(1,n); print(f"[GlobalSurrogate] epoch {ep}/{CFG['surrogate_epochs_global']} loss={avg:.6f}")
        if avg < best_loss - 1e-5:
            best_loss=avg; best={k:v.detach().cpu() for k,v in global_surr.state_dict().items()}
    if best is not None: global_surr.load_state_dict({k:v.to(device) for k,v in best.items()})
    global_surr.eval(); cleanup_cuda()

    # Candidate base models (binary) used ONLY to craft attacks
    candidates = [
        "DL-MLP","DL-FTTransformer",
        "Logistic_Regression","Calibrated_LinearSVC","Gaussian_Naive_Bayes",
        "Decision_Tree","Random_Forest","AdaBoost","XGBoost","LightGBM",
    ]
    if args.model != "all":
        candidates = [args.model]

    # Load available base (binary) models
    bases = []
    for name in candidates:
        mdir = models_root / name
        if not mdir.exists():
            print(f"[WARN] Skipping {name}: {mdir} not found.")
            continue
        if name in {"DL-MLP","DL-FTTransformer"}:
            # torch binary
            if name == "DL-MLP":
                mdl = MLPNetBin(in_dim=Xz.shape[1]).to(device)
            else:
                mdl = FTTransformerBin(n_features=Xz.shape[1], d_model=64, n_heads=8, n_layers=3, p_drop=0.1).to(device)
            state = torch.load(mdir / "model.pt", map_location=device)
            mdl.load_state_dict(state); mdl.eval()
            bases.append((name, mdl))
        else:
            mpath = mdir / "model.joblib"
            if mpath.exists():
                bases.append((name, joblib.load(mpath)))
            else:
                print(f"[WARN] Skipping {name}: {mpath} not found.")

    assert bases, "No base (binary) models found for attack crafting."

    # Pre-pick one stream (shared across bases)
    adv_idx_fixed, nat_idx_fixed = pick_stream_indices(N, CFG["STREAM_SIZE"], CFG["NAT_FRAC"], rng)

    # ---- craft helpers
    def distill_from_global(global_surr, teacher, Xz, device, epochs=6, batch=4096, sample_limit=150_000):
        Nloc = Xz.shape[0]; idx = np.arange(Nloc)
        if Nloc > sample_limit:
            idx = np.random.RandomState(42).choice(Nloc, size=sample_limit, replace=False)
        if hasattr(teacher, "predict_proba"):
            soft = teacher.predict_proba(Xz[idx])[:,1].astype(np.float32)
        elif hasattr(teacher, "decision_function"):
            df = teacher.decision_function(Xz[idx]).astype(np.float32); soft = 1./(1.+np.exp(-df))
        else:
            soft = teacher.predict(Xz[idx]).astype(np.float32)
        class SurrogateMLP(nn.Module):
            def __init__(self, D, H=CFG["surrogate_hidden"]):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(D, H), nn.ReLU(),
                    nn.Linear(H, H), nn.ReLU(),
                    nn.Linear(H, 1)
                )
            def forward(self, x): return self.net(x).squeeze(-1)
        mdl = SurrogateMLP(D=Xz.shape[1])
        mdl.load_state_dict(global_surr.state_dict()); mdl = mdl.to(device)
        opt = torch.optim.Adam(mdl.parameters(), lr=3e-4, weight_decay=1e-5)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(Xz[idx]).float(), torch.from_numpy(soft).float())
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
        mdl.train(); best=None; best_loss=float("inf"); patience=2; bad=0
        for ep in tqdm(range(1, epochs+1), desc="distill/finetune", disable=not CFG["use_progress"]):
            run=0.0; n=0
            for xb, sb in dl:
                xb=xb.to(device); sb=sb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(mdl(xb), sb)
                loss.backward(); opt.step()
                run+=loss.item()*xb.size(0); n+=xb.size(0)
            avg=run/max(1,n)
            if avg < best_loss - 1e-5:
                best_loss=avg; best={k:v.detach().cpu() for k,v in mdl.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=patience: break
        if best is not None: mdl.load_state_dict({k:v.to(device) for k,v in best.items()})
        mdl.eval(); return mdl

    def craft_adv_subset(base_name: str, base_est, which: str, Xz_sub: np.ndarray, y_sub: np.ndarray) -> np.ndarray:
        steps = CFG["steps_fgsm"] if which == "FGSM" else CFG["steps_pgd"]
        if base_name in {"DL-MLP","DL-FTTransformer"}:
            return attack_tabular_torch_batched(
                base_est, Xz_sub, y_sub,
                eps=CFG["eps_z"], alpha=CFG["alpha_z"], steps=steps,
                device=device, z_min=z_min, z_max=z_max, batch_size=CFG["torch_attack_bs"],
                desc=f"{base_name}/{which} torch attack", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"]
            )
        if base_name in {"Logistic_Regression","Calibrated_LinearSVC"}:
            loss_kind = "hinge" if base_name == "Calibrated_LinearSVC" else "logistic"
            return attack_linear_fgsm_pgd(
                base_est, Xz_sub, y_sub,
                steps=steps, eps=CFG["eps_z"], alpha=CFG["alpha_z"],
                loss_kind=loss_kind, batch=CFG["lin_attack_bs"],
                desc=f"{base_name}/{which} lin attack", use_progress=CFG["use_progress"]
            )
        surr = distill_from_global(
            global_surr, base_est, Xz,
            device=device, epochs=CFG["surrogate_epochs_finetune"],
            batch=CFG["surrogate_batch"], sample_limit=CFG["surrogate_sample_limit"]
        )
        return attack_tabular_torch_batched(
            surr, Xz_sub, y_sub,
            eps=CFG["eps_z"], alpha=CFG["alpha_z"], steps=steps,
            device=device, z_min=z_min, z_max=z_max, batch_size=CFG["bb_attack_bs"],
            desc=f"{base_name}/{which} bb attack", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"]
        )

    # ---- per-class metrics packer
    def per_class_metrics(y_true, y_pred, cls_labels):
        out = {}
        prc = precision_score(y_true, y_pred, labels=cls_labels, average=None, zero_division=0)
        rec = recall_score(y_true, y_pred, labels=cls_labels, average=None, zero_division=0)
        f1  = f1_score(y_true, y_pred, labels=cls_labels, average=None, zero_division=0)
        for i, c in enumerate(cls_labels):
            out[str(c)] = {
                "precision": float(prc[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int((y_true == c).sum()),
            }
        return out

    # 3c loader cache
    threeclass_cache: dict[str, tuple[str,str,object]] = {}

    def get_threeclass_for_base(base_name: str):
        # override for all bases?
        if CFG["classifier3c"] and CFG["classifier3c"] != "auto":
            name3 = CFG["classifier3c"]
        else:
            name3 = BASE_TO_3C.get(base_name, None)
            assert name3 is not None, f"No mapping from base '{base_name}' to 3-class model."
        if name3 in threeclass_cache:
            return threeclass_cache[name3]
        kind, name_loaded, est = load_threeclass_for_name(base3_root, name3, len(feat3c), device)
        threeclass_cache[name3] = (kind, name_loaded, est)
        print(f"[INFO] 3-class model for base '{base_name}': {name_loaded} ({kind})")
        return threeclass_cache[name3]

    # ---- main eval per base × attack
    def eval_3c_for(base_name: str, base_est, which: str, adv_idx: np.ndarray, nat_idx: np.ndarray):
        out_dir = ensure_dir(results_root / base_name / f"mixed_{which}")
        cache_dir = ensure_dir(results_root / base_name / "_cache")

        # Load the paired 3-class model for THIS base
        kind3, name3, est3 = get_threeclass_for_base(base_name)

        # 1) Craft ADV subset (optionally cached)
        steps_used = int(CFG["steps_fgsm"] if which == "FGSM" else CFG["steps_pgd"])
        cache_path = adv_cache_path(results_root / base_name, base_name, which, adv_idx, steps_used, CFG["eps_z"], CFG["alpha_z"])
        use_cached = False
        if CFG["cache_adv"] and cache_path.exists():
            try:
                data = np.load(cache_path)
                if np.array_equal(data["adv_idx"], adv_idx):
                    Xz_adv = data["Xz_adv"].astype(np.float32, copy=False)
                    use_cached = True; print(f"[CACHE] Loaded ADV subset from {cache_path.name}")
                del data
            except Exception:
                use_cached = False
        if not use_cached:
            Xz_adv = craft_adv_subset(base_name, base_est, which, Xz[adv_idx], y_base[adv_idx])
            if CFG["cache_adv"]:
                ensure_dir(cache_dir)
                np.savez_compressed(cache_path, adv_idx=adv_idx, Xz_adv=Xz_adv)
                print(f"[CACHE] Saved ADV subset to {cache_path.name}")

        # 2) Defense scores to compute accept masks for AFTER-DEF
        p_adv_nat = p_adv_nat_full[nat_idx]
        p_adv_adv = defense_p_adv_from_z(
            defense, scaler_base, Xz_adv, batch=CFG["eval_bs"], calibrator=calibrator,
            desc=f"defense p_adv ADV/{base_name}/{which}", use_progress=CFG["use_progress"],
            fast_predict=CFG["FAST_PREDICT"], early_stop_margin=CFG["EARLY_STOP_MARGIN"]
        )
        acc_nat_mask_3c = (p_adv_nat < tau3c)
        acc_adv_mask_3c = (p_adv_adv < tau3c)
        n_acc_nat = int(acc_nat_mask_3c.sum()); n_acc_adv = int(acc_adv_mask_3c.sum())
        n_blk_nat = int(p_adv_nat.size - n_acc_nat); n_blk_adv = int(p_adv_adv.size - n_acc_adv)

        # 3) Build 3-class GT: NAT keep {0,1}; ADV -> 2
        y3_nat = y_base[nat_idx].copy()
        y3_adv = np.full_like(y_base[adv_idx], 2)

        # 4) Prepare 3-class feature matrices (reorder + scale)
        Xz3_nat = scaler3c.transform(X_raw_reordered_for_3c[nat_idx])
        X_raw_adv = scaler_base.inverse_transform(Xz_adv).astype(np.float32, copy=False)
        X_raw_adv = _align_X_for_3c(X_raw_adv, feat_base, feat3c)
        Xz3_adv = scaler3c.transform(X_raw_adv)

        # 5) Get 3-class probabilities/predictions
        if kind3 == "torch":
            P3_nat = _torch_softmax_probs(est3, Xz3_nat, device, CFG["torch_eval_bs"], f"3c/{name3} NAT (no-def)", CFG["use_progress"], CFG["use_amp"])
            P3_adv = _torch_softmax_probs(est3, Xz3_adv, device, CFG["torch_eval_bs"], f"3c/{name3} ADV (no-def)", CFG["use_progress"], CFG["use_amp"])
        else:
            P3_nat = _chunked_probs_multi(est3, Xz3_nat, CFG["eval_bs"], f"3c/{name3} NAT (no-def)", CFG["use_progress"])
            P3_adv = _chunked_probs_multi(est3, Xz3_adv, CFG["eval_bs"], f"3c/{name3} ADV (no-def)", CFG["use_progress"])

        # ---------- (A) NO DEFENSE ----------
        y3_mixed = np.concatenate([y3_nat, y3_adv])
        P3_mixed = np.concatenate([P3_nat, P3_adv])
        yhat3_mixed = P3_mixed.argmax(axis=1)

        acc_nd = float(accuracy_score(y3_mixed, yhat3_mixed))
        f1_nd  = float(f1_score(y3_mixed, yhat3_mixed, average="macro"))
        auc_nd = None
        try:
            auc_nd = float(roc_auc_score(y3_mixed, P3_mixed, multi_class="ovr"))
        except Exception:
            pass
        per_class_nd = per_class_metrics(y3_mixed, yhat3_mixed, LABELS3)
        cm_nd = cm_as_dict(y3_mixed, yhat3_mixed, LABELS3)

        if CFG["use_plots"]:
            plot_confusion_3c(y3_mixed, yhat3_mixed, out_path=out_dir / "confusion_no_def.png", title=f"3-class Confusion (no defense) [{name3}]")
            if auc_nd is not None:
                plot_roc_multiclass(y3_mixed, P3_mixed, out_path=out_dir / "roc_ovr_no_def.png", title=f"ROC OvR (no defense) [{name3}]")
        try:
            save_text(classification_report(y3_mixed, yhat3_mixed, digits=4), out_dir / "classification_report_no_def.txt")
        except Exception:
            pass

        # NAT-only (0/1) NO DEFENSE
        nat_mask_all = (y3_mixed != 2)
        y_nat_only = y3_mixed[nat_mask_all]              # contains only {0,1}
        yhat_nat_only = yhat3_mixed[nat_mask_all]        # may contain {0,1,2}
        acc_nat_nd = float(accuracy_score(y_nat_only, yhat_nat_only))
        f1_nat_nd = float(f1_score(y_nat_only, yhat_nat_only, labels=LABELS_NAT, average="macro", zero_division=0))
        per_class_nat_nd = per_class_metrics(y_nat_only, yhat_nat_only, LABELS_NAT)
        cm_nat_nd = cm_as_dict(y_nat_only, yhat_nat_only, NAT_CM_LABELS)  # include pred==2 column
        if CFG["use_plots"]:
            plot_confusion_nat(y_nat_only, yhat_nat_only, out_path=out_dir / "confusion_nat_no_def.png", title=f"NAT-only Confusion (no defense) [{name3}]")

        # ---------- (B) AFTER DEFENSE (lenient tau_3c) ----------
        Pacc_nat = P3_nat[acc_nat_mask_3c]
        Yacc_nat = y3_nat[acc_nat_mask_3c]
        Pacc_adv = P3_adv[acc_adv_mask_3c]
        Yacc_adv = y3_adv[acc_adv_mask_3c]

        if (Yacc_nat.size + Yacc_adv.size) > 0:
            P3_acc = np.concatenate([Pacc_nat, Pacc_adv], axis=0)
            y3_acc = np.concatenate([Yacc_nat, Yacc_adv], axis=0)
            yhat3_acc = P3_acc.argmax(axis=1)

            acc_ad = float(accuracy_score(y3_acc, yhat3_acc))
            f1_ad  = float(f1_score(y3_acc, yhat3_acc, average="macro"))
            auc_ad = None
            try:
                auc_ad = float(roc_auc_score(y3_acc, P3_acc, multi_class="ovr"))
            except Exception:
                pass
            per_class_ad = per_class_metrics(y3_acc, yhat3_acc, LABELS3)
            cm_ad = cm_as_dict(y3_acc, yhat3_acc, LABELS3)

            if CFG["use_plots"]:
                plot_confusion_3c(y3_acc, yhat3_acc, out_path=out_dir / "confusion_after_def.png", title=f"3-class Confusion (after defense τ3c) [{name3}]")
                if auc_ad is not None:
                    plot_roc_multiclass(y3_acc, P3_acc, out_path=out_dir / "roc_ovr_after_def.png", title=f"ROC OvR (after defense τ3c) [{name3}]")
            try:
                save_text(classification_report(y3_acc, yhat3_acc, digits=4), out_dir / "classification_report_after_def.txt")
            except Exception:
                pass
        else:
            acc_ad = float("nan"); f1_ad = float("nan"); auc_ad = None; per_class_ad = {str(k): {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "support": 0} for k in LABELS3}
            cm_ad = {"labels": LABELS3, "matrix": [[0,0,0],[0,0,0],[0,0,0]], "total": 0, "accuracy_from_confusion": float("nan")}

        # NAT-only (0/1) AFTER DEFENSE (accepted NAT only)
        if Yacc_nat.size > 0:
            yhat_nat_acc = Pacc_nat.argmax(axis=1)       # may include 2
            acc_nat_ad = float(accuracy_score(Yacc_nat, yhat_nat_acc))
            f1_nat_ad = float(f1_score(Yacc_nat, yhat_nat_acc, labels=LABELS_NAT, average="macro", zero_division=0))
            per_class_nat_ad = per_class_metrics(Yacc_nat, yhat_nat_acc, LABELS_NAT)
            cm_nat_ad = cm_as_dict(Yacc_nat, yhat_nat_acc, NAT_CM_LABELS)  # include pred==2 column
            if CFG["use_plots"]:
                plot_confusion_nat(Yacc_nat, yhat_nat_acc, out_path=out_dir / "confusion_nat_after_def.png", title=f"NAT-only Confusion (after defense τ3c) [{name3}]")
        else:
            acc_nat_ad = float("nan"); f1_nat_ad = float("nan")
            per_class_nat_ad = {str(k): {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "support": 0} for k in LABELS_NAT}
            cm_nat_ad = {"labels": NAT_CM_LABELS, "matrix": [[0,0,0],[0,0,0],[0,0,0]], "total": 0, "accuracy_from_confusion": float("nan")}

        # ADV detection rates (class==2)
        det_rate_nd   = float((P3_adv.argmax(axis=1) == 2).mean()) if P3_adv.shape[0] else float("nan")
        det_rate_acc  = float((Pacc_adv.argmax(axis=1) == 2).mean()) if Pacc_adv.shape[0] else float("nan")

        # -------- Save metrics as TWO separate files with consistent names --------
        # BEFORE DEFENSE
        metrics_before = {
            "scenario": f"mixed_{which}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_base_model": base_name,
            "threeclass_model": name3,
            "lenient_tau_3c": float(tau3c),
            "stream_counts": {"nat": int(nat_idx.size), "adv": int(adv_idx.size), "total": int(nat_idx.size + adv_idx.size)},
            "three_class": {
                "accuracy": acc_nd,
                "f1_macro": f1_nd,
                "auc_ovr": auc_nd,
                "per_class": per_class_nd,
                "counts": {"total": int(y3_mixed.size), "nat": int(y3_nat.size), "adv": int(y3_adv.size)},
                "adv_pred2_rate": det_rate_nd,
                "confusion": cm_nd,
            },
            "nat_only_slice": {
                "accuracy": acc_nat_nd,
                "f1_macro": f1_nat_nd,
                "per_class": per_class_nat_nd,
                "counts": {"nat_total": int(y_nat_only.size)},
                "confusion": cm_nat_nd,
            },
            "attack": {
                "kind": which,
                "steps": steps_used,
                "eps_z": float(CFG["eps_z"]),
                "alpha_z": float(CFG["alpha_z"]),
            },
        }
        save_json(metrics_before, out_dir / "metrics_before_defense.json")

        # AFTER DEFENSE
        metrics_after = {
            "scenario": f"mixed_{which}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attack_base_model": base_name,
            "threeclass_model": name3,
            "lenient_tau_3c": float(tau3c),
            "stream_counts": {"nat": int(nat_idx.size), "adv": int(adv_idx.size), "total": int(nat_idx.size + adv_idx.size)},
            "gate": {
                "accept_rate_nat": float(acc_nat_mask_3c.mean()) if p_adv_nat.size else float("nan"),
                "accept_rate_adv": float(acc_adv_mask_3c.mean()) if p_adv_adv.size else float("nan"),
                "counts": {"nat_accepted": int(n_acc_nat), "nat_blocked": int(n_blk_nat), "adv_accepted": int(n_acc_adv), "adv_blocked": int(n_blk_adv)},
            },
            "three_class": {
                "accuracy": acc_ad,
                "f1_macro": f1_ad,
                "auc_ovr": auc_ad,
                "per_class": per_class_ad,
                "counts": {"total_accepted": int((Yacc_nat.size + Yacc_adv.size)), "nat_accepted": int(Yacc_nat.size), "adv_accepted": int(Yacc_adv.size)},
                "adv_pred2_rate_accepted": det_rate_acc,
                "confusion": cm_ad,
            },
            "nat_only_slice": {
                "accuracy": acc_nat_ad,
                "f1_macro": f1_nat_ad,
                "per_class": per_class_nat_ad,
                "counts": {"nat_accepted": int(Yacc_nat.size)},
                "confusion": cm_nat_ad,
            },
            "attack": {
                "kind": which,
                "steps": steps_used,
                "eps_z": float(CFG["eps_z"]),
                "alpha_z": float(CFG["alpha_z"]),
            },
        }
        save_json(metrics_after, out_dir / "metrics_after_defense.json")

        # free
        del Xz_adv, p_adv_nat, p_adv_adv, P3_nat, P3_adv, P3_mixed, y3_mixed
        cleanup_cuda()

    # ---------- Run per base ----------
    for base_name, base_est in bases:
        print(f"\n==== Using base (for ATTACK crafting only): {base_name} ====")
        eval_3c_for(base_name, base_est, "FGSM", adv_idx_fixed, nat_idx_fixed)
        eval_3c_for(base_name, base_est, "PGD",  adv_idx_fixed, nat_idx_fixed)

    print("\n[DONE] 3-class evaluation complete (no-defense & after-defense).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        print(traceback.format_exc())
        raise

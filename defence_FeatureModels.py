#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-stream evaluation for Defense-LGBM (gate trained on is_adv) with mixed NAT + adversarial traffic.
Optimized:
  - Streamed inverse_transform for ADV → raw (no giant float64 buffer).
  - LightGBM Booster fast-path + multi-threading (and optional pred_early_stop).
  - Cache NAT p_adv across runs keyed on model/scaler.
  - Fixed conditional accept-rate computation.
  - Reuse one (adv_idx, nat_idx) pair across all bases for fair + cheaper sampling.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, gc, warnings, argparse, hashlib, math
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Give LightGBM/OpenMP all threads unless user overrides externally
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay,
    accuracy_score, classification_report,
    ConfusionMatrixDisplay, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Config --------------------
CFG = {
    "models_dir": "models",
    "defense_dir": "models/Defense-LGBM",
    "results_dir": "results_defense_feat_stream",
    "dataset_csv": "features_extracted.csv",
    "label_col": "label",

    # z-space attack budgets
    "eps_z": 0.40,
    "alpha_z": 0.10,
    "steps_fgsm": 1,
    "steps_pgd": 5,

    # surrogate (for non-linear sklearn bases)
    "surrogate_hidden": 256,
    "surrogate_epochs_global": 5,
    "surrogate_epochs_finetune": 6,
    "surrogate_batch": 4096,
    "surrogate_sample_limit": 150_000,

    # batch sizes with back-off
    "eval_bs": 100_000,     # used for sklearn scoring & defense
    "lin_attack_bs": 32_768,
    "bb_attack_bs": 8_192,
    "torch_eval_bs": 8192,  # pytorch eval batch size
    "torch_attack_bs": 4096,

    # device
    "use_gpu": True,

    # NAT threshold selection
    "TAU_QUANTILE": 0.90,

    # Mixed-stream composition
    "NAT_FRAC": 0.5,
    "STREAM_SIZE": 1_000_000,

    # RNG
    "seed": 42,

    # Verbose
    "use_progress": True,
    "use_plots": True,
    "use_amp": False,
    "cache_adv": False,

    # Fast LightGBM inference
    "FAST_PREDICT": False,  # enable with --fast-predict
    "EARLY_STOP_MARGIN": 2.0,
}

LABELS = [0, 1]

# -------------------- IO helpers --------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(txt, p: Path):
    p.write_text(txt, encoding="utf-8")

def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    if len(y_true) == 0: return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, ax=ax)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def plot_roc(y_true, y_score, out_path: Path, title: str):
    if len(y_true) == 0: return
    try:
        y_true = np.asarray(y_true); y_score = np.asarray(y_score)
        if np.unique(y_true).size < 2: return
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title(title)
        fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    except Exception:
        pass

def plot_hist(x, out_path: Path, title: str, bins=50):
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.asarray(x), bins=bins, density=True)
        ax.set_title(title); ax.set_xlabel("value"); ax.set_ylabel("density")
        fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)
    except Exception:
        pass

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------- Globals & dataset --------------------
def load_globals(models_root: Path):
    gdir = models_root / "_global"
    scaler: StandardScaler = joblib.load(gdir / "scaler.joblib")
    feat_info = json.loads((gdir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_cols = feat_info["feature_columns"] if isinstance(feat_info, dict) and "feature_columns" in feat_info else feat_info
    return scaler, feature_cols

def load_dataset(csv_path: str, feature_cols: list[str], label_col: str):
    df = pd.read_csv(csv_path)
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column '{c}' missing in {csv_path}")
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_base = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    return X_raw, y_base

# -------------------- Sklearn helpers --------------------
def score_vector(estimator, X):
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if isinstance(p, list): p = p[0]
        if getattr(p, "ndim", 1) == 2 and p.shape[1] == 2: return p[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X); return np.asarray(s).ravel()
    pred = estimator.predict(X); return np.asarray(pred, dtype=float).ravel()

def chunked_scores(est, X: np.ndarray, chunk: int, desc: str = "scoring", use_progress: bool = True):
    N = X.shape[0]; out = np.empty((N,), dtype=np.float32)
    bs = max(4096, int(chunk)); s = 0
    total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while s < N:
        e = min(N, s + bs)
        try:
            out[s:e] = np.asarray(score_vector(est, X[s:e])).ravel().astype(np.float32, copy=False)
            s = e; pbar.update(1)
        except Exception as ex:
            if "MemoryError" in str(ex) or "CUDA out of memory" in str(ex) or bs > 4096:
                bs = max(4096, bs // 2); cleanup_cuda(); total = math.ceil((N - s) / bs)
                pbar.total = pbar.n + total; pbar.refresh(); continue
            pbar.close(); raise
    pbar.close()
    return out

def sigmoidify(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores)
    if (scores.min() < 0.0) or (scores.max() > 1.0):
        return 1.0 / (1.0 + np.exp(-scores))
    return scores

def unwrap_calibrated(estimator):
    if isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
            inner = estimator.calibrated_classifiers_[0]
            if hasattr(inner, "estimator") and inner.estimator is not None: return inner.estimator
            if hasattr(inner, "base_estimator") and inner.base_estimator is not None: return inner.base_estimator
    return estimator

# -------------------- Torch feature models --------------------
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
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(p_drop), nn.Linear(d_model, 1)
        )

    def forward(self, x):  # x: (B, F) float
        B, F = x.shape
        val_tok = self.value_emb(x.unsqueeze(-1)).reshape(B, F, -1)
        idx = torch.arange(F, device=x.device, dtype=torch.long)
        feat_tok = self.feat_emb(idx)[None, :, :].expand(B, -1, -1)
        z = self.encoder(val_tok + feat_tok)
        z = self.norm(z).mean(dim=1)
        return self.head(z).squeeze(-1)

@torch.no_grad()
def eval_torch_batched(model, Xz, device, batch_size, desc="torch/eval", use_progress=True, use_amp=False):
    """Evaluate a torch model in batches with optional CUDA AMP. No global autocast context on CPU."""
    N = Xz.shape[0]; probs_list = []; bs = max(1, int(batch_size)); i = 0
    model.eval()
    total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while i < N:
        j = min(N, i + bs)
        try:
            xb = torch.from_numpy(Xz[i:j]).to(device).float()
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(xb).float()
            else:
                logits = model(xb).float()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_list.append(probs); i = j; pbar.update(1)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and bs > 1 and torch.cuda.is_available():
                cleanup_cuda(); bs = max(1, bs // 2)
                total = math.ceil((N - i) / bs); pbar.total = pbar.n + total; pbar.refresh()
                print(f"[WARN] OOM during torch eval; retry with batch_size={bs}")
                continue
            pbar.close(); raise
    pbar.close()
    return np.concatenate(probs_list, axis=0)

def attack_tabular_torch_batched(model, Xz, y, eps, alpha, steps, device, z_min, z_max, batch_size,
                                 desc="torch/attack", use_progress=True, use_amp=False):
    N, F = Xz.shape
    adv_list = []; bs = max(1, int(batch_size))
    steps_eff = 1 if steps == 0 else steps
    alpha_eff = eps if steps == 0 else alpha
    zmin_t = torch.from_numpy(z_min).to(device)
    zmax_t = torch.from_numpy(z_max).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    ptr = 0
    total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while ptr < N:
        end = min(N, ptr + bs)
        try:
            X = torch.from_numpy(Xz[ptr:end]).to(device).float()
            Y = torch.from_numpy(y[ptr:end]).to(device).float()
            X0 = X.clone().detach()
            X_adv = X.clone().detach()
            for _ in range(steps_eff):
                X_adv.requires_grad_(True)
                if use_amp and device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits = model(X_adv).float()
                        loss = loss_fn(logits, Y)
                else:
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
            adv_list.append(X_adv.detach().cpu().numpy()); ptr = end; pbar.update(1)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and bs > 1 and torch.cuda.is_available():
                cleanup_cuda(); bs = max(1, bs // 2)
                total = math.ceil((N - ptr) / bs); pbar.total = pbar.n + total; pbar.refresh()
                print(f"[WARN] OOM during torch attack; retry with batch_size={bs}")
                continue
            pbar.close(); raise
    pbar.close()
    return np.concatenate(adv_list, axis=0)

# -------------------- LightGBM fast-path helpers --------------------
def _defense_set_threads(defense):
    try:
        if hasattr(defense, "set_params"):
            defense.set_params(n_jobs=os.cpu_count() or 1)
        booster = getattr(defense, "booster_", None)
        if booster is not None:
            booster.set_num_threads(os.cpu_count() or 1)
    except Exception:
        pass

def _booster_from_defense(defense):
    try:
        import lightgbm as lgb  # noqa
    except Exception:
        return None
    return getattr(defense, "booster_", None)

# -------------------- Defense (p_adv) --------------------
def defense_p_adv(def_model, X_raw, batch=100_000, calibrator=None, desc="defense/p_adv",
                  use_progress=True, fast_predict=False, early_stop_margin=2.0):
    """
    Scoring when inputs are already in raw feature space.
    Uses LightGBM booster fast-path if available.
    """
    N = X_raw.shape[0]; probs = np.empty(N, dtype=np.float32)
    bs = max(10_000, int(batch)); i = 0
    total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)

    booster = _booster_from_defense(def_model)
    if booster is not None:
        try: booster.set_num_threads(os.cpu_count() or 1)
        except Exception: pass

    while i < N:
        j = min(N, i + bs)
        xb = np.ascontiguousarray(X_raw[i:j], dtype=np.float32)
        try:
            if booster is not None:
                # class-1 prob fast path
                kw = dict(raw_score=False)
                if fast_predict:
                    kw.update(dict(pred_early_stop=True, pred_early_stop_margin=early_stop_margin))
                p1 = booster.predict(xb, **kw)
            else:
                p1 = score_vector(def_model, xb)
        except Exception as ex:
            # back off on batch size if memory-bound somewhere in the stack
            if "MemoryError" in str(ex) or bs > 10_000:
                bs = max(10_000, bs // 2); cleanup_cuda()
                total = math.ceil((N - i) / bs); pbar.total = pbar.n + total; pbar.refresh()
                continue
            pbar.close(); raise
        p1 = np.asarray(p1)
        if p1.ndim == 2 and p1.shape[1] == 2: p1 = p1[:, 1]
        if p1.min() < 0 or p1.max() > 1: p1 = 1.0 / (1.0 + np.exp(-p1))
        if calibrator is not None:
            try: p1 = calibrator.predict(p1)
            except Exception: pass
        probs[i:j] = p1.astype(np.float32, copy=False); i = j; pbar.update(1)
    pbar.close()
    return probs

def defense_p_adv_from_z(def_model, scaler: StandardScaler, Xz, batch=100_000, calibrator=None,
                         desc="defense/p_adv(z->raw)", use_progress=True,
                         fast_predict=False, early_stop_margin=2.0):
    """
    Scoring when inputs are in standardized z-space.
    Streams inverse_transform per chunk (RAM-constant) and uses LGBM fast-path.
    """
    N = Xz.shape[0]
    out = np.empty(N, dtype=np.float32)
    bs = max(10_000, int(batch))
    i = 0
    total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)

    booster = _booster_from_defense(def_model)
    if booster is not None:
        try: booster.set_num_threads(os.cpu_count() or 1)
        except Exception: pass

    while i < N:
        j = min(N, i + bs)
        xb_raw = scaler.inverse_transform(Xz[i:j]).astype(np.float32, copy=False)
        xb_raw = np.ascontiguousarray(xb_raw)
        if booster is not None:
            kw = dict(raw_score=False)
            if fast_predict:
                kw.update(dict(pred_early_stop=True, pred_early_stop_margin=early_stop_margin))
            p1 = booster.predict(xb_raw, **kw)
        else:
            p1 = score_vector(def_model, xb_raw)
        p1 = np.asarray(p1)
        if p1.ndim == 2 and p1.shape[1] == 2: p1 = p1[:, 1]
        if p1.min() < 0 or p1.max() > 1: p1 = 1.0 / (1.0 + np.exp(-p1))
        if calibrator is not None:
            try: p1 = calibrator.predict(p1)
            except Exception: pass
        out[i:j] = p1.astype(np.float32, copy=False)
        i = j; pbar.update(1)
    pbar.close()
    return out

def choose_tau_from_nat(p_adv_nat: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 0.50, 0.999))
    if len(p_adv_nat) == 0: return 0.5
    return float(np.quantile(p_adv_nat, q))

# -------------------- Linear attack helpers (sklearn) --------------------
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
    s = 0
    bs = int(batch)
    total = math.ceil(N / bs)
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

# -------------------- Metrics helpers --------------------
def safe_auc(y_true, scores) -> float:
    try:
        y_true = np.asarray(y_true); scores = np.asarray(scores)
        if np.unique(y_true).size < 2: return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")

def base_metrics(y_true, probs, out_dir: Path, tag: str, prefix: str, enable_plots=True):
    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, preds)) if len(y_true) else float("nan")
    auc = safe_auc(y_true, probs)
    if enable_plots:
        plot_confusion(y_true, preds, out_dir / f"{prefix}_confusion_{tag}.png", f"{prefix} [{tag}]")
        plot_roc(y_true, probs, out_dir / f"{prefix}_roc_{tag}.png", f"{prefix} [{tag}]")
    try:
        rep = classification_report(y_true, preds, digits=4)
        save_text(rep, out_dir / f"classification_report_{tag}.txt")
    except Exception:
        pass
    return acc, auc

def detector_confusion(y_is_adv, p_adv, tau, out_path: Path, title: str, enable_plots=True):
    if len(y_is_adv) == 0 or not enable_plots: return
    y_pred = (p_adv >= tau).astype(int)
    plot_confusion(y_is_adv, y_pred, out_path, title)

def _summ_stats(x: np.ndarray):
    x = np.asarray(x).astype(np.float64)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "p5": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(x)), "std": float(np.std(x)),
        "min": float(np.min(x)), "p5": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)), "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }

def _safe_ap(y_true, scores):
    try:
        y_true = np.asarray(y_true); scores = np.asarray(scores)
        if np.unique(y_true).size < 2: return float("nan")
        return float(average_precision_score(y_true, scores))
    except Exception:
        return float("nan")

# -------------------- Utility --------------------
def pick_stream_indices(N: int, stream_size, nat_frac: float, rng: np.random.RandomState):
    if stream_size is None:
        adv_idx = np.arange(N)                 # all ADV eligible
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

def _file_md5(path: Path) -> str:
    if not path.exists(): return "missing"
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Mixed-stream evaluation for Defense-LGBM gate (adversarial detector).")
    p.add_argument("--model", default="all",
                   choices=[
                       "all",
                       "DL-MLP","DL-FTTransformer",
                       "Logistic_Regression","Calibrated_LinearSVC","Gaussian_Naive_Bayes",
                       "Decision_Tree","Random_Forest","AdaBoost","XGBoost","LightGBM"
                   ])
    p.add_argument("--tau-q", type=float, default=None, help="Quantile q for NAT p_adv (default 0.90)")
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
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation for speed.")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    p.add_argument("--amp", action="store_true", help="Use AMP (bfloat16) for Torch eval/attacks.")
    p.add_argument("--cache-adv", action="store_true", help="Cache crafted ADV subsets on disk.")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--fast-predict", action="store_true", help="LightGBM booster fast predict with pred_early_stop.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

# -------------------- Main --------------------
def main():
    args = parse_args()
    if args.tau_q is not None:     CFG["TAU_QUANTILE"] = float(args.tau_q)
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

    rng = np.random.RandomState(CFG["seed"])
    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    models_root = Path(CFG["models_dir"])
    results_root = ensure_dir(Path(CFG["results_dir"]))
    defense_root = Path(CFG["defense_dir"])

    # Globals & dataset
    scaler, feature_cols = load_globals(models_root)
    X_raw, y_base = load_dataset(CFG["dataset_csv"], feature_cols, CFG["label_col"])
    Xz = scaler.transform(X_raw).astype(np.float32, copy=False)
    N = Xz.shape[0]
    z_min = Xz.min(axis=0); z_max = Xz.max(axis=0)
    print(f"[INFO] Loaded dataset: N={N:,}, D={Xz.shape[1]} (source sorted; mixed streams are shuffled)")

    # Load defense & calibrator
    def_path = defense_root / "model.joblib"
    assert def_path.exists(), "Missing Defense-LGBM at models/Defense-LGBM/model.joblib"
    defense = joblib.load(def_path)
    _defense_set_threads(defense)
    calibrator = None
    cal_path = defense_root / "calibrator.joblib"
    if cal_path.exists():
        try:
            calibrator = joblib.load(cal_path)
            print("[INFO] Using isotonic calibrator for defense probabilities.")
        except Exception:
            calibrator = None

    # NAT p_adv and τ (with caching)
    common_out = ensure_dir(results_root / "_common")
    finger = f"{_file_md5(def_path)}_{_file_md5(cal_path)}_{_file_md5(models_root / '_global' / 'scaler.joblib')}_{N}_{len(feature_cols)}"
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
        np.save(nat_cache, p_adv_nat_full)
        print(f"[CACHE] Saved NAT p_adv to {nat_cache.name}")

    tau = choose_tau_from_nat(p_adv_nat_full, CFG["TAU_QUANTILE"])
    print(f"[INFO] τ (NAT quantile={CFG['TAU_QUANTILE']:.2f}) = {tau:.6f}")
    if CFG["use_plots"]:
        plot_hist(p_adv_nat_full, common_out / "p_adv_nat_hist.png", "NAT p_adv distribution (full)")

    # Global surrogate (for sklearn black-box teachers only)
    print("[INFO] Training global surrogate (pretrain) for black-box sklearn attacks...")
    from torch.utils.data import DataLoader, TensorDataset
    class SurrogateMLP(nn.Module):
        def __init__(self, D, H=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(D, H), nn.ReLU(),
                nn.Linear(H, H), nn.ReLU(),
                nn.Linear(H, 1)
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

    # Candidate bases
    candidates = [
        "DL-MLP","DL-FTTransformer",
        "Logistic_Regression","Calibrated_LinearSVC","Gaussian_Naive_Bayes",
        "Decision_Tree","Random_Forest","AdaBoost","XGBoost","LightGBM",
    ]
    if args.model != "all":
        candidates = [args.model]

    # Load available bases
    bases = []
    for name in candidates:
        mdir = models_root / name
        if not mdir.exists():
            print(f"[WARN] Skipping {name}: {mdir} not found.")
            continue
        if name in {"DL-MLP","DL-FTTransformer"}:
            # Torch model
            if name == "DL-MLP":
                mdl = MLPNet(in_dim=Xz.shape[1]).to(device)
            else:
                mdl = FTTransformer(n_features=Xz.shape[1], d_model=64, n_heads=8, n_layers=3, p_drop=0.1).to(device)
            state = torch.load(mdir / "model.pt", map_location=device)
            mdl.load_state_dict(state); mdl.eval()
            bases.append((name, mdl))
        else:
            # Sklearn model
            mpath = mdir / "model.joblib"
            if mpath.exists():
                bases.append((name, joblib.load(mpath)))
            else:
                print(f"[WARN] Skipping {name}: {mpath} not found.")

    assert bases, "No base models found for the given selection."

    # Pre-pick one stream (reused across all bases for comparability & less overhead)
    adv_idx_fixed, nat_idx_fixed = pick_stream_indices(N, CFG["STREAM_SIZE"], CFG["NAT_FRAC"], rng)

    def nat_pure_eval(base_name: str, base_est):
        out_dir = ensure_dir(results_root / base_name / "NAT_pure")
        # Base-only on ALL NAT
        if base_name in {"DL-MLP","DL-FTTransformer"}:
            base_probs_all = eval_torch_batched(base_est, Xz, device, CFG["torch_eval_bs"],
                                                desc=f"{base_name}/eval NAT", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"])
        else:
            base_scores_all = chunked_scores(base_est, Xz, CFG["eval_bs"], desc=f"{base_name}/score NAT", use_progress=CFG["use_progress"])
            base_probs_all = sigmoidify(base_scores_all)
        acc_all, auc_all = base_metrics(y_base, base_probs_all, out_dir, tag="all_nat", prefix="base", enable_plots=CFG["use_plots"])
        # After-defense on ACCEPTED NAT
        accept_nat = (p_adv_nat_full < tau)
        base_probs_acc = base_probs_all[accept_nat]
        y_acc = y_base[accept_nat]
        acc_acc, auc_acc = base_metrics(y_acc, base_probs_acc, out_dir, tag="accepted_nat", prefix="base", enable_plots=CFG["use_plots"])
        save_json({
            "scenario": "NAT_pure",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "samples_nat": int(N),
            "accept_rate_nat": float(accept_nat.mean()),
            "tau": float(tau),
            "base_acc_nat_full": float(acc_all),
            "base_auc_nat_full": float(auc_all),
            "base_acc_nat_accepted": float(acc_acc),
            "base_auc_nat_accepted": float(auc_acc),
        }, out_dir / "metrics.json")
        # free memory
        del base_probs_all, base_probs_acc; cleanup_cuda()

    # --- Helpers: distill & craft ADV on subsets ---
    def distill_from_global(global_surr, teacher, Xz, device, epochs=6, batch=4096, hidden=256, sample_limit=150_000):
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
            def __init__(self, D, H=256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(D, H), nn.ReLU(),
                    nn.Linear(H, H), nn.ReLU(),
                    nn.Linear(H, 1)
                )
            def forward(self, x): return self.net(x).squeeze(-1)
        mdl = SurrogateMLP(D=Xz.shape[1], H=CFG["surrogate_hidden"])
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
            batch=CFG["surrogate_batch"], hidden=CFG["surrogate_hidden"],
            sample_limit=CFG["surrogate_sample_limit"]
        )
        return attack_tabular_torch_batched(
            surr, Xz_sub, y_sub,
            eps=CFG["eps_z"], alpha=CFG["alpha_z"], steps=steps,
            device=device, z_min=z_min, z_max=z_max, batch_size=CFG["bb_attack_bs"],
            desc=f"{base_name}/{which} bb attack", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"]
        )

    def mixed_eval(base_name: str, base_est, which: str, adv_idx: np.ndarray, nat_idx: np.ndarray):
        out_dir = ensure_dir(results_root / base_name / f"mixed_{which}")
        cache_dir = ensure_dir(results_root / base_name / "_cache")

        # Craft ADV on subset (with optional cache)
        steps_used = int(CFG["steps_fgsm"] if which == "FGSM" else CFG["steps_pgd"])
        cache_path = adv_cache_path(results_root / base_name, base_name, which, adv_idx, steps_used, CFG["eps_z"], CFG["alpha_z"])
        use_cached = False
        if CFG["cache_adv"] and cache_path.exists():
            try:
                data = np.load(cache_path)
                cached_idx = data["adv_idx"]
                if np.array_equal(cached_idx, adv_idx):
                    Xz_adv = data["Xz_adv"].astype(np.float32, copy=False)
                    use_cached = True
                    print(f"[CACHE] Loaded ADV subset from {cache_path.name}")
                del data
            except Exception:
                use_cached = False

        if not use_cached:
            Xz_adv = craft_adv_subset(base_name, base_est, which, Xz[adv_idx], y_base[adv_idx])
            if CFG["cache_adv"]:
                ensure_dir(cache_dir)
                np.savez_compressed(cache_path, adv_idx=adv_idx, Xz_adv=Xz_adv)
                print(f"[CACHE] Saved ADV subset to {cache_path.name}")

        # Defense on ADV subset (streamed z->raw; no big buffer)
        p_adv_adv = defense_p_adv_from_z(
            defense, scaler, Xz_adv,
            batch=CFG["eval_bs"], calibrator=calibrator,
            desc=f"defense p_adv ADV/{base_name}/{which}", use_progress=CFG["use_progress"],
            fast_predict=CFG["FAST_PREDICT"], early_stop_margin=CFG["EARLY_STOP_MARGIN"]
        )
        if CFG["use_plots"]:
            plot_hist(p_adv_adv, out_dir / "p_adv_adv_hist.png", f"{which} p_adv distribution")
        cleanup_cuda()

        # Base probabilities on NAT/ADV slices (before mixing)
        if base_name in {"DL-MLP","DL-FTTransformer"}:
            base_probs_nat = eval_torch_batched(base_est, Xz[nat_idx], device, CFG["torch_eval_bs"],
                                                desc=f"{base_name}/eval NAT slice", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"])
            base_probs_adv = eval_torch_batched(base_est, Xz_adv, device, CFG["torch_eval_bs"],
                                                desc=f"{base_name}/eval ADV slice", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"])
        else:
            base_scores_nat = chunked_scores(base_est, Xz[nat_idx], CFG["eval_bs"], desc=f"{base_name}/score NAT slice", use_progress=CFG["use_progress"])
            base_probs_nat = sigmoidify(base_scores_nat)
            base_scores_adv = chunked_scores(base_est, Xz_adv, CFG["eval_bs"], desc=f"{base_name}/score ADV slice", use_progress=CFG["use_progress"])
            base_probs_adv = sigmoidify(base_scores_adv)

        # Labels for slices
        y_base_nat = y_base[nat_idx]
        y_base_adv = y_base[adv_idx]

        # Detector probabilities for NAT slice
        p_adv_nat = p_adv_nat_full[nat_idx]

        # Build mixed arrays and shuffle
        y_is_adv = np.concatenate([np.zeros_like(p_adv_nat, dtype=int),
                                   np.ones_like(p_adv_adv, dtype=int)])
        p_adv_mixed = np.concatenate([p_adv_nat, p_adv_adv])
        base_probs_mixed = np.concatenate([base_probs_nat, base_probs_adv])
        y_base_mixed = np.concatenate([y_base_nat, y_base_adv])

        perm = rng.permutation(p_adv_mixed.shape[0])
        y_is_adv = y_is_adv[perm]
        p_adv_mixed = p_adv_mixed[perm]
        base_probs_mixed = base_probs_mixed[perm]
        y_base_mixed = y_base_mixed[perm]

        # -------- Detector metrics @ tau --------
        det_auc = safe_auc(y_is_adv, p_adv_mixed)
        det_ap  = _safe_ap(y_is_adv, p_adv_mixed)  # PR-AUC (ADV is positive)
        pred_adv = (p_adv_mixed >= tau)
        adv_mask = (y_is_adv == 1)
        nat_mask = (y_is_adv == 0)

        TP = int(np.sum(pred_adv & adv_mask))   # correctly blocked ADV
        FP = int(np.sum(pred_adv & nat_mask))   # mistakenly blocked NAT
        TN = int(np.sum(~pred_adv & nat_mask))  # accepted NAT
        FN = int(np.sum(~pred_adv & adv_mask))  # accepted ADV (got through)
        tpr_at_tau = float(TP / (TP + FN)) if (TP + FN) > 0 else float("nan")
        fpr_nat = float(FP / (FP + TN)) if (FP + TN) > 0 else float("nan")
        tnr_nat = 1.0 - fpr_nat if np.isfinite(fpr_nat) else float("nan")
        prec_at_tau = float(TP / (TP + FP)) if (TP + FP) > 0 else float("nan")
        f1_at_tau = float(2 * prec_at_tau * tpr_at_tau / (prec_at_tau + tpr_at_tau)) if np.isfinite(prec_at_tau) and np.isfinite(tpr_at_tau) and (prec_at_tau + tpr_at_tau) > 0 else float("nan")
        bal_acc_at_tau = float((tpr_at_tau + tnr_nat) / 2) if np.isfinite(tpr_at_tau) and np.isfinite(tnr_nat) else float("nan")

        accept_mask = (~pred_adv)
        acc_rate_overall = float(accept_mask.mean())
        # ---- FIXED: conditional acceptance rates
        nat_total = int(nat_mask.sum())
        adv_total = int(adv_mask.sum())
        acc_rate_nat = float((accept_mask & nat_mask).sum() / nat_total) if nat_total > 0 else float("nan")
        acc_rate_adv = float((accept_mask & adv_mask).sum() / adv_total) if adv_total > 0 else float("nan")

        # Confusion matrix for detector at τ + ROC
        detector_confusion(
            y_is_adv, p_adv_mixed, tau,
            out_dir / "detector_confusion_tau.png",
            f"{base_name} [{which}] detector @ τ",
            enable_plots=CFG["use_plots"]
        )
        if CFG["use_plots"]:
            plot_roc(y_is_adv, p_adv_mixed, out_dir / "detector_roc_nat_vs_adv.png", f"{base_name} [{which}] detector ROC")

        # -------- Base metrics BEFORE defense (slice-level)
        preds_nat_slice = (base_probs_nat >= 0.5).astype(int)
        acc_nat_slice = float(accuracy_score(y_base_nat, preds_nat_slice)) if y_base_nat.size else float("nan")
        auc_nat_slice = safe_auc(y_base_nat, base_probs_nat)

        preds_adv_slice = (base_probs_adv >= 0.5).astype(int)
        acc_adv_slice = float(accuracy_score(y_base_adv, preds_adv_slice)) if y_base_adv.size else float("nan")
        auc_adv_slice = safe_auc(y_base_adv, base_probs_adv)

        # -------- Base metrics AFTER defense (accepted vs rejected, NAT vs ADV)
        mask_acc_nat = (accept_mask & nat_mask)
        mask_rej_nat = (~accept_mask & nat_mask)
        mask_acc_adv = (accept_mask & adv_mask)
        mask_rej_adv = (~accept_mask & adv_mask)

        def _acc_auc(mask):
            if mask.any():
                y_ = y_base_mixed[mask]
                p_ = base_probs_mixed[mask]
                pred_ = (p_ >= 0.5).astype(int)
                acc_ = float(accuracy_score(y_, pred_)) if y_.size else float("nan")
                auc_ = safe_auc(y_, p_)
                return acc_, auc_, int(y_.size)
            else:
                return float("nan"), float("nan"), 0

        base_acc_accepted_nat, base_auc_accepted_nat, n_acc_nat = _acc_auc(mask_acc_nat)
        base_acc_rejected_nat, base_auc_rejected_nat, n_rej_nat = _acc_auc(mask_rej_nat)
        base_acc_accepted_adv, base_auc_accepted_adv, n_acc_adv = _acc_auc(mask_acc_adv)
        base_acc_rejected_adv, base_auc_rejected_adv, n_rej_adv = _acc_auc(mask_rej_adv)

        # -------- Acceptance by label (slice-level, pre-shuffle)
        acc_nat_sep = (p_adv_nat < tau)
        acc_adv_sep = (p_adv_adv < tau)
        def _rate_by_label(acc_mask, labels):
            out = {}
            for lab in [0, 1]:
                idx = (labels == lab)
                out[str(lab)] = float(acc_mask[idx].mean()) if idx.any() else float("nan")
            return out
        acc_rate_nat_by_label = _rate_by_label(acc_nat_sep, y_base_nat)
        acc_rate_adv_by_label = _rate_by_label(acc_adv_sep, y_base_adv)

        # -------- Summaries for research convenience
        p_adv_nat_stats = _summ_stats(p_adv_nat)
        p_adv_adv_stats = _summ_stats(p_adv_adv)
        base_prob_nat_stats = _summ_stats(base_probs_nat)
        base_prob_adv_stats = _summ_stats(base_probs_adv)

        # -------- Save rich metrics JSON
        metrics = {
            "scenario": f"mixed_{which}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_model": base_name,
            "defense_model": "Defense-LGBM",
            "defense_calibrator": bool(calibrator is not None),
            "device": str(device),
            "seed": int(CFG["seed"]),
            "tau": float(tau),
            "tau_quantile": float(CFG["TAU_QUANTILE"]),
            "nat_frac_target": float(CFG["NAT_FRAC"]),
            "stream_size_target": None if CFG["STREAM_SIZE"] is None else int(CFG["STREAM_SIZE"]),
            "stream_nat_count": int(nat_idx.size),
            "stream_adv_count": int(adv_idx.size),
            "samples_total": int(p_adv_mixed.shape[0]),

            # attack knobs
            "attack_kind": which,
            "attack_steps": steps_used,
            "attack_eps_z": float(CFG["eps_z"]),
            "attack_alpha_z": float(CFG["alpha_z"]),

            # batching
            "eval_bs": int(CFG["eval_bs"]),
            "torch_eval_bs": int(CFG["torch_eval_bs"]),
            "torch_attack_bs": int(CFG["torch_attack_bs"]),
            "lin_attack_bs": int(CFG["lin_attack_bs"]),
            "bb_attack_bs": int(CFG["bb_attack_bs"]),

            # --- Detector metrics (global & @tau)
            "detector_auroc_nat_vs_adv": float(det_auc),
            "detector_ap_nat_vs_adv": float(det_ap),
            "detector_counts_at_tau": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
            "tpr_adv_at_tau": float(tpr_at_tau),
            "fpr_nat_at_tau": float(fpr_nat),
            "tnr_nat_at_tau": float(tnr_nat),
            "precision_adv_at_tau": float(prec_at_tau),
            "f1_adv_at_tau": float(f1_at_tau),
            "balanced_accuracy_at_tau": float(bal_acc_at_tau),

            # ---- acceptance (overall + conditional)
            "accept_rate_overall": float(acc_rate_overall),
            "accept_rate_nat": float(acc_rate_nat),
            "accept_rate_adv": float(acc_rate_adv),
            "accept_rate_nat_by_label": acc_rate_nat_by_label,
            "accept_rate_adv_by_label": acc_rate_adv_by_label,

            # --- Base model BEFORE defense (slice-level)
            "base_acc_nat_slice": float(acc_nat_slice),
            "base_auc_nat_slice": float(auc_nat_slice),
            "base_acc_adv_slice": float(acc_adv_slice),
            "base_auc_adv_slice": float(auc_adv_slice),

            # --- Base model AFTER defense (accepted / rejected, NAT / ADV)
            "base_acc_accepted_nat": float(base_acc_accepted_nat),
            "base_auc_accepted_nat": float(base_auc_accepted_nat),
            "base_acc_rejected_nat": float(base_acc_rejected_nat),
            "base_auc_rejected_nat": float(base_auc_rejected_nat),
            "base_acc_accepted_adv": float(base_acc_accepted_adv),
            "base_auc_accepted_adv": float(base_auc_accepted_adv),
            "base_acc_rejected_adv": float(base_acc_rejected_adv),
            "base_auc_rejected_adv": float(base_auc_rejected_adv),
            "counts": {
                "accepted_nat": int(n_acc_nat),
                "rejected_nat": int(n_rej_nat),
                "accepted_adv": int(n_acc_adv),
                "rejected_adv": int(n_rej_adv),
            },

            # --- Distributions
            "p_adv_nat_stats": p_adv_nat_stats,
            "p_adv_adv_stats": p_adv_adv_stats,
            "base_prob_nat_stats": base_prob_nat_stats,
            "base_prob_adv_stats": base_prob_adv_stats,
        }

        save_json(metrics, out_dir / "metrics.json")

        # Visuals
        if CFG["use_plots"]:
            plot_hist(p_adv_nat, out_dir / "p_adv_nat_hist.png", f"{base_name} NAT p_adv ({which} mix)")
            plot_hist(p_adv_adv, out_dir / "p_adv_adv_hist_small.png", f"{base_name} ADV p_adv ({which} mix)")
            plot_roc(y_is_adv, p_adv_mixed, out_dir / "detector_roc_nat_vs_adv.png", f"{base_name} [{which}] detector ROC")

        # free memory
        del Xz_adv, p_adv_adv, base_probs_nat, base_probs_adv, p_adv_nat, p_adv_mixed, base_probs_mixed, y_base_mixed
        cleanup_cuda()

    # ---------- Run per base ----------
    for base_name, base_est in bases:
        print(f"\n==== Evaluating base: {base_name} ====")
        nat_pure_eval(base_name, base_est)
        mixed_eval(base_name, base_est, "FGSM", adv_idx_fixed, nat_idx_fixed)
        mixed_eval(base_name, base_est, "PGD",  adv_idx_fixed, nat_idx_fixed)

    print("\n[DONE] Mixed-stream Defense-LGBM evaluation complete.")

if __name__ == "__main__":
    main()

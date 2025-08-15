#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust adversarial *tabular* dataset generator for defense training (BALANCED on is_adv).

Key balancing behavior
----------------------
• Balance ONLY on is_adv: ADV total ≤ ratio × NAT total (default ratio=1.0).
• Evenly split the ADV budget across PRESENT models so each contributes.
• Optionally distribute each model's budget across attacks to ensure attack diversity.
• Guardrails: warn if parents × max_adv_per_parent < desired ADV budget and tell you the fix.

To hit NAT=2,000,000 and ADV=2,000,000 with your data:
  Use full test split (400k parents) and set max-per-parent=5.
    python prep_defensive_dataset_features.py --ratio 1.0 --max-per-parent 5
"""

from __future__ import annotations
from pathlib import Path
import os, json, gc, warnings, random, argparse, math, hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ------------------------- DEFAULT CONFIG -------------------------
CFG = {
    "input_csv": "features_extracted.csv",
    "out_csv": "features_adversarial_defense_dataset.csv",

    "models_dir": "models",
    "global_dir": "models/_global",

    # Pick ANY present models (we'll skip missing)
    "source_models": [
        "DL-MLP",
        "Logistic_Regression",
        "Decision_Tree",
        "LightGBM",
    ],

    # Which families to craft: any subset of {"FGSM","PGD","CW","NOISE"}
    "attacks": ["FGSM", "PGD", "CW", "NOISE"],

    # Prefer TEST split for attacks (read-only); else create & save once
    "use_test_split": True,

    # NAT is NOT capped by default
    "max_attack_samples": None,   # None => use full test split (recommended)

    "random_state": 42,

    # Streaming / flushing
    "flush_every_rows": 200_000,

    # Batching
    "tab_attack_bs": 4096,   # torch/surrogate attack batch
    "lin_attack_bs": 65536,  # linear white-box batch

    # Device
    "use_gpu": True,

    # Surrogate training (global base and per-teacher distill)
    "surr_base_epochs": 6,
    "surr_base_lr": 3e-4,
    "surr_base_wd": 1e-5,
    "surr_ft_epochs": 6,
    "surr_ft_lr": 3e-4,
    "surr_ft_wd": 1e-5,

    # ---------------- Attack suite knobs (z-space) ----------------
    # ε schedules for FGSM/PGD
    "fgsm_eps_list": [0.2, 0.4, 0.6],
    "pgd_recipes": [  # (eps, alpha, steps, restarts)
        (0.2, 0.05, 10, 1),
        (0.4, 0.05, 20, 2),
        (0.6, 0.10, 30, 4),
    ],
    # CW-L2-like (torch/surrogate)
    "cw_steps": 50,
    "cw_lr": 0.05,
    "cw_c": 1.0,        # tradeoff for loss term
    "cw_kappa": 0.0,    # confidence margin
    "cw_l2_clip": 0.6,  # optional L2 radius clamp around x (None disables)

    # Random noise baselines in z-space
    "noise_uniform_eps_list": [0.2, 0.4, 0.6],    # uniform in [-eps, +eps]
    "noise_gauss_sigma_list": [0.05, 0.10, 0.20], # N(0, sigma^2)
    "noise_dropout_p_list": [0.02, 0.05],         # random dropout + tiny jitter

    # ---------------- Balancing knobs (is_adv only) ----------------
    "target_adv_to_nat_ratio": 1.0,     # 1.0 == perfectly balanced cap on is_adv
    "max_adv_per_parent": 5,            # None for unlimited; 5*400k parents = 2M ADV
    "nat_cap": None,                    # e.g., 2_000_000 to downsample NAT; None = keep all
    "distribute_budget_by_attack": True # split each model's budget across attacks
}

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate adversarial tabular dataset for defense training (robust + balanced on is_adv)")
    p.add_argument("--input", default=None, help="Path to features_extracted.csv")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--models", default=None, help="Models directory (default: models)")
    p.add_argument("--global-dir", default=None, help="Global dir (default: models/_global)")
    p.add_argument("--attacks", default=None, help="Comma list subset of FGSM,PGD,CW,NOISE")
    p.add_argument("--max-attack-samples", type=int, default=None, help="Cap attack subset size; None for full test")
    p.add_argument("--gpu", type=int, default=None, help="1 to use GPU if available, 0 to force CPU")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default 42)")
    # Balancing CLI
    p.add_argument("--ratio", type=float, default=None, help="ADV-to-NAT ratio cap on is_adv (default 1.0)")
    p.add_argument("--max-per-parent", dest="max_per_parent", type=int, default=None,
                   help="Max adversarials per parent_idx across all attacks/models (default 5)")
    p.add_argument("--nat-cap", type=int, default=None, help="Downsample NAT to this many rows before balancing")
    p.add_argument("--no-attack-split", action="store_true", help="Do NOT split per-model budget across attacks")
    return p.parse_args()

# ------------------------- IO helpers -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def append_rows(out_path: Path, rows: list[dict], header_written: bool, feature_cols: list[str]):
    """Stream-append rows to CSV."""
    if not rows:
        return header_written
    cols = feature_cols + ["orig_label", "is_adv", "attack_type", "source_model", "parent_idx"]
    df = pd.DataFrame(rows, columns=cols)
    mode = "a" if (out_path.exists() or header_written) else "w"
    df.to_csv(out_path, mode=mode, header=(mode == "w"), index=False, encoding="utf-8")
    return True

# ------------------------- Index sanitization -------------------------
def _sanitize_indices(idx: np.ndarray, N: int, *, name: str) -> np.ndarray:
    """Drop out-of-range and duplicate indices; return sorted unique array. Warn if anything dropped."""
    arr = np.asarray(idx, dtype=np.int64).ravel()
    mask = (arr >= 0) & (arr < N)
    kept = np.unique(arr[mask])
    dropped = int(arr.size - kept.size)
    if dropped > 0:
        print(f"[WARN] {name}: dropped {dropped:,} invalid/duplicate indices "
              f"(kept {kept.size:,} of {arr.size:,}; dataset size={N:,}).")
    if kept.size == 0:
        print(f"[WARN] {name}: sanitization produced an empty index set; falling back to full range.")
        kept = np.arange(N, dtype=np.int64)
    return kept

# ------------------------- Data & schema -------------------------
def map_label(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip().lower()
    return 1 if s in {"1", "true", "malicious", "phishing", "malware", "bad"} else 0

def load_tabular_scaled(df: pd.DataFrame, global_dir: Path):
    """Load scaler + feature schema; build X (raw) & Xz (scaled) strictly in saved order."""
    scaler_path = global_dir / "scaler.joblib"
    feat_path   = global_dir / "feature_columns.json"
    assert scaler_path.exists(), f"Missing scaler at {scaler_path}"
    assert feat_path.exists(),   f"Missing schema at {feat_path}"

    scaler: StandardScaler = joblib.load(scaler_path)
    saved = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_cols = list(saved.get("feature_columns", [])) if isinstance(saved, dict) else list(saved)
    assert feature_cols, "feature_columns.json has empty feature_columns"

    # Construct X in saved order with numeric coercion and finite fill
    X_df = pd.DataFrame(index=df.index)
    means = getattr(scaler, "mean_", None)
    for i, col in enumerate(feature_cols):
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' missing in input CSV.")
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        mean_val = 0.0 if means is None else float(means[i])
        X_df[col] = s.fillna(mean_val).astype(np.float32)

    X = X_df[feature_cols].to_numpy(dtype=np.float32)
    Xz = scaler.transform(X).astype(np.float32, copy=False)

    # Empirical bounds in z-space for clamping PGD box
    z_min = Xz.min(axis=0)
    z_max = Xz.max(axis=0)
    return X, Xz, feature_cols, scaler, (z_min, z_max)

def infer_feature_domains(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    domains = {"binary": set(), "ratio": set(), "intish": set(), "nonneg": set()}
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        uniq = set(pd.unique(s.astype(np.float32)))
        lowc = c.lower()
        if uniq.issubset({0.0, 1.0}):
            domains["binary"].add(c)
        if "ratio" in lowc or "fraction" in lowc:
            domains["ratio"].add(c)
        if any(k in lowc for k in ["count","length"]) or lowc.startswith("num_"):
            domains["intish"].add(c); domains["nonneg"].add(c)
        if any(k in lowc for k in [
            "count","length","digit","underscore","dash","dot","semicolon",
            "question_mark","hash_char","equal","percent_char","ampersand",
            "at_char","tilde_char","double_slash"
        ]):
            domains["nonneg"].add(c)
    typical_flags = {"ip_as_hostname","exe_in_url","https_in_url","ftp_used","js_used","css_used"}
    for f in typical_flags.intersection(feature_cols):
        domains["binary"].add(f)
    return domains

def enforce_domains(X_raw: np.ndarray, feature_cols: list[str], domains: dict) -> np.ndarray:
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    for c in domains["ratio"]:
        i = col_idx[c]; X_raw[:, i] = np.clip(X_raw[:, i], 0.0, 1.0)
    for c in domains["nonneg"]:
        i = col_idx[c]; X_raw[:, i] = np.maximum(X_raw[:, i], 0.0)
    for c in domains["intish"]:
        i = col_idx[c]; X_raw[:, i] = np.rint(X_raw[:, i])
    for c in domains["binary"]:
        i = col_idx[c]; X_raw[:, i] = (X_raw[:, i] >= 0.5).astype(np.float32)
    return X_raw

# ------------------------- Models -------------------------
def unwrap_calibrated(estimator):
    if isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
            inner = estimator.calibrated_classifiers_[0]
            base = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
            if base is not None:
                return base
    return estimator

def linear_wb(model):
    base = unwrap_calibrated(model)
    assert hasattr(base, "coef_") and hasattr(base, "intercept_"), "Not a linear model with coef_/intercept_."
    w = np.array(base.coef_, dtype=np.float32).reshape(-1)
    b = float(np.array(base.intercept_, dtype=np.float32).reshape(-1)[0])
    return w, b

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
    def forward(self, x):  # returns logit
        return self.head(self.backbone(x)).squeeze(-1)

class SurrogateMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), p_drop=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(p_drop)]
            prev = h
        self.f = nn.Sequential(*layers, nn.Linear(prev, 1))
    def forward(self, x):  # logit
        return self.f(x).squeeze(-1)

# ------------------------- Utilities -------------------------
def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------- Attacks: Linear WB -------------------------
def attack_linear_batched(estimator, Xz, y, steps, eps, alpha, *, loss_kind="logistic", batch=65536, random_start=False):
    w, b = linear_wb(estimator)
    X0 = Xz.astype(np.float32, copy=False)
    Xa = X0.copy()
    if random_start:
        Xa = np.clip(Xa + np.random.uniform(-eps, eps, size=Xa.shape).astype(np.float32), X0 - eps, X0 + eps)
    steps = max(1, int(steps))
    alpha = float(alpha); eps = float(eps)
    y = y.astype(np.float32, copy=False)
    if loss_kind == "hinge":
        ypm = np.where(y > 0.5, 1.0, -1.0).astype(np.float32, copy=False)
    N, D = Xz.shape
    sign_w = np.sign(w).astype(np.float32, copy=False)
    for _ in range(steps):
        i = 0
        while i < N:
            j = min(N, i + batch)
            Z = Xa[i:j] @ w + b
            if loss_kind == "logistic":
                sig = 1.0 / (1.0 + np.exp(-Z))
                residual = sig - y[i:j]
                step_sign = np.sign(residual)[:, None] * sign_w[None, :]
            else:
                m = ypm[i:j] * Z
                active = (m < 1.0).astype(np.float32)
                step_sign = (active[:, None]) * ((-ypm[i:j])[:, None] * sign_w[None, :])
            Xa[i:j] = np.clip(Xa[i:j] + alpha * step_sign, X0[i:j] - eps, X0[i:j] + eps)
            i = j
    return Xa

def deepfool_linear_flip(estimator, Xz):
    w, b = linear_wb(estimator)
    z = Xz @ w + b
    denom = float(np.dot(w, w)) + 1e-12
    r = (-z / denom)[:, None] * w[None, :]
    return (Xz + r.astype(np.float32))

# ------------------------- Attacks: Torch/Surrogate -------------------------
def torch_model_step_grad(model, xa, yb):
    logit = model(xa)
    loss = F.binary_cross_entropy_with_logits(logit, yb)
    grad = torch.autograd.grad(loss, xa, retain_graph=False)[0]
    return grad, logit

def attack_torch_pgd_restarts(model, Xz, y, eps, alpha, steps, device, z_min, z_max, batch_size, restarts=1):
    N, D = Xz.shape
    eps = float(eps); alpha = float(alpha)
    steps = max(1, int(steps)); restarts = max(1, int(restarts))
    zmin_t = torch.from_numpy(z_min).to(device)
    zmax_t = torch.from_numpy(z_max).to(device)
    best_adv = np.empty_like(Xz, dtype=np.float32)
    bs = max(1, int(batch_size))
    ptr = 0
    while ptr < N:
        end = min(N, ptr + bs)
        try:
            x0 = torch.from_numpy(Xz[ptr:end]).float().to(device)
            yb = torch.from_numpy(y[ptr:end].astype(np.float32)).to(device)
            best_chunk = None
            best_loss = None
            for _ in range(restarts):
                xa = x0.clone() if steps == 1 else x0 + (2 * eps * torch.rand_like(x0) - eps)
                xa = torch.max(torch.min(xa, x0 + eps), x0 - eps)
                xa = torch.max(torch.min(xa, zmax_t), zmin_t).detach()
                for _ in range(steps):
                    xa.requires_grad_(True)
                    grad, _ = torch_model_step_grad(model, xa, yb)
                    xa = (xa + alpha * torch.sign(grad)).detach()
                    xa = torch.max(torch.min(xa, x0 + eps), x0 - eps)
                    xa = torch.max(torch.min(xa, zmax_t), zmin_t).detach()
                xa.requires_grad_(True)
                logit = model(xa)
                loss = F.binary_cross_entropy_with_logits(logit, yb, reduction="none")
                loss_sum = loss.sum().detach()
                if best_chunk is None or loss_sum > best_loss:
                    best_chunk = xa.detach()
                    best_loss = loss_sum
            best_adv[ptr:end] = best_chunk.cpu().numpy().astype(np.float32, copy=False)
            ptr = end
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1 and torch.cuda.is_available():
                cleanup_cuda()
                bs = max(1, bs // 2)
                print(f"[WARN] OOM during torch PGD; retrying with batch_size={bs}")
                continue
            raise
    return best_adv

def attack_torch_cw_l2(model, Xz, y, device, z_min, z_max, steps=50, lr=0.05, c=1.0, kappa=0.0, l2_clip=None, batch_size=4096):
    N, D = Xz.shape
    y_pm_np = (2.0 * y.astype(np.float32) - 1.0)
    zmin_t = torch.from_numpy(z_min).to(device)
    zmax_t = torch.from_numpy(z_max).to(device
    )
    adv = np.empty_like(Xz, dtype=np.float32)
    bs = max(1, int(batch_size))
    ptr = 0
    while ptr < N:
        end = min(N, ptr + bs)
        x0 = torch.from_numpy(Xz[ptr:end]).float().to(device)
        y_pm = torch.from_numpy(y_pm_np[ptr:end]).float().to(device)
        delta = torch.zeros_like(x0, requires_grad=True)
        opt = torch.optim.SGD([delta], lr=lr)
        for _ in range(max(1, int(steps))):
            opt.zero_grad()
            xa = x0 + delta
            xa = torch.max(torch.min(xa, zmax_t), zmin_t)
            logit = model(xa)
            loss_margin = F.relu(kappa - y_pm * logit).mean()
            loss_l2 = (delta * delta).sum(dim=1).mean()
            loss = loss_l2 + c * loss_margin
            loss.backward()
            opt.step()
            if l2_clip is not None and l2_clip > 0:
                with torch.no_grad():
                    d = delta
                    nrm = torch.norm(d.view(d.size(0), -1), dim=1, keepdim=True) + 1e-12
                    fac = torch.clamp(l2_clip / nrm, max=1.0)
                    delta.mul_(fac.view(-1, 1))
            with torch.no_grad():
                xa = torch.max(torch.min(x0 + delta, zmax_t), zmin_t)
                delta.copy_(xa - x0)
        adv[ptr:end] = (x0 + delta).detach().cpu().numpy().astype(np.float32, copy=False)
        ptr = end
    return adv

# ---------------- Surrogate training (base + fine-tune) ----------------
def train_surrogate_base(in_dim, Xz_tr, y_tr, device, epochs=6, lr=3e-4, wd=1e-5, num_workers=2):
    model = SurrogateMLP(in_dim=in_dim).to(device).train()
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], device=device)
    ds = TensorDataset(torch.from_numpy(Xz_tr).float(), torch.from_numpy(y_tr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=8192, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best, best_loss, patience, noimp = None, float("inf"), 2, 0
    for ep in range(1, epochs + 1):
        run, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_bce(logits, yb)
            loss.backward(); opt.step()
            run += loss.item() * xb.size(0); n += xb.size(0)
        avg = run / max(1, n)
        print(f"[Surr-Base] epoch {ep}/{epochs} loss={avg:.6f}")
        if avg < best_loss - 1e-6:
            best_loss = avg
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
            if noimp >= patience:
                break
    if best is None:
        best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k, v in best.items()})
    model.eval()
    return model

def distill_from_base(base_model: SurrogateMLP, teacher, Xz_tr, device, tag, epochs=6, lr=3e-4, wd=1e-5, num_workers=2):
    if hasattr(teacher, "predict_proba"):
        soft = teacher.predict_proba(Xz_tr)
        soft = soft[:, 1] if isinstance(soft, np.ndarray) and soft.ndim == 2 else np.asarray(soft).ravel()
    elif hasattr(teacher, "decision_function"):
        df = teacher.decision_function(Xz_tr)
        soft = 1.0 / (1.0 + np.exp(-np.asarray(df).ravel()))
    else:
        soft = teacher.predict(Xz_tr).astype(np.float32)
    mdl = SurrogateMLP(in_dim=Xz_tr.shape[1]).to(device)
    mdl.load_state_dict(base_model.state_dict())
    mdl.train()
    ds = TensorDataset(torch.from_numpy(Xz_tr).float(), torch.from_numpy(soft.astype(np.float32)))
    dl = DataLoader(ds, batch_size=8192, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=wd)
    best, best_loss, patience, noimp = None, float("inf"), 2, 0
    for ep in range(1, epochs + 1):
        run, n = 0.0, 0
        for xb, sb in dl:
            xb, sb = xb.to(device), sb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = mdl(xb)
            loss = F.binary_cross_entropy_with_logits(logits, sb)
            loss.backward(); opt.step()
            run += loss.item() * xb.size(0); n += xb.size(0)
        avg = run / max(1, n)
        print(f"[Distill {tag}] epoch {ep}/{epochs} loss={avg:.6f}")
        if avg < best_loss - 1e-6:
            best_loss = avg
            best = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
            if noimp >= patience:
                break
    if best is None:
        best = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
    mdl.load_state_dict({k: v.to(device) for k, v in best.items()})
    mdl.eval()
    return mdl

# ------------------------- Preprocessing drift check -------------------------
def write_preproc_drift_report(global_dir: Path, scaler: StandardScaler, Xz_full: np.ndarray, train_idx: np.ndarray):
    report = {}
    cur_mean = Xz_full[train_idx].mean(axis=0)
    cur_std  = Xz_full[train_idx].std(axis=0)
    report["z_train_mean_abs_max"] = float(np.max(np.abs(cur_mean)))
    report["z_train_std_deviation_from_1_abs_max"] = float(np.max(np.abs(cur_std - 1.0)))
    (global_dir / "preprocessing_drift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote preprocessing drift report -> {global_dir / 'preprocessing_drift.json'}")

# ------------------------- Main pipeline -------------------------
def main():
    args = parse_args()
    if args.input:       CFG["input_csv"] = args.input
    if args.out:         CFG["out_csv"] = args.out
    if args.models:      CFG["models_dir"] = args.models
    if args.global_dir:  CFG["global_dir"] = args.global_dir
    if args.attacks:     CFG["attacks"] = [s.strip().upper() for s in args.attacks.split(",") if s.strip()]
    if args.max_attack_samples is not None: CFG["max_attack_samples"] = args.max_attack_samples
    if args.gpu is not None: CFG["use_gpu"] = bool(args.gpu)
    if args.seed is not None: CFG["random_state"] = int(args.seed)
    if args.ratio is not None: CFG["target_adv_to_nat_ratio"] = float(args.ratio)
    if args.max_per_parent is not None: CFG["max_adv_per_parent"] = args.max_per_parent
    if args.nat_cap is not None: CFG["nat_cap"] = int(args.nat_cap)
    if args.no_attack_split: CFG["distribute_budget_by_attack"] = False

    seed_everything(CFG["random_state"])
    rng_global = np.random.default_rng(CFG["random_state"])

    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Device: {device}")

    models_root = Path(CFG["models_dir"])
    global_dir = Path(CFG["global_dir"])
    ensure_dir(global_dir)
    out_path = Path(CFG["out_csv"])
    if out_path.exists():
        out_path.unlink()

    # Load dataset
    df = pd.read_csv(CFG["input_csv"])
    assert "label" in df.columns, f"{CFG['input_csv']} must contain column 'label'"
    df["orig_label"] = df["label"].apply(map_label).astype(np.int64)

    # Build X (raw), Xz (scaled)
    X_raw_full, Xz_full, feature_cols, scaler, (z_min, z_max) = load_tabular_scaled(df, global_dir)
    domains = infer_feature_domains(df, feature_cols)
    N = Xz_full.shape[0]
    y_all = df["orig_label"].to_numpy()

    # Global split
    split_path = global_dir / "split_indices.json"
    if split_path.exists():
        info = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx_raw = np.array(info["train_idx"], dtype=np.int64)
        test_idx_raw  = np.array(info["test_idx"],  dtype=np.int64)
        train_idx = _sanitize_indices(train_idx_raw, N, name="global train_idx")
        test_idx  = _sanitize_indices(test_idx_raw,  N, name="global test_idx")
        print(f"[INFO] Loaded global split (sanitized): train={len(train_idx):,}, test={len(test_idx):,}")
    else:
        train_idx, test_idx = train_test_split(
            np.arange(N, dtype=np.int64),
            test_size=0.20, random_state=CFG["random_state"], stratify=y_all
        )
        save_json({"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}, split_path)
        print(f"[INFO] Created global split: train={len(train_idx):,}, test={len(test_idx):,} -> {split_path}")

    # Drift sanity
    write_preproc_drift_report(global_dir, scaler, Xz_full, train_idx)

    # NAT write (optional cap)
    if CFG["nat_cap"] is not None and CFG["nat_cap"] < N:
        nat_indices = np.sort(rng_global.choice(np.arange(N, dtype=np.int64), size=int(CFG["nat_cap"]), replace=False))
        print(f"[PHASE] NAT down-sampled: {len(nat_indices):,} from {N:,}")
    else:
        nat_indices = np.arange(N, dtype=np.int64)
        print("[PHASE] Writing ALL NAT rows (entire dataset)...")

    header_written = False
    rows = []
    for j in nat_indices:
        feats = {c: float(X_raw_full[j, k]) for k, c in enumerate(feature_cols)}
        rows.append({
            **feats,
            "orig_label": int(y_all[j]),
            "is_adv": 0,
            "attack_type": "NAT",
            "source_model": "N/A",
            "parent_idx": int(j)
        })
        if len(rows) >= CFG["flush_every_rows"]:
            header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()
    header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()
    print("[PHASE] NAT write complete.")

    # ADV budget based on NAT count (balance on is_adv only)
    nat_total = int(len(nat_indices))
    adv_budget_total = int(CFG["target_adv_to_nat_ratio"] * nat_total)
    print(f"[BALANCE] NAT={nat_total:,} | ADV budget total (is_adv=1)={adv_budget_total:,}")

    # Attack subset (parents)
    if CFG["use_test_split"]:
        idx = test_idx.copy(); src = "test split"
    else:
        idx = np.arange(N, dtype=np.int64); src = "full dataset"
    if CFG["max_attack_samples"] is not None and len(idx) > CFG["max_attack_samples"]:
        idx = np.sort(rng_global.choice(idx, size=int(CFG["max_attack_samples"]), replace=False))
    idx = _sanitize_indices(idx, N, name="attack subset")
    print(f"[INFO] Attack subset size (parents)={len(idx):,} (from {src})")

    # Capacity check
    parents = len(idx)
    per_parent_cap = CFG["max_adv_per_parent"] if CFG["max_adv_per_parent"] is not None else 1_000_000_000
    capacity = parents * per_parent_cap
    if capacity < adv_budget_total:
        need_k = math.ceil(adv_budget_total / max(1, parents))
        print(f"[WARN] ADV capacity too small: parents×per_parent={capacity:,} < budget {adv_budget_total:,}.")
        print(f"       Increase --max-attack-samples to grow parents (max {len(test_idx):,}), or set --max-per-parent to ≥ {need_k}.")
    Xz = Xz_full[idx]; y = y_all[idx]

    # Train surrogate base
    print("[PHASE] Training global surrogate (base) on train split...")
    Xz_tr = Xz_full[train_idx]; y_tr = y_all[train_idx]
    surr_base = train_surrogate_base(in_dim=Xz_tr.shape[1], Xz_tr=Xz_tr, y_tr=y_tr,
                                     device=device, epochs=CFG["surr_base_epochs"],
                                     lr=CFG["surr_base_lr"], wd=CFG["surr_base_wd"])
    base_path = global_dir / "surrogate_base.pt"
    torch.save(surr_base.state_dict(), base_path)
    print(f"[INFO] Saved global surrogate base -> {base_path}")

    def load_base_surrogate():
        m = SurrogateMLP(in_dim=Xz_tr.shape[1])
        m.load_state_dict(torch.load(base_path, map_location=device))
        m.to(device).eval()
        return m

    # Present models
    present_models = []
    for name in CFG["source_models"]:
        mdir = models_root / name
        if (mdir / "model.pt").exists() or (mdir / "model.joblib").exists():
            present_models.append(name)
        else:
            print(f"[WARN] Skipping {name}: no model found at {mdir}")
    if not present_models:
        raise RuntimeError("No source models present to attack.")
    M = len(present_models)

    # Per-model budget
    model_budget = {m: adv_budget_total // M for m in present_models}
    remainder = adv_budget_total - sum(model_budget.values())
    # give the remainder deterministically
    for m in present_models[:remainder]:
        model_budget[m] += 1
    print(f"[BALANCE] Per-model ADV budgets: " + ", ".join(f"{m}={model_budget[m]:,}" for m in present_models))

    # Per-attack sub-budgets (optional)
    att_order = []
    if "FGSM" in CFG["attacks"]:
        for eps in CFG["fgsm_eps_list"]:
            att_order.append(("FGSM", {"eps": eps}))
    if "PGD" in CFG["attacks"]:
        for (eps, alpha, steps, restarts) in CFG["pgd_recipes"]:
            att_order.append(("PGD", {"eps": eps, "alpha": alpha, "steps": steps, "restarts": restarts}))
    if "CW" in CFG["attacks"]:
        att_order.append(("CW", {}))
    if "NOISE" in CFG["attacks"]:
        for eps in CFG["noise_uniform_eps_list"]:
            att_order.append(("NOISE_UNIF", {"eps": eps}))
        for sigma in CFG["noise_gauss_sigma_list"]:
            att_order.append(("NOISE_GAUSS", {"sigma": sigma}))
        for p in CFG["noise_dropout_p_list"]:
            att_order.append(("NOISE_DROPOUT", {"p": p}))

    # Selection helpers/state
    header = {"header_written": header_written}
    parent_adv_count = defaultdict(int)

    def _salt(tag: str) -> int:
        h = hashlib.blake2b(tag.encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16) & 0x7fffffff

    def _write_selected_rows(X_adv_raw, selected_rows, atk_tag, model_name):
        rows = []
        for j in selected_rows:
            feats = {c: float(X_adv_raw[j, k]) for k, c in enumerate(feature_cols)}
            rows.append({
                **feats,
                "orig_label": int(y[j]),
                "is_adv": 1,
                "attack_type": atk_tag,
                "source_model": model_name,
                "parent_idx": int(idx[j]),
            })
            if len(rows) >= CFG["flush_every_rows"]:
                header["header_written"] = append_rows(out_path, rows, header["header_written"], feature_cols); rows.clear()
        header["header_written"] = append_rows(out_path, rows, header["header_written"], feature_cols); rows.clear()

    # Attack loop with per-model budgets
    for name in present_models:
        mdir = models_root / name
        is_torch = (mdir / "model.pt").exists()
        is_sklearn = (mdir / "model.joblib").exists()
        print(f"\n[PHASE] Attacking model: {name} (budget={model_budget[name]:,})")

        # Load teacher
        if is_torch:
            mdl = MLPNet(in_dim=Xz.shape[1]).to(device)
            state = torch.load(mdir / "model.pt", map_location=device)
            mdl.load_state_dict(state); mdl.eval()
        else:
            mdl = joblib.load(mdir / "model.joblib")

        # Determine family / surrogate
        is_linear = name in {"Logistic_Regression", "Calibrated_LinearSVC"}
        is_blackbox = name in {"Decision_Tree","Random_Forest","LightGBM","XGBoost","AdaBoost","Gaussian_Naive_Bayes"}

        surrogate_for_attacks = None
        if is_blackbox and any(t in CFG["attacks"] for t in ["PGD","FGSM","CW"]):
            print(f"[INFO] Distilling surrogate from global base for {name}...")
            base = load_base_surrogate()
            surrogate_for_attacks = distill_from_base(
                base_model=base, teacher=mdl, Xz_tr=Xz_tr, device=device, tag=f"{name}_Surr",
                epochs=CFG["surr_ft_epochs"], lr=CFG["surr_ft_lr"], wd=CFG["surr_ft_wd"]
            )

        remaining_for_model = model_budget[name]
        # Split per attack if requested
        if CFG["distribute_budget_by_attack"] and len(att_order) > 0:
            per_attack = {i: remaining_for_model // len(att_order) for i in range(len(att_order))}
            for i in range(remaining_for_model - sum(per_attack.values())):
                per_attack[i] += 1
        else:
            per_attack = {i: remaining_for_model if i == 0 else 0 for i in range(max(1, len(att_order)))}

        # For deterministic mixing, pre-shuffle candidate indices per (model,attack)
        shuffled = {}
        for i, (kind, params) in enumerate(att_order if att_order else [("NONE", {})]):
            rng = np.random.default_rng(CFG["random_state"] + _salt(f"{name}:{kind}:{json.dumps(params, sort_keys=True)}"))
            perm = np.arange(len(idx), dtype=np.int64)
            rng.shuffle(perm)
            shuffled[i] = perm

        def craft(kind, params):
            if kind == "FGSM":
                eps = params["eps"]
                if is_torch:
                    X_adv_z = attack_torch_pgd_restarts(mdl, Xz, y, eps=eps, alpha=eps, steps=1,
                                                        device=device, z_min=z_min, z_max=z_max,
                                                        batch_size=CFG["tab_attack_bs"], restarts=1)
                elif is_linear:
                    loss_kind = "hinge" if name == "Calibrated_LinearSVC" else "logistic"
                    X_adv_z = attack_linear_batched(mdl, Xz, y, steps=1, eps=eps, alpha=eps,
                                                    loss_kind=loss_kind, batch=CFG["lin_attack_bs"], random_start=False)
                else:
                    X_adv_z = attack_torch_pgd_restarts(surrogate_for_attacks, Xz, y, eps=eps, alpha=eps, steps=1,
                                                        device=device, z_min=z_min, z_max=z_max,
                                                        batch_size=CFG["tab_attack_bs"], restarts=1)
                tag = f"FGSM_eps{eps}"
                return tag, X_adv_z
            if kind == "PGD":
                eps, alpha, steps, restarts = params["eps"], params["alpha"], params["steps"], params["restarts"]
                if is_torch:
                    X_adv_z = attack_torch_pgd_restarts(mdl, Xz, y, eps=eps, alpha=alpha, steps=steps,
                                                        device=device, z_min=z_min, z_max=z_max,
                                                        batch_size=CFG["tab_attack_bs"], restarts=restarts)
                elif is_linear:
                    loss_kind = "hinge" if name == "Calibrated_LinearSVC" else "logistic"
                    X_adv_z = attack_linear_batched(mdl, Xz, y, steps=steps, eps=eps, alpha=alpha,
                                                    loss_kind=loss_kind, batch=CFG["lin_attack_bs"], random_start=True)
                else:
                    X_adv_z = attack_torch_pgd_restarts(surrogate_for_attacks, Xz, y, eps=eps, alpha=alpha, steps=steps,
                                                        device=device, z_min=z_min, z_max=z_max,
                                                        batch_size=CFG["tab_attack_bs"], restarts=restarts)
                tag = f"PGD_eps{eps}_a{alpha}_s{steps}_r{restarts}"
                return tag, X_adv_z
            if kind == "CW":
                if is_torch:
                    X_adv_z = attack_torch_cw_l2(mdl, Xz, y, device=device, z_min=z_min, z_max=z_max,
                                                 steps=CFG["cw_steps"], lr=CFG["cw_lr"],
                                                 c=CFG["cw_c"], kappa=CFG["cw_kappa"],
                                                 l2_clip=CFG["cw_l2_clip"], batch_size=CFG["tab_attack_bs"])
                elif is_linear:
                    Xa = deepfool_linear_flip(mdl, Xz)
                    if CFG["cw_l2_clip"] is not None:
                        r = Xa - Xz
                        nrm = np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
                        fac = np.minimum(1.0, CFG["cw_l2_clip"] / nrm)
                        Xa = Xz + r * fac
                    X_adv_z = Xa.astype(np.float32, copy=False)
                else:
                    X_adv_z = attack_torch_cw_l2(surrogate_for_attacks, Xz, y, device=device, z_min=z_min, z_max=z_max,
                                                 steps=CFG["cw_steps"], lr=CFG["cw_lr"],
                                                 c=CFG["cw_c"], kappa=CFG["cw_kappa"],
                                                 l2_clip=CFG["cw_l2_clip"], batch_size=CFG["tab_attack_bs"])
                tag = f"CW_L2_c{CFG['cw_c']}_k{CFG['cw_kappa']}_s{CFG['cw_steps']}"
                return tag, X_adv_z
            if kind == "NOISE_UNIF":
                eps = params["eps"]
                R = (np.random.rand(*Xz.shape).astype(np.float32) * 2.0 - 1.0) * float(eps)
                X_adv_z = np.clip(Xz + R, Xz - eps, Xz + eps).astype(np.float32, copy=False)
                return f"NOISE_UNIF_Linf_eps{eps}", X_adv_z
            if kind == "NOISE_GAUSS":
                sigma = params["sigma"]
                R = np.random.normal(loc=0.0, scale=float(sigma), size=Xz.shape).astype(np.float32)
                return f"NOISE_GAUSS_sigma{sigma}", (Xz + R).astype(np.float32, copy=False)
            if kind == "NOISE_DROPOUT":
                p = params["p"]
                mask = (np.random.rand(*Xz.shape) > float(p)).astype(np.float32)
                jitter = np.random.normal(0.0, 0.01, size=Xz.shape).astype(np.float32)
                return f"NOISE_DROPOUT_p{p}", (Xz * mask + jitter).astype(np.float32, copy=False)
            # Fallback (shouldn't happen)
            return "NONE", Xz.copy()

        # Iterate attacks; for each, pick only as many rows as that attack's share
        for i, (kind, params) in enumerate(att_order if att_order else [("NONE", {})]):
            need = per_attack.get(i, 0)
            if need <= 0:
                continue
            atk_tag, X_adv_z = craft(kind, params)
            X_adv_raw = enforce_domains(scaler.inverse_transform(X_adv_z).astype(np.float32, copy=False),
                                        feature_cols, domains)

            # walk shuffled candidates, respect per-parent cap and per-attack quota for this model
            perm = shuffled[i]
            picked = []
            for j in perm:
                if len(picked) >= need:
                    break
                pidx = int(idx[j])
                if CFG["max_adv_per_parent"] is not None and parent_adv_count[pidx] >= CFG["max_adv_per_parent"]:
                    continue
                picked.append(j)
                parent_adv_count[pidx] += 1

            if not picked:
                print(f"[BALANCE] {name}:{atk_tag} could not pick candidates (per-parent cap or exhaustion).")
                continue

            _write_selected_rows(X_adv_raw, picked, atk_tag, name)
            print(f"[BALANCE] {name}:{atk_tag} wrote {len(picked):,}/{need:,} for this attack.")

    print("\n[STATS] Wrote:", out_path)
    try:
        tmp = pd.read_csv(out_path, usecols=["attack_type","source_model","is_adv"])
        print("Counts by is_adv:\n", tmp["is_adv"].value_counts(dropna=False).to_string())
        if (tmp["is_adv"] == 1).any():
            print("\nADV rows by model:\n", tmp.loc[tmp["is_adv"]==1, "source_model"].value_counts().to_string())
            print("\nADV rows by attack (top 12):\n", tmp.loc[tmp["is_adv"]==1, "attack_type"].value_counts().head(12).to_string())
    except Exception as e:
        print(f"[WARN] Could not summarize counts: {e}")
    print("[OK] Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-stream evaluation for Defense-LGBM (gate trained on is_adv) with mixed NAT + adversarial traffic.

What this does (per base model)
-------------------------------
1) Loads GLOBAL schema/scaler + full dataset (features_extracted.csv).
2) Loads Defense-LGBM gate and optional isotonic calibrator.
3) Picks τ from NAT p_adv quantile (default q=0.90).
4) Evaluates:
   A) NAT_pure:
      - Base-only metrics on ALL NAT (before defense).
      - Base metrics on ACCEPTED NAT (after defense).
      - Confusion matrices for both.
   B) mixed_FGSM and mixed_PGD:
      - Craft ADV on whole dataset (chunked; linear=white-box, others=surrogate black-box).
      - Mix with NAT according to NAT_FRAC (default 0.5), shuffle to interleave.
      - Detector ROC-AUC (NAT vs ADV), TPR@τ on ADV, FPR@τ on NAT.
      - Base metrics on ACCEPTED NAT subset (post-gate) + confusion matrices.
      - Detector confusion matrix (NAT vs ADV) at τ.

Outputs
-------
results_defense_feat_stream/<Base>/
  NAT_pure/
    metrics.json
    base_confusion_all_nat.png
    base_confusion_accepted_nat.png
    base_roc_all_nat.png
    base_roc_accepted_nat.png
    classification_report_all_nat.txt
    classification_report_accepted_nat.txt
    p_adv_nat_hist.png
  mixed_FGSM/ (and mixed_PGD/)
    metrics.json
    detector_confusion_tau.png
    base_confusion_accepted_nat.png
    base_roc_accepted_nat.png
    classification_report_accepted_nat.txt
    p_adv_nat_hist.png
    p_adv_adv_hist.png
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, gc, warnings, argparse
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay,
    accuracy_score, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -------------------- Config --------------------
CFG = {
    "models_dir": "models",
    "defense_dir": "models/Defense-LGBM",
    "results_dir": "results_defense_feat_stream",
    "dataset_csv": "features_extracted.csv",
    "label_col": "label",      # base-task label (0/1)

    # z-space attack budgets
    "eps_z": 0.40,
    "alpha_z": 0.10,
    "steps_fgsm": 1,           # FGSM
    "steps_pgd": 5,            # small PGD

    # surrogate (for non-linear bases)
    "surrogate_hidden": 256,
    "surrogate_epochs_global": 5,
    "surrogate_epochs_finetune": 6,
    "surrogate_batch": 4096,
    "surrogate_sample_limit": 150_000,

    # batch sizes with back-off
    "eval_bs": 100_000,
    "lin_attack_bs": 32_768,
    "bb_attack_bs": 8_192,

    # device
    "use_gpu": True,

    # NAT threshold selection (q-quantile on NAT p_adv)
    "TAU_QUANTILE": 0.90,

    # Mixed-stream composition
    "NAT_FRAC": 0.5,           # fraction of NAT in the mixed stream
    "STREAM_SIZE": None,       # None => use all ADV (N), NAT = floor(N * NAT_FRAC). Set int to cap total.

    # RNG
    "seed": 42,
}

LABELS = [0, 1]  # always draw both axes on CMs

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

# -------------------- Model scoring utils --------------------
def score_vector(estimator, X):
    """Return proba for class 1 if available; else decision_function; else predictions as float."""
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if isinstance(p, list): p = p[0]
        if getattr(p, "ndim", 1) == 2 and p.shape[1] == 2:
            return p[:, 1]
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X); return np.asarray(s).ravel()
    pred = estimator.predict(X); return np.asarray(pred, dtype=float).ravel()

def chunked_scores(est, X: np.ndarray, chunk: int):
    """Chunked scoring with OOM back-off."""
    N = X.shape[0]; out = np.empty((N,), dtype=np.float32)
    bs = max(4096, int(chunk)); s = 0
    while s < N:
        e = min(N, s + bs)
        try:
            out[s:e] = np.asarray(score_vector(est, X[s:e])).ravel().astype(np.float32, copy=False)
            s = e
        except Exception as ex:
            if "MemoryError" in str(ex) or "CUDA out of memory" in str(ex) or bs > 4096:
                bs = max(4096, bs // 2); cleanup_cuda(); continue
            raise
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

# -------------------- Defense (p_adv) --------------------
def defense_p_adv(def_model, X_raw, batch=100_000, calibrator=None):
    """p_adv with isotonic calibrator.predict if available; chunked."""
    N = X_raw.shape[0]; probs = np.zeros(N, dtype=np.float32)
    bs = max(10_000, int(batch)); i = 0
    while i < N:
        j = min(N, i + bs)
        try:
            p1 = score_vector(def_model, X_raw[i:j])
        except Exception as ex:
            if "MemoryError" in str(ex) or bs > 10_000:
                bs = max(10_000, bs // 2); cleanup_cuda(); continue
            raise
        p1 = np.asarray(p1)
        if p1.min() < 0 or p1.max() > 1: p1 = 1.0 / (1.0 + np.exp(-p1))
        if calibrator is not None:
            try: p1 = calibrator.predict(p1)
            except Exception: pass
        probs[i:j] = p1.astype(np.float32, copy=False)
        i = j
    return probs  # p_adv

def choose_tau_from_nat(p_adv_nat: np.ndarray, q: float) -> float:
    q = float(np.clip(q, 0.50, 0.999))
    if len(p_adv_nat) == 0: return 0.5
    return float(np.quantile(p_adv_nat, q))

# -------------------- Attacks --------------------
def linear_params(estimator):
    est = unwrap_calibrated(estimator)
    if hasattr(est, "coef_") and hasattr(est, "intercept_"):
        w = np.array(est.coef_, dtype=np.float32).reshape(1, -1)
        b = np.array(getattr(est, "intercept_", np.zeros(w.shape[1], dtype=np.float32)), dtype=np.float32).reshape(-1)
        w = (w[0] if w.shape[0] == 1 else np.mean(w, axis=0)).astype(np.float32, copy=False)
        b = np.float32(b[0] if b.size else 0.0)
        return w, b
    raise RuntimeError("Linear parameters not found for white-box attack.")

def attack_linear_fgsm_pgd(estimator, Xz, y, steps, eps, alpha, loss_kind="logistic", batch=32_768):
    w, b = linear_params(estimator)
    sign_w = np.sign(w).astype(np.float32)
    X_adv = Xz.copy().astype(np.float32, copy=False); X0 = Xz.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    if loss_kind == "hinge": ypm = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    N = Xz.shape[0]; steps = max(1, int(steps)); eps = np.float32(eps); alpha = np.float32(alpha)
    s = 0
    while s < N:
        e = min(N, s + batch)
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
        X_adv[s:e] = Xa; s = e
    return X_adv

class SurrogateMLP(nn.Module):
    def __init__(self, D, H=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_global_surrogate(Xz, y, device, epochs=5, batch=4096, hidden=256):
    mdl = SurrogateMLP(D=Xz.shape[1], H=hidden).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=3e-4, weight_decay=1e-5)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xz).float(), torch.from_numpy(y.astype(np.float32)))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    mdl.train(); best, best_loss = None, float("inf")
    for ep in range(1, epochs+1):
        run=0.0; n=0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logit = mdl(xb); loss = F.binary_cross_entropy_with_logits(logit, yb)
            loss.backward(); opt.step()
            run += loss.item() * xb.size(0); n += xb.size(0)
        avg = run / max(1, n)
        print(f"[GlobalSurrogate] epoch {ep}/{epochs} loss={avg:.6f}")
        if avg < best_loss - 1e-5:
            best_loss = avg; best = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
    if best is not None:
        mdl.load_state_dict({k: v.to(device) for k, v in best.items()})
    mdl.eval(); return mdl

def distill_from_global(global_surr, teacher, Xz, device, epochs=6, batch=4096, hidden=256, sample_limit=150_000):
    N = Xz.shape[0]; idx = np.arange(N)
    if N > sample_limit:
        idx = np.random.RandomState(42).choice(N, size=sample_limit, replace=False)
    soft = score_vector(teacher, Xz[idx]); soft = sigmoidify(np.asarray(soft, dtype=np.float32))
    mdl = SurrogateMLP(D=Xz.shape[1], H=hidden)
    mdl.load_state_dict(global_surr.state_dict()); mdl = mdl.to(device)
    opt = torch.optim.Adam(mdl.parameters(), lr=3e-4, weight_decay=1e-5)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xz[idx]).float(), torch.from_numpy(soft).float())
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    mdl.train(); best, best_loss = None, float("inf"); patience, bad=2, 0
    for ep in range(1, epochs+1):
        run=0.0; n=0
        for xb, sb in dl:
            xb, sb = xb.to(device), sb.to(device)
            opt.zero_grad(set_to_none=True)
            logit = mdl(xb); loss = F.binary_cross_entropy_with_logits(logit, sb)
            loss.backward(); opt.step()
            run += loss.item() * xb.size(0); n += xb.size(0)
        avg = run / max(1, n)
        print(f"[Distill] epoch {ep}/{epochs} loss={avg:.6f}")
        if avg < best_loss - 1e-5:
            best_loss = avg; best = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}; bad=0
        else:
            bad += 1
            if bad >= patience: break
    if best is not None:
        mdl.load_state_dict({k: v.to(device) for k, v in best.items()})
    mdl.eval(); return mdl

def attack_surrogate_fgsm_pgd(surr, Xz, y, steps, eps, alpha, device, batch=8_192):
    X_adv = Xz.copy().astype(np.float32, copy=False)
    X0 = Xz.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    steps = max(1, int(steps)); eps = np.float32(eps); alpha = np.float32(alpha)
    i = 0
    while i < Xz.shape[0]:
        j = min(Xz.shape[0], i + batch)
        xa = torch.from_numpy(X_adv[i:j]).float().to(device)
        yb = torch.from_numpy(y[i:j]).float().to(device)
        x0 = torch.from_numpy(X0[i:j]).to(device)
        for _ in range(steps):
            xa.requires_grad_(True)
            logits = surr(xa); loss = F.binary_cross_entropy_with_logits(logits, yb)
            grad = torch.autograd.grad(loss, xa, retain_graph=False, create_graph=False)[0]
            xa = (xa + alpha * torch.sign(grad)).detach()
            xa = torch.max(torch.min(xa, x0 + eps), x0 - eps)
        X_adv[i:j] = xa.detach().cpu().numpy().astype(np.float32, copy=False); i = j
    return X_adv

# -------------------- Metrics helpers --------------------
def safe_auc(y_true, scores) -> float:
    try:
        y_true = np.asarray(y_true); scores = np.asarray(scores)
        if np.unique(y_true).size < 2: return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")

def base_metrics(y_true, probs, out_dir: Path, tag: str, prefix: str):
    """Write base confusion + ROC + report; return (acc, auc)."""
    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, preds)) if len(y_true) else float("nan")
    auc = safe_auc(y_true, probs)
    plot_confusion(y_true, preds, out_dir / f"{prefix}_confusion_{tag}.png", f"{prefix} [{tag}]")
    plot_roc(y_true, probs, out_dir / f"{prefix}_roc_{tag}.png", f"{prefix} [{tag}]")
    try:
        rep = classification_report(y_true, preds, digits=4)
        save_text(rep, out_dir / f"classification_report_{tag}.txt")
    except Exception:
        pass
    return acc, auc

def detector_confusion(y_is_adv, p_adv, tau, out_path: Path, title: str):
    if len(y_is_adv) == 0: return
    y_pred = (p_adv >= tau).astype(int)  # predict ADV when p_adv >= tau
    plot_confusion(y_is_adv, y_pred, out_path, title)

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Mixed-stream evaluation for Defense-LGBM gate (adversarial detector).")
    p.add_argument("--model", default="all",
                   choices=["all","Logistic_Regression","Calibrated_LinearSVC","Gaussian_Naive_Bayes",
                            "Decision_Tree","Random_Forest","AdaBoost","XGBoost","LightGBM"])
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
    p.add_argument("--cpu", action="store_true", help="Force CPU")
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
    if args.cpu:                   CFG["use_gpu"] = False
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
    print(f"[INFO] Loaded dataset: N={N:,}, D={Xz.shape[1]} (note: source is sorted; we will shuffle mixed streams)")

    # Load defense & calibrator
    def_path = defense_root / "model.joblib"
    assert def_path.exists(), "Missing Defense-LGBM at models/Defense-LGBM/model.joblib"
    defense = joblib.load(def_path)
    calibrator = None
    cal_path = defense_root / "calibrator.joblib"
    if cal_path.exists():
        try:
            calibrator = joblib.load(cal_path)
            print("[INFO] Using isotonic calibrator for defense probabilities.")
        except Exception:
            calibrator = None

    # NAT p_adv and τ
    p_adv_nat_full = defense_p_adv(defense, X_raw, batch=CFG["eval_bs"], calibrator=calibrator)
    tau = choose_tau_from_nat(p_adv_nat_full, CFG["TAU_QUANTILE"])
    print(f"[INFO] τ (NAT quantile={CFG['TAU_QUANTILE']:.2f}) = {tau:.6f}")
    # plot NAT p_adv histogram once per run
    common_out = ensure_dir(results_root / "_common")
    plot_hist(p_adv_nat_full, common_out / "p_adv_nat_hist.png", "NAT p_adv distribution (full)")

    # Train a GLOBAL surrogate on full z-space labels (for non-linear bases)
    print("[INFO] Training global surrogate (pretrain) for black-box attacks...")
    global_surr = train_global_surrogate(
        Xz, y_base.astype(np.float32, copy=False), device,
        epochs=CFG["surrogate_epochs_global"], batch=CFG["surrogate_batch"], hidden=CFG["surrogate_hidden"]
    )
    cleanup_cuda()

    # Candidate bases
    candidates = [
        "Logistic_Regression",
        "Calibrated_LinearSVC",
        "Gaussian_Naive_Bayes",
        "Decision_Tree",
        "Random_Forest",
        "AdaBoost",
        "XGBoost",
        "LightGBM",
    ]
    if args.model != "all":
        candidates = [args.model]

    bases = []
    for name in candidates:
        mpath = models_root / name / "model.joblib"
        if mpath.exists():
            bases.append((name, joblib.load(mpath)))
        else:
            print(f"[WARN] Skipping {name}: {mpath} not found.")
    assert bases, "No base models found for the given selection."

    def nat_pure_eval(base_name: str, base_est):
        out_dir = ensure_dir(results_root / base_name / "NAT_pure")
        # Base-only on ALL NAT
        base_scores_all = chunked_scores(base_est, Xz, CFG["eval_bs"])
        base_probs_all = sigmoidify(base_scores_all)
        acc_all, auc_all = base_metrics(y_base, base_probs_all, out_dir, tag="all_nat", prefix="base")
        # After-defense on ACCEPTED NAT
        accept_nat = (p_adv_nat_full < tau)
        base_probs_acc = base_probs_all[accept_nat]
        y_acc = y_base[accept_nat]
        acc_acc, auc_acc = base_metrics(y_acc, base_probs_acc, out_dir, tag="accepted_nat", prefix="base")
        # detector CM on NAT only is degenerate; skip
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

    # Craft ADV helpers
    def craft_adv(base_name: str, base_est, which: str) -> np.ndarray:
        """Return Xz_adv for all rows (may be heavy)."""
        print(f"[{base_name}] crafting {which} on FULL dataset...")
        steps = CFG["steps_fgsm"] if which == "FGSM" else CFG["steps_pgd"]
        is_linear = base_name in {"Logistic_Regression","Calibrated_LinearSVC"}
        if is_linear:
            loss_kind = "hinge" if base_name == "Calibrated_LinearSVC" else "logistic"
            return attack_linear_fgsm_pgd(
                base_est, Xz, y_base,
                steps=steps, eps=CFG["eps_z"], alpha=CFG["alpha_z"],
                loss_kind=loss_kind, batch=CFG["lin_attack_bs"]
            )
        # black-box via surrogate
        surr = distill_from_global(
            global_surr, base_est, Xz,
            device=device, epochs=CFG["surrogate_epochs_finetune"],
            batch=CFG["surrogate_batch"], hidden=CFG["surrogate_hidden"],
            sample_limit=CFG["surrogate_sample_limit"]
        )
        Xz_adv = attack_surrogate_fgsm_pgd(
            surr, Xz, y_base,
            steps=steps, eps=CFG["eps_z"], alpha=CFG["alpha_z"],
            device=device, batch=CFG["bb_attack_bs"]
        )
        del surr; cleanup_cuda()
        return Xz_adv

    def mixed_eval(base_name: str, base_est, which: str):
        out_dir = ensure_dir(results_root / base_name / f"mixed_{which}")
        # craft ADV on full set
        Xz_adv_full = craft_adv(base_name, base_est, which)
        Xraw_adv_full = scaler.inverse_transform(Xz_adv_full)
        p_adv_adv_full = defense_p_adv(defense, Xraw_adv_full, batch=CFG["eval_bs"], calibrator=calibrator)
        plot_hist(p_adv_adv_full, out_dir / "p_adv_adv_hist.png", f"{which} p_adv distribution")
        del Xraw_adv_full; gc.collect()

        # Determine stream sizes
        if CFG["STREAM_SIZE"] is None:
            adv_idx = np.arange(N)                 # use all ADV
            nat_count = int(np.floor(N * CFG["NAT_FRAC"]))
            nat_idx = rng.choice(N, size=nat_count, replace=False)
        else:
            total = int(max(1, CFG["STREAM_SIZE"]))
            nat_count = int(np.clip(int(total * CFG["NAT_FRAC"]), 1, total-1))
            adv_count = total - nat_count
            adv_idx = rng.choice(N, size=adv_count, replace=False)
            nat_idx = rng.choice(N, size=nat_count, replace=False)

        # Build mixed stream arrays
        p_adv_nat = p_adv_nat_full[nat_idx]
        base_scores_nat = chunked_scores(base_est, Xz[nat_idx], CFG["eval_bs"])
        base_probs_nat = sigmoidify(base_scores_nat)

        p_adv_adv = p_adv_adv_full[adv_idx]
        base_scores_adv = chunked_scores(base_est, Xz_adv_full[adv_idx], CFG["eval_bs"])
        base_probs_adv = sigmoidify(base_scores_adv)

        y_is_adv = np.concatenate([np.zeros_like(p_adv_nat, dtype=int),
                                   np.ones_like(p_adv_adv, dtype=int)])
        p_adv_mixed = np.concatenate([p_adv_nat, p_adv_adv])
        base_probs_mixed = np.concatenate([base_probs_nat, base_probs_adv])
        # base_preds_mixed not needed for metrics except accepted NAT CM; we’ll recompute slice

        # Shuffle to interleave (dataset is sorted)
        perm = rng.permutation(p_adv_mixed.shape[0])
        y_is_adv = y_is_adv[perm]
        p_adv_mixed = p_adv_mixed[perm]
        base_probs_mixed = base_probs_mixed[perm]

        # Detector metrics
        det_auc = safe_auc(y_is_adv, p_adv_mixed)
        adv_mask = (y_is_adv == 1)
        nat_mask = (y_is_adv == 0)
        tpr_at_tau = float((p_adv_mixed[adv_mask] >= tau).mean()) if adv_mask.any() else float("nan")
        fpr_at_tau = float((p_adv_mixed[nat_mask] >= tau).mean()) if nat_mask.any() else float("nan")
        accept = (p_adv_mixed < tau).astype(int)
        acc_rate_overall = float(accept.mean())
        acc_rate_nat = float(accept[nat_mask].mean()) if nat_mask.any() else float("nan")
        acc_rate_adv = float(accept[adv_mask].mean()) if adv_mask.any() else float("nan")

        # Confusion matrix for detector at τ
        detector_confusion(
            y_is_adv, p_adv_mixed, tau,
            out_dir / "detector_confusion_tau.png",
            f"{base_name} [{which}] detector @ τ"
        )

        # Base metrics on ACCEPTED NAT in the mixed stream
        # We need the NAT slice after the permutation. Rebuild sequences aligned to perm:
        nat_probs_seq = np.concatenate([base_probs_nat, base_probs_adv])[perm]  # NAT probs are only at nat_mask
        # Mask accepted NAT
        mask_acc_nat = (accept == 1) & (y_is_adv == 0)
        # For those, get true base labels (need them from original y_base)
        # nat_idx permuted subset positions correspond to first len(nat_idx) before perm. Build a mapping:
        # Build aligned y_base for mixed stream:
        y_base_nat = y_base[nat_idx]
        y_base_adv = y_base[adv_idx]  # not used for NAT metrics
        y_base_mixed = np.concatenate([y_base_nat, y_base_adv])[perm]
        y_nat_acc = y_base_mixed[mask_acc_nat]
        base_probs_nat_acc = nat_probs_seq[mask_acc_nat]

        # Write base metrics for accepted NAT
        acc_acc, auc_acc = base_metrics(y_nat_acc, base_probs_nat_acc, out_dir, tag="accepted_nat", prefix="base")

        # Save metrics.json
        save_json({
            "scenario": f"mixed_{which}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "samples_total": int(p_adv_mixed.shape[0]),
            "nat_count": int(nat_mask.sum()),
            "adv_count": int(adv_mask.sum()),
            "tau": float(tau),
            "detector_auroc_nat_vs_adv": float(det_auc),
            "tpr_adv_at_tau": float(tpr_at_tau),
            "fpr_nat_at_tau": float(fpr_at_tau),
            "accept_rate_overall": float(acc_rate_overall),
            "accept_rate_nat": float(acc_rate_nat),
            "accept_rate_adv": float(acc_rate_adv),
            "base_acc_accepted_nat": float(acc_acc),
            "base_auc_accepted_nat": float(auc_acc),
        }, out_dir / "metrics.json")

        # Histograms for visibility
        plot_hist(p_adv_nat, out_dir / "p_adv_nat_hist.png", f"{base_name} NAT p_adv ({which} mix)")
        plot_hist(p_adv_adv, out_dir / "p_adv_adv_hist_small.png", f"{base_name} ADV p_adv ({which} mix)")

        # ROC of detector on mixed stream
        plot_roc(y_is_adv, p_adv_mixed, out_dir / "detector_roc_nat_vs_adv.png", f"{base_name} [{which}] detector ROC")

        del Xz_adv_full, p_adv_adv_full
        cleanup_cuda()

    # ---------- Run per base ----------
    for base_name, base_est in bases:
        print(f"\n==== Evaluating base: {base_name} ====")
        nat_pure_eval(base_name, base_est)
        mixed_eval(base_name, base_est, "FGSM")
        mixed_eval(base_name, base_est, "PGD")

    print("\n[DONE] Mixed-stream Defense-LGBM evaluation complete.")

if __name__ == "__main__":
    main()

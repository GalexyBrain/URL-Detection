#!/usr/bin/env python3
"""
Generate adversarial *tabular* dataset (features-only) for defense training.

Input:
  - features_extracted.csv   (has: url, <numeric features...>, label)
  - models/_global/scaler.joblib
  - models/_global/feature_columns.json
  - models/_global/split_indices.json (optional; used if present)

Models searched under:
  - models/<ModelName>/model.pt         for DL (PyTorch)
  - models/<ModelName>/model.joblib     for sklearn

Default 3 models used to craft attacks (editable in CFG):
  - "DL-MLP"                (white-box PGD/FGSM in z-space)
  - "Logistic_Regression"   (white-box linear FGSM/PGD)
  - "Random_Forest"         (black-box via distilled surrogate + PGD/FGSM)

Output (streamed):
  - features_adversarial_defense_dataset.csv
    Columns: [<feature_columns...>, orig_label, is_adv, attack_type, source_model, parent_idx]

Notes:
  - We *ignore the url string completely*; only numeric features are used.
  - Attacks are done in standardized z-space, then inverse-transformed and post-processed
    to respect feature domains (ints/ratios/binaries).
  - NAT rows are included for balance (same subset as attacked samples).
"""

from __future__ import annotations
from pathlib import Path
import os, json, math, gc, warnings, random
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ------------------------- CONFIG -------------------------
CFG = {
    "input_csv": "features_extracted.csv",
    "out_csv": "features_adversarial_defense_dataset.csv",

    "models_dir": "models",
    "global_dir": "models/_global",

    # Pick ANY 3 models present in your repo:
    # Choices seen in your pipeline: 
    #   DL-MLP, DL-FTTransformer, Logistic_Regression, Calibrated_LinearSVC,
    #   Decision_Tree, Random_Forest, LightGBM, XGBoost, AdaBoost, Gaussian_Naive_Bayes
    "source_models": ["DL-MLP", "Logistic_Regression", "Random_Forest"],

    # Attacks to generate
    "attacks": ["FGSM", "PGD"],

    # Use test split if available (split_indices.json); else falls back to random subset
    "use_test_split": True,

    # Subset caps (to keep runtime/memory sane)
    "max_attack_samples": 50000,       # None for full test set
    "random_state": 42,

    # Streaming / flushing
    "flush_every_rows": 200_000,

    # Batching
    "tab_attack_bs": 4096,
    "tab_eval_bs": 8192,

    # Device
    "use_gpu": True,
}

# Attack knobs (z-space)
ATTACK = {
    # DL white-box (z-space PGD/FGSM)
    "eps_tab_fgsm": 0.5,
    "eps_tab_pgd": 0.5,
    "alpha_tab_pgd": 0.05,
    "steps_tab_pgd": 10,

    # Linear white-box
    "eps_lin_fgsm": 0.5,
    "eps_lin_pgd": 0.5,
    "alpha_lin_pgd": 0.05,
    "steps_lin_pgd": 10,
}

# ------------------------- IO helpers -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def append_rows(out_path: Path, rows: list[dict], header_written: bool, feature_cols: list[str]):
    """Stream-append rows to CSV."""
    if not rows:
        return header_written
    cols = feature_cols + ["orig_label", "is_adv", "attack_type", "source_model", "parent_idx"]
    df = pd.DataFrame(rows, columns=cols)
    mode = "a" if (out_path.exists() or header_written) else "w"
    df.to_csv(out_path, mode=mode, header=(mode=="w"), index=False, encoding="utf-8")
    return True

# ------------------------- Data & schema -------------------------
def map_label(v):
    if isinstance(v, (int, np.integer)): 
        return int(v)
    s = str(v).strip().lower()
    return 1 if s in {"1","true","malicious","phishing","malware","bad"} else 0

def load_tabular_scaled(df: pd.DataFrame, global_dir: Path):
    """Load scaler + feature schema; build X (raw) & Xz (scaled) strictly in saved order."""
    scaler_path = global_dir / "scaler.joblib"
    feat_path   = global_dir / "feature_columns.json"
    assert scaler_path.exists(), f"Missing scaler at {scaler_path}"
    assert feat_path.exists(),   f"Missing schema at {feat_path}"

    scaler: StandardScaler = joblib.load(scaler_path)
    saved = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_cols = list(saved.get("feature_columns", []))
    assert feature_cols, "feature_columns.json has empty feature_columns"

    # Construct X in saved order
    X_df = pd.DataFrame(index=df.index)
    for col in feature_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        # fill missing with training mean to keep z near 0
        mean_val = float(getattr(scaler, "mean_", np.zeros(len(feature_cols)))[feature_cols.index(col)])
        X_df[col] = s.fillna(mean_val).astype(np.float32)

    X = X_df[feature_cols].to_numpy(dtype=np.float32)
    Xz = scaler.transform(X).astype(np.float32)
    # Empirical bounds (used to clamp PGD box)
    z_min = Xz.min(axis=0)
    z_max = Xz.max(axis=0)
    return X, Xz, feature_cols, scaler, (z_min, z_max)

def infer_feature_domains(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Infer simple domain constraints from names and observed values on the *original* dataset:
      - binary: values in {0,1}
      - ratio:  name contains 'ratio' or 'fraction' -> [0,1]
      - intish: name contains 'count' or 'length' or startswith 'num_' -> integer >=0
      - nonneg: clip to >=0 for typical counts/lengths
    """
    domains = {"binary": set(), "ratio": set(), "intish": set(), "nonneg": set()}
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        uniq = set(pd.unique(s.astype(np.float32)))
        lowc = c.lower()
        if uniq.issubset({0.0, 1.0}):
            domains["binary"].add(c)
        if "ratio" in lowc or "fraction" in lowc:
            domains["ratio"].add(c)
        if any(k in lowc for k in ["count", "length"]) or lowc.startswith("num_"):
            domains["intish"].add(c); domains["nonneg"].add(c)
        # heuristic: obvious nonneg names
        if any(k in lowc for k in ["count", "length", "digit", "underscore", "dash", "dot", "semicolon",
                                   "question_mark", "hash_char", "equal", "percent_char", "ampersand",
                                   "at_char", "tilde_char", "double_slash"]):
            domains["nonneg"].add(c)
    # some typical binary flags by name if not caught
    typical_flags = {"ip_as_hostname","exe_in_url","https_in_url","ftp_used","js_used","css_used"}
    for f in typical_flags.intersection(feature_cols):
        domains["binary"].add(f)
    return domains

def enforce_domains(X_raw: np.ndarray, feature_cols: list[str], domains: dict) -> np.ndarray:
    """Apply domain constraints in-place; return X_raw."""
    col_idx = {c:i for i,c in enumerate(feature_cols)}
    # Ratios clipped to [0,1]
    for c in domains["ratio"]:
        i = col_idx[c]; X_raw[:, i] = np.clip(X_raw[:, i], 0.0, 1.0)
    # Non-negative
    for c in domains["nonneg"]:
        i = col_idx[c]; X_raw[:, i] = np.maximum(X_raw[:, i], 0.0)
    # Int-like counts/lengths -> round
    for c in domains["intish"]:
        i = col_idx[c]; X_raw[:, i] = np.rint(X_raw[:, i])
    # Binary flags -> hard round to {0,1}
    for c in domains["binary"]:
        i = col_idx[c]; X_raw[:, i] = (X_raw[:, i] >= 0.5).astype(np.float32)
    return X_raw

# ------------------------- Models -------------------------
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

# ------------------------- Attacks -------------------------
def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def attack_tabular_torch_batched(model, Xz, y, eps, alpha, steps, device, z_min, z_max, batch_size):
    """PGD/FGSM in z-space against a torch model."""
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

            for _ in range(steps_eff):
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
            if "out of memory" in str(e).lower() and bs > 1 and torch.cuda.is_available():
                cleanup_cuda()
                bs = max(1, bs // 2)
                print(f"[WARN] OOM during tab attack; retrying with batch_size={bs}")
                continue
            raise
    return np.concatenate(adv_list, axis=0)

def linear_wb_attack(model, Xz, y, eps, alpha, steps):
    """White-box FGSM/PGD for linear sklearn models in z-space."""
    w, b = linear_wb(model)
    X0 = Xz.copy()
    X_adv = Xz.copy()
    steps_eff = 1 if steps == 0 else steps
    alpha_eff = eps if steps == 0 else alpha
    y = y.astype(np.float32)
    # Logistic vs margin only changes sign; the following works as untargeted push
    for _ in range(steps_eff):
        margin = X_adv @ w + b
        # gradient for logistic is (p - y)*w; for hinge-like it's sign aligned; use logistic-ish:
        p = 1.0 / (1.0 + np.exp(-margin))
        grad = (p - y)[:, None] * w[None, :]
        X_adv = X_adv + alpha_eff * np.sign(grad)
        X_adv = np.clip(X_adv, X0 - eps, X0 + eps)
    return X_adv

def distill_surrogate(global_template: SurrogateMLP, teacher, Xz_tr, y_tr, device, tag, epochs=6, lr=3e-4, wd=1e-5, num_workers=2):
    """Distill any sklearn teacher into a torch MLP for gradient-based attacks."""
    if hasattr(teacher, "predict_proba"):
        soft = teacher.predict_proba(Xz_tr)[:, 1].astype(np.float32)
    elif hasattr(teacher, "decision_function"):
        df = teacher.decision_function(Xz_tr).astype(np.float32)
        soft = 1.0 / (1.0 + np.exp(-df))
    else:
        soft = teacher.predict(Xz_tr).astype(np.float32)

    model = SurrogateMLP(in_dim=Xz_tr.shape[1])
    model.load_state_dict(global_template.state_dict())
    model.to(device).train()

    ds = TensorDataset(torch.from_numpy(Xz_tr).float(), torch.from_numpy(soft).float())
    dl = DataLoader(ds, batch_size=8192, shuffle=True, num_workers=num_workers, pin_memory=(device.type=="cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_bce = nn.BCEWithLogitsLoss()

    best_state = None; best_loss = float("inf"); patience=2; noimp=0
    for ep in range(1, epochs+1):
        run=0.0; n=0
        for xb, sb in dl:
            xb = xb.to(device); sb = sb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_bce(logits, sb)
            loss.backward(); opt.step()
            run += loss.item()*xb.size(0); n += xb.size(0)
        avg = run/max(1,n)
        print(f"[Distill {tag}] epoch {ep}/{epochs} loss={avg:.6f}")
        if avg < best_loss - 1e-6:
            best_loss = avg
            best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
            if noimp >= patience: break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
    model.load_state_dict({k: v.to(device) for k,v in best_state.items()})
    model.eval()
    return model

# ------------------------- Main pipeline -------------------------
def main():
    rng = np.random.default_rng(CFG["random_state"])
    random.seed(CFG["random_state"])

    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Device: {device}")

    models_root = Path(CFG["models_dir"])
    global_dir = Path(CFG["global_dir"])
    out_path = Path(CFG["out_csv"])
    if out_path.exists():
        out_path.unlink()

    # Load dataset
    df = pd.read_csv(CFG["input_csv"])
    assert "label" in df.columns, "features_extracted.csv must contain 'label'"
    df["orig_label"] = df["label"].apply(map_label).astype(np.int64)

    # Build X (raw), Xz (scaled) in training order
    X_raw_full, Xz_full, feature_cols, scaler, (z_min, z_max) = load_tabular_scaled(df, global_dir)

    # Choose indices: test split preferred
    split_path = global_dir / "split_indices.json"
    if CFG["use_test_split"] and split_path.exists():
        info = json.loads(split_path.read_text(encoding="utf-8"))
        idx = np.array(info["test_idx"], dtype=np.int64)
        print(f"[INFO] Using test split indices: {len(idx)}")
    else:
        # stratified holdout ~20% if split missing
        y_all = df["orig_label"].to_numpy()
        _, idx, _, _ = train_test_split(
            np.arange(len(df)), y_all, test_size=0.20,
            random_state=CFG["random_state"], stratify=y_all
        )
        print(f"[WARN] split_indices.json missing; using a fresh 20% holdout ({len(idx)})")

    # Optional cap
    if CFG["max_attack_samples"] and len(idx) > CFG["max_attack_samples"]:
        idx = np.sort(rng.choice(idx, size=CFG["max_attack_samples"], replace=False))
        print(f"[INFO] Capped attack samples to {len(idx)}")

    Xz = Xz_full[idx]
    X_nat_raw = X_raw_full[idx]
    y = df["orig_label"].to_numpy()[idx]

    # NAT rows (balance)
    print("[PHASE] Writing NAT rows...")
    header_written = False
    rows = []
    for j, row_idx in enumerate(idx):
        row_feats = {c: X_nat_raw[j, k] for k, c in enumerate(feature_cols)}
        rows.append({**row_feats,
                     "orig_label": int(y[j]),
                     "is_adv": 0,
                     "attack_type": "NAT",
                     "source_model": "N/A",
                     "parent_idx": int(row_idx)})
        if len(rows) >= CFG["flush_every_rows"]:
            header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()
    header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()

    # Global surrogate template (for black-box teachers)
    global_surr = SurrogateMLP(in_dim=Xz.shape[1]).to(device)
    global_surr.load_state_dict(SurrogateMLP(in_dim=Xz.shape[1]).state_dict())

    # Attack over selected source models
    for name in CFG["source_models"]:
        mdir = models_root / name
        is_torch = (mdir / "model.pt").exists()
        is_sklearn = (mdir / "model.joblib").exists()
        if not (is_torch or is_sklearn):
            print(f"[WARN] Skipping {name}: no model found at {mdir}")
            continue

        print(f"\n[PHASE] Attacking model: {name}")

        # Load model
        if is_torch:
            if name == "DL-MLP":
                mdl = MLPNet(in_dim=Xz.shape[1]).to(device)
            else:
                # If you choose DL-FTTransformer in CFG, add loader here similarly.
                mdl = MLPNet(in_dim=Xz.shape[1]).to(device)
            state = torch.load(mdir / "model.pt", map_location=device)
            mdl.load_state_dict(state)
            mdl.eval()
        else:
            mdl = joblib.load(mdir / "model.joblib")

        # Prepare attacker by model type
        is_linear = name in {"Logistic_Regression", "Calibrated_LinearSVC"}
        is_blackbox = name in {"Decision_Tree","Random_Forest","LightGBM","XGBoost","AdaBoost","Gaussian_Naive_Bayes"}

        # FGSM
        if "FGSM" in CFG["attacks"]:
            if is_torch:
                X_adv_z = attack_tabular_torch_batched(
                    mdl, Xz, y,
                    eps=ATTACK["eps_tab_fgsm"], alpha=ATTACK["eps_tab_fgsm"], steps=0,
                    device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
                )
            elif is_linear:
                X_adv_z = linear_wb_attack(
                    mdl, Xz, y,
                    eps=ATTACK["eps_lin_fgsm"], alpha=ATTACK["eps_lin_fgsm"], steps=0
                )
            elif is_blackbox:
                # distill teacher -> surrogate -> attack surrogate
                print(f"[INFO] Distilling surrogate for {name} (FGSM)...")
                surr = distill_surrogate(global_surr, mdl, Xz, y, device, tag=f"{name}_FGSM")
                X_adv_z = attack_tabular_torch_batched(
                    surr, Xz, y,
                    eps=ATTACK["eps_tab_fgsm"], alpha=ATTACK["eps_tab_fgsm"], steps=0,
                    device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
                )
            else:
                print(f"[WARN] Unknown model type for {name}; skipping FGSM")
                X_adv_z = None

            if X_adv_z is not None:
                # inverse-transform to raw feature space and enforce domains
                X_adv_raw = joblib.load(global_dir / "scaler.joblib").inverse_transform(X_adv_z)
                domains = infer_feature_domains(pd.read_csv(CFG["input_csv"]), feature_cols)
                X_adv_raw = enforce_domains(X_adv_raw, feature_cols, domains)

                # stream write
                rows = []
                for j, row_idx in enumerate(idx):
                    feats = {c: X_adv_raw[j, k] for k, c in enumerate(feature_cols)}
                    rows.append({**feats,
                                 "orig_label": int(y[j]),
                                 "is_adv": 1,
                                 "attack_type": "FGSM",
                                 "source_model": name,
                                 "parent_idx": int(row_idx)})
                    if len(rows) >= CFG["flush_every_rows"]:
                        header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()
                header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()
                del X_adv_z, X_adv_raw
                cleanup_cuda()

        # PGD
        if "PGD" in CFG["attacks"]:
            if is_torch:
                X_adv_z = attack_tabular_torch_batched(
                    mdl, Xz, y,
                    eps=ATTACK["eps_tab_pgd"], alpha=ATTACK["alpha_tab_pgd"], steps=ATTACK["steps_lin_pgd"],
                    device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
                )
            elif is_linear:
                X_adv_z = linear_wb_attack(
                    mdl, Xz, y,
                    eps=ATTACK["eps_lin_pgd"], alpha=ATTACK["alpha_lin_pgd"], steps=ATTACK["steps_lin_pgd"]
                )
            elif is_blackbox:
                print(f"[INFO] Distilling surrogate for {name} (PGD)...")
                surr = distill_surrogate(global_surr, mdl, Xz, y, device, tag=f"{name}_PGD")
                X_adv_z = attack_tabular_torch_batched(
                    surr, Xz, y,
                    eps=ATTACK["eps_tab_pgd"], alpha=ATTACK["alpha_tab_pgd"], steps=ATTACK["steps_tab_pgd"],
                    device=device, z_min=z_min, z_max=z_max, batch_size=CFG["tab_attack_bs"]
                )
            else:
                print(f"[WARN] Unknown model type for {name}; skipping PGD")
                X_adv_z = None

            if X_adv_z is not None:
                X_adv_raw = joblib.load(global_dir / "scaler.joblib").inverse_transform(X_adv_z)
                domains = infer_feature_domains(pd.read_csv(CFG["input_csv"]), feature_cols)
                X_adv_raw = enforce_domains(X_adv_raw, feature_cols, domains)

                rows = []
                for j, row_idx in enumerate(idx):
                    feats = {c: X_adv_raw[j, k] for k, c in enumerate(feature_cols)}
                    rows.append({**feats,
                                 "orig_label": int(y[j]),
                                 "is_adv": 1,
                                 "attack_type": "PGD",
                                 "source_model": name,
                                 "parent_idx": int(row_idx)})
                    if len(rows) >= CFG["flush_every_rows"]:
                        header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()
                header_written = append_rows(out_path, rows, header_written, feature_cols); rows.clear()
                del X_adv_z, X_adv_raw
                cleanup_cuda()

    # ---- Final stats
    print("\n[STATS] Wrote:", out_path)
    try:
        tmp = pd.read_csv(out_path, usecols=["attack_type","source_model","is_adv"])
        print(tmp["attack_type"].value_counts(dropna=False).to_string())
    except Exception:
        pass
    print("[OK] Done.")

if __name__ == "__main__":
    main()

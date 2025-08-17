#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mixed-stream defense evaluation for URL character models (DL-CharCNN / DL-CharTransformer)
using a CharCNN detector.

What this does (per base URL model)
-----------------------------------
1) Loads tokenizer + full dataset (features_extracted.csv -> URL bytes -> token IDs).
2) Loads base URL model (DL-CharCNN or DL-CharTransformer) if present.
3) Loads Defense-CharCNN detector; computes NAT p_adv over the WHOLE dataset and
   chooses τ as a quantile (TAU_QUANTILE) of NAT p_adv.
4) Evaluates:
   A) NAT_pure:
      - Base-only metrics on ALL NAT (before defense).
      - Base metrics on ACCEPTED NAT (after defense).
      - Confusion matrices + ROC for both.
   B) mixed_FGSM and mixed_PGD:
      - Craft HotFlip adversaries on the FULL dataset (FGSM/PGD).
      - Build a mixed stream with NAT_FRAC of clean + the adversarial subset; shuffle to interleave.
      - Detector: AUROC (NAT vs ADV), TPR@τ on ADV, FPR@τ on NAT, confusion @ τ, ROC.
      - Base metrics on accepted/rejected NAT/ADV, plus tons of extra breakdowns.

Outputs
-------
results_defense_char_stream/<Base>/
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
    metrics.json  (rich)
    detector_confusion_tau.png
    detector_roc_nat_vs_adv.png
    base_confusion_accepted_nat.png
    base_roc_accepted_nat.png
    classification_report_accepted_nat.txt
    p_adv_nat_hist.png
    p_adv_adv_hist.png

Notes
-----
- Dataset is sorted by label in source; we shuffle mixed streams explicitly.
- All heavy ops run in batches with OOM backoff.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, gc, warnings, argparse
import numpy as np
import pandas as pd

# Plots (offscreen)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -------------------- Config --------------------
CFG = {
    "models_dir": "models",
    "defense_dir": "models/Defense-CharCNN",
    "results_dir": "results_defense_char_stream",
    "dataset_csv": "features_extracted.csv",
    "label_col": "label",
    "url_col": "url",

    # HotFlip attack knobs
    "char_topk_per_step": 1,
    "char_steps_fgsm": 1,   # 1 step => FGSM-like
    "char_steps_pgd": 5,    # multi-step PGD
    "freeze_cls_pos0": True,

    # Batch sizes (with OOM backoff)
    "eval_bs": 2048,
    "attack_bs": 1024,

    # Device
    "use_gpu": True,

    # NAT threshold selection (q-quantile on NAT p_adv)
    "TAU_QUANTILE": 0.90,

    # Mixed-stream composition
    "NAT_FRAC": 0.5,        # fraction of NAT in the mixed stream
    "STREAM_SIZE": None,    # None => use all ADV; else total mixed size

    # RNG
    "seed": 42,
}

PAD_ID = 256
CLS_ID = 257
VOCAB_SIZE = 258
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

# -------------------- Tokenizer & encoding --------------------
def read_tokenizer_info(global_dir: Path):
    tok_path = global_dir / "url_tokenizer.json"
    assert tok_path.exists(), "Missing models/_global/url_tokenizer.json"
    info = json.loads(tok_path.read_text(encoding="utf-8"))
    use_cls = bool(info.get("use_cls_token", True))
    max_len = int(info.get("max_len", 256))
    return max_len, use_cls

def encode_url(u: str, max_len: int, use_cls=True):
    b = str(u).encode("utf-8", "ignore")
    if use_cls:
        b = b[:max_len-1]; ids = [CLS_ID] + [x for x in b]
    else:
        b = b[:max_len];    ids = [x for x in b]
    if len(ids) < max_len: ids += [PAD_ID] * (max_len - len(ids))
    else:                  ids = ids[:max_len]
    ids = [i if i >= 256 else max(0, min(255, i)) for i in ids]
    return np.array(ids, dtype=np.int64)

def build_text_ids_from_df(df: pd.DataFrame, cfg: dict, global_dir: Path):
    max_len, use_cls = read_tokenizer_info(global_dir)
    urls = df[cfg["url_col"]].astype(str).tolist()
    y = pd.to_numeric(df[cfg["label_col"]], errors="coerce").fillna(0).astype(int).to_numpy()
    X_ids = np.stack([encode_url(u, max_len, use_cls) for u in urls])
    return X_ids, y, max_len, use_cls

# -------------------- URL models --------------------
class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, p_drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.conv1 = nn.Conv1d(d_model, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn1, self.bn2 = nn.BatchNorm1d(128), nn.BatchNorm1d(256)
        self.act, self.drop = nn.GELU(), nn.Dropout(p_drop)
        self.head = nn.Sequential(nn.Linear(256,128), nn.GELU(), nn.Dropout(p_drop), nn.Linear(128,1))
    def forward(self, ids):
        x = self.emb(ids).transpose(1,2)
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
    def forward(self, ids):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        z = self.emb(ids) + self.pos(pos)
        z = self.enc(z)
        z = self.norm(z)
        pooled = z[:,0,:] if (ids[:,0]==CLS_ID).all() else z.mean(dim=1)
        return self.head(pooled).squeeze(-1)

# -------------------- Eval / detector --------------------
@torch.no_grad()
def eval_url_batched(model, ids_np, device, batch_size):
    N = ids_np.shape[0]
    probs_list = []; bs = max(1, int(batch_size)); i = 0
    model.eval()
    while i < N:
        j = min(N, i + bs)
        try:
            ids = torch.from_numpy(ids_np[i:j]).to(device).long()
            logits = model(ids).float()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_list.append(probs); i = j
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1 and torch.cuda.is_available():
                cleanup_cuda(); bs = max(1, bs // 2)
                print(f"[WARN] OOM during eval; retry with batch_size={bs}")
                continue
            raise
    probs = np.concatenate(probs_list, axis=0)
    return probs  # probabilities for class 1

@torch.no_grad()
def defense_p_adv_batched(def_mdl, ids_np, device, batch_size):
    return eval_url_batched(def_mdl, ids_np, device, batch_size).reshape(-1)

# -------------------- HotFlip attacks --------------------
def hotflip_step(ids, y, model, emb_module, freeze_pos0=True, topk=1, device="cuda"):
    model.eval()
    ids = ids.clone().to(device).long()
    y_t = y.to(device).float()

    loss_fn = nn.BCEWithLogitsLoss()
    model.zero_grad(set_to_none=True)

    if isinstance(model, CharCNN):
        E = model.emb(ids); E.requires_grad_(True); E.retain_grad()
        x = E.transpose(1, 2)
        x = model.act(model.bn1(model.conv1(x)))
        x = model.act(model.bn2(model.conv2(x)))
        x = torch.amax(x, dim=2)
        logits = model.head(x).squeeze(-1)
    elif isinstance(model, CharTransformer):
        B, L = ids.shape
        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        E = model.emb(ids); E.requires_grad_(True); E.retain_grad()
        z = E + model.pos(pos_idx)
        z = model.enc(z)
        z = model.norm(z)
        pooled = z[:, 0, :] if (ids[:, 0] == CLS_ID).all() else z.mean(dim=1)
        logits = model.head(pooled).squeeze(-1)
    else:
        raise RuntimeError("Unsupported URL model")

    loss = loss_fn(logits, y_t)
    loss.backward()
    G = E.grad.detach()

    W = emb_module.weight.detach()
    B, L, D = G.shape
    ids_new = ids.clone()

    pos_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    if freeze_pos0: pos_mask[:, 0] = False
    pos_mask &= (ids != PAD_ID)

    curW = W[ids]
    scores = torch.einsum("vd,bld->blv", W, G) - torch.einsum("bld,bld->bl", curW, G)[..., None]
    scores.scatter_(2, ids.unsqueeze(-1), float("-inf"))
    scores[:, :, PAD_ID] = float("-inf")

    gains, cand = scores.max(dim=2)
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
    N = ids_np.shape[0]
    out = np.empty_like(ids_np)
    bs = max(1, int(batch_size)); i = 0
    while i < N:
        j = min(N, i + bs)
        try:
            ids = torch.from_numpy(ids_np[i:j]).long().to(device)
            y   = torch.from_numpy(y_np[i:j]).long().to(device)
            ids_adv = ids.clone()
            for _ in range(max(1, int(steps))):
                ids_adv = hotflip_step(ids_adv, y, model, model.emb,
                                       freeze_pos0=freeze_pos0, topk=topk, device=device)
            out[i:j] = ids_adv.detach().cpu().numpy()
            i = j
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1 and torch.cuda.is_available():
                cleanup_cuda(); bs = max(1, bs // 2)
                print(f"[WARN] OOM during HotFlip; retrying with batch_size={bs}")
                continue
            raise
    model.eval()
    return out

# -------------------- Metrics helpers --------------------
def safe_auc(y_true, scores) -> float:
    try:
        y_true = np.asarray(y_true); scores = np.asarray(scores)
        if np.unique(y_true).size < 2: return float("nan")
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float("nan")

def base_metrics(y_true, probs, out_dir: Path, tag: str, prefix: str):
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

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Mixed-stream CharCNN defense with HotFlip FGSM/PGD.")
    p.add_argument("--model", default="all", choices=["DL-CharCNN","DL-CharTransformer","all"])
    p.add_argument("--tau-q", type=float, default=None, help="Quantile q for NAT p_adv (default 0.90)")
    p.add_argument("--nat-frac", type=float, default=None, help="NAT fraction in mixed stream (default 0.5)")
    p.add_argument("--stream-size", type=int, default=None, help="Total mixed-stream size; None=use all ADV (N)")
    p.add_argument("--pgd-steps", type=int, default=None)
    p.add_argument("--fgsm-steps", type=int, default=None)
    p.add_argument("--topk", type=int, default=None, help="Characters to flip per step (default 1)")
    p.add_argument("--eval-bs", type=int, default=None)
    p.add_argument("--attack-bs", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

# -------------------- Main --------------------
def main():
    args = parse_args()
    if args.tau_q is not None:     CFG["TAU_QUANTILE"] = float(args.tau_q)
    if args.nat_frac is not None:  CFG["NAT_FRAC"] = float(args.nat_frac)
    if args.stream_size is not None: CFG["STREAM_SIZE"] = int(args.stream_size)
    if args.pgd_steps is not None: CFG["char_steps_pgd"] = int(args.pgd_steps)
    if args.fgsm_steps is not None: CFG["char_steps_fgsm"] = int(args.fgsm_steps)
    if args.topk is not None:      CFG["char_topk_per_step"] = int(args.topk)
    if args.eval_bs is not None:   CFG["eval_bs"] = int(args.eval_bs)
    if args.attack_bs is not None: CFG["attack_bs"] = int(args.attack_bs)
    if args.cpu:                   CFG["use_gpu"] = False
    if args.seed is not None:      CFG["seed"] = int(args.seed)

    rng = np.random.RandomState(CFG["seed"])
    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    models_root = Path(CFG["models_dir"])
    defense_root = Path(CFG["defense_dir"])
    results_root = ensure_dir(Path(CFG["results_dir"]))
    global_dir = models_root / "_global"

    # Load dataset and build token ids (WHOLE set)
    df = pd.read_csv(CFG["dataset_csv"], usecols=[CFG["url_col"], CFG["label_col"]])
    df[CFG["label_col"]] = pd.to_numeric(df[CFG["label_col"]], errors="coerce").fillna(0).astype(int)
    Xids_all, y_all, max_len, use_cls = build_text_ids_from_df(df, CFG, global_dir)
    N = Xids_all.shape[0]
    print(f"[INFO] Loaded dataset: N={N:,}, L={Xids_all.shape[1]} (source sorted; mixed streams are shuffled)")

    # Load base URL models if present and match selection
    requested_models = []
    if args.model in ("DL-CharCNN", "all"):
        if (models_root / "DL-CharCNN" / "model.pt").exists():
            m = CharCNN().to(device)
            m.load_state_dict(torch.load(models_root/"DL-CharCNN"/"model.pt", map_location=device))
            m.eval(); requested_models.append(("DL-CharCNN", m))
        else:
            print("[WARN] Skipping DL-CharCNN: model.pt not found.")
    if args.model in ("DL-CharTransformer", "all"):
        if (models_root / "DL-CharTransformer" / "model.pt").exists():
            m = CharTransformer(max_len=max_len).to(device)
            m.load_state_dict(torch.load(models_root/"DL-CharTransformer"/"model.pt", map_location=device))
            m.eval(); requested_models.append(("DL-CharTransformer", m))
        else:
            print("[WARN] Skipping DL-CharTransformer: model.pt not found.")

    assert requested_models, "No base URL models found for the given --model selection."

    # Load defense model (CharCNN)
    assert (defense_root / "model.pt").exists(), "Missing Defense-CharCNN model.pt"
    def_mdl = CharCNN().to(device).eval()
    def_mdl.load_state_dict(torch.load(defense_root/"model.pt", map_location=device))

    # NAT p_adv over full set and choose τ
    p_adv_nat_full = defense_p_adv_batched(def_mdl, Xids_all, device, CFG["eval_bs"])
    tau = float(np.quantile(p_adv_nat_full, CFG["TAU_QUANTILE"])) if len(p_adv_nat_full) else 0.5
    print(f"[INFO] τ (NAT quantile={CFG['TAU_QUANTILE']:.2f}) = {tau:.6f}")
    common_out = ensure_dir(results_root / "_common")
    plot_hist(p_adv_nat_full, common_out / "p_adv_nat_hist.png", "NAT p_adv distribution (full)")

    # ---- per base model ----
    for name, base_mdl in requested_models:
        print(f"\n==== Evaluating base: {name} ====")
        out_root = ensure_dir(results_root / name)

        # ---------- NAT_pure ----------
        out_nat = ensure_dir(out_root / "NAT_pure")
        base_probs_all = eval_url_batched(base_mdl, Xids_all, device, CFG["eval_bs"])
        # base metrics on ALL NAT
        acc_all, auc_all = base_metrics(y_all, base_probs_all, out_nat, tag="all_nat", prefix="base")
        # after-defense on ACCEPTED NAT
        accept_nat = (p_adv_nat_full < tau)
        base_probs_acc = base_probs_all[accept_nat]
        y_acc = y_all[accept_nat]
        acc_acc, auc_acc = base_metrics(y_acc, base_probs_acc, out_nat, tag="accepted_nat", prefix="base")
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
        }, out_nat / "metrics.json")

        # ---------- helper: craft ADV on full set ----------
        def craft_adv(which: str):
            steps = CFG["char_steps_fgsm"] if which == "FGSM" else CFG["char_steps_pgd"]
            print(f"[{name}] crafting {which} (steps={steps}, topk={CFG['char_topk_per_step']}) on FULL dataset...")
            return attack_url_hotflip_batched(
                base_mdl, Xids_all, y_all,
                steps=steps, topk=CFG["char_topk_per_step"],
                device=device, freeze_pos0=CFG["freeze_cls_pos0"], batch_size=CFG["attack_bs"]
            )

        # ---------- common mixed eval (RICH METRICS) ----------
        def mixed_eval(which: str):
            out_mix = ensure_dir(out_root / f"mixed_{which}")
            ids_adv_full = craft_adv(which)
            # detector scores on full ADV
            p_adv_adv_full = defense_p_adv_batched(def_mdl, ids_adv_full, device, CFG["eval_bs"])
            plot_hist(p_adv_adv_full, out_mix / "p_adv_adv_hist.png", f"{which} p_adv distribution")

            # stream sizes
            if CFG["STREAM_SIZE"] is None:
                adv_idx = np.arange(N)  # use all ADV
                nat_count = int(np.floor(N * CFG["NAT_FRAC"]))
                nat_idx = rng.choice(N, size=nat_count, replace=False)
            else:
                total = int(max(1, CFG["STREAM_SIZE"]))
                nat_count = int(np.clip(int(total * CFG["NAT_FRAC"]), 1, total-1))
                adv_count = total - nat_count
                adv_idx = rng.choice(N, size=adv_count, replace=False)
                nat_idx = rng.choice(N, size=nat_count, replace=False)

            # base probs on slices (before mixing)
            base_probs_nat = eval_url_batched(base_mdl, Xids_all[nat_idx], device, CFG["eval_bs"])
            base_probs_adv = eval_url_batched(base_mdl, ids_adv_full[adv_idx], device, CFG["eval_bs"])

            # labels for slices
            y_base_nat = y_all[nat_idx]
            y_base_adv = y_all[adv_idx]

            # detector probabilities for slices
            p_adv_nat = p_adv_nat_full[nat_idx]
            p_adv_adv = p_adv_adv_full[adv_idx]

            # build mixed arrays and shuffle
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
            det_ap = _safe_ap(y_is_adv, p_adv_mixed)  # PR-AUC (ADV is positive)
            pred_adv = (p_adv_mixed >= tau)
            adv_mask = (y_is_adv == 1)
            nat_mask = (y_is_adv == 0)

            TP = int(np.sum(pred_adv & adv_mask))   # correctly blocked ADV
            FP = int(np.sum(pred_adv & nat_mask))   # mistakenly blocked NAT
            TN = int(np.sum(~pred_adv & nat_mask))  # accepted NAT
            FN = int(np.sum(~pred_adv & adv_mask))  # accepted ADV (got through)
            tpr_at_tau = float(TP / (TP + FN)) if (TP + FN) > 0 else float("nan")
            fpr_at_tau = float(FP / (FP + TN)) if (FP + TN) > 0 else float("nan")
            tnr_at_tau = 1.0 - fpr_at_tau if np.isfinite(fpr_at_tau) else float("nan")
            prec_at_tau = float(TP / (TP + FP)) if (TP + FP) > 0 else float("nan")
            f1_at_tau = float(2 * prec_at_tau * tpr_at_tau / (prec_at_tau + tpr_at_tau)) if np.isfinite(prec_at_tau) and np.isfinite(tpr_at_tau) and (prec_at_tau + tpr_at_tau) > 0 else float("nan")
            bal_acc_at_tau = float((tpr_at_tau + tnr_at_tau) / 2) if np.isfinite(tpr_at_tau) and np.isfinite(tnr_at_tau) else float("nan")

            accept = (~pred_adv).astype(int)
            acc_rate_overall = float(accept.mean())
            acc_rate_nat = float((~pred_adv & nat_mask).mean()) if nat_mask.any() else float("nan")
            acc_rate_adv = float((~pred_adv & adv_mask).mean()) if adv_mask.any() else float("nan")

            # Detector visuals
            detector_confusion(
                y_is_adv, p_adv_mixed, tau,
                out_mix / "detector_confusion_tau.png",
                f"{name} [{which}] detector @ τ"
            )
            plot_roc(y_is_adv, p_adv_mixed, out_mix / "detector_roc_nat_vs_adv.png", f"{name} [{which}] detector ROC")

            # -------- Base metrics BEFORE defense (slice-level)
            preds_nat_slice = (base_probs_nat >= 0.5).astype(int)
            acc_nat_slice = float(accuracy_score(y_base_nat, preds_nat_slice)) if y_base_nat.size else float("nan")
            auc_nat_slice = safe_auc(y_base_nat, base_probs_nat)

            preds_adv_slice = (base_probs_adv >= 0.5).astype(int)
            acc_adv_slice = float(accuracy_score(y_base_adv, preds_adv_slice)) if y_base_adv.size else float("nan")
            auc_adv_slice = safe_auc(y_base_adv, base_probs_adv)

            # -------- Base metrics AFTER defense (accepted vs rejected, NAT vs ADV)
            mask_acc_nat = (~pred_adv & nat_mask)
            mask_rej_nat = (pred_adv & nat_mask)
            mask_acc_adv = (~pred_adv & adv_mask)
            mask_rej_adv = (pred_adv & adv_mask)

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
                # scenario & run context
                "scenario": f"mixed_{which}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "base_model": name,
                "defense_model": "Defense-CharCNN",
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
                "attack_steps": int(CFG["char_steps_fgsm"] if which == "FGSM" else CFG["char_steps_pgd"]),
                "attack_topk": int(CFG["char_topk_per_step"]),
                "freeze_cls_pos0": bool(CFG["freeze_cls_pos0"]),
                # batching
                "eval_bs": int(CFG["eval_bs"]),
                "attack_bs": int(CFG["attack_bs"]),

                # --- Detector metrics (global & @tau)
                "detector_auroc_nat_vs_adv": float(det_auc),
                "detector_ap_nat_vs_adv": float(det_ap),  # PR-AUC with ADV=1
                "detector_counts_at_tau": {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
                "tpr_adv_at_tau": float(tpr_at_tau),
                "fpr_nat_at_tau": float(fpr_at_tau),
                "tnr_nat_at_tau": float(tnr_at_tau),
                "precision_adv_at_tau": float(prec_at_tau),
                "f1_adv_at_tau": float(f1_at_tau),
                "balanced_accuracy_at_tau": float(bal_acc_at_tau),
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

            save_json(metrics, out_mix / "metrics.json")

            # small hists for visibility
            plot_hist(p_adv_nat, out_mix / "p_adv_nat_hist.png", f"{name} NAT p_adv ({which} mix)")
            plot_hist(p_adv_adv, out_mix / "p_adv_adv_hist_small.png", f"{name} ADV p_adv ({which} mix)")
            cleanup_cuda()

        # Run mixed FGSM + PGD
        mixed_eval("FGSM")
        mixed_eval("PGD")

    print("\n[DONE] Mixed-stream Char defense evaluation complete.")

if __name__ == "__main__":
    main()

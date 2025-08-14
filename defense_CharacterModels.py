"""
Defense test for URL models with a CharCNN detector.

What it does
------------
- Loads URL models: DL-CharCNN, DL-CharTransformer (if present)
- Builds test token IDs from features_extracted.csv using models/_global/url_tokenizer.json
- Creates FGSM/PGD HotFlip adversaries against each base model
- Loads Defense-CharCNN (binary detector: NAT=0, ADV=1) and computes p_adv on each sample
- Auto-chooses threshold tau as a percentile on NAT p_adv to target a given clean FPR budget
- Gates base predictions: if p_adv >= tau -> ABSTAIN; else pass to base model and score
- Saves metrics + classification reports

Outputs
-------
results_defense_char/<ModelName>/
  def_NAT/  def_FGSM/  def_PGD/
    metrics.json, classification_report.txt

Requirements
------------
- models/_global/url_tokenizer.json           (from your training)
- models/DL-CharCNN/model.pt                 (optional base)
- models/DL-CharTransformer/model.pt         (optional base)
- models/Defense-CharCNN/model.pt            (REQUIRED)
- features_extracted.csv                     (with columns: url,label)

Notes
-----
- Threshold tau is picked from NAT test p_adv at quantile NAT_FP_BUDGET (e.g., 0.90 -> ~10% FP on clean).
  For stricter gating, raise NAT_FP_BUDGET (e.g., 0.95). For looser, lower it.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, gc, warnings, argparse
import numpy as np
import pandas as pd

# Plots (optional; offscreen)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -------------------- Config --------------------
CFG = {
    "models_dir": "models",
    "defense_dir": "models/Defense-CharCNN",
    "results_dir": "results_defense_char",
    "dataset_csv": "features_extracted.csv",
    "label_col": "label",
    "url_col": "url",

    # Attack knobs (HotFlip)
    "char_topk_per_step": 1,
    "char_steps_fgsm": 1,   # 1 step => FGSM-like
    "char_steps_pgd": 3,    # MINIMAL CHANGE: reduced default from 5 -> 3
    "freeze_cls_pos0": True,

    # Batch sizes (with OOM backoff)
    "eval_bs": 2048,
    "attack_bs": 1024,

    # Device
    "use_gpu": True,

    # Detector threshold policy on NAT (clean FPR ~ (1 - NAT_FP_BUDGET))
    # e.g., 0.90 => ~10% of clean will be rejected
    "NAT_FP_BUDGET": 0.90,
}

PAD_ID = 256
CLS_ID = 257
VOCAB_SIZE = 258

# -------------------- IO helpers --------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(txt, p: Path):
    p.write_text(txt, encoding="utf-8")

def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    if len(y_true) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def plot_roc(y_true, y_score, out_path: Path, title: str):
    if len(y_true) == 0:
        return
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title(title)
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
    assert tok_path.exists(), "Missing models/_global/url_tokenizer.json (run DL trainer first)"
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
    y = df[cfg["label_col"]].astype(int).to_numpy()
    X_ids = np.stack([encode_url(u, max_len, use_cls) for u in urls])
    return X_ids, y, max_len, use_cls

# -------------------- Models --------------------
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

# -------------------- Eval (batched, OOM backoff) --------------------
@torch.no_grad()
def eval_url_batched(model, ids_np, device, batch_size):
    N = ids_np.shape[0]
    probs_list = []; bs = max(1, int(batch_size)); i = 0
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
                print(f"[WARN] OOM during eval; retrying with batch_size={bs}")
                continue
            raise
    probs = np.concatenate(probs_list, axis=0)
    y_pred = (probs >= 0.5).astype(int)
    return y_pred, probs

# -------------------- HotFlip attacks --------------------
def hotflip_step(ids, y, model, emb_module, freeze_pos0=True, topk=1, device="cuda"):
    # MINIMAL CHANGE: use model.eval() to disable dropout for speed/stability
    model.eval()
    ids = ids.clone().to(device).long()
    y_t = y.to(device).float()

    loss_fn = nn.BCEWithLogitsLoss()
    model.zero_grad(set_to_none=True)

    # manual forward to capture grads on embeddings
    if isinstance(model, CharCNN):
        E = model.emb(ids); E.requires_grad_(True); E.retain_grad()
        x = E.transpose(1, 2)
        x = model.act(model.bn1(model.conv1(x)))
        x = model.act(model.bn2(model.conv2(x)))
        x = torch.amax(x, dim=2)
        # dropout is inactive due to eval()
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

# -------------------- Defense utils --------------------
@torch.no_grad()
def defense_p_adv_batched(def_mdl, ids_np, device, batch_size):
    _, probs = eval_url_batched(def_mdl, ids_np, device, batch_size)
    return probs.reshape(-1)

def choose_tau_from_nat(p_adv_nat, q=0.90):
    q = float(np.clip(q, 0.50, 0.999))
    return float(np.quantile(p_adv_nat, q)) if len(p_adv_nat) else 0.5

def gate_and_score(base_probs, base_preds, p_adv, tau):
    accept = (p_adv < tau).astype(int)
    accepted_idx = np.where(accept == 1)[0]
    return accept, accepted_idx

def write_metrics(model_name, tag, y_true, base_preds, base_probs, p_adv, accept, out_dir: Path):
    out_dir = ensure_dir(out_dir)
    det_labels = np.zeros_like(p_adv, dtype=int) if tag == "NAT" else np.ones_like(p_adv, dtype=int)
    try:
        det_auc = float(roc_auc_score(det_labels, p_adv))
    except Exception:
        det_auc = float("nan")

    accepted_idx = np.where(accept == 1)[0]
    if accepted_idx.size > 0:
        y_true_acc = y_true[accepted_idx]
        preds_acc = base_preds[accepted_idx]
        probs_acc = base_probs[accepted_idx]
        acc = float(accuracy_score(y_true_acc, preds_acc))
        try:
            auc = float(roc_auc_score(y_true_acc, probs_acc)) if len(np.unique(y_true_acc)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        rep = classification_report(y_true_acc, preds_acc, digits=4)
        save_text(rep, out_dir / "classification_report.txt")
        plot_confusion(y_true_acc, preds_acc, out_dir / "confusion_matrix.png", f"{model_name} [{tag}] (accepted)")
        plot_roc(y_true_acc, probs_acc, out_dir / "roc_curve.png", f"{model_name} [{tag}] (accepted)")
    else:
        acc = auc = float("nan")

    coverage = float(accept.mean())

    metrics = {
        "model": model_name,
        "attack": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": int(len(p_adv)),
        "coverage_accept_rate": coverage,
        "detector_auc": det_auc,
        "classification_accuracy_on_accepted": acc,
        "classification_auc_on_accepted": auc,
    }
    save_json(metrics, out_dir / "metrics.json")

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="CharCNN defense test with HotFlip attacks")
    p.add_argument("--model", default="all",
                   choices=["DL-CharCNN","DL-CharTransformer","all"],
                   help="Which base model to evaluate")
    p.add_argument("--attack", default="all",
                   choices=["NAT","FGSM","PGD","all"],
                   help="Which attack to run")
    p.add_argument("--nat-fp-budget", type=float, default=None,
                   help="Quantile on NAT p_adv to choose tau (e.g., 0.95 -> ~5%% FP on clean)")
    p.add_argument("--pgd-steps", type=int, default=None, help="Override PGD steps (default 3)")
    p.add_argument("--fgsm-steps", type=int, default=None, help="Override FGSM steps (default 1)")
    p.add_argument("--topk", type=int, default=None, help="Characters to flip per step (default 1)")
    p.add_argument("--eval-bs", type=int, default=None, help="Eval batch size")
    p.add_argument("--attack-bs", type=int, default=None, help="Attack batch size")
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    return p.parse_args()

# -------------------- Main --------------------
def main():
    args = parse_args()

    # apply CLI overrides
    if args.nat_fp_budget is not None: CFG["NAT_FP_BUDGET"] = float(args.nat_fp_budget)
    if args.pgd_steps is not None:     CFG["char_steps_pgd"] = int(args.pgd_steps)
    if args.fgsm_steps is not None:    CFG["char_steps_fgsm"] = int(args.fgsm_steps)
    if args.topk is not None:          CFG["char_topk_per_step"] = int(args.topk)
    if args.eval_bs is not None:       CFG["eval_bs"] = int(args.eval_bs)
    if args.attack_bs is not None:     CFG["attack_bs"] = int(args.attack_bs)
    if args.cpu:                       CFG["use_gpu"] = False

    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    models_root = Path(CFG["models_dir"])
    defense_root = Path(CFG["defense_dir"])
    results_root = ensure_dir(Path(CFG["results_dir"]))
    global_dir = models_root / "_global"

    # Load dataset
    df = pd.read_csv(CFG["dataset_csv"], usecols=[CFG["url_col"], CFG["label_col"]])
    df[CFG["label_col"]] = df[CFG["label_col"]].astype(int)

    # Build token ids for entire dataset
    Xids_all, y_all, max_len, use_cls = build_text_ids_from_df(df, CFG, global_dir)

    # Load base URL models if present and match selection
    requested_models = []
    if args.model in ("DL-CharCNN", "all"):
        if (models_root / "DL-CharCNN" / "model.pt").exists():
            m = CharCNN().to(device).eval()
            m.load_state_dict(torch.load(models_root/"DL-CharCNN"/"model.pt", map_location=device))
            requested_models.append(("DL-CharCNN", m))
    if args.model in ("DL-CharTransformer", "all"):
        if (models_root / "DL-CharTransformer" / "model.pt").exists():
            m = CharTransformer(max_len=max_len).to(device).eval()
            m.load_state_dict(torch.load(models_root/"DL-CharTransformer"/"model.pt", map_location=device))
            requested_models.append(("DL-CharTransformer", m))

    assert requested_models, "No base URL models found for the given --model selection."

    # Load defense model (CharCNN)
    assert (defense_root / "model.pt").exists(), "Missing Defense-CharCNN model.pt"
    def_mdl = CharCNN().to(device).eval()
    def_mdl.load_state_dict(torch.load(defense_root/"model.pt", map_location=device))

    # Choose attacks
    if args.attack == "all":
        attacks = ["NAT", "FGSM", "PGD"]
    else:
        attacks = [args.attack]

    for name, base_mdl in requested_models:
        print(f"\n==== Defense test for: {name} ====")
        out_base = ensure_dir(results_root / name)

        # Always compute NAT p_adv to pick tau, even if user didn't request saving NAT metrics
        p_nat = defense_p_adv_batched(def_mdl, Xids_all, device, CFG["eval_bs"])
        tau = choose_tau_from_nat(p_nat, q=CFG["NAT_FP_BUDGET"])
        print(f"[{name}] chosen tau from NAT (quantile={CFG['NAT_FP_BUDGET']:.2f}) -> tau={tau:.4f}")

        if "NAT" in attacks:
            base_pred_nat, base_prob_nat = eval_url_batched(base_mdl, Xids_all, device, CFG["eval_bs"])
            accept_nat, _ = gate_and_score(base_prob_nat, base_pred_nat, p_nat, tau)
            write_metrics(name, "NAT", y_all, base_pred_nat, base_prob_nat, p_nat, accept_nat, out_base / "def_NAT")

        if "FGSM" in attacks:
            print(f"[{name}] crafting FGSM (steps={CFG['char_steps_fgsm']}, topk={CFG['char_topk_per_step']})")
            ids_adv_fgsm = attack_url_hotflip_batched(
                base_mdl, Xids_all, y_all,
                steps=CFG["char_steps_fgsm"], topk=CFG["char_topk_per_step"],
                device=device, freeze_pos0=CFG["freeze_cls_pos0"], batch_size=CFG["attack_bs"]
            )
            base_pred_fgsm, base_prob_fgsm = eval_url_batched(base_mdl, ids_adv_fgsm, device, CFG["eval_bs"])
            p_fgsm = defense_p_adv_batched(def_mdl, ids_adv_fgsm, device, CFG["eval_bs"])
            accept_fgsm, _ = gate_and_score(base_prob_fgsm, base_pred_fgsm, p_fgsm, tau)
            write_metrics(name, "FGSM", y_all, base_pred_fgsm, base_prob_fgsm, p_fgsm, accept_fgsm, out_base / "def_FGSM")
            cleanup_cuda()

        if "PGD" in attacks:
            print(f"[{name}] crafting PGD (steps={CFG['char_steps_pgd']}, topk={CFG['char_topk_per_step']})")
            ids_adv_pgd = attack_url_hotflip_batched(
                base_mdl, Xids_all, y_all,
                steps=CFG["char_steps_pgd"], topk=CFG["char_topk_per_step"],
                device=device, freeze_pos0=CFG["freeze_cls_pos0"], batch_size=CFG["attack_bs"]
            )
            base_pred_pgd, base_prob_pgd = eval_url_batched(base_mdl, ids_adv_pgd, device, CFG["eval_bs"])
            p_pgd = defense_p_adv_batched(def_mdl, ids_adv_pgd, device, CFG["eval_bs"])
            accept_pgd, _ = gate_and_score(base_prob_pgd, base_pred_pgd, p_pgd, tau)
            write_metrics(name, "PGD", y_all, base_pred_pgd, base_prob_pgd, p_pgd, accept_pgd, out_base / "def_PGD")
            cleanup_cuda()

    print("\n[DONE] CharCNN defense evaluation finished.")

if __name__ == "__main__":
    main()

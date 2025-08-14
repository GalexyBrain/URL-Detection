# train_defense_cnn.py (RAM-safe + tqdm progress)
from __future__ import annotations
from pathlib import Path
import os, json, random, warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from contextlib import nullcontext
from tqdm.auto import tqdm  # <-- NEW

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CFG = {
    "data_csv": "urls_adversarial_defense_dataset.csv",   # expects: url,label[,attack_type]
    "models_out": "models/Defense-CharCNN",
    "global_dir": "models/_global",
    "random_state": 42,

    # More conservative by default; auto-fallback if OOM
    "batch_size": 1024,
    "min_batch_size": 32,
    "max_epochs": 3,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "patience": 2,                 # early stop on val AUROC
    "num_workers": 0,              # workers copy data -> RAM spike; set >0 only if plenty of RAM
    "label_smoothing": 0.0,

    # Progress bars
    "progress_bar": True,          # set False to disable
}

PAD_ID, CLS_ID, VOCAB_SIZE = 256, 257, 258

def read_tokenizer(global_dir: Path):
    info = json.loads((global_dir / "url_tokenizer.json").read_text())
    return int(info.get("max_len", 256)), bool(info.get("use_cls_token", True))

def encode_url(u: str, max_len: int, use_cls=True):
    b = u.encode("utf-8", "ignore")
    if use_cls:
        b = b[:max_len - 1]
        ids = [CLS_ID] + [x for x in b]
    else:
        b = b[:max_len]
        ids = [x for x in b]
    if len(ids) < max_len:
        ids += [PAD_ID] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    # Keep compact in memory (uint16) → cast to Long only when feeding the model
    return np.asarray([i if i >= 256 else max(0, min(255, i)) for i in ids], dtype=np.uint16)

class URLDataset(Dataset):
    """Lazy encoding (RAM-friendly): encode on access; store only string & tiny label."""
    def __init__(self, urls, labels, max_len, use_cls):
        self.urls = list(urls)
        self.labels = np.asarray(labels, dtype=np.uint8)  # small dtype
        self.max_len = max_len
        self.use_cls = use_cls
    def __len__(self): return len(self.urls)
    def __getitem__(self, i):
        ids = encode_url(self.urls[i], self.max_len, self.use_cls)  # uint16
        return torch.from_numpy(ids).long(), torch.tensor(self.labels[i], dtype=torch.float32)

class CharCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=64, p_drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.conv1 = nn.Conv1d(d_model, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn1, self.bn2 = nn.BatchNorm1d(128), nn.BatchNorm1d(256)
        self.act, self.drop = nn.GELU(), nn.Dropout(p_drop)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(p_drop), nn.Linear(128, 1)
        )
    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = torch.amax(x, dim=2)
        x = self.drop(x)
        return self.head(x).squeeze(-1)

def main():
    # Repro
    random.seed(CFG["random_state"])
    np.random.seed(CFG["random_state"])
    torch.manual_seed(CFG["random_state"])

    out_dir = Path(CFG["models_out"]); out_dir.mkdir(parents=True, exist_ok=True)
    max_len, use_cls = read_tokenizer(Path(CFG["global_dir"]))
    print(f"[INFO] tokenizer: max_len={max_len}, use_cls={use_cls}")

    # Read only what we need, with compact dtypes
    df = pd.read_csv(
        CFG["data_csv"],
        usecols=["url", "label"],
        dtype={"url": "string", "label": "int8"},
    )
    df["label"] = df["label"].astype("int8")

    # Stratified split (indices only; keep strings in RAM once)
    idx = np.arange(len(df))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, df["label"].values, test_size=0.3, random_state=CFG["random_state"], stratify=df["label"].values
    )
    idx_val, idx_te, y_val, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=2/3, random_state=CFG["random_state"], stratify=y_tmp
    )
    json.dump(
        {"train_idx": idx_tr.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_te.tolist()},
        open(out_dir / "split_indices.json", "w"),
    )

    # Datasets (lazy encoding)
    tr = URLDataset(df.loc[idx_tr, "url"], df.loc[idx_tr, "label"], max_len, use_cls)
    va = URLDataset(df.loc[idx_val, "url"], df.loc[idx_val, "label"], max_len, use_cls)
    te = URLDataset(df.loc[idx_te, "url"], df.loc[idx_te, "label"], max_len, use_cls)

    # Class imbalance → pos_weight
    pos = int((df.loc[idx_tr, "label"] == 1).sum())
    neg = int((df.loc[idx_tr, "label"] == 0).sum())
    pos_weight = torch.tensor([max(1.0, neg / max(1, pos))], dtype=torch.float32)
    print(f"[INFO] train counts: NAT={neg:,} ADV={pos:,}  pos_weight={pos_weight.item():.3f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"  # pin only on CUDA

    model = CharCNN(p_drop=CFG["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    amp_enabled = device.type == "cuda"
    autocast = torch.cuda.amp.autocast if amp_enabled else nullcontext
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def make_loaders(bs: int):
        dl_tr = DataLoader(tr, batch_size=bs, shuffle=True,  num_workers=CFG["num_workers"],
                           pin_memory=pin_memory, persistent_workers=False)
        dl_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=CFG["num_workers"],
                           pin_memory=pin_memory, persistent_workers=False)
        dl_te = DataLoader(te, batch_size=bs, shuffle=False, num_workers=CFG["num_workers"],
                           pin_memory=pin_memory, persistent_workers=False)
        return dl_tr, dl_va, dl_te

    def run_epoch(dl, train=True, epoch=0, phase="train"):
        model.train(train)
        tot, n = 0.0, 0
        yprob, ytrue = [], []

        # nice live readout with smoothed loss
        disable_bar = not CFG.get("progress_bar", True)
        pbar = tqdm(
            dl, total=len(dl), dynamic_ncols=True, leave=False,
            desc=f"[{phase.title()} E{epoch:02d}]", disable=disable_bar
        )
        ema_loss = None

        for ids, y in pbar:
            ids = ids.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)

            if train:
                opt.zero_grad(set_to_none=True)

            with autocast():
                logits = model(ids).float()
                y_ls = y
                if CFG["label_smoothing"] > 0 and train:
                    y_ls = y * (1 - CFG["label_smoothing"]) + 0.5 * CFG["label_smoothing"]
                loss = crit(logits, y_ls)

            if train:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward(); opt.step()

            # running stats
            cur = loss.item()
            ema_loss = cur if ema_loss is None else (0.9 * ema_loss + 0.1 * cur)
            tot += cur * ids.size(0); n += ids.size(0)

            with torch.no_grad():
                yprob.append(torch.sigmoid(logits).detach().cpu().numpy())
                ytrue.append(y.detach().cpu().numpy().astype(np.int32))

            pbar.set_postfix({"loss": f"{ema_loss:.4f}"})
        pbar.close()

        yprob = np.concatenate(yprob, axis=0)
        ytrue = np.concatenate(ytrue, axis=0)
        auroc = roc_auc_score(ytrue, yprob) if ytrue.sum() and (1 - ytrue).sum() else float("nan")
        ap    = average_precision_score(ytrue, yprob)
        return tot / max(1, n), auroc, ap

    # Training with automatic batch-size fallback (avoids OOM crashes)
    bs = CFG["batch_size"]
    dl_tr, dl_va, dl_te = make_loaders(bs)
    best_auroc, best_state, noimp = -1.0, None, 0

    epoch = 0
    while epoch < CFG["max_epochs"]:
        try:
            epoch += 1
            ltr, au_tr, ap_tr = run_epoch(dl_tr, train=True,  epoch=epoch, phase="train")
            lva, au_va, ap_va = run_epoch(dl_va, train=False, epoch=epoch, phase="val")
            print(f"[E{epoch:02d}] train loss {ltr:.4f} | AUROC {au_tr:.4f} AP {ap_tr:.4f} "
                  f"|| val loss {lva:.4f} | AUROC {au_va:.4f} AP {ap_va:.4f}")
            if au_va > best_auroc + 1e-4:
                best_auroc, noimp = au_va, 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                noimp += 1
                if noimp >= CFG["patience"]:
                    print("[INFO] early stopping.")
                    break
        except RuntimeError as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda error" in msg) and bs > CFG["min_batch_size"]:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                bs = max(CFG["min_batch_size"], bs // 2)
                print(f"[WARN] OOM detected → reducing batch_size to {bs} and recreating loaders.")
                dl_tr, dl_va, dl_te = make_loaders(bs)
                epoch -= 1  # repeat this epoch at smaller batch
                continue
            raise

    # Save best
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, Path(CFG["models_out"]) / "model.pt")
    json.dump({"max_len": max_len, "use_cls_token": use_cls},
              open(Path(CFG["models_out"]) / "tokenizer_snapshot.json", "w"))
    print(f"[OK] saved to {CFG['models_out']}")

    # Quick test AUROC (with progress bar)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    lte, au_te, ap_te = run_epoch(dl_te, train=False, epoch=0, phase="test")
    print(f"[TEST] loss {lte:.4f} | AUROC {au_te:.4f}  AP {ap_te:.4f}")

if __name__ == "__main__":
    main()

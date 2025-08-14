"""
Generate adversarial URL dataset for CNN-based defense (streaming + batched HotFlip + tqdm).

Input:
  - features_extracted.csv  (must contain: url, label)

Output:
  - urls_adversarial_defense_dataset.csv  (url,label,attack_type,parent_url)

Memory-safe:
  - Streams rows to CSV in chunks (no huge DataFrames)
  - Global de-dup via a compact Bloom filter (~16MB)
  - Heuristics in CPU batches with periodic flush
  - HotFlip (CharCNN) in GPU batches with OOM backoff
"""

from __future__ import annotations
from pathlib import Path
import os, json, random, warnings, time, hashlib, math
import numpy as np
import pandas as pd
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------- CONFIG ----------------------------
CFG = {
    "input_csv": "features_extracted.csv",
    "out_csv": "urls_adversarial_defense_dataset.csv",
    "models_dir": "models",
    "global_dir": "models/_global",
    "random_state": 42,

    # Dataset throttles
    "MAX_URLS": None,             # e.g., 200_000 to cap input rows
    "ADV_PER_URL": 6,             # max heuristic variants per url

    # Heuristics batching (CPU)
    "heur_batch_urls": 100_000,   # process this many source URLs per CPU batch
    "flush_every_rows": 200_000,  # flush to CSV after collecting this many output rows

    # HotFlip controls (CharCNN only for speed/VRAM)
    "use_hotflip": True,
    "hotflip_attacks": ["FGSM", "PGD"],  # choose subset: ["FGSM"], ["PGD"], or both
    "hotflip_frac": 0.10,         # fraction of URLs for HotFlip subset
    "hotflip_topk": 1,
    "hotflip_steps_fgsm": 1,
    "hotflip_steps_pgd": 3,
    "hotflip_batch_size": 1024,   # auto-backoff on CUDA OOM
    "hotflip_chunk": 20_000,      # GPU-process this many URLs per chunk

    # Heuristic strengths
    "confusable_prob": 0.25,
    "percent_prob": 0.25,
    "case_prob": 0.25,
    "subdomain_count": (2, 5),
    "path_tokens": (3, 8),

    # De-dup (Bloom filter)
    "bloom_bits": 1 << 27,        # ~134M bits -> 16MB
    "bloom_k": 4,

    # tqdm
    "tqdm_disable": False,        # set True to silence progress bars
}

# --------------------- tqdm (with graceful fallback) ----------------
try:
    from tqdm.auto import tqdm as _tqdm
    def make_pbar(total=None, desc="", unit="it"):
        return _tqdm(total=total, desc=desc, unit=unit, leave=False, disable=CFG["tqdm_disable"])
    def tqdmit(iterable, **kw):
        kw.setdefault("leave", False)
        kw.setdefault("disable", CFG["tqdm_disable"])
        return _tqdm(iterable, **kw)
    tqdm_write = _tqdm.write
except Exception:  # fallback if tqdm missing
    def make_pbar(total=None, desc="", unit="it"):
        class _NoOp: 
            def update(self, n=1): pass
            def close(self): pass
        return _NoOp()
    def tqdmit(iterable, **kw): return iterable
    def tqdm_write(msg): print(msg)

# --------------------- Utility / Reproducibility ------------------
rng = np.random.default_rng(CFG["random_state"])
random.seed(CFG["random_state"])

def map_label(v):
    try:
        iv = int(v);  return 1 if iv != 0 else 0
    except Exception:
        s = str(v).strip().lower()
        return 1 if s in {"1","true","malicious","phishing","malware","bad"} else 0

# --------------------- Tokenizer + Model (optional) ----------------
PAD_ID = 256
CLS_ID = 257
VOCAB_SIZE = 258

def have_tokenizer(global_dir: Path):
    return (global_dir / "url_tokenizer.json").exists()

def read_tokenizer(global_dir: Path):
    info = json.loads((global_dir / "url_tokenizer.json").read_text(encoding="utf-8"))
    return int(info.get("max_len", 256)), bool(info.get("use_cls_token", True))

def encode_url(u: str, max_len: int, use_cls=True):
    # return uint16 to save memory (IDs in [0..257])
    b = u.encode("utf-8","ignore")
    if use_cls:
        b = b[:max_len-1]; ids = [CLS_ID] + [x for x in b]
    else:
        b = b[:max_len];    ids = [x for x in b]
    if len(ids) < max_len: ids += [PAD_ID] * (max_len - len(ids))
    else:                  ids = ids[:max_len]
    ids = [i if i >= 256 else max(0, min(255, i)) for i in ids]
    return np.array(ids, dtype=np.uint16)

def clip_url(u: str, max_len: int):
    b = u.encode("utf-8","ignore")
    if len(b) <= max_len-1: return u
    return b[:max_len-1].decode("utf-8","ignore")

def try_load_charcnn(models_dir: Path):
    """
    Load DL-CharCNN only. Returns (model, max_len, use_cls, device) or None.
    """
    try:
        import torch
        import torch.nn as nn
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_len, use_cls = read_tokenizer(Path(CFG["global_dir"]))

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

        mpath = models_dir / "DL-CharCNN" / "model.pt"
        if not mpath.exists():
            return None
        import torch
        mdl = CharCNN().to(device)
        state = torch.load(mpath, map_location=device)
        mdl.load_state_dict(state)
        mdl.eval()
        try:
            mdl = torch.compile(mdl)
        except Exception:
            pass
        return mdl, max_len, use_cls, device
    except Exception:
        return None

# ---------------- HotFlip (batched + OOM backoff) ------------------
def hotflip_attack_batch(model, ids_np, steps, topk, device, freeze_pos0=True, batch_size=1024, pbar=None):
    import torch
    import torch.nn as nn
    PAD = PAD_ID

    def hotflip_step(ids, model, emb_module, freeze_pos0, topk, device):
        model.train()
        ids = ids.clone().to(device).long()
        y_t = torch.ones(ids.size(0), device=device).float()   # untargeted push to 1
        loss_fn = nn.BCEWithLogitsLoss()
        model.zero_grad(set_to_none=True)

        # CharCNN path
        E = model.emb(ids); E.requires_grad_(True); E.retain_grad()
        x = E.transpose(1, 2)
        x = model.act(model.bn1(model.conv1(x)))
        x = model.act(model.bn2(model.conv2(x)))
        x = torch.amax(x, dim=2)
        x = model.drop(x)
        logits = model.head(x).squeeze(-1)

        loss = loss_fn(logits, y_t)
        loss.backward()
        G = E.grad.detach()                          # (B,L,D)
        W = emb_module.weight.detach()               # (V,D)
        B, L, D = G.shape
        ids_new = ids.clone()

        pos_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        if freeze_pos0: pos_mask[:, 0] = False
        pos_mask &= (ids != PAD)

        curW = W[ids]
        scores = torch.einsum("vd,bld->blv", W, G) - torch.einsum("bld,bld->bl", curW, G)[..., None]
        scores.scatter_(2, ids.unsqueeze(-1), float("-inf"))  # forbid identity
        scores[:, :, PAD] = float("-inf")                     # forbid PAD

        gains, cand = scores.max(dim=2)
        gains = gains.masked_fill(~pos_mask, float("-inf"))
        k = min(max(1, int(topk)), L)
        top_gain, top_pos = torch.topk(gains, k=k, dim=1)

        for b in range(B):
            for j in range(k):
                if top_gain[b, j].item() > 0:
                    p = int(top_pos[b, j].item())
                    ids_new[b, p] = int(cand[b, p].item())
        return ids_new.detach()

    N = ids_np.shape[0]
    out = np.empty_like(ids_np)

    # own progress bar if not provided
    close_pbar = False
    if pbar is None:
        pbar = make_pbar(total=N, desc="HotFlip", unit="url")
        close_pbar = True

    i = 0
    bs = max(1, int(batch_size))
    while i < N:
        j = min(N, i + bs)
        import torch
        try:
            ids = torch.from_numpy(ids_np[i:j]).long().to(device, non_blocking=True)
            ids_adv = ids.clone()
            for _ in range(max(1, int(steps))):
                ids_adv = hotflip_step(ids_adv, model, getattr(model, "emb"), True, topk, device)
            out[i:j] = ids_adv.detach().cpu().numpy()
            pbar.update(j - i)
            i = j  # advance window only on success
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and bs > 1 and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                new_bs = max(1, bs // 2)
                tqdm_write(f"[WARN] CUDA OOM during HotFlip; reducing batch_size {bs} -> {new_bs}")
                bs = new_bs
                continue
            raise

    if close_pbar:
        pbar.close()
    model.eval()
    return out

# --------------------- Heuristic perturbations ---------------------
CONFUSABLES = {
    "a":"а","e":"е","o":"о","p":"р","c":"с","y":"у","x":"х","i":"і","l":"ӏ",
    "d":"ԁ","g":"ɡ","m":"м","n":"п","h":"һ","k":"к","t":"т","B":"Β","H":"Н","M":"Μ"
}
import string as _str
ALNUM = _str.ascii_letters + _str.digits
DELIMS = [";", "_", "-", ".", "~", "@", "%", "=", "&"]

def percent_encode_char(ch): return "%" + format(ord(ch), "02X")

def confusable_swap(s: str, prob=0.2):
    out = []
    for ch in s:
        low = ch.lower()
        if low in CONFUSABLES and rng.random() < prob:
            out.append(CONFUSABLES[low])
        else:
            out.append(ch)
    return "".join(out)

def percent_encode_some(s: str, prob=0.2):
    out = []
    for ch in s:
        if ch.isalnum() and rng.random() < prob:
            out.append(percent_encode_char(ch))
        elif ch in ".-_" and rng.random() < prob:
            out.append(percent_encode_char(ch))
        else:
            out.append(ch)
    return "".join(out)

def case_jitter(s: str, prob=0.2):
    out = []
    for ch in s:
        if ch.isalpha() and rng.random() < prob:
            out.append(ch.upper() if ch.islower() else ch.lower())
        else:
            out.append(ch)
    return "".join(out)

def rand_token(k=8): return "".join(rng.choice(list(ALNUM)) for _ in range(k))

def with_extra_subdomains(host: str, kmin=2, kmax=5):
    parts = host.split(".")
    k = int(rng.integers(kmin, kmax+1))
    extra = [rand_token(int(rng.integers(3,8))) for _ in range(k)]
    return ".".join(extra + parts)

def with_path_padding(path: str, tmin=3, tmax=8):
    toks = [p for p in path.split("/") if p]
    k = int(rng.integers(tmin, tmax+1))
    toks += [rand_token(int(rng.integers(4,12))) for _ in range(k)]
    return "/" + "/".join(toks)

def with_random_insert(s: str, k=3):
    chars = list(s)
    for _ in range(k):
        pos = int(rng.integers(0, len(chars)+1))
        chars.insert(pos, rng.choice(DELIMS))
    return "".join(chars)

def add_userinfo_at(urlp):
    user = rand_token(int(rng.integers(5,12)))
    host = urlp.netloc or "example.com"
    netloc = f"{user}@{host}"
    return urlunparse((urlp.scheme or "http", netloc, urlp.path, urlp.params, urlp.query, urlp.fragment))

def add_extra_port(urlp):
    host = urlp.netloc
    base = host.split(":")[0] if ":" in host else host
    port = rng.choice([":80", ":443", f":{int(rng.integers(1025, 65535))}"])
    return urlunparse((urlp.scheme or "http", base + port, urlp.path, urlp.params, urlp.query, urlp.fragment))

def rebuild_url(urlp, host=None, path=None, query=None):
    netloc = host if host is not None else urlp.netloc
    pth = path if path is not None else urlp.path
    qry = query if query is not None else urlp.query
    return urlunparse((urlp.scheme or "http", netloc, pth, urlp.params, qry, urlp.fragment))

def heuristic_attacks(u: str, max_len: int) -> list[tuple[str, str]]:
    out = []
    try:
        p = urlparse(u)
        host = p.netloc or ""
        path = p.path or "/"
        if host:
            h2 = confusable_swap(host, CFG["confusable_prob"])
            out.append((rebuild_url(p, host=h2), "CONFUSABLE"))
        p2 = percent_encode_some(path, CFG["percent_prob"])
        out.append((rebuild_url(p, path=p2), "PERCENT"))
        out.append((add_userinfo_at(p), "USERINFO_AT"))
        if host:
            h3 = with_extra_subdomains(host, *CFG["subdomain_count"])
            out.append((rebuild_url(p, host=h3), "SUBDOMAIN_PAD"))
        out.append((rebuild_url(p, path=with_path_padding(path, *CFG["path_tokens"])), "PATH_PAD"))
        cq = case_jitter((p.path or "") + ("?" + p.query if p.query else ""), CFG["case_prob"])
        if "?" in cq:
            new_path, new_query = cq.split("?", 1)
        else:
            new_path, new_query = cq, ""
        out.append((rebuild_url(p, path=new_path, query=new_query), "CASE_JITTER"))
        out.append((add_extra_port(p), "EXTRA_PORT"))
        out.append((with_random_insert(u, k=3), "RANDOM_INSERT"))
    except Exception:
        out.append((with_random_insert(u, k=3), "RANDOM_INSERT"))
        out.append((percent_encode_some(u, CFG["percent_prob"]), "PERCENT"))
    clipped = [(clip_url(v, max_len), tag) for v, tag in out]
    return clipped

# ---- worker wrapper for multiprocess (must be top-level picklable)
def _heuristic_mp(args):
    u, max_len, adv_per_url, seed = args
    import numpy as _np, random as _rand
    _np.random.seed(seed); _rand.seed(seed)
    try:
        vs = heuristic_attacks(u, max_len)
        _np.random.shuffle(vs)
        vs = vs[:max(0, adv_per_url)]
        return [{"url": v, "label": 1, "attack_type": tag, "parent_url": u} for (v, tag) in vs]
    except Exception:
        return [{"url": u, "label": 1, "attack_type": "HEUR_FAIL", "parent_url": u}]

# --------------------------- Bloom Filter --------------------------
class Bloom:
    def __init__(self, bits=1<<27, k=4, salt=b"malurl"):
        self.bits = int(bits)
        self.k = int(k)
        self.buf = np.zeros((self.bits + 7) // 8, dtype=np.uint8)
        self.salt = salt

    def _hashes(self, s: str):
        b = s.encode("utf-8", "ignore")
        h = hashlib.blake2b(b + self.salt, digest_size=16).digest()  # 128-bit
        # derive 4×32-bit ints
        return [(int.from_bytes(h[i*4:(i+1)*4], "little") % self.bits) for i in range(4)]

    def check_add(self, s: str) -> bool:
        """Return True if newly added (i.e., wasn't present)."""
        idxs = self._hashes(s)
        # probe
        present = True
        for pos in idxs[:self.k]:
            byte = pos >> 3
            bit = pos & 7
            if (self.buf[byte] >> bit) & 1 == 0:
                present = False
        if present:
            return False
        # set
        for pos in idxs[:self.k]:
            byte = pos >> 3
            bit = pos & 7
            self.buf[byte] |= (1 << bit)
        return True

# --------------------------- IO helpers ---------------------------
def append_rows(out_path: Path, rows: list[dict], header_written: bool):
    if not rows:
        return header_written
    df = pd.DataFrame(rows, columns=["url","label","attack_type","parent_url"])
    mode = "a" if out_path.exists() or header_written else "w"
    df.to_csv(out_path, mode=mode, header=(mode=="w"), index=False, encoding="utf-8")
    return True

# --------------------------- MAIN PIPELINE ------------------------
def main():
    t0 = time.perf_counter()
    out_path = Path(CFG["out_csv"])
    if out_path.exists():
        out_path.unlink()  # start fresh

    tqdm_write("[INFO] Loading input CSV...")
    df = pd.read_csv(CFG["input_csv"], usecols=["url","label"])
    assert "url" in df.columns and "label" in df.columns, "features_extracted.csv must have columns: url,label"
    df["label"] = df["label"].apply(map_label).astype(np.int64)

    if CFG["MAX_URLS"] is not None and len(df) > CFG["MAX_URLS"]:
        tqdm_write(f"[INFO] Sampling MAX_URLS={CFG['MAX_URLS']} from {len(df):,}")
        df = df.sample(CFG["MAX_URLS"], random_state=CFG["random_state"]).reset_index(drop=True)

    urls = df["url"].astype(str).tolist()
    N = len(urls)

    # Tokenizer info
    if have_tokenizer(Path(CFG["global_dir"])):
        max_len, use_cls = read_tokenizer(Path(CFG["global_dir"]))
        tqdm_write(f"[INFO] Tokenizer found -> max_len={max_len}, use_cls={use_cls}")
    else:
        max_len, use_cls = 256, True
        tqdm_write("[WARN] Tokenizer not found; defaulting max_len=256")

    # ---- Global de-dup filter
    bloom = Bloom(bits=CFG["bloom_bits"], k=CFG["bloom_k"])
    header_written = False

    # ---- NAT (streamed)
    tqdm_write("[PHASE] Writing NAT (natural) samples (streamed)...")
    nat_rows = []
    pbar_nat = make_pbar(total=N, desc="NAT", unit="url")
    for u in urls:
        u_str = str(u)
        if bloom.check_add(u_str):
            nat_rows.append({"url": u_str, "label": 0, "attack_type": "NAT", "parent_url": u_str})
        if len(nat_rows) >= CFG["flush_every_rows"]:
            header_written = append_rows(out_path, nat_rows, header_written); nat_rows.clear()
        pbar_nat.update(1)
    pbar_nat.close()
    header_written = append_rows(out_path, nat_rows, header_written); nat_rows.clear()

    # ---- Heuristics in CPU batches (streamed)
    tqdm_write("[PHASE] Generating heuristic variants (CPU batches, streaming)...")
    batch_size = int(CFG["heur_batch_urls"])
    flush_every = int(CFG["flush_every_rows"])
    pbar_heu = make_pbar(total=N, desc="Heuristics (src URLs)", unit="url")

    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            batch_urls = urls[start:end]
            # map with per-item seeds for variability
            args_iter = ((u, max_len, CFG["ADV_PER_URL"], CFG["random_state"] + start + i)
                          for i, u in enumerate(batch_urls))
            rows_buffer = []
            for out_list in ex.map(_heuristic_mp, args_iter, chunksize=64):
                # out_list is list[dict]
                # stream-filter with bloom
                for row in out_list:
                    if bloom.check_add(row["url"]):
                        rows_buffer.append(row)
                    if len(rows_buffer) >= flush_every:
                        header_written = append_rows(out_path, rows_buffer, header_written); rows_buffer.clear()
            # flush remaining for this batch
            header_written = append_rows(out_path, rows_buffer, header_written); rows_buffer.clear()
            pbar_heu.update(len(batch_urls))
    pbar_heu.close()

    # ---- HotFlip (CharCNN only, chunked + OOM backoff, streamed)
    added_hotflip = 0
    if CFG["use_hotflip"] and have_tokenizer(Path(CFG["global_dir"])):
        tqdm_write("[PHASE] Loading DL-CharCNN for HotFlip...")
        env = try_load_charcnn(Path(CFG["models_dir"]))
        if env is None:
            tqdm_write("[WARN] HotFlip disabled (no DL-CharCNN model found).")
        else:
            mdl, max_len_tok, use_cls_tok, device = env
            k = max(1, int(CFG["hotflip_frac"] * N))
            idx_subset = sorted(rng.choice(np.arange(N), size=k, replace=False).tolist())
            sub_urls = [clip_url(urls[i], max_len_tok) for i in idx_subset]
            tqdm_write(f"[INFO] HotFlip subset: {len(sub_urls):,} / {N:,} URLs (~{100*CFG['hotflip_frac']:.1f}%) using DL-CharCNN")

            attacks = [a for a in CFG["hotflip_attacks"] if a in {"FGSM","PGD"}]
            total_steps = len(sub_urls) * max(1, len(attacks))
            pbar_hf = make_pbar(total=total_steps, desc="HotFlip", unit="url")

            chunk = int(CFG["hotflip_chunk"])
            bs = int(CFG["hotflip_batch_size"])

            for s in range(0, len(sub_urls), chunk):
                sub = sub_urls[s:s+chunk]
                # encode chunk (uint16 for memory)
                ids_all = np.stack([encode_url(u, max_len_tok, use_cls_tok) for u in sub])

                # For each attack type, run and stream results
                if "FGSM" in attacks:
                    ids_fgsm = hotflip_attack_batch(mdl, ids_all, steps=CFG["hotflip_steps_fgsm"],
                                                    topk=CFG["hotflip_topk"], device=device,
                                                    batch_size=bs, pbar=pbar_hf)
                    rows = []
                    for parent, row in zip(sub, ids_fgsm):
                        b = bytes([int(x) for x in row.tolist() if int(x) < 256])
                        v = clip_url(b.decode("utf-8","ignore"), max_len_tok)
                        if v and v != parent and bloom.check_add(v):
                            rows.append({"url": v, "label": 1, "attack_type": "HOTFLIP_FGSM", "parent_url": parent})
                    header_written = append_rows(out_path, rows, header_written)
                    added_hotflip += len(rows)
                    del ids_fgsm, rows

                if "PGD" in attacks:
                    ids_pgd = hotflip_attack_batch(mdl, ids_all, steps=CFG["hotflip_steps_pgd"],
                                                   topk=CFG["hotflip_topk"], device=device,
                                                   batch_size=bs, pbar=pbar_hf)
                    rows = []
                    for parent, row in zip(sub, ids_pgd):
                        b = bytes([int(x) for x in row.tolist() if int(x) < 256])
                        v = clip_url(b.decode("utf-8","ignore"), max_len_tok)
                        if v and v != parent and bloom.check_add(v):
                            rows.append({"url": v, "label": 1, "attack_type": "HOTFLIP_PGD", "parent_url": parent})
                    header_written = append_rows(out_path, rows, header_written)
                    added_hotflip += len(rows)
                    del ids_pgd, rows

                # free chunk memory
                del ids_all
                try:
                    import torch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            pbar_hf.close()
    else:
        if not CFG["use_hotflip"]:
            tqdm_write("[INFO] HotFlip disabled by config.")
        else:
            tqdm_write("[WARN] HotFlip needs tokenizer; skipping.")

    # ---- Final stats (streamed file; re-read for quick counts)
    tqdm_write("[PHASE] Final stats...")
    # For large files, sampling head is safer; but we can count attack types with chunks:
    nat_count = 0
    adv_count = 0
    type_counts = {}
    for chunk in pd.read_csv(out_path, usecols=["attack_type"], chunksize=500_000):
        c = chunk["attack_type"].value_counts()
        for k, v in c.items():
            type_counts[k] = type_counts.get(k, 0) + int(v)
    nat_count = type_counts.get("NAT", 0)
    adv_count = sum(v for k, v in type_counts.items() if k != "NAT")

    tqdm_write(f"[STATS] NAT={nat_count:,}  ADV={adv_count:,}")
    top_items = sorted(type_counts.items(), key=lambda x: -x[1])[:12]
    tqdm_write("[TOP ATTACKS]")
    for k, v in top_items:
        tqdm_write(f"  {k:16s} {v:,}")

    dt = time.perf_counter() - t0
    tqdm_write(f"[OK] Wrote dataset to {out_path} (elapsed {dt:.1f}s)")
    try:
        head = pd.read_csv(out_path, nrows=5)
        tqdm_write(head.to_string(index=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a 3-class base dataset (0=benign NAT, 1=malicious NAT, 2=ADV) with:
  • Exact layout as features_extracted.csv minus `url`: <features in saved order>, label
  • Hard cap of CAP rows per class using reservoir sampling (uniform random)
  • Chunked I/O + threaded transform; single-threaded reservoir updates for safety

Input:
  features_adversarial_defense_dataset.csv  (has orig_label, is_adv, and features)
  models/_global/feature_columns.json       (feature order)

Output:
  features_base3_strict_cap.csv

CLI:
  --cap-per-class INT   (default 1000000)
  --chunksize INT       (default 500000)
  --workers INT         (default min(8, CPU))
  --buffer INT          (in-flight transformed chunks, default 6)
  --seed INT            (default 42)
  --shuffle-final 0/1   (default 1 -> shuffle final rows)
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

IN_CSV  = Path("features_adversarial_defense_dataset.csv")
OUT_CSV = Path("features_base3_strict_cap.csv")
GLOBAL  = Path("models/_global/feature_columns.json")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(IN_CSV))
    p.add_argument("--output", default=str(OUT_CSV))
    p.add_argument("--global-schema", default=str(GLOBAL))
    p.add_argument("--cap-per-class", type=int, default=1_000_000)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    p.add_argument("--buffer", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-final", type=int, default=1)
    return p.parse_args()

def transform_chunk(seq: int, chunk: pd.DataFrame, feature_cols: list[str]) -> tuple[int, dict[int, np.ndarray]]:
    """
    Returns per-class feature blocks for this chunk.
    dict: {0: arr0, 1: arr1, 2: arr2}, where each arr is (k, D+1) -> [features..., label]
    We pack label as last column to keep moving data compact.
    """
    # Compute 3-class label
    orig   = chunk["orig_label"].astype("int8").to_numpy()
    is_adv = chunk["is_adv"].astype("int8").to_numpy()
    label3 = np.where(is_adv == 1, 2, orig).astype("int8")

    # Extract features in saved order
    X = chunk[feature_cols].to_numpy(dtype=np.float32, copy=False)

    # Split by class
    out = {}
    for cls in (0, 1, 2):
        mask = (label3 == cls)
        if not mask.any():
            out[cls] = None
            continue
        # pack features + label column for compact transfer
        k = int(mask.sum())
        packed = np.empty((k, X.shape[1] + 1), dtype=np.float32)
        packed[:, :-1] = X[mask]
        packed[:, -1]  = cls  # label stored as float32 for compactness; will cast later
        out[cls] = packed
    return seq, out

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    in_csv  = Path(args.input)
    out_csv = Path(args.output)
    global_p = Path(args.global_schema)
    assert in_csv.exists(), f"Missing input: {in_csv}"
    assert global_p.exists(), f"Missing schema: {global_p}"

    saved = json.loads(global_p.read_text(encoding="utf-8"))
    feature_cols = list(saved.get("feature_columns", [])) if isinstance(saved, dict) else list(saved)
    assert feature_cols, "feature_columns.json missing 'feature_columns'."
    n_feat = len(feature_cols)
    CAP = int(args.cap_per_class)
    CHUNK = max(50_000, int(args.chunksize))
    WORKERS = max(1, int(args.workers))
    BUFFER = max(1, int(args.buffer))
    SHUFFLE = bool(args.shuffle_final)

    if out_csv.exists():
        out_csv.unlink()

    # Per-class reservoir storage
    # We'll store features per class and labels separately for clean typing on write.
    res_X = {0: np.empty((CAP, n_feat), dtype=np.float32),
             1: np.empty((CAP, n_feat), dtype=np.float32),
             2: np.empty((CAP, n_feat), dtype=np.float32)}
    res_y = {0: np.empty((CAP,), dtype=np.int8),
             1: np.empty((CAP,), dtype=np.int8),
             2: np.empty((CAP,), dtype=np.int8)}
    seen  = {0: 0, 1: 0, 2: 0}   # total seen per class
    size  = {0: 0, 1: 0, 2: 0}   # current filled size per class

    # Only read what we need: features + meta to compute label
    usecols = feature_cols + ["orig_label", "is_adv"]
    dtype_map = {c: "float32" for c in feature_cols}
    dtype_map.update({"orig_label": "float32", "is_adv": "float32"})

    next_write = 0
    pending = {}

    def apply_reservoir_for_class(cls: int, block: np.ndarray):
        """block shape: (k, n_feat+1) with label in last column (as float32)"""
        if block is None or block.size == 0:
            return
        Xb = block[:, :-1]
        kb = Xb.shape[0]
        # First fill up to CAP in one vectorized shot
        to_fill = min(CAP - size[cls], kb)
        if to_fill > 0:
            start = size[cls]
            end = start + to_fill
            res_X[cls][start:end, :] = Xb[:to_fill, :]
            res_y[cls][start:end] = cls
            size[cls] += to_fill
            seen[cls] += to_fill
        # Handle overflow part with true reservoir sampling
        remaining = kb - to_fill
        if remaining <= 0:
            return
        # Process row-by-row for correct probabilities
        # t = current seen + i (1-based math handled with seen increment)
        idx = to_fill
        while idx < kb:
            seen[cls] += 1
            t = seen[cls]
            # keep with prob CAP / t
            if t <= CAP:
                # shouldn't happen because we filled to CAP already, but keep logic safe
                pos = t - 1
                res_X[cls][pos, :] = Xb[idx, :]
                res_y[cls][pos] = cls
                size[cls] = max(size[cls], pos + 1)
            else:
                if rng.random() < (CAP / t):
                    r = int(rng.integers(0, CAP))
                    res_X[cls][r, :] = Xb[idx, :]
                    res_y[cls][r] = cls
            idx += 1

    def drain_ready():
        nonlocal next_write
        while next_write in pending and pending[next_write].done():
            _, per_class = pending[next_write].result()
            for cls in (0, 1, 2):
                apply_reservoir_for_class(cls, per_class.get(cls))
            del pending[next_write]
            next_write += 1

    # Thread pool to transform chunks; main thread updates reservoirs (deterministic & safe)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        reader = pd.read_csv(in_csv, chunksize=CHUNK, usecols=usecols, dtype=dtype_map)
        for seq, chunk in enumerate(reader):
            fut = ex.submit(transform_chunk, seq, chunk, feature_cols)
            pending[seq] = fut
            if len(pending) >= BUFFER:
                drain_ready()
        # Drain remaining
        while pending:
            drain_ready()

    # Now we have up to CAP rows per class in reservoirs (uniform random)
    # If some classes saw fewer than CAP, 'size[cls]' < CAP; we just use that many.
    c0, c1, c2 = size[0], size[1], size[2]
    print(f"[RESERVOIR] sizes -> class0={c0:,}, class1={c1:,}, class2={c2:,}")

    # Build final combined table
    parts = []
    for cls in (0, 1, 2):
        if size[cls] == 0:
            continue
        df = pd.DataFrame(res_X[cls][:size[cls], :], columns=feature_cols)
        df["label"] = res_y[cls][:size[cls]].astype("int8")
        parts.append(df)

    if not parts:
        raise RuntimeError("No rows selected; check inputs.")

    out_df = pd.concat(parts, axis=0, ignore_index=True)

    # Optional shuffle to avoid class blocks (recommended for training)
    if SHUFFLE:
        out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Write final CSV
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[OK] Wrote {out_csv} rows={len(out_df):,}")
    print(f"Final per-class counts -> 0={c0:,}, 1={c1:,}, 2={c2:,}")
    if not SHUFFLE:
        print("Note: rows are grouped by class; enable --shuffle-final 1 to mix classes.")

if __name__ == "__main__":
    main()

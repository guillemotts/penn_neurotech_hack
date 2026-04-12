"""
cache_val.py -- Cache val-split predictions and ground truth for post-hoc sweeps.

Usage:
    python -m src.decoder.cache_val --mode easy
    python -m src.decoder.cache_val --mode hard
"""
import argparse
import json
import os
import pickle

import numpy as np

from decoder.data import (
    HARD_CFG, EASY_CFG,
    load_all, split_hard, split_easy_temporal,
    make_windows, BIN_SIZE, FS,
)
from decoder.ridge_train import scale_y, clip_pred


FEATURE_MODE = "mean_rms"
CONTEXT_BINS = 20


def main(args):
    mode = args.mode
    out_dir = f"outputs/{mode}_ridge_rms_ctx20"

    # ── 1. Reconstruct val split (identical to ridge_train.py) ────────
    if mode == "hard":
        cfg = HARD_CFG
        files = load_all(cfg)
        _, val_files = split_hard(files, cfg.val_seed)
        print(f"Hard mode: val seed(s) = {[f[0] for f in val_files]}")
    else:
        cfg = EASY_CFG
        files = load_all(cfg)
        assert len(files) == 1, f"Expected 1 easy file, got {len(files)}"
        seed, Xm, Xr, Yb = files[0]
        _, _, _, Xm_val, Xr_val, Yb_val = split_easy_temporal(
            Xm, Xr, Yb, val_frac=0.2)
        val_files = [(seed, Xm_val, Xr_val, Yb_val)]
        print(f"Easy mode: seed {seed}, val bins = {Xm_val.shape[0]}")

    # ── 2. Build windows ─────────────────────────────────────────────
    Xw_parts, Yw_parts = [], []
    for seed, Xb_mean, Xb_rms, Yb in val_files:
        Xw, Yw = make_windows(Xb_mean, Xb_rms, Yb,
                               context_bins=CONTEXT_BINS,
                               feature_mode=FEATURE_MODE)
        Xw_parts.append(Xw)
        Yw_parts.append(Yw)
    Xw_val = np.concatenate(Xw_parts, axis=0)
    Yw_val_raw = np.concatenate(Yw_parts, axis=0)

    # ── 3. Scale ground truth ────────────────────────────────────────
    Y_true = scale_y(Yw_val_raw)[:, :10].astype(np.float32)

    # ── 4. Load model and predict ────────────────────────────────────
    model_path = os.path.join(out_dir, "model.pkl")
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    print(f"Loaded model: {model_path}")

    pred_active = pipe.predict(Xw_val)  # (N, 10)
    pred_full = np.zeros((Xw_val.shape[0], 12), dtype=np.float64)
    pred_full[:, :10] = pred_active
    pred_full = clip_pred(pred_full)  # same clipping as ridge_train.py
    Y_pred = pred_full[:, :10].astype(np.float32)

    # ── 5. Save ──────────────────────────────────────────────────────
    assert Y_pred.shape == Y_true.shape, (
        f"Shape mismatch: Y_pred={Y_pred.shape}, Y_true={Y_true.shape}")
    assert Y_pred.dtype == np.float32
    assert Y_true.dtype == np.float32

    np.save(os.path.join(out_dir, "Y_pred.npy"), Y_pred)
    np.save(os.path.join(out_dir, "Y_true.npy"), Y_true)

    # Determine active channels from val_metrics.json
    metrics_path = os.path.join(out_dir, "val_metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)
    active_channels = metrics["active_channels"]

    meta = {
        "n_bins": int(Y_pred.shape[0]),
        "bin_size_samples": BIN_SIZE,
        "fs": FS,
        "bin_seconds": BIN_SIZE / FS,
        "active_channels": active_channels,
        "mode": mode,
    }
    meta_path = os.path.join(out_dir, "val_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved Y_pred.npy: {Y_pred.shape}, dtype={Y_pred.dtype}")
    print(f"Saved Y_true.npy: {Y_true.shape}, dtype={Y_true.dtype}")
    print(f"Saved val_meta.json: {meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["easy", "hard"], required=True)
    args = parser.parse_args()
    main(args)

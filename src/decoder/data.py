"""
data.py -- Streaming HDF5 loader, binning, windowing, and splits for decoder training.
"""
import re
import glob as glob_mod
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np

# ── Constants ────────────────────────────────────────────────────────
FS = 30000
BIN_SIZE = 300              # 10 ms at 30 kHz
CONTEXT_BINS = 20           # 200 ms
CHUNK_BINS = 256
DATASET_KEY = "acquisition/ElectricalSeries"


# ── Config ───────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    total_channels: int = 76
    encoded_channels: int = 64
    gt_channels: int = 12
    data_glob: str = ""
    exclude_seeds: list = field(default_factory=list)
    val_seed: Optional[int] = None


HARD_CFG = DataConfig(
    data_glob="data/newdata_apr12/smoke_Ani_seed*.h5",
    exclude_seeds=[6],
    val_seed=15,
)

EASY_CFG = DataConfig(
    data_glob="data/easy/smoke_Ani_seed*_easy.h5",
    exclude_seeds=[],
    val_seed=None,  # temporal split, one file only
)


# ── 1. bin_file ──────────────────────────────────────────────────────
def bin_file(h5_path, cfg):
    """
    Stream ElectricalSeries, bin to 10ms frames, split encoded/gt channels.
    Returns (Xb_mean: (T, C_enc) f32, Xb_rms: (T, C_enc) f32, Yb: (T, C_gt) f32).
    """
    chunk_values = CHUNK_BINS * BIN_SIZE * cfg.total_channels

    with h5py.File(h5_path, "r") as f:
        ds = f[DATASET_KEY]
        flat_len = ds.shape[0]
        assert flat_len % cfg.total_channels == 0, (
            f"flat_len={flat_len} not divisible by total_channels={cfg.total_channels}"
        )
        n_frames = flat_len // cfg.total_channels
        usable_frames = (n_frames // BIN_SIZE) * BIN_SIZE
        usable_values = usable_frames * cfg.total_channels

        xm_parts, xr_parts, y_parts = [], [], []
        for offset in range(0, usable_values, chunk_values):
            end = min(offset + chunk_values, usable_values)
            flat_chunk = ds[offset:end]
            frame_data = flat_chunk.reshape(-1, cfg.total_channels).astype(np.float32)
            n_local_frames = frame_data.shape[0]
            n_local_bins = n_local_frames // BIN_SIZE
            if n_local_bins == 0:
                continue
            trimmed = frame_data[:n_local_bins * BIN_SIZE]
            reshaped = trimmed.reshape(n_local_bins, BIN_SIZE, cfg.total_channels)
            binned_mean = reshaped.mean(axis=1)
            binned_rms = np.sqrt((reshaped ** 2).mean(axis=1))
            xm_parts.append(binned_mean[:, :cfg.encoded_channels])
            xr_parts.append(binned_rms[:, :cfg.encoded_channels])
            y_parts.append(binned_mean[:, cfg.encoded_channels:])

    Xb_mean = np.concatenate(xm_parts, axis=0)
    Xb_rms = np.concatenate(xr_parts, axis=0)
    Yb = np.concatenate(y_parts, axis=0)
    return Xb_mean, Xb_rms, Yb


# ── 2. make_windows ─────────────────────────────────────────────────
def _strided_windows(arr, context_bins):
    """Sliding windows via stride tricks on (T, C) array. Returns (N, context, C)."""
    T, C = arr.shape
    N = T - context_bins + 1
    stride_t, stride_c = arr.strides
    return np.lib.stride_tricks.as_strided(
        arr,
        shape=(N, context_bins, C),
        strides=(stride_t, stride_t, stride_c),
    ).copy()


def make_windows(Xb_mean, Xb_rms, Yb, context_bins=CONTEXT_BINS,
                 feature_mode="mean"):
    """
    Sliding windows with optional RMS features.
    feature_mode:
      "mean"     -> Xw shape (N, context*C_enc)         = (N, 1280)
      "mean_rms" -> Xw shape (N, context*C_enc*2)       = (N, 2560)
    Returns (Xw: (N, D), Yw: (N, C_gt)).
    """
    assert feature_mode in ("mean", "mean_rms"), f"Unknown feature_mode: {feature_mode}"
    T = Xb_mean.shape[0]
    N = T - context_bins + 1
    assert N > 0, f"Not enough bins ({T}) for context ({context_bins})"

    Xw_mean = _strided_windows(Xb_mean, context_bins)  # (N, ctx, C)
    Xw_flat = Xw_mean.reshape(N, -1)

    if feature_mode == "mean_rms":
        Xw_rms = _strided_windows(Xb_rms, context_bins)
        Xw_flat = np.concatenate([Xw_flat, Xw_rms.reshape(N, -1)], axis=1)

    Yw = Yb[context_bins - 1:].copy()
    return Xw_flat, Yw


# ── 3. load_all ──────────────────────────────────────────────────────
def load_all(cfg):
    """
    Glob h5 files, parse seed, filter, sort.
    Returns list of (seed, Xb_mean, Xb_rms, Yb).
    """
    paths = sorted(glob_mod.glob(cfg.data_glob))
    results = []
    for p in paths:
        m = re.search(r"seed(\d+)", p)
        if m is None:
            continue
        seed = int(m.group(1))
        if seed in cfg.exclude_seeds:
            continue
        Xb_mean, Xb_rms, Yb = bin_file(p, cfg)
        results.append((seed, Xb_mean, Xb_rms, Yb))
    results.sort(key=lambda x: x[0])
    assert len(results) > 0, f"No files found for glob={cfg.data_glob!r}"
    return results


# ── 4. split_hard ────────────────────────────────────────────────────
def split_hard(files, val_seed):
    """
    Seed-holdout split. Returns (train_list, val_list).
    Each element is (seed, Xb_mean, Xb_rms, Yb).
    """
    train_list = [f for f in files if f[0] != val_seed]
    val_list = [f for f in files if f[0] == val_seed]
    assert len(val_list) > 0, f"val_seed={val_seed} not found in files"
    assert len(train_list) > 0, "No train seeds remaining after holdout"
    return train_list, val_list


# ── 5. split_easy_temporal ───────────────────────────────────────────
def split_easy_temporal(Xb_mean, Xb_rms, Yb, val_frac=0.2):
    """
    Temporal split: last val_frac of time for validation.
    Returns (Xm_tr, Xr_tr, Yb_tr, Xm_val, Xr_val, Yb_val).
    """
    T = Xb_mean.shape[0]
    cut = int((1 - val_frac) * T)
    return (Xb_mean[:cut], Xb_rms[:cut], Yb[:cut],
            Xb_mean[cut:], Xb_rms[cut:], Yb[cut:])


# ── 6. compute_norm_stats ────────────────────────────────────────────
def compute_norm_stats(X_list, Y_list, predict_channels):
    """
    Compute normalization stats from pre-window frames.
    X_list: list of (T, D) arrays. Y_list: list of (T, C_gt) arrays.
    predict_channels: slice into C_gt dimension.
    Returns dict with x_mean, x_std, y_mean, y_std (all f32).
    """
    X_cat = np.concatenate(X_list, axis=0)
    Y_cat = np.concatenate(Y_list, axis=0)[:, predict_channels]

    x_mean = X_cat.mean(axis=0).astype(np.float32)
    x_std = X_cat.std(axis=0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0

    y_mean = Y_cat.mean(axis=0).astype(np.float32)
    y_std = Y_cat.std(axis=0).astype(np.float32)
    y_std[y_std < 1e-6] = 1.0

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

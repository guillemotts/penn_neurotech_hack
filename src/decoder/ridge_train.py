"""
ridge_train.py -- Ridge regression decoder, matching teammate's notebook approach.

Usage:
    python -m decoder.ridge_train --mode easy --out-dir outputs/easy_ridge
    python -m decoder.ridge_train --mode hard --feature-mode mean_rms --out-dir outputs/hard_ridge_rms
    python -m decoder.ridge_train --mode hard --feature-mode mean_rms --context-bins 40 --out-dir outputs/hard_ridge_rms_ctx40
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decoder.data import (
    HARD_CFG, EASY_CFG, CONTEXT_BINS,
    load_all, split_hard, split_easy_temporal,
    make_windows, compute_norm_stats,
)

ALPHAS = [3, 10, 30, 100]


# ── Scaling ──────────────────────────────────────────────────────────
def scale_y(Y_raw):
    """Scale raw GT to controller range. Returns (N, 12) float32."""
    Y_s = np.zeros_like(Y_raw)
    Y_s[:, :4] = np.clip(Y_raw[:, :4] / 32767.0, -1, 1)
    Y_s[:, 4:10] = np.clip(Y_raw[:, 4:10] / 32767.0, 0, 1)
    Y_s[:, 10:12] = 0.0
    return Y_s


def clip_pred(Y_pred):
    """Clip predictions to valid controller ranges."""
    Y_c = Y_pred.copy()
    Y_c[:, :4] = np.clip(Y_c[:, :4], -1, 1)
    Y_c[:, 4:10] = np.clip(Y_c[:, 4:10], 0, 1)
    Y_c[:, 10:12] = 0.0
    return Y_c


# ── Sample weights ───────────────────────────────────────────────────
def build_sample_weights(Y_scaled):
    """Per-sample weights based on controller activity."""
    analog_active = np.abs(Y_scaled[:, :4]).max(axis=1) > 0.10
    sparse_active = Y_scaled[:, 4:10].max(axis=1) > 0.05
    W = 1.0 + 1.0 * analog_active.astype(np.float64) + 2.0 * sparse_active.astype(np.float64)
    print(f"  Sample weights: analog_active={analog_active.mean():.4f}, "
          f"sparse_active={sparse_active.mean():.4f}")
    return W


# ── Pearson correlation ──────────────────────────────────────────────
def pearson_corr(pred, target):
    """Per-channel Pearson correlation. Returns (C,) array."""
    C = pred.shape[1]
    corrs = np.zeros(C, dtype=np.float64)
    for c in range(C):
        p, t = pred[:, c], target[:, c]
        if t.std() < 1e-8 or p.std() < 1e-8:
            corrs[c] = 0.0
        else:
            corrs[c] = np.corrcoef(p, t)[0, 1]
    return corrs


# ── Proxy bitrate ────────────────────────────────────────────────────
def proxy_bitrate(Y_true_scaled, Y_pred_scaled, N=8, mag_thresh=0.3, window_sec=0.2):
    """
    Proxy ITR from stick channels (0, 1) mapped to N angular symbols.
    Returns (bitrate, n_correct, n_incorrect, total_time_s).
    """
    def to_symbol(y):
        x, yv = y[:, 0], y[:, 1]
        mag = np.sqrt(x**2 + yv**2)
        angle = np.arctan2(yv, x)
        sym = ((angle + np.pi) / (2 * np.pi) * N).astype(int) % N
        sym[mag < mag_thresh] = -1
        return sym

    s_true = to_symbol(Y_true_scaled)
    s_pred = to_symbol(Y_pred_scaled)

    mask = s_true >= 0
    pred_emit = s_pred >= 0
    correct = (s_true == s_pred) & mask & pred_emit
    incorrect = (s_true != s_pred) & mask & pred_emit
    Sc, Si = int(correct.sum()), int(incorrect.sum())
    t = float(mask.sum()) * window_sec
    if t == 0:
        return 0.0, 0, 0, 0.0
    bitrate = float(np.log2(N - 1) * max(Sc - Si, 0) / t)
    return bitrate, Sc, Si, t


# ── Windows helper ───────────────────────────────────────────────────
def build_windows(file_list, feature_mode, context_bins):
    """Build flattened Xw and raw Yw from file list."""
    Xw_parts, Yw_parts = [], []
    for seed, Xb_mean, Xb_rms, Yb in file_list:
        Xw, Yw = make_windows(Xb_mean, Xb_rms, Yb,
                              context_bins=context_bins,
                              feature_mode=feature_mode)
        Xw_parts.append(Xw)
        Yw_parts.append(Yw)
    return np.concatenate(Xw_parts, axis=0), np.concatenate(Yw_parts, axis=0)


# ── Main ─────────────────────────────────────────────────────────────
def train(args):
    t_start = time.time()
    feature_mode = args.feature_mode
    context_bins = args.context_bins

    print(f"context_bins={context_bins}, feature_mode={feature_mode}")

    # ── 1. Load ──────────────────────────────────────────────────────
    if args.mode == "hard":
        cfg = HARD_CFG
        print(f"Loading hard mode: {cfg.data_glob}")
        files = load_all(cfg)
        train_files, val_files = split_hard(files, cfg.val_seed)
        print(f"  Train seeds: {[f[0] for f in train_files]}")
        print(f"  Val seeds:   {[f[0] for f in val_files]}")
    else:
        cfg = EASY_CFG
        print(f"Loading easy mode: {cfg.data_glob}")
        files = load_all(cfg)
        assert len(files) == 1, f"Expected 1 easy file, got {len(files)}"
        seed, Xm, Xr, Yb = files[0]
        Xm_tr, Xr_tr, Yb_tr, Xm_val, Xr_val, Yb_val = split_easy_temporal(
            Xm, Xr, Yb, val_frac=0.2)
        train_files = [(seed, Xm_tr, Xr_tr, Yb_tr)]
        val_files = [(seed, Xm_val, Xr_val, Yb_val)]
        print(f"  Seed {seed}: train={Xm_tr.shape[0]}, val={Xm_val.shape[0]} bins")

    # ── 2. Build windows ─────────────────────────────────────────────
    print(f"Building windows (feature_mode={feature_mode}) ...")
    Xw_tr, Yw_tr_raw = build_windows(train_files, feature_mode, context_bins)
    Xw_val, Yw_val_raw = build_windows(val_files, feature_mode, context_bins)
    print(f"  Train: {Xw_tr.shape} windows, Val: {Xw_val.shape} windows")

    # ── 3. Scale Y ───────────────────────────────────────────────────
    Ys_tr = scale_y(Yw_tr_raw)
    Ys_val = scale_y(Yw_val_raw)

    # ── 4. Sample weights ────────────────────────────────────────────
    print("Computing sample weights (train) ...")
    W_tr = build_sample_weights(Ys_tr)

    # ── 5. Norm stats (for saving) ───────────────────────────────────
    X_list = [xm for _, xm, _, _ in train_files]
    Y_list = [yb for _, _, _, yb in train_files]
    norm_stats = compute_norm_stats(X_list, Y_list, predict_channels=slice(0, 12))

    # ── 6. Sweep alpha ───────────────────────────────────────────────
    print(f"\nSweeping alpha in {ALPHAS} ...")
    best_alpha = None
    best_mean_corr = -1.0
    best_pipe = None
    best_pred = None

    fit_channels = slice(0, 10)

    for alpha in ALPHAS:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        pipe.fit(Xw_tr, Ys_tr[:, fit_channels], **{"ridge__sample_weight": W_tr})

        pred_active = pipe.predict(Xw_val)
        pred_full = np.zeros((Xw_val.shape[0], 12), dtype=np.float64)
        pred_full[:, fit_channels] = pred_active
        pred_full = clip_pred(pred_full)

        corrs = pearson_corr(pred_full, Ys_val)
        y_val_std = Ys_val.std(axis=0)
        active_mask = y_val_std > 1e-4
        mean_corr = float(corrs[active_mask].mean()) if active_mask.any() else 0.0

        print(f"  alpha={alpha:>4d}: mean_corr(active)={mean_corr:.4f}  "
              f"corrs={np.array2string(corrs[:10], precision=4, separator=', ')}")

        if mean_corr > best_mean_corr:
            best_mean_corr = mean_corr
            best_alpha = alpha
            best_pipe = pipe
            best_pred = pred_full

    print(f"\n  Best alpha: {best_alpha} (mean_corr={best_mean_corr:.4f})")

    # ── 7. Final eval ────────────────────────────────────────────────
    corrs = pearson_corr(best_pred, Ys_val)
    mae = np.abs(best_pred - Ys_val).mean(axis=0)
    y_val_std = Ys_val.std(axis=0)
    active_mask = y_val_std > 1e-4

    print(f"\nPer-channel Pearson corr (val, best alpha={best_alpha}):")
    for c in range(12):
        tag = "ACTIVE" if active_mask[c] else "idle"
        print(f"  ch {c:2d}: corr={corrs[c]:+.4f}  mae={mae[c]:.4f}  "
              f"y_std={y_val_std[c]:.4f}  ({tag})")

    mean_corr_active = float(corrs[active_mask].mean()) if active_mask.any() else 0.0
    print(f"\n  Mean corr (active): {mean_corr_active:.4f}")

    # ── 8. Proxy bitrate ─────────────────────────────────────────────
    print(f"\nProxy bitrate table (stick channels 0,1):")
    print(f"  {'N':>3s}  {'mag_thresh':>10s}  {'bitrate':>8s}  {'correct':>8s}  "
          f"{'incorrect':>10s}  {'time_s':>8s}")
    print(f"  {'---':>3s}  {'----------':>10s}  {'--------':>8s}  {'--------':>8s}  "
          f"{'----------':>10s}  {'--------':>8s}")

    bitrate_results = {}
    for N in [4, 8]:
        for mt in [0.2, 0.3, 0.4]:
            br, sc, si, t = proxy_bitrate(Ys_val, best_pred, N=N, mag_thresh=mt)
            key = f"N{N}_mt{mt}"
            bitrate_results[key] = {"bitrate": br, "correct": sc, "incorrect": si, "time_s": t}
            print(f"  {N:>3d}  {mt:>10.1f}  {br:>8.4f}  {sc:>8d}  {si:>10d}  {t:>8.1f}")

    # ── 9. Save ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    model_path = os.path.join(args.out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_pipe, f)
    print(f"\n  Saved model: {model_path}")

    stats_path = os.path.join(args.out_dir, "norm_stats.npz")
    np.savez(stats_path, **norm_stats)
    print(f"  Saved norm stats: {stats_path}")

    metrics = {
        "best_alpha": best_alpha,
        "feature_mode": feature_mode,
        "context_bins": context_bins,
        "per_channel_corr": corrs.tolist(),
        "per_channel_mae": mae.tolist(),
        "mean_corr_active": mean_corr_active,
        "active_channels": np.where(active_mask)[0].tolist(),
        "bitrate": bitrate_results,
    }
    metrics_path = os.path.join(args.out_dir, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics: {metrics_path}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["easy", "hard"], required=True)
    parser.add_argument("--feature-mode", choices=["mean", "mean_rms", "mean_logrms"],
                        default="mean_rms")
    parser.add_argument("--context-bins", type=int, default=CONTEXT_BINS)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    train(args)

"""
export_onnx.py -- Retrain Ridge on ALL data (no val holdout) and export to ONNX.

Usage:
    python src/decoder/export_onnx.py --mode easy --out-dir outputs/easy_ridge_final
    python src/decoder/export_onnx.py --mode hard --out-dir outputs/hard_ridge_final
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from decoder.data import (
    HARD_CFG, EASY_CFG,
    load_all, make_windows, compute_norm_stats,
)
from decoder.ridge_train import scale_y, build_sample_weights, build_windows

ALPHA = 100
CONTEXT_BINS = 20
FEATURE_MODE = "mean_logrms"
FIT_CHANNELS = slice(0, 10)


# ── ONNX wrapper ────────────────────────────────────────────────────
class RidgeDecoderWrapper(nn.Module):
    """Torch wrapper around sklearn StandardScaler + Ridge for ONNX export."""

    def __init__(self, scaler, ridge):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(scaler.mean_.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scaler.scale_.astype(np.float32)))
        self.register_buffer("coef", torch.from_numpy(ridge.coef_.astype(np.float32)))
        self.register_buffer("bias", torch.from_numpy(ridge.intercept_.astype(np.float32)))

    def forward(self, x):  # x: [B, 2560]
        x = (x - self.mean) / self.scale            # StandardScaler
        active = x @ self.coef.T + self.bias         # Ridge linear: [B, 10]
        active = torch.cat([
            torch.clamp(active[:, :4], -1.0, 1.0),   # analog sticks
            torch.clamp(active[:, 4:10], 0.0, 1.0),  # triggers/buttons
        ], dim=1)
        raw = active * 32767.0                        # scale to raw GT units
        zeros = torch.zeros(x.shape[0], 2, device=x.device)  # gt_10, gt_11
        return torch.cat([raw, zeros], dim=1)         # [B, 12]


# ── Main ────────────────────────────────────────────────────────────
def main(args):
    t_start = time.time()
    mode = args.mode
    out_dir = args.out_dir
    feature_mode = args.feature_mode
    cfg = HARD_CFG if mode == "hard" else EASY_CFG

    print(f"{'='*60}")
    print(f"  RETRAIN-ALL + ONNX EXPORT  |  mode={mode}")
    print(f"  alpha={ALPHA}, context_bins={CONTEXT_BINS}, feature_mode={feature_mode}")
    print(f"{'='*60}\n")

    # ── Phase A: Retrain on ALL data ────────────────────────────────
    print("[Phase A] Loading ALL data (no train/val split) ...")
    files = load_all(cfg)
    seeds = [f[0] for f in files]
    print(f"  Seeds loaded: {seeds} ({len(files)} files)")

    print(f"  Building windows ...")
    Xw_all, Yw_all_raw = build_windows(files, feature_mode, CONTEXT_BINS)
    print(f"  Windows: {Xw_all.shape}  (features={Xw_all.shape[1]})")

    Ys_all = scale_y(Yw_all_raw)
    W_all = build_sample_weights(Ys_all)

    # Norm stats on full dataset
    X_list = [xm for _, xm, _, _ in files]
    Y_list = [yb for _, _, _, yb in files]
    norm_stats = compute_norm_stats(X_list, Y_list, predict_channels=slice(0, 12))

    print(f"\n  Fitting Pipeline(StandardScaler + Ridge(alpha={ALPHA})) on {Xw_all.shape[0]} samples ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])
    pipe.fit(Xw_all, Ys_all[:, FIT_CHANNELS], **{"ridge__sample_weight": W_all})
    print(f"  Done fitting.")

    # Save sklearn model
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Saved: {model_path}")

    stats_path = os.path.join(out_dir, "norm_stats.npz")
    np.savez(stats_path, **norm_stats)
    print(f"  Saved: {stats_path}")

    # ── Phase B: ONNX export ────────────────────────────────────────
    print(f"\n[Phase B] ONNX export ...")
    scaler = pipe.named_steps["scaler"]
    ridge = pipe.named_steps["ridge"]

    wrapper = RidgeDecoderWrapper(scaler, ridge).eval()
    feature_dim = Xw_all.shape[1]
    dummy_input = torch.randn(1, feature_dim)

    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(
        wrapper, dummy_input, onnx_path,
        input_names=["features"],
        output_names=["controller"],
        dynamic_axes={
            "features": {0: "batch"},
            "controller": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"  Saved: {onnx_path}  input=(B, {feature_dim})  output=(B, 12)")

    # ── Phase C: Metadata ───────────────────────────────────────────
    print(f"\n[Phase C] Saving metadata ...")
    meta = {
        "sample_rate_hz": 30000,
        "bin_size_raw_frames": 300,
        "context_bins": CONTEXT_BINS,
        "feature_mode": feature_mode,
        "best_alpha": ALPHA,
        "n_train_samples": int(Xw_all.shape[0]),
        "n_seeds": len(files),
        "seeds_used": seeds,
        "onnx_input_shape": [1, feature_dim],
        "onnx_output_shape": [1, 12],
        "synapse_app_parameters": {
            "input_channels": 64,
            "frames_per_step": 300,
            "feature_window": CONTEXT_BINS,
            "feature_mode": FEATURE_MODE,
            "model_name": "model",
        },
    }
    meta_path = os.path.join(out_dir, "decoder_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    # ── Phase D: ONNX sanity check ──────────────────────────────────
    print(f"\n[Phase D] ONNX roundtrip sanity check ...")
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(5, feature_dim).astype(np.float32)

    onnx_out = sess.run(None, {"features": test_input})[0]
    torch_out = wrapper(torch.from_numpy(test_input)).detach().numpy()

    max_diff = float(np.abs(onnx_out - torch_out).max())
    # Tolerance: the 32767x output scaling amplifies float32 matmul rounding.
    # A diff of ~0.05 in raw units is < 2e-6 in controller-scale, totally fine.
    tol = 0.1
    print(f"  ONNX shape: {onnx_out.shape}")
    print(f"  max_abs_diff: {max_diff:.2e}  (tol={tol})")
    assert max_diff < tol, f"ONNX roundtrip failed! max_diff={max_diff}"
    print(f"  ROUNDTRIP OK")

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  SUMMARY ({mode} mode)")
    print(f"  Seeds: {seeds}")
    print(f"  Samples: {Xw_all.shape[0]}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Alpha: {ALPHA}")
    print(f"  Output dir: {out_dir}")
    print(f"  Files: model.pkl, norm_stats.npz, model.onnx, decoder_meta.json")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    # ── RMS compatibility warning ───────────────────────────────────
    if feature_mode in ("mean_rms", "mean_logrms"):
        print(f"\n{'!'*60}")
        print(f"  WARNING: This model uses feature_mode='{feature_mode}' ({feature_dim} inputs).")
        print(f"  The Synapse app may expect mean-only features (1280 inputs).")
        print(f"  If the app cannot compute RMS features cheaply, you will need")
        print(f"  a mean-only fallback ONNX (retrain with feature_mode='mean').")
        if feature_mode == "mean_logrms":
            print(f"  NOTE: The RMS half requires np.log1p transform at inference time.")
        print(f"{'!'*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain Ridge on all data + ONNX export")
    parser.add_argument("--mode", choices=["easy", "hard"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-mode", choices=["mean", "mean_rms", "mean_logrms"],
                        default=FEATURE_MODE)
    args = parser.parse_args()
    main(args)

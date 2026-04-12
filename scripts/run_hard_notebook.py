"""
Local runner: mirrors Colab Files/Decoder_hard.ipynb end-to-end using local hard data.
Hard data lives in two directories:
  - data/hard/          -> hard-ksenija-smoke-{1,2,3}.h5
  - data/newdata_apr12/ -> smoke_Ani_seed{7..15}.h5  (seed6 excluded)
"""
import os, gc, glob, json, time, random

import h5py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pickle
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Constants (matching notebook) ──
DATA_DIRS = ["data/hard", "data/newdata_apr12"]
DATASET_KEY = "acquisition/ElectricalSeries"
TOTAL_CHANNELS = 76
ENCODED_CHANNELS = 64
GT_CHANNELS = 12
FS = 32000
BIN_SIZE = 320           # 10 ms at 32 kHz
CONTEXT_BINS = 20        # 200 ms context
CHUNK_BINS = 256
FEATURE_MODE = "mean_rms"
CACHE_DIR = "outputs/hard_cache"
OUT_DIR = "outputs/hard_ridge_final"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_DIM = CONTEXT_BINS * ENCODED_CHANNELS * (2 if FEATURE_MODE == "mean_rms" else 1)

# ── Glob both Ani and ksenija hard files, exclude seed6 ──
FILES = sorted(
    glob.glob(os.path.join(DATA_DIRS[1], "smoke_Ani_seed*.h5")) +
    glob.glob(os.path.join(DATA_DIRS[0], "hard-ksenija-smoke-*.h5"))
)
FILES = [f for f in FILES if "seed6" not in f]
assert len(FILES) > 0, f"No .h5 files found in {DATA_DIRS}"

# ── Verify every file ──
print("=" * 70)
print("File verification:")
for fpath in FILES:
    with h5py.File(fpath, "r") as f:
        ds = f[DATASET_KEY]
        flat_len = int(ds.shape[0])
        assert flat_len % TOTAL_CHANNELS == 0, (
            f"{os.path.basename(fpath)}: flat_len={flat_len} not divisible by {TOTAL_CHANNELS}"
        )
        n_frames = flat_len // TOTAL_CHANNELS
        print(f"  {os.path.basename(fpath):40s}  n_frames={n_frames:>10d}")

print(f"\nFeature mode: {FEATURE_MODE}, feature dim: {FEATURE_DIM}")
print(f"Total files: {len(FILES)}")
print("=" * 70)


# ── bin_file_to_arrays ──
def bin_file_to_arrays(h5_path):
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    cache_path = os.path.join(CACHE_DIR, f"{stem}_bin{BIN_SIZE}_v2.npz")

    if os.path.exists(cache_path):
        z = np.load(cache_path)
        Xb_mean = z["X_mean"].astype(np.float32)
        Xb_rms = z["X_rms"].astype(np.float32)
        Yb = z["Y"].astype(np.float32)
        print(f"[cache] {stem}: X_mean={Xb_mean.shape}, X_rms={Xb_rms.shape}, Y={Yb.shape}")
        return Xb_mean, Xb_rms, Yb

    Xm_parts, Xr_parts, Y_parts = [], [], []
    with h5py.File(h5_path, "r") as f:
        dset = f[DATASET_KEY]
        flat_len = int(dset.shape[0])
        n_frames = flat_len // TOTAL_CHANNELS
        usable_frames = (n_frames // BIN_SIZE) * BIN_SIZE
        usable_values = usable_frames * TOTAL_CHANNELS
        values_per_chunk = CHUNK_BINS * BIN_SIZE * TOTAL_CHANNELS
        print(f"[read ] {stem}: n_frames={n_frames}, usable_bins={usable_frames // BIN_SIZE}")

        for start in range(0, usable_values, values_per_chunk):
            end = min(start + values_per_chunk, usable_values)
            flat = np.asarray(dset[start:end], dtype=np.float32)
            frames = flat.reshape(-1, TOTAL_CHANNELS)
            reshaped = frames.reshape(-1, BIN_SIZE, TOTAL_CHANNELS)
            binned_mean = reshaped.mean(axis=1)
            binned_rms = np.sqrt((reshaped ** 2).mean(axis=1))
            Xm_parts.append(binned_mean[:, :ENCODED_CHANNELS].astype(np.float32))
            Xr_parts.append(binned_rms[:, :ENCODED_CHANNELS].astype(np.float32))
            Y_parts.append(binned_mean[:, ENCODED_CHANNELS:].astype(np.float32))
            del flat, frames, reshaped, binned_mean, binned_rms
            gc.collect()

    Xb_mean = np.concatenate(Xm_parts, axis=0).astype(np.float32)
    Xb_rms = np.concatenate(Xr_parts, axis=0).astype(np.float32)
    Yb = np.concatenate(Y_parts, axis=0).astype(np.float32)
    np.savez(cache_path, X_mean=Xb_mean, X_rms=Xb_rms, Y=Yb)
    print(f"[saved] {stem}: X_mean={Xb_mean.shape}, X_rms={Xb_rms.shape}, Y={Yb.shape}")
    return Xb_mean, Xb_rms, Yb


print("\nLoading and binning files...")
binned_files = []
for path in FILES:
    Xb_mean, Xb_rms, Yb = bin_file_to_arrays(path)
    binned_files.append((path, Xb_mean, Xb_rms, Yb))

print("\nSummary:")
for path, Xb_mean, Xb_rms, Yb in binned_files:
    print(f"  {os.path.basename(path):40s}  X_mean={Xb_mean.shape}  Y={Yb.shape}")


# ── make_windows ──
def make_windows(Xb_mean, Xb_rms, Yb):
    T, C = Xb_mean.shape
    N = T - CONTEXT_BINS + 1
    assert N > 0

    def _strided_windows(arr):
        shape = (N, CONTEXT_BINS, C)
        strides = (arr.strides[0], arr.strides[0], arr.strides[1])
        return np.lib.stride_tricks.as_strided(
            arr, shape=shape, strides=strides
        ).copy().reshape(N, CONTEXT_BINS * C).astype(np.float32)

    Xw = _strided_windows(Xb_mean)
    if FEATURE_MODE == "mean_rms":
        Xw = np.concatenate([Xw, _strided_windows(Xb_rms)], axis=1)

    Y = Yb[CONTEXT_BINS - 1:, :10].astype(np.float32)
    Y_scaled = np.empty_like(Y, dtype=np.float32)
    Y_scaled[:, :4] = np.clip(Y[:, :4] / 32767.0, -1.0, 1.0)
    Y_scaled[:, 4:10] = np.clip(Y[:, 4:10] / 32767.0, 0.0, 1.0)
    return Xw, Y_scaled


def make_sample_weights(Y_scaled):
    w = np.ones(len(Y_scaled), dtype=np.float32)
    w += 1.0 * (np.max(np.abs(Y_scaled[:, :4]), axis=1) > 0.10).astype(np.float32)
    w += 2.0 * (np.max(Y_scaled[:, 4:10], axis=1) > 0.05).astype(np.float32)
    return w


# ── Train/val split: last file holdout ──
train_indices = list(range(len(binned_files) - 1))
val_indices = [len(binned_files) - 1]

print("\nTrain files:")
for i in train_indices:
    print(f"  {os.path.basename(binned_files[i][0])}")
print("Val file:")
for i in val_indices:
    print(f"  {os.path.basename(binned_files[i][0])}")


def build_split(indices):
    X_list, Y_list, W_list = [], [], []
    for i in indices:
        _, Xb_mean, Xb_rms, Yb = binned_files[i]
        Xw, Yw = make_windows(Xb_mean, Xb_rms, Yb)
        W = make_sample_weights(Yw)
        X_list.append(Xw); Y_list.append(Yw); W_list.append(W)
        del Xw, Yw, W; gc.collect()
    return (np.concatenate(X_list, axis=0).astype(np.float32),
            np.concatenate(Y_list, axis=0).astype(np.float32),
            np.concatenate(W_list, axis=0).astype(np.float32))


X_train, Y_train, W_train = build_split(train_indices)
X_val, Y_val, W_val = build_split(val_indices)
print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}")


# ── Helpers ──
def safe_corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_predictions(y_true, y_pred):
    corrs = [safe_corr(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    maes = np.mean(np.abs(y_true - y_pred), axis=0)
    valid = np.array([not np.isnan(c) for c in corrs], dtype=bool)
    mean_corr = float(np.mean(np.array(corrs)[valid])) if valid.any() else np.nan
    return {"corrs": corrs, "mean_corr": mean_corr, "maes": maes}


# ── Alpha sweep ──
print("\n" + "=" * 70)
print("Alpha sweep [3, 10, 30, 100]")
print("=" * 70)

alphas = [3.0, 10.0, 30.0, 100.0]
results = []
for alpha in alphas:
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    t0 = time.time()
    pipe.fit(X_train, Y_train, ridge__sample_weight=W_train)
    y_val_pred = pipe.predict(X_val).astype(np.float32)
    y_val_pred[:, :4] = np.clip(y_val_pred[:, :4], -1.0, 1.0)
    y_val_pred[:, 4:10] = np.clip(y_val_pred[:, 4:10], 0.0, 1.0)
    metrics = evaluate_predictions(Y_val, y_val_pred)
    dt = time.time() - t0
    results.append((alpha, metrics["mean_corr"], metrics["corrs"], metrics["maes"], y_val_pred))
    print(f"  alpha={alpha:6.1f} | val_mean_corr={metrics['mean_corr']:.4f} | time={dt:.1f}s")

best_idx = max(range(len(results)), key=lambda i: results[i][1])
best_alpha = results[best_idx][0]
best_val_pred = results[best_idx][4]
print(f"\nBest alpha: {best_alpha}")

print(f"\nPer-channel corrs (best alpha={best_alpha}):")
_, _, corrs, maes, _ = results[best_idx]
for i, c in enumerate(corrs):
    print(f"  gt_{i}: corr={c:.4f}  mae={maes[i]:.4f}")


# ── Proxy bitrate table ──
def proxy_bitrate(Y_true_scaled, Y_pred_scaled, N=8, mag_thresh=0.3, window_sec=0.2):
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
    Sc = int(((s_true == s_pred) & mask & pred_emit).sum())
    Si = int(((s_true != s_pred) & mask & pred_emit).sum())
    t = float(mask.sum()) * window_sec
    if t == 0:
        return 0.0
    return float(np.log2(max(N - 1, 1)) * max(Sc - Si, 0) / t)


print(f"\n{'='*70}")
print("Proxy bitrate table (stick channels 0,1 on val set)")
print(f"{'='*70}")
print(f"{'N':>3s}  {'mag_thresh':>10s}  {'bitrate':>8s}")
print(f"{'---':>3s}  {'----------':>10s}  {'--------':>8s}")
for N_sym in [4, 6, 8]:
    for mt in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        br = proxy_bitrate(Y_val, best_val_pred, N=N_sym, mag_thresh=mt)
        print(f"{N_sym:>3d}  {mt:>10.2f}  {br:>8.4f}")


# ── Retrain all + ONNX export ──
print(f"\n{'='*70}")
print("Retrain on all data + ONNX export")
print(f"{'='*70}")
del X_train, Y_train, W_train, X_val, Y_val, W_val, best_val_pred
gc.collect()

X_all, Y_all, W_all = build_split(list(range(len(binned_files))))
print(f"X_all: {X_all.shape}")

final_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])
t0 = time.time()
final_pipe.fit(X_all, Y_all, ridge__sample_weight=W_all)
print(f"Final fit: {time.time() - t0:.1f}s")

y_all_pred = final_pipe.predict(X_all).astype(np.float32)
y_all_pred[:, :4] = np.clip(y_all_pred[:, :4], -1.0, 1.0)
y_all_pred[:, 4:10] = np.clip(y_all_pred[:, 4:10], 0.0, 1.0)
tm = evaluate_predictions(Y_all, y_all_pred)
print(f"Train mean corr: {tm['mean_corr']:.4f}")

# ONNX wrapper
scaler = final_pipe.named_steps["scaler"]
ridge = final_pipe.named_steps["ridge"]


class RidgeDecoderWrapper(nn.Module):
    def __init__(self, scaler, ridge):
        super().__init__()
        x_mean = scaler.mean_.astype(np.float32)
        x_scale = scaler.scale_.astype(np.float32)
        x_scale = np.where(x_scale < 1e-8, 1.0, x_scale).astype(np.float32)
        self.register_buffer("x_mean", torch.tensor(x_mean).view(1, -1))
        self.register_buffer("x_scale", torch.tensor(x_scale).view(1, -1))
        self.register_buffer("W", torch.tensor(ridge.coef_.astype(np.float32)))
        self.register_buffer("b", torch.tensor(ridge.intercept_.astype(np.float32)).view(1, -1))

    def forward(self, x):
        x = (x - self.x_mean) / self.x_scale
        y10 = x @ self.W.T + self.b
        y_signed = torch.clamp(y10[:, :4], -1.0, 1.0)
        y_pos = torch.clamp(y10[:, 4:10], 0.0, 1.0)
        active = torch.cat([y_signed, y_pos], dim=1)
        raw = active * 32767.0
        zeros = torch.zeros((x.shape[0], 2), dtype=x.dtype, device=x.device)
        return torch.cat([raw, zeros], dim=1)


wrapper = RidgeDecoderWrapper(scaler, ridge).eval()
onnx_path = os.path.join(OUT_DIR, "model.onnx")
dummy = torch.randn(1, FEATURE_DIM)
torch.onnx.export(wrapper, dummy, onnx_path,
                   input_names=["features"], output_names=["controller"],
                   dynamic_axes={"features": {0: "batch"}, "controller": {0: "batch"}},
                   opset_version=17)
print(f"Saved ONNX: {onnx_path}")

# Roundtrip check
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
x_test = np.random.randn(4, FEATURE_DIM).astype(np.float32)
onnx_out = sess.run(None, {"features": x_test})[0]
torch_out = wrapper(torch.from_numpy(x_test)).detach().numpy()
max_diff = np.max(np.abs(onnx_out - torch_out))
print(f"ONNX roundtrip max_abs_diff: {max_diff:.2e}")
assert max_diff < 0.1

# ── Save model.pkl ──
pkl_path = os.path.join(OUT_DIR, "model.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(final_pipe, f)
print(f"Saved model.pkl: {pkl_path}")

# ── Save norm_stats.npz ──
norm_path = os.path.join(OUT_DIR, "norm_stats.npz")
scaler_obj = final_pipe.named_steps["scaler"]
y_mean = np.mean(Y_all, axis=0).astype(np.float32)
y_std = np.std(Y_all, axis=0).astype(np.float32)
y_std[y_std < 1e-8] = 1.0
np.savez(norm_path,
         x_mean=scaler_obj.mean_.astype(np.float32),
         x_std=scaler_obj.scale_.astype(np.float32),
         y_mean=y_mean,
         y_std=y_std)
print(f"Saved norm_stats.npz: {norm_path}")

# ── IR version patch ──
import onnx as _onnx
m = _onnx.load(onnx_path)
m.ir_version = 9
ir9_path = os.path.join(OUT_DIR, "model_ir9.onnx")
_onnx.save(m, ir9_path)
print(f"Saved IR9-patched: {ir9_path}")

# Save metadata
meta = {
    "sample_rate_hz": FS,
    "bin_size_raw_frames": BIN_SIZE,
    "context_bins": CONTEXT_BINS,
    "feature_mode": FEATURE_MODE,
    "best_alpha": float(best_alpha),
    "n_train_samples": int(X_all.shape[0]),
    "n_files": len(FILES),
    "files_used": [os.path.basename(f) for f in FILES],
    "onnx_input_shape": [1, FEATURE_DIM],
    "onnx_output_shape": [1, GT_CHANNELS],
    "synapse_app_parameters": {
        "input_channels": ENCODED_CHANNELS,
        "frames_per_step": BIN_SIZE,
        "feature_window": CONTEXT_BINS,
        "feature_mode": FEATURE_MODE,
        "model_name": "model",
    },
}
meta_path = os.path.join(OUT_DIR, "decoder_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved metadata: {meta_path}")

print("\nDone.")

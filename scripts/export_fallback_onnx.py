"""
export_fallback_onnx.py -- Train 4-cardinal ljoy probe on 80% data, export ONNX.

ONNX graph: input (1,128,8) -> FeatureStack z-score -> flatten -> sklearn scaler
            -> linear -> log_softmax -> scatter into (1,14).
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")

from load_h5_session import align_and_build_session, extract_windows
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# ── 0. Load & pool sessions one at a time (exclude seed 6) ───────────
data_dir = Path(__file__).resolve().parent.parent / "data" / "newdata_apr12"
seeds = [s for s in range(1, 16) if s != 6]

all_X, all_y = [], []
for s in seeds:
    h5 = data_dir / "smoke_Ani_seed{}.h5".format(s)
    cj = data_dir / "cues_Ani_seed{}.json".format(s)
    if not h5.exists() or not cj.exists():
        continue
    print("Loading seed {} ...".format(s))
    try:
        sess = align_and_build_session(h5, cj, n_channels=76)
        X, y, kept = extract_windows(sess)
        X = X[:, :64, :]  # neural channels only
        all_X.append(X)
        all_y.append(y)
        print("  X={}  y={}  kept={}".format(X.shape, y.shape, len(kept)))
        del sess, X, y, kept
        gc.collect()
    except Exception as e:
        print("  ERROR: {}".format(e))
        continue

if not all_X:
    print("NO DATA")
    sys.exit(1)

X_all = np.concatenate(all_X)
y_all = np.concatenate(all_y)
del all_X, all_y
gc.collect()
print("\npooled raw: X={} unique_labels={}".format(X_all.shape, sorted(set(y_all.tolist()))))

# ── 1. Filter to ljoy 4-cardinal, remap ──────────────────────────────
CARD_IDS = [0, 2, 4, 6]
REMAP = {0: 0, 2: 1, 4: 2, 6: 3}
mask = np.isin(y_all, CARD_IDS)
X_raw = X_all[mask]
y = np.array([REMAP[v] for v in y_all[mask]])
del X_all, y_all
gc.collect()
N_CLS = 4

counts = np.bincount(y, minlength=N_CLS)
print("\n4-cardinal subset:")
for i in range(N_CLS):
    print("  class {} (ljoy_{}): {} trials".format(i, CARD_IDS[i], counts[i]))
print("  total: {} trials".format(len(y)))

# ── 2. 80/20 stratified split on RAW windows, seed 42 ────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=42,
)
del X_raw, y
gc.collect()
print("\ntrain: {} test: {}".format(X_train_raw.shape[0], X_test_raw.shape[0]))

# ── 3. FeatureStack: fit z-score on full 80% train ───────────────────
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
print("extracting train features (pass 1, un-zscored) ...")
train_feat_raw = [fs.transform(X_train_raw[i].astype(np.float32))
                  for i in range(len(X_train_raw))]
print("  single feature shape: {}".format(train_feat_raw[0].shape))

# Check shapes before proceeding
feat_shape = train_feat_raw[0].shape  # expect (128, 8)
if feat_shape != (128, 8):
    print("ERROR: unexpected feature shape {}, expected (128, 8)".format(feat_shape))
    sys.exit(1)

fs.fit_zscore(train_feat_raw)
del train_feat_raw
gc.collect()
print("fit_zscore done: _mean={} _std={}".format(fs._mean.shape, fs._std.shape))

# Check broadcast compatibility
if fs._mean.shape[0] != 128:
    print("ERROR: _mean shape {} does not broadcast to (128, 8)".format(fs._mean.shape))
    sys.exit(1)
if fs._mean.shape != (128, 1):
    print("ERROR: _mean shape {} != (128,1), cannot broadcast cleanly to (128,8)".format(
        fs._mean.shape))
    sys.exit(1)

# Re-transform with z-score applied
print("extracting train features (pass 2, z-scored) ...")
Xf_train = np.stack([fs.transform(X_train_raw[i].astype(np.float32))
                     for i in range(len(X_train_raw))]).astype(np.float32)
print("Xf_train={}".format(Xf_train.shape))

# ── 4. Fit probe on z-scored train features ──────────────────────────
factory = LinearProbeFactory(C=1.0, max_iter=2000)
fitted = factory.fit(Xf_train, y_train, n=N_CLS)
del Xf_train
gc.collect()
print("probe fitted")

# ── 5. Evaluate on 20% test ──────────────────────────────────────────
print("extracting test features (z-scored from train stats) ...")
Xf_test = np.stack([fs.transform(X_test_raw[i].astype(np.float32))
                    for i in range(len(X_test_raw))]).astype(np.float32)
log_probs_test = fitted.score(Xf_test)
pred_test = log_probs_test.argmax(axis=1)
bal_acc = balanced_accuracy_score(y_test, pred_test)
print("balanced accuracy on 20% test: {:.4f}".format(bal_acc))

# ── 6. Extract weights for ONNX ──────────────────────────────────────
scaler = fitted.pipeline.named_steps["scaler"]
lr = fitted.pipeline.named_steps["lr"]

fs_mean = fs._mean.astype(np.float32)       # (128, 1)
fs_std = fs._std.astype(np.float32)          # (128, 1)
sk_mean = scaler.mean_.astype(np.float32)    # (1024,)
sk_scale = scaler.scale_.astype(np.float32)  # (1024,)
W = lr.coef_.astype(np.float32)              # (4, 1024)
b = lr.intercept_.astype(np.float32)         # (4,)

print("\nONNX weight shapes:")
print("  fs_mean={} fs_std={}".format(fs_mean.shape, fs_std.shape))
print("  sk_mean={} sk_scale={}".format(sk_mean.shape, sk_scale.shape))
print("  W={} b={}".format(W.shape, b.shape))

# ── 7. Build ONNX graph ──────────────────────────────────────────────
import onnx
from onnx import TensorProto, helper, numpy_helper

# Scatter indices: 4-card classes map to positions [0,2,4,6] in 14-class output
SCATTER_IDX = [0, 2, 4, 6]
fill_val = np.float32(-1e9)


def make_graph():
    # --- Initializers (constant tensors) ---
    init_fs_mean = numpy_helper.from_array(fs_mean, name="fs_mean")
    init_fs_std = numpy_helper.from_array(fs_std, name="fs_std")
    init_sk_mean = numpy_helper.from_array(sk_mean.reshape(1, -1), name="sk_mean")
    init_sk_scale = numpy_helper.from_array(sk_scale.reshape(1, -1), name="sk_scale")
    init_W = numpy_helper.from_array(W.T, name="W")  # (1024, 4) for matmul
    init_b = numpy_helper.from_array(b.reshape(1, -1), name="b")
    init_flat_shape = numpy_helper.from_array(
        np.array([1, 1024], dtype=np.int64), name="flat_shape")
    init_scatter_idx = numpy_helper.from_array(
        np.array(SCATTER_IDX, dtype=np.int64).reshape(1, -1), name="scatter_idx")
    init_fill = numpy_helper.from_array(
        np.full((1, 14), fill_val, dtype=np.float32), name="fill_val")

    # --- Nodes ---
    nodes = []

    # 1. FeatureStack z-score: (x - fs_mean) / fs_std
    nodes.append(helper.make_node("Sub", ["input", "fs_mean"], ["zs1"]))
    nodes.append(helper.make_node("Div", ["zs1", "fs_std"], ["zscored"]))

    # 2. Flatten to (1, 1024)
    nodes.append(helper.make_node("Reshape", ["zscored", "flat_shape"], ["flat"]))

    # 3. Sklearn StandardScaler: (flat - sk_mean) / sk_scale
    nodes.append(helper.make_node("Sub", ["flat", "sk_mean"], ["sc1"]))
    nodes.append(helper.make_node("Div", ["sc1", "sk_scale"], ["scaled"]))

    # 4. Linear: scaled @ W + b -> (1, 4)
    nodes.append(helper.make_node("MatMul", ["scaled", "W"], ["logits_raw"]))
    nodes.append(helper.make_node("Add", ["logits_raw", "b"], ["logits"]))

    # 5. LogSoftmax over axis=1 -> (1, 4) log-probs
    nodes.append(helper.make_node("LogSoftmax", ["logits"], ["lp4"],
                                  axis=1))

    # 6. Scatter into (1, 14): fill with -1e9, then scatter 4 real values
    nodes.append(helper.make_node(
        "ScatterElements", ["fill_val", "scatter_idx", "lp4"], ["log_probs"],
        axis=1))

    # --- I/O ---
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128, 8])
    out = helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, [1, 14])

    graph = helper.make_graph(
        nodes,
        "fallback_decoder",
        [inp],
        [out],
        initializer=[init_fs_mean, init_fs_std, init_sk_mean, init_sk_scale,
                      init_W, init_b, init_flat_shape, init_scatter_idx, init_fill],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


onnx_model = make_graph()

# ── 8. Save artifacts ────────────────────────────────────────────────
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

onnx_path = artifacts_dir / "decoder_fallback.onnx"
onnx.save(onnx_model, str(onnx_path))
onnx_size = onnx_path.stat().st_size
print("\nSAVED {} ({} bytes, {:.1f} KB)".format(onnx_path, onnx_size, onnx_size / 1024))

meta = {
    "val_bal_acc": round(float(bal_acc), 4),
    "val_n": int(len(y_test)),
    "theta": 0.0,
    "N": 4,
    "symbol_ids": [0, 2, 4, 6],
}

meta_path = artifacts_dir / "decoder_fallback_meta.json"
with open(str(meta_path), "w") as f:
    json.dump(meta, f, indent=2)
print("SAVED {}".format(meta_path))
print("\n" + json.dumps(meta, indent=2))

# ── 9. Roundtrip check: ONNX vs Python pipeline on 10 test windows ──
import onnxruntime as ort

sess = ort.InferenceSession(str(onnx_path))

rng = np.random.RandomState(0)
test_indices = rng.choice(len(X_test_raw), size=min(10, len(X_test_raw)), replace=False)
max_diff = 0.0
FILLED = SCATTER_IDX

print("\nRoundtrip check (10 test windows):")
# Fresh FS without z-score for un-zscored ONNX input
fs_nozs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))

for idx in test_indices:
    # Python pipeline (z-scored features -> sklearn pipeline)
    feat = fs.transform(X_test_raw[idx].astype(np.float32))  # (128, 8), z-scored
    lp_py_4 = fitted.pipeline.predict_log_proba(feat.reshape(1, -1))[0]  # (4,)
    lp_py_14 = np.full(14, -1e9, dtype=np.float32)
    for j, si in enumerate(SCATTER_IDX):
        lp_py_14[si] = lp_py_4[j]

    # ONNX (un-zscored features -> model applies FS z-score internally)
    feat_raw = fs_nozs.transform(X_test_raw[idx].astype(np.float32))  # NO z-score
    onnx_input = feat_raw.reshape(1, 128, 8).astype(np.float32)
    lp_onnx_14 = sess.run(None, {"input": onnx_input})[0][0]  # (14,)

    # Compare on filled indices only
    diff = np.max(np.abs(lp_py_14[FILLED] - lp_onnx_14[FILLED]))
    max_diff = max(max_diff, diff)
    print("  test[{}]: max|diff| on filled indices = {:.2e}".format(idx, diff))

print("\nMAX ABS DIFF (4 filled indices, 10 windows): {:.2e}".format(max_diff))
if max_diff < 1e-4:
    print("PASS: roundtrip diff < 1e-4")
else:
    print("FAIL: roundtrip diff >= 1e-4")

print("\nONNX size: {} bytes".format(onnx_size))

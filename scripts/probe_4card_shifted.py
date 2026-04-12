"""
probe_4card_shifted.py -- Sweep onset offset on 4-cardinal ljoy data.
Tests where discriminative information lives in time relative to cue.
"""
import numpy as np, sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from load_h5_session import align_and_build_session, extract_windows
from penn_nh.features.stack import FeatureConfig, FeatureStack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# ── Config ───────────────────────────────────────────────────────────
CARD_IDS = [0, 2, 4, 6]
REMAP = {0: 0, 2: 1, 4: 2, 6: 3}
N_CLS = 4
W_POST_MS = 400.0
FS_HZ = 32000.0
OFFSETS_MS = [0, 100, 200, 300, 400]

data_dir = Path("data/newdata_apr12")
seeds = [s for s in range(1, 16) if s != 6]

# ── Feature extractors ──────────────────────────────────────────────
def feat_featurestack(X):
    fs = FeatureStack(FeatureConfig(fs_hz=FS_HZ, bin_ms=50.0, spike_threshold_k=2.0))
    return np.stack([fs.transform(X[i].astype(np.float32)).flatten()
                     for i in range(len(X))])

def feat_var(X):
    return X.var(axis=2)

def feat_log_var(X):
    return np.log(X.var(axis=2) + 1e-12)

FEATURES = [
    ("FeatureStack(50ms,k=2)", feat_featurestack),
    ("variance",               feat_var),
    ("log-variance",           feat_log_var),
]

# ── Load sessions once ──────────────────────────────────────────────
print("Loading sessions ...")
sessions = []
for s in seeds:
    h5 = data_dir / f"smoke_Ani_seed{s}.h5"
    cj = data_dir / f"cues_Ani_seed{s}.json"
    if not h5.exists() or not cj.exists():
        continue
    try:
        sess = align_and_build_session(h5, cj)
        sessions.append((s, sess))
        print(f"  seed {s}: {sess.neural.shape[1]} samples, {len(sess.cue_times_ns)} cues")
    except ValueError as e:
        print(f"  SKIP seed {s}: {e}")

if not sessions:
    print("NO DATA")
    sys.exit(1)

def evaluate(offset_ms):
    """Extract windows at given offset, filter to 4-cardinal, return (X, y)."""
    all_X, all_y = [], []
    for s, sess in sessions:
        X, y_raw, _ = extract_windows(sess, w_post_ms=W_POST_MS,
                                       onset_offset_ms=offset_ms)
        if len(X) == 0:
            continue
        mask = np.isin(y_raw, CARD_IDS)
        if mask.sum() == 0:
            continue
        all_X.append(X[mask, :64, :])
        all_y.append(np.array([REMAP[v] for v in y_raw[mask]]))
    if not all_X:
        return None, None
    return np.concatenate(all_X), np.concatenate(all_y)

def cv_score(Xf, y):
    """5-fold stratified LogisticRegression bal_acc."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    bals = []
    for tr, te in skf.split(Xf, y):
        sc = StandardScaler().fit(Xf[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(sc.transform(Xf[tr]), y[tr])
        pred = clf.predict(sc.transform(Xf[te]))
        bals.append(balanced_accuracy_score(y[te], pred))
    return bals

# ── Sweep offsets ────────────────────────────────────────────────────
print(f"\nSweeping offsets: {OFFSETS_MS} ms")
print(f"Window duration: {W_POST_MS} ms")
print(f"Features: {[n for n, _ in FEATURES]}")

# results[offset_ms][feat_name] = (mean_bal, folds, n_trials)
all_results = {}

for offset_ms in OFFSETS_MS:
    print(f"\n{'='*70}")
    print(f"  OFFSET = {offset_ms} ms  ->  window [{offset_ms}ms .. {offset_ms + W_POST_MS:.0f}ms]")
    print(f"{'='*70}")

    X, y = evaluate(offset_ms)
    if X is None:
        print("  NO DATA at this offset")
        all_results[offset_ms] = {}
        continue

    counts = np.bincount(y, minlength=N_CLS)
    print(f"  {len(y)} trials  classes: {counts.tolist()}  X_trial: {X.shape[1:]}")

    all_results[offset_ms] = {}
    for feat_name, feat_fn in FEATURES:
        Xf = feat_fn(X)
        bals = cv_score(Xf, y)
        mean_bal = np.mean(bals)
        print(f"  {feat_name:<28s} shape={str(Xf.shape):<14s} "
              f"folds={[f'{b:.3f}' for b in bals]}  mean={mean_bal:.4f}")
        all_results[offset_ms][feat_name] = (mean_bal, bals, len(y))

# ── Summary table ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY: mean balanced accuracy by offset x feature")
print(f"{'='*70}")

feat_names = [n for n, _ in FEATURES]
header = f"{'offset':>8s}  {'window':>18s}  " + "  ".join(f"{n:>14s}" for n in feat_names)
print(header)
print("─" * len(header))

for offset_ms in OFFSETS_MS:
    window_str = f"[{offset_ms}..{offset_ms + W_POST_MS:.0f}ms]"
    vals = []
    for fn in feat_names:
        if fn in all_results[offset_ms]:
            vals.append(f"{all_results[offset_ms][fn][0]:>14.4f}")
        else:
            vals.append(f"{'N/A':>14s}")
    print(f"{offset_ms:>6d}ms  {window_str:>18s}  " + "  ".join(vals))

print(f"\nchance = 0.250")

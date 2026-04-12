"""
diagnose_signal.py -- Diagnostic feature sweep on 4-cardinal ljoy data.
Compare raw statistical features vs current FeatureStack pipeline.
"""
import numpy as np, sys, subprocess
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.signal import welch
from scipy.stats import f_oneway

sys.path.insert(0, "src")
from penn_nh.features.stack import FeatureConfig, FeatureStack

# ── 0. Load & pool all sessions (exclude seed 6) ────────────────────
data_dir = Path("data/newdata_apr12")
seeds = [s for s in range(1, 16) if s != 6]

all_X, all_y = [], []
for s in seeds:
    h5 = data_dir / f"smoke_Ani_seed{s}.h5"
    cj = data_dir / f"cues_Ani_seed{s}.json"
    if not h5.exists() or not cj.exists():
        continue
    out = f"/tmp/diag_s{s}.npz"
    if not Path(out).exists():
        r = subprocess.run(
            ["python", "scripts/load_h5_session.py",
             "--h5", str(h5), "--cues", str(cj), "--out", out],
            capture_output=True,
        )
        if r.returncode != 0:
            print(f"SKIP seed {s}: {r.stderr.decode()[-200:]}")
            continue
    d = np.load(out)
    all_X.append(d["X"])
    all_y.append(d["y"])
    print(f"seed {s}: {d['X'].shape[0]} trials")

if not all_X:
    print("NO DATA")
    sys.exit(1)

X = np.concatenate(all_X)
y = np.concatenate(all_y)
print(f"\npooled raw: X={X.shape} unique_labels={sorted(set(y.tolist()))}")

# ── 1. Filter to ljoy 4-cardinal {0,2,4,6}, remap to {0,1,2,3} ────
CARD_IDS = [0, 2, 4, 6]
REMAP = {0: 0, 2: 1, 4: 2, 6: 3}
mask = np.isin(y, CARD_IDS)
X = X[mask, :64, :]       # neural ch 0..63, full aligned window
y = np.array([REMAP[v] for v in y[mask]])
N_CLS = 4

counts = np.bincount(y, minlength=N_CLS)
print(f"\n4-cardinal subset: {len(y)} trials")
for i in range(N_CLS):
    print(f"  class {i} (ljoy_{CARD_IDS[i]}): {counts[i]} trials")
print(f"  X shape per trial: {X.shape[1:]}  (channels, time)")

# ── 2. Feature extraction functions ──────────────────────────────────
FS_HZ = 32000.0

def feat_mean(X):
    """1. Per-channel raw mean voltage over window -> (N, 64)"""
    return X.mean(axis=2)

def feat_var(X):
    """2. Per-channel raw variance over window -> (N, 64)"""
    return X.var(axis=2)

def feat_diff_std(X):
    """3. Per-channel raw std of diff(x) (high-freq energy) -> (N, 64)"""
    return np.std(np.diff(X, axis=2), axis=2)

def feat_welch_bands(X):
    """4. Per-channel band power welch, 4 bands -> (N, 256)"""
    bands = [(1, 30), (30, 100), (100, 300), (300, 3000)]
    N_trials, C, T = X.shape
    out = np.zeros((N_trials, C * len(bands)), dtype=np.float64)
    for i in range(N_trials):
        freqs, psd = welch(X[i], fs=FS_HZ, nperseg=min(T, 1024), axis=1)
        for bi, (lo, hi) in enumerate(bands):
            idx = (freqs >= lo) & (freqs <= hi)
            if idx.any():
                out[i, bi * C:(bi + 1) * C] = psd[:, idx].mean(axis=1)
    return out

def feat_featurestack(X):
    """5. Current FeatureStack(bin_ms=50, k=2.0) flat -> (N, ?)"""
    fs = FeatureStack(FeatureConfig(fs_hz=FS_HZ, bin_ms=50.0, spike_threshold_k=2.0))
    feats = []
    for i in range(len(X)):
        f = fs.transform(X[i].astype(np.float32))
        feats.append(f.flatten())
    return np.stack(feats)

def feat_log_var(X):
    """6. Per-channel log-variance -> (N, 64)"""
    v = X.var(axis=2)
    return np.log(v + 1e-12)

def feat_ptp(X):
    """7. Per-channel peak-to-peak -> (N, 64)"""
    return X.max(axis=2) - X.min(axis=2)


CONFIGS = [
    ("1. mean voltage",            feat_mean),
    ("2. variance",                feat_var),
    ("3. diff_std (HF energy)",    feat_diff_std),
    ("4. welch 4-band power",      feat_welch_bands),
    ("5. FeatureStack(50ms,k=2)",  feat_featurestack),
    ("6. log-variance",            feat_log_var),
    ("7. peak-to-peak",            feat_ptp),
]

# ── 3. Evaluate each config ─────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
results = []

for name, fn in CONFIGS:
    print(f"\n{'─'*60}")
    print(f"  {name}")
    Xf = fn(X)
    shape = Xf.shape
    print(f"  shape: {shape}")

    bals = []
    for fold, (tr, te) in enumerate(skf.split(Xf, y)):
        sc = StandardScaler().fit(Xf[tr])
        Xtr = sc.transform(Xf[tr])
        Xte = sc.transform(Xf[te])
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        bal = balanced_accuracy_score(y[te], pred)
        bals.append(bal)
    mean_bal = np.mean(bals)
    print(f"  folds: {[f'{b:.3f}' for b in bals]}")
    print(f"  mean bal_acc: {mean_bal:.4f}  (chance=0.250)")
    results.append((name, shape, mean_bal))

# ── 4. ANOVA F-stats for config 2 (variance) ────────────────────────
print(f"\n{'='*60}")
print("ANOVA F-stats for config 2 (variance), 64 channels")
Xvar = feat_var(X)
f_stats = []
for ch in range(64):
    groups = [Xvar[y == c, ch] for c in range(N_CLS)]
    f, _ = f_oneway(*groups)
    f_stats.append(f)
f_stats = np.array(f_stats)
print(f"  F mean across 64 ch: {f_stats.mean():.4f}")
print(f"  F max  across 64 ch: {f_stats.max():.4f}")

top10 = np.argsort(f_stats)[::-1][:10]
print(f"\n  Top-10 F-stat channels:")
for rank, ch in enumerate(top10):
    print(f"    #{rank+1:2d}  ch {ch:2d}  F={f_stats[ch]:.4f}")

# ── 5. Summary table ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Name':<32s} {'Shape':<14s} {'Bal Acc':>8s}")
print(f"{'─'*32} {'─'*14} {'─'*8}")
for name, shape, bal in results:
    print(f"{name:<32s} {str(shape):<14s} {bal:>8.4f}")
print(f"{'─'*32} {'─'*14} {'─'*8}")
print(f"chance = 0.250")

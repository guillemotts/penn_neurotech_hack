import numpy as np, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from load_training_csv import load_session, SYMBOLS, extract_trial_windows
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

s = load_session(Path('data/train_ani-1.csv'), Path('data/train_ani_1_seed10.csv'))

# Extract raw-channel windows parallel to neural windows
fs = s.fs_hz
w_samples = int(0.4 * fs)
offset = int(0.05 * fs)
X_raw, y = [], []
for i, t0 in enumerate(s.cue_times):
    start_idx = int(np.searchsorted(s.time_s, t0)) + offset
    end_idx = start_idx + w_samples
    if end_idx > s.raw.shape[1]: continue
    X_raw.append(s.raw[:, start_idx:end_idx])
    y.append(s.cue_indices[i])
X_raw = np.stack(X_raw).astype(np.float32)
y = np.array(y)
print(f"X_raw={X_raw.shape} y={y.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
    probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(X_raw[tr], y[tr], n=22)
    pred = probe.score(X_raw[te]).argmax(axis=1)
    acc = (pred == y[te]).mean()
    accs.append(acc)
    print(f"fold {fold}: acc={acc:.3f}")
print(f"\nRAW MEAN={np.mean(accs):.3f}  chance=0.045")

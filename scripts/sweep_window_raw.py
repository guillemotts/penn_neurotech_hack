import numpy as np, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from load_training_csv import load_session
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

s = load_session(Path('data/train_ani-1.csv'), Path('data/train_ani_1_seed10.csv'))
fs = s.fs_hz

# Sweep (offset_ms, window_ms)
offsets = [0, 100, 200, 300, 400, 500]
windows = [200, 400, 600, 800]

print(f"{'off_ms':>6} {'win_ms':>6} {'n':>4}  {'acc':>5}")
for off_ms in offsets:
    for win_ms in windows:
        off_s = int(off_ms/1000 * fs)
        w_s = int(win_ms/1000 * fs)
        X, y = [], []
        for i, t0 in enumerate(s.cue_times):
            start = int(np.searchsorted(s.time_s, t0)) + off_s
            end = start + w_s
            if end > s.raw.shape[1] or start < 0: continue
            X.append(s.raw[:, start:end])
            y.append(s.cue_indices[i])
        if len(X) < 50:
            print(f"{off_ms:>6} {win_ms:>6} {len(X):>4}  skip")
            continue
        X = np.stack(X).astype(np.float32)
        y = np.array(y)
        # Check all 22 classes present
        if len(np.unique(y)) < 22:
            continue
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        accs = []
        for tr, te in skf.split(X, y):
            try:
                probe = LinearProbeFactory(C=1.0, max_iter=1000).fit(X[tr], y[tr], n=22)
                pred = probe.score(X[te]).argmax(axis=1)
                accs.append((pred == y[te]).mean())
            except ValueError:
                accs.append(float('nan'))
        print(f"{off_ms:>6} {win_ms:>6} {len(X):>4}  {np.nanmean(accs):.3f}")

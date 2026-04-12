import numpy as np, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from load_training_csv import load_session, SYMBOLS
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

s = load_session(Path('data/train_ani-1.csv'), Path('data/train_ani_1_seed10.csv'))
fs = s.fs_hz

# Macro label: 0=ljoy, 1=rjoy, 2=button
def macro(i):
    t, _ = SYMBOLS[i]
    return {'ljoy':0, 'rjoy':1, 'button':2}[t]

w_s = int(0.4 * fs); off_s = int(0.05 * fs)
X, y = [], []
for i, t0 in enumerate(s.cue_times):
    start = int(np.searchsorted(s.time_s, t0)) + off_s
    end = start + w_s
    if end > s.raw.shape[1]: continue
    X.append(s.raw[:, start:end])
    y.append(macro(s.cue_indices[i]))
X = np.stack(X).astype(np.float32); y = np.array(y)
print(f"X={X.shape} y={y.shape} macro_counts={np.bincount(y)}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
for tr, te in skf.split(X, y):
    p = LinearProbeFactory(C=1.0, max_iter=1000).fit(X[tr], y[tr], n=3)
    pred = p.score(X[te]).argmax(axis=1)
    accs.append((pred == y[te]).mean())
print(f"MACRO (3-class): MEAN={np.mean(accs):.3f}  chance=0.333")

# Also try stick-vs-button (2-class)
y2 = (y == 2).astype(int)
accs2 = []
for tr, te in skf.split(X, y2):
    p = LinearProbeFactory(C=1.0, max_iter=1000).fit(X[tr], y2[tr], n=2)
    pred = p.score(X[te]).argmax(axis=1)
    accs2.append((pred == y2[te]).mean())
print(f"STICK-vs-BUTTON: MEAN={np.mean(accs2):.3f}  chance=0.500")

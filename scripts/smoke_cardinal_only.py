import numpy as np, sys
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

Xs, ys = [], []
for seed in range(1, 6):
    d = np.load(f'/tmp/ani_s{seed}.npz')
    Xs.append(d['X']); ys.append(d['y'])
X = np.concatenate(Xs); y = np.concatenate(ys)

Xn = X[:, :64, :]
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
Xf = np.stack([fs.transform(Xn[i].astype(np.float32)) for i in range(len(Xn))])

# Cardinal-only groupings (drop diagonals 1,3,5,7 and 9,11,13,15)
groups = [
    ('ljoy cardinals (4)', {0:0, 2:1, 4:2, 6:3}),
    ('rjoy cardinals (4)', {8:0, 10:1, 12:2, 14:3}),
    ('both stick cardinals (8)', {0:0,2:1,4:2,6:3,8:4,10:5,12:6,14:7}),
    ('buttons only (6)', {16:0,17:1,18:2,19:3,20:4,21:5}),
    ('sticks+buttons no diag (14)', {0:0,2:1,4:2,6:3,8:4,10:5,12:6,14:7,16:8,17:9,18:10,19:11,20:12,21:13}),
]
for name, m in groups:
    mask = np.isin(y, list(m.keys()))
    Xm = Xf[mask]; ym = np.array([m[v] for v in y[mask]])
    k = len(set(m.values()))
    counts = np.bincount(ym, minlength=k)
    if min(counts) < 3 or len(Xm) < k*3:
        print(f'{name}: skip counts={counts.tolist()}')
        continue
    nf = min(5, int(min(counts)))
    skf = StratifiedKFold(n_splits=nf, shuffle=True, random_state=0)
    bals = []
    for tr,te in skf.split(Xm, ym):
        p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xm[tr], ym[tr], n=k)
        pred = p.score(Xm[te]).argmax(axis=1)
        bals.append(balanced_accuracy_score(ym[te], pred))
    print(f'{name} k={k} n={len(Xm)} counts={counts.tolist()}: bal={np.mean(bals):.3f} chance={1/k:.3f}')

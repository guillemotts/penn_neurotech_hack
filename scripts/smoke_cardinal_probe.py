import numpy as np, sys
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

# Cue symbol indices in SYMBOLS list:
# 0-7=ljoy, 8-15=rjoy, 16-21=buttons(UP,DOWN,LEFT,RIGHT,LB,RB)
Xs, ys = [], []
for seed in range(1, 6):
    d = np.load(f'/tmp/ani_s{seed}.npz')
    Xs.append(d['X']); ys.append(d['y'])
X = np.concatenate(Xs); y = np.concatenate(ys)

# Collapse to 4 cardinals: ljoy_0=E(0), ljoy_2=N(2), ljoy_4=W(4), ljoy_6=S(6)
# Plus rjoy equivalents 8,10,12,14
# Plus buttons UP/DOWN/LEFT/RIGHT (16-19)
# Try 3 groupings:
groups = [
    ('4 button dirs', {16:0, 17:1, 18:2, 19:3}),
    ('4 ljoy cardinals', {0:0, 2:1, 4:2, 6:3}),
    ('stick vs button', {0:0,2:0,4:0,6:0,8:0,10:0,12:0,14:0, 16:1,17:1,18:1,19:1,20:1,21:1}),
]
Xn = X[:, :64, :]
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
Xf = np.stack([fs.transform(Xn[i].astype(np.float32)) for i in range(len(Xn))])

for name, mapping in groups:
    mask = np.isin(y, list(mapping.keys()))
    Xm = Xf[mask]
    ym = np.array([mapping[v] for v in y[mask]])
    k = len(set(mapping.values()))
    if len(Xm) < 20 or min(np.bincount(ym)) < 3:
        print(f'{name}: skip (n={len(Xm)} counts={np.bincount(ym).tolist()})')
        continue
    n_folds = min(5, min(np.bincount(ym)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    bals = []
    for tr,te in skf.split(Xm, ym):
        p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xm[tr], ym[tr], n=k)
        pred = p.score(Xm[te]).argmax(axis=1)
        bals.append(balanced_accuracy_score(ym[te], pred))
    print(f'{name} k={k} n={len(Xm)} counts={np.bincount(ym).tolist()}: bal={np.mean(bals):.3f} chance={1/k:.3f}')

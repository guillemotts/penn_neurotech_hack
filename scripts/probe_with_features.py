import numpy as np, sys
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold

d = np.load('/tmp/pooled_ss.npz', allow_pickle=True)
X, y = d['X'], d['y']
print(f'raw X={X.shape}')

# Prune classes with <20 samples
counts = np.bincount(y)
keep = set(np.where(counts >= 20)[0].tolist())
mask = np.array([v in keep for v in y])
Xs = X[mask]
remap = {s: i for i, s in enumerate(sorted(keep))}
ys = np.array([remap[v] for v in y[mask]])
k = len(remap)
print(f'pruned: {Xs.shape[0]} trials, {k} classes, counts={np.bincount(ys).tolist()}')

cfg = FeatureConfig(fs_hz=32000.0, bin_ms=1.0)
fs = FeatureStack(cfg)
Xf = np.stack([fs.transform(Xs[i].astype(np.float32)) for i in range(len(Xs))])
print(f'feat X={Xf.shape} dtype={Xf.dtype}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
for fold, (tr, te) in enumerate(skf.split(Xf, ys)):
    probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xf[tr], ys[tr], n=k)
    pred = probe.score(Xf[te]).argmax(axis=1)
    acc = (pred == ys[te]).mean()
    accs.append(acc)
    print(f'fold {fold}: acc={acc:.3f}')
print(f'MEAN={np.mean(accs):.3f} chance={1/k:.3f}')

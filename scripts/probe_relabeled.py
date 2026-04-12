import numpy as np, sys
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

d = np.load('/tmp/relabeled.npz')
X, y = d['X'], d['y']
counts = np.bincount(y, minlength=14)
print(f'raw counts={counts.tolist()}')

keep = sorted(np.where(counts >= 15)[0].tolist())
mask = np.isin(y, keep)
Xs = X[mask]
remap = {s: i for i, s in enumerate(keep)}
ys = np.array([remap[v] for v in y[mask]])
k = len(keep)
print(f'kept classes (orig idx): {keep}  counts={np.bincount(ys).tolist()}  k={k}')
print(f'majority baseline: {np.bincount(ys).max()/len(ys):.3f}  chance: {1/k:.3f}')

fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=1.0))
Xf = np.stack([fs.transform(Xs[i].astype(np.float32)) for i in range(len(Xs))])
print(f'feat X={Xf.shape}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
all_pred, all_true = [], []
for fold, (tr, te) in enumerate(skf.split(Xf, ys)):
    probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xf[tr], ys[tr], n=k)
    pred = probe.score(Xf[te]).argmax(axis=1)
    all_pred.append(pred); all_true.append(ys[te])
    print(f'fold {fold}: acc={(pred==ys[te]).mean():.3f}')
pred = np.concatenate(all_pred); true = np.concatenate(all_true)
print(f'\naccuracy={(pred==true).mean():.3f}  balanced={balanced_accuracy_score(true, pred):.3f}')
print(f'confusion:\n{confusion_matrix(true, pred)}')

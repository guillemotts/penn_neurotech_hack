import numpy as np, sys
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

d = np.load('/tmp/pooled_ss.npz', allow_pickle=True)
X, y = d['X'], d['y']
pattern_map = d['pattern_map'].item() if 'pattern_map' in d.files else {}
print('pattern_map:', pattern_map)

counts = np.bincount(y)
keep = sorted(np.where(counts >= 20)[0].tolist())
mask = np.isin(y, keep)
Xs = X[mask]
remap = {s: i for i, s in enumerate(keep)}
ys = np.array([remap[v] for v in y[mask]])
k = len(keep)
print(f'kept classes (orig idx): {keep}')
print(f'counts: {np.bincount(ys).tolist()}')

# Majority baseline
maj = np.bincount(ys).max() / len(ys)
print(f'majority baseline: {maj:.3f}')

fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=1.0))
Xf = np.stack([fs.transform(Xs[i].astype(np.float32)) for i in range(len(Xs))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
all_pred, all_true = [], []
for tr, te in skf.split(Xf, ys):
    probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xf[tr], ys[tr], n=k)
    pred = probe.score(Xf[te]).argmax(axis=1)
    all_pred.append(pred); all_true.append(ys[te])
pred = np.concatenate(all_pred); true = np.concatenate(all_true)

print(f'\naccuracy: {(pred==true).mean():.3f}')
print(f'balanced_accuracy: {balanced_accuracy_score(true, pred):.3f}')
print(f'\nconfusion matrix (rows=true, cols=pred):')
cm = confusion_matrix(true, pred)
print(cm)
print(f'\nper-class recall:')
for i in range(k):
    r = cm[i,i] / max(1, cm[i].sum())
    print(f'  class {i} (orig {keep[i]}, n={cm[i].sum()}): recall={r:.3f}')

# Also: check if the two mega-classes are actually distinguishable from each other
print(f'\nbinary: mega-class 0 vs mega-class 1 only')
mega = np.where(np.isin(ys, [0, 1]))[0]  # assuming 0,1 are remapped megas
Xm = Xf[mega]; ym = ys[mega]
# Relabel 0/1
ym_bin = (ym == ys[mega][0]).astype(int) if len(np.unique(ym))==2 else ym
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
for tr, te in skf2.split(Xm, ym):
    p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xm[tr], ym[tr], n=2)
    pr = p.score(Xm[te]).argmax(axis=1)
    accs.append((pr==ym[te]).mean())
print(f'  mega-vs-mega acc: {np.mean(accs):.3f} (chance 0.5)')

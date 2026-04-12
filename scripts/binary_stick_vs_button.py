import numpy as np, sys
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold

d = np.load('/tmp/relabeled.npz')
X, y = d['X'], d['y']
y_bin = (y >= 8).astype(np.int64)
print(f'stick={np.sum(y_bin==0)} button={np.sum(y_bin==1)}')
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=1.0))
Xf = np.stack([fs.transform(X[i].astype(np.float32)) for i in range(len(X))])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs=[]
for tr,te in skf.split(Xf,y_bin):
    p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xf[tr], y_bin[tr], n=2)
    pred = p.score(Xf[te]).argmax(axis=1)
    accs.append((pred==y_bin[te]).mean())
    print(f'acc={accs[-1]:.3f}')
print(f'MEAN={np.mean(accs):.3f} chance=0.5 majority={max(np.bincount(y_bin))/len(y_bin):.3f}')

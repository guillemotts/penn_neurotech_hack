import numpy as np, sys
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from sklearn.feature_selection import f_classif

d = np.load('/tmp/smoke_ani.npz')
X, y = d['X'], d['y']
print(f'X={X.shape} y={y} counts={np.bincount(y, minlength=22).tolist()}')

# Strip to neural only (first 64 of 76)
Xn = X[:, :64, :]
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
Xf = np.stack([fs.transform(Xn[i].astype(np.float32)) for i in range(len(Xn))])
print(f'Xf={Xf.shape} mean_spike_count={Xf[:,:64,:].mean():.3f}')

# Per-class means
for cls in sorted(set(y.tolist())):
    mask = y == cls
    sc = Xf[mask, :64, :].mean()
    print(f'  cls {cls} n={mask.sum()}: spike_count={sc:.3f}')

# F-stat if >1 class
if len(set(y.tolist())) > 1:
    Xflat = Xf.reshape(len(Xf), -1)
    F, p = f_classif(Xflat, y)
    F_clean = F[np.isfinite(F)]
    print(f'F max={F_clean.max():.2f} median={np.median(F_clean):.2f}')
    print(f'sig p<0.01: {(p<0.01).sum()}/{len(p)}  ({(p<0.01).mean()*100:.1f}%)')

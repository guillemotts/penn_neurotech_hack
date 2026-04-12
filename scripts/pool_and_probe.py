import numpy as np, sys, json
from pathlib import Path
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Load all 5 npz (assume already extracted)
import subprocess
data_dir = Path('data/newdata_apr12')
all_X, all_y = [], []
for seed in [1, 2, 3, 4, 5]:
    h5 = data_dir / f'smoke_Ani_seed{seed}.h5'
    cj = data_dir / f'cues_Ani_seed{seed}.json'
    out = f'/tmp/ani_s{seed}.npz'
    if not Path(out).exists():
        subprocess.run(['python', 'scripts/load_h5_session.py',
            '--h5', str(h5), '--cues', str(cj), '--out', out], check=True)
    d = np.load(out)
    all_X.append(d['X']); all_y.append(d['y'])
X = np.concatenate(all_X); y = np.concatenate(all_y)
print(f'pooled X={X.shape} counts={np.bincount(y, minlength=22).tolist()}')

# Keep neural only
Xn = X[:, :64, :]
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
Xf = np.stack([fs.transform(Xn[i].astype(np.float32)) for i in range(len(Xn))])
print(f'Xf={Xf.shape}')

# Full 22-class probe (drop classes with <5)
counts = np.bincount(y, minlength=22)
keep = sorted(np.where(counts >= 5)[0].tolist())
mask = np.isin(y, keep)
Xs = Xf[mask]
remap = {s: i for i, s in enumerate(keep)}
ys = np.array([remap[v] for v in y[mask]])
k = len(keep)
print(f'keep k={k} counts={np.bincount(ys).tolist()} chance={1/k:.3f}')

min_cls = np.bincount(ys).min()
n_folds = min(5, min_cls)
if n_folds >= 2:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    accs, bals = [], []
    all_pred, all_true = [], []
    for tr, te in skf.split(Xs, ys):
        p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xs[tr], ys[tr], n=k)
        pred = p.score(Xs[te]).argmax(axis=1)
        accs.append((pred == ys[te]).mean())
        bals.append(balanced_accuracy_score(ys[te], pred))
        all_pred.append(pred); all_true.append(ys[te])
    print(f'{k}-class: acc={np.mean(accs):.3f} bal={np.mean(bals):.3f}')
    cm = confusion_matrix(np.concatenate(all_true), np.concatenate(all_pred))
    print(f'confusion:\n{cm}')

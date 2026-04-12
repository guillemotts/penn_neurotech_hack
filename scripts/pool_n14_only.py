import numpy as np, sys, json, subprocess
from pathlib import Path
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

data_dir = Path('data/newdata_apr12')
# Only n14 sessions
good_seeds = [7, 8, 9, 10, 11, 12, 13, 14, 15]
all_X, all_y = [], []
failed = []
for s in good_seeds:
    h5 = data_dir / f'smoke_Ani_seed{s}.h5'
    cj = data_dir / f'cues_Ani_seed{s}.json'
    if not h5.exists(): continue
    out = f'/tmp/ani14_s{s}.npz'
    if not Path(out).exists():
        r = subprocess.run(['python', 'scripts/load_h5_session.py',
            '--h5', str(h5), '--cues', str(cj), '--out', out], capture_output=True)
        if r.returncode != 0:
            print(f'FAIL seed{s}: {r.stderr.decode()[-300:]}')
            failed.append(s)
            continue
    d = np.load(out)
    all_X.append(d['X']); all_y.append(d['y'])
    print(f'OK seed{s}: {d["X"].shape[0]} trials')

if not all_X:
    print('NO DATA'); sys.exit(1)

X = np.concatenate(all_X); y = np.concatenate(all_y)
print(f'\npooled X={X.shape}')
# Remap n14 symbols to 0..13
n14_ids = [0,2,4,6,8,10,12,14,16,17,18,19,20,21]
mask = np.isin(y, n14_ids)
Xs = X[mask, :64, :]
remap = {s:i for i,s in enumerate(n14_ids)}
ys = np.array([remap[v] for v in y[mask]])
print(f'n14: X={Xs.shape} counts={np.bincount(ys, minlength=14).tolist()}')

fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
Xf = np.stack([fs.transform(Xs[i].astype(np.float32)) for i in range(len(Xs))])

# Drop classes <5
counts = np.bincount(ys, minlength=14)
keep_cls = [c for c in range(14) if counts[c] >= 5]
m2 = np.isin(ys, keep_cls)
Xf = Xf[m2]
remap2 = {c:i for i,c in enumerate(keep_cls)}
ys2 = np.array([remap2[v] for v in ys[m2]])
k = len(keep_cls)
print(f'after <5 prune: k={k} counts={np.bincount(ys2).tolist()}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
bals = []
for tr,te in skf.split(Xf, ys2):
    p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xf[tr], ys2[tr], n=k)
    pred = p.score(Xf[te]).argmax(axis=1)
    bals.append(balanced_accuracy_score(ys2[te], pred))
    print(f'  fold bal={bals[-1]:.3f}')
print(f'LINEAR n14-pruned: bal={np.mean(bals):.3f} chance={1/k:.3f}')

# Also 4-class cardinal-only
card = [0, 2, 4, 6]
mc = np.isin(y, card)
Xc = X[mc, :64, :]
Xcf = np.stack([fs.transform(Xc[i].astype(np.float32)) for i in range(len(Xc))])
yc = np.array([{0:0,2:1,4:2,6:3}[v] for v in y[mc]])
print(f'\n4-card: n={len(Xcf)} counts={np.bincount(yc).tolist()}')
if len(Xcf) >= 20 and min(np.bincount(yc)) >= 3:
    bals4 = []
    for tr,te in skf.split(Xcf, yc):
        p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xcf[tr], yc[tr], n=4)
        pred = p.score(Xcf[te]).argmax(axis=1)
        bals4.append(balanced_accuracy_score(yc[te], pred))
    print(f'LINEAR 4-card: bal={np.mean(bals4):.3f} chance=0.250')

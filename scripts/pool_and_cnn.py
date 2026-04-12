import numpy as np, sys, json, subprocess, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0, 'src')
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

data_dir = Path('data/newdata_apr12')
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
all_X, all_y = [], []
for s in seeds:
    h5 = data_dir / f'smoke_Ani_seed{s}.h5'
    cj = data_dir / f'cues_Ani_seed{s}.json'
    if not h5.exists() or not cj.exists(): continue
    out = f'/tmp/ani_s{s}.npz'
    if not Path(out).exists():
        r = subprocess.run(['python', 'scripts/load_h5_session.py',
            '--h5', str(h5), '--cues', str(cj), '--out', out])
        if r.returncode != 0:
            print(f'SKIP seed {s} (load failed)')
            continue
    d = np.load(out)
    all_X.append(d['X']); all_y.append(d['y'])
X = np.concatenate(all_X); y = np.concatenate(all_y)
print(f'pooled X={X.shape} counts={np.bincount(y, minlength=22).tolist()}')

Xn = X[:, :64, :]
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
Xf = np.stack([fs.transform(Xn[i].astype(np.float32)) for i in range(len(Xn))])

# Drop classes <5
counts = np.bincount(y, minlength=22)
keep = sorted(np.where(counts >= 5)[0].tolist())
mask = np.isin(y, keep)
Xs = Xf[mask]; remap = {s:i for i,s in enumerate(keep)}
ys = np.array([remap[v] for v in y[mask]])
k = len(keep)
print(f'k={k} counts={np.bincount(ys).tolist()} chance={1/k:.3f}')

# Linear baseline
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
bals_lin = []
for tr,te in skf.split(Xs, ys):
    p = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xs[tr], ys[tr], n=k)
    pred = p.score(Xs[te]).argmax(axis=1)
    bals_lin.append(balanced_accuracy_score(ys[te], pred))
print(f'LINEAR: bal={np.mean(bals_lin):.3f}')

# CNN
class CNN(nn.Module):
    def __init__(self, cin, tin, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(256, k))
    def forward(self,x): return self.net(x)

bals_cnn = []
for fold, (tr,te) in enumerate(skf.split(Xs, ys)):
    Xtr = torch.from_numpy(Xs[tr]).float()
    ytr = torch.from_numpy(ys[tr]).long()
    Xte = torch.from_numpy(Xs[te]).float()
    mean, std = Xtr.mean(0,keepdim=True), Xtr.std(0,keepdim=True)+1e-6
    Xtr = (Xtr-mean)/std; Xte = (Xte-mean)/std
    model = CNN(Xs.shape[1], Xs.shape[2], k)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lf = nn.CrossEntropyLoss()
    for epoch in range(100):
        model.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(idx), 32):
            b = idx[i:i+32]
            opt.zero_grad()
            lf(model(Xtr[b]), ytr[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    bal = balanced_accuracy_score(ys[te], pred)
    bals_cnn.append(bal)
    print(f'  cnn fold {fold}: bal={bal:.3f}')
print(f'CNN: bal={np.mean(bals_cnn):.3f}')

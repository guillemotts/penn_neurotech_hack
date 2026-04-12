"""
probe_4card.py -- Linear probe + CNN on ljoy 4-cardinal subset only.
Pool all data/newdata_apr12/smoke_Ani_seed*.h5 (excluding seed 6).
"""
import numpy as np, sys, subprocess, torch, torch.nn as nn
from pathlib import Path

sys.path.insert(0, "src")
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory
from penn_nh.eval.calibration_model import predict_bitrate_curve, HickHymanFit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

# ── 0. Load & pool all sessions (exclude seed 6) ────────────────────
data_dir = Path("data/newdata_apr12")
seeds = [s for s in range(1, 16) if s != 6]

all_X, all_y = [], []
for s in seeds:
    h5 = data_dir / f"smoke_Ani_seed{s}.h5"
    cj = data_dir / f"cues_Ani_seed{s}.json"
    if not h5.exists() or not cj.exists():
        continue
    out = f"/tmp/probe4c_s{s}.npz"
    if not Path(out).exists():
        r = subprocess.run(
            ["python", "scripts/load_h5_session.py",
             "--h5", str(h5), "--cues", str(cj), "--out", out],
            capture_output=True,
        )
        if r.returncode != 0:
            print(f"SKIP seed {s}: {r.stderr.decode()[-200:]}")
            continue
    d = np.load(out)
    all_X.append(d["X"])
    all_y.append(d["y"])
    print(f"seed {s}: {d['X'].shape[0]} trials")

if not all_X:
    print("NO DATA")
    sys.exit(1)

X = np.concatenate(all_X)
y = np.concatenate(all_y)
print(f"\npooled raw: X={X.shape} unique_labels={sorted(set(y.tolist()))}")

# ── 1. Filter to ljoy 4-cardinal {0,2,4,6}, remap to {0,1,2,3} ────
CARD_IDS = [0, 2, 4, 6]
REMAP = {0: 0, 2: 1, 4: 2, 6: 3}
mask = np.isin(y, CARD_IDS)
X = X[mask, :64, :]
y = np.array([REMAP[v] for v in y[mask]])
N_CLS = 4

counts = np.bincount(y, minlength=N_CLS)
print(f"\n4-cardinal subset:")
for i in range(N_CLS):
    print(f"  class {i} (ljoy_{CARD_IDS[i]}): {counts[i]} trials")
print(f"  total: {len(y)} trials")

# ── Feature extraction (bin_ms=50, spike_threshold_k=2.0) ───────────
fs = FeatureStack(FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0))
print("\nextracting features ...")
Xf = np.stack([fs.transform(X[i].astype(np.float32)) for i in range(len(X))])
print(f"Xf={Xf.shape}")

# ── 2. 5-fold stratified CV ─────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
bals_lin, bals_cnn = [], []

for fold, (tr, te) in enumerate(skf.split(Xf, y)):
    # ── 3a. Linear probe ────────────────────────────────────────────
    probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xf[tr], y[tr], n=N_CLS)
    pred_lin = probe.score(Xf[te]).argmax(axis=1)
    bal_lin = balanced_accuracy_score(y[te], pred_lin)
    bals_lin.append(bal_lin)

    # ── 3b. CNN (2-layer 1D, 400 ch, kernel 5, BN+ReLU, GAP, 4-head) ──
    class CNN(nn.Module):
        def __init__(self, cin, k):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(cin, 400, 5, padding=2),
                nn.BatchNorm1d(400), nn.ReLU(),
                nn.Conv1d(400, 400, 5, padding=2),
                nn.BatchNorm1d(400), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(400, k),
            )

        def forward(self, x):
            return self.net(x)

    Xtr = torch.from_numpy(Xf[tr]).float()
    ytr = torch.from_numpy(y[tr]).long()
    Xte = torch.from_numpy(Xf[te]).float()

    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    model = CNN(Xf.shape[1], N_CLS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    n_epochs = 30
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.0, total_iters=n_epochs,
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(idx), 64):
            b = idx[i : i + 64]
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred_cnn = model(Xte).argmax(1).numpy()
    bal_cnn = balanced_accuracy_score(y[te], pred_cnn)
    bals_cnn.append(bal_cnn)

    print(f"fold {fold}: linear={bal_lin:.3f}  cnn={bal_cnn:.3f}")

# ── 4. Summary ──────────────────────────────────────────────────────
mean_lin = np.mean(bals_lin)
mean_cnn = np.mean(bals_cnn)
print(f"\n{'='*50}")
print(f"LINEAR  per-fold: {[f'{b:.3f}' for b in bals_lin]}")
print(f"LINEAR  mean bal acc: {mean_lin:.3f}  (chance=0.250)")
print(f"CNN     per-fold: {[f'{b:.3f}' for b in bals_cnn]}")
print(f"CNN     mean bal acc: {mean_cnn:.3f}  (chance=0.250)")

# ── 5. Shuffle control (permuted labels) ──────────────────────────────
rng = np.random.RandomState(0)
y_shuf = rng.permutation(y)

bals_shuf_lin, bals_shuf_cnn = [], []
for fold, (tr, te) in enumerate(skf.split(Xf, y_shuf)):
    # linear probe
    probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xf[tr], y_shuf[tr], n=N_CLS)
    pred_lin = probe.score(Xf[te]).argmax(axis=1)
    bal_lin = balanced_accuracy_score(y_shuf[te], pred_lin)
    bals_shuf_lin.append(bal_lin)

    # CNN
    Xtr = torch.from_numpy(Xf[tr]).float()
    ytr = torch.from_numpy(y_shuf[tr]).long()
    Xte = torch.from_numpy(Xf[te]).float()

    mean = Xtr.mean(0, keepdim=True)
    std = Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    model = CNN(Xf.shape[1], N_CLS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.0, total_iters=n_epochs,
    )

    for epoch in range(n_epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(idx), 64):
            b = idx[i : i + 64]
            opt.zero_grad()
            loss_fn(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred_cnn = model(Xte).argmax(1).numpy()
    bal_cnn = balanced_accuracy_score(y_shuf[te], pred_cnn)
    bals_shuf_cnn.append(bal_cnn)

    print(f"shuffle fold {fold}: linear={bal_lin:.3f}  cnn={bal_cnn:.3f}")

mean_shuf_lin = np.mean(bals_shuf_lin)
mean_shuf_cnn = np.mean(bals_shuf_cnn)
print(f"\n{'='*50}")
print(f"SHUFFLE LINEAR  per-fold: {[f'{b:.3f}' for b in bals_shuf_lin]}")
print(f"SHUFFLE LINEAR  mean bal acc: {mean_shuf_lin:.3f}  (chance=0.250)")
print(f"SHUFFLE CNN     per-fold: {[f'{b:.3f}' for b in bals_shuf_cnn]}")
print(f"SHUFFLE CNN     mean bal acc: {mean_shuf_cnn:.3f}  (chance=0.250)")

# ── 6. Shenoy B (N=4, best accuracy, t=0.5s placeholder) ───────────
best_acc = max(mean_lin, mean_cnn)
rt_s = 0.5
fit = HickHymanFit(p0=best_acc, c=0.0, r2=1.0)
curve = predict_bitrate_curve(fit, [N_CLS], rt_s)
B = curve[0]["bitrate"]
print(f"\nShenoy B (N={N_CLS}, acc={best_acc:.3f}, t={rt_s}s): {B:.2f} bits/s")

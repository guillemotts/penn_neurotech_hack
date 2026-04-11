import numpy as np, sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from load_training_csv import SYMBOLS
from penn_nh.features.stack import FeatureConfigLowRate, LowRateFeatureStack
from penn_nh.decoder.linear_probe import LinearProbeFactory

d = np.load('/tmp/ani1_out.npz')
X, y = d['X'], d['y']
print(f"X={X.shape} y={y.shape} n_classes={len(np.unique(y))}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
conf = np.zeros((22, 22), dtype=int)
for fold, (tr, te) in enumerate(skf.split(X, y)):
    fs = LowRateFeatureStack(FeatureConfigLowRate(fs_hz=103.73))
    fs.fit_zscore([X[i] for i in tr])
    Xtr = np.stack([fs.transform(X[i]) for i in tr])
    Xte = np.stack([fs.transform(X[i]) for i in te])
    probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xtr, y[tr], n=22)
    pred = probe.score(Xte).argmax(axis=1)
    acc = (pred == y[te]).mean()
    accs.append(acc)
    for t_, p_ in zip(y[te], pred):
        conf[t_, p_] += 1
    print(f"fold {fold}: acc={acc:.3f}")

print(f"\nMEAN={np.mean(accs):.3f} STD={np.std(accs):.3f}  chance=0.045")

recall = np.array([conf[i, i] / max(1, conf[i].sum()) for i in range(22)])
order = np.argsort(recall)
print("\nPer-symbol recall (ascending):")
for i in order:
    sym = f"{SYMBOLS[i][0]}_{SYMBOLS[i][1]}"
    print(f"  {sym:<15} recall={recall[i]:.2f}  n={conf[i].sum()}")

off = conf.copy()
np.fill_diagonal(off, 0)
print("\nTop 5 confusions:")
flat = [(off[i, j], i, j) for i in range(22) for j in range(22)]
flat.sort(reverse=True)
for c, i, j in flat[:5]:
    si = f"{SYMBOLS[i][0]}_{SYMBOLS[i][1]}"
    sj = f"{SYMBOLS[j][0]}_{SYMBOLS[j][1]}"
    print(f"  {si} -> {sj}: {c}")

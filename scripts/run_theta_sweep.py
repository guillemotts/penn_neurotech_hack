import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from penn_nh.decoder.linear_probe import LinearProbeFactory

def sweep_theta(X, y, n, thetas, rt_per_symbol_s=0.8, n_folds=5):
    """Assumes X, y already restricted to n classes remapped 0..n-1."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    fold_probs = []
    fold_labels = []
    for tr, te in skf.split(X, y):
        probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(X[tr], y[tr], n=n)
        logp = probe.score(X[te])
        probs = np.exp(logp - logp.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        fold_probs.append(probs)
        fold_labels.append(y[te])
    all_probs = np.concatenate(fold_probs)
    all_labels = np.concatenate(fold_labels)
    M = len(all_labels)
    sorted_probs = np.sort(all_probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margins = top1 - top2
    pred = all_probs.argmax(axis=1)
    results = []
    for theta in thetas:
        accept = margins > theta
        sc = int(((pred == all_labels) & accept).sum())
        si = int(((pred != all_labels) & accept).sum())
        nop = int((~accept).sum())
        t = M * rt_per_symbol_s
        B = np.log2(n - 1) * max(sc - si, 0) / max(t, 1e-9) if n >= 3 else 0.0
        results.append({"theta": float(theta), "sc": sc, "si": si, "no_op": nop, "B": B})
    return results

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--n", type=int, required=True,
                   help="Restrict to top-N most frequent classes")
    p.add_argument("--thetas", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--rt-s", type=float, default=0.8)
    args = p.parse_args(argv)
    d = np.load(args.npz)
    X, y = d["X"], d["y"]
    counts = np.bincount(y, minlength=22)
    keep = set(np.argsort(-counts)[:args.n].tolist())
    mask = np.array([v in keep for v in y])
    Xs = X[mask]
    remap = {s: i for i, s in enumerate(sorted(keep))}
    ys = np.array([remap[v] for v in y[mask]])
    results = sweep_theta(Xs, ys, args.n, args.thetas, args.rt_s)
    print(f"n={args.n}  M={len(ys)}")
    print(f"{'theta':>6} {'sc':>5} {'si':>5} {'noop':>5} {'B_bps':>7}")
    for r in results:
        print(f"{r['theta']:>6.2f} {r['sc']:>5d} {r['si']:>5d} {r['no_op']:>5d} {r['B']:>7.3f}")
    best = max(results, key=lambda r: r["B"])
    print(f"\nBEST theta={best['theta']:.2f} B={best['B']:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

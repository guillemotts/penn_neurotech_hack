import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from penn_nh.decoder.linear_probe import LinearProbeFactory

def inline_bitrate(n, sc, si, t):
    if n < 3: return 0.0
    return np.log2(n - 1) * max(sc - si, 0) / max(t, 1e-9)

def top_n_symbols_by_count(y, n):
    counts = np.bincount(y, minlength=22)
    return np.argsort(-counts)[:n]

def sweep_n(X, y, n_values, rt_per_symbol_s, n_folds=5):
    results = []
    for n in n_values:
        keep = set(top_n_symbols_by_count(y, n).tolist())
        mask = np.array([yi in keep for yi in y])
        Xs = X[mask]; ys_raw = y[mask]
        remap = {s: i for i, s in enumerate(sorted(keep))}
        ys = np.array([remap[v] for v in ys_raw])
        if len(Xs) < n * 5 or len(np.unique(ys)) < n:
            results.append({"n": n, "acc": 0.0, "sc": 0, "si": 0, "t": 0.0, "bitrate": 0.0, "skipped": True})
            continue
        k_folds = min(n_folds, int(np.bincount(ys).min()))
        if k_folds < 2:
            results.append({"n": n, "acc": 0.0, "sc": 0, "si": 0, "t": 0.0, "bitrate": 0.0, "skipped": True})
            continue
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0)
        accs = []; sc_total = si_total = 0
        for tr, te in skf.split(Xs, ys):
            probe = LinearProbeFactory(C=1.0, max_iter=2000).fit(Xs[tr], ys[tr], n=n)
            pred = probe.score(Xs[te]).argmax(axis=1)
            correct = int((pred == ys[te]).sum())
            sc_total += correct
            si_total += int(len(te) - correct)
            accs.append(correct / len(te))
        t_total = (sc_total + si_total) * rt_per_symbol_s
        results.append({
            "n": n, "acc": float(np.mean(accs)),
            "sc": sc_total, "si": si_total, "t": t_total,
            "bitrate": inline_bitrate(n, sc_total, si_total, t_total),
            "skipped": False,
        })
    return results

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--n-values", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16, 22])
    p.add_argument("--rt-s", type=float, default=0.8)
    args = p.parse_args(argv)
    d = np.load(args.npz)
    X, y = d["X"], d["y"]
    results = sweep_n(X, y, args.n_values, args.rt_s)
    print(f"{'N':>3} {'acc':>6} {'sc':>5} {'si':>5} {'t_s':>7} {'B_bps':>7}")
    for r in results:
        tag = " SKIP" if r["skipped"] else ""
        print(f"{r['n']:>3} {r['acc']:>6.3f} {r['sc']:>5d} {r['si']:>5d} {r['t']:>7.1f} {r['bitrate']:>7.3f}{tag}")
    live = [r for r in results if not r["skipped"]]
    if live:
        best = max(live, key=lambda r: r["bitrate"])
        print(f"\nARGMAX_N={best['n']} B={best['bitrate']:.3f} bps acc={best['acc']:.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

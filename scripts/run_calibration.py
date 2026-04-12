#!/usr/bin/env python
"""Per-session calibration: sweep N, fit Hick-Hyman, find (N*, theta*), prune symbols, write calib.json."""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from penn_nh.decoder.linear_probe import LinearProbeFactory
from penn_nh.eval.bitrate import shenoy_b
from penn_nh.eval.calibration_model import (
    fit_hick_hyman,
    panzeri_treves_mi,
    per_symbol_breakeven,
    predict_bitrate_curve,
)
from penn_nh.features.stack import FeatureConfig, FeatureStack

# ── Symbol groupings by alphabet size ──────────────────────────────────
# Indices: 0-7 ljoy, 8-15 rjoy, 16-21 buttons (UP,DOWN,LEFT,RIGHT,LB,RB)
GROUPINGS = {
    4: [0, 2, 4, 6],
    6: [16, 17, 18, 19, 20, 21],
    8: [0, 2, 4, 6, 16, 17, 18, 19],
    10: [0, 2, 4, 6, 16, 17, 18, 19, 20, 21],
    14: [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21],
}


def _load_data(recording_path, cue_json_path):
    """Load recording .npz and cue JSON. Returns X, y, rt_s."""
    d = np.load(recording_path)
    X, y = d["X"], d["y"]

    with open(cue_json_path) as f:
        cue = json.load(f)

    if "isi_s" in cue:
        rt_s = float(cue["isi_s"])
    else:
        ts = [t["t_mono_ns"] for t in cue["trials"]]
        rt_s = float(np.median(np.diff(ts))) * 1e-9 if len(ts) >= 2 else 1.0

    return X, y, rt_s


def _extract_features(X, seed):
    """(trials, >=64ch, T) -> (trials, S*C, n_bins) via FeatureStack."""
    Xn = X[:, :64, :].astype(np.float32)
    cfg = FeatureConfig(fs_hz=32000.0, bin_ms=50.0, spike_threshold_k=2.0)
    stack = FeatureStack(cfg)

    raw_feats = [stack.transform(Xn[i]) for i in range(len(Xn))]
    stack.fit_zscore(raw_feats)
    feats = [stack.transform(Xn[i]) for i in range(len(Xn))]
    return np.stack(feats)


def _cv_sweep(Xf, y, seed):
    """5-fold stratified CV per N in GROUPINGS. Returns measured_ns, measured_accs, confusions."""
    measured_ns, measured_accs, confusions = [], [], {}

    for N, sym_ids in sorted(GROUPINGS.items()):
        mask = np.isin(y, sym_ids)
        if mask.sum() == 0:
            continue

        id_to_idx = {s: i for i, s in enumerate(sym_ids)}
        Xm, ym = Xf[mask], np.array([id_to_idx[v] for v in y[mask]])

        counts = np.bincount(ym, minlength=N)
        if counts.min() < 5:
            warnings.warn(f"N={N}: skipping, min trials/class={counts.min()}")
            continue

        n_folds = min(5, int(counts.min()))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        C = np.zeros((N, N), dtype=int)

        for tr, te in skf.split(Xm, ym):
            fitted = LinearProbeFactory(
                C=1.0, max_iter=1000, random_state=seed
            ).fit(Xm[tr], ym[tr], n=N)
            preds = fitted.score(Xm[te]).argmax(axis=1)
            for t_lab, p_lab in zip(ym[te], preds):
                C[t_lab, p_lab] += 1

        acc = float(np.trace(C) / C.sum())
        measured_ns.append(N)
        measured_accs.append(acc)
        confusions[N] = C
        print(f"  N={N:2d}: acc={acc:.3f} n={mask.sum()}")

    return measured_ns, measured_accs, confusions


def _find_optimal_n(measured_ns, measured_accs, rt_s):
    """Hick-Hyman fit -> B(N) curve -> N*."""
    if len(measured_ns) < 2:
        raise ValueError(f"Need >=2 measured N values, got {len(measured_ns)}")

    hh = fit_hick_hyman(measured_ns, measured_accs)
    n_grid = list(range(2, 15))
    curve = predict_bitrate_curve(hh, n_grid, rt_s)

    best = max(curve, key=lambda c: c["bitrate"])
    n_star = min(GROUPINGS, key=lambda n: abs(n - best["n"]))
    return hh, curve, n_star


def _prune_symbols(confusions, n_star, rt_s, Xf, y, seed):
    """Drop symbols with negative break-even contribution. Re-run CV if pruned."""
    C = confusions[n_star]
    sym_ids = GROUPINGS[n_star]

    result = per_symbol_breakeven(C, rt_s, n_star)
    keep_idx = result["keep"]
    contribs = result["contributions"]

    pruned_ids = [sym_ids[i] for i in keep_idx]
    n_pruned = len(pruned_ids)

    if n_pruned == len(sym_ids) or n_pruned < 3:
        return sym_ids, C, contribs, {s: i for i, s in enumerate(sym_ids)}

    print(f"  Pruned {len(sym_ids)} -> {n_pruned} symbols")

    # Re-run CV on pruned set
    mask = np.isin(y, pruned_ids)
    id_to_idx = {s: i for i, s in enumerate(pruned_ids)}
    Xm, ym = Xf[mask], np.array([id_to_idx[v] for v in y[mask]])
    counts = np.bincount(ym, minlength=n_pruned)
    n_folds = min(5, int(counts.min()))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    C_new = np.zeros((n_pruned, n_pruned), dtype=int)
    for tr, te in skf.split(Xm, ym):
        fitted = LinearProbeFactory(
            C=1.0, max_iter=1000, random_state=seed
        ).fit(Xm[tr], ym[tr], n=n_pruned)
        preds = fitted.score(Xm[te]).argmax(axis=1)
        for t_lab, p_lab in zip(ym[te], preds):
            C_new[t_lab, p_lab] += 1

    return pruned_ids, C_new, contribs, id_to_idx


def _tune_theta(Xf, y, symbol_ids, seed, rt_s):
    """Grid-search theta* that maximises achieved bitrate.

    Returns (theta*, B, C_accepted, n_heldout).
    """
    n = len(symbol_ids)
    mask = np.isin(y, symbol_ids)
    id_to_idx = {s: i for i, s in enumerate(symbol_ids)}
    Xm, ym = Xf[mask], np.array([id_to_idx[v] for v in y[mask]])

    # 80/20 split
    rng = np.random.RandomState(seed)
    idx = np.arange(len(Xm))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr, te = idx[:split], idx[split:]

    fitted = LinearProbeFactory(
        C=1.0, max_iter=1000, random_state=seed
    ).fit(Xm[tr], ym[tr], n=n)
    softmax = np.exp(fitted.score(Xm[te]))  # (M, n) probabilities
    preds = softmax.argmax(axis=1)
    true_labels = ym[te]
    n_heldout = len(te)

    best_b, best_theta, best_C = -1.0, 0.0, np.zeros((n, n), dtype=int)
    for theta in np.linspace(0.0, 0.9, 19):
        accepted = softmax.max(axis=1) >= theta
        if accepted.sum() == 0:
            continue
        sc = int((preds[accepted] == true_labels[accepted]).sum())
        si = int(accepted.sum()) - sc
        b = (
            np.log2(n - 1) * max(sc - si, 0) / (n_heldout * rt_s)
            if n >= 3
            else 0.0
        )
        if b > best_b:
            best_b, best_theta = b, float(theta)
            C_acc = np.zeros((n, n), dtype=int)
            for tl, pl in zip(true_labels[accepted], preds[accepted]):
                C_acc[tl, pl] += 1
            best_C = C_acc

    return best_theta, best_b, best_C, n_heldout


def _make_plot(path, curve, n_star, measured_ns, measured_accs, contribs, sym_ids):
    """Two-panel PNG: B(N) curve + per-symbol break-even bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: B(N) curve
    ns = [c["n"] for c in curve]
    bs = [c["bitrate"] for c in curve]
    ax1.plot(ns, bs, "b-o", label="B_pred(N)")
    ax1.axvline(n_star, color="r", ls="--", label=f"N*={n_star}")
    ax1b = ax1.twinx()
    ax1b.scatter(
        measured_ns, measured_accs, c="orange", marker="s", zorder=5, label="measured acc"
    )
    ax1.set_xlabel("N")
    ax1.set_ylabel("Bitrate (bits/s)")
    ax1b.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")
    ax1b.legend(loc="upper right")
    ax1.set_title("Predicted Bitrate Curve")

    # Right: per-symbol bars
    if contribs:
        vals = [c["contribution"] for c in contribs]
        colors = ["green" if c["keep"] else "red" for c in contribs]
        labels = [
            str(sym_ids[c["class"]]) if c["class"] < len(sym_ids) else str(c["class"])
            for c in contribs
        ]
        ax2.bar(range(len(vals)), vals, color=colors, tick_label=labels)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xlabel("Symbol")
        ax2.set_ylabel("Bit contribution")
        ax2.set_title(f"Per-symbol Break-even (N={n_star})")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-session calibration pipeline")
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--cue-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("=== Calibration Pipeline ===")

    # 1. Load
    print("[1] Loading data...")
    X, y, rt_s = _load_data(args.recording, args.cue_json)
    print(f"  X={X.shape} y={y.shape} rt_s={rt_s:.3f}s")

    # 2. Features
    print("[2] Extracting features...")
    Xf = _extract_features(X, args.seed)
    print(f"  Features: {Xf.shape}")

    # 3. N-sweep
    print("[3] Cross-validated N-sweep...")
    measured_ns, measured_accs, confusions = _cv_sweep(Xf, y, args.seed)

    # 4. Hick-Hyman -> N*
    print("[4] Fitting Hick-Hyman...")
    hh, curve, n_star = _find_optimal_n(measured_ns, measured_accs, rt_s)
    print(f"  p0={hh.p0:.3f} c={hh.c:.4f} r2={hh.r2:.3f} -> N*={n_star}")

    # Ensure N* was actually measured
    if n_star not in confusions:
        n_star = min(confusions, key=lambda n: abs(n - n_star))
        print(f"  Snapped to measured N*={n_star}")

    # 5. Prune
    print("[5] Per-symbol pruning...")
    pruned_ids, C_pruned, contribs, _ = _prune_symbols(
        confusions, n_star, rt_s, Xf, y, args.seed
    )
    n_final = len(pruned_ids)
    print(f"  Final alphabet: {n_final} symbols {pruned_ids}")

    # 6. Theta tuning
    print("[6] Theta tuning...")
    theta_star, b_theta, C_accepted, n_heldout = _tune_theta(
        Xf, y, pruned_ids, args.seed, rt_s
    )
    print(f"  theta*={theta_star:.2f} B(theta*)={b_theta:.3f}")

    # 7. PT MI
    mi = (
        panzeri_treves_mi(C_accepted)
        if C_accepted.sum() > 0
        else {"mi": 0.0, "mi_uncorrected": 0.0, "bias": 0.0}
    )
    print(f"[7] MI={mi['mi']:.3f} bits")

    # 8. Final Shenoy B (time includes rejected trials)
    if C_accepted.sum() > 0 and n_final >= 3:
        sc = int(np.trace(C_accepted))
        si = int(C_accepted.sum()) - sc
        shenoy_b_val = float(shenoy_b(sc, si, n_final, t=n_heldout * rt_s))
    else:
        shenoy_b_val = 0.0
    print(f"[8] Shenoy B = {shenoy_b_val:.3f} bits/s")

    # 9. Write calib.json
    calib = {
        "N_star": n_star,
        "N_final": n_final,
        "theta_star": theta_star,
        "pruned_symbol_ids": pruned_ids,
        "shenoy_b_bits_per_s": shenoy_b_val,
        "mi_bits": mi["mi"],
        "mi_uncorrected_bits": mi["mi_uncorrected"],
        "mi_bias": mi["bias"],
        "hick_hyman": {"p0": hh.p0, "c": hh.c, "r2": hh.r2},
        "rt_s": rt_s,
        "measured_ns": measured_ns,
        "measured_accs": measured_accs,
        "bitrate_curve": curve,
        "contributions": contribs,
        "confusion_at_theta": C_accepted.tolist(),
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2, cls=_NumpyEncoder)
    print(f"[9] Saved {args.out}")

    # 10. Plot
    if args.plot:
        _make_plot(
            args.plot, curve, n_star, measured_ns, measured_accs, contribs, GROUPINGS[n_star]
        )

    print("=== Done ===")
    return calib


if __name__ == "__main__":
    main()

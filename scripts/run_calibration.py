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

    hh = fit_hick_hyman(measured_ns, measured_accs, rt_s=rt_s)
    n_grid = list(range(2, 15))
    curve = predict_bitrate_curve(hh, n_grid, rt_s)

    # Snap the continuous N* to the nearest measured grouping
    n_star = min(GROUPINGS, key=lambda n: abs(n - hh.n_star))

    print(f"  OLS:   p0={hh.p0:.3f}  c={hh.c:.4f}  R²={hh.r2:.3f}")
    print(f"  N* analytic (parabola vertex) = {hh.n_star_analytic}")
    print(f"  N* sweep    (argmax B(N))     = {hh.n_star_sweep}")
    if hh.safety_fallback:
        print(f"  ⚠ Safety fallback triggered (accuracy < 0.75 at sweep N*)")
    if hh.bayes_n_star is not None:
        agree = "✓" if hh.bayes_n_star == hh.n_star_sweep else "⚠ disagree"
        print(f"  Bayes: p0={hh.bayes_p0:.3f}  c={hh.bayes_c:.4f}  N*={hh.bayes_n_star}  [{agree}]")
    print(f"  → snapped to measured grouping N*={n_star}")

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


def _make_plot(
    path, curve, n_star, measured_ns, measured_accs,
    contribs, sym_ids, hh, confusion_at_n_star, rt_s,
):
    """
    Four-panel calibration report:
      [0,0] B(N) curve  – OLS + Bayesian overlay, analytic N* and sweep N* marked
      [0,1] Confusion matrix heatmap at N* (row-normalised)
      [1,0] Per-symbol break-even bars  – green=keep, red=prune
      [1,1] Speed-accuracy operating point on iso-bit-rate curves
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BCI Calibration Analytics", fontsize=14, fontweight="bold", y=1.01)

    # ── [0,0] B(N) curve ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    ns  = [c["n"] for c in curve]
    bs  = [c["bitrate"] for c in curve]
    acs = [c["accuracy"] for c in curve]

    ax.plot(ns, bs, "b-o", ms=4, label="B̂(N) – OLS")

    # Bayesian overlay (if parameters available)
    if hh.bayes_p0 is not None and hh.bayes_c is not None:
        bs_bayes = [
            np.log2(max(n - 1, 1))
            * max(2 * max(0, min(1, hh.bayes_p0 - hh.bayes_c * np.log2(n))) - 1, 0)
            / rt_s
            for n in ns
        ]
        ax.plot(ns, bs_bayes, "m--", lw=1.2, alpha=0.7, label="B̂(N) – Bayes")

    # Mark N* lines
    ax.axvline(hh.n_star_sweep, color="red",    ls="--", lw=1.5,
               label=f"N* sweep = {hh.n_star_sweep}")
    ax.axvline(hh.n_star_analytic, color="darkorange", ls=":", lw=1.5,
               label=f"N* analytic = {hh.n_star_analytic}")
    if hh.bayes_n_star and hh.bayes_n_star != hh.n_star_sweep:
        ax.axvline(hh.bayes_n_star, color="purple", ls="-.", lw=1.2,
                   label=f"N* Bayes = {hh.bayes_n_star}")

    ax2b = ax.twinx()
    ax2b.scatter(measured_ns, measured_accs, c="orange", marker="s",
                 zorder=5, s=60, label="Measured acc")
    ax2b.plot(ns, acs, "orange", lw=1, alpha=0.4)
    ax2b.set_ylabel("Accuracy p(N)", color="darkorange")
    ax2b.tick_params(axis="y", labelcolor="darkorange")
    ax2b.set_ylim(0.4, 1.05)

    ax.set_xlabel("Alphabet size N")
    ax.set_ylabel("Bit rate (bits/s)")
    ax.set_title("B(N) Curve — Optimal Alphabet Size")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7, loc="upper left")

    if hh.safety_fallback:
        ax.text(0.98, 0.05, "⚠ safety fallback active",
                transform=ax.transAxes, ha="right", color="red", fontsize=8)

    # ── [0,1] Confusion matrix ────────────────────────────────────────────────
    ax = axes[0, 1]
    C = np.array(confusion_at_n_star, dtype=float)
    if C.size > 0 and C.sum() > 0:
        row_sums = C.sum(axis=1, keepdims=True)
        C_norm = np.divide(C, row_sums, where=row_sums > 0)
        im = ax.imshow(C_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, label="p(predicted | true)")

        n_sym = C.shape[0]
        sym_labels = [str(s) for s in sym_ids[:n_sym]] if sym_ids else [str(i) for i in range(n_sym)]
        ax.set_xticks(range(n_sym))
        ax.set_yticks(range(n_sym))
        ax.set_xticklabels(sym_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(sym_labels, fontsize=7)

        # Annotate cells with p values
        for i in range(n_sym):
            for j in range(n_sym):
                val = C_norm[i, j]
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if val > 0.55 else "black")

        # Diagonal accuracy overlay
        diag_acc = np.trace(C) / C.sum()
        ax.set_title(f"Confusion Matrix — N={C.shape[0]}  (acc={diag_acc:.1%})")
    else:
        ax.text(0.5, 0.5, "No confusion data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted symbol")
    ax.set_ylabel("True symbol")

    # ── [1,0] Per-symbol break-even bars ──────────────────────────────────────
    ax = axes[1, 0]
    if contribs:
        vals   = [c["contribution"] for c in contribs]
        colors = ["#2ecc71" if c["keep"] else "#e74c3c" for c in contribs]
        labels = [
            str(sym_ids[c["class"]]) if c["class"] < len(sym_ids) else str(c["class"])
            for c in contribs
        ]
        x_pos = range(len(vals))
        ax.bar(x_pos, vals, color=colors, edgecolor="white", lw=0.5)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Symbol")
        ax.set_ylabel("Marginal bit-rate contribution (bits/s)")
        ax.set_title(
            f"Per-symbol Break-even — N={n_star}\n"
            "Green = net positive (keep)  |  Red = net negative (prune)"
        )
        # Annotation: how many pruned
        n_pruned = sum(1 for c in contribs if not c["keep"])
        if n_pruned:
            ax.text(0.98, 0.98, f"{n_pruned} symbol(s) below break-even",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="#e74c3c")
    else:
        ax.text(0.5, 0.5, "No contribution data", ha="center", va="center",
                transform=ax.transAxes)

    # ── [1,1] Speed-accuracy operating point on iso-bit-rate curves ───────────
    ax = axes[1, 1]
    p_grid = np.linspace(0.51, 1.0, 200)   # accuracy axis
    r_grid = np.linspace(0.0, 2.0 / rt_s, 200)  # selections/s axis
    P, R = np.meshgrid(p_grid, r_grid)

    # Draw iso-bit-rate contours for each measured N at its N*
    # B = log2(N-1) * R * (2p-1)
    # We show contours for B = 1, 2, 3, ... bits/s (alphabet-agnostic)
    b_levels = [0.5, 1, 2, 3, 4, 5, 6, 8]
    for b_target in b_levels:
        # R = B / (2p - 1) for the B-agnostic version (log2(N-1) absorbed into y-axis)
        r_iso = b_target / np.maximum(2 * p_grid - 1, 1e-6)
        r_iso = np.clip(r_iso, 0, r_grid.max())
        ax.plot(p_grid, r_iso, "lightgray", lw=0.8, zorder=1)
        # Label at right edge
        right_r = b_target / max(2 * p_grid[-1] - 1, 1e-6)
        if right_r <= r_grid.max():
            ax.text(p_grid[-1] + 0.002, right_r, f"{b_target}b/s",
                    fontsize=6, va="center", color="gray")

    # Plot each measured operating point (one per N)
    cmap = plt.cm.get_cmap("tab10", len(measured_ns))
    for idx, (n, p, acc) in enumerate(zip(measured_ns, measured_accs, measured_accs)):
        r_meas = 1.0 / rt_s  # fixed ISI → fixed rate
        # effective bitrate at this operating point
        b_eff = np.log2(max(n - 1, 1)) * r_meas * max(2 * p - 1, 0)
        ax.scatter(p, r_meas, color=cmap(idx), s=90, zorder=5,
                   label=f"N={n}  B={b_eff:.2f} b/s")
        ax.annotate(f"N={n}", (p, r_meas),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=cmap(idx))

    # Mark N* operating point
    p_nstar = max(0.0, min(1.0, hh.p0 - hh.c * np.log2(n_star)))
    r_nstar = 1.0 / rt_s
    b_nstar = np.log2(max(n_star - 1, 1)) * r_nstar * max(2 * p_nstar - 1, 0)
    ax.scatter(p_nstar, r_nstar, marker="*", s=220, color="red", zorder=6,
               label=f"N*={n_star}  B={b_nstar:.2f} b/s")

    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0, r_grid.max() * 1.05)
    ax.set_xlabel("Accuracy p(N)")
    ax.set_ylabel("Selection rate R (selections/s)")
    ax.set_title("Speed–Accuracy Operating Point\n(iso-bit-rate curves)")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
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
        "hick_hyman": {
            "p0": hh.p0, "c": hh.c, "r2": hh.r2,
            "n_star_analytic": hh.n_star_analytic,
            "n_star_sweep": hh.n_star_sweep,
            "safety_fallback": hh.safety_fallback,
            "bayes_p0": hh.bayes_p0, "bayes_c": hh.bayes_c,
            "bayes_n_star": hh.bayes_n_star,
        },
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
        # Use the confusion matrix at N* (pre-theta, so cell counts are richer)
        C_for_plot = confusions.get(n_star, C_accepted)
        _make_plot(
            args.plot, curve, n_star, measured_ns, measured_accs,
            contribs, GROUPINGS.get(n_star, list(range(n_star))),
            hh, C_for_plot.tolist(), rt_s,
        )

    print("=== Done ===")
    return calib


if __name__ == "__main__":
    main()

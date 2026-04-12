"""
sweep_postproc.py -- Sweep post-hoc smoother/discretizer hyperparams and report proxy bitrate.

Usage:
    python -m src.decoder.sweep_postproc --mode easy
    python -m src.decoder.sweep_postproc --mode hard
"""
import argparse
import csv
import json
import os

import numpy as np

from decoder.postprocess import (
    causal_moving_average,
    discretize_stick,
    discretize_button,
    combine,
    proxy_bitrate,
)

K_VALUES = [10, 15, 20, 30, 50]
STICK_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BUTTON_THRESHOLDS = [0.20, 0.35, 0.50, 0.65]
N_VALUES = [4, 6, 8]
BIN_MS = 10  # ms per bin

# Assumption: Hick-Hyman floor reaction time for N~4 symbols.
# Every emission pays base_rt + K*bin_seconds (cue → smoothed-emission delay).
# Every no-op pays base_rt (user acts, decoder doesn't emit, user waits out trial).
# Grader's actual timing may differ; this is a modelling assumption.
BASE_RT_SEC = 0.30


def main(args):
    mode = args.mode
    out_dir = f"outputs/{mode}_ridge_rms_ctx20"

    Y_pred = np.load(os.path.join(out_dir, "Y_pred.npy"))
    Y_true = np.load(os.path.join(out_dir, "Y_true.npy"))
    with open(os.path.join(out_dir, "val_meta.json")) as f:
        meta = json.load(f)

    bin_seconds = meta["bin_seconds"]
    print(f"Mode: {mode}, n_bins={meta['n_bins']}, bin_seconds={bin_seconds}")
    print(f"Y_pred: {Y_pred.shape}, Y_true: {Y_true.shape}")
    print(f"base_rt_sec={BASE_RT_SEC}")

    rows = []

    for K in K_VALUES:
        Y_pred_sm = causal_moving_average(Y_pred, K)
        Y_true_sm = causal_moving_average(Y_true, K)

        for N in N_VALUES:
            for stick_thresh in STICK_THRESHOLDS:
                for button_thresh in BUTTON_THRESHOLDS:
                    gt_stick = discretize_stick(Y_true_sm, N, stick_thresh)
                    gt_button = discretize_button(Y_true_sm, button_thresh, stick_offset=N)
                    gt_combined = combine(gt_stick, gt_button)

                    pred_stick = discretize_stick(Y_pred_sm, N, stick_thresh)
                    pred_button = discretize_button(Y_pred_sm, button_thresh, stick_offset=N)
                    pred_combined = combine(pred_stick, pred_button)

                    for scoring in ["stick_only", "stick_button"]:
                        if scoring == "stick_only":
                            gt_sym = gt_stick
                            pred_sym = pred_stick
                            N_eff = N
                        else:
                            gt_sym = gt_combined
                            pred_sym = pred_combined
                            N_eff = N + 6

                        for silence_mode in ["neutral", "correct"]:
                            result = proxy_bitrate(gt_sym, pred_sym, N_eff,
                                                   bin_seconds, silence_mode)

                            n_emitted = result["S_c"] + result["S_i"]
                            t_per_emission = BASE_RT_SEC + K * bin_seconds
                            t_per_noop = BASE_RT_SEC
                            t_adj = (n_emitted * t_per_emission
                                     + result["n_noop"] * t_per_noop)
                            log_term = np.log2(max(N_eff - 1, 1))
                            B_lat = (float(log_term * max(result["S_c"] - result["S_i"], 0)
                                           / t_adj) if t_adj > 0 else 0.0)

                            row = {
                                "mode": mode,
                                "scoring": scoring,
                                "silence_mode": silence_mode,
                                "K": K,
                                "stick_thresh": stick_thresh,
                                "button_thresh": button_thresh,
                                "N": N,
                                "N_eff": N_eff,
                                "S_c": result["S_c"],
                                "S_i": result["S_i"],
                                "n_emitted": n_emitted,
                                "n_noop": result["n_noop"],
                                "n_total": result["n_total"],
                                "t_sec": result["t_sec"],
                                "B": result["B"],
                                "B_latency_adjusted": B_lat,
                                "t_adjusted_sec": t_adj,
                                "accuracy_emitted": result["accuracy_emitted"],
                                "effective_latency_ms": K * BIN_MS,
                            }
                            rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "sweep_postproc_v3.csv")
    fieldnames = [
        "mode", "scoring", "silence_mode", "K", "stick_thresh",
        "button_thresh", "N", "N_eff", "S_c", "S_i", "n_emitted",
        "n_noop", "n_total", "t_sec", "B", "B_latency_adjusted",
        "t_adjusted_sec", "accuracy_emitted", "effective_latency_ms",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # ── Reporting ─────────────────────────────────────────────────────
    COL_HDR = (f"  {'K':>3s} {'lat':>5s} {'st_th':>5s} {'bt_th':>5s} {'N':>3s} "
               f"{'S_c':>6s} {'S_i':>6s} {'emit':>6s} {'noop':>6s} "
               f"{'B':>8s} {'B_lat':>8s} {'acc':>6s}")

    def fmt(r):
        return (f"  {r['K']:>3d} {r['effective_latency_ms']:>4d}m "
                f"{r['stick_thresh']:>5.2f} {r['button_thresh']:>5.2f} "
                f"{r['N']:>3d} {r['S_c']:>6d} {r['S_i']:>6d} "
                f"{r['n_emitted']:>6d} {r['n_noop']:>6d} "
                f"{r['B']:>8.4f} {r['B_latency_adjusted']:>8.4f} "
                f"{r['accuracy_emitted']:>6.3f}")

    BOUNDED_K = {10, 15, 20}

    print(f"\n{'#'*78}")
    print(f"# silence_mode = neutral")
    print(f"{'#'*78}")

    for scoring in ["stick_only", "stick_button"]:
        neutral = [r for r in rows
                   if r["scoring"] == scoring and r["silence_mode"] == "neutral"]

        # (a) Top-5 by B, K bounded to {10,15,20}
        bounded = [r for r in neutral if r["K"] in BOUNDED_K]
        bounded.sort(key=lambda r: r["B"], reverse=True)
        print(f"\n{'='*78}")
        print(f"Top-5 by B (K≤20): scoring={scoring}")
        print(f"{'='*78}")
        print(COL_HDR)
        for r in bounded[:5]:
            print(fmt(r))

        # (b) Top-5 by B_latency_adjusted, all K
        neutral.sort(key=lambda r: r["B_latency_adjusted"], reverse=True)
        print(f"\n{'='*78}")
        print(f"Top-5 by B_latency_adjusted (all K): scoring={scoring}")
        print(f"{'='*78}")
        print(COL_HDR)
        for r in neutral[:5]:
            print(fmt(r))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["easy", "hard"], required=True)
    args = parser.parse_args()
    main(args)

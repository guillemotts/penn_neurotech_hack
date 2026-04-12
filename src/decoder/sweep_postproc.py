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

K_VALUES = [1, 3, 5, 7, 10]
STICK_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BUTTON_THRESHOLDS = [0.0, 0.20, 0.35, 0.50, 0.65]
N_VALUES = [4, 8]


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

    rows = []

    for K in K_VALUES:
        Y_pred_sm = causal_moving_average(Y_pred, K)
        Y_true_sm = causal_moving_average(Y_true, K)

        for N in N_VALUES:
            for stick_thresh in STICK_THRESHOLDS:
                for button_thresh in BUTTON_THRESHOLDS:
                    # ── GT symbols ────────────────────────────
                    gt_stick = discretize_stick(Y_true_sm, N, stick_thresh)
                    gt_button = discretize_button(Y_true_sm, button_thresh, stick_offset=N)
                    gt_combined = combine(gt_stick, gt_button)

                    # ── Pred symbols ──────────────────────────
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
                                "n_noop": result["n_noop"],
                                "n_total": result["n_total"],
                                "t_sec": result["t_sec"],
                                "B": result["B"],
                                "accuracy_emitted": result["accuracy_emitted"],
                            }
                            rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "sweep_postproc.csv")
    fieldnames = [
        "mode", "scoring", "silence_mode", "K", "stick_thresh",
        "button_thresh", "N", "N_eff", "S_c", "S_i", "n_noop",
        "n_total", "t_sec", "B", "accuracy_emitted",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # ── Top-5 per (scoring, silence_mode) ─────────────────────────────
    for scoring in ["stick_only", "stick_button"]:
        for silence_mode in ["neutral", "correct"]:
            subset = [r for r in rows
                      if r["scoring"] == scoring and r["silence_mode"] == silence_mode]
            subset.sort(key=lambda r: r["B"], reverse=True)
            print(f"\n{'='*70}")
            print(f"Top-5: scoring={scoring}, silence_mode={silence_mode}")
            print(f"{'='*70}")
            print(f"  {'K':>3s}  {'st_th':>5s}  {'bt_th':>5s}  {'N':>3s}  "
                  f"{'B':>8s}  {'acc':>6s}  {'S_c':>6s}  {'S_i':>6s}  {'noop':>6s}")
            for r in subset[:5]:
                print(f"  {r['K']:>3d}  {r['stick_thresh']:>5.2f}  "
                      f"{r['button_thresh']:>5.2f}  {r['N']:>3d}  "
                      f"{r['B']:>8.4f}  {r['accuracy_emitted']:>6.3f}  "
                      f"{r['S_c']:>6d}  {r['S_i']:>6d}  {r['n_noop']:>6d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["easy", "hard"], required=True)
    args = parser.parse_args()
    main(args)

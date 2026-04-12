"""
check_alignment.py -- Verify window alignment to action onset, not cue time.

Loads seed 7, inspects controller channels around cue times to find
when the subject actually starts moving.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "src")

from load_h5_session import align_and_build_session

# ── 1. Load seed 7 ───────────────────────────────────────────────────
data_dir = Path(__file__).resolve().parent.parent / "data" / "newdata_apr12"
h5 = data_dir / "smoke_Ani_seed7.h5"
cj = data_dir / "cues_Ani_seed7.json"

print("Loading seed 7 ...")
sess = align_and_build_session(h5, cj, n_channels=76)
fs = sess.fs_hz
T0 = int(sess.frame_timestamps_ns[0])
n_trials = len(sess.cue_times_ns)
print("  neural shape: {}  fs: {}  n_trials: {}".format(sess.neural.shape, fs, n_trials))

# Cue sample indices
cue_samples = []
for t_ns in sess.cue_times_ns:
    rel_s = (int(t_ns) - T0) / 1e9
    cue_samples.append(int(round(rel_s * fs)))
cue_samples = np.array(cue_samples, dtype=np.int64)

T_total = sess.neural.shape[1]
n_check = min(20, n_trials)

# ── 2 & 3. Per-trial action onset detection ─────────────────────────
pre_samples = int(0.5 * fs)    # 0.5 s before cue
post_samples = int(1.5 * fs)   # 1.5 s after cue
baseline_end = int(0.1 * fs)   # baseline ends 0.1 s before cue

onset_ms_list = []

print("\nPer-trial action onset (first 20 trials):")
print("{:>5s} {:>8s} {:>10s}  {}".format("trial", "symbol", "onset_ms", "note"))

for i in range(n_check):
    t_cue = cue_samples[i]
    win_start = t_cue - pre_samples
    win_end = t_cue + post_samples

    if win_start < 0 or win_end > T_total:
        print("{:5d} {:>8s} {:>10s}  skipped (out of bounds)".format(
            i, sess.cue_symbols[i], "---"))
        continue

    # Controller channels 64..75 over [t_cue - 0.5s : t_cue + 1.5s]
    ctrl = sess.neural[64:76, win_start:win_end].astype(np.float64)
    # L2 norm per sample across 12 controller channels
    ctrl_norm = np.linalg.norm(ctrl, axis=0)  # (win_len,)

    # Baseline: [t_cue - 0.5s : t_cue - 0.1s] = samples [0 : pre - baseline_end]
    baseline_len = pre_samples - baseline_end
    baseline = ctrl_norm[:baseline_len]
    bl_mean = np.mean(baseline)
    bl_std = np.std(baseline)
    threshold = bl_mean + 5.0 * bl_std

    # Find first sample in full window exceeding threshold
    above = np.where(ctrl_norm > threshold)[0]
    if len(above) == 0:
        onset_ms = float("nan")
        note = "no threshold crossing"
    else:
        # onset relative to cue = (sample_in_window - pre_samples) / fs * 1000
        onset_sample = above[0]
        onset_ms = (onset_sample - pre_samples) / fs * 1000.0
        note = ""

    onset_ms_list.append(onset_ms)
    print("{:5d} {:>8s} {:10.1f}  {}".format(
        i, sess.cue_symbols[i], onset_ms, note))

# ── 4. Summary statistics ────────────────────────────────────────────
valid = np.array([v for v in onset_ms_list if not np.isnan(v)])
print("\n--- Summary ({} valid of {} checked) ---".format(len(valid), n_check))
if len(valid) > 0:
    print("  mean:   {:.1f} ms".format(np.mean(valid)))
    print("  median: {:.1f} ms".format(np.median(valid)))
    print("  std:    {:.1f} ms".format(np.std(valid)))

    # Histogram buckets
    buckets = [
        ("<0",      np.sum(valid < 0)),
        ("0-100",   np.sum((valid >= 0) & (valid < 100))),
        ("100-200", np.sum((valid >= 100) & (valid < 200))),
        ("200-400", np.sum((valid >= 200) & (valid < 400))),
        ("400-800", np.sum((valid >= 400) & (valid < 800))),
        (">800",    np.sum(valid >= 800)),
    ]
    print("\n  Histogram (ms):")
    for label, count in buckets:
        bar = "#" * int(count)
        print("    {:>8s}: {:3d}  {}".format(label, count, bar))

# ── 5. Neural variance in 200ms windows for 3 example trials ────────
print("\n--- Neural channel 0..3 variance in 200ms windows (3 example trials) ---")
win_ms = 200
win_samp = int(win_ms / 1000.0 * fs)  # 6400 samples

example_trials = [0, 1, 2]
for i in example_trials:
    if i >= n_trials:
        break
    t_cue = cue_samples[i]
    print("\nTrial {} (symbol: {}):".format(i, sess.cue_symbols[i]))
    windows = [
        ("cue-200:cue",     t_cue - win_samp, t_cue),
        ("cue:cue+200",     t_cue,            t_cue + win_samp),
        ("cue+200:cue+400", t_cue + win_samp, t_cue + 2 * win_samp),
        ("cue+400:cue+600", t_cue + 2 * win_samp, t_cue + 3 * win_samp),
    ]
    header = "  {:>20s}".format("window")
    for ch in range(4):
        header += "    ch{:d}".format(ch)
    print(header)
    for label, s, e in windows:
        if s < 0 or e > T_total:
            print("  {:>20s}  (out of bounds)".format(label))
            continue
        row = "  {:>20s}".format(label)
        for ch in range(4):
            v = np.var(sess.neural[ch, s:e].astype(np.float64))
            row += "  {:10.2f}".format(v)
        print(row)

print("\nDone.")

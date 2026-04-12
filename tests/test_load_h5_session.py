import json
import numpy as np
import h5py
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from load_h5_session import (
    load_h5, load_cues, detect_lb_edge, align_and_build_session, extract_windows, SYMBOL_STRS
)

def _make_synthetic_h5(path, n_channels=76, fs_hz=32000, duration_s=10, samples_per_frame=320,
                      lb_channel=68, lb_start_s=2.0, lb_hold_s=2.0):
    n_samples = int(duration_s * fs_hz)
    n_frames = n_samples // samples_per_frame
    data = np.random.RandomState(0).randn(n_frames, n_channels, samples_per_frame).astype(np.float32) * 10
    # Inject LB edge
    lb_start_sample = int(lb_start_s * fs_hz)
    lb_end_sample = int((lb_start_s + lb_hold_s) * fs_hz)
    for s in range(lb_start_sample, lb_end_sample):
        frame_idx = s // samples_per_frame
        sample_in_frame = s % samples_per_frame
        if frame_idx < n_frames:
            data[frame_idx, lb_channel, sample_in_frame] = 30000.0
    # Frame timestamps: 1000 ns per sample at 32 kHz ≈ 31250, but use round numbers
    ns_per_sample = int(1e9 / fs_hz)
    ns_per_frame = ns_per_sample * samples_per_frame
    t0 = 1_000_000_000_000
    ts = np.array([t0 + i * ns_per_frame for i in range(n_frames)], dtype=np.int64)
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=data)
        f.create_dataset("frame_timestamps_ns", data=ts)
        f.attrs["sample_rate_hz"] = float(fs_hz)
    return t0, ns_per_sample, lb_start_sample, lb_end_sample

def _make_synthetic_cues(path, sync_prompt_mono_ns, sync_release_mono_ns, trial_ts_mono_ns):
    log = {
        "schema_version": 1,
        "symbols": SYMBOL_STRS,
        "n_symbols": 22,
        "seed": 0,
        "n_trials": len(trial_ts_mono_ns),
        "isi_s": 0.8,
        "session_start_t_mono_ns": sync_prompt_mono_ns - 3_000_000_000,
        "sync_gesture": {
            "instruction": "hold LB",
            "prompt_t_mono_ns": sync_prompt_mono_ns,
            "release_t_mono_ns": sync_release_mono_ns,
        },
        "trials": [
            {"idx": i, "symbol_idx": i % 22, "symbol": SYMBOL_STRS[i % 22], "t_mono_ns": int(t)}
            for i, t in enumerate(trial_ts_mono_ns)
        ],
    }
    with open(path, "w") as f:
        json.dump(log, f)

def test_load_h5_shape(tmp_path):
    h5p = tmp_path / "s.h5"
    _make_synthetic_h5(h5p)
    neural, ts, spf, fs = load_h5(h5p)
    assert neural.shape[0] == 76
    assert neural.ndim == 2
    assert spf == 320
    assert fs == 32000.0

def test_detect_lb_edge(tmp_path):
    h5p = tmp_path / "s.h5"
    t0, nsps, lb_start, _ = _make_synthetic_h5(h5p)
    neural, ts, spf, fs = load_h5(h5p)
    _, hw_ns, ch = detect_lb_edge(neural, fs, ts, spf, min_hold_s=1.5)
    assert ch == 4  # lb_channel 68 - raw_start 64
    expected_hw_ns = t0 + lb_start * nsps
    assert abs(hw_ns - expected_hw_ns) < nsps * 10

def test_align_and_extract(tmp_path):
    h5p = tmp_path / "s.h5"
    cp = tmp_path / "c.json"
    t0, nsps, lb_start, _ = _make_synthetic_h5(h5p, duration_s=10)
    sync_prompt_mono = 500_000_000_000
    sync_release_mono = sync_prompt_mono + 2_000_000_000
    # 5 trials starting 4s after sync prompt (mono), within h5 range
    trial_mono = [sync_prompt_mono + int((4 + 0.8 * i) * 1e9) for i in range(5)]
    _make_synthetic_cues(cp, sync_prompt_mono, sync_release_mono, trial_mono)
    sess = align_and_build_session(h5p, cp, n_channels=76)
    assert sess.neural.shape[0] == 76
    assert len(sess.cue_times_ns) == 5
    X, y, kept = extract_windows(sess, w_post_ms=200.0, onset_offset_ms=100.0)
    assert X.shape[0] >= 3  # some trials should land inside h5
    assert X.shape[1] == 76
    assert X.dtype == np.float32

def test_detect_lb_edge_no_hold_raises(tmp_path):
    h5p = tmp_path / "s.h5"
    _make_synthetic_h5(h5p, lb_hold_s=0.5)
    neural, ts, spf, fs = load_h5(h5p)
    with pytest.raises(ValueError):
        detect_lb_edge(neural, fs, ts, spf, min_hold_s=1.5)

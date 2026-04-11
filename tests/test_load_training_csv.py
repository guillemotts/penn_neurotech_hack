"""Tests for scripts/load_training_csv.py."""
import numpy as np
import pandas as pd
import pytest

from scripts.load_training_csv import (
    SYMBOLS,
    N_SYMBOLS,
    Session,
    symbol_to_index,
    load_session,
    recover_channel_map,
    extract_trial_windows,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_neural_csv(path, T, rng=None, time_s=None):
    """Write a neural CSV with T rows, ch_0..ch_75, time_s from 0 with step 0.001."""
    if rng is None:
        rng = np.random.default_rng(42)
    if time_s is None:
        time_s = np.arange(T) * 0.001
    ts_ns = (time_s * 1e9).astype(np.int64)
    channels = rng.standard_normal((T, 76)).astype(np.float32)
    cols = {"time_s": time_s, "timestamp_ns": ts_ns}
    for i in range(76):
        cols[f"ch_{i}"] = channels[:, i]
    df = pd.DataFrame(cols)
    df.to_csv(path, index=False)
    return time_s, channels


def _write_cue_csv(path, t_monos, cues):
    """Write a cue CSV with trial, t_mono, cue columns."""
    df = pd.DataFrame({
        "trial": list(range(len(t_monos))),
        "t_mono": t_monos,
        "cue": cues,
    })
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_symbol_to_index_roundtrip():
    """Parse all 22 symbols, indices 0..21."""
    for i, (typ, val) in enumerate(SYMBOLS):
        s = f"{typ}_{val}"
        assert symbol_to_index(s) == i
    # Check we covered 0..21
    assert len(SYMBOLS) == N_SYMBOLS == 22


def test_symbol_to_index_unknown_raises():
    """'foo_bar' -> ValueError."""
    with pytest.raises(ValueError, match="Unknown symbol"):
        symbol_to_index("foo_bar")


def test_load_session_shapes(tmp_path):
    """
    Write 1000-row neural CSV, 5-row cue CSV.
    Assert shapes and fs ~ 1000.
    """
    neural_path = tmp_path / "neural.csv"
    cue_path = tmp_path / "cues.csv"

    _write_neural_csv(neural_path, 1000)
    t_monos = [0.1, 0.3, 0.5, 0.7, 0.9]
    cues = ["ljoy_0", "button_UP", "ljoy_0", "button_UP", "ljoy_0"]
    _write_cue_csv(cue_path, t_monos, cues)

    session = load_session(neural_path, cue_path)
    assert session.neural.shape == (64, 1000)
    assert session.raw.shape == (12, 1000)
    assert len(session.cue_times) == 5
    # fs = len(time_s) / span = 1000 / 0.999 ≈ 1001
    assert abs(session.fs_hz - 1000.0) < 2.0


def test_load_session_misaligned_raises(tmp_path):
    """After rebase, cue span still exceeds neural span + 1.0 -> ValueError."""
    neural_path = tmp_path / "neural.csv"
    cue_path = tmp_path / "cues.csv"

    _write_neural_csv(neural_path, 1000)  # time_s 0..0.999, span ~1s
    # Cues span 5s: after rebase cue_times.max()=4.9 >> 0.999+1.0
    _write_cue_csv(cue_path, [0.1, 5.0], ["ljoy_0", "button_UP"])

    with pytest.raises(ValueError, match="misaligned"):
        load_session(neural_path, cue_path)


def test_load_session_rebases_time(tmp_path):
    """Neural starting at 5.0 and cues at ~1005 are rebased to start near 0."""
    neural_path = tmp_path / "neural.csv"
    cue_path = tmp_path / "cues.csv"

    time_s = np.linspace(5.0, 6.0, 1000)
    _write_neural_csv(neural_path, 1000, time_s=time_s)
    _write_cue_csv(cue_path, [1005.1, 1005.5, 1005.8], ["ljoy_0", "button_UP", "ljoy_0"])

    session = load_session(neural_path, cue_path)
    # cue_times rebased: first cue should be ~0
    assert abs(session.cue_times[0]) < 1e-9
    # All cue times within neural time_s range
    assert session.cue_times.max() <= session.time_s.max()
    # neural time_s rebased to start at 0
    assert abs(session.time_s[0]) < 1e-9
    assert abs(session.time_s.max() - 1.0) < 0.01


def test_recover_channel_map_synthetic():
    """
    Build session where ch_64 deflects +100 only during 'ljoy_0' cues,
    ch_65 deflects only during 'button_UP'. Assert mapping finds them.
    """
    T = 2000
    fs = 1000.0
    time_s = np.arange(T) / fs
    neural = np.zeros((64, T), dtype=np.float32)
    raw = np.zeros((12, T), dtype=np.float32)

    cue_times = np.array([0.2, 0.6, 1.0, 1.4])
    cue_symbols = ["ljoy_0", "button_UP", "ljoy_0", "button_UP"]
    cue_indices = np.array([symbol_to_index(c) for c in cue_symbols], dtype=np.int64)

    # Inject deflections: raw channel 0 (=ch_64) during ljoy_0, channel 1 (=ch_65) during button_UP
    for i, sym in enumerate(cue_symbols):
        t_start = cue_times[i]
        t_end = t_start + 0.3
        mask = (time_s >= t_start) & (time_s < t_end)
        if sym == "ljoy_0":
            raw[0, mask] = 100.0
        elif sym == "button_UP":
            raw[1, mask] = 100.0

    session = Session(
        time_s=time_s,
        neural=neural,
        raw=raw,
        fs_hz=fs,
        cue_times=cue_times,
        cue_symbols=cue_symbols,
        cue_indices=cue_indices,
    )

    mapping = recover_channel_map(session)
    assert mapping["ljoy_0"]["channel"] == 0
    assert mapping["button_UP"]["channel"] == 1


def test_extract_trial_windows_shape():
    """
    Session fs=1000, 5 cues at t=[0.2, 0.4, 0.6, 0.8, 1.5].
    window 0-150ms, offset 50ms.
    Last cue exceeds session -> kept has 4 trials. X shape (4, 64, 150).
    """
    T = 2000  # 2 seconds at 1000 Hz
    fs = 1000.0
    time_s = np.arange(T) / fs
    neural = np.random.default_rng(7).standard_normal((64, T)).astype(np.float32)
    raw = np.zeros((12, T), dtype=np.float32)

    cue_times = np.array([0.2, 0.4, 0.6, 0.8, 1.5])
    cue_symbols = ["ljoy_0"] * 5
    cue_indices = np.zeros(5, dtype=np.int64)

    session = Session(
        time_s=time_s,
        neural=neural,
        raw=raw,
        fs_hz=fs,
        cue_times=cue_times,
        cue_symbols=cue_symbols,
        cue_indices=cue_indices,
    )

    X, y, kept = extract_trial_windows(session, w0_ms=0.0, w1_ms=150.0, onset_offset_ms=50.0)
    # Last cue at 1.5: window = [1.55, 1.70] but session ends at 1.999 -> fits
    # Actually 1.5 + 0.05 + 0.15 = 1.70, which is <= 1.999. All 5 should fit.
    # Let's recalculate: time_s[-1] = 1.999.
    # cue 4 at 1.5: win_end = 1.5 + 0.05 + 0.15 = 1.70 <= 1.999. Still fits.
    # We need a cue that truly exceeds. Let's adjust:
    # With T=2000, time_s[-1] = 1.999. cue at 1.5 + 0.05 = 1.55, end 1.70 -> fits.
    # Need cue at ~1.9: 1.9+0.05+0.15=2.10 > 1.999 -> skip.
    # Re-do with cue at 1.9 instead of 1.5.
    cue_times2 = np.array([0.2, 0.4, 0.6, 0.8, 1.9])
    session2 = Session(
        time_s=time_s,
        neural=neural,
        raw=raw,
        fs_hz=fs,
        cue_times=cue_times2,
        cue_symbols=cue_symbols,
        cue_indices=cue_indices,
    )
    X, y, kept = extract_trial_windows(session2, w0_ms=0.0, w1_ms=150.0, onset_offset_ms=50.0)
    assert X.shape == (4, 64, 150)
    assert y.shape == (4,)
    assert len(kept) == 4
    assert 4 not in kept  # trial index 4 was skipped


def test_main_smoke(tmp_path):
    """Write both CSVs, call main, assert return 0, output .npz has expected keys."""
    neural_path = tmp_path / "neural.csv"
    cue_path = tmp_path / "cues.csv"
    out_path = tmp_path / "out.npz"

    _write_neural_csv(neural_path, 1000)
    _write_cue_csv(cue_path, [0.1, 0.3, 0.5], ["ljoy_0", "button_UP", "rjoy_2"])

    ret = main([
        "--neural", str(neural_path),
        "--cues", str(cue_path),
        "--out", str(out_path),
    ])
    assert ret == 0
    assert out_path.exists()
    data = np.load(out_path)
    assert "X" in data
    assert "y" in data
    assert "kept_idx" in data
    assert data["X"].ndim == 3
    assert data["y"].ndim == 1

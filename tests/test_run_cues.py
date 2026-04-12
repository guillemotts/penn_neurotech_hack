import json
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_cues import (
    SYMBOL_STRS, N_SYMBOLS, ALPHABETS, generate_cue_sequence, build_log, write_log, main
)

def test_symbol_strs_exact_match():
    expected = (
        [f"ljoy_{k}" for k in range(8)]
        + [f"rjoy_{k}" for k in range(8)]
        + ["button_UP", "button_DOWN", "button_LEFT", "button_RIGHT", "button_LB", "button_RB"]
    )
    assert SYMBOL_STRS == expected
    assert N_SYMBOLS == 22

def test_generate_cue_sequence_uniform_chi_square():
    cues = generate_cue_sequence(22000, seed=0)
    assert cues.shape == (22000,)
    assert cues.min() >= 0 and cues.max() < 22
    counts = np.bincount(cues, minlength=22)
    expected = 22000 / 22
    chi2 = ((counts - expected) ** 2 / expected).sum()
    assert chi2 < 50.0  # df=21, critical ~32.7 at p=0.05; 50 is loose

def test_generate_cue_sequence_deterministic():
    a = generate_cue_sequence(500, seed=42)
    b = generate_cue_sequence(500, seed=42)
    assert np.array_equal(a, b)

def test_generate_cue_sequence_seed_differs():
    a = generate_cue_sequence(500, seed=1)
    b = generate_cue_sequence(500, seed=2)
    assert not np.array_equal(a, b)

def test_build_log_schema():
    trials = [{"idx": i, "symbol_idx": i % 22, "t_mono_ns": 1000 + i} for i in range(5)]
    log = build_log(
        session_start_ns=100,
        sync_prompt_ns=200,
        sync_release_ns=2200,
        trial_records=trials,
        seed=7,
        n_trials=5,
        isi_s=0.8,
    )
    assert log["schema_version"] == 1
    assert log["symbols"] == SYMBOL_STRS
    assert log["n_symbols"] == 22
    assert log["seed"] == 7
    assert log["n_trials"] == 5
    assert log["isi_s"] == 0.8
    assert log["session_start_t_mono_ns"] == 100
    assert log["sync_gesture"]["prompt_t_mono_ns"] == 200
    assert log["sync_gesture"]["release_t_mono_ns"] == 2200
    assert len(log["trials"]) == 5
    for i, tr in enumerate(log["trials"]):
        assert tr["idx"] == i
        assert tr["symbol"] == SYMBOL_STRS[i % 22]
        assert isinstance(tr["t_mono_ns"], int)

def test_build_log_integer_timestamps():
    trials = [{"idx": 0, "symbol_idx": 3, "t_mono_ns": 1234567890123456789}]
    log = build_log(1, 2, 3, trials, 0, 1, 0.8)
    assert log["trials"][0]["t_mono_ns"] == 1234567890123456789
    assert isinstance(log["trials"][0]["t_mono_ns"], int)

def test_write_log_roundtrip(tmp_path):
    trials = [{"idx": 0, "symbol_idx": 0, "t_mono_ns": 42}]
    log = build_log(0, 0, 0, trials, 0, 1, 0.8)
    out = tmp_path / "sub" / "cues.json"
    write_log(out, log)
    loaded = json.loads(out.read_text())
    assert loaded == log

def test_main_dry_run_writes_valid_log(tmp_path):
    out = tmp_path / "cues.json"
    rc = main(["--n-trials", "50", "--seed", "13", "--isi", "0.8", "--out", str(out), "--dry-run"])
    assert rc == 0
    log = json.loads(out.read_text())
    assert log["n_trials"] == 50
    assert len(log["trials"]) == 50
    assert log["seed"] == 13
    assert log["isi_s"] == 0.8
    assert all(t["symbol_idx"] in range(22) for t in log["trials"])

def test_main_rejects_bad_args(tmp_path):
    out = tmp_path / "cues.json"
    with pytest.raises(SystemExit):
        main(["--n-trials", "0", "--seed", "1", "--out", str(out), "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--n-trials", "10", "--seed", "1", "--isi", "0", "--out", str(out), "--dry-run"])

def test_cue_color_left():
    from run_cues import cue_color, COLOR_LEFT
    assert cue_color("ljoy", 0) == COLOR_LEFT
    assert cue_color("ljoy", 7) == COLOR_LEFT
    assert cue_color("button", "LEFT") == COLOR_LEFT
    assert cue_color("button", "LB") == COLOR_LEFT

def test_cue_color_right():
    from run_cues import cue_color, COLOR_RIGHT
    assert cue_color("rjoy", 0) == COLOR_RIGHT
    assert cue_color("rjoy", 7) == COLOR_RIGHT
    assert cue_color("button", "RIGHT") == COLOR_RIGHT
    assert cue_color("button", "RB") == COLOR_RIGHT

def test_cue_color_vertical():
    from run_cues import cue_color, COLOR_VERT
    assert cue_color("button", "UP") == COLOR_VERT
    assert cue_color("button", "DOWN") == COLOR_VERT

def test_cue_color_unknown_raises():
    from run_cues import cue_color
    import pytest
    with pytest.raises(ValueError):
        cue_color("trigger", "LT")

def test_alphabet_n10_coverage():
    n10 = ALPHABETS["n10"]
    cues = generate_cue_sequence(2000, seed=0, alphabet=n10)
    assert cues.shape == (2000,)
    unique = set(cues.tolist())
    assert unique.issubset(set(n10)), f"unexpected symbols: {unique - set(n10)}"
    assert set(n10).issubset(unique), f"missing symbols: {set(n10) - unique}"

def test_alphabet_full_default():
    cues = generate_cue_sequence(22000, seed=0)
    assert cues.min() >= 0 and cues.max() < 22

def test_build_log_includes_alphabet():
    trials = [{"idx": 0, "symbol_idx": 2, "t_mono_ns": 42}]
    log = build_log(0, 0, 0, trials, 0, 1, 0.8,
                    alphabet_name="n10", alphabet_indices=ALPHABETS["n10"])
    assert log["alphabet"] == "n10"
    assert log["alphabet_symbols"] == [SYMBOL_STRS[i] for i in ALPHABETS["n10"]]

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("run_calibration", _SCRIPTS / "run_calibration.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
main = _mod.main


def _make_trials(tmp_path):
    """Synthetic trials: raw (400,8,40), stick_xy on annulus r in [0.6,1.0]."""
    rng = np.random.default_rng(0)
    M = 400
    angles = rng.uniform(-np.pi, np.pi, M)
    r = rng.uniform(0.6, 1.0, M)
    stick_xy = np.column_stack([r * np.cos(angles), r * np.sin(angles)]).astype(
        np.float32
    )
    raw = rng.standard_normal((M, 8, 40)).astype(np.float32)
    trials_path = tmp_path / "trials.npz"
    np.savez(str(trials_path), raw=raw, stick_xy=stick_xy)
    return trials_path


def _run(tmp_path):
    trials_path = _make_trials(tmp_path)
    out_path = tmp_path / "calibration.yaml"
    rc = main(
        [
            "--trials", str(trials_path),
            "--out", str(out_path),
            "--player", "test_player",
            "--arch", "linear",
            "--n-grid", "4", "6", "8", "10",
        ]
    )
    return rc, out_path


def test_calibration_smoke(tmp_path, capsys):
    rc, out_path = _run(tmp_path)
    assert rc == 0
    with open(out_path) as f:
        cfg = yaml.safe_load(f)
    for key in ("player", "n", "theta", "alpha", "tau", "feature_config", "sweep_result"):
        assert key in cfg, f"missing key {key}"
    assert cfg["n"] in [4, 6, 8, 10]
    assert cfg["theta"] >= 0


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        main(
            [
                "--trials", str(tmp_path / "nonexistent.npz"),
                "--out", str(tmp_path / "out.yaml"),
                "--player", "test",
            ]
        )


def test_yaml_roundtrip(tmp_path):
    rc, out_path = _run(tmp_path)
    assert rc == 0
    with open(out_path) as f:
        loaded = yaml.safe_load(f)
    dumped = yaml.safe_dump(loaded, sort_keys=False)
    reloaded = yaml.safe_load(dumped)
    assert loaded == reloaded

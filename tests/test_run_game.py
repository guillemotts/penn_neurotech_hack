import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location("run_game", _SCRIPTS / "run_game.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
main = _mod.main


def _save_confusion(tmp_path, key, array):
    path = tmp_path / "conf.npz"
    np.savez(str(path), **{key: array})
    return str(path)


def test_perfect_confusion_smoke(tmp_path, capsys):
    C = np.eye(8, dtype=float)
    path = _save_confusion(tmp_path, "C", C)
    rc = main([
        "--confusion", path,
        "--time-source", "iterator",
        "--session-seconds", "2.0",
        "--mean-interval", "0.05",
        "--quiet",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "FINAL" in out
    m = re.search(r"B=([\d.]+)", out)
    assert m is not None
    assert float(m.group(1)) > 0


def test_missing_C_key(tmp_path, capsys):
    path = _save_confusion(tmp_path, "X", np.eye(4))
    rc = main(["--confusion", path, "--quiet"])
    assert rc == 2


def test_nonsquare_C(tmp_path, capsys):
    path = _save_confusion(tmp_path, "C", np.ones((3, 4)))
    rc = main(["--confusion", path, "--quiet"])
    assert rc == 2


def test_rejection_rate_propagates(tmp_path, capsys):
    C = np.eye(8, dtype=float)
    path = _save_confusion(tmp_path, "C", C)
    rc = main([
        "--confusion", path,
        "--time-source", "iterator",
        "--session-seconds", "2.0",
        "--mean-interval", "0.05",
        "--rejection-rate", "1.0",
        "--quiet",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    m_rej = re.search(r"rej=(\d+)", out)
    assert m_rej is not None
    assert int(m_rej.group(1)) > 0
    m_b = re.search(r"B=([\d.]+)", out)
    assert m_b is not None
    assert float(m_b.group(1)) == pytest.approx(0.0)

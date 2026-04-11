import numpy as np
import pytest
from numpy import log2

from penn_nh.eval.bitrate import (
    _mi_plugin,
    mutual_information_pt,
    per_symbol_breakeven,
    shenoy_b,
    shenoy_b_from_confusion,
)


def test_shenoy_b_hand_computed():
    assert shenoy_b(s_c=30, s_i=5, n=8, t=60.0) == pytest.approx(log2(7) * 25 / 60)


def test_shenoy_b_floor_at_zero():
    assert shenoy_b(s_c=5, s_i=10, n=8) == 0.0


def test_shenoy_b_validation():
    with pytest.raises(ValueError):
        shenoy_b(s_c=10, s_i=5, n=2)
    with pytest.raises(ValueError):
        shenoy_b(s_c=10, s_i=5, n=8, t=0.0)
    with pytest.raises(ValueError):
        shenoy_b(s_c=-1, s_i=5, n=8)
    with pytest.raises(ValueError):
        shenoy_b(s_c=10, s_i=-1, n=8)


def test_confusion_matches_scalar():
    rng = np.random.default_rng(42)
    C = rng.integers(0, 20, size=(8, 8))
    s_c = int(np.trace(C))
    s_i = int(C.sum()) - s_c
    assert shenoy_b_from_confusion(C) == pytest.approx(shenoy_b(s_c, s_i, 8))


def test_confusion_rejects_nonsquare():
    with pytest.raises(ValueError):
        shenoy_b_from_confusion(np.zeros((4, 5), dtype=int))


def test_mi_pt_perfect_decoder():
    C = 100 * np.eye(8, dtype=int)
    mi = mutual_information_pt(C)
    assert 0 < mi <= 3.0
    assert mi == pytest.approx(3.0 - 7 / (1600 * np.log(2)), abs=1e-9)


def test_mi_pt_uniform_noise():
    C = np.full((4, 4), 25, dtype=int)
    assert mutual_information_pt(C) == 0.0


def test_mi_pt_n2_closed_form():
    C = np.array([[40, 10], [10, 40]], dtype=int)
    expected_plugin = 0.8 * log2(1.6) + 0.2 * log2(0.4)
    assert _mi_plugin(C) == pytest.approx(expected_plugin, abs=1e-12)


def test_breakeven_signs():
    C = np.array(
        [
            [10, 0, 0],
            [5, 0, 5],
            [3, 3, 4],
        ],
        dtype=int,
    )
    b = per_symbol_breakeven(C, t_per_trial=1.0)
    assert b[0] > 0
    assert b[1] < 0


def test_breakeven_empty_row():
    C = np.array(
        [
            [10, 0, 0],
            [0, 0, 0],
            [0, 0, 10],
        ],
        dtype=int,
    )
    b = per_symbol_breakeven(C, t_per_trial=1.0)
    assert b[1] == 0.0
    assert not np.isnan(b[1])

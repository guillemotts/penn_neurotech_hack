import numpy as np
import pytest

from penn_nh.decoder.stage3_reject import decide


def test_clear_margin_accepts():
    assert decide(np.array([0.9, 0.1, 0.05]), theta=0.3) == 0


def test_small_margin_rejects():
    assert decide(np.array([0.5, 0.45, 0.1]), theta=0.3) is None


def test_tie_rejects_even_zero_theta():
    assert decide(np.array([0.5, 0.5, 0.0]), theta=0.0) is None


def test_two_class():
    assert decide(np.array([0.9, 0.1]), theta=0.5) == 0


def test_length_one_raises():
    with pytest.raises(ValueError):
        decide(np.array([0.9]), theta=0.3)


def test_negative_theta_raises():
    with pytest.raises(ValueError):
        decide(np.array([0.9, 0.1]), theta=-0.1)


def test_2d_input_raises():
    with pytest.raises(ValueError):
        decide(np.array([[0.9, 0.1]]), theta=0.3)

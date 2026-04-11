import numpy as np
import pytest
from numpy import log2

from penn_nh.eval.bitrate import shenoy_b
from penn_nh.eval.sweep_theta import sweep_theta


def _make_perfect_scores(n_classes, n_trials, rng):
    """Scores with margin 1.0 for correct class."""
    labels = rng.integers(0, n_classes, size=n_trials)
    scores = np.zeros((n_trials, n_classes))
    for i, lbl in enumerate(labels):
        scores[i, lbl] = 1.0
    return scores, labels


def test_perfect_scores_high_theta():
    rng = np.random.default_rng(0)
    scores, labels = _make_perfect_scores(4, 50, rng)
    theta_grid = np.array([0.0, 0.3, 0.5, 0.9])
    result = sweep_theta(scores, labels, theta_grid, t=60.0)

    # All thetas <= 0.9 accept all trials (margin is 1.0)
    for i in range(len(theta_grid)):
        assert result["n_accepted"][i] == 50
        assert result["S_c"][i] == 50
        assert result["S_i"][i] == 0

    expected_B = shenoy_b(50, 0, 4, 60.0)
    for i in range(len(theta_grid)):
        assert result["B"][i] == pytest.approx(expected_B)


def test_theta_above_max_margin_rejects_all():
    rng = np.random.default_rng(0)
    scores, labels = _make_perfect_scores(4, 50, rng)
    # margin is exactly 1.0, theta=1.0 means (top1-top2) > 1.0 fails
    theta_grid = np.array([1.0, 2.0])
    result = sweep_theta(scores, labels, theta_grid, t=60.0)
    for i in range(len(theta_grid)):
        assert result["B"][i] == 0.0
        assert result["S_c"][i] == 0


def test_argmax_theta_in_grid():
    rng = np.random.default_rng(0)
    scores, labels = _make_perfect_scores(4, 50, rng)
    theta_grid = np.array([0.0, 0.5, 0.9, 1.5])
    result = sweep_theta(scores, labels, theta_grid, t=60.0)
    assert result["argmax_theta"] in theta_grid


def test_shape_validation():
    # 1D scores
    with pytest.raises(ValueError):
        sweep_theta(np.array([1.0, 2.0]), np.array([0, 1]), np.array([0.0]))

    # labels length mismatch
    with pytest.raises(ValueError):
        sweep_theta(np.ones((5, 3)), np.array([0, 1]), np.array([0.0]))


def test_theta_grid_validation():
    scores = np.ones((5, 3))
    labels = np.zeros(5, dtype=int)

    # Unsorted
    with pytest.raises(ValueError):
        sweep_theta(scores, labels, np.array([0.5, 0.3, 0.8]))

    # Negative
    with pytest.raises(ValueError):
        sweep_theta(scores, labels, np.array([-0.1, 0.0, 0.5]))


def test_rejections_hurt_B():
    """Sweep completes and returns finite values with noisy scores."""
    rng = np.random.default_rng(42)
    n_classes = 4
    n_trials = 200
    labels = rng.integers(0, n_classes, size=n_trials)
    # Noisy scores: correct class gets ~0.7, others ~0.1
    scores = rng.uniform(0, 0.2, size=(n_trials, n_classes))
    for i in range(n_trials):
        scores[i, labels[i]] += 0.6

    theta_grid = np.linspace(0.0, 1.0, 20)
    result = sweep_theta(scores, labels, theta_grid, t=60.0)

    assert np.all(np.isfinite(result["B"]))
    assert len(result["B"]) == len(theta_grid)

import numpy as np
import pytest

from penn_nh.collect.labeler import label_from_xy
from penn_nh.decoder.linear_probe import LinearProbeFactory
from penn_nh.eval.sweep_n import sweep_n


def _make_annulus_data(M, rng, n_true_sectors=8):
    """Synthetic data on annulus with one-hot features matching 8-sector ground truth."""
    angles = rng.uniform(0, 2 * np.pi, M)
    radii = rng.uniform(0.6, 1.0, M)
    xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    labels = np.array([label_from_xy(xy[i], n_true_sectors, deadzone=0.5) for i in range(M)])
    # One-hot of true label + noise, shape (M, 8, 1)
    X = rng.standard_normal((M, n_true_sectors, 1)) * 0.1
    for i in range(M):
        if labels[i] is not None:
            X[i, labels[i], 0] += 3.0

    return X, xy


def test_sweep_recovers_best_n():
    rng = np.random.default_rng(42)
    X_train, xy_train = _make_annulus_data(400, rng)
    X_val, xy_val = _make_annulus_data(200, rng)
    n_grid = np.array([4, 6, 8, 10, 12])
    factory = LinearProbeFactory(C=1.0, max_iter=2000, random_state=0)

    result = sweep_n(
        X_train, xy_train, X_val, xy_val,
        factory=factory, n_grid=n_grid, deadzone=0.5, t=60.0,
    )

    # N=8 should be among the top-2 performers
    order = np.argsort(result["B"])[::-1]
    top2_N = set(n_grid[order[:2]])
    assert 8 in top2_N


def test_deadzone_drops_samples():
    rng = np.random.default_rng(0)
    X_good, xy_good = _make_annulus_data(200, rng)
    # 50 samples inside deadzone
    xy_dead = rng.uniform(-0.1, 0.1, (50, 2))
    X_dead = rng.standard_normal((50, 8, 1))
    X_train = np.vstack([X_good, X_dead])
    xy_train = np.vstack([xy_good, xy_dead])

    X_val, xy_val = _make_annulus_data(100, rng)
    factory = LinearProbeFactory(max_iter=500)
    result = sweep_n(
        X_train, xy_train, X_val, xy_val,
        factory=factory, n_grid=np.array([4]), deadzone=0.5, t=60.0,
    )
    assert result["n_train_kept"][0] < X_train.shape[0]


def test_shape_validation():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 8, 1))
    xy = rng.standard_normal((100, 2))
    factory = LinearProbeFactory()

    # N=2 in grid
    with pytest.raises(ValueError):
        sweep_n(X, xy, X, xy, factory, np.array([2, 4]))

    # Mismatched lengths
    with pytest.raises(ValueError):
        sweep_n(X, xy[:50], X, xy, factory, np.array([4]))


def test_theta_grid_default():
    rng = np.random.default_rng(0)
    X_train, xy_train = _make_annulus_data(200, rng)
    X_val, xy_val = _make_annulus_data(100, rng)
    factory = LinearProbeFactory(max_iter=500)

    result = sweep_n(
        X_train, xy_train, X_val, xy_val,
        factory=factory, n_grid=np.array([4]), deadzone=0.5, t=60.0,
    )
    # Default theta_grid is linspace(0, 0.5, 26), so best theta in [0, 0.5]
    assert 0.0 <= result["theta"][0] <= 0.5

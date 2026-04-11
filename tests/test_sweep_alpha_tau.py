import numpy as np
import pytest

from penn_nh.decoder.stage2_proto import _stable_softmax
from penn_nh.eval.sweep_alpha_tau import sweep_alpha_tau
from penn_nh.eval.sweep_theta import sweep_theta


def _make_separable(n_classes=4, m=100, seed=0):
    """Logits with perfect class separation; cos is random."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=m)
    logits = np.zeros((m, n_classes))
    for i, lbl in enumerate(labels):
        logits[i, lbl] = 5.0
    cos = rng.uniform(-1, 1, (m, n_classes))
    return logits, cos, labels


def test_alpha_extremes_match_components():
    logits, cos, labels = _make_separable()
    tau = 0.5
    theta_grid = np.linspace(0.0, 0.5, 6)

    # alpha=1 -> pure softmax(logits)
    res = sweep_alpha_tau(
        logits, cos, labels,
        alpha_grid=np.array([1.0]),
        tau_grid=np.array([tau]),
        theta_grid=theta_grid,
    )
    expected = sweep_theta(_stable_softmax(logits), labels, theta_grid)
    assert res["B"][0, 0] == pytest.approx(expected["B"][np.argmax(expected["B"])])

    # alpha=0 -> pure softmax(cos/tau)
    res0 = sweep_alpha_tau(
        logits, cos, labels,
        alpha_grid=np.array([0.0]),
        tau_grid=np.array([tau]),
        theta_grid=theta_grid,
    )
    expected0 = sweep_theta(_stable_softmax(cos / tau), labels, theta_grid)
    assert res0["B"][0, 0] == pytest.approx(expected0["B"][np.argmax(expected0["B"])])


def test_argmax_in_grid():
    logits, cos, labels = _make_separable()
    alpha_grid = np.array([0.0, 0.3, 0.7, 1.0])
    tau_grid = np.array([0.1, 0.5, 1.0])
    theta_grid = np.linspace(0.0, 0.5, 6)
    res = sweep_alpha_tau(logits, cos, labels, alpha_grid, tau_grid, theta_grid)
    assert res["argmax_alpha"] in alpha_grid
    assert res["argmax_tau"] in tau_grid


def test_b_max_is_grid_max():
    logits, cos, labels = _make_separable()
    alpha_grid = np.array([0.0, 0.5, 1.0])
    tau_grid = np.array([0.1, 1.0])
    theta_grid = np.linspace(0.0, 0.3, 4)
    res = sweep_alpha_tau(logits, cos, labels, alpha_grid, tau_grid, theta_grid)
    assert res["B_max"] == pytest.approx(res["B"].max())


def test_shape_validation():
    a = np.ones((10, 4))
    labels = np.zeros(10, dtype=int)
    tg = np.array([0.0])

    # mismatched logits/cos
    with pytest.raises(ValueError):
        sweep_alpha_tau(a, np.ones((10, 3)), labels,
                        np.array([0.5]), np.array([1.0]), tg)
    # mismatched labels length
    with pytest.raises(ValueError):
        sweep_alpha_tau(a, a, np.zeros(5, dtype=int),
                        np.array([0.5]), np.array([1.0]), tg)


def test_alpha_out_of_range():
    a = np.ones((10, 4))
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        sweep_alpha_tau(a, a, labels,
                        np.array([0.5, 1.5]), np.array([1.0]),
                        np.array([0.0]))


def test_tau_nonpositive():
    a = np.ones((10, 4))
    labels = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        sweep_alpha_tau(a, a, labels,
                        np.array([0.5]), np.array([0.0, 1.0]),
                        np.array([0.0]))

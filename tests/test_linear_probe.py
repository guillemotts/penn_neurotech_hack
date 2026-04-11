import numpy as np
import pytest

from penn_nh.decoder.linear_probe import FittedLinearProbe, LinearProbeFactory


def _make_synthetic(M, n_classes, C_ch, T, rng):
    """Class-specific bias in channel dimension + Gaussian noise."""
    X = rng.standard_normal((M, C_ch, T)) * 0.1
    y = rng.integers(0, n_classes, size=M)
    for i in range(M):
        X[i, y[i] % C_ch, :] += 2.0
    return X, y


def test_fit_recovers_linearly_separable():
    rng = np.random.default_rng(0)
    X, y = _make_synthetic(200, 4, 4, 10, rng)
    factory = LinearProbeFactory()
    fitted = factory.fit(X, y, n=4)
    scores = fitted.score(X)
    preds = np.argmax(scores, axis=1)
    acc = (preds == y).mean()
    assert acc > 0.95


def test_fit_2d_input():
    rng = np.random.default_rng(1)
    M, D, n_classes = 200, 40, 4
    X = rng.standard_normal((M, D)) * 0.1
    y = rng.integers(0, n_classes, size=M)
    for i in range(M):
        X[i, y[i] * (D // n_classes)] += 2.0
    factory = LinearProbeFactory()
    fitted = factory.fit(X, y, n=n_classes)
    scores = fitted.score(X)
    preds = np.argmax(scores, axis=1)
    acc = (preds == y).mean()
    assert acc > 0.8


def test_fit_missing_class_raises():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4, 10))
    y = rng.integers(0, 3, size=100)  # only classes 0,1,2
    factory = LinearProbeFactory()
    with pytest.raises(ValueError, match="3"):
        factory.fit(X, y, n=4)


def test_score_shape():
    rng = np.random.default_rng(0)
    X, y = _make_synthetic(100, 4, 4, 10, rng)
    factory = LinearProbeFactory()
    fitted = factory.fit(X, y, n=4)
    X_test = rng.standard_normal((50, 4, 10))
    scores = fitted.score(X_test)
    assert scores.shape == (50, 4)


def test_score_before_fit():
    # FittedLinearProbe without valid pipeline: AttributeError is acceptable
    probe = FittedLinearProbe.__new__(FittedLinearProbe)
    with pytest.raises(AttributeError):
        probe.score(np.zeros((5, 10)))

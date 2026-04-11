import numpy as np
import pytest

from penn_nh.decoder.stage2_proto import combine, cosine_scores, fit_prototypes


def test_fit_orthogonal():
    rng = np.random.default_rng(0)
    # Class 0 near e1, class 1 near e2
    e0 = np.array([1.0, 0.0]) + rng.normal(0, 0.01, size=(10, 2))
    e1 = np.array([0.0, 1.0]) + rng.normal(0, 0.01, size=(10, 2))
    embeddings = np.vstack([e0, e1])
    labels = np.array([0] * 10 + [1] * 10)

    protos = fit_prototypes(embeddings, labels, n=2)
    assert protos.shape == (2, 2)
    for i in range(2):
        assert np.linalg.norm(protos[i]) == pytest.approx(1.0, abs=1e-9)

    # Cosine scores recover labels
    for i, emb in enumerate(embeddings):
        scores = cosine_scores(emb, protos)
        assert np.argmax(scores) == labels[i]


def test_fit_missing_class():
    embeddings = np.array([[1.0, 0.0], [2.0, 0.0]])
    labels = np.array([0, 0])
    with pytest.raises(ValueError, match="1"):
        fit_prototypes(embeddings, labels, n=2)


def test_cosine_shape_single():
    protos = np.eye(3)
    e = np.array([1.0, 0.0, 0.0])
    out = cosine_scores(e, protos)
    assert out.shape == (3,)


def test_cosine_shape_batch():
    protos = np.eye(3)
    e = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = cosine_scores(e, protos)
    assert out.shape == (2, 3)


def test_combine_alpha_extremes():
    logits = np.array([2.0, 1.0, 0.5])
    cos = np.array([0.1, 0.9, 0.5])
    tau = 0.5

    # alpha=1: pure softmax(logits)
    out1 = combine(logits, cos, alpha=1.0, tau=tau)
    x = logits - logits.max()
    expected = np.exp(x) / np.exp(x).sum()
    np.testing.assert_allclose(out1, expected, atol=1e-9)

    # alpha=0: pure softmax(cos/tau)
    out0 = combine(logits, cos, alpha=0.0, tau=tau)
    x = cos / tau
    x = x - x.max()
    expected = np.exp(x) / np.exp(x).sum()
    np.testing.assert_allclose(out0, expected, atol=1e-9)


def test_combine_sums_to_one():
    logits = np.array([[2.0, 1.0, 0.5], [0.1, 0.2, 0.3]])
    cos = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]])
    out = combine(logits, cos, alpha=0.5, tau=1.0)
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], atol=1e-12)


def test_combine_validation():
    a = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        combine(a, a, alpha=-0.1, tau=1.0)
    with pytest.raises(ValueError):
        combine(a, a, alpha=1.1, tau=1.0)
    with pytest.raises(ValueError):
        combine(a, a, alpha=0.5, tau=0.0)
    with pytest.raises(ValueError):
        combine(a, np.array([1.0, 2.0, 3.0]), alpha=0.5, tau=1.0)


def test_combine_numerical_stability():
    logits = np.array([1000.0, 1001.0, 999.0])
    cos = np.array([0.5, 0.6, 0.4])
    out = combine(logits, cos, alpha=0.5, tau=1.0)
    assert np.all(np.isfinite(out))

import numpy as np
import pytest
import torch

from penn_nh.decoder.stage1_cnn import Stage1Config, Stage1Factory


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def make_clusters(n_classes, m_per_class, c, t, sep=3.0):
    labels = np.repeat(np.arange(n_classes), m_per_class).astype(np.int64)
    M = n_classes * m_per_class
    means = (sep * np.eye(n_classes, c)).astype(np.float32)
    X = means[labels][:, :, None] + np.random.randn(M, c, t).astype(np.float32)
    return X, labels


def test_eegnet_fits_clusters():
    X, y = make_clusters(4, 50, c=8, t=40)
    cfg = Stage1Config(arch="eegnet", epochs=20, device="cpu")
    fitted = Stage1Factory(cfg).fit(X, y, 4)
    scores = fitted.score(X)
    acc = (scores.argmax(axis=1) == y).mean()
    assert acc > 0.90


def test_dense_fits_clusters():
    X, y = make_clusters(4, 50, c=8, t=40)
    cfg = Stage1Config(arch="dense", epochs=20, device="cpu")
    fitted = Stage1Factory(cfg).fit(X, y, 4)
    scores = fitted.score(X)
    acc = (scores.argmax(axis=1) == y).mean()
    assert acc > 0.90


def test_forward_returns_logits_and_embedding():
    X, y = make_clusters(4, 20, c=8, t=40)
    cfg = Stage1Config(arch="eegnet", epochs=5, device="cpu")
    fitted = Stage1Factory(cfg).fit(X, y, 4)
    logits, emb = fitted.forward(X)
    assert logits.shape == (X.shape[0], 4)
    assert emb.shape[0] == X.shape[0]
    assert emb.shape[1] > 0
    assert np.all(np.isfinite(logits))
    assert np.all(np.isfinite(emb))


def test_score_is_log_softmax():
    X, y = make_clusters(4, 20, c=8, t=40)
    cfg = Stage1Config(arch="eegnet", epochs=5, device="cpu")
    fitted = Stage1Factory(cfg).fit(X, y, 4)
    scores = fitted.score(X)
    sums = np.exp(scores).sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-4)


def test_missing_class_raises():
    X = np.random.randn(40, 8, 40).astype(np.float32)
    y = np.zeros(40, dtype=np.int64)
    with pytest.raises(ValueError, match="Missing"):
        Stage1Factory(Stage1Config(device="cpu")).fit(X, y, 4)


def test_onnx_roundtrip(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    X, y = make_clusters(4, 30, c=8, t=40)
    cfg = Stage1Config(arch="eegnet", epochs=10, device="cpu")
    fitted = Stage1Factory(cfg).fit(X, y, 4)
    path = str(tmp_path / "model.onnx")
    fitted.export_onnx(path, (1, 8, 40))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    X_sub = X[:5].astype(np.float32)
    ort_out = sess.run(None, {"input": X_sub})
    logits_py, emb_py = fitted.forward(X_sub)
    np.testing.assert_allclose(ort_out[0], logits_py, atol=1e-4)
    np.testing.assert_allclose(ort_out[1], emb_py, atol=1e-4)


def test_device_override_cpu():
    X, y = make_clusters(4, 20, c=8, t=40)
    cfg = Stage1Config(device="cpu", epochs=5)
    fitted = Stage1Factory(cfg).fit(X, y, 4)
    assert fitted.device.type == "cpu"


def test_reproducibility():
    X, y = make_clusters(4, 30, c=8, t=40)
    cfg = Stage1Config(epochs=10, device="cpu", seed=42)
    scores1 = Stage1Factory(cfg).fit(X, y, 4).score(X)
    scores2 = Stage1Factory(cfg).fit(X, y, 4).score(X)
    np.testing.assert_allclose(scores1, scores2, atol=1e-6)

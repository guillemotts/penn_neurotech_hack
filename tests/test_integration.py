import numpy as np
import onnxruntime as ort
import pytest

from penn_nh.decoder.linear_probe import LinearProbeFactory
from penn_nh.decoder.stage1_cnn import Stage1Config, Stage1Factory
from penn_nh.decoder.stage2_proto import combine, cosine_scores, fit_prototypes
from penn_nh.decoder.stage3_reject import decide
from penn_nh.eval.bitrate import shenoy_b_from_confusion
from penn_nh.eval.sweep_theta import sweep_theta
from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.game.interface import UnpacedTargetHit
from penn_nh.game.session import Session
from penn_nh.game.ui import NullRenderer
from penn_nh.integration.mock_decoder import mock_decoder_stream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raw_poisson(n_classes=8, trials_per_class=60, c=32, fs=32000,
                     dur_s=0.1, seed=0):
    rng = np.random.default_rng(seed)
    T = int(fs * dur_s)
    M = n_classes * trials_per_class
    X = rng.normal(0, 5, (M, c, T)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), trials_per_class).astype(np.int64)
    ch_per = c // n_classes
    spike = np.array([-25, -50, -25], dtype=np.float32)
    for i in range(M):
        cls = y[i]
        for ch in range(c):
            rate = 120.0 if ch_per * cls <= ch < ch_per * (cls + 1) else 5.0
            n_sp = rng.poisson(rate * dur_s)
            if n_sp > 0:
                for st in rng.integers(1, T - 1, size=n_sp):
                    X[i, ch, st - 1 : st + 2] += spike
    perm = rng.permutation(M)
    return X[perm], y[perm]


def stratified_split(y, fracs=(0.7, 0.15, 0.15), seed=0):
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    for c in np.unique(y):
        idx = np.where(y == c)[0].copy()
        rng.shuffle(idx)
        n = len(idx)
        c0, c1 = int(n * fracs[0]), int(n * (fracs[0] + fracs[1]))
        splits[0].extend(idx[:c0])
        splits[1].extend(idx[c0:c1])
        splits[2].extend(idx[c1:])
    return [np.array(s) for s in splits]


def featurize_all(X_raw, tr, va, te, cfg=None):
    cfg = cfg or FeatureConfig()
    stack = FeatureStack(cfg)
    train_feat = [stack.transform(X_raw[i]) for i in tr]
    stack.fit_zscore(train_feat)
    def xform(idx):
        return np.stack([stack.transform(X_raw[i]) for i in idx]).astype(np.float32)
    return xform(tr), xform(va), xform(te)


def build_confusion(scores, labels, theta, n):
    C = np.zeros((n, n), dtype=int)
    for i in range(len(scores)):
        d = decide(scores[i], theta)
        if d is not None:
            C[labels[i], d] += 1
    return C


def row_stochastic(C):
    n = C.shape[0]
    P = np.zeros_like(C, dtype=float)
    for i in range(n):
        s = C[i].sum()
        P[i] = C[i] / s if s > 0 else np.full(n, 1.0 / n)
    return P


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory_name", ["linear", "eegnet"])
def test_end_to_end_pipeline(factory_name):
    N = 8
    X_raw, y = make_raw_poisson(n_classes=N, trials_per_class=50, seed=0)
    tr, va, te = stratified_split(y, seed=0)
    X_tr, X_va, X_te = featurize_all(X_raw, tr, va, te)
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    if factory_name == "linear":
        fitted = LinearProbeFactory().fit(X_tr, y_tr, N)
        scores_va = fitted.score(X_va)
        scores_te = fitted.score(X_te)
    else:
        cfg = Stage1Config(arch="eegnet", epochs=15, seed=0)
        fitted = Stage1Factory(cfg).fit(X_tr, y_tr, N)
        logits_va, emb_va = fitted.forward(X_va)
        protos = fit_prototypes(emb_va, y_va, N)
        cos_va = cosine_scores(emb_va, protos)
        scores_va = combine(logits_va, cos_va, alpha=0.5, tau=0.1)
        logits_te, emb_te = fitted.forward(X_te)
        cos_te = cosine_scores(emb_te, protos)
        scores_te = combine(logits_te, cos_te, alpha=0.5, tau=0.1)

    theta_grid = np.linspace(0.0, 0.5, 11)
    sw = sweep_theta(scores_va, y_va, theta_grid, t=60.0)
    best_theta = sw["argmax_theta"]

    C = build_confusion(scores_te, y_te, best_theta, N)
    P = row_stochastic(C)

    game = UnpacedTargetHit(n=N, session_seconds=5.0,
                            rng=np.random.default_rng(99))
    stream = mock_decoder_stream(game, P, mean_interval_s=0.25,
                                 rng=np.random.default_rng(100))
    sess = Session(game=game, decoder_stream=stream,
                   renderer=NullRenderer(), time_source="iterator")
    report = sess.run()

    assert report["B"] > 1.5
    assert report["S_c"] > report["S_i"]


def test_feature_stack_fit_transform_parity():
    X_raw, _ = make_raw_poisson(n_classes=4, trials_per_class=10,
                                c=16, fs=32000, dur_s=0.05, seed=7)
    cfg = FeatureConfig()
    stack = FeatureStack(cfg)
    train_feat = [stack.transform(X_raw[i]) for i in range(20)]
    stack.fit_zscore(train_feat)
    out1 = stack.transform(X_raw[25])
    out2 = stack.transform(X_raw[25])
    np.testing.assert_array_equal(out1, out2)


def test_onnx_score_matches_torch(tmp_path):
    X_raw, y = make_raw_poisson(n_classes=4, trials_per_class=20,
                                c=16, fs=32000, dur_s=0.1, seed=42)
    cfg = FeatureConfig()
    stack = FeatureStack(cfg)
    feats = [stack.transform(X_raw[i]) for i in range(len(X_raw))]
    stack.fit_zscore(feats[: len(feats) // 2])
    X = np.stack([stack.transform(X_raw[i]) for i in range(len(X_raw))]
                 ).astype(np.float32)

    fitted = Stage1Factory(Stage1Config(arch="eegnet", epochs=5, seed=0)
                           ).fit(X, y, n=4)
    torch_logits, torch_emb = fitted.forward(X[:10])

    path = str(tmp_path / "model.onnx")
    fitted.export_onnx(path, example_shape=(1, X.shape[1], X.shape[2]))

    ort_sess = ort.InferenceSession(path)
    ort_logits, ort_emb = ort_sess.run(None, {"input": X[:10]})

    np.testing.assert_allclose(torch_logits, ort_logits, atol=1e-4)
    np.testing.assert_allclose(torch_emb, ort_emb, atol=1e-4)

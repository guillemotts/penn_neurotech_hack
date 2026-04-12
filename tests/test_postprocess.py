"""
tests/test_postprocess.py -- Tests for src/decoder/postprocess.py
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "src")
from decoder.postprocess import (
    causal_moving_average,
    discretize_stick,
    discretize_button,
    combine,
    proxy_bitrate,
)


# ── causal_moving_average ─────────────────────────────────────────────

def test_cma_k1_identity():
    Y = np.random.randn(20, 10).astype(np.float32)
    out = causal_moving_average(Y, K=1)
    np.testing.assert_array_equal(out, Y)


def test_cma_matches_naive():
    rng = np.random.RandomState(42)
    Y = rng.randn(50, 10).astype(np.float64)
    for K in [2, 5, 10]:
        out = causal_moving_average(Y, K)
        # Naive reference
        T, C = Y.shape
        naive = np.empty_like(Y)
        for t in range(T):
            start = max(0, t - K + 1)
            naive[t] = Y[start:t + 1].mean(axis=0)
        np.testing.assert_allclose(out, naive, atol=1e-12,
                                   err_msg=f"K={K} mismatch")


def test_cma_no_future_leak():
    rng = np.random.RandomState(7)
    Y = rng.randn(30, 10).astype(np.float64)
    K = 5
    out_full = causal_moving_average(Y, K)
    # Mutate Y[15:] and recompute; out[:15] must be unchanged
    Y2 = Y.copy()
    Y2[15:] = rng.randn(15, 10)
    out_partial = causal_moving_average(Y2, K)
    np.testing.assert_array_equal(out_full[:15], out_partial[:15])


# ── discretize_stick ──────────────────────────────────────────────────

def test_stick_sector0_n4():
    """Point at (1, 0) → sector 0 for N=4."""
    Y = np.zeros((1, 10), dtype=np.float32)
    Y[0, 0] = 1.0  # x
    Y[0, 1] = 0.0  # y
    sym = discretize_stick(Y, N=4, stick_thresh=0.5)
    assert sym[0] == 0


def test_stick_sector0_n8():
    """Point at (1, 0) → sector 0 for N=8."""
    Y = np.zeros((1, 10), dtype=np.float32)
    Y[0, 0] = 1.0
    Y[0, 1] = 0.0
    sym = discretize_stick(Y, N=8, stick_thresh=0.5)
    assert sym[0] == 0


def test_stick_up_n4():
    """Point at (0, 1) → sector 1 for N=4 (90 degrees, sector width 90)."""
    Y = np.zeros((1, 10), dtype=np.float32)
    Y[0, 0] = 0.0
    Y[0, 1] = 1.0
    sym = discretize_stick(Y, N=4, stick_thresh=0.5)
    assert sym[0] == 1


def test_stick_up_n8():
    """Point at (0, 1) → sector 2 for N=8 (90 degrees, sector width 45)."""
    Y = np.zeros((1, 10), dtype=np.float32)
    Y[0, 0] = 0.0
    Y[0, 1] = 1.0
    sym = discretize_stick(Y, N=8, stick_thresh=0.5)
    assert sym[0] == 2


def test_stick_below_threshold():
    Y = np.zeros((3, 10), dtype=np.float32)
    Y[0, 0] = 0.01
    Y[1, 0] = 0.05
    Y[2, 0] = 0.09
    sym = discretize_stick(Y, N=4, stick_thresh=0.10)
    np.testing.assert_array_equal(sym, [-1, -1, -1])


# ── proxy_bitrate ─────────────────────────────────────────────────────

def test_proxy_bitrate_hand_example():
    """10-bin example: S_c=3, S_i=1, n_noop=6."""
    gt   = np.array([-1, -1, -1,  0,  1,  2, -1, -1, -1,  0])
    pred = np.array([-1, -1, -1,  0,  1,  3, -1, -1, -1, -1])
    # pred_emit (pred != -1): indices 3, 4, 5 → 3 emitted
    # S_c: (pred==gt & emit): idx 3(0==0), idx 4(1==1) → 2
    # S_i: (pred!=gt & emit): idx 5(3!=2) → 1
    # n_noop: pred==-1 → 7

    r_neutral = proxy_bitrate(gt, pred, N_eff=4, bin_seconds=0.01,
                              silence_mode="neutral")
    assert r_neutral["S_c"] == 2
    assert r_neutral["S_i"] == 1
    assert r_neutral["n_noop"] == 7

    r_correct = proxy_bitrate(gt, pred, N_eff=4, bin_seconds=0.01,
                              silence_mode="correct")
    # correct: S_c += count(pred==-1 & gt==-1) = 6 (indices 0,1,2,6,7,8)
    assert r_correct["S_c"] == 2 + 6  # 8
    assert r_correct["S_i"] == 1

    # B values differ
    assert r_correct["B"] > r_neutral["B"]


def test_proxy_bitrate_perfect():
    gt   = np.array([0, 1, 2, 3])
    pred = np.array([0, 1, 2, 3])
    r = proxy_bitrate(gt, pred, N_eff=4, bin_seconds=0.01,
                      silence_mode="neutral")
    assert r["S_c"] == 4
    assert r["S_i"] == 0
    assert r["accuracy_emitted"] == 1.0
    assert r["B"] > 0

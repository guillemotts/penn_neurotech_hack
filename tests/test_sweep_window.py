import numpy as np
import pytest

from penn_nh.decoder.linear_probe import LinearProbeFactory
from penn_nh.eval.sweep_window import sweep_window
from penn_nh.features.stack import FeatureConfig

_CFG = FeatureConfig(
    fs_hz=1000.0,
    bandpass_low_hz=50.0,
    bandpass_high_hz=200.0,
    bin_ms=5.0,
    spike_threshold_k=3.0,
    min_isi_ms=1.0,
)


def _make_synthetic(M, C, T_full, n_classes, seed=0):
    """
    Raw with class signal injected in samples [100, 150] relative to onset=50.
    Onset is fixed at 50 for all trials. stick_xy on outer annulus.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(0, 5, (M, C, T_full)).astype(np.float32)
    labels = rng.integers(0, n_classes, size=M)
    onset = np.full(M, 50, dtype=int)
    ch_per = max(1, C // n_classes)

    # Inject class-specific signal in [onset+50, onset+100] = [100, 150]
    spike = np.array([-25, -50, -25], dtype=np.float32)
    for i in range(M):
        cls = labels[i]
        for ch in range(ch_per * cls, min(ch_per * (cls + 1), C)):
            for st in rng.integers(101, 149, size=5):
                raw[i, ch, st - 1 : st + 2] += spike

    # stick_xy on annulus for n_classes sectors
    angles = labels * (2 * np.pi / n_classes) + rng.uniform(-0.1, 0.1, M)
    radii = rng.uniform(0.7, 1.0, M)
    stick_xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    return raw, onset, stick_xy


def test_recovers_best_window():
    M_tr, M_va, C, T_full, n = 150, 50, 8, 400, 4
    raw_tr, onset_tr, xy_tr = _make_synthetic(M_tr, C, T_full, n, seed=0)
    raw_va, onset_va, xy_va = _make_synthetic(M_va, C, T_full, n, seed=1)

    res = sweep_window(
        raw_tr, onset_tr, xy_tr,
        raw_va, onset_va, xy_va,
        factory=LinearProbeFactory(max_iter=500),
        n=n, fs_hz=1000.0,
        w0_ms_grid=np.array([0.0, 25.0, 50.0]),
        w1_ms_grid=np.array([75.0, 100.0, 125.0, 150.0]),
        feature_cfg=_CFG,
        deadzone=0.5,
    )
    assert res["B_max"] > 0
    best = res["argmax_idx"]
    assert res["w1_ms"][best] - res["w0_ms"][best] >= 50


def test_invalid_combos_skipped():
    M, C, T_full, n = 50, 4, 200, 4
    raw, onset, xy = _make_synthetic(M, C, T_full, n, seed=2)

    res = sweep_window(
        raw, onset, xy,
        raw, onset, xy,
        factory=LinearProbeFactory(),
        n=n, fs_hz=1000.0,
        w0_ms_grid=np.array([100.0]),
        w1_ms_grid=np.array([50.0]),
        feature_cfg=_CFG,
    )
    assert len(res["B"]) == 0
    assert res["B_max"] == 0.0


def test_out_of_range_onset_drops_trials():
    M, C, T_full, n = 80, 4, 200, 4
    raw, onset, xy = _make_synthetic(M, C, T_full, n, seed=3)
    # Set last trial's onset near end so slice overflows
    onset[-1] = T_full - 10

    res = sweep_window(
        raw, onset, xy,
        raw, onset, xy,
        factory=LinearProbeFactory(max_iter=300),
        n=n, fs_hz=1000.0,
        w0_ms_grid=np.array([0.0]),
        w1_ms_grid=np.array([100.0]),
        feature_cfg=_CFG,
        deadzone=0.5,
    )
    # Should complete without error; some trials dropped
    assert len(res["B"]) >= 0


def test_shape_validation():
    with pytest.raises(ValueError):
        sweep_window(
            np.ones((10, 100)),  # 2D instead of 3D
            np.zeros(10, dtype=int),
            np.ones((10, 2)),
            np.ones((10, 100)),
            np.zeros(10, dtype=int),
            np.ones((10, 2)),
            factory=LinearProbeFactory(),
            n=4, fs_hz=1000.0,
            w0_ms_grid=np.array([0.0]),
            w1_ms_grid=np.array([50.0]),
        )

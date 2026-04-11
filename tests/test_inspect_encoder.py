import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from scripts.inspect_encoder import analyze, build_config, plot


def _make_synthetic(C=32, T=30000, fs=30000.0, seed=0):
    """Gaussian noise + injected negative spikes on ch 0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 10, (C, T)).astype(np.float32)
    spike = np.array([-30, -60, -30], dtype=np.float32)
    for st in rng.integers(2, T - 2, size=50):
        x[0, st - 1 : st + 2] += spike
    return x, fs


def test_analyze_shapes():
    x, fs = _make_synthetic()
    stats = analyze(x, fs)

    expected_keys = {
        "per_channel_rms", "median_rms", "threshold_crossings",
        "spectral_centroid", "isi_hist_bins", "isi_hist_counts",
    }
    assert expected_keys.issubset(stats.keys())

    assert len(stats["per_channel_rms"]) == 32
    assert isinstance(stats["median_rms"], float)
    assert stats["median_rms"] > 0
    assert len(stats["threshold_crossings"]) == 32
    assert stats["threshold_crossings"][0] > 0  # injected spikes
    assert len(stats["spectral_centroid"]) == 32
    assert len(stats["isi_hist_bins"]) == len(stats["isi_hist_counts"]) + 1


def test_plot_writes_png(tmp_path):
    x, fs = _make_synthetic()
    stats = analyze(x, fs)
    out = tmp_path / "test_plot.png"
    plot(x, fs, stats, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_build_config_shape():
    syn = pytest.importorskip("synapse")
    config = build_config(peripheral_id=104, n_channels=32, fs_hz=30000, udp_port=50038)
    assert isinstance(config, syn.Config)
    assert len(config.nodes) >= 1
    has_broadband = any(
        isinstance(node, syn.BroadbandSource) for node in config.nodes
    )
    assert has_broadband

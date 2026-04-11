import numpy as np
import pytest

from penn_nh.features.filters import apply_bandpass, car, design_bandpass
from penn_nh.features.power import spike_band_power
from penn_nh.features.spikes import bin_spike_counts, rms_per_channel, threshold_crossings
from penn_nh.features.stack import FeatureConfig, FeatureStack


# --- filters ---

def test_bandpass_passes_inband():
    fs = 32000.0
    b, a = design_bandpass(300.0, 3000.0, fs)
    t = np.arange(10000) / fs
    x = np.sin(2 * np.pi * 1000 * t)
    y = apply_bandpass(x, b, a)
    # Skip warmup (first 200 samples)
    rms_in = np.sqrt(np.mean(x[200:] ** 2))
    rms_out = np.sqrt(np.mean(y[200:] ** 2))
    assert abs(rms_out - rms_in) / rms_in < 0.05


def test_bandpass_rejects_outband():
    fs = 32000.0
    b, a = design_bandpass(300.0, 3000.0, fs)
    t = np.arange(10000) / fs
    x = np.sin(2 * np.pi * 50 * t)
    y = apply_bandpass(x, b, a)
    rms_in = np.sqrt(np.mean(x[200:] ** 2))
    rms_out = np.sqrt(np.mean(y[200:] ** 2))
    assert rms_out / rms_in < 0.10


def test_bandpass_validation():
    with pytest.raises(ValueError):
        design_bandpass(3000.0, 300.0, 32000.0)  # low >= high
    with pytest.raises(ValueError):
        design_bandpass(300.0, 16000.0, 32000.0)  # high >= fs/2
    with pytest.raises(ValueError):
        design_bandpass(0.0, 3000.0, 32000.0)  # low <= 0


def test_car_zero_mean():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 100))
    y = car(x)
    np.testing.assert_allclose(y.mean(axis=0), 0.0, atol=1e-12)


# --- spikes ---

def test_threshold_crossings_basic():
    x = np.zeros((1, 100))
    # Create negative-going spikes at t=10, 30, 60
    for t in [10, 30, 60]:
        x[0, t] = -2.0
    crossings = threshold_crossings(x, np.array([-1.0]), min_isi_samples=1)
    assert len(crossings) == 1
    np.testing.assert_array_equal(crossings[0], [10, 30, 60])


def test_threshold_refractory():
    x = np.zeros((1, 100))
    for t in [10, 11, 12]:
        x[0, t] = -2.0
    crossings = threshold_crossings(x, np.array([-1.0]), min_isi_samples=5)
    assert len(crossings[0]) == 1
    assert crossings[0][0] == 10


def test_bin_counts_shape():
    crossings = [
        np.array([5, 15, 25]),  # channel 0: bins 0, 1, 2
        np.array([95]),          # channel 1: bin 9
    ]
    counts = bin_spike_counts(crossings, n_bins=10, bin_samples=10)
    assert counts.shape == (2, 10)
    assert counts[0, 0] == 1
    assert counts[0, 1] == 1
    assert counts[0, 2] == 1
    assert counts[1, 9] == 1


# --- power ---

def test_spike_band_power_constant():
    x = np.ones((2, 100))
    p = spike_band_power(x, bin_samples=10)
    assert p.shape == (2, 10)
    np.testing.assert_allclose(p, 1.0)


def test_spike_band_power_drops_partial():
    x = np.ones((2, 95))
    p = spike_band_power(x, bin_samples=10)
    assert p.shape == (2, 9)


# --- stack ---

def test_stack_transform_shape():
    cfg = FeatureConfig()
    stack = FeatureStack(cfg)
    raw = np.random.default_rng(0).standard_normal((32, 640))
    out = stack.transform(raw)
    # Both streams: 32 spike_counts + 32 spike_power = 64 channels
    # 640 samples / 160 bin_samples = 4 bins
    assert out.shape == (64, 4)
    assert out.dtype == np.float32


def test_stack_single_stream():
    cfg = FeatureConfig(use_spike_power=False)
    stack = FeatureStack(cfg)
    raw = np.random.default_rng(0).standard_normal((32, 640))
    out = stack.transform(raw)
    assert out.shape == (32, 4)


def test_stack_no_streams_raises():
    cfg = FeatureConfig(use_spike_counts=False, use_spike_power=False)
    stack = FeatureStack(cfg)
    raw = np.random.default_rng(0).standard_normal((32, 640))
    with pytest.raises(ValueError):
        stack.transform(raw)


def test_stack_zscore_applied():
    cfg = FeatureConfig()
    stack = FeatureStack(cfg)
    rng = np.random.default_rng(42)
    # Use longer windows (3200 samples = 20 bins) and 30 windows for stable stats
    windows = [rng.standard_normal((32, 3200)) for _ in range(30)]
    features = [stack.transform(w) for w in windows]
    stack.fit_zscore(features)
    # Transform one of the training windows
    out = stack.transform(windows[0])
    # Mean across bins should be near zero, std near 1 (tolerance 0.3 given small n)
    per_feature_mean = out.mean(axis=1)
    per_feature_std = out.std(axis=1)
    assert np.abs(per_feature_mean).mean() < 0.3
    assert np.abs(per_feature_std - 1.0).mean() < 0.3


def test_stack_causal_filter_state_fresh():
    cfg = FeatureConfig()
    stack = FeatureStack(cfg)
    raw = np.random.default_rng(0).standard_normal((32, 640))
    out1 = stack.transform(raw)
    out2 = stack.transform(raw)
    np.testing.assert_array_equal(out1, out2)


def test_lowrate_shape_preserved():
    from penn_nh.features.stack import FeatureConfigLowRate, LowRateFeatureStack
    cfg = FeatureConfigLowRate()
    fs = LowRateFeatureStack(cfg)
    raw = np.random.RandomState(0).randn(64, 42).astype(np.float32)
    out = fs.transform(raw)
    assert out.shape == (64, 42)
    assert out.dtype == np.float32


def test_lowrate_bandpass_passes_inband():
    from penn_nh.features.stack import FeatureConfigLowRate, LowRateFeatureStack
    cfg = FeatureConfigLowRate(use_zscore=False, use_car=False)
    fs = LowRateFeatureStack(cfg)
    t = np.arange(2000) / 105.0
    sig = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)  # 10 Hz, in band
    raw = np.tile(sig, (4, 1))
    out = fs.transform(raw)
    # skip warmup
    assert np.abs(out[:, 200:]).mean() > 0.3


def test_lowrate_bandpass_rejects_outband():
    from penn_nh.features.stack import FeatureConfigLowRate, LowRateFeatureStack
    cfg = FeatureConfigLowRate(use_zscore=False, use_car=False)
    fs = LowRateFeatureStack(cfg)
    t = np.arange(2000) / 105.0
    sig = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)  # 50 Hz, above band
    raw = np.tile(sig, (4, 1))
    out = fs.transform(raw)
    assert np.abs(out[:, 200:]).mean() < 0.1


def test_lowrate_zscore_fit():
    from penn_nh.features.stack import FeatureConfigLowRate, LowRateFeatureStack
    cfg = FeatureConfigLowRate(use_bandpass=False, use_car=False, use_zscore=True)
    fs = LowRateFeatureStack(cfg)
    rng = np.random.RandomState(0)
    train = [rng.randn(8, 200).astype(np.float32) * 5.0 + 3.0 for _ in range(5)]
    fs.fit_zscore(train)
    out = fs.transform(train[0])
    assert abs(out.mean()) < 0.5
    assert abs(out.std() - 1.0) < 0.5

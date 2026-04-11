from dataclasses import dataclass

import numpy as np

from penn_nh.features.filters import apply_bandpass, car, design_bandpass
from penn_nh.features.power import spike_band_power
from penn_nh.features.spikes import bin_spike_counts, rms_per_channel, threshold_crossings


@dataclass
class FeatureConfig:
    fs_hz: float = 32000.0
    bandpass_low_hz: float = 300.0
    bandpass_high_hz: float = 3000.0
    bandpass_order: int = 4
    bin_ms: float = 5.0
    spike_threshold_k: float = 3.0
    min_isi_ms: float = 1.0
    use_spike_counts: bool = True
    use_spike_power: bool = True


class FeatureStack:
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self._b, self._a = design_bandpass(
            cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.fs_hz, cfg.bandpass_order
        )
        self._bin_samples = int(round(cfg.bin_ms * 1e-3 * cfg.fs_hz))
        self._min_isi = max(1, int(round(cfg.min_isi_ms * 1e-3 * cfg.fs_hz)))
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit_zscore(self, X_list: list[np.ndarray]) -> None:
        """Accumulate per-feature mean/std across training windows. Each X is (S*C, n_bins)."""
        all_data = np.concatenate(X_list, axis=1)  # (S*C, total_bins)
        self._mean = all_data.mean(axis=1, keepdims=True)
        self._std = all_data.std(axis=1, keepdims=True)
        self._std[self._std < 1e-12] = 1.0

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """
        raw (C, T) single trial window.
        Pipeline: bandpass -> CAR -> features -> z-score -> (S*C, n_bins) float32.
        """
        if not self.cfg.use_spike_counts and not self.cfg.use_spike_power:
            raise ValueError("At least one feature stream must be enabled")

        # Bandpass (fresh zi per window)
        filtered = apply_bandpass(raw, self._b, self._a)
        # CAR
        filtered = car(filtered)

        C, T = filtered.shape
        n_bins = T // self._bin_samples

        streams = []

        if self.cfg.use_spike_counts:
            rms = rms_per_channel(filtered)
            thresholds = -self.cfg.spike_threshold_k * rms
            crossings = threshold_crossings(filtered, thresholds, self._min_isi)
            counts = bin_spike_counts(crossings, n_bins, self._bin_samples)
            streams.append(counts.astype(float))

        if self.cfg.use_spike_power:
            power = spike_band_power(filtered, self._bin_samples)
            streams.append(power)

        result = np.concatenate(streams, axis=0)

        if self._mean is not None and self._std is not None:
            result = (result - self._mean) / self._std

        return result.astype(np.float32)


@dataclass
class FeatureConfigLowRate:
    fs_hz: float = 105.0
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    bandpass_order: int = 4
    use_bandpass: bool = True
    use_car: bool = True
    use_zscore: bool = True


class LowRateFeatureStack:
    """
    LFP-band feature stack for low-sample-rate data (~50-500 Hz).
    Pipeline: causal bandpass -> CAR -> z-score. No binning, no spike features.
    Input and output shape (C, T) unchanged.
    """
    def __init__(self, cfg: FeatureConfigLowRate):
        self.cfg = cfg
        if cfg.use_bandpass:
            self._b, self._a = design_bandpass(
                cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.fs_hz, cfg.bandpass_order
            )
        else:
            self._b, self._a = None, None
        self._mean = None
        self._std = None

    def _filter_car(self, raw: np.ndarray) -> np.ndarray:
        x = raw.astype(np.float32)
        if self.cfg.use_bandpass:
            x = apply_bandpass(x, self._b, self._a)
        if self.cfg.use_car:
            x = car(x)
        return x

    def fit_zscore(self, raw_list: list) -> None:
        """Fit per-channel mean and std across concatenated filtered training windows."""
        if not self.cfg.use_zscore:
            return
        processed = [self._filter_car(r) for r in raw_list]
        cat = np.concatenate(processed, axis=1)  # (C, total_T)
        self._mean = cat.mean(axis=1, keepdims=True).astype(np.float32)
        self._std = cat.std(axis=1, keepdims=True).astype(np.float32)
        self._std[self._std < 1e-8] = 1.0

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """(C, T) -> (C, T) float32."""
        if raw.ndim != 2:
            raise ValueError(f"raw must be 2D (C, T), got shape {raw.shape}")
        x = self._filter_car(raw)
        if self.cfg.use_zscore and self._mean is not None:
            x = (x - self._mean) / self._std
        return x.astype(np.float32)

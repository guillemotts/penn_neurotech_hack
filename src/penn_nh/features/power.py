import numpy as np


def spike_band_power(x: np.ndarray, bin_samples: int) -> np.ndarray:
    """
    x (C, T) already bandpassed. Returns (C, n_bins) where n_bins = T // bin_samples.
    Power = mean(x**2) within each bin. Drop trailing partial bin.
    """
    C, T = x.shape
    n_bins = T // bin_samples
    trimmed = x[:, : n_bins * bin_samples]
    reshaped = trimmed.reshape(C, n_bins, bin_samples)
    return np.mean(reshaped ** 2, axis=2)

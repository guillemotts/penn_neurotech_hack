import numpy as np


def rms_per_channel(x: np.ndarray) -> np.ndarray:
    """x (C, T) -> (C,) RMS."""
    return np.sqrt(np.mean(x ** 2, axis=-1))


def threshold_crossings(
    x: np.ndarray, thresholds: np.ndarray, min_isi_samples: int
) -> list[np.ndarray]:
    """
    x (C, T), thresholds (C,) negative values (e.g., -3*RMS).
    Returns list of length C, each a 1D int array of crossing sample indices.
    Crossing = x[c, t] < thresholds[c] AND x[c, t-1] >= thresholds[c].
    Enforce min_isi_samples refractory per channel.
    """
    C, T = x.shape
    result = []
    for c in range(C):
        crossings = []
        last = -min_isi_samples - 1  # allow first crossing
        for t in range(1, T):
            if x[c, t] < thresholds[c] and x[c, t - 1] >= thresholds[c]:
                if t - last >= min_isi_samples:
                    crossings.append(t)
                    last = t
        result.append(np.array(crossings, dtype=int))
    return result


def bin_spike_counts(
    crossings: list[np.ndarray], n_bins: int, bin_samples: int
) -> np.ndarray:
    """Returns (C, n_bins) int array. Counts per bin."""
    C = len(crossings)
    counts = np.zeros((C, n_bins), dtype=int)
    for c in range(C):
        for t in crossings[c]:
            b = t // bin_samples
            if 0 <= b < n_bins:
                counts[c, b] += 1
    return counts

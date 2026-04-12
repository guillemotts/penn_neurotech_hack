import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# =====================================================
# 1) robust threshold
# =====================================================
def robust_threshold(x, k=4.0, mode="negative"):
    """
    Estimate threshold from median/MAD.

    Parameters
    ----------
    x : array-like
        1D signal
    k : float
        Threshold multiplier
    mode : {"positive", "negative", "absolute"}

    Returns
    -------
    thr : float
        Threshold value
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(x)

    if mode == "positive":
        return med + k * sigma
    elif mode == "negative":
        return med - k * sigma
    elif mode == "absolute":
        return k * sigma
    else:
        raise ValueError("mode must be 'positive', 'negative', or 'absolute'")


# =====================================================
# 2) threshold mask
# =====================================================
def threshold_mask(x, thr, mode="negative"):
    """
    Binary mask of samples above/below threshold.
    """
    x = np.asarray(x, dtype=float)

    if mode == "positive":
        return (x > thr).astype(int)
    elif mode == "negative":
        return (x < thr).astype(int)
    elif mode == "absolute":
        return (np.abs(x) > thr).astype(int)
    else:
        raise ValueError("mode must be 'positive', 'negative', or 'absolute'")


# =====================================================
# 3) crossing events on raw signal
# =====================================================
def detect_crossings(x, thr, mode="negative", refractory_samples=1):
    """
    Detect threshold crossing events on raw signal.

    Returns
    -------
    events_raw : np.ndarray
        0/1 vector at raw sample rate
    idx_raw : np.ndarray
        indices of detected crossings at raw sample rate
    """
    x = np.asarray(x, dtype=float)

    if mode == "positive":
        raw = (x[1:] > thr) & (x[:-1] <= thr)
    elif mode == "negative":
        raw = (x[1:] < thr) & (x[:-1] >= thr)
    elif mode == "absolute":
        raw = (np.abs(x[1:]) > thr) & (np.abs(x[:-1]) <= thr)
    else:
        raise ValueError("mode must be 'positive', 'negative', or 'absolute'")

    idx_raw = np.where(raw)[0] + 1

    if refractory_samples > 1 and len(idx_raw) > 0:
        keep = [idx_raw[0]]
        for i in idx_raw[1:]:
            if i - keep[-1] >= refractory_samples:
                keep.append(i)
        idx_raw = np.array(keep, dtype=int)

    events_raw = np.zeros_like(x, dtype=int)
    events_raw[idx_raw] = 1

    return events_raw, idx_raw


# =====================================================
# 4) bin raw events into lower-rate counts/rate
# =====================================================
def bin_events_to_rate(idx_raw, n_raw, fs_raw, fs_rate=1000, output="rate"):
    """
    Bin raw event indices into a lower-rate timeline.

    Parameters
    ----------
    idx_raw : array-like
        Event indices at raw sampling rate
    n_raw : int
        Number of raw samples
    fs_raw : float
        Raw sampling rate
    fs_rate : float
        Target rate sampling rate
    output : {"rate", "count"}
        Whether to return counts/bin or Hz

    Returns
    -------
    out : np.ndarray
        Binned counts or rate on the target timeline
    t_rate : np.ndarray
        Time vector for target timeline
    """
    idx_raw = np.asarray(idx_raw, dtype=int)

    duration_s = n_raw / fs_raw
    n_bins = int(np.floor(duration_s * fs_rate))

    if n_bins <= 0:
        raise ValueError("Target rate timeline has zero bins")

    event_times_s = idx_raw / fs_raw
    bin_idx = np.floor(event_times_s * fs_rate).astype(int)
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bins)]

    counts = np.bincount(bin_idx, minlength=n_bins).astype(float)

    if output == "count":
        out = counts
    elif output == "rate":
        out = counts * fs_rate
    else:
        raise ValueError("output must be 'rate' or 'count'")

    t_rate = np.arange(n_bins) / fs_rate
    return out, t_rate


# =====================================================
# 5) optional smoothing on low-rate timeline
# =====================================================
def smooth_rate(rate, fs_rate, method="gaussian", sigma_s=0.01):
    """
    Smooth a lower-rate spike rate.

    Parameters
    ----------
    rate : array-like
        Rate on target timeline
    fs_rate : float
        Sampling rate of rate signal
    method : {"gaussian", "none"}
    sigma_s : float
        Gaussian sigma in seconds

    Returns
    -------
    rate_smooth : np.ndarray
    """
    rate = np.asarray(rate, dtype=float)

    if method == "none":
        return rate
    elif method == "gaussian":
        sigma_samples = max(1e-12, sigma_s * fs_rate)
        return gaussian_filter1d(rate, sigma=sigma_samples, mode="nearest")
    else:
        raise ValueError("method must be 'gaussian' or 'none'")


# =====================================================
# 6) full pipeline: raw thresholding -> low-rate spike rate
# =====================================================
def threshold_and_rate(
    x,
    fs_raw,
    fs_rate=1000,
    k=4.0,
    mode="negative",
    refractory_s=0.001,
    smooth_method="gaussian",
    smooth_sigma_s=0.01,
):
    """
    Detect spikes on raw signal, then produce spike rate on lower-rate timeline.

    Parameters
    ----------
    x : array-like
        Raw 1D signal
    fs_raw : float
        Raw signal sampling rate, e.g. 31000
    fs_rate : float
        Target spike-rate sampling rate, e.g. 1000
    k : float
        Threshold multiplier
    mode : {"positive", "negative", "absolute"}
    refractory_s : float
        Refractory interval in seconds
    smooth_method : {"gaussian", "none"}
    smooth_sigma_s : float
        Smoothing sigma in seconds on the fs_rate timeline

    Returns
    -------
    out : dict with keys
        threshold
        mask_raw
        events_raw
        idx_raw
        event_times_s
        rate
        rate_raw_bins
        t_rate
        fs_raw
        fs_rate
    """
    x = np.asarray(x, dtype=float)

    thr = robust_threshold(x, k=k, mode=mode)
    mask_raw = threshold_mask(x, thr, mode=mode)

    refractory_samples = max(1, int(round(refractory_s * fs_raw)))
    events_raw, idx_raw = detect_crossings(
        x,
        thr,
        mode=mode,
        refractory_samples=refractory_samples
    )

    rate_raw_bins, t_rate = bin_events_to_rate(
        idx_raw=idx_raw,
        n_raw=len(x),
        fs_raw=fs_raw,
        fs_rate=fs_rate,
        output="rate"
    )

    rate = smooth_rate(
        rate_raw_bins,
        fs_rate=fs_rate,
        method=smooth_method,
        sigma_s=smooth_sigma_s
    )

    return {
        "threshold": thr,
        "mask_raw": mask_raw,
        "events_raw": events_raw,
        "idx_raw": idx_raw,
        "event_times_s": idx_raw / fs_raw,
        "rate_raw_bins": rate_raw_bins,
        "rate": rate,
        "t_rate": t_rate,
        "fs_raw": fs_raw,
        "fs_rate": fs_rate,
    }


# =====================================================
# 7) plot raw thresholding result
# =====================================================
def plot_thresholding_raw(t_raw, x, result, channel_name="signal", start_time=None, duration=1.0):
    """
    Plot raw signal with threshold and detected events.
    """
    t_raw = np.asarray(t_raw)
    x = np.asarray(x)

    thr = result["threshold"]
    idx_raw = result["idx_raw"]

    if start_time is None:
        start_time = t_raw[0]

    mask = (t_raw >= start_time) & (t_raw <= start_time + duration)

    plt.figure(figsize=(12, 4))
    plt.plot(t_raw[mask], x[mask], linewidth=1, label=channel_name)
    plt.axhline(thr, linestyle="--", label=f"threshold = {thr:.3f}")

    idx_plot = idx_raw[(t_raw[idx_raw] >= start_time) & (t_raw[idx_raw] <= start_time + duration)]
    if len(idx_plot) > 0:
        plt.scatter(t_raw[idx_plot], x[idx_plot], s=18, label="crossings")

    plt.xlabel("Time (s)")
    plt.ylabel(channel_name)
    plt.title(f"{channel_name} raw threshold crossings")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================
# 8) plot low-rate spike rate
# =====================================================
def plot_spike_rate(t_rate, rate, channel_name="signal", start_time=None, duration=5.0):
    """
    Plot lower-rate spike rate.
    """
    t_rate = np.asarray(t_rate)
    rate = np.asarray(rate)

    if start_time is None:
        start_time = t_rate[0]

    mask = (t_rate >= start_time) & (t_rate <= start_time + duration)

    plt.figure(figsize=(12, 3))
    plt.plot(t_rate[mask], rate[mask], linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.title(f"{channel_name} spike rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
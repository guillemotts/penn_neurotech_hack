import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        Direction of threshold

    Returns
    -------
    thr : float
        Threshold value
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    # robust std approximation
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
# 3) crossing events
# =====================================================
def detect_crossings(x, thr, mode="negative", refractory_samples=1):
    """
    Detect threshold crossing events.

    Returns
    -------
    events : np.ndarray
        0/1 vector, 1 at crossing times
    idx : np.ndarray
        indices of crossings
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

    idx = np.where(raw)[0] + 1

    if refractory_samples > 1 and len(idx) > 0:
        keep = [idx[0]]
        for i in idx[1:]:
            if i - keep[-1] >= refractory_samples:
                keep.append(i)
        idx = np.array(keep, dtype=int)

    events = np.zeros_like(x, dtype=int)
    events[idx] = 1
    return events, idx

# =====================================================
# 4) rolling spike rate
# =====================================================
def spike_rate(events, fs, window_s=0.1):
    """
    Rolling spike/event rate in Hz.

    Parameters
    ----------
    events : array-like
        0/1 event vector
    fs : float
        Sampling rate
    window_s : float
        Window length in seconds

    Returns
    -------
    rate : np.ndarray
        Rolling event rate in Hz
    """
    events = np.asarray(events, dtype=float)
    win = max(1, int(round(window_s * fs)))

    rate = (
        pd.Series(events)
        .rolling(window=win, min_periods=1)
        .sum()
        .to_numpy()
    ) / window_s

    return rate

# =====================================================
# 5) full pipeline helper
# =====================================================
def threshold_and_rate(x, fs, k=4.0, mode="negative", refractory_s=0.001, window_s=0.1):
    """
    Full spike-thresholding pipeline.

    Returns
    -------
    out : dict with keys
        threshold
        mask
        events
        idx
        rate
    """
    thr = robust_threshold(x, k=k, mode=mode)

    if mode == "absolute":
        mask = threshold_mask(x, thr, mode="absolute")
    else:
        mask = threshold_mask(x, thr, mode=mode)

    refractory_samples = max(1, int(round(refractory_s * fs)))
    events, idx = detect_crossings(x, thr, mode=mode, refractory_samples=refractory_samples)
    rate = spike_rate(events, fs, window_s=window_s)

    return {
        "threshold": thr,
        "mask": mask,
        "events": events,
        "idx": idx,
        "rate": rate
    }
    
    
# =====================================================
# 6) plot thresholding result
# =====================================================
def plot_thresholding(t, x, result, channel_name="signal", start_time=None, duration=5):
    """
    Plot signal with threshold and detected events.
    """
    t = np.asarray(t)
    x = np.asarray(x)

    thr = result["threshold"]
    idx = result["idx"]

    if start_time is None:
        start_time = t[0]

    mask = (t >= start_time) & (t <= start_time + duration)

    plt.figure(figsize=(12, 4))
    plt.plot(t[mask], x[mask], label=channel_name, linewidth=1)

    # draw threshold
    if thr >= 0:
        plt.axhline(thr, linestyle="--", label=f"threshold = {thr:.3f}")
    else:
        plt.axhline(thr, linestyle="--", label=f"threshold = {thr:.3f}")

    # overlay event markers
    idx_plot = idx[(t[idx] >= start_time) & (t[idx] <= start_time + duration)]
    if len(idx_plot) > 0:
        plt.scatter(t[idx_plot], x[idx_plot], s=20, label="crossings")

    plt.xlabel("Time (s)")
    plt.ylabel(channel_name)
    plt.title(f"{channel_name} threshold crossings")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
# =====================================================
# 7) plot spike/event rate
# =====================================================
def plot_spike_rate(t, rate, channel_name="signal", start_time=None, duration=5):
    """
    Plot rolling event rate.
    """
    t = np.asarray(t)
    rate = np.asarray(rate)

    if start_time is None:
        start_time = t[0]

    mask = (t >= start_time) & (t <= start_time + duration)

    plt.figure(figsize=(12, 3))
    plt.plot(t[mask], rate[mask], linewidth=1.2)
    plt.xlabel("Time (s)")
    plt.ylabel("Rate (Hz)")
    plt.title(f"{channel_name} event rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
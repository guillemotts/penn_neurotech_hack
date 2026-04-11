import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


def design_bandpass(
    low_hz: float, high_hz: float, fs_hz: float, order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Butterworth bandpass. Returns (b, a). Validate 0 < low < high < fs/2."""
    if low_hz <= 0:
        raise ValueError("low_hz must be > 0")
    if high_hz <= low_hz:
        raise ValueError("high_hz must be > low_hz")
    if high_hz >= fs_hz / 2:
        raise ValueError("high_hz must be < fs_hz / 2")
    nyq = fs_hz / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    return b, a


def apply_bandpass(
    x: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zi: np.ndarray | None = None,
) -> np.ndarray:
    """
    Causal lfilter along last axis. x shape (..., T).
    If zi provided, use as initial state; else lfilter_zi * x[..., :1] per channel.
    Returns filtered array same shape.
    """
    if zi is not None:
        y, _ = lfilter(b, a, x, axis=-1, zi=zi)
        return y

    zi_base = lfilter_zi(b, a)  # shape (max(len(a), len(b)) - 1,)
    # Broadcast zi across leading dimensions
    orig_shape = x.shape
    if x.ndim == 1:
        zi_init = zi_base * x[0]
        y, _ = lfilter(b, a, x, zi=zi_init)
        return y

    # Flatten leading dims
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        zi_init = zi_base * flat[i, 0]
        out[i], _ = lfilter(b, a, flat[i], zi=zi_init)
    return out.reshape(orig_shape)


def car(x: np.ndarray) -> np.ndarray:
    """Common average reference. x shape (C, T). Subtract mean across channel axis per timestep."""
    return x - x.mean(axis=0, keepdims=True)

"""
postprocess.py -- Post-hoc smoother, discretizer, and proxy bitrate.

Pure functions, no side effects.
"""
import numpy as np


def causal_moving_average(Y: np.ndarray, K: int) -> np.ndarray:
    """
    Causal moving average over axis 0.  Y: (T, 10).  K >= 1.
    output[t] = mean(Y[max(0, t-K+1) : t+1]).
    Left edge uses shorter windows (no zero-padding, no future leak).
    """
    if K == 1:
        return Y
    T, C = Y.shape
    cs = np.cumsum(Y, axis=0)  # cs[t] = sum(Y[0..t])
    out = np.empty_like(Y)
    # t < K: window is Y[0..t], length t+1
    for t in range(min(K - 1, T)):
        out[t] = cs[t] / (t + 1)
    # t >= K-1: window is Y[t-K+1..t], length K
    if K - 1 < T:
        out[K - 1:] = (cs[K - 1:] - np.vstack([np.zeros((1, C)), cs[:T - K]])) / K
    return out


def discretize_stick(Y_smooth: np.ndarray, N: int,
                     stick_thresh: float) -> np.ndarray:
    """
    Map (x, y) = Y_smooth[:, 0:2] to one of N equal angular sectors.
    Sector 0 centered on +x.  Returns int array {-1, 0, ..., N-1}.
    -1 = below threshold (no-op).
    """
    x = Y_smooth[:, 0]
    y = Y_smooth[:, 1]
    mag = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)  # atan2(y, x) — correct convention

    # Shift so sector 0 is centered on +x: offset by half-sector
    half = np.pi / N
    shifted = (angle + half) % (2 * np.pi)
    sector = (shifted / (2 * np.pi) * N).astype(int) % N

    sym = np.where(mag >= stick_thresh, sector, -1)
    return sym


def discretize_button(Y_smooth: np.ndarray, button_thresh: float,
                      stick_offset: int) -> np.ndarray:
    """
    Map button channels Y_smooth[:, 4:10] to symbols.
    If max < button_thresh → -1. Else argmax + stick_offset.
    Returns int array {-1, stick_offset, ..., stick_offset+5}.
    """
    buttons = Y_smooth[:, 4:10]
    max_val = buttons.max(axis=1)
    sym = np.where(max_val >= button_thresh,
                   buttons.argmax(axis=1) + stick_offset, -1)
    return sym


def combine(stick_sym: np.ndarray, button_sym: np.ndarray) -> np.ndarray:
    """
    Combine stick and button symbols.
    Precedence: button wins if != -1 (discrete events); else stick (may be -1).
    """
    return np.where(button_sym != -1, button_sym, stick_sym)


def proxy_bitrate(gt_sym: np.ndarray, pred_sym: np.ndarray,
                  N_eff: int, bin_seconds: float,
                  silence_mode: str = "neutral") -> dict:
    """
    Proxy ITR.
    silence_mode:
      "neutral" (b): only bins where pred != -1 count toward S_c/S_i.
      "correct" (a): additionally S_c += count(pred==-1 & gt==-1).

    B = log2(max(N_eff-1, 1)) * max(S_c - S_i, 0) / t
    t = T * bin_seconds (total val duration).
    """
    T = len(gt_sym)
    t = T * bin_seconds

    pred_emit = pred_sym != -1
    S_c = int(((pred_sym == gt_sym) & pred_emit).sum())
    S_i = int(((pred_sym != gt_sym) & pred_emit).sum())
    n_noop = int((~pred_emit).sum())

    if silence_mode == "correct":
        S_c += int(((pred_sym == -1) & (gt_sym == -1)).sum())

    B = float(np.log2(max(N_eff - 1, 1)) * max(S_c - S_i, 0) / t) if t > 0 else 0.0
    accuracy_emitted = S_c / (S_c + S_i) if (S_c + S_i) > 0 else 0.0

    return {
        "S_c": S_c,
        "S_i": S_i,
        "n_noop": n_noop,
        "n_total": T,
        "t_sec": t,
        "N_eff": N_eff,
        "B": B,
        "accuracy_emitted": accuracy_emitted,
    }

import numpy as np
from typing import Optional


def decide(scores: np.ndarray, theta: float) -> Optional[int]:
    """
    Margin-based rejection.
    scores: shape (N,), float, higher = more confident.
    theta: float, margin threshold.
    Returns argmax index iff (top1 - top2) > theta, else None.
    Requires scores 1D, len >= 2, theta >= 0. ValueError otherwise.
    Ties (top1 == top2) return None regardless of theta.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    if len(scores) < 2:
        raise ValueError("scores must have length >= 2")
    if theta < 0:
        raise ValueError("theta must be >= 0")

    idx = np.argsort(scores)
    top1 = scores[idx[-1]]
    top2 = scores[idx[-2]]

    if top1 == top2:
        return None
    if (top1 - top2) > theta:
        return int(idx[-1])
    return None

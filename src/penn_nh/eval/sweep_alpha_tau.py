import numpy as np

from penn_nh.decoder.stage2_proto import combine
from penn_nh.eval.sweep_theta import sweep_theta


def sweep_alpha_tau(
    logits_val: np.ndarray,
    cos_val: np.ndarray,
    labels_val: np.ndarray,
    alpha_grid: np.ndarray,
    tau_grid: np.ndarray,
    theta_grid: np.ndarray,
    t: float = 60.0,
) -> dict:
    """
    For each (alpha, tau) cell: compute combined scores via combine(), run inner
    sweep_theta, record best B and best theta.
    """
    logits_val = np.asarray(logits_val, dtype=float)
    cos_val = np.asarray(cos_val, dtype=float)
    labels_val = np.asarray(labels_val, dtype=int)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    theta_grid = np.asarray(theta_grid, dtype=float)

    if logits_val.shape != cos_val.shape:
        raise ValueError("logits_val and cos_val must have the same shape")
    if logits_val.ndim != 2:
        raise ValueError("logits_val must be 2D (M, N)")
    if labels_val.ndim != 1 or labels_val.shape[0] != logits_val.shape[0]:
        raise ValueError("labels_val length must match logits_val rows")
    if alpha_grid.ndim != 1:
        raise ValueError("alpha_grid must be 1D")
    if tau_grid.ndim != 1:
        raise ValueError("tau_grid must be 1D")
    if np.any(alpha_grid < 0) or np.any(alpha_grid > 1):
        raise ValueError("alpha_grid values must be in [0, 1]")
    if np.any(tau_grid <= 0):
        raise ValueError("tau_grid values must be > 0")

    na, nt = len(alpha_grid), len(tau_grid)
    B = np.zeros((na, nt), dtype=float)
    theta_best = np.zeros((na, nt), dtype=float)

    for ai, alpha in enumerate(alpha_grid):
        for ti, tau in enumerate(tau_grid):
            scores = combine(logits_val, cos_val, alpha, tau)
            sw = sweep_theta(scores, labels_val, theta_grid, t)
            best_idx = int(np.argmax(sw["B"]))
            B[ai, ti] = sw["B"][best_idx]
            theta_best[ai, ti] = sw["argmax_theta"]

    flat_idx = int(np.argmax(B))
    ai_best, ti_best = np.unravel_index(flat_idx, (na, nt))

    return {
        "alpha": alpha_grid,
        "tau": tau_grid,
        "B": B,
        "theta_best": theta_best,
        "argmax_alpha": float(alpha_grid[ai_best]),
        "argmax_tau": float(tau_grid[ti_best]),
        "argmax_theta": float(theta_best[ai_best, ti_best]),
        "B_max": float(B[ai_best, ti_best]),
    }

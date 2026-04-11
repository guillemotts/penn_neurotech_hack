import numpy as np

from penn_nh.decoder.stage3_reject import decide
from penn_nh.eval.bitrate import shenoy_b_from_confusion


def sweep_theta(
    scores: np.ndarray,
    labels: np.ndarray,
    theta_grid: np.ndarray,
    t: float = 60.0,
) -> dict:
    """
    For each theta, apply decide() to each row of scores, accumulate a confusion
    matrix over accepted trials, compute B via shenoy_b_from_confusion.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    theta_grid = np.asarray(theta_grid, dtype=float)

    if scores.ndim != 2:
        raise ValueError("scores must be 2D (M, N)")
    if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
        raise ValueError("labels must be 1D with length matching scores rows")
    if theta_grid.ndim != 1:
        raise ValueError("theta_grid must be 1D")
    if len(theta_grid) > 1 and np.any(np.diff(theta_grid) < 0):
        raise ValueError("theta_grid must be monotonic nondecreasing")
    if np.any(theta_grid < 0):
        raise ValueError("theta_grid values must be >= 0")

    M, N = scores.shape
    n_thetas = len(theta_grid)

    B_array = np.zeros(n_thetas, dtype=float)
    n_accepted = np.zeros(n_thetas, dtype=int)
    S_c_array = np.zeros(n_thetas, dtype=int)
    S_i_array = np.zeros(n_thetas, dtype=int)

    for ti, theta in enumerate(theta_grid):
        C = np.zeros((N, N), dtype=int)
        for row in range(M):
            result = decide(scores[row], theta)
            if result is not None:
                C[labels[row], result] += 1

        accepted = int(C.sum())
        n_accepted[ti] = accepted
        s_c = int(np.trace(C))
        s_i = accepted - s_c
        S_c_array[ti] = s_c
        S_i_array[ti] = s_i

        if accepted > 0 and N >= 3:
            B_array[ti] = shenoy_b_from_confusion(C, t)
        else:
            B_array[ti] = 0.0

    best = int(np.argmax(B_array))

    return {
        "theta": theta_grid,
        "B": B_array,
        "argmax_theta": float(theta_grid[best]),
        "n_accepted": n_accepted,
        "S_c": S_c_array,
        "S_i": S_i_array,
    }

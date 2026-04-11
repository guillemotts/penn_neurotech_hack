import numpy as np
from typing import Protocol, Optional

from penn_nh.collect.labeler import label_batch
from penn_nh.eval.sweep_theta import sweep_theta


class DecoderFactory(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, n: int): ...


def sweep_n(
    X_train: np.ndarray,
    stick_xy_train: np.ndarray,
    X_val: np.ndarray,
    stick_xy_val: np.ndarray,
    factory: DecoderFactory,
    n_grid: np.ndarray,
    deadzone: float = 0.5,
    theta_grid: Optional[np.ndarray] = None,
    t: float = 60.0,
) -> dict:
    """
    Sweep over alphabet sizes N, relabeling, fitting, and evaluating for each.
    """
    n_grid = np.asarray(n_grid, dtype=int)
    if n_grid.ndim != 1:
        raise ValueError("n_grid must be 1D")
    if np.any(n_grid < 3):
        raise ValueError("All N in n_grid must be >= 3")

    X_train = np.asarray(X_train)
    stick_xy_train = np.asarray(stick_xy_train, dtype=float)
    X_val = np.asarray(X_val)
    stick_xy_val = np.asarray(stick_xy_val, dtype=float)

    if X_train.shape[0] != stick_xy_train.shape[0]:
        raise ValueError("X_train and stick_xy_train must have same number of rows")
    if X_val.shape[0] != stick_xy_val.shape[0]:
        raise ValueError("X_val and stick_xy_val must have same number of rows")

    if theta_grid is None:
        theta_grid = np.linspace(0.0, 0.5, 26)

    B_out = np.zeros(len(n_grid), dtype=float)
    theta_out = np.zeros(len(n_grid), dtype=float)
    n_train_kept = np.zeros(len(n_grid), dtype=int)
    n_val_kept = np.zeros(len(n_grid), dtype=int)

    for idx, N in enumerate(n_grid):
        N = int(N)

        y_train, mask_train = label_batch(stick_xy_train, N, deadzone)
        y_val, mask_val = label_batch(stick_xy_val, N, deadzone)

        if mask_train.sum() == 0:
            raise ValueError(f"N={N}: zero kept train samples")
        if mask_val.sum() == 0:
            raise ValueError(f"N={N}: zero kept val samples")

        X_tr = X_train[mask_train]
        X_v = X_val[mask_val]

        n_train_kept[idx] = int(mask_train.sum())
        n_val_kept[idx] = int(mask_val.sum())

        fitted = factory.fit(X_tr, y_train, N)
        scores_val = fitted.score(X_v)

        result = sweep_theta(scores_val, y_val, theta_grid, t)
        B_out[idx] = result["B"][np.argmax(result["B"])]
        theta_out[idx] = result["argmax_theta"]

    best_idx = int(np.argmax(B_out))

    return {
        "N": n_grid,
        "B": B_out,
        "theta": theta_out,
        "n_train_kept": n_train_kept,
        "n_val_kept": n_val_kept,
        "argmax_N": int(n_grid[best_idx]),
    }

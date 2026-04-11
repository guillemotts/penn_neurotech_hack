import numpy as np


def shenoy_b(s_c: int, s_i: int, n: int, t: float = 60.0) -> float:
    """
    Shenoy achieved bit rate.
    B = log2(N - 1) * max(S_c - S_i, 0) / t
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    if t <= 0:
        raise ValueError("t must be > 0")
    if s_c < 0:
        raise ValueError("s_c must be >= 0")
    if s_i < 0:
        raise ValueError("s_i must be >= 0")

    diff = s_c - s_i
    if diff <= 0:
        return 0.0
    return np.log2(n - 1) * diff / t


def _validate_confusion(C: np.ndarray) -> np.ndarray:
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")
    C = np.asarray(C, dtype=int)
    if C.shape[0] < 3:
        raise ValueError("N must be >= 3")
    if np.any(C < 0):
        raise ValueError("C must have no negative entries")
    return C


def shenoy_b_from_confusion(C: np.ndarray, t: float = 60.0) -> float:
    """
    Compute Shenoy bit rate from an (N, N) confusion matrix.
    """
    C = _validate_confusion(C)
    s_c = int(np.trace(C))
    s_i = int(C.sum()) - s_c
    n = C.shape[0]
    return shenoy_b(s_c, s_i, n, t)


def _mi_plugin(C: np.ndarray) -> float:
    """Plug-in estimate of mutual information in bits."""
    N_total = C.sum()
    if N_total == 0:
        return 0.0
    p_joint = C / N_total
    p_i = C.sum(axis=1) / N_total
    p_j = C.sum(axis=0) / N_total
    mi = 0.0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                mi += p_joint[i, j] * np.log2(p_joint[i, j] / (p_i[i] * p_j[j]))
    return float(mi)


def mutual_information_pt(C: np.ndarray) -> float:
    """
    Empirical mutual information in bits with Panzeri-Treves bias correction.
    """
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix")
    C = np.asarray(C, dtype=int)
    N_total = C.sum()
    if N_total == 0:
        return 0.0

    I_plugin = _mi_plugin(C)

    p_i = C.sum(axis=1)
    p_j = C.sum(axis=0)

    R = int(np.count_nonzero(C))

    bias = (R - 1) / (2.0 * N_total * np.log(2.0))

    result = I_plugin - bias
    return max(result, 0.0)


def per_symbol_breakeven(C: np.ndarray, t_per_trial: float) -> np.ndarray:
    """
    Per-class signed bit-rate contribution.
    """
    if t_per_trial <= 0:
        raise ValueError("t_per_trial must be > 0")
    C = np.asarray(C)
    n = C.shape[0]
    result = np.zeros(n, dtype=float)
    log_factor = np.log2(n - 1) if n > 1 else 0.0
    for i in range(n):
        n_i = C[i, :].sum()
        if n_i == 0:
            continue
        s_c_i = C[i, i]
        s_i_i = n_i - s_c_i
        result[i] = log_factor * (s_c_i - s_i_i) / (n_i * t_per_trial)
    return result

import numpy as np
from typing import Protocol

from penn_nh.collect.labeler import label_batch
from penn_nh.eval.sweep_theta import sweep_theta
from penn_nh.features.stack import FeatureConfig, FeatureStack


class DecoderFactory(Protocol):
    def fit(self, X, y, n): ...


def sweep_window(
    raw_train: np.ndarray,
    onset_idx_train: np.ndarray,
    stick_xy_train: np.ndarray,
    raw_val: np.ndarray,
    onset_idx_val: np.ndarray,
    stick_xy_val: np.ndarray,
    factory: DecoderFactory,
    n: int,
    fs_hz: float,
    w0_ms_grid: np.ndarray,
    w1_ms_grid: np.ndarray,
    feature_cfg: FeatureConfig | None = None,
    theta_grid: np.ndarray | None = None,
    deadzone: float = 0.5,
    t: float = 60.0,
) -> dict:
    raw_train = np.asarray(raw_train)
    raw_val = np.asarray(raw_val)
    if raw_train.ndim != 3 or raw_val.ndim != 3:
        raise ValueError("raw_train and raw_val must be 3D (M, C, T)")
    onset_idx_train = np.asarray(onset_idx_train, dtype=int)
    onset_idx_val = np.asarray(onset_idx_val, dtype=int)
    stick_xy_train = np.asarray(stick_xy_train, dtype=float)
    stick_xy_val = np.asarray(stick_xy_val, dtype=float)
    w0_ms_grid = np.asarray(w0_ms_grid, dtype=float)
    w1_ms_grid = np.asarray(w1_ms_grid, dtype=float)

    if w0_ms_grid.ndim != 1 or w1_ms_grid.ndim != 1:
        raise ValueError("w0_ms_grid and w1_ms_grid must be 1D")
    if np.any(w0_ms_grid < 0):
        raise ValueError("w0_ms_grid values must be >= 0")
    if np.any(w1_ms_grid <= 0):
        raise ValueError("w1_ms_grid values must be > 0")

    if theta_grid is None:
        theta_grid = np.linspace(0.0, 0.5, 26)
    if feature_cfg is None:
        feature_cfg = FeatureConfig(fs_hz=fs_hz)

    T_full_train = raw_train.shape[2]
    T_full_val = raw_val.shape[2]

    w0_list, w1_list, B_list, theta_list = [], [], [], []

    for w0_ms in w0_ms_grid:
        for w1_ms in w1_ms_grid:
            if w1_ms <= w0_ms:
                continue

            w0_samp = int(round(w0_ms * 1e-3 * fs_hz))
            w1_samp = int(round(w1_ms * 1e-3 * fs_hz))

            # Slice train
            keep_tr = []
            slices_tr = []
            for i in range(len(raw_train)):
                s = onset_idx_train[i] + w0_samp
                e = onset_idx_train[i] + w1_samp
                if s >= 0 and e <= T_full_train:
                    keep_tr.append(i)
                    slices_tr.append(raw_train[i, :, s:e])
            if len(slices_tr) == 0:
                continue

            # Slice val
            keep_va = []
            slices_va = []
            for i in range(len(raw_val)):
                s = onset_idx_val[i] + w0_samp
                e = onset_idx_val[i] + w1_samp
                if s >= 0 and e <= T_full_val:
                    keep_va.append(i)
                    slices_va.append(raw_val[i, :, s:e])
            if len(slices_va) == 0:
                continue

            X_tr_raw = np.stack(slices_tr)
            X_va_raw = np.stack(slices_va)

            # Featurize
            stack = FeatureStack(feature_cfg)
            tr_feat = [stack.transform(X_tr_raw[i]) for i in range(len(X_tr_raw))]
            stack.fit_zscore(tr_feat)
            X_tr = np.stack([stack.transform(X_tr_raw[i]) for i in range(len(X_tr_raw))]).astype(np.float32)
            X_va = np.stack([stack.transform(X_va_raw[i]) for i in range(len(X_va_raw))]).astype(np.float32)

            # Relabel
            y_tr, mask_tr = label_batch(stick_xy_train[keep_tr], n, deadzone)
            y_va, mask_va = label_batch(stick_xy_val[keep_va], n, deadzone)
            if len(y_tr) == 0 or len(y_va) == 0:
                continue

            X_tr = X_tr[mask_tr]
            X_va = X_va[mask_va]

            # Check all classes present
            if len(np.unique(y_tr)) < n or len(np.unique(y_va)) < n:
                continue

            fitted = factory.fit(X_tr, y_tr, n)
            scores = fitted.score(X_va)
            sw = sweep_theta(scores, y_va, theta_grid, t)

            w0_list.append(w0_ms)
            w1_list.append(w1_ms)
            B_list.append(float(sw["B"][np.argmax(sw["B"])]))
            theta_list.append(float(sw["argmax_theta"]))

    w0_arr = np.array(w0_list, dtype=float)
    w1_arr = np.array(w1_list, dtype=float)
    B_arr = np.array(B_list, dtype=float)
    theta_arr = np.array(theta_list, dtype=float)

    if len(B_arr) > 0:
        argmax_idx = int(np.argmax(B_arr))
        B_max = float(B_arr[argmax_idx])
    else:
        argmax_idx = 0
        B_max = 0.0

    return {
        "w0_ms": w0_arr,
        "w1_ms": w1_arr,
        "B": B_arr,
        "theta": theta_arr,
        "argmax_idx": argmax_idx,
        "B_max": B_max,
    }

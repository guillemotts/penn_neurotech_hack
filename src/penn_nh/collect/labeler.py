import numpy as np
from typing import Optional


def label_from_xy(xy: np.ndarray, n_sectors: int, deadzone: float = 0.5) -> Optional[int]:
    """
    Single trial, xy shape (2,).
    r = sqrt(x^2 + y^2). If r < deadzone: return None.
    theta = atan2(y, x). Sector:
        s = floor((theta + pi/n_sectors + 2*pi) / (2*pi/n_sectors)) mod n_sectors
    Sector 0 centered on +x, counter-clockwise. Requires n_sectors >= 3.
    """
    if n_sectors < 3:
        raise ValueError("n_sectors must be >= 3")
    xy = np.asarray(xy, dtype=float)
    x, y = xy[0], xy[1]
    r = np.sqrt(x * x + y * y)
    if r < deadzone:
        return None
    theta = np.arctan2(y, x)
    sector_width = 2.0 * np.pi / n_sectors
    s = int(np.floor((theta + np.pi / n_sectors + 2.0 * np.pi) / sector_width)) % n_sectors
    return s


def label_batch(
    xy: np.ndarray, n_sectors: int, deadzone: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    xy shape (M, 2). Returns (labels, kept_mask):
      labels: (K,) int, K = kept count
      kept_mask: (M,) bool
    """
    if n_sectors < 3:
        raise ValueError("n_sectors must be >= 3")
    xy = np.asarray(xy, dtype=float)
    m = xy.shape[0]
    kept_mask = np.zeros(m, dtype=bool)
    labels_list = []
    for i in range(m):
        lbl = label_from_xy(xy[i], n_sectors, deadzone)
        if lbl is not None:
            kept_mask[i] = True
            labels_list.append(lbl)
    return np.array(labels_list, dtype=int), kept_mask

import numpy as np
import pytest

from penn_nh.collect.labeler import label_batch, label_from_xy


def test_eight_unit_vectors():
    for k in range(8):
        angle = k * np.pi / 4
        xy = np.array([np.cos(angle), np.sin(angle)])
        assert label_from_xy(xy, n_sectors=8) == k


def test_deadzone_drops():
    for angle in [0, np.pi / 4, np.pi / 2, np.pi]:
        xy = 0.3 * np.array([np.cos(angle), np.sin(angle)])
        assert label_from_xy(xy, n_sectors=8) is None


def test_four_cardinal():
    cardinals = [
        (np.array([1.0, 0.0]), 0),
        (np.array([0.0, 1.0]), 1),
        (np.array([-1.0, 0.0]), 2),
        (np.array([0.0, -1.0]), 3),
    ]
    for xy, expected in cardinals:
        assert label_from_xy(xy, n_sectors=4) == expected


def test_label_batch_mixed():
    rng = np.random.default_rng(0)
    # 10 inside deadzone
    inside = 0.3 * rng.standard_normal((10, 2))
    inside = inside / np.linalg.norm(inside, axis=1, keepdims=True) * 0.3
    # 10 outside deadzone
    angles = rng.uniform(0, 2 * np.pi, 10)
    outside = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.8
    xy = np.vstack([inside, outside])
    labels, kept_mask = label_batch(xy, n_sectors=8)
    assert kept_mask.sum() == 10
    assert len(labels) == 10


def test_n_sectors_2_raises():
    with pytest.raises(ValueError):
        label_from_xy(np.array([1.0, 0.0]), n_sectors=2)

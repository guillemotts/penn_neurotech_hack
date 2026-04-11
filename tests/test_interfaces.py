import numpy as np
import pytest
from numpy import log2
from scipy.stats import chisquare

from penn_nh.eval.bitrate import shenoy_b
from penn_nh.game.interface import Game, UnpacedTargetHit


def test_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        Game()


def test_n_validation():
    with pytest.raises(ValueError):
        UnpacedTargetHit(n=2)


def test_queue_depth_default():
    game = UnpacedTargetHit(n=8, rng=np.random.default_rng(seed=0))
    assert game.queue_depth == 1
    assert len(game.upcoming_targets()) == 1


def test_queue_depth_two():
    game = UnpacedTargetHit(n=8, queue_depth=2, rng=np.random.default_rng(seed=0))
    assert len(game.upcoming_targets()) == 2


def test_submit_correct():
    game = UnpacedTargetHit(n=8, rng=np.random.default_rng(seed=0))
    target = game.current_target()
    result = game.submit(target, t=1.0)
    assert result["outcome"] == "correct"
    assert game.S_c == 1
    assert game.S_i == 0


def test_submit_incorrect():
    game = UnpacedTargetHit(n=8, rng=np.random.default_rng(seed=0))
    target = game.current_target()
    wrong = (target + 1) % game.N
    result = game.submit(wrong, t=1.0)
    assert result["outcome"] == "incorrect"
    assert game.S_c == 0
    assert game.S_i == 1


def test_submit_advances_queue():
    rng = np.random.default_rng(seed=0)
    game = UnpacedTargetHit(n=8, rng=rng)
    before = game.current_target()
    game.submit(before, t=1.0)
    after = game.current_target()
    # With seed=0 and N=8, consecutive draws differ
    assert after != before


def test_rejection_no_counter_update():
    game = UnpacedTargetHit(n=8, rng=np.random.default_rng(seed=0))
    before = game.current_target()
    game.on_rejection(t=1.0)
    assert game.S_c == 0
    assert game.S_i == 0
    assert game.rejections == 1
    # Queue advanced
    after = game.current_target()
    assert after != before


def test_bitrate_empty():
    game = UnpacedTargetHit(n=8, rng=np.random.default_rng(seed=0))
    assert game.bitrate() == 0.0


def test_bitrate_matches_shenoy():
    game = UnpacedTargetHit(n=8, rng=np.random.default_rng(seed=0))
    for _ in range(30):
        target = game.current_target()
        game.submit(target, t=0.0)
    for _ in range(5):
        target = game.current_target()
        wrong = (target + 1) % game.N
        game.submit(wrong, t=0.0)
    assert game.S_c == 30
    assert game.S_i == 5
    expected = shenoy_b(30, 5, 8, 60.0)
    assert game.bitrate() == pytest.approx(expected)
    assert game.bitrate() == pytest.approx(log2(7) * 25 / 60)


def test_iid_uniform():
    n = 8
    rng = np.random.default_rng(seed=0)
    game = UnpacedTargetHit(n=n, rng=rng)
    counts = np.zeros(n, dtype=int)
    for i in range(10000):
        target = game.current_target()
        counts[target] += 1
        game.submit(target, t=float(i))
    stat, p = chisquare(counts)
    assert p > 0.01

import numpy as np
import pytest

from penn_nh.game.interface import UnpacedTargetHit
from penn_nh.integration.mock_decoder import mock_decoder_stream


def test_perfect_decoder():
    rng = np.random.default_rng(0)
    game = UnpacedTargetHit(n=4, session_seconds=1000.0, rng=rng)
    confusion = np.eye(4)
    stream = mock_decoder_stream(
        game, confusion, mean_interval_s=1.0, rng=np.random.default_rng(1)
    )
    count = 0
    for symbol, conf, t in stream:
        if symbol is not None:
            game.submit(symbol, t)
            count += 1
        if count >= 100:
            break
    assert game.S_c == 100
    assert game.S_i == 0


def test_rejection_rate_zero():
    rng = np.random.default_rng(0)
    game = UnpacedTargetHit(n=4, session_seconds=100.0, rng=rng)
    confusion = np.eye(4)
    stream = mock_decoder_stream(
        game, confusion, mean_interval_s=1.0, rejection_rate=0.0,
        rng=np.random.default_rng(1),
    )
    for symbol, conf, t in stream:
        assert symbol is not None
        game.submit(symbol, t)


def test_rejection_rate_one():
    rng = np.random.default_rng(0)
    game = UnpacedTargetHit(n=4, session_seconds=100.0, rng=rng)
    confusion = np.eye(4)
    stream = mock_decoder_stream(
        game, confusion, mean_interval_s=1.0, rejection_rate=1.0,
        rng=np.random.default_rng(1),
    )
    for symbol, conf, t in stream:
        assert symbol is None
    assert game.S_c == 0
    assert game.S_i == 0


def test_validation():
    rng = np.random.default_rng(0)
    game = UnpacedTargetHit(n=4, session_seconds=60.0, rng=rng)

    # Non-square
    with pytest.raises(ValueError):
        list(mock_decoder_stream(game, np.ones((3, 4)), mean_interval_s=1.0))

    # Rows don't sum to 1
    with pytest.raises(ValueError):
        list(mock_decoder_stream(game, np.ones((4, 4)), mean_interval_s=1.0))

    # rejection_rate > 1
    with pytest.raises(ValueError):
        list(mock_decoder_stream(
            game, np.eye(4), mean_interval_s=1.0, rejection_rate=1.5
        ))


def test_stops_at_session_end():
    rng = np.random.default_rng(0)
    session = 10.0
    game = UnpacedTargetHit(n=4, session_seconds=session, rng=rng)
    confusion = np.eye(4)
    stream = mock_decoder_stream(
        game, confusion, mean_interval_s=0.1, rng=np.random.default_rng(1)
    )
    max_t = 0.0
    for symbol, conf, t in stream:
        max_t = t
        if symbol is not None:
            game.submit(symbol, t)
    assert max_t <= session

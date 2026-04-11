import numpy as np
import pytest

from penn_nh.game.interface import UnpacedTargetHit
from penn_nh.game.session import Session
from penn_nh.integration.mock_decoder import mock_decoder_stream


def _make_session(
    n=4,
    confusion=None,
    mean_interval=0.1,
    rejection_rate=0.0,
    session_seconds=60.0,
    seed_game=0,
    seed_decoder=1,
    time_source="iterator",
    bitrate_tick_hz=4.0,
    return_history=False,
    renderer=None,
):
    game = UnpacedTargetHit(
        n=n, session_seconds=session_seconds, rng=np.random.default_rng(seed_game)
    )
    if confusion is None:
        confusion = np.eye(n)
    stream = mock_decoder_stream(
        game,
        confusion,
        mean_interval_s=mean_interval,
        rejection_rate=rejection_rate,
        rng=np.random.default_rng(seed_decoder),
    )
    sess = Session(
        game=game,
        decoder_stream=stream,
        renderer=renderer,
        time_source=time_source,
        bitrate_tick_hz=bitrate_tick_hz,
        return_history=return_history,
    )
    return sess, game


def test_perfect_run_iterator_time():
    sess, game = _make_session(
        n=4, confusion=np.eye(4), mean_interval=0.1, rejection_rate=0.0
    )
    report = sess.run()
    assert report["S_c"] > 0
    assert report["S_i"] == 0
    assert report["B"] > 0
    assert report["duration"] <= 60.0


def test_session_respects_hard_stop():
    sess, game = _make_session(
        n=4, mean_interval=0.01, session_seconds=1.0
    )
    report = sess.run()
    assert report["duration"] <= 1.0
    assert report["S_c"] + report["S_i"] <= 200


def test_noisy_confusion_produces_errors():
    n = 4
    confusion = np.full((n, n), 0.1)
    np.fill_diagonal(confusion, 0.7)
    sess, game = _make_session(n=n, confusion=confusion)
    report = sess.run()
    assert report["S_c"] > 0
    assert report["S_i"] > 0


def test_rejections_counted():
    sess, game = _make_session(rejection_rate=0.3)
    report = sess.run()
    assert report["rejections"] > 0
    total_events = report["S_c"] + report["S_i"] + report["rejections"]
    assert total_events == len(game.history)


def test_return_history_flag():
    sess_on, _ = _make_session(return_history=True)
    report_on = sess_on.run()
    assert "history" in report_on
    assert len(report_on["history"]) > 0

    sess_off, _ = _make_session(return_history=False, seed_game=2, seed_decoder=3)
    report_off = sess_off.run()
    assert "history" not in report_off


def test_null_renderer_default(capsys):
    sess, _ = _make_session(renderer=None)
    sess.run()
    captured = capsys.readouterr().out
    assert captured == ""


def test_bitrate_tick_validation():
    game = UnpacedTargetHit(n=4, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        Session(game=game, decoder_stream=iter([]), bitrate_tick_hz=0)


def test_time_source_validation():
    game = UnpacedTargetHit(n=4, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        Session(game=game, decoder_stream=iter([]), time_source="bogus")


def test_empty_stream_returns_zero():
    game = UnpacedTargetHit(n=4, rng=np.random.default_rng(0))
    sess = Session(
        game=game,
        decoder_stream=iter([]),
        time_source="iterator",
    )
    report = sess.run()
    assert report["S_c"] == 0
    assert report["S_i"] == 0
    assert report["B"] == 0.0
    assert report["duration"] == 0.0

"""Tests for scripts/collect_training.py — no hardware, no display."""
import sys
import types
from unittest import mock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock synapse and pygame so the module can be imported without hardware
# ---------------------------------------------------------------------------

def _build_mock_synapse():
    """Return a fake 'synapse' package with the minimum API surface."""
    synapse = types.ModuleType("synapse")
    synapse.Device = mock.MagicMock
    synapse.Config = mock.MagicMock
    synapse.Channel = mock.MagicMock
    synapse.BroadbandSource = mock.MagicMock
    synapse.SignalConfig = mock.MagicMock
    synapse.ElectrodeConfig = mock.MagicMock

    client = types.ModuleType("synapse.client")
    taps = types.ModuleType("synapse.client.taps")
    taps.Tap = mock.MagicMock
    client.taps = taps

    api = types.ModuleType("synapse.api")
    datatype_pb2 = types.ModuleType("synapse.api.datatype_pb2")
    datatype_pb2.BroadbandFrame = mock.MagicMock
    api.datatype_pb2 = datatype_pb2

    synapse.client = client
    synapse.api = api
    return {
        "synapse": synapse,
        "synapse.client": client,
        "synapse.client.taps": taps,
        "synapse.api": api,
        "synapse.api.datatype_pb2": datatype_pb2,
    }


def _build_mock_pygame():
    """Return a minimal mock pygame so the import succeeds."""
    pg = types.ModuleType("pygame")
    pg.init = mock.MagicMock()
    pg.quit = mock.MagicMock()
    pg.FULLSCREEN = 0x80000000
    pg.HIDDEN = 0x00000008
    pg.QUIT = 256
    pg.KEYDOWN = 768
    pg.K_ESCAPE = 27

    display = mock.MagicMock()
    pg.display = display

    font_mod = mock.MagicMock()
    pg.font = font_mod

    draw = mock.MagicMock()
    pg.draw = draw

    pg.event = mock.MagicMock()
    pg.time = mock.MagicMock()
    return {"pygame": pg}


# Patch modules before importing collect_training
_syn_mocks = _build_mock_synapse()
_pg_mocks = _build_mock_pygame()

_all_mocks = {**_syn_mocks, **_pg_mocks}
_originals = {k: sys.modules.get(k) for k in _all_mocks}

for k, v in _all_mocks.items():
    sys.modules[k] = v

from scripts.collect_training import (
    SYMBOLS,
    N_SYMBOLS,
    CaptureState,
    tap_reader,
    build_config,
    render_cue,
    main,
)

# Restore originals after import so other tests are not affected
for k, v in _originals.items():
    if v is None:
        sys.modules.pop(k, None)
    else:
        sys.modules[k] = v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_symbols_n22():
    """SYMBOLS has exactly 22 entries: 8 ljoy + 8 rjoy + 6 buttons."""
    assert len(SYMBOLS) == 22
    assert N_SYMBOLS == 22
    ljoy = [s for s in SYMBOLS if s[0] == "ljoy"]
    rjoy = [s for s in SYMBOLS if s[0] == "rjoy"]
    buttons = [s for s in SYMBOLS if s[0] == "button"]
    assert len(ljoy) == 8
    assert len(rjoy) == 8
    assert len(buttons) == 6


def test_cue_sequence_reproducible():
    """default_rng(10).integers(0,22,100) is deterministic across runs."""
    a = np.random.default_rng(10).integers(0, 22, 100)
    b = np.random.default_rng(10).integers(0, 22, 100)
    np.testing.assert_array_equal(a, b)


def test_build_config_shape():
    """build_config returns a Config with one node added."""
    syn = pytest.importorskip("synapse")
    cfg = build_config(108, 64, 30000)
    # cfg is a syn.Config mock; add_node was called once
    cfg.add_node.assert_called_once()


def test_render_cue_does_not_crash():
    """render_cue runs for ljoy, rjoy, button without raising."""
    pytest.importorskip("pygame")
    # render_cue's module-level pygame is the mock from import time,
    # so we use a fully mocked screen/font to stay consistent.
    screen = mock.MagicMock()
    screen.get_size.return_value = (800, 600)

    font = mock.MagicMock()
    surf = mock.MagicMock()
    surf.get_width.return_value = 100
    surf.get_height.return_value = 40
    font.render.return_value = surf

    render_cue(screen, font, "ljoy", 3, 0, 10)
    render_cue(screen, font, "rjoy", 5, 1, 10)
    render_cue(screen, font, "button", "A", 2, 10)


def test_tap_reader_parses_frames():
    """tap_reader consumes 3 fake frames and appends them to state.frames."""
    n_channels = 4
    samples_per_frame = 8
    frame_size = n_channels * samples_per_frame

    # Build 3 fake raw bytes via a mock BroadbandFrame
    fake_frames = []
    for i in range(3):
        data = np.arange(frame_size, dtype=np.float32) + i * 100
        fake_frames.append((1000 + i, list(data)))

    call_count = 0

    def fake_stream(timeout_ms=100):
        nonlocal call_count
        for ts_ns, fdata in fake_frames:
            call_count += 1
            yield f"raw_{call_count}".encode()
        # After yielding all, stop
        return

    # Monkey-patch BroadbandFrame
    frame_iter = iter(fake_frames)

    class FakeBBFrame:
        def __init__(self):
            self.timestamp_ns = 0
            self.frame_data = []

        def ParseFromString(self, raw):
            ts_ns, fdata = next(frame_iter)
            self.timestamp_ns = ts_ns
            self.frame_data = fdata

    mock_tap = mock.MagicMock()
    mock_tap.stream = fake_stream

    state = CaptureState()

    # Patch BroadbandFrame in the module's namespace
    import scripts.collect_training as ct_mod
    orig_bb = ct_mod.BroadbandFrame
    ct_mod.BroadbandFrame = FakeBBFrame
    try:
        tap_reader(mock_tap, state, n_channels)
    finally:
        ct_mod.BroadbandFrame = orig_bb

    assert len(state.frames) == 3
    for ts_ns, chunk in state.frames:
        assert chunk.shape == (n_channels, samples_per_frame)
    assert state.t0_frame_ns == 1000
    assert state.dropped == 0


def test_dry_run_end_to_end(tmp_path):
    """--dry-run skips hardware, writes .npz with synthetic neural data."""
    import glob as glob_mod
    import scripts.collect_training as ct_mod

    # Grab the mock pygame that ct_mod is bound to
    pg = ct_mod.pygame

    # Mock screen / font / clock returned by pygame calls
    mock_screen = mock.MagicMock()
    mock_screen.get_size.return_value = (800, 600)
    mock_screen.get_width.return_value = 800
    mock_screen.get_height.return_value = 600
    pg.display.set_mode.return_value = mock_screen
    pg.display.flip = mock.MagicMock()

    mock_font = mock.MagicMock()
    mock_surf = mock.MagicMock()
    mock_surf.get_width.return_value = 100
    mock_surf.get_height.return_value = 40
    mock_font.render.return_value = mock_surf
    pg.font.SysFont.return_value = mock_font

    mock_clock = mock.MagicMock()
    pg.time.Clock.return_value = mock_clock

    # pygame.event.get returns no events (no ESC)
    pg.event.get.return_value = []

    # No-op time.sleep so the test is instant
    with mock.patch.object(ct_mod.time, "sleep"):
        ret = main([
            "--dry-run", "--player", "test", "--seed", "10",
            "--n-trials", "5", "--isi", "0.1",
            "--out-dir", str(tmp_path),
        ])

    assert ret == 0

    # Find the saved .npz
    npz_files = list(tmp_path.joinpath("test").glob("hard_*.npz"))
    assert len(npz_files) == 1
    data = np.load(npz_files[0], allow_pickle=True)
    assert data["neural"].ndim == 2

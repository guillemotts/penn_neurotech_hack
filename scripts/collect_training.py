import argparse, sys, time, threading, queue
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pygame
import synapse as syn
from synapse.client.taps import Tap
from synapse.api.datatype_pb2 import BroadbandFrame

# Alphabet for N=22 hard mode: 8 LJoy sectors + 8 RJoy sectors + 6 buttons
SYMBOLS = (
    [("ljoy", k) for k in range(8)]
    + [("rjoy", k) for k in range(8)]
    + [("button", name) for name in ["A", "B", "X", "Y", "LB", "RB"]]
)
N_SYMBOLS = 22

@dataclass
class CaptureState:
    frames: list = field(default_factory=list)  # list of (timestamp_ns, (C, T_chunk) float32)
    dropped: int = 0
    running: bool = True
    t0_mono: float = 0.0
    t0_frame_ns: int = 0

def tap_reader(tap: Tap, state: CaptureState, n_channels: int):
    """Background thread. Reads tap.stream(), parses BroadbandFrame, appends to state.frames."""
    first = True
    for raw in tap.stream(timeout_ms=100):
        if not state.running:
            break
        if raw is None or len(raw) == 0:
            continue
        try:
            f = BroadbandFrame()
            f.ParseFromString(raw)
        except Exception:
            state.dropped += 1
            continue
        if first:
            state.t0_frame_ns = f.timestamp_ns
            first = False
        # frame_data is flat float32; channel_ranges describes layout.
        # Assume channel-major: frame_data = concat over channels of T samples.
        data = np.array(list(f.frame_data), dtype=np.float32)
        if data.size % n_channels != 0:
            state.dropped += 1
            continue
        T = data.size // n_channels
        chunk = data.reshape(n_channels, T)
        state.frames.append((f.timestamp_ns, chunk))

def build_config(peripheral_id: int, n_channels: int, fs_hz: int) -> syn.Config:
    channels = [syn.Channel(id=i, electrode_id=i, reference_id=i) for i in range(n_channels)]
    bb = syn.BroadbandSource(
        peripheral_id=peripheral_id,
        sample_rate_hz=fs_hz,
        bit_width=16,
        gain=20.0,
        signal=syn.SignalConfig(
            electrode=syn.ElectrodeConfig(
                channels=channels,
                low_cutoff_hz=0.0,
                high_cutoff_hz=fs_hz / 2.0,
            )
        ),
    )
    cfg = syn.Config()
    cfg.add_node(bb)
    return cfg

def find_tap_name(tap: Tap) -> str:
    """List taps, pick one matching 'broadband' in name. Raise if none."""
    taps = tap.list_taps()
    for t in taps:
        if "broadband" in t.name.lower():
            return t.name
    if taps:
        return taps[0].name
    raise RuntimeError("No taps found on device")

def render_cue(screen, font, sym_type, sym_val, trial_idx, total):
    """Draw fullscreen cue. sym_type in {'ljoy','rjoy','button'}."""
    W, H = screen.get_size()
    screen.fill((26, 26, 46))
    # Progress text (top, fixed width container)
    progress = font.render(f"{trial_idx + 1:5d} / {total:5d}", True, (136, 136, 136))
    screen.blit(progress, (W // 2 - progress.get_width() // 2, 40))
    # Cue body
    if sym_type in ("ljoy", "rjoy"):
        label = "LEFT STICK" if sym_type == "ljoy" else "RIGHT STICK"
        lbl = font.render(label, True, (170, 170, 170))
        screen.blit(lbl, (W // 2 - lbl.get_width() // 2, H // 2 - 220))
        # Draw 8-wedge pie, highlight sym_val
        cx, cy, r = W // 2, H // 2, 160
        for k in range(8):
            a0 = (k * 45 - 22.5) * np.pi / 180
            a1 = ((k + 1) * 45 - 22.5) * np.pi / 180
            color = (255, 200, 60) if k == sym_val else (60, 60, 80)
            pts = [(cx, cy)]
            for a in np.linspace(a0, a1, 12):
                pts.append((cx + r * np.cos(a), cy - r * np.sin(a)))
            pygame.draw.polygon(screen, color, pts)
        pygame.draw.circle(screen, (26, 26, 46), (cx, cy), 50)
    else:
        # Button cue: large centered letter
        big = pygame.font.SysFont("monospace", 220, bold=True)
        btn = big.render(str(sym_val), True, (255, 200, 60))
        screen.blit(btn, (W // 2 - btn.get_width() // 2, H // 2 - btn.get_height() // 2))
    pygame.display.flip()

def run_session(uri, player, peripheral_id, n_trials, seed, isi_s, n_channels, fs_hz, out_dir,
                dry_run=False):
    rng = np.random.default_rng(seed)
    cue_order = rng.integers(0, N_SYMBOLS, size=n_trials, dtype=np.int64)
    tap = None
    device = None
    if not dry_run:
        # Connect, configure, start
        device = syn.Device(uri)
        info = device.info()
        if info is None:
            raise RuntimeError(f"Cannot reach device at {uri}")
        cfg = build_config(peripheral_id, n_channels, fs_hz)
        if not device.configure(cfg):
            raise RuntimeError("device.configure failed")
        if not device.start():
            raise RuntimeError("device.start failed")
        tap = Tap(uri)
        tap_name = find_tap_name(tap)
        if not tap.connect(tap_name):
            raise RuntimeError(f"Tap connect failed: {tap_name}")
    # Background capture
    state = CaptureState()
    state.t0_mono = time.monotonic()
    reader = None
    if not dry_run:
        reader = threading.Thread(target=tap_reader, args=(tap, state, n_channels), daemon=True)
        reader.start()
    # Pygame loop
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    font = pygame.font.SysFont("monospace", 36)
    clock = pygame.time.Clock()
    cue_log = []  # (t_cue_mono, symbol_idx)
    try:
        # Countdown
        for k in [3, 2, 1]:
            screen.fill((26, 26, 46))
            t = font.render(f"Starting in {k}", True, (255, 255, 255))
            screen.blit(t, (screen.get_width() // 2 - t.get_width() // 2,
                            screen.get_height() // 2))
            pygame.display.flip()
            time.sleep(1.0)
        # Main loop
        for i, sym_idx in enumerate(cue_order):
            sym_type, sym_val = SYMBOLS[int(sym_idx)]
            cue_t = time.monotonic()
            cue_log.append((cue_t, int(sym_idx)))
            render_cue(screen, font, sym_type, sym_val, i, n_trials)
            # ESC to abort
            deadline = cue_t + isi_s
            while time.monotonic() < deadline:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                        raise KeyboardInterrupt
                clock.tick(120)
    finally:
        state.running = False
        time.sleep(0.2)
        if tap is not None:
            try:
                tap.disconnect()
            except Exception:
                pass
        if device is not None:
            try:
                device.stop()
            except Exception:
                pass
        pygame.quit()
    # Assemble neural
    if dry_run:
        T_total = int(n_trials * isi_s * fs_hz)
        neural = np.random.randn(n_channels, T_total).astype(np.float32)
        frame_timestamps = np.zeros(1, dtype=np.int64)
        state.t0_frame_ns = 0
        state.dropped = 0
    else:
        if not state.frames:
            raise RuntimeError("No frames captured")
        neural = np.concatenate([c for _, c in state.frames], axis=1)
        frame_timestamps = np.array([ts for ts, _ in state.frames], dtype=np.int64)
    cue_log_arr = np.array(cue_log, dtype=[("t_mono", "f8"), ("symbol_idx", "i8")])
    out_dir = Path(out_dir) / player
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"hard_{ts}.npz"
    np.savez(
        out_path,
        neural=neural,
        cue_log=cue_log_arr,
        frame_timestamps_ns=frame_timestamps,
        t0_mono=state.t0_mono,
        t0_frame_ns=state.t0_frame_ns,
        sample_rate_hz=fs_hz,
        peripheral_id=peripheral_id,
        seed=seed,
        n_symbols=N_SYMBOLS,
    )
    print(f"SAVED shape={neural.shape} n_trials={len(cue_log)} cues_logged={len(cue_log)} "
          f"tap_frames_dropped={state.dropped} path={out_path}")
    return 0

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--uri", default=None, help="SciFi IP or name")
    p.add_argument("--player", required=True)
    p.add_argument("--peripheral", type=int, default=108)
    p.add_argument("--n-trials", type=int, default=2200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--isi", type=float, default=0.6)
    p.add_argument("--n-channels", type=int, default=64)
    p.add_argument("--fs-hz", type=int, default=30000)
    p.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--dry-run", action="store_true", help="Skip hardware; use synthetic neural data")
    args = p.parse_args(argv)
    if not args.dry_run and args.uri is None:
        p.error("--uri is required unless --dry-run is set")
    return run_session(
        uri=args.uri, player=args.player, peripheral_id=args.peripheral,
        n_trials=args.n_trials, seed=args.seed, isi_s=args.isi,
        n_channels=args.n_channels, fs_hz=args.fs_hz, out_dir=args.out_dir,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    sys.exit(main())

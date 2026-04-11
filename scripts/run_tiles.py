"""
piano_tiles.py  –  BCI Piano Tiles Game
Built on top of collect_training.py infrastructure.

Dry-run (keyboard) usage:
    python piano_tiles.py --dry-run

Live (Synapse tap) usage:
    python piano_tiles.py --uri <device-ip>

Controls in dry-run mode:
    Joystick sectors  →  keys 1-8 (sector index 0-7)
    Buttons           →  D=DOWN  R=RIGHT  L=LEFT  U=UP  B=LB  N=RB
    ESC               →  quit
"""

import argparse, sys, time, threading, math
from dataclasses import dataclass, field
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Try to import Synapse deps – gracefully degrade in dry-run
# ---------------------------------------------------------------------------
try:
    import synapse as syn
    from synapse.client.taps import Tap
    from synapse.api.datatype_pb2 import BroadbandFrame
    HAS_SYNAPSE = True
except ImportError:
    HAS_SYNAPSE = False

# ---------------------------------------------------------------------------
# Symbol alphabet  (same as collect_training.py)
# ---------------------------------------------------------------------------
SYMBOLS = (
    [("ljoy", k) for k in range(8)]
    + [("rjoy", k) for k in range(8)]
    + [("button", name) for name in ["DOWN", "RIGHT", "LEFT", "UP", "LB", "RB"]]
)
N_SYMBOLS = len(SYMBOLS)   # 22

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
N_LANES         = 4
GAME_DURATION   = 60.0
TILE_H          = 100
TILE_W_FRAC     = 0.90
HIT_ZONE_H      = 90
HIT_ZONE_Y_FRAC = 0.88
PIE_R           = 28
COUNTDOWN_SECS  = 3
FLASH_DURATION  = 0.25

# Adaptive speed bounds
SPEED_INIT      = 200.0
SPEED_MIN       = 200.0
SPEED_MAX       = 700.0
SPAWN_INIT      = 0.55
SPAWN_MIN       = 0.25
SPAWN_MAX       = 0.80
ADAPT_EVERY     = 3.0    # seconds between adaptation checks
ADAPT_WINDOW    = 5.0    # seconds of history to measure hit rate
HIT_RATE_UP     = 1.5    # hits/sec threshold to speed up
HIT_RATE_DOWN   = 0.5    # hits/sec threshold to slow down

# Colours
BG              = (10,  10,  10)
LANE_LINE       = (50,  50,  50)
HIT_ZONE_COL    = (30,  30,  30)
HIT_ZONE_BORDER = (180, 180, 180)
TILE_IDLE       = (20,  20,  20)
TILE_HIT_OK     = (50, 220, 100)
TILE_HIT_MISS   = (220, 60,  60)
TILE_EXPIRED    = (80,  60,  60)
TEXT_COL        = (210, 210, 230)
ACCENT          = (255, 255,  255)
DIM             = (120, 120, 120)

# ---------------------------------------------------------------------------
# Keyboard mapping for dry-run
# ---------------------------------------------------------------------------
DRY_RUN_KEY_MAP = {
    pygame.K_1: ("joy_sector", 0),
    pygame.K_2: ("joy_sector", 1),
    pygame.K_3: ("joy_sector", 2),
    pygame.K_4: ("joy_sector", 3),
    pygame.K_5: ("joy_sector", 4),
    pygame.K_6: ("joy_sector", 5),
    pygame.K_7: ("joy_sector", 6),
    pygame.K_8: ("joy_sector", 7),
    pygame.K_d: ("button", "DOWN"),
    pygame.K_r: ("button", "RIGHT"),
    pygame.K_l: ("button", "LEFT"),
    pygame.K_u: ("button", "UP"),
    pygame.K_b: ("button", "LB"),
    pygame.K_n: ("button", "RB"),
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Tile:
    sym_idx:  int
    lane:     int
    y:        float
    state:    str   = "falling"   # falling | hit_ok | hit_miss | expired
    flash_t:  float = 0.0

@dataclass
class GameState:
    tiles:          list  = field(default_factory=list)
    sc:             int   = 0
    si:             int   = 0
    last_spawn:     float = 0.0
    start_t:        float = 0.0
    elapsed:        float = 0.0
    speed:          float = SPEED_INIT
    spawn_interval: float = SPAWN_INIT
    last_adapt:     float = 0.0
    recent_hits:    list  = field(default_factory=list)  # monotonic times of correct hits

@dataclass
class TapState:
    latest_sym: int  = -1
    running:    bool = True

# ---------------------------------------------------------------------------
# Synapse tap reader
# ---------------------------------------------------------------------------
def decoded_tap_reader(tap, state: TapState):
    for raw in tap.stream(timeout_ms=50):
        if not state.running:
            break
        if raw is None or len(raw) == 0:
            continue
        try:
            sym_idx = int.from_bytes(raw[:4], "little")
            if 0 <= sym_idx < N_SYMBOLS:
                state.latest_sym = sym_idx
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Input matching
# ---------------------------------------------------------------------------
def input_matches(sym_idx: int, input_type: str, input_val) -> bool:
    stype, sval = SYMBOLS[sym_idx]
    if stype in ("ljoy", "rjoy"):
        return input_type == "joy_sector" and input_val == sval
    else:
        return input_type == "button" and input_val == sval

# ---------------------------------------------------------------------------
# Bit rate formula (Shenoy et al.)
# ---------------------------------------------------------------------------
def bit_rate(n: int, sc: int, si: int, t: float) -> float:
    if t <= 0 or n < 3:
        return 0.0
    return max(0.0, math.log2(n - 1) * max(sc - si, 0) / t)

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_pie_on_tile(surf, cx, cy, r, highlight_sector):
    for k in range(8):
        a0 = (k * 45 - 22.5) * math.pi / 180
        a1 = ((k + 1) * 45 - 22.5) * math.pi / 180
        color = ACCENT if k == highlight_sector else (50, 50, 80)
        pts = [(cx, cy)]
        for a in np.linspace(a0, a1, 10):
            pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
        pygame.draw.polygon(surf, color, pts)
    pygame.draw.circle(surf, (20, 20, 40), (cx, cy), r // 3)


def draw_tile(surf, tile: Tile, lane_x: int, lane_w: int, font_sm, font_xs):
    tw = int(lane_w * TILE_W_FRAC)
    tx = lane_x + (lane_w - tw) // 2
    ty = int(tile.y)

    col = {
        "falling":  TILE_IDLE,
        "hit_ok":   TILE_HIT_OK,
        "hit_miss": TILE_HIT_MISS,
        "expired":  TILE_EXPIRED,
    }.get(tile.state, TILE_IDLE)

    pygame.draw.rect(surf, col, (tx, ty, tw, TILE_H), border_radius=8)
    pygame.draw.rect(surf, (255, 255, 255), (tx, ty, tw, TILE_H), 2, border_radius=8)

    stype, sval = SYMBOLS[tile.sym_idx]
    cx = tx + tw // 2
    cy = ty + TILE_H // 2

    if stype in ("ljoy", "rjoy"):
        draw_pie_on_tile(surf, cx, cy - 8, PIE_R, sval)
        lbl = font_xs.render("L" if stype == "ljoy" else "R", True, (200, 200, 200))
        surf.blit(lbl, (cx - lbl.get_width() // 2, ty + TILE_H - 22))
    else:
        lbl = font_sm.render(str(sval), True, (255, 255, 255))
        surf.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))


def draw_hud(surf, gs: GameState, W: int, H: int, font_md, font_xs):
    elapsed   = min(gs.elapsed, GAME_DURATION)
    remaining = max(0.0, GAME_DURATION - elapsed)
    br        = bit_rate(N_SYMBOLS, gs.sc, gs.si, elapsed)

    pygame.draw.rect(surf, (25, 25, 45), (0, 0, W, 50))

    timer_col = ACCENT if remaining > 10 else (220, 80, 80)
    surf.blit(font_md.render(f"{remaining:5.1f}s", True, timer_col), (20, 8))

    br_txt = font_md.render(f"{br:5.2f} bps", True, TEXT_COL)
    surf.blit(br_txt, (W // 2 - br_txt.get_width() // 2, 8))

    spd_txt = font_xs.render(f"speed {gs.speed:.0f}  interval {gs.spawn_interval:.2f}s", True, DIM)
    surf.blit(spd_txt, (W - spd_txt.get_width() - 20, 4))

    score_txt = font_xs.render(f"✓ {gs.sc}   ✗ {gs.si}   N={N_SYMBOLS}", True, DIM)
    surf.blit(score_txt, (W - score_txt.get_width() - 20, 28))


def draw_hit_zone(surf, W: int, H: int):
    hz_y = int(H * HIT_ZONE_Y_FRAC)
    pygame.draw.rect(surf, HIT_ZONE_COL, (0, hz_y, W, HIT_ZONE_H))
    pygame.draw.line(surf, HIT_ZONE_BORDER, (0, hz_y), (W, hz_y), 2)
    pygame.draw.line(surf, HIT_ZONE_BORDER, (0, hz_y + HIT_ZONE_H), (W, hz_y + HIT_ZONE_H), 2)


def draw_lane_lines(surf, lane_xs, H):
    for x in lane_xs:
        pygame.draw.line(surf, LANE_LINE, (x, 50), (x, H), 1)


def draw_key_legend(surf, W: int, H: int, font_xs):
    lines = [
        "1-8: joystick sector   D=DOWN  R=RIGHT  L=LEFT  U=UP  B=LB  N=RB",
        "ESC: quit",
    ]
    for i, line in enumerate(lines):
        t = font_xs.render(line, True, (60, 60, 90))
        surf.blit(t, (W // 2 - t.get_width() // 2, H - 38 + i * 18))


def draw_results(surf, gs: GameState, W: int, H: int, font_lg, font_md, font_sm):
    surf.fill(BG)
    br = bit_rate(N_SYMBOLS, gs.sc, gs.si, GAME_DURATION)
    lines = [
        ("GAME OVER",                    font_lg, ACCENT),
        (f"Bit Rate:  {br:.3f} bps",     font_md, TEXT_COL),
        (f"N = {N_SYMBOLS}",             font_sm, DIM),
        (f"Correct  (Sc) = {gs.sc}",     font_sm, TILE_HIT_OK),
        (f"Incorrect (Si) = {gs.si}",    font_sm, TILE_HIT_MISS),
        (f"Duration = {GAME_DURATION:.1f}s", font_sm, DIM),
        ("",                             font_sm, DIM),
        ("Press ESC or Q to exit",       font_sm, (60, 60, 90)),
    ]
    y = H // 2 - 180
    for text, fnt, col in lines:
        rendered = fnt.render(text, True, col)
        surf.blit(rendered, (W // 2 - rendered.get_width() // 2, y))
        y += rendered.get_height() + 14
    pygame.display.flip()

# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------
def run_game(uri, peripheral_id, n_channels, fs_hz, seed, dry_run=False):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Synapse setup
    # ------------------------------------------------------------------
    tap_state  = TapState()
    tap_thread = None

    if not dry_run:
        if not HAS_SYNAPSE:
            print("ERROR: synapse package not installed. Use --dry-run.")
            return 1
        device = syn.Device(uri)
        if device.info() is None:
            print(f"ERROR: Cannot reach device at {uri}")
            return 1
        cfg = _build_config(peripheral_id, n_channels, fs_hz)
        if not device.configure(cfg):
            print("ERROR: device.configure failed")
            return 1
        if not device.start():
            print("ERROR: device.start failed")
            return 1
        tap = Tap(uri)
        tap_name = _find_tap_name(tap)
        if not tap.connect(tap_name):
            print(f"ERROR: Tap connect failed: {tap_name}")
            return 1
        tap_thread = threading.Thread(
            target=decoded_tap_reader, args=(tap, tap_state), daemon=True
        )
        tap_thread.start()

    # ------------------------------------------------------------------
    # Pygame setup
    # ------------------------------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("BCI Piano Tiles")
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    font_lg = pygame.font.SysFont("monospace", 72, bold=True)
    font_md = pygame.font.SysFont("monospace", 36, bold=True)
    font_sm = pygame.font.SysFont("monospace", 26)
    font_xs = pygame.font.SysFont("monospace", 18)

    lane_w  = W // N_LANES
    lane_xs = [i * lane_w for i in range(N_LANES + 1)]
    hit_zone_y = int(H * HIT_ZONE_Y_FRAC)

    # ------------------------------------------------------------------
    # Countdown
    # ------------------------------------------------------------------
    for k in range(COUNTDOWN_SECS, 0, -1):
        screen.fill(BG)
        t = font_lg.render(str(k), True, ACCENT)
        screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - t.get_height() // 2))
        sub = font_sm.render("Get ready!", True, DIM)
        screen.blit(sub, (W // 2 - sub.get_width() // 2, H // 2 + 60))
        pygame.display.flip()
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); return 0
            clock.tick(60)

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------
    gs = GameState()
    gs.start_t    = time.monotonic()
    gs.last_spawn = gs.start_t
    gs.last_adapt = gs.start_t

    sym_sequence = rng.integers(0, N_SYMBOLS, size=10000, dtype=np.int64)
    sym_cursor   = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    game_over = False
    while True:
        now        = time.monotonic()
        gs.elapsed = now - gs.start_t
        dt         = clock.tick(120) / 1000.0

        # ---- Input ----
        pressed_inputs = []

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                game_over = True
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    game_over = True
                if dry_run and e.key in DRY_RUN_KEY_MAP:
                    pressed_inputs.append(DRY_RUN_KEY_MAP[e.key])

        if not dry_run and tap_state.latest_sym >= 0:
            stype, sval = SYMBOLS[tap_state.latest_sym]
            pressed_inputs.append(
                ("joy_sector", sval) if stype in ("ljoy", "rjoy") else ("button", sval)
            )
            tap_state.latest_sym = -1

        # ---- Time up? ----
        if gs.elapsed >= GAME_DURATION:
            game_over = True
        if game_over:
            break

        # ---- Spawn tile ----
        if now - gs.last_spawn >= gs.spawn_interval:
            tile = Tile(
                sym_idx=int(sym_sequence[sym_cursor % len(sym_sequence)]),
                lane=int(rng.integers(0, N_LANES)),
                y=-TILE_H,
            )
            sym_cursor += 1
            gs.tiles.append(tile)
            gs.last_spawn = now

        # ---- Adaptive difficulty (every ADAPT_EVERY seconds) ----
        if now - gs.last_adapt >= ADAPT_EVERY and gs.elapsed > ADAPT_EVERY:
            gs.recent_hits  = [t for t in gs.recent_hits if now - t < ADAPT_WINDOW]
            hit_rate        = len(gs.recent_hits) / ADAPT_WINDOW
            if hit_rate > HIT_RATE_UP:
                gs.speed          = min(gs.speed * 1.10, SPEED_MAX)
                gs.spawn_interval = max(gs.spawn_interval * 0.90, SPAWN_MIN)
            elif hit_rate < HIT_RATE_DOWN:
                gs.speed          = max(gs.speed * 0.95, SPEED_MIN)
                gs.spawn_interval = min(gs.spawn_interval * 1.05, SPAWN_MAX)
            gs.last_adapt = now

        # ---- Update tiles ----
        for tile in gs.tiles:
            if tile.state != "falling":
                continue
            tile.y += gs.speed * dt
            tile_bottom = tile.y + TILE_H
            in_zone = tile.y < hit_zone_y + HIT_ZONE_H and tile_bottom > hit_zone_y

            if in_zone:
                for inp_type, inp_val in pressed_inputs:
                    if input_matches(tile.sym_idx, inp_type, inp_val):
                        tile.state  = "hit_ok"
                        tile.flash_t = now
                        gs.sc       += 1
                        gs.recent_hits.append(now)
                        pressed_inputs = []
                        break

            if tile.y > hit_zone_y + HIT_ZONE_H and tile.state == "falling":
                tile.state  = "expired"
                tile.flash_t = now

        # ---- Wrong inputs ----
        if pressed_inputs:
            zone_tiles = [
                t for t in gs.tiles
                if t.state == "falling"
                and t.y < hit_zone_y + HIT_ZONE_H
                and t.y + TILE_H > hit_zone_y
            ]
            if zone_tiles:
                gs.si += len(pressed_inputs)
                closest = min(
                    zone_tiles,
                    key=lambda t: abs((t.y + TILE_H / 2) - (hit_zone_y + HIT_ZONE_H / 2))
                )
                closest.state   = "hit_miss"
                closest.flash_t = now

        # ---- Cull old tiles ----
        gs.tiles = [
            t for t in gs.tiles
            if not (t.state in ("hit_ok", "hit_miss", "expired") and now - t.flash_t > FLASH_DURATION)
            and t.y < H + TILE_H
        ]

        # ---- Draw ----
        screen.fill(BG)
        draw_lane_lines(screen, lane_xs, H)
        draw_hit_zone(screen, W, H)
        for tile in gs.tiles:
            draw_tile(screen, tile, tile.lane * lane_w, lane_w, font_sm, font_xs)
        draw_hud(screen, gs, W, H, font_md, font_xs)
        if dry_run:
            draw_key_legend(screen, W, H, font_xs)
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Cleanup & results
    # ------------------------------------------------------------------
    tap_state.running = False
    if not dry_run:
        try:
            tap.disconnect()
            device.stop()
        except Exception:
            pass

    draw_results(screen, gs, W, H, font_lg, font_md, font_sm)
    waiting = True
    while waiting:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                waiting = False
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                waiting = False
        clock.tick(30)

    pygame.quit()

    br = bit_rate(N_SYMBOLS, gs.sc, gs.si, GAME_DURATION)
    print(f"\n=== FINAL RESULTS ===")
    print(f"N={N_SYMBOLS}  Sc={gs.sc}  Si={gs.si}  t={GAME_DURATION:.1f}s")
    print(f"Bit Rate: {br:.4f} bps")
    return 0


# ---------------------------------------------------------------------------
# Synapse helpers
# ---------------------------------------------------------------------------
def _build_config(peripheral_id, n_channels, fs_hz):
    import synapse as syn
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


def _find_tap_name(tap):
    taps = tap.list_taps()
    for t in taps:
        if "broadband" in t.name.lower():
            return t.name
    if taps:
        return taps[0].name
    raise RuntimeError("No taps found on device")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="BCI Piano Tiles Game")
    p.add_argument("--uri",        default=None,            help="SciFi IP or device name")
    p.add_argument("--peripheral", type=int, default=108,   help="Peripheral ID")
    p.add_argument("--n-channels", type=int, default=64,    help="Number of neural channels")
    p.add_argument("--fs-hz",      type=int, default=30000, help="Sample rate Hz")
    p.add_argument("--seed",       type=int, default=0,     help="RNG seed (default 0)")
    p.add_argument("--dry-run",    action="store_true",     help="Keyboard input, no hardware")
    args = p.parse_args(argv)

    if not args.dry_run and args.uri is None:
        p.error("--uri is required unless --dry-run is set")

    return run_game(
        uri=args.uri,
        peripheral_id=args.peripheral,
        n_channels=args.n_channels,
        fs_hz=args.fs_hz,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
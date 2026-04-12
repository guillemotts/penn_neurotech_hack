import argparse, json, sys, time
from pathlib import Path
import numpy as np

SYMBOLS = (
    [("ljoy", k) for k in range(8)]
    + [("rjoy", k) for k in range(8)]
    + [("button", name) for name in ["UP", "DOWN", "LEFT", "RIGHT", "LB", "RB"]]
)
N_SYMBOLS = 22
SYMBOL_STRS = [f"{t}_{v}" for t, v in SYMBOLS]

ALPHABETS = {
    "full": list(range(22)),  # all 22
    "n10": [0, 2, 4, 6, 16, 17, 18, 19, 20, 21],  # 4 ljoy cardinals + 6 buttons
    "n14": [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21],
}

COLOR_LEFT = (80, 200, 255)
COLOR_RIGHT = (255, 160, 60)
COLOR_VERT = (255, 220, 80)
COLOR_BG = (26, 26, 46)
COLOR_DIM = (60, 60, 80)
COLOR_LABEL = (170, 170, 170)

def cue_color(sym_type: str, sym_val) -> tuple:
    """Return highlight color for a given symbol. Pure function, testable."""
    if sym_type == "ljoy":
        return COLOR_LEFT
    if sym_type == "rjoy":
        return COLOR_RIGHT
    if sym_type == "button":
        if sym_val in ("LEFT", "LB"):
            return COLOR_LEFT
        if sym_val in ("RIGHT", "RB"):
            return COLOR_RIGHT
        if sym_val in ("UP", "DOWN"):
            return COLOR_VERT
    raise ValueError(f"Unknown symbol: {sym_type}_{sym_val}")

def symbol_str(idx: int) -> str:
    return SYMBOL_STRS[idx]

def generate_cue_sequence(n_trials: int, seed: int, alphabet: list = None) -> np.ndarray:
    """iid uniform sampling over N_SYMBOLS. Returns int64 array (n_trials,)."""
    rng = np.random.default_rng(seed)
    if alphabet is None:
        return rng.integers(0, N_SYMBOLS, size=n_trials, dtype=np.int64)
    pool = np.array(alphabet, dtype=np.int64)
    return pool[rng.integers(0, len(pool), size=n_trials)]

def build_log(
    session_start_ns: int,
    sync_prompt_ns: int,
    sync_release_ns: int,
    trial_records: list,
    seed: int,
    n_trials: int,
    isi_s: float,
    alphabet_name: str = "full",
    alphabet_indices: list = None,
) -> dict:
    if alphabet_indices is None:
        alphabet_indices = ALPHABETS["full"]
    return {
        "schema_version": 1,
        "symbols": SYMBOL_STRS,
        "n_symbols": N_SYMBOLS,
        "alphabet": alphabet_name,
        "alphabet_symbols": [SYMBOL_STRS[i] for i in alphabet_indices],
        "seed": int(seed),
        "n_trials": int(n_trials),
        "isi_s": float(isi_s),
        "session_start_t_mono_ns": int(session_start_ns),
        "sync_gesture": {
            "instruction": "Hold LB for 2 seconds, then release.",
            "prompt_t_mono_ns": int(sync_prompt_ns),
            "release_t_mono_ns": int(sync_release_ns),
        },
        "trials": [
            {
                "idx": int(r["idx"]),
                "symbol_idx": int(r["symbol_idx"]),
                "symbol": SYMBOL_STRS[int(r["symbol_idx"])],
                "t_mono_ns": int(r["t_mono_ns"]),
            }
            for r in trial_records
        ],
    }

def write_log(out_path: Path, log: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(log, f, indent=2)

def run_dry(n_trials: int, seed: int, isi_s: float, out_path: Path, alphabet_name: str = "full") -> int:
    alphabet_indices = ALPHABETS[alphabet_name]
    cues = generate_cue_sequence(n_trials, seed, alphabet_indices)
    t0 = time.monotonic_ns()
    sync_prompt = t0 + 1_000_000_000
    sync_release = sync_prompt + 2_000_000_000
    isi_ns = int(isi_s * 1e9)
    trials = [
        {"idx": i, "symbol_idx": int(s), "t_mono_ns": sync_release + 1_000_000_000 + i * isi_ns}
        for i, s in enumerate(cues)
    ]
    log = build_log(t0, sync_prompt, sync_release, trials, seed, n_trials, isi_s,
                    alphabet_name=alphabet_name, alphabet_indices=alphabet_indices)
    write_log(out_path, log)
    print(f"DRY n_trials={n_trials} out={out_path}")
    return 0

def run_live(n_trials: int, seed: int, isi_s: float, out_path: Path, alphabet_name: str = "full") -> int:
    import pygame
    alphabet_indices = ALPHABETS[alphabet_name]
    cues = generate_cue_sequence(n_trials, seed, alphabet_indices)
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    W, H = screen.get_size()
    font = pygame.font.SysFont("monospace", 36)
    big = pygame.font.SysFont("monospace", 200, bold=True)
    clock = pygame.time.Clock()
    session_start_ns = time.monotonic_ns()

    def draw_text_centered(text, y, f, color=(255, 255, 255)):
        surf = f.render(text, True, color)
        screen.blit(surf, (W // 2 - surf.get_width() // 2, y))

    def pump_events_until(deadline_ns: int):
        while time.monotonic_ns() < deadline_ns:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    raise KeyboardInterrupt
            clock.tick(120)

    def draw_stick(sym_type, sym_val, highlight_color):
        screen.fill(COLOR_BG)
        label = "LEFT STICK" if sym_type == "ljoy" else "RIGHT STICK"
        draw_text_centered(label, H // 2 - 260, font, highlight_color)
        cx, cy, r = W // 2, H // 2, 160
        for k in range(8):
            a0 = (k * 45 - 22.5) * np.pi / 180
            a1 = ((k + 1) * 45 - 22.5) * np.pi / 180
            color = highlight_color if k == sym_val else COLOR_DIM
            pts = [(cx, cy)]
            for a in np.linspace(a0, a1, 12):
                pts.append((cx + r * np.cos(a), cy - r * np.sin(a)))
            pygame.draw.polygon(screen, color, pts)
        pygame.draw.circle(screen, COLOR_BG, (cx, cy), 50)

    def draw_button(sym_val, highlight_color):
        screen.fill(COLOR_BG)
        draw_text_centered("BUTTON", H // 2 - 260, font, COLOR_LABEL)
        surf = big.render(str(sym_val), True, highlight_color)
        screen.blit(surf, (W // 2 - surf.get_width() // 2, H // 2 - surf.get_height() // 2))

    trial_records = []
    sync_prompt_ns = 0
    sync_release_ns = 0
    try:
        # Countdown
        for k in [3, 2, 1]:
            screen.fill((26, 26, 46))
            draw_text_centered(f"Starting in {k}", H // 2, font)
            pygame.display.flip()
            pump_events_until(time.monotonic_ns() + 1_000_000_000)
        # Sync gesture
        screen.fill((26, 26, 46))
        draw_text_centered("Hold LB for 2 seconds", H // 2 - 40, font, (255, 200, 60))
        draw_text_centered("then release", H // 2 + 20, font, (255, 200, 60))
        pygame.display.flip()
        sync_prompt_ns = time.monotonic_ns()
        pump_events_until(sync_prompt_ns + 2_000_000_000)
        sync_release_ns = time.monotonic_ns()
        screen.fill((26, 26, 46))
        draw_text_centered("Release now", H // 2, font, (255, 200, 60))
        pygame.display.flip()
        pump_events_until(sync_release_ns + 1_000_000_000)
        # Main loop
        for i, sym_idx in enumerate(cues):
            sym_type, sym_val = SYMBOLS[int(sym_idx)]
            if i > 0 and cues[i] == cues[i - 1]:
                screen.fill(COLOR_BG)
                pygame.display.flip()
                flash_until = time.monotonic_ns() + 100_000_000
                pump_events_until(flash_until)
            highlight_color = cue_color(sym_type, sym_val)
            cue_t_ns = time.monotonic_ns()
            trial_records.append({"idx": i, "symbol_idx": int(sym_idx), "t_mono_ns": cue_t_ns})
            if sym_type in ("ljoy", "rjoy"):
                draw_stick(sym_type, sym_val, highlight_color)
            else:
                draw_button(sym_val, highlight_color)
            progress = font.render(f"{i + 1:5d} / {n_trials:5d}", True, (136, 136, 136))
            screen.blit(progress, (W // 2 - progress.get_width() // 2, 40))
            pygame.display.flip()
            pump_events_until(cue_t_ns + int(isi_s * 1e9))
    except KeyboardInterrupt:
        print("ABORTED by user")
    finally:
        pygame.quit()

    log = build_log(session_start_ns, sync_prompt_ns, sync_release_ns,
                    trial_records, seed, n_trials, isi_s,
                    alphabet_name=alphabet_name, alphabet_indices=alphabet_indices)
    write_log(out_path, log)
    print(f"SAVED trials_written={len(trial_records)} out={out_path}")
    return 0

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=2200)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--isi", type=float, default=0.8)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--alphabet", choices=list(ALPHABETS.keys()), default="full")
    args = p.parse_args(argv)
    if args.n_trials < 1:
        p.error("--n-trials must be >= 1")
    if args.isi <= 0:
        p.error("--isi must be > 0")
    if args.dry_run:
        return run_dry(args.n_trials, args.seed, args.isi, args.out, alphabet_name=args.alphabet)
    return run_live(args.n_trials, args.seed, args.isi, args.out, alphabet_name=args.alphabet)

if __name__ == "__main__":
    sys.exit(main())

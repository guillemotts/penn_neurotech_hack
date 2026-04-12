import math
import os
import queue
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pygame


# ------------------------------------------------------------
# Dance Simulation Game Prototype
#
# Install:
#   pip install pygame
#
# Run:
#   python dance_sim_game.py
#
# Controls:
#   SPACE  -> Start / stop run
#   R      -> Reset run
#   ESC    -> Quit
#
# Keyboard test-mode inputs:
#   Buttons:
#       1=A, 2=B, 3=X, 4=Y, 5=LB, 6=RB
#   Left stick 4-way:
#         W
#       A   D
#         X
#   Right stick 4-way:
#         I
#       J   L
#         ,
#
# UDP input mode:
#   Send UTF-8 strings like "A", "RB", "L_UP", "R_DOWN"
#   to UDP port 5005 on this machine.
# ------------------------------------------------------------
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()

# ------------------------------------------------------------
# Layout tuning
# ------------------------------------------------------------
LEFT_PANEL_RATIO = 0.56
LEFT_PANEL_W = int(WIDTH * LEFT_PANEL_RATIO)
RIGHT_PANEL_X = LEFT_PANEL_W
RIGHT_PANEL_W = WIDTH - LEFT_PANEL_W

FPS = 60
TRIAL_DURATION = 60
VISIBLE_PROMPTS = 3
UDP_PORT = 5005
CONTROLLER_IMAGE_PATH = "Controller.jpg"
JUDGE_FACE_PATH = "judge_face.png"  # Optional future stretch asset.

# Top-left white status box
STATUS_BOX_X = 48
STATUS_BOX_Y = 28
STATUS_BOX_W = LEFT_PANEL_W - 116
STATUS_BOX_H = 150

# Right-side queue panel (outer container)
QUEUE_PANEL_X = RIGHT_PANEL_X + 16
QUEUE_PANEL_Y = 18
QUEUE_PANEL_W = RIGHT_PANEL_W - 32
QUEUE_PANEL_H = HEIGHT - 36

# Inner spacing for the 3 controller cards
QUEUE_CARD_SIDE_PAD = 16
QUEUE_CARD_TOP_PAD = 16
QUEUE_CARD_BOTTOM_PAD = 16
QUEUE_CARD_GAP = 14

BG = (243, 236, 218)
INK = (34, 34, 34)
PANEL = (252, 248, 238)
PANEL_ALT = (242, 234, 214)
GLOW = (255, 220, 120)
GREEN = (58, 184, 95)
RED = (207, 72, 72)
BLUE = (88, 130, 235)
PURPLE = (123, 91, 214)
WHITE = (250, 250, 250)
NOTE = (120, 91, 214)
SILHOUETTE = (45, 45, 55)
ACCENT = (255, 176, 67)

# ── Calibration constants ──────────────────────────────────────────────────────
ARCH_CHOICES = ["eegnet", "dense", "linear"]
_SCRIPT_DIR = Path(__file__).parent
CALIB_DATA_ROOTS = [
    _SCRIPT_DIR.parent / "data" / "raw",
    Path("data") / "raw",
]
CALIB_CONFIG_DIR = _SCRIPT_DIR.parent / "data" / "configs"
CALIB_FILE_ROWS = 7
CALIB_MAX_NAME  = 32

CALIB_H5_ROOTS = [
    _SCRIPT_DIR.parent / "data" / "raw",
    Path("data") / "raw",
]
COLLECT_N_TRIALS_DEFAULT = 500
COLLECT_ISI_DEFAULT      = 0.8
COLLECT_ALPHABETS = {
    "full": list(range(22)),
    "n14":  [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21],
    "n10":  [0, 2, 4, 6, 16, 17, 18, 19, 20, 21],
}
# Symbol table matching run_cues.py
_COLLECT_SYMBOLS = (
    [("ljoy", k) for k in range(8)]
    + [("rjoy", k) for k in range(8)]
    + [("button", name) for name in ["UP", "DOWN", "LEFT", "RIGHT", "LB", "RB"]]
)

# ── Synapse decoded-controls tap ───────────────────────────────────────────────
SYNAPSE_URI_DEFAULT = "127.0.0.1:647"
SYNAPSE_TAP_DEFAULT = "decoded_controls"

# Maps decoder symbol index (0-21, matching _COLLECT_SYMBOLS ordering) → INPUT_NAMES string.
# _COLLECT_SYMBOLS: [ljoy 0-7][rjoy 0-7][button UP/DOWN/LEFT/RIGHT/LB/RB]
# ljoy/rjoy sector 0 = RIGHT, sectors go clockwise (from run_cues.py wedge math).
_DECODE_IDX_TO_ACTION = [
    # ljoy sectors 0-7 (RIGHT → RIGHT → DOWN → DOWN → LEFT → LEFT → UP → UP, diagonals mapped to nearest cardinal)
    "L_RIGHT", "L_RIGHT",    "L_DOWN", "L_DOWN",
    "L_LEFT",  "L_LEFT",     "L_UP",   "L_UP",
    # rjoy sectors 0-7
    "R_RIGHT", "R_RIGHT",    "R_DOWN", "R_DOWN",
    "R_LEFT",  "R_LEFT",     "R_UP",   "R_UP",
    # buttons: UP→A, DOWN→B, LEFT→X, RIGHT→Y, LB, RB
    "A", "B", "X", "Y", "LB", "RB",
]
assert len(_DECODE_IDX_TO_ACTION) == 22


BUTTON_INPUTS = ["A", "B", "X", "Y", "RB", "LB"]
LEFT_STICK_INPUTS = [
    "L_UP",
    "L_LEFT", "L_RIGHT",
    "L_DOWN",
]
RIGHT_STICK_INPUTS = [
    "R_UP",
    "R_LEFT", "R_RIGHT",
    "R_DOWN",
]
INPUT_NAMES = BUTTON_INPUTS + LEFT_STICK_INPUTS + RIGHT_STICK_INPUTS
assert len(INPUT_NAMES) == 14

KEY_TO_ACTION = {
    pygame.K_1: "A",
    pygame.K_2: "B",
    pygame.K_3: "X",
    pygame.K_4: "Y",
    pygame.K_5: "LB",
    pygame.K_6: "RB",
    pygame.K_w: "L_UP",
    pygame.K_a: "L_LEFT",
    pygame.K_d: "L_RIGHT",
    pygame.K_x: "L_DOWN",
    pygame.K_i: "R_UP",
    pygame.K_j: "R_LEFT",
    pygame.K_l: "R_RIGHT",
    pygame.K_COMMA: "R_DOWN",
}

ALIAS_MAP = {
    "LUP": "L_UP",
    "LDOWN": "L_DOWN",
    "LLEFT": "L_LEFT",
    "LRIGHT": "L_RIGHT",
    "RUP": "R_UP",
    "RDOWN": "R_DOWN",
    "RLEFT": "R_LEFT",
    "RRIGHT": "R_RIGHT",
}


@dataclass
class FloatingNote:
    x: float
    y: float
    speed: float
    phase: float
    size: float


@dataclass
class Sardine:
    x: float
    y: float
    vx: float
    vy: float
    rotation: float = 0.0
    age: float = 0.0
    max_age: float = 1.2


@dataclass
class GameState:
    running: bool = False
    trial_ended: bool = False
    total_inputs: int = 0
    incorrect_inputs: int = 0
    streak: int = 0
    frowning: bool = False
    prompt_queue: list[str] = field(default_factory=list)
    last_result_color: tuple[int, int, int] | None = None
    last_result_until: float = 0.0
    last_input_text: str = "—"
    elapsed_before_pause: float = 0.0
    run_start_time: float | None = None
    dancer_scale: float = 0.60
    dancer_scale_target: float = 0.60

    def reset(self) -> None:
        self.running = False
        self.trial_ended = False
        self.total_inputs = 0
        self.incorrect_inputs = 0
        self.streak = 0
        self.frowning = False
        self.prompt_queue = [random_prompt() for _ in range(VISIBLE_PROMPTS)]
        self.last_result_color = None
        self.last_result_until = 0.0
        self.last_input_text = "—"
        self.elapsed_before_pause = 0.0
        self.run_start_time = None
        self.dancer_scale = 0.60
        self.dancer_scale_target = 0.60

    def start(self) -> None:
        if self.running or self.trial_ended:
            return
        self.running = True
        self.run_start_time = time.time()

    def end_trial(self) -> None:
        if self.run_start_time is not None:
            self.elapsed_before_pause += time.time() - self.run_start_time
        self.elapsed_before_pause = min(self.elapsed_before_pause, float(TRIAL_DURATION))
        self.running = False
        self.run_start_time = None
        self.trial_ended = True

    def elapsed(self) -> float:
        if self.running and self.run_start_time is not None:
            return self.elapsed_before_pause + (time.time() - self.run_start_time)
        return self.elapsed_before_pause

    def time_remaining(self) -> float:
        return max(0.0, TRIAL_DURATION - self.elapsed())

    def bitrate(self, N: int) -> float:
        elapsed = self.elapsed()
        if elapsed <= 0:
            return 0.0
        Sc = self.total_inputs - self.incorrect_inputs
        Si = self.incorrect_inputs
        return math.log2(N - 1) * max(Sc - Si, 0) / elapsed

    def process_input(self, action: str) -> bool | None:
        if not self.running or self.trial_ended or not self.prompt_queue:
            return None

        expected = self.prompt_queue[0]
        self.total_inputs += 1
        self.last_input_text = action
        correct = action == expected
        if correct:
            self.streak += 1
            self.frowning = False
        else:
            self.incorrect_inputs += 1
            self.streak = 0
            self.frowning = True

        self.last_result_color = GREEN if correct else RED
        self.last_result_until = time.time() + 0.30
        if correct:
            self.dancer_scale_target = min(1.0, self.dancer_scale_target + 0.05)
        else:
            self.dancer_scale_target = max(0.15, self.dancer_scale_target - 0.10)

        self.prompt_queue.pop(0)
        self.prompt_queue.append(random_prompt())
        return correct



@dataclass
class CalibState:
    phase: str = "setup"          # "setup" | "running" | "done" | "error"
    player_name: str = ""
    npz_files: list = field(default_factory=list)
    file_scroll: int = 0
    selected_file: int = 0
    arch_idx: int = 0
    active_field: str = "player"  # "player" | "files"
    progress_msg: str = ""
    result_N: int = 0
    result_theta: float = 0.0
    result_B: float = 0.0
    result_path: str = ""
    error_msg: str = ""
    _msg_queue: queue.Queue = field(default_factory=queue.Queue)

    def refresh_files(self) -> None:
        found: list[Path] = []
        for root in CALIB_DATA_ROOTS:
            if root.exists():
                found.extend(sorted(root.rglob("*.npz")))
        self.npz_files = found
        self.selected_file = min(self.selected_file, max(0, len(self.npz_files) - 1))
        self.file_scroll = 0

    def poll(self) -> None:
        while True:
            try:
                kind, payload = self._msg_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.progress_msg = payload
            elif kind == "done":
                self.result_N, self.result_theta, self.result_B, self.result_path = payload
                self.phase = "done"
            elif kind == "error":
                self.error_msg = payload
                self.phase = "error"

    def start_calibration(self) -> None:
        self.phase = "running"
        self.progress_msg = "Starting calibration..."
        threading.Thread(target=_calib_worker, args=(self,), daemon=True).start()



@dataclass
class CollectState:
    # ── Config (editable in setup) ────────────────────────────────────────────
    n_trials: int    = COLLECT_N_TRIALS_DEFAULT
    isi_s: float     = COLLECT_ISI_DEFAULT
    alphabet_name: str = "full"
    seed: int        = 0          # 0 → random at session start

    # ── Phase state machine ────────────────────────────────────────────────────
    # "setup" | "countdown" | "sync_prompt" | "sync_wait" | "sync_release"
    # | "cueing" | "h5_select" | "processing" | "done" | "error"
    # | "calibrating" | "calib_done" | "calib_error"
    phase: str = "setup"
    # calibration sub-results (populated after auto-calibration)
    calib_N: int = 0
    calib_theta: float = 0.0
    calib_B: float = 0.0
    calib_path: str = ""
    calib_arch_idx: int = 0   # arch index for auto-calibration (default eegnet)

    # ── Session runtime ────────────────────────────────────────────────────────
    cue_seq: list   = field(default_factory=list)   # list[int] symbol indices
    trial_records: list = field(default_factory=list)
    current_trial: int  = 0
    display_sym_idx: int = -2   # -2=not started, -1=flash/blank, ≥0=symbol
    session_start_ns: int = 0
    sync_prompt_ns: int   = 0
    sync_release_ns: int  = 0
    json_out_path: Path | None = None

    # ── Frame-based timing ────────────────────────────────────────────────────
    _phase_start_ns: int  = 0   # when the current phase began
    _next_change_ns: int  = 0   # when to advance to next cue / sub-phase

    # ── H5 pairing ────────────────────────────────────────────────────────────
    h5_files: list = field(default_factory=list)
    selected_h5: int = 0
    active_h5_field: str = "list"

    # ── Processing output ──────────────────────────────────────────────────────
    npz_out_path: Path | None = None
    progress_msg: str = ""
    error_msg: str    = ""
    _msg_queue: queue.Queue = field(default_factory=queue.Queue)

    def refresh_h5_files(self) -> None:
        found: list[Path] = []
        for root in CALIB_H5_ROOTS:
            if root.exists():
                found.extend(sorted(root.rglob("*.h5")))
        self.h5_files = found
        self.selected_h5 = min(self.selected_h5, max(0, len(self.h5_files) - 1))

    def poll(self) -> None:
        while True:
            try:
                kind, payload = self._msg_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.progress_msg = payload
            elif kind == "done":
                self.npz_out_path = payload
                self.phase = "done"
                # Automatically start calibration with the processed file
                self.start_auto_calibration()
            elif kind == "error":
                self.error_msg = payload
                self.phase = "error"
            elif kind == "calib_done":
                self.calib_N, self.calib_theta, self.calib_B, self.calib_path = payload
                self.phase = "calib_done"
            elif kind == "calib_error":
                self.error_msg = payload
                self.phase = "calib_error"

    def start_session(self) -> None:
        import numpy as _np
        actual_seed = self.seed if self.seed != 0 else int(time.time() * 1000) % (2**31)
        rng = _np.random.default_rng(actual_seed)
        alphabet = COLLECT_ALPHABETS[self.alphabet_name]
        pool = _np.array(alphabet, dtype=_np.int64)
        self.cue_seq = pool[rng.integers(0, len(pool), size=self.n_trials)].tolist()
        self.trial_records = []
        self.current_trial = 0
        self.display_sym_idx = -2
        self.session_start_ns = time.monotonic_ns()
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.json_out_path = _SCRIPT_DIR.parent / "data" / "cue_logs" / f"cues_{ts}.json"
        self.phase = "countdown"
        self._phase_start_ns = time.monotonic_ns()

    def start_processing(self, json_path: Path, h5_path: Path) -> None:
        self.phase = "processing"
        self.progress_msg = "Starting..."
        threading.Thread(
            target=_process_h5_worker,
            args=(self, json_path, h5_path),
            daemon=True,
        ).start()

    def start_auto_calibration(self) -> None:
        self.phase = "calibrating"
        self.progress_msg = "Starting calibration..."
        threading.Thread(
            target=_auto_calib_worker,
            args=(self,),
            daemon=True,
        ).start()


def random_prompt() -> str:
    return random.choice(INPUT_NAMES)


def normalize_action(raw: str) -> str | None:
    cleaned = raw.strip().upper().replace("-", "_").replace(" ", "_")
    if cleaned in INPUT_NAMES:
        return cleaned
    squashed = cleaned.replace("_", "")
    if squashed in ALIAS_MAP:
        return ALIAS_MAP[squashed]
    if squashed in BUTTON_INPUTS:
        return squashed
    return None


def _calib_worker(calib: CalibState) -> None:
    """Background thread: runs sweep_n and posts progress to calib._msg_queue."""
    q = calib._msg_queue
    try:
        q.put(("progress", "Importing calibration modules..."))
        try:
            import sys as _sys
            import numpy as _np
            import yaml as _yaml
            _sys.path.insert(0, str(Path(__file__).parent))
            from run_calibration import load_trials, build_factory
            from penn_nh.eval.sweep_n import sweep_n
        except ImportError as exc:
            q.put(("error", f"Import failed: {exc}\nEnsure penn_nh is installed (pip install -e .)."))
            return

        if not calib.npz_files:
            q.put(("error", "No .npz files found. Check CALIB_DATA_ROOTS."))
            return

        path = calib.npz_files[calib.selected_file]
        q.put(("progress", f"Loading {path.name}..."))
        data = load_trials(path)
        X, stick = data["raw"], data["stick_xy"]
        M = X.shape[0]
        split = int(M * 0.7)

        arch = ARCH_CHOICES[calib.arch_idx]
        q.put(("progress", f"Loaded {M} trials  •  arch: {arch}  •  running sweep_n..."))
        factory = build_factory(arch)
        result = sweep_n(
            X_train=X[:split], stick_xy_train=stick[:split],
            X_val=X[split:],   stick_xy_val=stick[split:],
            factory=factory,
            n_grid=_np.array([6, 8, 10, 12, 16, 20, 24, 32], dtype=int),
            theta_grid=_np.linspace(0.0, 0.5, 26),
        )

        best = int(_np.argmax(result["B"]))
        N     = int(result["N"][best])
        theta = float(result["theta"][best])
        B     = float(result["B"][best])
        q.put(("progress", f"Sweep done — N={N}  θ={theta:.3f}  B={B:.3f} bits/s"))

        out_path = CALIB_CONFIG_DIR / f"{calib.player_name or 'player'}_config.yaml"
        cfg = {
            "player": calib.player_name or "player",
            "n": N, "theta": theta, "alpha": 0.5, "tau": 0.1,
            "window_w0_ms": 0.0, "window_w1_ms": 100.0,
            "stage1_arch": arch,
            "feature_config": {
                "fs_hz": 32000.0, "bandpass_low_hz": 300.0,
                "bandpass_high_hz": 3000.0, "bin_ms": 5.0,
                "spike_threshold_k": 3.0, "min_isi_ms": 1.0,
                "use_spike_counts": True, "use_spike_power": True,
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as _f:
            _yaml.safe_dump(cfg, _f, sort_keys=False)

        q.put(("done", (N, theta, B, str(out_path))))
    except Exception:
        import traceback as _tb
        q.put(("error", _tb.format_exc()))




def _finish_cueing(collect: CollectState) -> None:
    """Save JSON cue log and move to h5_select phase."""
    import json as _json
    from run_cues import build_log as _build_log
    alphabet = COLLECT_ALPHABETS[collect.alphabet_name]
    log = _build_log(
        session_start_ns=collect.session_start_ns,
        sync_prompt_ns=collect.sync_prompt_ns,
        sync_release_ns=collect.sync_release_ns,
        trial_records=collect.trial_records,
        seed=collect.seed,
        n_trials=collect.n_trials,
        isi_s=collect.isi_s,
        alphabet_name=collect.alphabet_name,
        alphabet_indices=alphabet,
    )
    collect.json_out_path.parent.mkdir(parents=True, exist_ok=True)
    with collect.json_out_path.open("w") as _f:
        _json.dump(log, _f, indent=2)
    collect.refresh_h5_files()
    collect.phase = "h5_select"


def update_collect_timing(collect: CollectState) -> None:
    """Call once per frame to advance timing-based collection phases."""
    now = time.monotonic_ns()

    if collect.phase == "countdown":
        elapsed = now - collect._phase_start_ns
        if elapsed >= 3_000_000_000:
            collect.phase = "sync_prompt"
            collect._phase_start_ns = now
            collect.sync_prompt_ns = now

    elif collect.phase == "sync_prompt":
        if now - collect._phase_start_ns >= 200_000_000:   # 200 ms display
            collect.phase = "sync_wait"
            collect._next_change_ns = now + 2_000_000_000  # 2-second hold

    elif collect.phase == "sync_wait":
        if now >= collect._next_change_ns:
            collect.sync_release_ns = now
            collect.phase = "sync_release"
            collect._next_change_ns = now + 1_000_000_000  # 1-second release

    elif collect.phase == "sync_release":
        if now >= collect._next_change_ns:
            collect.phase = "cueing"
            collect.display_sym_idx = -2
            collect._next_change_ns = 0  # trigger first cue immediately

    elif collect.phase == "cueing":
        if now < collect._next_change_ns:
            return
        if collect.display_sym_idx == -1:
            # Flash gap over — show next cue
            _show_next_cue(collect, now)
        else:
            # Current cue ISI elapsed — check if done or advance
            if collect.current_trial >= collect.n_trials:
                _finish_cueing(collect)
                return
            prev = collect.display_sym_idx
            nxt  = collect.cue_seq[collect.current_trial]
            if prev >= 0 and nxt == prev:
                collect.display_sym_idx = -1
                collect._next_change_ns = now + 100_000_000  # 100 ms flash
            else:
                _show_next_cue(collect, now)


def _show_next_cue(collect: CollectState, now_ns: int) -> None:
    if collect.current_trial >= collect.n_trials:
        _finish_cueing(collect)
        return
    sym_idx = collect.cue_seq[collect.current_trial]
    cue_t = time.monotonic_ns()
    collect.trial_records.append({
        "idx": collect.current_trial,
        "symbol_idx": int(sym_idx),
        "t_mono_ns": cue_t,
    })
    collect.display_sym_idx = int(sym_idx)
    collect.current_trial += 1
    collect._next_change_ns = cue_t + int(collect.isi_s * 1e9)


def _process_h5_worker(collect: CollectState, json_path: Path, h5_path: Path) -> None:
    """Background thread: run load_h5_session alignment + extraction → .npz."""
    q = collect._msg_queue
    try:
        q.put(("progress", "Importing processing modules..."))
        try:
            import sys as _sys
            import numpy as _np
            _sys.path.insert(0, str(_SCRIPT_DIR))
            from load_h5_session import align_and_build_session, extract_windows
        except ImportError as exc:
            q.put(("error", f"Import failed: {exc}\nEnsure h5py is installed."))
            return

        q.put(("progress", f"Aligning {h5_path.name} with cue log..."))
        sess = align_and_build_session(h5_path, json_path)
        q.put(("progress",
               f"Loaded  fs={sess.fs_hz:.0f} Hz  shape={sess.neural.shape}  "
               f"n_cues={len(sess.cue_times_ns)}"))

        q.put(("progress", "Extracting trial windows..."))
        X, y, kept = extract_windows(sess, w_post_ms=400.0, onset_offset_ms=150.0)
        q.put(("progress", f"Extracted {X.shape[0]}/{len(sess.cue_times_ns)} trials"))

        ts = time.strftime("%Y%m%d_%H%M%S")
        out = _SCRIPT_DIR.parent / "data" / "processed" / f"session_{ts}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        _np.savez(out, raw=X, stick_xy=y, fs_hz=sess.fs_hz,
                  clock_offset_ns=sess.clock_offset_ns)
        q.put(("done", out))
    except Exception:
        import traceback as _tb
        q.put(("error", _tb.format_exc()))



def _auto_calib_worker(collect: "CollectState") -> None:
    """Background thread: run calibration on collect.npz_out_path, post results to collect._msg_queue."""
    q = collect._msg_queue
    try:
        q.put(("progress", "Importing calibration modules..."))
        try:
            import sys as _sys
            import numpy as _np
            import yaml as _yaml
            _sys.path.insert(0, str(Path(__file__).parent))
            from run_calibration import load_trials, build_factory
            from penn_nh.eval.sweep_n import sweep_n
        except ImportError as exc:
            q.put(("calib_error", f"Import failed: {exc}\nEnsure penn_nh is installed."))
            return

        path = collect.npz_out_path
        if path is None or not path.exists():
            q.put(("calib_error", f"Processed .npz not found: {path}"))
            return

        q.put(("progress", f"Loading {path.name}..."))
        data = load_trials(path)
        X, stick = data["raw"], data["stick_xy"]
        M = X.shape[0]
        split = int(M * 0.7)

        arch = ARCH_CHOICES[collect.calib_arch_idx]
        q.put(("progress", f"Loaded {M} trials  •  arch: {arch}  •  running sweep..."))
        factory = build_factory(arch)
        result = sweep_n(
            X_train=X[:split], stick_xy_train=stick[:split],
            X_val=X[split:],   stick_xy_val=stick[split:],
            factory=factory,
            n_grid=_np.array([6, 8, 10, 12, 16, 20, 24, 32], dtype=int),
            theta_grid=_np.linspace(0.0, 0.5, 26),
        )
        best = int(_np.argmax(result["B"]))
        N     = int(result["N"][best])
        theta = float(result["theta"][best])
        B     = float(result["B"][best])
        q.put(("progress", f"Sweep done — N={N}  theta={theta:.3f}  B={B:.3f} bits/s"))

        out_path = CALIB_CONFIG_DIR / "session_config.yaml"
        cfg = {
            "n": N, "theta": theta, "alpha": 0.5, "tau": 0.1,
            "window_w0_ms": 0.0, "window_w1_ms": 100.0,
            "stage1_arch": arch,
            "feature_config": {
                "fs_hz": 32000.0, "bandpass_low_hz": 300.0,
                "bandpass_high_hz": 3000.0, "bin_ms": 5.0,
                "spike_threshold_k": 3.0, "min_isi_ms": 1.0,
                "use_spike_counts": True, "use_spike_power": True,
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as _f:
            _yaml.safe_dump(cfg, _f, sort_keys=False)
        q.put(("calib_done", (N, theta, B, str(out_path))))
    except Exception:
        import traceback as _tb
        q.put(("calib_error", _tb.format_exc()))


class UDPInputReceiver(threading.Thread):
    def __init__(self, out_queue: queue.Queue[str], port: int = UDP_PORT):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.port = port
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.25)

        try:
            while not self._stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break

                action = normalize_action(data.decode("utf-8", errors="ignore"))
                if action is not None:
                    self.out_queue.put(action)
        finally:
            sock.close()


class SynapseInputReceiver(threading.Thread):
    """Background thread: reads decoded_controls Tensor tap → action strings → input_queue."""

    def __init__(
        self,
        out_queue: "queue.Queue[str]",
        uri: str = SYNAPSE_URI_DEFAULT,
        tap_name: str = SYNAPSE_TAP_DEFAULT,
    ):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.uri = uri
        self.tap_name = tap_name
        self._stop_event = threading.Event()
        self.connected: bool = False
        self.error: str | None = None   # set if connection failed

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            import numpy as _np
            import synapse as syn
            from synapse.client.taps import Tap
            from synapse.api.datatype_pb2 import Tensor
        except ImportError as exc:
            self.error = f"synapse not installed: {exc}"
            return

        try:
            dev = syn.Device(self.uri)
            if dev.info() is None:
                self.error = f"Cannot reach Synapse at {self.uri}"
                return
        except Exception as exc:
            self.error = str(exc)
            return

        tap = Tap(self.uri)
        if not tap.connect(self.tap_name):
            self.error = f"Tap connect failed: {self.tap_name!r}"
            return

        self.connected = True
        try:
            for raw_msg in tap.stream(timeout_ms=100):
                if self._stop_event.is_set():
                    break
                if not raw_msg:
                    continue
                tensor = Tensor()
                tensor.ParseFromString(raw_msg)

                arr = _np.frombuffer(tensor.data, dtype=_np.float32).copy()
                if tensor.shape:
                    expected = 1
                    for d in tensor.shape:
                        expected *= d
                    if expected == arr.size:
                        arr = arr.reshape(tuple(tensor.shape))
                arr = arr.flatten()
                if arr.size == 0:
                    continue

                if arr.size == 1:
                    # Scalar decoded index
                    idx = int(arr[0])
                else:
                    # Probability / logit vector — pick argmax
                    idx = int(_np.argmax(arr))

                if 0 <= idx < len(_DECODE_IDX_TO_ACTION):
                    self.out_queue.put(_DECODE_IDX_TO_ACTION[idx])
        except Exception:
            pass
        finally:
            try:
                tap.disconnect()
            except Exception:
                pass



def load_controller_image() -> pygame.Surface | None:
    if os.path.exists(CONTROLLER_IMAGE_PATH):
        try:
            return pygame.image.load(CONTROLLER_IMAGE_PATH).convert_alpha()
        except pygame.error:
            return None
    return None


def load_judge_face() -> pygame.Surface | None:
    if os.path.exists(JUDGE_FACE_PATH):
        try:
            return pygame.image.load(JUDGE_FACE_PATH).convert_alpha()
        except pygame.error:
            return None
    return None


def fit_rect(src_w: int, src_h: int, dst_rect: pygame.Rect, padding: int = 0) -> pygame.Rect:
    inner = dst_rect.inflate(-padding * 2, -padding * 2)
    if src_w <= 0 or src_h <= 0 or inner.w <= 0 or inner.h <= 0:
        return inner
    scale = min(inner.w / src_w, inner.h / src_h)
    w = max(1, int(src_w * scale))
    h = max(1, int(src_h * scale))
    return pygame.Rect(inner.centerx - w // 2, inner.centery - h // 2, w, h)


def draw_text(surface: pygame.Surface, text: str, font: pygame.font.Font, color: tuple[int, int, int], pos: tuple[int, int], *, center: bool = False) -> None:
    img = font.render(text, True, color)
    rect = img.get_rect(center=pos) if center else img.get_rect(topleft=pos)
    surface.blit(img, rect)


def draw_round_rect(surface: pygame.Surface, rect: pygame.Rect, color: tuple[int, int, int], radius: int = 18, width: int = 0) -> None:
    pygame.draw.rect(surface, color, rect, width=width, border_radius=radius)


def make_music_notes() -> list[FloatingNote]:
    notes = []
    for _ in range(10):
        notes.append(
            FloatingNote(
                x=random.uniform(60, LEFT_PANEL_W - 120),
                y=random.uniform(120, HEIGHT - 160),
                speed=random.uniform(20, 50),
                phase=random.uniform(0, math.tau),
                size=random.uniform(18, 34),
            )
        )
    return notes


def update_notes(notes: list[FloatingNote], dt: float) -> None:
    for note in notes:
        note.y -= note.speed * dt
        note.x += math.sin(time.time() * 1.8 + note.phase) * 12 * dt
        if note.y < 80:
            note.y = HEIGHT - random.uniform(20, 120)
            note.x = random.uniform(50, LEFT_PANEL_W - 120)


def draw_note(surface: pygame.Surface, x: float, y: float, size: float) -> None:
    stem_h = size * 1.5
    head_r = size * 0.28
    pygame.draw.line(surface, NOTE, (x, y - stem_h), (x, y), 4)
    pygame.draw.line(surface, NOTE, (x, y - stem_h), (x + size * 0.55, y - stem_h + size * 0.15), 4)
    pygame.draw.ellipse(surface, NOTE, (x - head_r, y - head_r, head_r * 1.7, head_r * 1.3))


def draw_arrow(
    surface: pygame.Surface,
    color: tuple,
    start: tuple[float, float],
    end: tuple[float, float],
    width: int = 2,
    head_size: int = 10,
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.sqrt(dx * dx + dy * dy) < 4:
        return
    s = (int(start[0]), int(start[1]))
    e = (int(end[0]), int(end[1]))
    pygame.draw.line(surface, color, s, e, width)
    angle = math.atan2(dy, dx)
    span = math.radians(28)
    p1 = (int(end[0] - head_size * math.cos(angle - span)), int(end[1] - head_size * math.sin(angle - span)))
    p2 = (int(end[0] - head_size * math.cos(angle + span)), int(end[1] - head_size * math.sin(angle + span)))
    pygame.draw.polygon(surface, color, [e, p1, p2])


def draw_disco_ball(surface: pygame.Surface, t: float) -> None:
    cx = LEFT_PANEL_W // 2 - 10
    cy = 72
    r = 68
    tile = 9
    palette = [
        (255, 60, 60), (60, 255, 110), (60, 110, 255),
        (255, 240, 50), (255, 55, 220), (55, 225, 255),
        (255, 150, 50), (150, 255, 60), (255, 120, 200),
        (120, 200, 255),
    ]
    spin = t * 1.1

    # Wire
    pygame.draw.line(surface, (180, 170, 155), (cx, 0), (cx, cy - r), 4)
    # Sphere base
    pygame.draw.circle(surface, (18, 18, 28), (cx, cy), r)
    # Mirror tiles
    for row in range(-8, 9):
        for col in range(-8, 9):
            nx, ny = col * tile, row * tile
            if nx * nx + ny * ny < (r - 4) ** 2:
                dist_frac = math.sqrt(nx * nx + ny * ny) / r
                brightness = max(0.12, 1.0 - dist_frac * 0.7)
                ci = (col * 3 + row * 2 + int(spin * 5)) % len(palette)
                c = tuple(max(0, min(255, int(v * brightness))) for v in palette[ci])
                pygame.draw.rect(surface, c, pygame.Rect(cx + nx - tile // 2 + 1, cy + ny - tile // 2 + 1, tile - 2, tile - 2))
    # Glints
    pygame.draw.circle(surface, (255, 255, 255), (cx - 18, cy - 22), 8)
    pygame.draw.circle(surface, (220, 220, 255), (cx + 12, cy - 28), 5)
    pygame.draw.circle(surface, (255, 255, 200), (cx - 5,  cy - 34), 3)
    pygame.draw.circle(surface, (18, 18, 28), (cx, cy), r, 3)


def draw_lasers(surface: pygame.Surface, t: float) -> None:
    cx = LEFT_PANEL_W // 2 - 10
    cy = 72
    laser_surf = pygame.Surface((LEFT_PANEL_W, HEIGHT), pygame.SRCALPHA)
    beams = [
        ((255, 20,  70),  t * 38,  True),
        ((20,  255, 110), t * 52,  True),
        ((70,  110, 255), t * 31,  True),
        ((255, 220, 20),  t * 44,  True),
        ((255, 30,  220), t * 35,  True),
        ((20,  220, 255), t * 58,  True),
        ((255, 130, 30),  t * 27,  False),
        ((130, 30,  255), t * 63,  False),
        ((30,  255, 200), t * 41,  False),
        ((255, 255, 80),  t * 49,  False),
        ((255, 80,  150), t * 33,  False),
        ((80,  200, 255), t * 55,  False),
        ((200, 255, 100), t * 22,  False),
        ((255, 100, 255), t * 67,  False),
    ]
    for color, angle_deg, primary in beams:
        angle = math.radians(angle_deg)
        ex = int(cx + math.cos(angle) * 1600)
        ey = int(cy + math.sin(angle) * 1600)
        if primary:
            pygame.draw.line(laser_surf, (*color, 35), (cx, cy), (ex, ey), 18)
            pygame.draw.line(laser_surf, (*color, 80), (cx, cy), (ex, ey), 6)
            pygame.draw.line(laser_surf, (*color, 200), (cx, cy), (ex, ey), 2)
        else:
            pygame.draw.line(laser_surf, (*color, 25), (cx, cy), (ex, ey), 10)
            pygame.draw.line(laser_surf, (*color, 130), (cx, cy), (ex, ey), 2)
    surface.blit(laser_surf, (0, 0))


def draw_dancer_vectors(surface: pygame.Surface, t: float) -> None:
    cx = LEFT_PANEL_W // 2 - 10
    floor_y = HEIGHT - 110
    SCALE = 1 / 18  # px/s → display px, capped at 80

    joints = [
        # (px, py, vx, vy, color)
        (
            cx - 150 + math.sin(t * 3.2) * 36,
            265 + math.sin(t * 6.0) * 30,
            math.cos(t * 3.2) * 36 * 3.2,
            math.cos(t * 6.0) * 30 * 6.0,
            (255, 200, 50, 210),
        ),
        (
            cx + 145 + math.cos(t * 3.4) * 28,
            255 + math.sin(t * 5.4 + 1.0) * 42,
            -math.sin(t * 3.4) * 28 * 3.4,
            math.cos(t * 5.4 + 1.0) * 42 * 5.4,
            (100, 220, 255, 210),
        ),
        (
            cx - 92 + math.sin(t * 5.0) * 22,
            floor_y,
            math.cos(t * 5.0) * 22 * 5.0,
            0.0,
            (255, 120, 80, 210),
        ),
        (
            cx + 95 + math.cos(t * 4.8) * 18,
            floor_y,
            -math.sin(t * 4.8) * 18 * 4.8,
            0.0,
            (150, 255, 120, 210),
        ),
        (
            cx - 65 + math.sin(t * 3.6 + 0.9) * 28,
            545 + math.sin(t * 4.2) * 12,
            math.cos(t * 3.6 + 0.9) * 28 * 3.6,
            math.cos(t * 4.2) * 12 * 4.2,
            (220, 100, 255, 210),
        ),
        (
            cx + 58 + math.cos(t * 3.9 + 0.4) * 24,
            540 + math.cos(t * 4.0) * 12,
            -math.sin(t * 3.9 + 0.4) * 24 * 3.9,
            -math.sin(t * 4.0) * 12 * 4.0,
            (80, 255, 200, 210),
        ),
    ]

    vec_surf = pygame.Surface((LEFT_PANEL_W, HEIGHT), pygame.SRCALPHA)
    for px, py, vx, vy, color in joints:
        mag = math.sqrt(vx * vx + vy * vy)
        if mag < 1:
            continue
        disp_len = min(80, mag * SCALE)
        ex = px + (vx / mag) * disp_len
        ey = py + (vy / mag) * disp_len
        draw_arrow(vec_surf, color, (px, py), (ex, ey), width=2, head_size=9)
    surface.blit(vec_surf, (0, 0))


def spawn_sardines(sardines: list[Sardine], count: int = 2) -> None:
    dancer_cx = LEFT_PANEL_W // 2 - 10
    for _ in range(count):
        from_left = random.random() < 0.5
        sx = random.uniform(10, 60) if from_left else random.uniform(LEFT_PANEL_W - 60, LEFT_PANEL_W - 10)
        sy = random.uniform(160, 420)
        target_x = dancer_cx + random.uniform(-40, 40)
        target_y = random.uniform(220, 380)
        dx, dy = target_x - sx, target_y - sy
        dist = math.sqrt(dx * dx + dy * dy)
        speed = random.uniform(500, 750)
        vx, vy = dx / dist * speed, dy / dist * speed
        sardines.append(Sardine(x=sx, y=sy, vx=vx, vy=vy))


def update_sardines(sardines: list[Sardine], dt: float) -> None:
    for s in sardines:
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.vy += 300 * dt  # gravity arc
        s.rotation = math.degrees(math.atan2(-s.vy, s.vx))
        s.age += dt
    sardines[:] = [s for s in sardines if s.age < s.max_age]


def draw_sardine(surface: pygame.Surface, s: Sardine) -> None:
    alpha = max(0, int(255 * (1.0 - s.age / s.max_age)))
    length, body_w = 80, 26
    pad = 28
    fish = pygame.Surface((length + pad * 2, body_w * 2 + pad), pygame.SRCALPHA)
    cx, cy = length // 2 + pad, body_w + pad // 2

    # Body
    pygame.draw.ellipse(fish, (118, 158, 172, alpha), (cx - length // 2, cy - body_w // 2, length, body_w))
    # Dorsal sheen
    pygame.draw.ellipse(fish, (80, 120, 145, alpha), (cx - length // 2, cy - body_w // 2, length, body_w // 2))
    # Belly highlight
    pygame.draw.ellipse(fish, (210, 230, 235, alpha // 2), (cx - length // 3, cy - body_w // 5, length // 2, body_w // 3))
    # Tail fork
    tail_x = cx - length // 2
    pygame.draw.polygon(fish, (95, 135, 150, alpha), [
        (tail_x + 2, cy), (tail_x - 20, cy - 16), (tail_x - 20, cy + 16),
    ])
    # Side stripe
    pygame.draw.line(fish, (180, 210, 220, alpha), (cx - length // 2 + 8, cy), (cx + length // 4, cy), 3)
    # Eye
    pygame.draw.circle(fish, (15, 15, 25, alpha), (cx + length // 4, cy - 3), 5)
    pygame.draw.circle(fish, (255, 255, 255, alpha), (cx + length // 4 + 2, cy - 5), 2)

    rotated = pygame.transform.rotate(fish, s.rotation)
    rect = rotated.get_rect(center=(int(s.x), int(s.y)))
    surface.blit(rotated, rect)


def draw_capsule(
    surface: pygame.Surface,
    color: tuple,
    p1: tuple[float, float],
    p2: tuple[float, float],
    radius: int,
) -> None:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    px, py = -dy / length * radius, dx / length * radius
    pts = [
        (int(p1[0] + px), int(p1[1] + py)),
        (int(p2[0] + px), int(p2[1] + py)),
        (int(p2[0] - px), int(p2[1] - py)),
        (int(p1[0] - px), int(p1[1] - py)),
    ]
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.circle(surface, color, (int(p1[0]), int(p1[1])), radius)
    pygame.draw.circle(surface, color, (int(p2[0]), int(p2[1])), radius)


_LIGHT = (-0.707, -0.707)  # upper-left light source for 3-D shading


def draw_capsule_3d(
    surface: pygame.Surface,
    color: tuple,
    p1: tuple[float, float],
    p2: tuple[float, float],
    radius: int,
) -> None:
    """Capsule with cylindrical 3-D shading: shadow base → main fill → highlight stripe."""
    shadow = tuple(max(0, c - 28) for c in color)
    draw_capsule(surface, shadow, p1, p2, radius)

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    nx, ny = -dy / length, dx / length
    dot = nx * _LIGHT[0] + ny * _LIGHT[1]
    sign = 1 if dot >= 0 else -1
    ox = nx * sign * radius * 0.24
    oy = ny * sign * radius * 0.24

    # Main fill, offset slightly toward light
    draw_capsule(surface, color,
                 (p1[0] + ox * 0.5, p1[1] + oy * 0.5),
                 (p2[0] + ox * 0.5, p2[1] + oy * 0.5),
                 int(radius * 0.86))

    # Highlight stripe on lit face
    bright = tuple(min(255, c + 60) for c in color)
    draw_capsule(surface, bright,
                 (p1[0] + ox * 1.9, p1[1] + oy * 1.9),
                 (p2[0] + ox * 1.9, p2[1] + oy * 1.9),
                 max(2, int(radius * 0.32)))


def draw_joint_3d(
    surface: pygame.Surface,
    center: tuple[float, float],
    radius: int,
    color: tuple,
) -> None:
    """Sphere-shaded ball joint: shadow rim → main → highlight → specular."""
    cx, cy = int(center[0]), int(center[1])
    shadow = tuple(max(0, c - 28) for c in color)
    pygame.draw.circle(surface, shadow, (cx, cy), radius)
    pygame.draw.circle(surface, color,  (cx, cy), int(radius * 0.88))
    bright = tuple(min(255, c + 65) for c in color)
    hx = int(cx - radius * 0.28)
    hy = int(cy - radius * 0.30)
    pygame.draw.circle(surface, bright, (hx, hy), max(2, int(radius * 0.46)))
    pygame.draw.circle(surface, (245, 245, 255), (hx, hy), max(1, int(radius * 0.20)))


def draw_dancer(surface: pygame.Surface, t: float, judge_face: pygame.Surface | None = None, *, frown: bool = False, scale: float = 1.0) -> None:
    cx = LEFT_PANEL_W // 2 - 10
    floor_y = HEIGHT - 110
    bounce = math.sin(t * 4.0) * 12

    # ── Primary joints ──────────────────────────────────────────
    head_y = 180 + bounce
    neck_y = 235 + bounce
    hip_y  = 405 + bounce

    left_hand  = (cx - 150 + math.sin(t * 3.2) * 44,  265 + math.sin(t * 6.0) * 38)
    right_hand = (cx + 145 + math.cos(t * 3.4) * 36,  255 + math.sin(t * 5.4 + 1.0) * 50)
    left_knee  = (cx - 65  + math.sin(t * 3.6 + 0.9) * 34, 545 + math.sin(t * 4.2) * 16)
    right_knee = (cx + 58  + math.cos(t * 3.9 + 0.4) * 30, 540 + math.cos(t * 4.0) * 16)
    left_foot  = (cx - 92  + math.sin(t * 5.0) * 28,  floor_y)
    right_foot = (cx + 95  + math.cos(t * 4.8) * 24,  floor_y)

    # ── Derived joints ───────────────────────────────────────────
    l_shoulder = (cx - 76, neck_y + 26)
    r_shoulder = (cx + 72, neck_y + 26)
    l_hip_pt   = (cx - 42, hip_y)
    r_hip_pt   = (cx + 40, hip_y)

    l_elbow = (
        (l_shoulder[0] + left_hand[0])  / 2 + math.sin(t * 2.8) * 26,
        (l_shoulder[1] + left_hand[1])  / 2 + 20,
    )
    r_elbow = (
        (r_shoulder[0] + right_hand[0]) / 2 + math.cos(t * 3.1) * 22,
        (r_shoulder[1] + right_hand[1]) / 2 + 20,
    )
    l_wrist = (
        l_elbow[0] * 0.28 + left_hand[0]  * 0.72 + math.sin(t * 7.3) * 10,
        l_elbow[1] * 0.28 + left_hand[1]  * 0.72 + math.cos(t * 6.9) *  7,
    )
    r_wrist = (
        r_elbow[0] * 0.28 + right_hand[0] * 0.72 + math.cos(t * 6.8) *  9,
        r_elbow[1] * 0.28 + right_hand[1] * 0.72 + math.sin(t * 7.5) *  6,
    )
    l_ankle = (
        left_knee[0]  * 0.18 + left_foot[0]  * 0.82 + math.sin(t * 5.3) * 8,
        left_knee[1]  * 0.18 + left_foot[1]  * 0.82,
    )
    r_ankle = (
        right_knee[0] * 0.18 + right_foot[0] * 0.82 + math.cos(t * 4.9) * 7,
        right_knee[1] * 0.18 + right_foot[1] * 0.82,
    )

    # ── Colors ───────────────────────────────────────────────────
    BODY   = (38, 35, 52)
    FRONT  = (62, 58, 84)
    SKIN   = (198, 158, 115)
    SHOE   = (25, 22, 35)
    SHOE_F = (42, 38, 58)

    # ── Floor shadow — always at full screen coords ───────────────
    pygame.draw.ellipse(surface, (225, 214, 192), (60, floor_y - 18, LEFT_PANEL_W - 120, 64))

    # ── Route drawing to a temp surface when scaling ──────────────
    # After drawing, it is scaled and blitted anchored at (cx, floor_y)
    # so the dancer's feet always stay on the floor line.
    if scale < 0.995:
        dst = pygame.Surface((LEFT_PANEL_W, HEIGHT), pygame.SRCALPHA)
    else:
        dst = surface

    # ── Draw order: back → torso → front → head ──────────────────

    # Back arm (right)
    draw_capsule_3d(dst, BODY, r_shoulder, r_elbow,    14)
    draw_capsule_3d(dst, BODY, r_elbow,    r_wrist,    11)
    draw_capsule_3d(dst, BODY, r_wrist,    right_hand, 8)
    draw_joint_3d(dst, r_shoulder, 15, BODY)
    draw_joint_3d(dst, r_elbow,    13, BODY)
    draw_joint_3d(dst, r_wrist,    10, BODY)
    draw_joint_3d(dst, right_hand, 13, SKIN)

    # Back leg (right)
    draw_capsule_3d(dst, BODY, r_hip_pt,   right_knee, 19)
    draw_capsule_3d(dst, BODY, right_knee, r_ankle,    15)
    draw_capsule_3d(dst, BODY, r_ankle,    right_foot, 11)
    draw_joint_3d(dst, r_hip_pt,   18, BODY)
    draw_joint_3d(dst, right_knee, 17, BODY)
    draw_joint_3d(dst, r_ankle,    13, BODY)
    pygame.draw.ellipse(dst, SHOE, (int(right_foot[0]) - 26, int(right_foot[1]) - 10, 44, 20))

    # ── Torso ────────────────────────────────────────────────────
    waist_y = int((neck_y + hip_y) / 2) + 10
    torso_pts = [
        (int(l_shoulder[0]),  int(l_shoulder[1])),
        (int(r_shoulder[0]),  int(r_shoulder[1])),
        (cx + 34,             waist_y),
        (int(r_hip_pt[0]),    int(r_hip_pt[1])),
        (int(l_hip_pt[0]),    int(l_hip_pt[1])),
        (cx - 34,             waist_y),
    ]
    pygame.draw.polygon(dst, BODY, torso_pts)
    # Chest highlight plane (lit face of torso)
    chest_pts = [
        (int(l_shoulder[0]) + 12, int(l_shoulder[1]) + 5),
        (int(r_shoulder[0]) - 12, int(r_shoulder[1]) + 5),
        (cx + 16,  waist_y - 8),
        (cx - 16,  waist_y - 8),
    ]
    pygame.draw.polygon(dst, FRONT, chest_pts)

    # Neck
    draw_capsule_3d(dst, BODY, (cx, int(neck_y)), (cx, int(neck_y + 28)), 14)

    # Front arm (left)
    draw_capsule_3d(dst, FRONT, l_shoulder, l_elbow,   14)
    draw_capsule_3d(dst, FRONT, l_elbow,    l_wrist,   11)
    draw_capsule_3d(dst, FRONT, l_wrist,    left_hand,  8)
    draw_joint_3d(dst, l_shoulder, 15, FRONT)
    draw_joint_3d(dst, l_elbow,    13, FRONT)
    draw_joint_3d(dst, l_wrist,    10, FRONT)
    draw_joint_3d(dst, left_hand,  13, SKIN)

    # Front leg (left)
    draw_capsule_3d(dst, FRONT, l_hip_pt,  left_knee, 19)
    draw_capsule_3d(dst, FRONT, left_knee, l_ankle,   15)
    draw_capsule_3d(dst, FRONT, l_ankle,   left_foot, 11)
    draw_joint_3d(dst, l_hip_pt,  18, FRONT)
    draw_joint_3d(dst, left_knee, 17, FRONT)
    draw_joint_3d(dst, l_ankle,   13, FRONT)
    pygame.draw.ellipse(dst, SHOE_F, (int(left_foot[0]) - 26, int(left_foot[1]) - 10, 44, 20))

    # ── Head ─────────────────────────────────────────────────────
    head_center = (cx, int(head_y))
    draw_joint_3d(dst, head_center, 54, BODY)
    # Ears
    draw_joint_3d(dst, (head_center[0] - 52, head_center[1] + 4), 12, SKIN)
    draw_joint_3d(dst, (head_center[0] + 52, head_center[1] + 4), 12, SKIN)
    # Face
    pygame.draw.circle(dst, SKIN, head_center, 46)
    # Eyes
    for ex, ey in [(head_center[0] - 16, head_center[1] - 8), (head_center[0] + 16, head_center[1] - 8)]:
        pygame.draw.circle(dst, (255, 255, 255), (ex, ey), 10)
        pygame.draw.circle(dst, (35, 25, 15),    (ex, ey),  6)
        pygame.draw.circle(dst, (255, 255, 255), (ex + 2, ey - 3), 2)
    # Mouth
    mouth_start, mouth_end = (0, math.pi) if frown else (math.pi, 2 * math.pi)
    pygame.draw.arc(dst, (155, 75, 65),
                    pygame.Rect(head_center[0] - 15, head_center[1] + 10, 30, 16),
                    mouth_start, mouth_end, 3)

    if judge_face is not None:
        face = pygame.transform.smoothscale(judge_face, (86, 86))
        face_rect = face.get_rect(center=head_center)
        clip = pygame.Surface((86, 86), pygame.SRCALPHA)
        pygame.draw.circle(clip, (255, 255, 255), (43, 43), 43)
        clip.blit(face, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        dst.blit(clip, face_rect)

    # ── Scale and composite when not at full size ─────────────────
    if scale < 0.995:
        new_w = max(1, int(LEFT_PANEL_W * scale))
        new_h = max(1, int(HEIGHT * scale))
        scaled_dst = pygame.transform.smoothscale(dst, (new_w, new_h))
        # Anchor at (cx, floor_y): blit so scaled (cx, floor_y) lands on (cx, floor_y)
        surface.blit(scaled_dst, (int(cx * (1.0 - scale)), int(floor_y * (1.0 - scale))))

def draw_streak_indicator(
    surface: pygame.Surface,
    streak: int,
    t: float,
    font_big: pygame.font.Font,
    font_small: pygame.font.Font,
) -> None:
    if streak < 3:
        return

    pulse = (math.sin(t * 6.0) + 1) / 2  # 0..1

    base_w, base_h = 290, 110
    expand = int(pulse * 14)
    w, h = base_w + expand, base_h + expand
    margin = 28
    x = LEFT_PANEL_W - w - margin
    y = HEIGHT - h - margin
    box = pygame.Rect(x, y, w, h)

    # Background: warm pulsing orange
    r_val = 255
    g_val = int(120 + pulse * 70)
    bg_color = (r_val, g_val, 30)
    draw_round_rect(surface, box, bg_color, radius=16)

    # Pulsing border
    border_alpha = int(180 + pulse * 75)
    border_color = (255, 230, border_alpha)
    pygame.draw.rect(surface, border_color, box, width=3 + int(pulse * 2), border_radius=16)

    # Star icon on the left
    star_cx = box.x + 52
    star_cy = box.centery
    star_outer = int(32 + pulse * 6)
    star_inner = int(star_outer * 0.42)
    star_pts = []
    for i in range(10):
        angle = math.radians(-90 + i * 36)
        r = star_outer if i % 2 == 0 else star_inner
        star_pts.append((star_cx + r * math.cos(angle), star_cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, WHITE, star_pts)

    text_cx = (box.x + 100 + box.right) // 2
    draw_text(surface, "HOT STREAK", font_small, WHITE, (text_cx, box.y + 32), center=True)
    draw_text(surface, f"x{streak}", font_big, WHITE, (text_cx, box.y + 72), center=True)


def draw_status_panel(
    surface: pygame.Surface,
    state: GameState,
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_small: pygame.font.Font,
    syn_connected: bool = False,
    syn_error: bool = False,
) -> pygame.Rect:
    box = pygame.Rect(STATUS_BOX_X, STATUS_BOX_Y, STATUS_BOX_W, STATUS_BOX_H)
    draw_round_rect(surface, box, WHITE, radius=22)
    pygame.draw.rect(surface, (224, 212, 185), box, width=2, border_radius=22)

    # Synapse connection indicator dot
    dot_cx = box.right - 16
    dot_cy = box.y + 16
    dot_color = GREEN if syn_connected else (RED if syn_error else (200, 200, 200))
    pygame.draw.circle(surface, dot_color, (dot_cx, dot_cy), 7)
    pygame.draw.circle(surface, (80, 80, 80), (dot_cx, dot_cy), 7, 1)
    syn_label = "BCI" if syn_connected else ("ERR" if syn_error else "SYN")
    draw_text(surface, syn_label, font_small, dot_color, (dot_cx - 18, dot_cy - 8))

    # Bit rate
    bitrate = state.bitrate(14)
    draw_text(surface, f"Bit rate: {bitrate:.2f} bits/sec", font_big, INK, (box.x + 22, box.y + 14))

    # Countdown timer
    remaining = state.time_remaining()
    if state.trial_ended:
        timer_text = "TRIAL OVER"
        timer_color = RED
    elif state.running:
        timer_text = f"{int(math.ceil(remaining))}s"
        timer_color = BLUE if remaining > 10 else RED
    else:
        timer_text = "PRESS SPACE TO START"
        timer_color = (140, 130, 110)
    draw_text(surface, timer_text, font_med, timer_color, (box.x + 22, box.y + 58))

    # Stats row
    draw_text(surface, f"Seen: {state.total_inputs}", font_small, INK, (box.x + 22, box.y + 118))
    draw_text(surface, f"Wrong: {state.incorrect_inputs}", font_small, INK, (box.x + 110, box.y + 118))

    # Reset button
    reset_btn = pygame.Rect(box.right - 100, box.y + 108, 88, 32)
    draw_round_rect(surface, reset_btn, ACCENT, radius=10)
    draw_text(surface, "RESET", font_small, WHITE, reset_btn.center, center=True)

    return reset_btn


def draw_placeholder_controller(surface: pygame.Surface, rect: pygame.Rect) -> tuple[dict[str, tuple[int, int]], dict[str, pygame.Rect], dict[str, tuple[tuple[int, int], int]]]:
    pad = int(min(rect.w, rect.h) * 0.05)
    body = pygame.Rect(rect.x + pad, rect.y + pad, rect.w - pad * 2, rect.h - pad * 2)

    base_color = (232, 223, 203)
    edge_color = (198, 186, 162)
    detail_color = (176, 164, 142)
    stick_ring_color = (145, 134, 116)
    stick_inner_color = (120, 110, 96)
    button_color = (200, 188, 164)

    # Main rectangular body
    pygame.draw.rect(surface, base_color, body, border_radius=18)
    pygame.draw.rect(surface, edge_color, body, width=3, border_radius=18)

    # LB / RB shoulder buttons — straddle the top edge of the body
    shoulder_w = int(body.w * 0.35)
    shoulder_h = int(body.h * 0.18)
    shoulder_y = body.y - int(shoulder_h * 0.55)  # raised to extend above body

    lb_rect = pygame.Rect(body.x + int(body.w * 0.04), shoulder_y, shoulder_w, shoulder_h)
    rb_rect = pygame.Rect(body.right - shoulder_w - int(body.w * 0.04), shoulder_y, shoulder_w, shoulder_h)
    pygame.draw.rect(surface, detail_color, lb_rect, border_radius=10)
    pygame.draw.rect(surface, detail_color, rb_rect, border_radius=10)
    pygame.draw.rect(surface, edge_color, lb_rect, width=2, border_radius=10)
    pygame.draw.rect(surface, edge_color, rb_rect, width=2, border_radius=10)

    # Content area anchored to body top, leaving margin below bumpers
    content_y = body.y + int(body.h * 0.12)
    content_h = body.h - int(body.h * 0.17)

    # Left joystick
    stick_r = max(18, int(body.w * 0.070))
    left_stick_center = (
        body.x + int(body.w * 0.27),
        content_y + int(content_h * 0.42),
    )
    pygame.draw.circle(surface, detail_color, left_stick_center, stick_r + 6)
    pygame.draw.circle(surface, stick_ring_color, left_stick_center, stick_r)
    pygame.draw.circle(surface, stick_inner_color, left_stick_center, max(7, stick_r - 7))

    # Right joystick
    right_stick_center = (
        body.x + int(body.w * 0.60),
        content_y + int(content_h * 0.72),
    )
    pygame.draw.circle(surface, detail_color, right_stick_center, stick_r + 6)
    pygame.draw.circle(surface, stick_ring_color, right_stick_center, stick_r)
    pygame.draw.circle(surface, stick_inner_color, right_stick_center, max(7, stick_r - 7))

    # XYAB diamond — grey
    face_cx = body.x + int(body.w * 0.79)
    face_cy = content_y + int(content_h * 0.42)
    face_gap = max(20, int(body.w * 0.062))
    face_r = max(14, int(body.w * 0.040))

    diamond_positions = {
        "Y": (face_cx, face_cy - face_gap),
        "A": (face_cx, face_cy + face_gap),
        "X": (face_cx - face_gap, face_cy),
        "B": (face_cx + face_gap, face_cy),
    }
    for pos in diamond_positions.values():
        pygame.draw.circle(surface, button_color, pos, face_r)
        pygame.draw.circle(surface, edge_color, pos, face_r, 2)

    # Build positions dict for overlay alignment
    positions: dict[str, tuple[int, int]] = {}
    positions["LB"] = lb_rect.center
    positions["RB"] = rb_rect.center
    positions.update(diamond_positions)

    d = stick_r + 6
    lx, ly = left_stick_center
    positions["L_UP"]        = (lx,        ly - d)
    positions["L_DOWN"]      = (lx,        ly + d)
    positions["L_LEFT"]      = (lx - d,    ly)
    positions["L_RIGHT"]     = (lx + d,    ly)
    rx, ry = right_stick_center
    positions["R_UP"]        = (rx,        ry - d)
    positions["R_DOWN"]      = (rx,        ry + d)
    positions["R_LEFT"]      = (rx - d,    ry)
    positions["R_RIGHT"]     = (rx + d,    ry)

    shoulder_rects = {"LB": lb_rect, "RB": rb_rect}
    stick_info = {
        "L": (left_stick_center, stick_r + 8),
        "R": (right_stick_center, stick_r + 8),
    }
    return positions, shoulder_rects, stick_info

STICK_DIRECTION_ANGLES = {
    "L_RIGHT": 0, "L_DOWN": 90, "L_LEFT": 180, "L_UP": 270,
    "R_RIGHT": 0, "R_DOWN": 90, "R_LEFT": 180, "R_UP": 270,
}


def _pie_polygon(cx: int, cy: int, r: int, center_deg: float, n: int = 20) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = [(cx, cy)]
    start = math.radians(center_deg - 22.5)
    end = math.radians(center_deg + 22.5)
    for i in range(n + 1):
        a = start + (end - start) * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def draw_action_overlay(
    surface: pygame.Surface,
    img_rect: pygame.Rect,
    action: str,
    positions: dict[str, tuple[int, int]] | None = None,
    shoulder_rects: dict[str, pygame.Rect] | None = None,
    stick_info: dict[str, tuple[tuple[int, int], int]] | None = None,
) -> None:
    overlay = pygame.Surface((img_rect.w, img_rect.h), pygame.SRCALPHA)

    # Full-bar highlight for LB / RB
    if shoulder_rects is not None and action in shoulder_rects:
        r = shoulder_rects[action]
        local = pygame.Rect(r.x - img_rect.x, r.y - img_rect.y, r.w, r.h)
        pygame.draw.rect(overlay, (*ACCENT, 200), local, border_radius=10)
        surface.blit(overlay, img_rect.topleft)
        return

    # Pie-slice highlight for joystick directions
    if stick_info is not None and action in STICK_DIRECTION_ANGLES:
        side = action[0]  # "L" or "R"
        (abs_cx, abs_cy), r = stick_info[side]
        cx = abs_cx - img_rect.x
        cy = abs_cy - img_rect.y
        pts = _pie_polygon(cx, cy, r, STICK_DIRECTION_ANGLES[action])
        pygame.draw.polygon(overlay, (*ACCENT, 200), pts)
        surface.blit(overlay, img_rect.topleft)
        return

    if positions is not None:
        # Exact pixel positions returned by draw_placeholder_controller
        if action not in positions:
            surface.blit(overlay, img_rect.topleft)
            return
        abs_x, abs_y = positions[action]
        x = abs_x - img_rect.x
        y = abs_y - img_rect.y
    else:
        # Normalized fallback for a real controller image
        highlight_positions = {
            "LB": (0.20, 0.12),
            "RB": (0.80, 0.12),
            "X":  (0.63, 0.35),
            "Y":  (0.73, 0.28),
            "A":  (0.73, 0.48),
            "B":  (0.84, 0.39),

            "L_UP":    (0.26, 0.31),
            "L_LEFT":  (0.18, 0.43),
            "L_RIGHT": (0.34, 0.43),
            "L_DOWN":  (0.26, 0.55),

            "R_UP":    (0.65, 0.45),
            "R_LEFT":  (0.60, 0.55),
            "R_RIGHT": (0.70, 0.55),
            "R_DOWN":  (0.65, 0.66),
        }
        if action not in highlight_positions:
            surface.blit(overlay, img_rect.topleft)
            return
        nx, ny = highlight_positions[action]
        x = int(nx * img_rect.w)
        y = int(ny * img_rect.h)

    pygame.draw.circle(overlay, (*ACCENT, 255), (x, y), 16)
    surface.blit(overlay, img_rect.topleft)


def draw_prompt_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    action: str,
    controller_image: pygame.Surface | None,
    font_label: pygame.font.Font,
    font_small: pygame.font.Font,
    *,
    active: bool,
    flash_color: tuple[int, int, int] | None,
) -> None:
    bg_color = WHITE if active else PANEL
    draw_round_rect(surface, rect, bg_color, radius=20)

    border_color = GLOW if active else (215, 202, 178)
    border_width = 4 if active else 2
    if active and flash_color is not None:
        border_color = flash_color
        border_width = 5
    pygame.draw.rect(surface, border_color, rect, width=border_width, border_radius=20)

    label_h = 30
    label_rect = pygame.Rect(rect.x + 12, rect.y + 10, rect.w - 24, label_h)
    draw_round_rect(surface, label_rect, PANEL_ALT if active else (232, 224, 208), radius=12)
    draw_text(surface, action, font_label, INK, label_rect.center, center=True)

    image_area = pygame.Rect(rect.x + 12, rect.y + 48, rect.w - 24, rect.h - 60)
    if controller_image is None:
        positions, shoulder_rects, stick_info = draw_placeholder_controller(surface, image_area)
        img_rect = image_area
        draw_action_overlay(surface, img_rect, action, positions, shoulder_rects, stick_info)
    else:
        img_rect = fit_rect(controller_image.get_width(), controller_image.get_height(), image_area, padding=4)
        scaled = pygame.transform.smoothscale(controller_image, (img_rect.w, img_rect.h))
        surface.blit(scaled, img_rect)
        draw_action_overlay(surface, img_rect, action)


def draw_queue_panel(
    surface: pygame.Surface,
    state: GameState,
    controller_image: pygame.Surface | None,
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_small: pygame.font.Font,
) -> None:
    panel_rect = pygame.Rect(QUEUE_PANEL_X, QUEUE_PANEL_Y, QUEUE_PANEL_W, QUEUE_PANEL_H)
    draw_round_rect(surface, panel_rect, PANEL_ALT, radius=24)
    pygame.draw.rect(surface, (220, 206, 183), panel_rect, width=2, border_radius=24)

    available_h = panel_rect.h - QUEUE_CARD_TOP_PAD - QUEUE_CARD_BOTTOM_PAD
    card_h = int((available_h - QUEUE_CARD_GAP * (VISIBLE_PROMPTS - 1)) / VISIBLE_PROMPTS)

    flash_color = state.last_result_color if time.time() < state.last_result_until else None

    for idx, action in enumerate(state.prompt_queue[:VISIBLE_PROMPTS]):
        rect = pygame.Rect(
            panel_rect.x + QUEUE_CARD_SIDE_PAD,
            panel_rect.y + QUEUE_CARD_TOP_PAD + idx * (card_h + QUEUE_CARD_GAP),
            panel_rect.w - QUEUE_CARD_SIDE_PAD * 2,
            card_h,
        )
        draw_prompt_card(
            surface,
            rect,
            action,
            controller_image,
            font_med,
            font_small,
            active=(idx == 0),
            flash_color=flash_color if idx == 0 else None,
        )




# Cue colors matching run_cues.py
_CUE_BG      = (26,  26,  46)
_CUE_DIM     = (60,  60,  80)
_CUE_LEFT    = (80,  200, 255)
_CUE_RIGHT   = (255, 160, 60)
_CUE_VERT    = (255, 220, 80)
_CUE_LABEL   = (170, 170, 170)


def _cue_highlight_color(sym_type: str, sym_val) -> tuple:
    if sym_type == "ljoy":
        return _CUE_LEFT
    if sym_type == "rjoy":
        return _CUE_RIGHT
    if sym_type == "button":
        if sym_val in ("LEFT", "LB"):
            return _CUE_LEFT
        if sym_val in ("RIGHT", "RB"):
            return _CUE_RIGHT
    return _CUE_VERT


def _draw_cue(surface: pygame.Surface, sym_idx: int, font_cue: pygame.font.Font) -> None:
    """Render one cue frame (dark background + stick or button graphic)."""
    surface.fill(_CUE_BG)
    sym_type, sym_val = _COLLECT_SYMBOLS[sym_idx]
    hi = _cue_highlight_color(sym_type, sym_val)
    cx, cy = WIDTH // 2, HEIGHT // 2

    if sym_type in ("ljoy", "rjoy"):
        label = "LEFT STICK" if sym_type == "ljoy" else "RIGHT STICK"
        lbl = font_cue.render(label, True, hi)
        surface.blit(lbl, (cx - lbl.get_width() // 2, cy - 280))
        r = 180
        for k in range(8):
            a0 = math.radians(k * 45 - 22.5)
            a1 = math.radians((k + 1) * 45 - 22.5)
            color = hi if k == sym_val else _CUE_DIM
            pts = [(cx, cy)]
            for a in [a0 + (a1 - a0) * i / 12 for i in range(13)]:
                pts.append((cx + r * math.cos(a), cy - r * math.sin(a)))
            pygame.draw.polygon(surface, color, pts)
        pygame.draw.circle(surface, _CUE_BG, (cx, cy), 58)
    else:
        big_surf = font_cue.render(str(sym_val), True, hi)
        surface.blit(big_surf, (cx - big_surf.get_width() // 2,
                                cy - big_surf.get_height() // 2))


def draw_collect_screen(
    surface: pygame.Surface,
    t: float,
    collect: CollectState,
    font_title: pygame.font.Font,
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_small: pygame.font.Font,
    font_cue: pygame.font.Font,
) -> dict:
    """Render the collection screen; returns dict of clickable rects."""
    rects: dict = {}
    cx = WIDTH // 2

    # ── Dark full-screen cue phases ──────────────────────────────────────────
    if collect.phase in ("countdown", "sync_prompt", "sync_wait", "sync_release", "cueing"):
        surface.fill(_CUE_BG)
        now_ns = time.monotonic_ns()

        if collect.phase == "countdown":
            elapsed = now_ns - collect._phase_start_ns
            n = max(1, 3 - int(elapsed // 1_000_000_000))
            lbl = font_title.render(f"Starting in {n}", True, (255, 255, 255))
            surface.blit(lbl, (cx - lbl.get_width() // 2, HEIGHT // 2 - lbl.get_height() // 2))

        elif collect.phase in ("sync_prompt", "sync_wait"):
            draw_text(surface, "Hold LB for 2 seconds", font_big,
                      (255, 200, 60), (cx, HEIGHT // 2 - 44), center=True)
            draw_text(surface, "then release", font_big,
                      (255, 200, 60), (cx, HEIGHT // 2 + 16), center=True)
            if collect.phase == "sync_wait":
                elapsed_s = (now_ns - collect.sync_prompt_ns) / 1e9
                frac = min(1.0, elapsed_s / 2.0)
                bar_w = int((WIDTH - 200) * frac)
                pygame.draw.rect(surface, (60, 60, 80),
                                 pygame.Rect(100, HEIGHT // 2 + 80, WIDTH - 200, 24),
                                 border_radius=12)
                pygame.draw.rect(surface, (255, 200, 60),
                                 pygame.Rect(100, HEIGHT // 2 + 80, bar_w, 24),
                                 border_radius=12)

        elif collect.phase == "sync_release":
            draw_text(surface, "Release now", font_big,
                      (255, 200, 60), (cx, HEIGHT // 2), center=True)

        elif collect.phase == "cueing":
            if collect.display_sym_idx >= 0:
                _draw_cue(surface, collect.display_sym_idx, font_cue)
                # Progress counter
                prog = font_small.render(
                    f"{collect.current_trial} / {collect.n_trials}",
                    True, _CUE_LABEL,
                )
                surface.blit(prog, (cx - prog.get_width() // 2, 36))
            # else: flash (black already from fill)

        # ESC hint
        hint = font_small.render("ESC to abort", True, (80, 80, 100))
        surface.blit(hint, (WIDTH - hint.get_width() - 24, HEIGHT - hint.get_height() - 16))
        return rects

    # ── Light-background setup / result phases ───────────────────────────────
    surface.fill(BG)
    draw_text(surface, "CALIBRATION", font_title, INK, (cx, 44), center=True)
    draw_text(surface, "Record a cue session for decoder training",
              font_med, (130, 108, 72), (cx, 116), center=True)
    pygame.draw.line(surface, (210, 198, 175), (80, 146), (WIDTH - 80, 146), 2)

    panel = pygame.Rect(60, 162, WIDTH - 120, HEIGHT - 242)
    draw_round_rect(surface, panel, WHITE, radius=20)
    pygame.draw.rect(surface, (215, 202, 178), panel, width=2, border_radius=20)
    px, py = panel.x + 28, panel.y + 24
    pw = panel.w - 56

    if collect.phase == "setup":
        # ── Row 1: TRIALS preset buttons (left) + ISI stepper (right) ──────────
        half = (pw - 40) // 2

        # TRIALS – 3 preset buttons: 20 / 50 / 100
        draw_text(surface, "TRIALS", font_med, INK, (px, py))
        trial_presets = [20, 50, 100]
        preset_w = (half - 20) // 3
        for i, val in enumerate(trial_presets):
            bx = px + i * (preset_w + 10)
            br = pygame.Rect(bx, py + 34, preset_w, 48)
            selected = (collect.n_trials == val)
            bg = BLUE if selected else PANEL_ALT
            bd = BLUE if selected else (200, 188, 165)
            draw_round_rect(surface, br, bg, radius=8)
            pygame.draw.rect(surface, bd, br, width=2, border_radius=8)
            draw_text(surface, str(val), font_big, WHITE if selected else INK,
                      br.center, center=True)
            rects[f"trials_{val}"] = br

        # ISI
        isi_x = px + half + 40
        draw_text(surface, "ISI (s)", font_med, INK, (isi_x, py))
        btn_im = pygame.Rect(isi_x, py + 34, 44, 48)
        isi_rect = pygame.Rect(btn_im.right + 8, py + 34, half - 44 - 8 - 44 - 8, 48)
        btn_ip = pygame.Rect(isi_rect.right + 8, py + 34, 44, 48)
        draw_round_rect(surface, isi_rect, PANEL_ALT, radius=8)
        pygame.draw.rect(surface, (200, 188, 165), isi_rect, width=1, border_radius=8)
        draw_text(surface, f"{collect.isi_s:.1f}", font_big, INK,
                  isi_rect.center, center=True)
        for r, lbl in [(btn_im, "−"), (btn_ip, "+")]:
            draw_round_rect(surface, r, PANEL_ALT, radius=6)
            pygame.draw.rect(surface, (200, 188, 165), r, width=1, border_radius=6)
            draw_text(surface, lbl, font_big, INK, r.center, center=True)
        rects["isi_minus"] = btn_im
        rects["isi_plus"]  = btn_ip

        # Info box
        sep1 = py + 34 + 48 + 22
        pygame.draw.line(surface, (210, 198, 175), (px, sep1), (px + pw, sep1), 1)
        iy = sep1 + 20
        est_s = collect.n_trials * collect.isi_s
        draw_text(surface,
                  f"Estimated session:  {int(est_s // 60)} min {int(est_s % 60)} s  "
                  f"({collect.n_trials} cues at {collect.isi_s:.1f} s each)",
                  font_med, (100, 90, 75), (cx, iy), center=True)
        iy += 38
        draw_text(surface,
                  "A sync gesture (hold LB 2 s) will appear first to align the hardware clock.",
                  font_small, (140, 128, 108), (cx, iy), center=True)
        iy += 28
        draw_text(surface,
                  "Ensure Synapse is running and recording an .h5 file before starting.",
                  font_small, RED, (cx, iy), center=True)
        iy += 28
        draw_text(surface,
                  "Calibration will run automatically once recording is complete.",
                  font_small, (100, 140, 80), (cx, iy), center=True)

        # Buttons
        bty = panel.bottom - 74
        pygame.draw.line(surface, (210, 198, 175), (px, bty - 18), (px + pw, bty - 18), 1)
        start_r = pygame.Rect(px, bty, int(pw * 0.65), 52)
        skip_r  = pygame.Rect(start_r.right + 16, bty, pw - start_r.w - 16, 52)
        draw_round_rect(surface, start_r, BLUE, radius=12)
        draw_text(surface, "START COLLECTION", font_big, WHITE, start_r.center, center=True)
        draw_round_rect(surface, skip_r, PANEL_ALT, radius=12)
        pygame.draw.rect(surface, (200, 188, 165), skip_r, width=2, border_radius=12)
        draw_text(surface, "SKIP TO GAME", font_big, INK, skip_r.center, center=True)
        rects["start"] = start_r
        rects["skip"]  = skip_r

    elif collect.phase == "h5_select":
        draw_text(surface, "PAIR NEURAL RECORDING  (.h5)", font_big, INK, (px, py))
        pygame.draw.line(surface, (210, 198, 175), (px, py + 42), (px + pw, py + 42), 1)
        hy = py + 58

        draw_text(surface,
                  f"Cue log saved to:  {collect.json_out_path}",
                  font_small, GREEN, (px, hy))
        hy += 32
        draw_text(surface,
                  "Select the .h5 file recorded simultaneously by Synapse:",
                  font_med, INK, (px, hy))
        hy += 38
        row_h = 38
        visible = min(CALIB_FILE_ROWS, len(collect.h5_files))

        if not collect.h5_files:
            no_r = pygame.Rect(px, hy, pw, row_h * 2)
            draw_round_rect(surface, no_r, PANEL_ALT, radius=8)
            draw_text(surface, "No .h5 files found — check CALIB_H5_ROOTS",
                      font_small, RED, no_r.center, center=True)
        else:
            fa = pygame.Rect(px, hy, pw, visible * row_h + 8)
            draw_round_rect(surface, fa, PANEL_ALT, radius=8)
            pygame.draw.rect(surface, (200, 150, 50), fa, width=2, border_radius=8)
            rects["h5_list"] = fa
            for i in range(visible):
                fi = i
                if fi >= len(collect.h5_files):
                    break
                rr = pygame.Rect(px + 4, hy + 4 + i * row_h, pw - 8, row_h - 2)
                if fi == collect.selected_h5:
                    draw_round_rect(surface, rr, GLOW, radius=6)
                p = collect.h5_files[fi]
                try:
                    disp = str(p.relative_to(_SCRIPT_DIR.parent))
                except ValueError:
                    disp = str(p)
                marker = "► " if fi == collect.selected_h5 else "  "
                draw_text(surface, marker + disp, font_small,
                          INK if fi == collect.selected_h5 else (110, 100, 80),
                          (rr.x + 10, rr.y + 10))
                rects[f"h5_{fi}"] = rr

        bty = panel.bottom - 74
        pygame.draw.line(surface, (210, 198, 175), (px, bty - 18), (px + pw, bty - 18), 1)
        proc_r = pygame.Rect(px, bty, int(pw * 0.56), 52)
        skip_r = pygame.Rect(proc_r.right + 16, bty, pw - proc_r.w - 16, 52)
        can_proc = bool(collect.h5_files)
        draw_round_rect(surface, proc_r, BLUE if can_proc else (180, 180, 180), radius=12)
        draw_text(surface, "PROCESS + CALIBRATE", font_big, WHITE, proc_r.center, center=True)
        draw_round_rect(surface, skip_r, PANEL_ALT, radius=12)
        pygame.draw.rect(surface, (200, 188, 165), skip_r, width=2, border_radius=12)
        draw_text(surface, "SKIP TO GAME", font_big, INK, skip_r.center, center=True)
        if can_proc:
            rects["process"] = proc_r
        rects["skip"] = skip_r

    elif collect.phase == "processing":
        draw_text(surface, "PROCESSING...", font_title, INK, (cx, py + 50), center=True)
        sp_cx, sp_cy, sp_r = cx, py + 180, 46
        for i in range(12):
            angle = math.radians(i * 30 - t * 200)
            alpha = int(255 * (i + 1) / 12)
            dot_s = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(dot_s, (*BLUE[:3], alpha), (6, 6), 5)
            surface.blit(dot_s, (int(sp_cx + sp_r * math.cos(angle)) - 6,
                                 int(sp_cy + sp_r * math.sin(angle)) - 6))
        draw_text(surface, collect.progress_msg, font_med, INK,
                  (cx, sp_cy + sp_r + 32), center=True)

    elif collect.phase == "done":
        # Transient: poll() auto-calls start_auto_calibration(), so this is brief
        draw_text(surface, "PROCESSING COMPLETE", font_title, GREEN,
                  (cx, py + 40), center=True)
        draw_text(surface, "Starting calibration...", font_big, INK,
                  (cx, py + 130), center=True)

    elif collect.phase == "error":
        draw_text(surface, "PROCESSING FAILED", font_title, RED,
                  (cx, py + 40), center=True)
        err_box = pygame.Rect(px, py + 110, pw, 160)
        draw_round_rect(surface, err_box, (248, 230, 230), radius=14)
        pygame.draw.rect(surface, RED, err_box, width=2, border_radius=14)
        for li, line in enumerate(collect.error_msg.strip().splitlines()[:5]):
            draw_text(surface, line[:110], font_small, INK,
                      (err_box.x + 14, err_box.y + 12 + li * 28))
        retry_r = pygame.Rect(cx - 290, panel.bottom - 74, 250, 52)
        skip_r  = pygame.Rect(cx + 40,  panel.bottom - 74, 250, 52)
        draw_round_rect(surface, retry_r, ACCENT, radius=12)
        draw_text(surface, "RETRY", font_big, WHITE, retry_r.center, center=True)
        draw_round_rect(surface, skip_r, PANEL_ALT, radius=12)
        pygame.draw.rect(surface, (200, 188, 165), skip_r, width=2, border_radius=12)
        draw_text(surface, "SKIP TO GAME", font_big, INK, skip_r.center, center=True)
        rects["retry"] = retry_r
        rects["skip"]  = skip_r

    elif collect.phase == "calibrating":
        draw_text(surface, "CALIBRATING...", font_title, INK, (cx, py + 50), center=True)
        sp_cx, sp_cy, sp_r = cx, py + 180, 46
        for i in range(12):
            angle = math.radians(i * 30 - t * 200)
            alpha = int(255 * (i + 1) / 12)
            dot_s = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(dot_s, (*BLUE[:3], alpha), (6, 6), 5)
            surface.blit(dot_s, (int(sp_cx + sp_r * math.cos(angle)) - 6,
                                 int(sp_cy + sp_r * math.sin(angle)) - 6))
        draw_text(surface, collect.progress_msg, font_med, (100, 90, 75),
                  (cx, sp_cy + sp_r + 32), center=True)
        draw_text(surface,
                  "Training the neural decoder on your recorded session...",
                  font_small, (140, 128, 108), (cx, sp_cy + sp_r + 74), center=True)

    elif collect.phase == "calib_done":
        draw_text(surface, "CALIBRATION COMPLETE", font_title, GREEN,
                  (cx, py + 40), center=True)
        res_box = pygame.Rect(cx - 360, py + 120, 720, 80)
        draw_round_rect(surface, res_box, (228, 248, 232), radius=14)
        pygame.draw.rect(surface, GREEN, res_box, width=2, border_radius=14)
        draw_text(surface,
                  f"N = {collect.calib_N}     θ = {collect.calib_theta:.3f}     "
                  f"Bit rate = {collect.calib_B:.3f} bits/s",
                  font_big, INK, res_box.center, center=True)
        draw_text(surface, f"Config → {collect.calib_path}",
                  font_small, (110, 100, 80), (cx, res_box.bottom + 24), center=True)
        cont_r = pygame.Rect(cx - 220, panel.bottom - 74, 440, 52)
        draw_round_rect(surface, cont_r, GREEN, radius=12)
        draw_text(surface, "CONTINUE TO GAME", font_big, WHITE, cont_r.center, center=True)
        rects["continue"] = cont_r

    elif collect.phase == "calib_error":
        draw_text(surface, "CALIBRATION FAILED", font_title, RED,
                  (cx, py + 40), center=True)
        err_box = pygame.Rect(px, py + 110, pw, 160)
        draw_round_rect(surface, err_box, (248, 230, 230), radius=14)
        pygame.draw.rect(surface, RED, err_box, width=2, border_radius=14)
        for li, line in enumerate(collect.error_msg.strip().splitlines()[:5]):
            draw_text(surface, line[:110], font_small, INK,
                      (err_box.x + 14, err_box.y + 12 + li * 28))
        skip_r2 = pygame.Rect(cx - 200, panel.bottom - 74, 400, 52)
        draw_round_rect(surface, skip_r2, PANEL_ALT, radius=12)
        pygame.draw.rect(surface, (200, 188, 165), skip_r2, width=2, border_radius=12)
        draw_text(surface, "SKIP TO GAME", font_big, INK, skip_r2.center, center=True)
        rects["skip"] = skip_r2

    return rects


def draw_calibration_screen(
    surface: pygame.Surface,
    t: float,
    calib: CalibState,
    font_title: pygame.font.Font,
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_small: pygame.font.Font,
) -> dict:
    """Draw calibration screen; returns {name: pygame.Rect} of clickable areas."""
    surface.fill(BG)
    cx = WIDTH // 2
    rects: dict = {}

    # Title
    draw_text(surface, "CALIBRATION", font_title, INK, (cx, 44), center=True)
    draw_text(surface, "Configure and run the neural decoder before your first trial",
              font_med, (130, 108, 72), (cx, 116), center=True)
    pygame.draw.line(surface, (210, 198, 175), (80, 146), (WIDTH - 80, 146), 2)

    # Main panel
    panel = pygame.Rect(60, 162, WIDTH - 120, HEIGHT - 242)
    draw_round_rect(surface, panel, WHITE, radius=20)
    pygame.draw.rect(surface, (215, 202, 178), panel, width=2, border_radius=20)
    px, py = panel.x + 28, panel.y + 24
    pw = panel.w - 56

    if calib.phase == "setup":
        half = (pw - 32) // 2

        # Player name field
        draw_text(surface, "PLAYER NAME", font_med, INK, (px, py))
        name_rect = pygame.Rect(px, py + 34, half, 42)
        fc = GLOW if calib.active_field == "player" else PANEL_ALT
        fb = (200, 150, 50) if calib.active_field == "player" else (200, 188, 165)
        draw_round_rect(surface, name_rect, fc, radius=8)
        pygame.draw.rect(surface, fb, name_rect, width=2, border_radius=8)
        cursor = "|" if calib.active_field == "player" and int(t * 2) % 2 == 0 else ""
        draw_text(surface, calib.player_name + cursor, font_med, INK,
                  (name_rect.x + 10, name_rect.y + 9))
        rects["player_field"] = name_rect

        # Architecture selector
        ax0 = px + half + 32
        draw_text(surface, "ARCHITECTURE", font_med, INK, (ax0, py))
        btn_l = pygame.Rect(ax0, py + 34, 38, 42)
        btn_r = pygame.Rect(ax0 + half - 38, py + 34, 38, 42)
        aname = pygame.Rect(ax0 + 42, py + 34, half - 84, 42)
        for r, lbl in [(btn_l, "<"), (btn_r, ">")]:
            draw_round_rect(surface, r, PANEL_ALT, radius=6)
            pygame.draw.rect(surface, (200, 188, 165), r, width=1, border_radius=6)
            draw_text(surface, lbl, font_big, INK, r.center, center=True)
        draw_round_rect(surface, aname, PANEL_ALT, radius=6)
        draw_text(surface, ARCH_CHOICES[calib.arch_idx].upper(), font_med, INK,
                  aname.center, center=True)
        rects["arch_left"] = btn_l
        rects["arch_right"] = btn_r

        # Separator
        sep1 = py + 34 + 42 + 18
        pygame.draw.line(surface, (210, 198, 175), (px, sep1), (px + pw, sep1), 1)

        # File list
        fy = sep1 + 18
        draw_text(surface, "TRAINING FILE  (.npz)", font_med, INK, (px, fy))
        fy += 36
        row_h = 38

        if not calib.npz_files:
            no_rect = pygame.Rect(px, fy, pw, row_h * 2 + 8)
            draw_round_rect(surface, no_rect, PANEL_ALT, radius=8)
            draw_text(surface, "No .npz files found in data/raw/  —  check CALIB_DATA_ROOTS",
                      font_small, RED, no_rect.center, center=True)
        else:
            visible = min(CALIB_FILE_ROWS, len(calib.npz_files))
            fa = pygame.Rect(px, fy, pw, visible * row_h + 8)
            fb2 = (200, 150, 50) if calib.active_field == "files" else (200, 188, 165)
            draw_round_rect(surface, fa, PANEL_ALT, radius=8)
            pygame.draw.rect(surface, fb2, fa, width=2, border_radius=8)
            rects["files_area"] = fa
            for i in range(visible):
                fi = calib.file_scroll + i
                if fi >= len(calib.npz_files):
                    break
                rr = pygame.Rect(px + 4, fy + 4 + i * row_h, pw - 8, row_h - 2)
                if fi == calib.selected_file:
                    draw_round_rect(surface, rr, GLOW, radius=6)
                p = calib.npz_files[fi]
                try:
                    disp = str(p.relative_to(_SCRIPT_DIR.parent))
                except ValueError:
                    disp = str(p)
                marker = "► " if fi == calib.selected_file else "  "
                draw_text(surface, marker + disp, font_small,
                          INK if fi == calib.selected_file else (110, 100, 80),
                          (rr.x + 10, rr.y + 10))
                rects[f"file_{fi}"] = rr

        # Buttons
        bty = panel.bottom - 74
        pygame.draw.line(surface, (210, 198, 175), (px, bty - 18), (px + pw, bty - 18), 1)
        run_r = pygame.Rect(px, bty, int(pw * 0.56), 52)
        skip_r = pygame.Rect(run_r.right + 16, bty, pw - run_r.w - 16, 52)
        can_run = bool(calib.npz_files)
        draw_round_rect(surface, run_r, BLUE if can_run else (180, 180, 180), radius=12)
        draw_text(surface, "RUN CALIBRATION", font_big, WHITE, run_r.center, center=True)
        draw_round_rect(surface, skip_r, PANEL_ALT, radius=12)
        pygame.draw.rect(surface, (200, 188, 165), skip_r, width=2, border_radius=12)
        draw_text(surface, "SKIP TO GAME", font_big, INK, skip_r.center, center=True)
        if can_run:
            rects["run"] = run_r
        rects["skip"] = skip_r
        draw_text(surface, "TAB to switch field  •  ↑ ↓ to navigate  •  ← → to change arch",
                  font_small, (160, 148, 128), (cx, HEIGHT - 28), center=True)

    elif calib.phase == "running":
        draw_text(surface, "CALIBRATING...", font_title, INK, (cx, py + 50), center=True)
        # Spinner
        sp_cx, sp_cy, sp_r = cx, py + 180, 46
        for i in range(12):
            angle = math.radians(i * 30 - t * 200)
            alpha = int(255 * (i + 1) / 12)
            dot_surf = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (*BLUE[:3], alpha), (6, 6), 5)
            surface.blit(dot_surf, (int(sp_cx + sp_r * math.cos(angle)) - 6,
                                    int(sp_cy + sp_r * math.sin(angle)) - 6))
        draw_text(surface, calib.progress_msg, font_med, INK,
                  (cx, sp_cy + sp_r + 32), center=True)
        skip_r = pygame.Rect(cx - 130, panel.bottom - 74, 260, 52)
        draw_round_rect(surface, skip_r, PANEL_ALT, radius=12)
        pygame.draw.rect(surface, (200, 188, 165), skip_r, width=2, border_radius=12)
        draw_text(surface, "SKIP TO GAME", font_big, INK, skip_r.center, center=True)
        rects["skip"] = skip_r

    elif calib.phase == "done":
        draw_text(surface, "CALIBRATION COMPLETE", font_title, GREEN,
                  (cx, py + 40), center=True)
        res_box = pygame.Rect(cx - 340, py + 120, 680, 80)
        draw_round_rect(surface, res_box, (228, 248, 232), radius=14)
        pygame.draw.rect(surface, GREEN, res_box, width=2, border_radius=14)
        draw_text(surface,
                  f"N = {calib.result_N}     θ = {calib.result_theta:.3f}     "
                  f"Bit rate = {calib.result_B:.3f} bits/s",
                  font_big, INK, res_box.center, center=True)
        draw_text(surface, f"Config saved → {calib.result_path}",
                  font_small, (110, 100, 80), (cx, res_box.bottom + 24), center=True)
        cont_r = pygame.Rect(cx - 200, panel.bottom - 74, 400, 52)
        draw_round_rect(surface, cont_r, GREEN, radius=12)
        draw_text(surface, "CONTINUE TO GAME", font_big, WHITE, cont_r.center, center=True)
        rects["continue"] = cont_r

    elif calib.phase == "error":
        draw_text(surface, "CALIBRATION FAILED", font_title, RED,
                  (cx, py + 40), center=True)
        err_box = pygame.Rect(px, py + 110, pw, 160)
        draw_round_rect(surface, err_box, (248, 230, 230), radius=14)
        pygame.draw.rect(surface, RED, err_box, width=2, border_radius=14)
        for li, line in enumerate(calib.error_msg.strip().splitlines()[:5]):
            draw_text(surface, line[:110], font_small, INK,
                      (err_box.x + 14, err_box.y + 12 + li * 28))
        retry_r = pygame.Rect(cx - 290, panel.bottom - 74, 250, 52)
        skip_r  = pygame.Rect(cx + 40,  panel.bottom - 74, 250, 52)
        draw_round_rect(surface, retry_r, ACCENT, radius=12)
        draw_text(surface, "RETRY", font_big, WHITE, retry_r.center, center=True)
        draw_round_rect(surface, skip_r, PANEL_ALT, radius=12)
        pygame.draw.rect(surface, (200, 188, 165), skip_r, width=2, border_radius=12)
        draw_text(surface, "SKIP TO GAME", font_big, INK, skip_r.center, center=True)
        rects["retry"] = retry_r
        rects["skip"] = skip_r

    return rects



def draw_intro_screen(
    surface: pygame.Surface,
    t: float,
    font_title: pygame.font.Font,
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_small: pygame.font.Font,
) -> None:
    surface.fill(BG)
    cx = WIDTH // 2

    # ── Title ─────────────────────────────────────────────────
    draw_text(surface, "SCIENCE DANCE", font_title, INK, (cx, 44), center=True)
    draw_text(
        surface, "Brain-Computer Interface Dance Game",
        font_med, (130, 108, 72), (cx, 116), center=True,
    )
    pygame.draw.line(surface, (210, 198, 175), (80, 146), (WIDTH - 80, 146), 2)

    # ── Two-column content area ────────────────────────────────
    panel_top = 162
    panel_bot = HEIGHT - 108
    panel_h = panel_bot - panel_top
    gap = 28
    panel_w = (WIDTH - 120 - gap) // 2

    left_rect = pygame.Rect(60, panel_top, panel_w, panel_h)
    right_rect = pygame.Rect(left_rect.right + gap, panel_top, panel_w, panel_h)

    for r in (left_rect, right_rect):
        draw_round_rect(surface, r, WHITE, radius=20)
        pygame.draw.rect(surface, (215, 202, 178), r, width=2, border_radius=20)

    # ── Left panel: How to play ────────────────────────────────
    lx = left_rect.x + 28
    ly = left_rect.y + 24

    draw_text(surface, "HOW TO PLAY", font_big, INK, (lx, ly))
    pygame.draw.line(surface, (210, 198, 175), (lx, ly + 42), (left_rect.right - 28, ly + 42), 1)
    ly += 58

    bullets = [
        ("Goal",
         "Use your gaming controller to make a funky dude dance! The cue panel on ",
         "the right shows which controller input sequence you must make.",
         "Match it as quickly and accurately as possible."),
        ("Calibration",
         "Before each session, a short recording is made while you respond",
         "to practice cues on screen. This trains the decoder to recognize your",
         "patterns for each button."),
         ("Trial",
         "60-second timed run — try to maximize the number of correct inputs."),
        ("Score",
         "Bit rate (bits/sec): how much information your performance",
         "is successfully conveyed per second.", ""),
    ]
    for item in bullets:
        label = item[0]
        lines = [l for l in item[1:] if l]
        draw_text(surface, f"{label}:", font_med, ACCENT, (lx, ly))
        ly += 34
        for line in lines:
            draw_text(surface, line, font_small, INK, (lx + 10, ly))
            ly += 24
        ly += 18

    # ── Right panel: Input alphabet ────────────────────────────
    rx = right_rect.x + 28
    ry = right_rect.y + 24

    draw_text(surface, "INPUT SYMBOLS", font_big, INK, (rx, ry))
    pygame.draw.line(surface, (210, 198, 175), (rx, ry + 42), (right_rect.right - 28, ry + 42), 1)
    ry += 58

    dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
    col_w = (right_rect.w - 56) // 4

    for stick_label, prefix in [
        ("Left Joystick  (4 directions)", "L_"),
        ("Right Joystick  (4 directions)", "R_"),
    ]:
        draw_text(surface, stick_label, font_med, BLUE, (rx, ry))
        ry += 32
        for i, d in enumerate(dirs):
            draw_text(surface, f"{prefix}{d}", font_small, INK,
                      (rx + i * col_w, ry))
        ry += 42

    draw_text(surface, "Buttons  (6)", font_med, BLUE, (rx, ry))
    ry += 32
    btn_labels = ["A", "B", "X", "Y", "LB", "RB"]
    col_w2 = (right_rect.w - 56) // 6
    for i, b in enumerate(btn_labels):
        draw_text(surface, b, font_med, INK, (rx + i * col_w2, ry))

    # ── Pulsing "Press SPACE to begin" ────────────────────────
    pulse = (math.sin(t * 2.5) + 1) / 2
    box_color = (int(200 + pulse * 55), int(100 + pulse * 76), 20)
    space_box = pygame.Rect(cx - 240, HEIGHT - 96, 480, 106)
    draw_round_rect(surface, space_box, box_color, radius=18)
    pygame.draw.rect(surface, (255, 230, int(180 + pulse * 75)),
                     space_box, width=3, border_radius=18)
    draw_text(surface, "PRESS SPACE TO BEGIN", font_big, WHITE, space_box.center, center=True)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Dance Simulation Prototype")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("arial", 72, bold=True)
    font_cue   = pygame.font.SysFont("monospace", 200, bold=True)
    font_big = pygame.font.SysFont("arial", 34, bold=True)
    font_med = pygame.font.SysFont("arial", 24, bold=True)
    font_small = pygame.font.SysFont("arial", 18)

    controller_image = load_controller_image()
    judge_face = load_judge_face()

    state = GameState()
    state.reset()

    notes = make_music_notes()
    sardines: list[Sardine] = []
    input_queue: queue.Queue[str] = queue.Queue()
    # Parse optional Synapse args: --uri <addr> --tap <name>
    import sys as _sys
    _args = _sys.argv[1:]
    synapse_uri = SYNAPSE_URI_DEFAULT
    synapse_tap = SYNAPSE_TAP_DEFAULT
    i = 0
    while i < len(_args):
        if _args[i] == "--uri" and i + 1 < len(_args):
            synapse_uri = _args[i + 1]; i += 2
        elif _args[i] == "--tap" and i + 1 < len(_args):
            synapse_tap = _args[i + 1]; i += 2
        else:
            i += 1

    udp_receiver = UDPInputReceiver(input_queue)
    udp_receiver.start()
    synapse_receiver = SynapseInputReceiver(input_queue, uri=synapse_uri, tap_name=synapse_tap)
    synapse_receiver.start()

    reset_btn_rect: pygame.Rect | None = None
    show_intro = True
    show_collect = False
    collect = CollectState()
    _collect_btn_rects: dict = {}
    running = True
    try:
        while running:
            dt = clock.tick(FPS) / 1000.0
            now = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif show_intro:
                        if event.key == pygame.K_SPACE:
                            show_intro = False
                            show_collect = True
                    elif show_collect:
                        if collect.phase == "setup":
                            if event.key == pygame.K_ESCAPE:
                                show_collect = False
                        elif collect.phase in ("countdown", "sync_prompt", "sync_wait",
                                               "sync_release", "cueing"):
                            if event.key == pygame.K_ESCAPE:
                                collect.phase = "setup"
                        elif collect.phase in ("h5_select",):
                            if event.key == pygame.K_UP:
                                collect.selected_h5 = max(0, collect.selected_h5 - 1)
                            elif event.key == pygame.K_DOWN and collect.h5_files:
                                collect.selected_h5 = min(
                                    len(collect.h5_files) - 1, collect.selected_h5 + 1)
                            elif event.key == pygame.K_RETURN and collect.h5_files:
                                collect.start_processing(
                                    collect.json_out_path,
                                    collect.h5_files[collect.selected_h5])
                            elif event.key == pygame.K_ESCAPE:
                                show_collect = False
                        elif collect.phase in ("calib_done", "calib_error"):
                            if event.key == pygame.K_SPACE:
                                show_collect = False
                        elif collect.phase == "error":
                            if event.key == pygame.K_ESCAPE:
                                show_collect = False
                    elif event.key == pygame.K_SPACE:
                        state.start()
                    elif event.key == pygame.K_r:
                        state.reset()
                    else:
                        action = KEY_TO_ACTION.get(event.key)
                        if action is not None:
                            if state.process_input(action) is False:
                                spawn_sardines(sardines, count=random.randint(7, 12))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if show_collect:
                        for name, rect in _collect_btn_rects.items():
                            if rect.collidepoint(event.pos):
                                if name == "start":
                                    collect.start_session()
                                elif name == "skip":
                                    show_collect = False
                                elif name.startswith("trials_") and name[7:].isdigit():
                                    collect.n_trials = int(name[7:])
                                elif name == "isi_minus":
                                    collect.isi_s = round(max(0.3, collect.isi_s - 0.1), 1)
                                elif name == "isi_plus":
                                    collect.isi_s = round(min(3.0, collect.isi_s + 0.1), 1)
                                elif name == "process":
                                    collect.start_processing(
                                        collect.json_out_path,
                                        collect.h5_files[collect.selected_h5])
                                elif name == "continue":
                                    show_collect = False
                                elif name == "retry":
                                    collect.phase = "h5_select"
                                elif name.startswith("h5_") and name[3:].isdigit():
                                    collect.selected_h5 = int(name[3:])
                                break
                    elif reset_btn_rect is not None and reset_btn_rect.collidepoint(event.pos):
                        state.reset()

            if show_intro:
                draw_intro_screen(screen, now, font_title, font_big, font_med, font_small)
                pygame.display.flip()
                continue

            if show_collect:
                update_collect_timing(collect)
                if collect.phase in ("processing", "done", "calibrating"):
                    collect.poll()
                _collect_btn_rects = draw_collect_screen(
                    screen, now, collect, font_title, font_big, font_med, font_small, font_cue
                )
                pygame.display.flip()
                continue

            # Auto-end trial when time runs out
            if state.running and state.time_remaining() <= 0:
                state.end_trial()

            while True:
                try:
                    action = input_queue.get_nowait()
                except queue.Empty:
                    break
                if state.process_input(action) is False:
                    spawn_sardines(sardines, count=random.randint(7, 12))

            update_notes(notes, dt)
            update_sardines(sardines, dt)
            state.dancer_scale += (state.dancer_scale_target - state.dancer_scale) * min(1.0, dt * 8)

            screen.fill(BG)
            pygame.draw.line(screen, (220, 210, 188), (LEFT_PANEL_W, 20), (LEFT_PANEL_W, HEIGHT - 20), 3)
            reset_btn_rect = draw_status_panel(
            screen, state, font_big, font_med, font_small,
            syn_connected=synapse_receiver.connected,
            syn_error=synapse_receiver.error is not None,
        )
            if state.streak >= 10:
                draw_lasers(screen, now)
                draw_disco_ball(screen, now)
            draw_dancer(screen, now, judge_face, frown=state.frowning, scale=state.dancer_scale)
            if state.streak >= 10:
                draw_dancer_vectors(screen, now)
            for s in sardines:
                draw_sardine(screen, s)
            for note in notes:
                draw_note(screen, note.x, note.y, note.size)
            draw_streak_indicator(screen, state.streak, now, font_big, font_small)
            draw_queue_panel(screen, state, controller_image, font_big, font_med, font_small)
            pygame.display.flip()
    finally:
        udp_receiver.stop()
        synapse_receiver.stop()
        pygame.quit()


if __name__ == "__main__":
    main()

import math
import os
import queue
import random
import socket
import threading
import time
from dataclasses import dataclass, field

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
#   Left stick 8-way:
#       Q W E
#       A   D
#       Z X C
#   Right stick 8-way:
#       U I O
#       J   L
#       M , .
#
# UDP input mode:
#   Send UTF-8 strings like "A", "RB", "L_UP", "R_DOWNLEFT"
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
VISIBLE_PROMPTS = 3
UDP_PORT = 5005
CONTROLLER_IMAGE_PATH = "Controller.jpg"
JUDGE_FACE_PATH = "judge_face.png"  # Optional future stretch asset.

# Top-left white status box
STATUS_BOX_X = 48
STATUS_BOX_Y = 28
STATUS_BOX_W = LEFT_PANEL_W - 116
STATUS_BOX_H = 110

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

BUTTON_INPUTS = ["A", "B", "X", "Y", "RB", "LB"]
LEFT_STICK_INPUTS = [
    "L_UPLEFT", "L_UP", "L_UPRIGHT",
    "L_LEFT", "L_RIGHT",
    "L_DOWNLEFT", "L_DOWN", "L_DOWNRIGHT",
]
RIGHT_STICK_INPUTS = [
    "R_UPLEFT", "R_UP", "R_UPRIGHT",
    "R_LEFT", "R_RIGHT",
    "R_DOWNLEFT", "R_DOWN", "R_DOWNRIGHT",
]
INPUT_NAMES = BUTTON_INPUTS + LEFT_STICK_INPUTS + RIGHT_STICK_INPUTS
assert len(INPUT_NAMES) == 22

KEY_TO_ACTION = {
    pygame.K_1: "A",
    pygame.K_2: "B",
    pygame.K_3: "X",
    pygame.K_4: "Y",
    pygame.K_5: "LB",
    pygame.K_6: "RB",
    pygame.K_q: "L_UPLEFT",
    pygame.K_w: "L_UP",
    pygame.K_e: "L_UPRIGHT",
    pygame.K_a: "L_LEFT",
    pygame.K_d: "L_RIGHT",
    pygame.K_z: "L_DOWNLEFT",
    pygame.K_x: "L_DOWN",
    pygame.K_c: "L_DOWNRIGHT",
    pygame.K_u: "R_UPLEFT",
    pygame.K_i: "R_UP",
    pygame.K_o: "R_UPRIGHT",
    pygame.K_j: "R_LEFT",
    pygame.K_l: "R_RIGHT",
    pygame.K_m: "R_DOWNLEFT",
    pygame.K_COMMA: "R_DOWN",
    pygame.K_PERIOD: "R_DOWNRIGHT",
}

ALIAS_MAP = {
    "LUP": "L_UP",
    "LDOWN": "L_DOWN",
    "LLEFT": "L_LEFT",
    "LRIGHT": "L_RIGHT",
    "LUPLEFT": "L_UPLEFT",
    "LUPRIGHT": "L_UPRIGHT",
    "LDOWNLEFT": "L_DOWNLEFT",
    "LDOWNRIGHT": "L_DOWNRIGHT",
    "RUP": "R_UP",
    "RDOWN": "R_DOWN",
    "RLEFT": "R_LEFT",
    "RRIGHT": "R_RIGHT",
    "RUPLEFT": "R_UPLEFT",
    "RUPRIGHT": "R_UPRIGHT",
    "RDOWNLEFT": "R_DOWNLEFT",
    "RDOWNRIGHT": "R_DOWNRIGHT",
}


@dataclass
class FloatingNote:
    x: float
    y: float
    speed: float
    phase: float
    size: float


@dataclass
class GameState:
    running: bool = False
    total_inputs: int = 0
    incorrect_inputs: int = 0
    prompt_queue: list[str] = field(default_factory=list)
    last_result_color: tuple[int, int, int] | None = None
    last_result_until: float = 0.0
    last_input_text: str = "—"
    elapsed_before_pause: float = 0.0
    run_start_time: float | None = None

    def reset(self) -> None:
        self.running = False
        self.total_inputs = 0
        self.incorrect_inputs = 0
        self.prompt_queue = [random_prompt() for _ in range(VISIBLE_PROMPTS)]
        self.last_result_color = None
        self.last_result_until = 0.0
        self.last_input_text = "—"
        self.elapsed_before_pause = 0.0
        self.run_start_time = None

    def start_stop(self) -> None:
        now = time.time()
        if self.running:
            self.running = False
            if self.run_start_time is not None:
                self.elapsed_before_pause += now - self.run_start_time
            self.run_start_time = None
        else:
            self.running = True
            self.run_start_time = now

    def elapsed(self) -> float:
        if self.running and self.run_start_time is not None:
            return self.elapsed_before_pause + (time.time() - self.run_start_time)
        return self.elapsed_before_pause

    def bitrate(self, N: int) -> float:
        elapsed = self.elapsed()
        if elapsed <= 0:
            return 0.0
        return math.log2(N) * (self.total_inputs - self.incorrect_inputs) / elapsed

    def process_input(self, action: str) -> None:
        if not self.running or not self.prompt_queue:
            return

        expected = self.prompt_queue[0]
        self.total_inputs += 1
        self.last_input_text = action
        correct = action == expected
        if not correct:
            self.incorrect_inputs += 1

        self.last_result_color = GREEN if correct else RED
        self.last_result_until = time.time() + 0.30

        self.prompt_queue.pop(0)
        self.prompt_queue.append(random_prompt())


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


def draw_dancer(surface: pygame.Surface, t: float, judge_face: pygame.Surface | None = None) -> None:
    cx = LEFT_PANEL_W // 2 - 10
    floor_y = HEIGHT - 110
    bounce = math.sin(t * 4.0) * 10

    head_y = 180 + bounce
    neck_y = 235 + bounce
    hip_y = 405 + bounce

    left_hand_x = cx - 150 + math.sin(t * 3.2) * 36
    left_hand_y = 265 + math.sin(t * 6.0) * 30
    right_hand_x = cx + 145 + math.cos(t * 3.4) * 28
    right_hand_y = 255 + math.sin(t * 5.4 + 1.0) * 42

    left_knee_x = cx - 65 + math.sin(t * 3.6 + 0.9) * 28
    left_knee_y = 545 + math.sin(t * 4.2) * 12
    right_knee_x = cx + 58 + math.cos(t * 3.9 + 0.4) * 24
    right_knee_y = 540 + math.cos(t * 4.0) * 12

    left_foot_x = cx - 92 + math.sin(t * 5.0) * 22
    left_foot_y = floor_y
    right_foot_x = cx + 95 + math.cos(t * 4.8) * 18
    right_foot_y = floor_y

    pygame.draw.ellipse(surface, (225, 214, 192), (80, floor_y - 20, LEFT_PANEL_W - 160, 70))

    line_w = 18
    pygame.draw.line(surface, SILHOUETTE, (cx, neck_y), (cx, hip_y), line_w)
    pygame.draw.line(surface, SILHOUETTE, (cx, 295 + bounce), (left_hand_x, left_hand_y), line_w)
    pygame.draw.line(surface, SILHOUETTE, (cx, 300 + bounce), (right_hand_x, right_hand_y), line_w)
    pygame.draw.line(surface, SILHOUETTE, (cx - 18, hip_y), (left_knee_x, left_knee_y), line_w)
    pygame.draw.line(surface, SILHOUETTE, (left_knee_x, left_knee_y), (left_foot_x, left_foot_y), line_w)
    pygame.draw.line(surface, SILHOUETTE, (cx + 18, hip_y), (right_knee_x, right_knee_y), line_w)
    pygame.draw.line(surface, SILHOUETTE, (right_knee_x, right_knee_y), (right_foot_x, right_foot_y), line_w)

    for point in [
        (cx, neck_y), (cx, hip_y), (left_hand_x, left_hand_y), (right_hand_x, right_hand_y),
        (left_knee_x, left_knee_y), (right_knee_x, right_knee_y), (left_foot_x, left_foot_y), (right_foot_x, right_foot_y)
    ]:
        pygame.draw.circle(surface, SILHOUETTE, (int(point[0]), int(point[1])), 12)

    head_center = (cx, int(head_y))
    pygame.draw.circle(surface, SILHOUETTE, head_center, 48)

    if judge_face is not None:
        face = pygame.transform.smoothscale(judge_face, (82, 82))
        face_rect = face.get_rect(center=head_center)
        clip = pygame.Surface((82, 82), pygame.SRCALPHA)
        pygame.draw.circle(clip, (255, 255, 255), (41, 41), 41)
        clip.blit(face, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surface.blit(clip, face_rect)

def draw_status_panel(
    surface: pygame.Surface,
    state: GameState,
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_small: pygame.font.Font,
) -> None:
    box = pygame.Rect(STATUS_BOX_X, STATUS_BOX_Y, STATUS_BOX_W, STATUS_BOX_H)
    draw_round_rect(surface, box, WHITE, radius=22)
    pygame.draw.rect(surface, (224, 212, 185), box, width=2, border_radius=22)

    bitrate = state.bitrate(22)
    draw_text(surface, f"Bit rate: {bitrate:.2f} bits/sec", font_big, INK, (box.x + 22, box.y + 14))
    run_state = "RUNNING" if state.running else "STOPPED"
    draw_text(surface, f"Judge: {run_state}", font_small, BLUE if state.running else RED, (box.x + 22, box.y + 84))
    draw_text(surface, f"Seen: {state.total_inputs}", font_small, INK, (box.x + 170, box.y + 84))
    draw_text(surface, f"Wrong: {state.incorrect_inputs}", font_small, INK, (box.x + 260, box.y + 84))


def draw_placeholder_controller(surface: pygame.Surface, rect: pygame.Rect) -> tuple[dict[str, tuple[int, int]], dict[str, pygame.Rect], dict[str, tuple[tuple[int, int], int]]]:
    body = rect.inflate(-int(rect.w * 0.08), -int(rect.h * 0.16))

    base_color = (232, 223, 203)
    edge_color = (198, 186, 162)
    detail_color = (176, 164, 142)
    stick_ring_color = (145, 134, 116)
    stick_inner_color = (120, 110, 96)
    button_color = (200, 188, 164)

    # Main rectangular controller body
    main_body = pygame.Rect(
        body.x + int(body.w * 0.05),
        body.y + int(body.h * 0.16),
        int(body.w * 0.90),
        int(body.h * 0.62),
    )
    pygame.draw.rect(surface, base_color, main_body, border_radius=34)
    pygame.draw.rect(surface, edge_color, main_body, width=3, border_radius=34)

    # LB / RB shoulder buttons
    shoulder_w = int(main_body.w * 0.24)
    shoulder_h = int(main_body.h * 0.12)
    shoulder_y = main_body.y - int(shoulder_h * 0.65)

    lb_rect = pygame.Rect(
        main_body.x + int(main_body.w * 0.03),
        shoulder_y,
        shoulder_w,
        shoulder_h,
    )
    rb_rect = pygame.Rect(
        main_body.right - shoulder_w - int(main_body.w * 0.03),
        shoulder_y,
        shoulder_w,
        shoulder_h,
    )

    pygame.draw.rect(surface, detail_color, lb_rect, border_radius=10)
    pygame.draw.rect(surface, detail_color, rb_rect, border_radius=10)
    pygame.draw.rect(surface, edge_color, lb_rect, width=2, border_radius=10)
    pygame.draw.rect(surface, edge_color, rb_rect, width=2, border_radius=10)

    # Left joystick
    stick_r = max(14, int(main_body.w * 0.055))
    left_stick_center = (
        main_body.x + int(main_body.w * 0.28),
        main_body.y + int(main_body.h * 0.48),
    )
    pygame.draw.circle(surface, detail_color, left_stick_center, stick_r + 7)
    pygame.draw.circle(surface, stick_ring_color, left_stick_center, stick_r)
    pygame.draw.circle(surface, stick_inner_color, left_stick_center, max(6, stick_r - 6))

    # Right joystick (same size/style as left)
    right_stick_center = (
        main_body.x + int(main_body.w * 0.58),
        main_body.y + int(main_body.h * 0.58),
    )
    pygame.draw.circle(surface, detail_color, right_stick_center, stick_r + 7)
    pygame.draw.circle(surface, stick_ring_color, right_stick_center, stick_r)
    pygame.draw.circle(surface, stick_inner_color, right_stick_center, max(6, stick_r - 6))

    # XYAB diamond layout
    face_cx = main_body.x + int(main_body.w * 0.79)
    face_cy = main_body.y + int(main_body.h * 0.40)
    face_gap = max(18, int(main_body.w * 0.055))
    face_r = max(9, int(main_body.w * 0.026))

    diamond_positions = {
        "Y": (face_cx, face_cy - face_gap),
        "A": (face_cx, face_cy + face_gap),
        "X": (face_cx - face_gap, face_cy),
        "B": (face_cx + face_gap, face_cy),
    }

    for pos in diamond_positions.values():
        pygame.draw.circle(surface, button_color, pos, face_r)
        pygame.draw.circle(surface, edge_color, pos, face_r, 2)

    # Build exact positions dict for overlay alignment
    positions: dict[str, tuple[int, int]] = {}
    positions["LB"] = lb_rect.center
    positions["RB"] = rb_rect.center
    positions.update(diamond_positions)

    d = stick_r + 6
    diag = int(d * 0.707)
    lx, ly = left_stick_center
    positions["L_UP"]        = (lx,        ly - d)
    positions["L_DOWN"]      = (lx,        ly + d)
    positions["L_LEFT"]      = (lx - d,    ly)
    positions["L_RIGHT"]     = (lx + d,    ly)
    positions["L_UPLEFT"]    = (lx - diag, ly - diag)
    positions["L_UPRIGHT"]   = (lx + diag, ly - diag)
    positions["L_DOWNLEFT"]  = (lx - diag, ly + diag)
    positions["L_DOWNRIGHT"] = (lx + diag, ly + diag)

    rx, ry = right_stick_center
    positions["R_UP"]        = (rx,        ry - d)
    positions["R_DOWN"]      = (rx,        ry + d)
    positions["R_LEFT"]      = (rx - d,    ry)
    positions["R_RIGHT"]     = (rx + d,    ry)
    positions["R_UPLEFT"]    = (rx - diag, ry - diag)
    positions["R_UPRIGHT"]   = (rx + diag, ry - diag)
    positions["R_DOWNLEFT"]  = (rx - diag, ry + diag)
    positions["R_DOWNRIGHT"] = (rx + diag, ry + diag)

    shoulder_rects = {"LB": lb_rect, "RB": rb_rect}
    stick_info = {
        "L": (left_stick_center, stick_r + 7),
        "R": (right_stick_center, stick_r + 7),
    }
    return positions, shoulder_rects, stick_info

STICK_DIRECTION_ANGLES = {
    "L_RIGHT": 0, "L_DOWNRIGHT": 45, "L_DOWN": 90, "L_DOWNLEFT": 135,
    "L_LEFT": 180, "L_UPLEFT": 225, "L_UP": 270, "L_UPRIGHT": 315,
    "R_RIGHT": 0, "R_DOWNRIGHT": 45, "R_DOWN": 90, "R_DOWNLEFT": 135,
    "R_LEFT": 180, "R_UPLEFT": 225, "R_UP": 270, "R_UPRIGHT": 315,
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

            "L_UPLEFT":    (0.18, 0.35),
            "L_UP":        (0.26, 0.31),
            "L_UPRIGHT":   (0.34, 0.35),
            "L_LEFT":      (0.18, 0.43),
            "L_RIGHT":     (0.34, 0.43),
            "L_DOWNLEFT":  (0.18, 0.51),
            "L_DOWN":      (0.26, 0.55),
            "L_DOWNRIGHT": (0.34, 0.51),

            "R_UPLEFT":    (0.62, 0.47),
            "R_UP":        (0.65, 0.45),
            "R_UPRIGHT":   (0.68, 0.47),
            "R_LEFT":      (0.60, 0.55),
            "R_RIGHT":     (0.70, 0.55),
            "R_DOWNLEFT":  (0.62, 0.63),
            "R_DOWN":      (0.65, 0.66),
            "R_DOWNRIGHT": (0.68, 0.63),
        }
        if action not in highlight_positions:
            surface.blit(overlay, img_rect.topleft)
            return
        nx, ny = highlight_positions[action]
        x = int(nx * img_rect.w)
        y = int(ny * img_rect.h)

    pygame.draw.circle(overlay, (*ACCENT, 255), (x, y), 4)
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




def main() -> None:
    pygame.init()
    pygame.display.set_caption("Dance Simulation Prototype")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("arial", 34, bold=True)
    font_med = pygame.font.SysFont("arial", 24, bold=True)
    font_small = pygame.font.SysFont("arial", 18)

    controller_image = load_controller_image()
    judge_face = load_judge_face()

    state = GameState()
    state.reset()

    notes = make_music_notes()
    input_queue: queue.Queue[str] = queue.Queue()
    udp_receiver = UDPInputReceiver(input_queue)
    udp_receiver.start()

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
                    elif event.key == pygame.K_SPACE:
                        state.start_stop()
                    elif event.key == pygame.K_r:
                        state.reset()
                    else:
                        action = KEY_TO_ACTION.get(event.key)
                        if action is not None:
                            state.process_input(action)

            while True:
                try:
                    action = input_queue.get_nowait()
                except queue.Empty:
                    break
                state.process_input(action)

            update_notes(notes, dt)

            screen.fill(BG)
            pygame.draw.line(screen, (220, 210, 188), (LEFT_PANEL_W, 20), (LEFT_PANEL_W, HEIGHT - 20), 3)
            draw_status_panel(screen, state, font_big, font_med, font_small)
            draw_dancer(screen, now, judge_face)
            for note in notes:
                draw_note(screen, note.x, note.y, note.size)
            draw_queue_panel(screen, state, controller_image, font_big, font_med, font_small)
            pygame.display.flip()
    finally:
        udp_receiver.stop()
        pygame.quit()


if __name__ == "__main__":
    main()

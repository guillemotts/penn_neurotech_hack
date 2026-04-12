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
        return math.log2(N) * (self.total_inputs - self.incorrect_inputs) / elapsed

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

        self.prompt_queue.pop(0)
        self.prompt_queue.append(random_prompt())
        return correct


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


def draw_dancer(surface: pygame.Surface, t: float, judge_face: pygame.Surface | None = None, *, frown: bool = False) -> None:
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

    # ── Floor shadow ─────────────────────────────────────────────
    pygame.draw.ellipse(surface, (225, 214, 192), (60, floor_y - 18, LEFT_PANEL_W - 120, 64))

    # ── Draw order: back → torso → front → head ──────────────────

    # Back arm (right)
    draw_capsule_3d(surface, BODY, r_shoulder, r_elbow,    14)
    draw_capsule_3d(surface, BODY, r_elbow,    r_wrist,    11)
    draw_capsule_3d(surface, BODY, r_wrist,    right_hand, 8)
    draw_joint_3d(surface, r_shoulder, 15, BODY)
    draw_joint_3d(surface, r_elbow,    13, BODY)
    draw_joint_3d(surface, r_wrist,    10, BODY)
    draw_joint_3d(surface, right_hand, 13, SKIN)

    # Back leg (right)
    draw_capsule_3d(surface, BODY, r_hip_pt,   right_knee, 19)
    draw_capsule_3d(surface, BODY, right_knee, r_ankle,    15)
    draw_capsule_3d(surface, BODY, r_ankle,    right_foot, 11)
    draw_joint_3d(surface, r_hip_pt,   18, BODY)
    draw_joint_3d(surface, right_knee, 17, BODY)
    draw_joint_3d(surface, r_ankle,    13, BODY)
    pygame.draw.ellipse(surface, SHOE, (int(right_foot[0]) - 26, int(right_foot[1]) - 10, 44, 20))

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
    pygame.draw.polygon(surface, BODY, torso_pts)
    # Chest highlight plane (lit face of torso)
    chest_pts = [
        (int(l_shoulder[0]) + 12, int(l_shoulder[1]) + 5),
        (int(r_shoulder[0]) - 12, int(r_shoulder[1]) + 5),
        (cx + 16,  waist_y - 8),
        (cx - 16,  waist_y - 8),
    ]
    pygame.draw.polygon(surface, FRONT, chest_pts)

    # Neck
    draw_capsule_3d(surface, BODY, (cx, int(neck_y)), (cx, int(neck_y + 28)), 14)

    # Front arm (left)
    draw_capsule_3d(surface, FRONT, l_shoulder, l_elbow,   14)
    draw_capsule_3d(surface, FRONT, l_elbow,    l_wrist,   11)
    draw_capsule_3d(surface, FRONT, l_wrist,    left_hand,  8)
    draw_joint_3d(surface, l_shoulder, 15, FRONT)
    draw_joint_3d(surface, l_elbow,    13, FRONT)
    draw_joint_3d(surface, l_wrist,    10, FRONT)
    draw_joint_3d(surface, left_hand,  13, SKIN)

    # Front leg (left)
    draw_capsule_3d(surface, FRONT, l_hip_pt,  left_knee, 19)
    draw_capsule_3d(surface, FRONT, left_knee, l_ankle,   15)
    draw_capsule_3d(surface, FRONT, l_ankle,   left_foot, 11)
    draw_joint_3d(surface, l_hip_pt,  18, FRONT)
    draw_joint_3d(surface, left_knee, 17, FRONT)
    draw_joint_3d(surface, l_ankle,   13, FRONT)
    pygame.draw.ellipse(surface, SHOE_F, (int(left_foot[0]) - 26, int(left_foot[1]) - 10, 44, 20))

    # ── Head ─────────────────────────────────────────────────────
    head_center = (cx, int(head_y))
    draw_joint_3d(surface, head_center, 54, BODY)
    # Ears
    draw_joint_3d(surface, (head_center[0] - 52, head_center[1] + 4), 12, SKIN)
    draw_joint_3d(surface, (head_center[0] + 52, head_center[1] + 4), 12, SKIN)
    # Face
    pygame.draw.circle(surface, SKIN, head_center, 46)
    # Eyes
    for ex, ey in [(head_center[0] - 16, head_center[1] - 8), (head_center[0] + 16, head_center[1] - 8)]:
        pygame.draw.circle(surface, (255, 255, 255), (ex, ey), 10)
        pygame.draw.circle(surface, (35, 25, 15),    (ex, ey),  6)
        pygame.draw.circle(surface, (255, 255, 255), (ex + 2, ey - 3), 2)
    # Mouth
    mouth_start, mouth_end = (0, math.pi) if frown else (math.pi, 2 * math.pi)
    pygame.draw.arc(surface, (155, 75, 65),
                    pygame.Rect(head_center[0] - 15, head_center[1] + 10, 30, 16),
                    mouth_start, mouth_end, 3)

    if judge_face is not None:
        face = pygame.transform.smoothscale(judge_face, (86, 86))
        face_rect = face.get_rect(center=head_center)
        clip = pygame.Surface((86, 86), pygame.SRCALPHA)
        pygame.draw.circle(clip, (255, 255, 255), (43, 43), 43)
        clip.blit(face, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surface.blit(clip, face_rect)

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

    base_w, base_h = 210, 64
    expand = int(pulse * 10)
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

    draw_text(surface, "HOT STREAK", font_small, WHITE, (box.centerx, box.y + 20), center=True)
    draw_text(surface, f"x{streak}", font_big, WHITE, (box.centerx, box.y + 42), center=True)


def draw_status_panel(
    surface: pygame.Surface,
    state: GameState,
    font_big: pygame.font.Font,
    font_med: pygame.font.Font,
    font_small: pygame.font.Font,
) -> pygame.Rect:
    box = pygame.Rect(STATUS_BOX_X, STATUS_BOX_Y, STATUS_BOX_W, STATUS_BOX_H)
    draw_round_rect(surface, box, WHITE, radius=22)
    pygame.draw.rect(surface, (224, 212, 185), box, width=2, border_radius=22)

    # Bit rate
    bitrate = state.bitrate(22)
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
    sardines: list[Sardine] = []
    input_queue: queue.Queue[str] = queue.Queue()
    udp_receiver = UDPInputReceiver(input_queue)
    udp_receiver.start()

    reset_btn_rect: pygame.Rect | None = None
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
                        state.start()
                    elif event.key == pygame.K_r:
                        state.reset()
                    else:
                        action = KEY_TO_ACTION.get(event.key)
                        if action is not None:
                            if state.process_input(action) is False:
                                spawn_sardines(sardines, count=random.randint(7, 12))
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if reset_btn_rect is not None and reset_btn_rect.collidepoint(event.pos):
                        state.reset()

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

            screen.fill(BG)
            pygame.draw.line(screen, (220, 210, 188), (LEFT_PANEL_W, 20), (LEFT_PANEL_W, HEIGHT - 20), 3)
            reset_btn_rect = draw_status_panel(screen, state, font_big, font_med, font_small)
            if state.streak >= 10:
                draw_lasers(screen, now)
                draw_disco_ball(screen, now)
            draw_dancer(screen, now, judge_face, frown=state.frowning)
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
        pygame.quit()


if __name__ == "__main__":
    main()

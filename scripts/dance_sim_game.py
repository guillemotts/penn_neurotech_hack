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

WIDTH, HEIGHT = 1440, 860
LEFT_PANEL_W = int(WIDTH * 0.58)
RIGHT_PANEL_X = LEFT_PANEL_W
RIGHT_PANEL_W = WIDTH - LEFT_PANEL_W
FPS = 60
VISIBLE_PROMPTS = 5
UDP_PORT = 5005
CONTROLLER_IMAGE_PATH = "Controller.jpg"
JUDGE_FACE_PATH = "judge_face.png"  # Optional future stretch asset.

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

    def bitrate(self) -> float:
        elapsed = self.elapsed()
        if elapsed <= 0:
            return 0.0
        return math.log2(22) * (self.total_inputs - self.incorrect_inputs) / elapsed

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


def draw_status_panel(surface: pygame.Surface, state: GameState, font_big: pygame.font.Font, font_med: pygame.font.Font, font_small: pygame.font.Font) -> None:
    box = pygame.Rect(60, 36, LEFT_PANEL_W - 120, 112)
    draw_round_rect(surface, box, WHITE, radius=22)
    pygame.draw.rect(surface, (224, 212, 185), box, width=2, border_radius=22)

    bitrate = state.bitrate()
    draw_text(surface, f"Bit rate: {bitrate:.2f} bits/sec", font_big, INK, (box.x + 22, box.y + 16))
    draw_text(
        surface,
        f"Total: {state.total_inputs}   Correct: {state.total_inputs - state.incorrect_inputs}   Incorrect: {state.incorrect_inputs}",
        font_med,
        INK,
        (box.x + 22, box.y + 54),
    )
    run_state = "RUNNING" if state.running else "STOPPED"
    draw_text(surface, f"Judge: {run_state}", font_small, BLUE if state.running else RED, (box.x + 22, box.y + 82))
    draw_text(surface, f"Last input: {state.last_input_text}", font_small, INK, (box.x + 170, box.y + 82))
    draw_text(surface, f"UDP port: {UDP_PORT}", font_small, INK, (box.right - 160, box.y + 82))


def draw_placeholder_controller(surface: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, (248, 242, 228), rect, border_radius=26)
    body = rect.inflate(-rect.w * 0.12, -rect.h * 0.18)
    pygame.draw.rect(surface, (230, 220, 198), body, border_radius=50)
    pygame.draw.circle(surface, (211, 200, 178), (body.x + body.w // 4, body.y + body.h // 2), max(12, body.w // 10))
    for offset in [(-16, -16), (16, -16), (-16, 16), (16, 16)]:
        pygame.draw.circle(surface, (211, 200, 178), (body.x + int(body.w * 0.73) + offset[0], body.y + int(body.h * 0.38) + offset[1]), 10)


def draw_action_overlay(surface: pygame.Surface, img_rect: pygame.Rect, action: str) -> None:
    overlay = pygame.Surface((img_rect.w, img_rect.h), pygame.SRCALPHA)

    button_positions = {
        "LB": (0.20, 0.12),
        "RB": (0.80, 0.12),
        "X": (0.63, 0.35),
        "Y": (0.73, 0.28),
        "A": (0.73, 0.48),
        "B": (0.84, 0.39),
    }
    left_stick_center = (0.26, 0.43)
    right_stick_center = (0.50, 0.65)

    def to_px(nx: float, ny: float) -> tuple[int, int]:
        return int(nx * img_rect.w), int(ny * img_rect.h)

    if action in button_positions:
        x, y = to_px(*button_positions[action])
        pygame.draw.circle(overlay, (*ACCENT, 190), (x, y), max(14, img_rect.w // 20))
    else:
        is_left = action.startswith("L_")
        center = left_stick_center if is_left else right_stick_center
        cx, cy = to_px(*center)
        pygame.draw.circle(overlay, (255, 255, 255, 90), (cx, cy), max(10, img_rect.w // 24), 2)

        direction = action.split("_", 1)[1]
        dx = 0.0
        dy = 0.0
        if "LEFT" in direction:
            dx -= 0.12
        if "RIGHT" in direction:
            dx += 0.12
        if "UP" in direction:
            dy -= 0.12
        if "DOWN" in direction:
            dy += 0.12

        tx, ty = to_px(center[0] + dx, center[1] + dy)
        pygame.draw.line(overlay, (*ACCENT, 180), (cx, cy), (tx, ty), 5)
        pygame.draw.circle(overlay, (*ACCENT, 220), (tx, ty), max(12, img_rect.w // 24))

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
        draw_placeholder_controller(surface, image_area)
        img_rect = image_area
    else:
        img_rect = fit_rect(controller_image.get_width(), controller_image.get_height(), image_area, padding=4)
        scaled = pygame.transform.smoothscale(controller_image, (img_rect.w, img_rect.h))
        surface.blit(scaled, img_rect)

    draw_action_overlay(surface, img_rect, action)
    draw_text(surface, "top prompt" if active else "incoming", font_small, INK, (rect.x + 16, rect.bottom - 24))


def draw_queue_panel(surface: pygame.Surface, state: GameState, controller_image: pygame.Surface | None, font_big: pygame.font.Font, font_med: pygame.font.Font, font_small: pygame.font.Font) -> None:
    panel_rect = pygame.Rect(RIGHT_PANEL_X + 18, 22, RIGHT_PANEL_W - 36, HEIGHT - 44)
    draw_round_rect(surface, panel_rect, PANEL_ALT, radius=24)
    pygame.draw.rect(surface, (220, 206, 183), panel_rect, width=2, border_radius=24)

    draw_text(surface, "Prompt Queue", font_big, INK, (panel_rect.x + 18, panel_rect.y + 14))
    draw_text(surface, "Green = correct  •  Red = wrong  •  queue always advances", font_small, INK, (panel_rect.x + 18, panel_rect.y + 52))

    card_area_top = panel_rect.y + 84
    card_gap = 10
    available_h = panel_rect.h - 110
    card_h = int((available_h - card_gap * (VISIBLE_PROMPTS - 1)) / VISIBLE_PROMPTS)

    flash_color = state.last_result_color if time.time() < state.last_result_until else None

    for idx, action in enumerate(state.prompt_queue[:VISIBLE_PROMPTS]):
        rect = pygame.Rect(panel_rect.x + 16, card_area_top + idx * (card_h + card_gap), panel_rect.w - 32, card_h)
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


def draw_footer_help(surface: pygame.Surface, font_small: pygame.font.Font) -> None:
    footer_text = "SPACE start/stop • R reset • keyboard test inputs enabled • UDP strings like A, RB, L_UP, R_DOWNLEFT"
    draw_text(surface, footer_text, font_small, INK, (28, HEIGHT - 28))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Dance Simulation Prototype")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
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
            draw_footer_help(screen, font_small)

            pygame.display.flip()
    finally:
        udp_receiver.stop()
        pygame.quit()


if __name__ == "__main__":
    main()

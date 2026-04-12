"""
synapse_to_udp.py  --  Bridge: Synapse decoder tap → dance_sim_game.py UDP

Applies causal smoothing (ring buffer of K ticks) + scaled-space thresholds
matching the ridge decoder's output space ([-1,1] sticks, [0,1] buttons).
Emits bimanual: left stick, right stick, and button actions can each fire
independently per tick.
"""

import argparse
import json
import socket
import sys
from collections import deque
from pathlib import Path

import numpy as np
from synapse.client.taps import Tap
from synapse.api.datatype_pb2 import Tensor


DEFAULTS_PATH = Path(__file__).resolve().parent / "src" / "decoder" / "postproc_defaults.json"

AXIS_NAMES = {
    0: ("L_UP",    "L_DOWN"),
    1: ("L_RIGHT", "L_LEFT"),
    2: ("R_UP",    "R_DOWN"),
    3: ("R_RIGHT", "R_LEFT"),
}

BUTTON_NAMES = {
    0: "Y",
    1: "A",
    2: "X",
    3: "B",
    4: "LB",
    5: "RB",
    6: "LT",
    7: "RT",
}


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    arr = np.frombuffer(tensor.data, dtype=np.float32)
    if len(tensor.shape) > 0:
        expected = 1
        for d in tensor.shape:
            expected *= d
        if expected == arr.size:
            arr = arr.reshape(tuple(tensor.shape))
    return np.asarray(arr).reshape(-1)


def scale_to_model_space(raw: np.ndarray) -> np.ndarray:
    """raw: length-12 int16-ish → [-1,1] sticks (0:4), [0,1] buttons (4:12)."""
    scaled = raw.astype(np.float32) / 32767.0
    scaled[0:4] = np.clip(scaled[0:4], -1.0, 1.0)
    scaled[4:12] = np.clip(scaled[4:12], 0.0, 1.0)
    return scaled


def decode_actions(y_smooth: np.ndarray,
                   joy_thresh: float,
                   btn_thresh: float) -> list[str]:
    """Bimanual: left stick, right stick, and button emit independently.
    Returns 0, 1, 2, or 3 action strings per tick."""
    actions = []

    # Left stick (channels 0, 1)
    lx, ly = float(y_smooth[0]), float(y_smooth[1])
    l_mag = np.sqrt(lx * lx + ly * ly)
    if l_mag >= joy_thresh:
        # Pick dominant axis of left stick
        if abs(lx) >= abs(ly):
            actions.append(AXIS_NAMES[1][0] if lx >= 0 else AXIS_NAMES[1][1])
        else:
            actions.append(AXIS_NAMES[0][0] if ly >= 0 else AXIS_NAMES[0][1])

    # Right stick (channels 2, 3)
    rx, ry = float(y_smooth[2]), float(y_smooth[3])
    r_mag = np.sqrt(rx * rx + ry * ry)
    if r_mag >= joy_thresh:
        if abs(rx) >= abs(ry):
            actions.append(AXIS_NAMES[3][0] if rx >= 0 else AXIS_NAMES[3][1])
        else:
            actions.append(AXIS_NAMES[2][0] if ry >= 0 else AXIS_NAMES[2][1])

    # Buttons (channels 4..11)
    buttons = y_smooth[4:12]
    best_btn = int(np.argmax(buttons))
    if float(buttons[best_btn]) >= btn_thresh:
        actions.append(BUTTON_NAMES[best_btn])

    return actions


def main(argv=None):
    p = argparse.ArgumentParser(description="Bridge Synapse tap to dance game UDP")
    p.add_argument("uri",           nargs="?", default="127.0.0.1:647")
    p.add_argument("tap_name",      nargs="?", default="decoded_controls")
    p.add_argument("--host",        default="127.0.0.1")
    p.add_argument("--port",        type=int, default=5005)
    p.add_argument("--mode",        choices=["easy", "hard"], default="easy")
    p.add_argument("--K",           type=int, default=None,
                   help="Override smoother window size from defaults")
    p.add_argument("--joy-thresh",  type=float, default=None,
                   help="Override stick threshold (scaled space, e.g. 0.10)")
    p.add_argument("--btn-thresh",  type=float, default=None,
                   help="Override button threshold (scaled space, e.g. 0.15)")
    args = p.parse_args(argv)

    with open(DEFAULTS_PATH) as f:
        all_defaults = json.load(f)
    cfg = dict(all_defaults[args.mode])
    if args.K is not None:
        cfg["K"] = args.K
    if args.joy_thresh is not None:
        cfg["stick_thresh"] = args.joy_thresh
    if args.btn_thresh is not None:
        cfg["button_thresh"] = args.btn_thresh

    K = cfg["K"]
    joy_thresh = cfg["stick_thresh"]
    btn_thresh = cfg["button_thresh"]

    print(f"Connecting to {args.uri}, tap '{args.tap_name}'")
    print(f"Forwarding decoded actions → UDP {args.host}:{args.port}")
    print(f"mode={args.mode}  K={K}  joy_thresh={joy_thresh}  btn_thresh={btn_thresh}")

    tap = Tap(args.uri, verbose=True)

    taps = tap.list_taps()
    tap_names = [t.name for t in taps]
    print(f"Available taps: {tap_names}")

    if args.tap_name not in tap_names:
        print(f"ERROR: tap '{args.tap_name}' not found. Available: {tap_names}")
        sys.exit(1)

    tap.connect(args.tap_name)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    buf = deque(maxlen=K)

    try:
        for raw_msg in tap.stream(timeout_ms=100):
            if raw_msg is None or len(raw_msg) == 0:
                continue

            tensor = Tensor()
            tensor.ParseFromString(raw_msg)
            values = tensor_to_numpy(tensor)

            if values.size < 12:
                continue

            scaled = scale_to_model_space(values[:12].copy())
            buf.append(scaled)

            if len(buf) < K:
                continue  # warmup

            y_smooth = np.mean(buf, axis=0)
            actions = decode_actions(y_smooth, joy_thresh, btn_thresh)

            for action in actions:
                sock.sendto(action.encode("utf-8"), (args.host, args.port))
                print(f"→  {action}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        tap.disconnect()
        sock.close()


if __name__ == "__main__":
    main()
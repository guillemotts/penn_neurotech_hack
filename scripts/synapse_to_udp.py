"""
synapse_to_udp.py  --  Bridge: Synapse decoder tap → dance_sim_game.py UDP

Usage:
    python synapse_to_udp.py [<device_uri>] [<tap_name>] [--host <host>] [--port <port>]
                             [--joy-thresh <float>] [--btn-thresh <float>]

Defaults:
    device_uri  = 127.0.0.1:647
    tap_name    = decoded_controls
    host        = 127.0.0.1
    port        = 5005  (matches dance_sim_game.py UDP_PORT)

The tap sends a Tensor protobuf with a float32 array of shape [12]:
    [0] L Up/Down axis      (+→ L_UP,    -→ L_DOWN)
    [1] L Left/Right axis   (+→ L_RIGHT, -→ L_LEFT)
    [2] R Up/Down axis      (+→ R_UP,    -→ R_DOWN)
    [3] R Left/Right axis   (+→ R_RIGHT, -→ R_LEFT)
    [4] Y  [5] A  [6] X  [7] B  [8] LB  [9] RB  [10] LT  [11] RT

The dominant action across all axes and buttons is forwarded as a UTF-8 UDP
packet to the game, covering all 14 possible inputs.
"""

import argparse
import socket
import sys

import numpy as np
from synapse.client.taps import Tap
from synapse.api.datatype_pb2 import Tensor


# Axis index → (positive_name, negative_name)
AXIS_NAMES = {
    0: ("L_UP",    "L_DOWN"),
    1: ("L_RIGHT", "L_LEFT"),
    2: ("R_UP",    "R_DOWN"),
    3: ("R_RIGHT", "R_LEFT"),
}

# Button index within values[4:10] → UDP string
BUTTON_NAMES = {
    0: "Y",
    1: "A",
    2: "X",
    3: "B",
    4: "LB",
    5: "RB",
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


def decode_action(values: np.ndarray, joy_thresh: float, btn_thresh: float) -> str:
    """Return the dominant action string, or 'NOTHING' if below thresholds."""
    if values.shape[0] < 10:
        return "NOTHING"

    axes = values[:4]
    buttons = values[4:10]

    best_axis_idx = int(np.argmax(np.abs(axes)))
    best_axis_val = float(axes[best_axis_idx])
    best_axis_mag = abs(best_axis_val)

    best_btn_local = int(np.argmax(buttons))
    best_btn_val = float(buttons[best_btn_local])

    candidates = []

    if best_axis_mag >= joy_thresh:
        pos_name, neg_name = AXIS_NAMES[best_axis_idx]
        candidates.append((best_axis_mag, pos_name if best_axis_val >= 0 else neg_name))

    if best_btn_val >= btn_thresh:
        candidates.append((best_btn_val, BUTTON_NAMES[best_btn_local]))

    if not candidates:
        return "NOTHING"

    return max(candidates, key=lambda x: x[0])[1]


def main(argv=None):
    p = argparse.ArgumentParser(description="Bridge Synapse tap to dance game UDP")
    p.add_argument("uri",           nargs="?", default="127.0.0.1:647",      help="Synapse device URI")
    p.add_argument("tap_name",      nargs="?", default="decoded_controls",    help="Tap name")
    p.add_argument("--host",        default="127.0.0.1",                      help="UDP destination host")
    p.add_argument("--port",        type=int,  default=5005,                  help="UDP destination port")
    p.add_argument("--joy-thresh",  type=float, default=2500.0,               help="Joystick activation threshold")
    p.add_argument("--btn-thresh",  type=float, default=1200.0,               help="Button activation threshold")
    args = p.parse_args(argv)

    print(f"Connecting to {args.uri}, tap '{args.tap_name}'")
    print(f"Forwarding decoded actions → UDP {args.host}:{args.port}")
    print(f"Thresholds: joystick={args.joy_thresh}  button={args.btn_thresh}")

    tap = Tap(args.uri, verbose=True)

    taps = tap.list_taps()
    tap_names = [t.name for t in taps]
    print(f"Available taps: {tap_names}")

    if args.tap_name not in tap_names:
        print(f"ERROR: tap '{args.tap_name}' not found. Available: {tap_names}")
        sys.exit(1)

    tap.connect(args.tap_name)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        for raw_msg in tap.stream(timeout_ms=100):
            if raw_msg is None or len(raw_msg) == 0:
                continue

            tensor = Tensor()
            tensor.ParseFromString(raw_msg)
            values = tensor_to_numpy(tensor)

            if values.size == 0:
                continue

            action = decode_action(values, args.joy_thresh, args.btn_thresh)
            if action == "NOTHING":
                continue

            sock.sendto(action.encode("utf-8"), (args.host, args.port))
            print(f"→  {action}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        tap.disconnect()
        sock.close()


if __name__ == "__main__":
    main()

"""
listen_decoded_controls.py -- Stream decoded controller tensors from a
Synapse tap, apply causal smoothing + discretization using sweep-tuned
defaults from src/decoder/postproc_defaults.json, and emit symbol names.

Usage:
    python listen_decoded_controls.py <device-ip> [tap_name] [--mode easy|hard]
    python listen_decoded_controls.py --dry-run [--mode easy|hard]
"""
import argparse
import json
import os
from collections import deque
from pathlib import Path

import numpy as np

from src.decoder.postprocess import discretize_stick, discretize_button, combine

PRINT_ONLY_ON_CHANGE = False
PRINT_RAW = True

DEFAULTS_PATH = Path(__file__).resolve().parent / "src" / "decoder" / "postproc_defaults.json"


# ── Symbol name maps ─────────────────────────────────────────────────

def build_symbol_names(N):
    if N == 4:
        stick = {0: "E", 1: "N", 2: "W", 3: "S"}
    elif N == 6:
        stick = {0: "E", 1: "ENE", 2: "NNW", 3: "W", 4: "WSW", 5: "SSE"}
    elif N == 8:
        stick = {0: "E", 1: "NE", 2: "N", 3: "NW", 4: "W", 5: "SW", 6: "S", 7: "SE"}
    else:
        stick = {i: f"S{i}" for i in range(N)}
    buttons = {
        N: "A", N + 1: "B", N + 2: "X",
        N + 3: "Y", N + 4: "LB", N + 5: "RB",
    }
    return {**stick, **buttons}


# ── Scale conversion (matches ridge_train.py:scale_y) ────────────────

def scale_to_model_space(raw: np.ndarray) -> np.ndarray:
    scaled = raw / 32767.0
    scaled[0:4] = np.clip(scaled[0:4], -1.0, 1.0)
    scaled[4:10] = np.clip(scaled[4:10], 0.0, 1.0)
    return scaled


# ── Tensor parsing (Synapse protobuf) ────────────────────────────────

def tensor_to_numpy(tensor) -> np.ndarray:
    arr = np.frombuffer(tensor.data, dtype=np.float32)
    if len(tensor.shape) > 0:
        expected = 1
        for d in tensor.shape:
            expected *= d
        if expected == arr.size:
            arr = arr.reshape(tuple(tensor.shape))
    return np.asarray(arr).reshape(-1)


# ── Main ────────────────────────────────��────────────────────────────

def run_live(args, cfg):
    from synapse.client.taps import Tap
    try:
        from synapse.api.datatype_pb2 import Tensor
    except ImportError:
        from api.datatype_pb2 import Tensor

    K = cfg["K"]
    N = cfg["N"]
    stick_thresh = cfg["stick_thresh"]
    button_thresh = cfg["button_thresh"]
    names = build_symbol_names(N)
    buf = deque(maxlen=K)

    tap = Tap(args.device, verbose=True)
    taps = tap.list_taps()
    tap_names = [t.name for t in taps]

    print("Available taps:")
    for name in tap_names:
        print(" -", name)

    if args.tap not in tap_names:
        print(f"Tap '{args.tap}' not found.")
        return

    print(f"\nConnecting to tap '{args.tap}' on {args.device} ...")
    print(f"mode={args.mode}  K={K}  N={N}  stick_thresh={stick_thresh}  "
          f"button_thresh={button_thresh}\n")
    tap.connect(args.tap)

    last_sym = None

    try:
        for message in tap.stream():
            tensor = Tensor()
            tensor.ParseFromString(message)
            values = tensor_to_numpy(tensor)

            scaled = scale_to_model_space(values[:10].copy())
            buf.append(scaled)

            ts_ns = getattr(tensor, "timestamp_ns", 0)

            if len(buf) < K:
                print(f"[{ts_ns}] warmup ({len(buf)}/{K})")
                continue

            y_smooth = np.mean(buf, axis=0)
            row = y_smooth[None, :]

            stick_sym = discretize_stick(row, N, stick_thresh)[0]
            button_sym = discretize_button(row, button_thresh, stick_offset=N)[0]
            sym = int(combine(
                np.array([stick_sym]), np.array([button_sym]))[0])

            if PRINT_ONLY_ON_CHANGE and sym == last_sym:
                continue

            name = names.get(sym, "NO_OP") if sym >= 0 else "NO_OP"
            if PRINT_RAW:
                print(f"[{ts_ns}] symbol={sym:3d} ({name:4s})  "
                      f"smooth=[{y_smooth[0]:+.3f},{y_smooth[1]:+.3f}] "
                      f"btn_max={y_smooth[4:10].max():.3f}")
            else:
                print(f"[{ts_ns}] symbol={sym:3d} ({name})")

            last_sym = sym

    except KeyboardInterrupt:
        print("\nStopping listener.")
    finally:
        tap.disconnect()


def run_dry(cfg):
    K = cfg["K"]
    N = cfg["N"]
    stick_thresh = cfg["stick_thresh"]
    button_thresh = cfg["button_thresh"]
    names = build_symbol_names(N)
    buf = deque(maxlen=K)

    # Synthetic tensors: K east, K button-A, K idle (total 3*K)
    # First K fill the buffer with east → must emit E at t=K-1
    raw_tensors = []
    for _ in range(K):
        raw_tensors.append(np.array([30000, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    for _ in range(K):
        raw_tensors.append(np.array([0, 0, 0, 0, 0, 25000, 0, 0, 0, 0], dtype=np.float32))
    for _ in range(K):
        raw_tensors.append(np.zeros(10, dtype=np.float32))

    print(f"Dry run: mode cfg K={K} N={N} stick_thresh={stick_thresh} "
          f"button_thresh={button_thresh}")
    print(f"Feeding {len(raw_tensors)} synthetic tensors\n")

    east_emitted = False

    for i, raw in enumerate(raw_tensors):
        scaled = scale_to_model_space(raw.copy())
        buf.append(scaled)

        if len(buf) < K:
            print(f"[t={i:3d}] warmup ({len(buf)}/{K})")
            continue

        y_smooth = np.mean(buf, axis=0)
        row = y_smooth[None, :]

        stick_sym = discretize_stick(row, N, stick_thresh)[0]
        button_sym = discretize_button(row, button_thresh, stick_offset=N)[0]
        sym = int(combine(
            np.array([stick_sym]), np.array([button_sym]))[0])

        name = names.get(sym, "NO_OP") if sym >= 0 else "NO_OP"
        print(f"[t={i:3d}] symbol={sym:3d} ({name:4s})  "
              f"smooth=[{y_smooth[0]:+.3f},{y_smooth[1]:+.3f}] "
              f"btn_max={y_smooth[4:10].max():.3f}")

        # After K east tensors fill the buffer, must emit sector 0
        if i == K - 1 and sym == 0:
            east_emitted = True

    # Assertion: pure east vector repeated K times → symbol 0 (E)
    assert east_emitted, (
        f"FAIL: expected symbol 0 (E) at t={K - 1} after K={K} "
        f"east tensors filled buffer"
    )
    print("\nASSERT PASSED: pure east [30000,0,...] × K → symbol 0 (E)")


def main():
    parser = argparse.ArgumentParser(
        description="Listen to decoded controls with smoother + discretizer")
    parser.add_argument("device", nargs="?", default=None,
                        help="Synapse device IP or name")
    parser.add_argument("tap", nargs="?", default="decoded_controls",
                        help="Tap name (default: decoded_controls)")
    parser.add_argument("--mode", choices=["easy", "hard"], default="easy")
    parser.add_argument("--dry-run", action="store_true",
                        help="Feed synthetic tensors without connecting to Synapse")
    args = parser.parse_args()

    with open(DEFAULTS_PATH) as f:
        all_defaults = json.load(f)
    cfg = all_defaults[args.mode]
    print(f"Loaded defaults for mode={args.mode}: {cfg}")

    if args.dry_run:
        run_dry(cfg)
    else:
        if args.device is None:
            parser.error("device is required unless --dry-run is used")
        run_live(args, cfg)


if __name__ == "__main__":
    main()

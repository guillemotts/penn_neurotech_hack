import sys
import numpy as np

from synapse.client.taps import Tap

try:
    from synapse.api.datatype_pb2 import Tensor
except ImportError:
    from api.datatype_pb2 import Tensor


# ----------------------------
# Tune these if needed
# ----------------------------
JOY_THRESHOLD = 2500.0
BUTTON_THRESHOLD = 1200.0

PRINT_ONLY_ON_CHANGE = False   # set True to reduce spam
PRINT_RAW = True


AXIS_NAMES = {
    0: ("L_UP", "L_DOWN"),
    1: ("L_RIGHT", "L_LEFT"),
    2: ("R_UP", "R_DOWN"),
    3: ("R_RIGHT", "R_LEFT"),
}

BUTTON_NAMES = {
    4: "Y",
    5: "A",
    6: "X",
    7: "B",
    8: "LB",
    9: "RB",
    # 10: "LT",   # ignored
    # 11: "RT",   # ignored
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


def decode_action(values: np.ndarray) -> str:
    if values.shape[0] < 12:
        return "INVALID_OUTPUT"

    axes = values[:4]
    buttons = values[4:10]

    best_axis_idx = int(np.argmax(np.abs(axes)))
    best_axis_val = float(axes[best_axis_idx])
    best_axis_mag = abs(best_axis_val)

    best_button_local_idx = int(np.argmax(buttons))
    best_button_val = float(buttons[best_button_local_idx])
    best_button_global_idx = 4 + best_button_local_idx

    candidates = []

    if best_axis_mag >= JOY_THRESHOLD:
        pos_name, neg_name = AXIS_NAMES[best_axis_idx]
        axis_action = pos_name if best_axis_val >= 0 else neg_name
        candidates.append(("axis", best_axis_mag, axis_action))

    if best_button_val >= BUTTON_THRESHOLD:
        button_action = BUTTON_NAMES[best_button_global_idx]
        candidates.append(("button", best_button_val, button_action))

    if not candidates:
        return "NOTHING"

    # Winner-take-all, since only one control should be active at a time
    winner = max(candidates, key=lambda x: x[1])
    return winner[2]


def format_named_values(values: np.ndarray) -> str:
    return (
        f"LUD={values[0]:8.1f}  "
        f"LLR={values[1]:8.1f}  "
        f"RUD={values[2]:8.1f}  "
        f"RLR={values[3]:8.1f}  "
        f"Y={values[4]:8.1f}  "
        f"A={values[5]:8.1f}  "
        f"X={values[6]:8.1f}  "
        f"B={values[7]:8.1f}  "
        f"LB={values[8]:8.1f}  "
        f"RB={values[9]:8.1f}  "
        f"LT={values[10]:8.1f}  "
        f"RT={values[11]:8.1f}"
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python listen_decoded_controls.py <device-ip-or-name> [tap_name]")
        sys.exit(1)

    device_uri = sys.argv[1]
    tap_name = sys.argv[2] if len(sys.argv) > 2 else "decoded_controls"

    tap = Tap(device_uri, verbose=True)

    taps = tap.list_taps()
    tap_names = [t.name for t in taps]

    print("Available taps:")
    for name in tap_names:
        print(" -", name)

    if tap_name not in tap_names:
        print(f"Tap '{tap_name}' not found.")
        sys.exit(1)

    print(f"\nConnecting to tap '{tap_name}' on {device_uri} ...")
    print(f"Using JOY_THRESHOLD={JOY_THRESHOLD}, BUTTON_THRESHOLD={BUTTON_THRESHOLD}\n")
    tap.connect(tap_name)

    last_action = None

    try:
        for message in tap.stream():
            tensor = Tensor()
            tensor.ParseFromString(message)

            values = tensor_to_numpy(tensor)
            action = decode_action(values)

            if PRINT_ONLY_ON_CHANGE and action == last_action:
                continue

            ts_ns = getattr(tensor, "timestamp_ns", 0)

            if PRINT_RAW:
                print(f"[{ts_ns}] action={action:8s} raw={values}")
            else:
                print(f"[{ts_ns}] action={action:8s} {format_named_values(values)}")

            last_action = action

    except KeyboardInterrupt:
        print("\nStopping listener.")
    finally:
        tap.disconnect()


if __name__ == "__main__":
    main()
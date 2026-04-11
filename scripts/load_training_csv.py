import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

SYMBOLS = (
    [("ljoy", k) for k in range(8)]
    + [("rjoy", k) for k in range(8)]
    + [("button", name) for name in ["UP", "DOWN", "LEFT", "RIGHT", "LB", "RB"]]
)
N_SYMBOLS = 22
NEURAL_CHANNELS = list(range(64))   # ch_0..ch_63
RAW_CHANNELS = list(range(64, 76))  # ch_64..ch_75

# Pre-build lookup for symbol_to_index
_SYMBOL_STR_TO_IDX = {}
for _i, (_typ, _val) in enumerate(SYMBOLS):
    _SYMBOL_STR_TO_IDX[f"{_typ}_{_val}"] = _i


@dataclass
class Session:
    time_s: np.ndarray           # (T,) float
    neural: np.ndarray           # (64, T) float32
    raw: np.ndarray              # (12, T) float32
    fs_hz: float
    cue_times: np.ndarray        # (n_trials,) float, from cue log t_mono
    cue_symbols: list[str]       # (n_trials,) str, like "ljoy_3", "button_A"
    cue_indices: np.ndarray      # (n_trials,) int in [0, 22)
    cue_offset: float = 0.0     # original min of cue t_mono before rebase
    neural_offset: float = 0.0  # original min of neural time_s before rebase


def symbol_to_index(cue: str) -> int:
    """Parse 'ljoy_3' / 'button_A' -> index in SYMBOLS list. Raise ValueError on unknown."""
    try:
        return _SYMBOL_STR_TO_IDX[cue]
    except KeyError:
        raise ValueError(f"Unknown symbol: {cue!r}")


def load_session(neural_csv: Path, cue_csv: Path) -> Session:
    """
    Load a paired (neural, cue) session.
    neural_csv: time_s, timestamp_ns, ch_0..ch_75
    cue_csv: trial, t_mono, cue
    Returns Session with neural transposed to (64, T), raw to (12, T).
    fs_hz estimated from median diff of time_s.
    """
    ndf = pd.read_csv(neural_csv)
    cdf = pd.read_csv(cue_csv)

    time_s = ndf["time_s"].values.astype(np.float64)
    neural_cols = [f"ch_{i}" for i in NEURAL_CHANNELS]
    raw_cols = [f"ch_{i}" for i in RAW_CHANNELS]
    neural = ndf[neural_cols].values.astype(np.float32).T  # (64, T)
    raw = ndf[raw_cols].values.astype(np.float32).T        # (12, T)

    cue_times = cdf["t_mono"].values.astype(np.float64)

    # Rebase both time axes to start at 0
    cue_offset = float(cue_times.min())
    neural_offset = float(time_s.min())
    cue_times = cue_times - cue_offset
    time_s = time_s - neural_offset

    # Estimate sampling rate from span
    span = time_s.max() - time_s.min()
    if span <= 0:
        raise ValueError("neural time_s has zero span")
    fs_hz = len(time_s) / span
    if fs_hz < 50 or fs_hz > 100000:
        raise ValueError(f"fs_hz={fs_hz:.1f} out of sane range")

    cue_symbols = cdf["cue"].tolist()
    cue_indices = np.array([symbol_to_index(c) for c in cue_symbols], dtype=np.int64)

    if not (cue_times.max() < time_s.max() + 1.0):
        raise ValueError(
            f"Cue times misaligned: cue range [0, {cue_times.max():.4f}] "
            f"vs neural range [0, {time_s.max():.4f}]"
        )

    return Session(
        time_s=time_s,
        neural=neural,
        raw=raw,
        fs_hz=fs_hz,
        cue_times=cue_times,
        cue_symbols=cue_symbols,
        cue_indices=cue_indices,
        cue_offset=cue_offset,
        neural_offset=neural_offset,
    )


def recover_channel_map(session: Session, window_s: float = 0.8) -> dict:
    """
    For each raw channel c in [0, 12), compute per-symbol mean abs deflection
    in [cue_t, cue_t + window_s]. Return dict:
        {symbol_str: {'channel': int, 'score': float, 'sign': +1|-1}}
    """
    n_raw = session.raw.shape[0]
    baselines = np.median(session.raw, axis=1)  # (12,)

    # Group trials by symbol string
    sym_trials: dict[str, list[int]] = {}
    for i, sym in enumerate(session.cue_symbols):
        sym_trials.setdefault(sym, []).append(i)

    mapping = {}
    for sym, trial_idxs in sym_trials.items():
        # Compute mean deflection per raw channel across trials for this symbol
        deflections = np.zeros(n_raw, dtype=np.float64)
        count = 0
        for ti in trial_idxs:
            t_start = session.cue_times[ti]
            t_end = t_start + window_s
            mask = (session.time_s >= t_start) & (session.time_s < t_end)
            if not np.any(mask):
                continue
            segment = session.raw[:, mask]  # (12, W)
            deflections += np.abs(segment - baselines[:, None]).mean(axis=1)
            count += 1
        if count > 0:
            deflections /= count
        best_ch = int(np.argmax(np.abs(deflections)))
        score = float(np.abs(deflections[best_ch]))
        mapping[sym] = {"channel": best_ch, "score": score, "sign": 1}

    return mapping


def extract_trial_windows(
    session: Session,
    w0_ms: float = 0.0,
    w1_ms: float = 150.0,
    onset_offset_ms: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each cue, extract neural window [cue_t + onset_offset, cue_t + onset_offset + (w1-w0)].
    Skip trials where window exceeds session duration.
    Returns:
      X (M, 64, T_window) float32
      y (M,) int64 in [0, 22)
      kept_idx (M,) int, original trial indices
    """
    fs = session.fs_hz
    onset_s = onset_offset_ms / 1000.0
    win_len_s = (w1_ms - w0_ms) / 1000.0
    T_window = int(round(win_len_s * fs))
    t_max = session.time_s[-1]

    Xs, ys, kept = [], [], []
    for i, cue_t in enumerate(session.cue_times):
        win_start = cue_t + onset_s
        win_end = win_start + win_len_s
        if win_end > t_max:
            continue
        # Find nearest sample index for win_start
        idx_start = int(np.searchsorted(session.time_s, win_start))
        idx_end = idx_start + T_window
        if idx_end > session.neural.shape[1]:
            continue
        Xs.append(session.neural[:, idx_start:idx_end])
        ys.append(session.cue_indices[i])
        kept.append(i)

    if not Xs:
        return (
            np.empty((0, 64, T_window), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.intp),
        )
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    kept_idx = np.array(kept, dtype=np.intp)
    return X, y, kept_idx


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--neural", type=Path, required=True)
    p.add_argument("--cues", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output .npz with X, y")
    p.add_argument("--w0-ms", type=float, default=0.0)
    p.add_argument("--w1-ms", type=float, default=400.0)
    args = p.parse_args(argv)
    session = load_session(args.neural, args.cues)
    print(f"LOADED fs={session.fs_hz:.1f} T={session.neural.shape[1]} "
          f"n_cues={len(session.cue_times)}")
    print(f"ALIGN cue_offset={session.cue_offset:.3f} neural_span={session.time_s.max():.3f} fs_hz={session.fs_hz:.2f}")
    mapping = recover_channel_map(session)
    print("CHANNEL_MAP:")
    for sym, info in mapping.items():
        print(f"  {sym}: ch={info['channel']} score={info['score']:.3f} sign={info['sign']}")
    X, y, kept = extract_trial_windows(session, args.w0_ms, args.w1_ms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, X=X.astype(np.float32), y=y.astype(np.int64), kept_idx=kept)
    print(f"SAVED X={X.shape} y={y.shape} kept={len(kept)}/{len(session.cue_times)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

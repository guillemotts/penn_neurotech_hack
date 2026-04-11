import argparse, json, sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py

SYMBOLS = (
    [("ljoy", k) for k in range(8)]
    + [("rjoy", k) for k in range(8)]
    + [("button", name) for name in ["UP", "DOWN", "LEFT", "RIGHT", "LB", "RB"]]
)
N_SYMBOLS = 22
SYMBOL_STRS = [f"{t}_{v}" for t, v in SYMBOLS]

@dataclass
class H5Session:
    neural: np.ndarray           # (C, T) float32, C=76 for hard training
    frame_timestamps_ns: np.ndarray  # (n_frames,) int64, hardware timestamps
    samples_per_frame: int
    fs_hz: float
    cue_times_ns: np.ndarray     # (n_trials,) int64, hardware-clock aligned
    cue_symbols: list            # (n_trials,) str
    cue_indices: np.ndarray      # (n_trials,) int64
    session_start_t_mono_ns: int
    sync_prompt_t_mono_ns: int
    sync_release_t_mono_ns: int
    clock_offset_ns: int         # hw_ns - mono_ns, constant added to mono times

def load_h5(h5_path: Path) -> tuple:
    """
    Load synapsectl-produced HDF5. Returns (neural, frame_timestamps_ns, samples_per_frame, fs_hz).

    Structure assumed (to be verified against real file):
      - frames dataset: (n_frames, n_channels, samples_per_frame) or (n_frames, n_channels*samples_per_frame)
      - frame_timestamps_ns dataset: (n_frames,) int64
      - attrs: sample_rate_hz, n_channels

    Parse both common layouts. Raise ValueError with dataset listing if
    structure is unrecognized.
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if "frames" in f and "frame_timestamps_ns" in f:
            frames = f["frames"][:]
            ts = f["frame_timestamps_ns"][:].astype(np.int64)
            fs_hz = float(f.attrs.get("sample_rate_hz", 32000.0))
        elif "broadband" in f:
            g = f["broadband"]
            frames = g["data"][:]
            ts = g["timestamps_ns"][:].astype(np.int64) if "timestamps_ns" in g else g["frame_timestamps_ns"][:].astype(np.int64)
            fs_hz = float(g.attrs.get("sample_rate_hz", 32000.0))
        else:
            raise ValueError(f"Unrecognized h5 structure. Root keys: {keys}")
    if frames.ndim == 2:
        n_frames, flat = frames.shape
        raise ValueError(
            f"Flat frame layout (n_frames={n_frames}, flat={flat}); "
            f"need n_channels to reshape. Pass n_channels explicitly."
        )
    if frames.ndim == 3:
        n_frames, n_channels, samples_per_frame = frames.shape
        neural = frames.transpose(1, 0, 2).reshape(n_channels, n_frames * samples_per_frame).astype(np.float32)
        return neural, ts, samples_per_frame, fs_hz
    raise ValueError(f"Unexpected frames ndim={frames.ndim} shape={frames.shape}")

def load_cues(cues_json: Path) -> dict:
    with cues_json.open() as f:
        return json.load(f)

def detect_lb_edge(
    neural: np.ndarray,
    fs_hz: float,
    frame_timestamps_ns: np.ndarray,
    samples_per_frame: int,
    raw_channel_range: tuple = (64, 76),
    min_hold_s: float = 1.5,
) -> tuple:
    """
    Find the sync LB press in the raw controller channels.

    Strategy: per raw channel, find longest contiguous run of |value - median| > 5*MAD.
    Return (rising_edge_sample_idx, rising_edge_hw_ns, channel) for the longest hold.
    Raise ValueError if no hold >= min_hold_s found.
    """
    c0, c1 = raw_channel_range
    raw = neural[c0:c1, :]
    n_channels, T = raw.shape
    fs_int = int(round(fs_hz))
    min_samples = int(min_hold_s * fs_hz)
    best = None  # (length, start_idx, channel)
    for ch in range(n_channels):
        x = raw[ch]
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        active = np.abs(x - med) > 5 * mad
        # Find longest run of active
        diffs = np.diff(active.astype(np.int8), prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        for s, e in zip(starts, ends):
            length = e - s
            if length >= min_samples and (best is None or length > best[0]):
                best = (length, s, ch)
    if best is None:
        raise ValueError(f"No sync hold >= {min_hold_s}s found in channels {c0}..{c1}")
    length, start_idx, ch = best
    frame_idx = start_idx // samples_per_frame
    sample_in_frame = start_idx % samples_per_frame
    hw_ns = int(frame_timestamps_ns[frame_idx] + sample_in_frame * int(1e9 / fs_hz))
    return start_idx, hw_ns, ch

def align_and_build_session(
    h5_path: Path,
    cues_json_path: Path,
    n_channels: int = 76,
) -> H5Session:
    neural, frame_ts_ns, spf, fs_hz = load_h5(h5_path)
    assert neural.shape[0] == n_channels, f"Expected {n_channels} channels, got {neural.shape[0]}"
    cues = load_cues(cues_json_path)
    sync_prompt_mono = int(cues["sync_gesture"]["prompt_t_mono_ns"])
    sync_release_mono = int(cues["sync_gesture"]["release_t_mono_ns"])
    hold_s = (sync_release_mono - sync_prompt_mono) / 1e9
    min_hold = max(1.2, hold_s - 0.3)
    _, edge_hw_ns, _ = detect_lb_edge(neural, fs_hz, frame_ts_ns, spf, min_hold_s=min_hold)
    offset_ns = edge_hw_ns - sync_prompt_mono
    cue_times_ns = np.array(
        [int(t["t_mono_ns"]) + offset_ns for t in cues["trials"]],
        dtype=np.int64,
    )
    cue_symbols = [t["symbol"] for t in cues["trials"]]
    cue_indices = np.array([int(t["symbol_idx"]) for t in cues["trials"]], dtype=np.int64)
    return H5Session(
        neural=neural,
        frame_timestamps_ns=frame_ts_ns,
        samples_per_frame=spf,
        fs_hz=fs_hz,
        cue_times_ns=cue_times_ns,
        cue_symbols=cue_symbols,
        cue_indices=cue_indices,
        session_start_t_mono_ns=int(cues["session_start_t_mono_ns"]),
        sync_prompt_t_mono_ns=sync_prompt_mono,
        sync_release_t_mono_ns=sync_release_mono,
        clock_offset_ns=int(offset_ns),
    )

def extract_windows(
    session: H5Session,
    w_pre_ms: float = 0.0,
    w_post_ms: float = 400.0,
    onset_offset_ms: float = 150.0,
) -> tuple:
    """
    For each cue, extract neural window from frame timestamp.
    Returns (X: (M, C, T), y: (M,), kept_idx: (M,)).
    """
    fs = session.fs_hz
    T0 = int(session.frame_timestamps_ns[0])
    samples_per_frame = session.samples_per_frame
    n_frames = len(session.frame_timestamps_ns)
    total_samples = session.neural.shape[1]
    pre = int(w_pre_ms / 1000 * fs)
    post = int(w_post_ms / 1000 * fs)
    offset = int(onset_offset_ms / 1000 * fs)
    w_len = pre + post
    X_list, y_list, kept = [], [], []
    for i, t_ns in enumerate(session.cue_times_ns):
        rel_s = (int(t_ns) - T0) / 1e9
        center_idx = int(round(rel_s * fs)) + offset
        start = center_idx - pre
        end = start + w_len
        if start < 0 or end > total_samples:
            continue
        X_list.append(session.neural[:, start:end])
        y_list.append(int(session.cue_indices[i]))
        kept.append(i)
    if not X_list:
        return (
            np.empty((0, session.neural.shape[0], w_len), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=np.int64),
        np.array(kept, dtype=np.int64),
    )

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--cues", type=Path, required=True)
    p.add_argument("--n-channels", type=int, default=76)
    p.add_argument("--w-post-ms", type=float, default=400.0)
    p.add_argument("--onset-offset-ms", type=float, default=150.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    sess = align_and_build_session(args.h5, args.cues, n_channels=args.n_channels)
    print(f"LOADED fs={sess.fs_hz:.1f} shape={sess.neural.shape} n_cues={len(sess.cue_times_ns)}")
    print(f"CLOCK_OFFSET_NS={sess.clock_offset_ns}")
    X, y, kept = extract_windows(sess, w_post_ms=args.w_post_ms, onset_offset_ms=args.onset_offset_ms)
    print(f"EXTRACTED X={X.shape} y={y.shape} kept={len(kept)}/{len(sess.cue_times_ns)}")
    np.savez(args.out, X=X, y=y, kept_idx=kept, fs_hz=sess.fs_hz, clock_offset_ns=sess.clock_offset_ns)
    print(f"SAVED {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

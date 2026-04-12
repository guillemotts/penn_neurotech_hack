import argparse, sys
from pathlib import Path
import numpy as np
import h5py

N_NEURAL = 64
N_RAW = 12
N_TOTAL = 76
FS_HZ = 32000

def load_raw_decimated(h5_path: Path, start_pct: float = 0.0,
                       decimate_factor: int = 100) -> tuple:
    """Pass 1: read only raw channels (ch_64..ch_75), decimated by factor.

    Reads the h5 file in 1-second chunks and extracts only the 12 raw
    controller channels at every `decimate_factor`-th frame. At
    decimate_factor=100, produces ~320 Hz raw (12, T_dec) arrays, using
    ~50MB regardless of file duration.

    Returns (raw_dec (12, T_dec), start_frame_global, n_frames_total).
    start_frame_global is the full-rate frame index corresponding to
    raw_dec[:, 0] and is needed by Pass 2 to locate onsets in the file.
    """
    chunk_frames = FS_HZ  # 1s of frames per read
    assert chunk_frames % decimate_factor == 0, \
        "chunk_frames must be divisible by decimate_factor for consistent decimation"
    with h5py.File(h5_path, 'r') as f:
        ds = f['acquisition/ElectricalSeries']
        total = ds.shape[0]
        n_frames_total = total // N_TOTAL
        start_frame_global = int(start_pct / 100.0 * n_frames_total)
        # Align start_frame to decimate_factor so decimation phase is consistent
        start_frame_global = (start_frame_global // decimate_factor) * decimate_factor
        raw_chunks = []
        for i in range(start_frame_global, n_frames_total, chunk_frames):
            end_f = min(i + chunk_frames, n_frames_total)
            # Clip trailing partial chunk to a multiple of decimate_factor
            chunk_len = end_f - i
            chunk_len = (chunk_len // decimate_factor) * decimate_factor
            if chunk_len == 0:
                break
            end_f = i + chunk_len
            flat_start = i * N_TOTAL
            flat_end = end_f * N_TOTAL
            chunk = ds[flat_start:flat_end]
            reshaped = chunk.reshape(chunk_len, N_TOTAL)  # (chunk_len, 76)
            # Take raw channels only, decimated
            decimated = reshaped[::decimate_factor, N_NEURAL:].astype(np.float32)  # (T_dec_chunk, 12)
            raw_chunks.append(decimated.T)  # (12, T_dec_chunk)
    if not raw_chunks:
        return np.empty((N_RAW, 0), dtype=np.float32), start_frame_global, n_frames_total
    raw_dec = np.concatenate(raw_chunks, axis=1)
    return raw_dec, start_frame_global, n_frames_total

def find_active_region(raw: np.ndarray, window_frames: int) -> int:
    """Find first index where >4 raw channels have std > 5.0 in a window.
    Returns sample index (in the same rate as `raw`)."""
    if raw.shape[1] < window_frames:
        return 0
    step = max(window_frames // 2, 1)
    for start in range(0, raw.shape[1] - window_frames, step):
        chunk = raw[:, start:start + window_frames]
        active = (chunk.std(axis=1) > 5.0).sum()
        if active >= 4:
            return start
    return 0

def detect_onsets(raw: np.ndarray, fs_hz: float,
                  min_gap_ms: float = 150.0,
                  pattern_window_ms: float = 100.0,
                  baseline_seconds: float = 1.0) -> list:
    """
    Detect controller events as rising-edge transitions on any raw channel.

    Parameters are given in physical units so the same function works on
    either full-rate or decimated raw data.

    Returns list of (sample_idx, active_channel_pattern) where sample_idx
    is an index into `raw` (at the rate `fs_hz`) and
    active_channel_pattern is a tuple of 12 booleans.
    """
    n_ch, T = raw.shape
    min_gap_samples = max(int(min_gap_ms / 1000.0 * fs_hz), 1)
    pattern_window_samples = max(int(pattern_window_ms / 1000.0 * fs_hz), 1)
    baseline_samples = min(max(int(baseline_seconds * fs_hz), 1), T)
    baseline = np.median(raw[:, :baseline_samples], axis=1, keepdims=True)
    deviation = np.abs(raw - baseline)
    mad = np.median(deviation, axis=1, keepdims=True) + 1e-6
    active = deviation > 5 * mad  # (12, T) bool
    any_active = active.any(axis=0)  # (T,)
    edges = np.where(np.diff(any_active.astype(np.int8)) > 0)[0]
    onsets = []
    last = -min_gap_samples - 1
    for e in edges:
        if e - last < min_gap_samples:
            continue
        window_end = min(e + pattern_window_samples, T)
        pattern = tuple(active[ch, e:window_end].any() for ch in range(n_ch))
        onsets.append((int(e), pattern))
        last = e
    return onsets

def pattern_to_label(pattern: tuple) -> int:
    """Convert 12-bool channel pattern to integer label (0..2^12-1)."""
    return sum(1 << i for i, v in enumerate(pattern) if v)

def extract_neural_window(ds, onset_frame_fullrate: int,
                          w_post: int, onset_offset: int) -> np.ndarray:
    """Pass 2: read a single neural window (64, w_post) from the h5 dataset.

    Reads only `w_post * 76` samples starting at the requested frame.
    Returns None if the window falls outside the file.
    """
    start_frame = onset_frame_fullrate + onset_offset
    end_frame = start_frame + w_post
    if start_frame < 0:
        return None
    flat_start = start_frame * N_TOTAL
    flat_end = end_frame * N_TOTAL
    if flat_end > ds.shape[0]:
        return None
    chunk = ds[flat_start:flat_end]
    if chunk.size != w_post * N_TOTAL:
        return None
    reshaped = chunk.reshape(w_post, N_TOTAL)  # (w_post, 76)
    return reshaped[:, :N_NEURAL].T.astype(np.float32)  # (64, w_post)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, nargs="+", required=True,
                   help="One or more h5 files")
    p.add_argument("--start-pct", type=float, default=0.0,
                   help="Start reading from this percent of each file")
    p.add_argument("--w-post-ms", type=float, default=400.0)
    p.add_argument("--onset-offset-ms", type=float, default=150.0)
    p.add_argument("--min-gap-ms", type=float, default=150.0)
    p.add_argument("--decimate-factor", type=int, default=100,
                   help="Decimation factor for Pass 1 raw-channel scan")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    w_post = int(args.w_post_ms / 1000 * FS_HZ)
    onset_offset = int(args.onset_offset_ms / 1000 * FS_HZ)
    fs_dec = FS_HZ / args.decimate_factor

    all_X, all_y = [], []
    for h5_path in args.h5:
        print(f"Loading {h5_path}...")
        # --- Pass 1: decimated raw for onset detection ---
        raw_dec, start_frame_global, n_frames_total = load_raw_decimated(
            h5_path, start_pct=args.start_pct, decimate_factor=args.decimate_factor,
        )
        print(f"  raw_dec={raw_dec.shape} fs_dec={fs_dec:.1f}Hz "
              f"start_frame={start_frame_global} total_frames={n_frames_total}")

        # Optional: skip idle period at the start of the decimated raw
        active_window = max(int(0.5 * fs_dec), 1)  # 500ms window
        active_start_dec = find_active_region(raw_dec, window_frames=active_window)
        if active_start_dec > 0:
            raw_dec = raw_dec[:, active_start_dec:]
            start_frame_global += active_start_dec * args.decimate_factor
            print(f"  active region starts at dec-sample {active_start_dec} "
                  f"({active_start_dec / fs_dec:.1f}s)")

        # Detect onsets in the decimated raw (indices are in decimated space)
        onsets_dec = detect_onsets(
            raw_dec,
            fs_hz=fs_dec,
            min_gap_ms=args.min_gap_ms,
            pattern_window_ms=100.0,
            baseline_seconds=1.0,
        )
        print(f"  detected {len(onsets_dec)} onsets")

        # --- Pass 2: for each onset, read neural window from the h5 file ---
        X_list, y_list = [], []
        with h5py.File(h5_path, 'r') as f:
            ds = f['acquisition/ElectricalSeries']
            for onset_idx_dec, pattern in onsets_dec:
                onset_frame_fullrate = start_frame_global + onset_idx_dec * args.decimate_factor
                neural_win = extract_neural_window(
                    ds, onset_frame_fullrate, w_post, onset_offset,
                )
                if neural_win is None:
                    continue
                X_list.append(neural_win)
                y_list.append(pattern_to_label(pattern))
        if X_list:
            X = np.stack(X_list)
            y = np.array(y_list, dtype=np.int64)
        else:
            X = np.empty((0, N_NEURAL, w_post), dtype=np.float32)
            y = np.array([], dtype=np.int64)
        print(f"  extracted {X.shape[0]} trials, window=({N_NEURAL},{w_post})")
        print(f"  unique patterns: {len(np.unique(y))}")
        all_X.append(X)
        all_y.append(y)

    X_cat = np.concatenate(all_X)
    y_cat = np.concatenate(all_y)
    # Remap raw pattern labels to contiguous 0..K-1
    unique = np.unique(y_cat)
    remap = {int(v): i for i, v in enumerate(unique)}
    y_remapped = np.array([remap[int(v)] for v in y_cat], dtype=np.int64)
    print(f"\nTOTAL: X={X_cat.shape} unique_classes={len(unique)}")
    print(f"Per-class counts: {np.bincount(y_remapped).tolist()}")
    np.savez(args.out, X=X_cat.astype(np.float32), y=y_remapped,
             pattern_map={str(k): v for k, v in remap.items()},
             fs_hz=FS_HZ)
    print(f"SAVED {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

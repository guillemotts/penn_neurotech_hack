import numpy as np, sys, h5py
sys.path.insert(0, 'src')

H5_FILES = [
    '/lcars/home/k/ktasich/hack_training_data/train_1.h5',
    '/lcars/home/k/ktasich/hack_training_data/train_2.h5',
    '/lcars/home/k/ktasich/hack_training_data/train_5.h5',
    '/lcars/home/k/ktasich/hack_training_data/train_6.h5',
]
FS = 32000
N_TOTAL = 76
N_NEURAL = 64
DEC = 100  # decimation for onset detection

def stick_sector(x_neg, x_pos, y_neg, y_pos, thresh=10.0):
    """Map ch_64..67 values to 8-way sector or None (dead zone)."""
    x = x_pos - x_neg  # signed X
    y = y_pos - y_neg  # signed Y
    if abs(x) < thresh and abs(y) < thresh:
        return None
    angle = np.arctan2(y, x) * 180 / np.pi  # -180..180
    # Map to 0..7 (8 sectors, 45 deg each, starting from East=0)
    sector = int(((angle + 22.5) % 360) // 45)
    return sector

def process_file(h5_path, w_post_samp=12800, onset_offset_samp=4800):
    f = h5py.File(h5_path, 'r')
    ds = f['acquisition/ElectricalSeries']
    total_frames = ds.shape[0] // N_TOTAL
    # Decimated raw-only load
    raw_dec = np.zeros((12, (total_frames + DEC - 1) // DEC), dtype=np.float32)
    chunk_frames = FS  # 1s chunks
    for i in range(0, total_frames, chunk_frames):
        end_f = min(i + chunk_frames, total_frames)
        flat = ds[i*N_TOTAL:end_f*N_TOTAL].reshape(-1, N_TOTAL).T
        raw_chunk = flat[N_NEURAL:, ::DEC]
        end_col = i//DEC + raw_chunk.shape[1]
        end_col = min(end_col, raw_dec.shape[1])
        raw_dec[:, i//DEC:end_col] = raw_chunk[:, :end_col - i//DEC]
    # Find active region: any channel std > 5 in 5s window
    win = 5 * FS // DEC
    active_start_dec = 0
    for s in range(0, raw_dec.shape[1] - win, win // 2):
        if (raw_dec[:, s:s+win].std(axis=1) > 5).sum() >= 4:
            active_start_dec = s
            break
    raw_active = raw_dec[:, active_start_dec:]
    # Detect onsets on any_active signal
    baseline = np.median(raw_active[:, :FS//DEC], axis=1, keepdims=True)
    active_mask = (np.abs(raw_active - baseline) > 5 * (np.median(np.abs(raw_active - baseline), axis=1, keepdims=True) + 1e-6)).any(axis=0)
    edges = np.where(np.diff(active_mask.astype(np.int8)) > 0)[0]
    min_gap_dec = int(0.15 * FS / DEC)
    onsets_dec = []
    last = -min_gap_dec - 1
    for e in edges:
        if e - last >= min_gap_dec:
            onsets_dec.append(e + active_start_dec)
            last = e
    # For each onset, extract analog stick values + button state
    X_list, y_list = [], []
    for o_dec in onsets_dec:
        o_full = o_dec * DEC
        win_start = o_full + onset_offset_samp
        win_end = win_start + w_post_samp
        if win_start < 0 or win_end * N_TOTAL > ds.shape[0]:
            continue
        # Load full-rate window
        flat = ds[win_start*N_TOTAL:win_end*N_TOTAL].reshape(-1, N_TOTAL).T
        neural = flat[:N_NEURAL].astype(np.float32)
        raw_full = flat[N_NEURAL:].astype(np.float32)
        # Use mean over window to classify
        ch64, ch65, ch66, ch67 = raw_full[0:4].mean(axis=1)
        buttons = raw_full[4:10].mean(axis=1)  # ch_68..73
        btn_idx = int(np.argmax(np.abs(buttons))) if np.abs(buttons).max() > 20 else -1
        btn_active = np.abs(buttons[btn_idx]) > 20 if btn_idx >= 0 else False
        sector = stick_sector(ch64, ch65, ch66, ch67)
        # Label: 0..7 = stick sectors, 8..13 = buttons (ch_68..73)
        if btn_active:
            label = 8 + btn_idx
        elif sector is not None:
            label = sector
        else:
            continue
        X_list.append(neural)
        y_list.append(label)
    f.close()
    return X_list, y_list

all_X, all_y = [], []
for h5 in H5_FILES:
    print(f'Processing {h5}...')
    Xs, ys = process_file(h5)
    print(f'  {len(Xs)} trials, labels={np.bincount(ys, minlength=14).tolist()}')
    all_X.extend(Xs); all_y.extend(ys)

X = np.stack(all_X)
y = np.array(all_y, dtype=np.int64)
print(f'\nTOTAL X={X.shape} counts={np.bincount(y, minlength=14).tolist()}')
np.savez('/tmp/relabeled.npz', X=X.astype(np.float32), y=y, fs_hz=FS)
print('SAVED /tmp/relabeled.npz')

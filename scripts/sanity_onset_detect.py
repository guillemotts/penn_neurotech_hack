import numpy as np, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from load_training_csv import load_session
from penn_nh.decoder.linear_probe import LinearProbeFactory
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

s = load_session(Path('data/train_ani-1.csv'), Path('data/train_ani_1_seed10.csv'))
fs = s.fs_hz
raw = s.raw  # (12, T)

# Detect onsets: any raw channel exceeds 3*MAD from median
med = np.median(raw, axis=1, keepdims=True)
mad = np.median(np.abs(raw - med), axis=1, keepdims=True) + 1e-6
active = (np.abs(raw - med) > 5 * mad).any(axis=0)  # (T,)
# Rising edges
edges = np.where(np.diff(active.astype(int)) > 0)[0]
# Debounce: min 500ms between onsets
min_gap = int(0.5 * fs)
kept = [edges[0]] if len(edges) else []
for e in edges[1:]:
    if e - kept[-1] >= min_gap:
        kept.append(e)
onsets = np.array(kept)
print(f"Detected {len(onsets)} onsets, {len(s.cue_times)} cues")

# Assign each onset the nearest cue within 1s
cue_idx_sec = s.cue_times
onset_sec = s.time_s[onsets]
assigned = []
for os_ in onset_sec:
    diffs = np.abs(cue_idx_sec - os_)
    j = np.argmin(diffs)
    if diffs[j] < 1.0:
        assigned.append((os_, s.cue_indices[j]))
print(f"Assigned {len(assigned)} onsets to cues within 1s")
if len(assigned) < 50:
    print("TOO FEW"); sys.exit(0)

# Report median onset - cue offset
offsets_ms = [(o - cue_idx_sec[np.argmin(np.abs(cue_idx_sec - o))])*1000 for o,_ in assigned]
print(f"onset-cue offset: median={np.median(offsets_ms):.0f}ms  p10={np.percentile(offsets_ms,10):.0f}  p90={np.percentile(offsets_ms,90):.0f}  std={np.std(offsets_ms):.0f}")

# Build windows around detected onsets
w_s = int(0.4 * fs)
X, y = [], []
for o_sec, sym in assigned:
    start = int(np.searchsorted(s.time_s, o_sec))
    end = start + w_s
    if end > raw.shape[1]: continue
    X.append(raw[:, start:end])
    y.append(sym)
X = np.stack(X).astype(np.float32); y = np.array(y)
print(f"X={X.shape} y={y.shape} n_classes={len(np.unique(y))}")
if len(np.unique(y)) < 22:
    print(f"Only {len(np.unique(y))} classes present, adjusting n")

n_cls = int(y.max()) + 1
# Map classes to 0..k-1
uniq = np.unique(y)
remap = {u:i for i,u in enumerate(uniq)}
y2 = np.array([remap[v] for v in y])
k = len(uniq)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
for tr, te in skf.split(X, y2):
    probe = LinearProbeFactory(C=1.0, max_iter=1000).fit(X[tr], y2[tr], n=k)
    pred = probe.score(X[te]).argmax(axis=1)
    accs.append((pred == y2[te]).mean())
print(f"ONSET-aligned RAW probe: MEAN={np.mean(accs):.3f}  chance={1/k:.3f}  k={k}")

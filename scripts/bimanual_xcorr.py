"""
Bimanual cross-correlation analysis.

Cross-correlates ch_0..31 vs ch_32..63 activation patterns on
ljoy-only trials vs rjoy-only trials.

Verdict: if correlation > 0.8 on matched channels across stick sides,
bimanual control is dead -- it's a unimanual game.
"""
import sys, subprocess
from pathlib import Path
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# 1. Locate all session h5/cue pairs
# ---------------------------------------------------------------------------
data_dirs = [Path("data/newdata_apr11"), Path("data/newdata_apr12")]
pairs = []
for d in data_dirs:
    if not d.exists():
        continue
    for h5 in sorted(d.glob("smoke_Ani_seed*.h5")):
        seed = h5.stem.replace("smoke_Ani_seed", "")
        cue = d / f"cues_Ani_seed{seed}.json"
        if cue.exists():
            pairs.append((str(h5), str(cue), seed))

if not pairs:
    print("ERROR: no data found")
    sys.exit(1)

print(f"Found {len(pairs)} session pairs\n")

# ---------------------------------------------------------------------------
# 2. Extract windows for each session (reuse existing loader)
# ---------------------------------------------------------------------------
all_X, all_y = [], []
for h5, cue, seed in pairs:
    out = f"/tmp/biman_s{seed}.npz"
    if not Path(out).exists():
        r = subprocess.run(
            ["python", "scripts/load_h5_session.py",
             "--h5", h5, "--cues", cue, "--out", out],
            capture_output=True,
        )
        if r.returncode != 0:
            print(f"FAIL seed{seed}: {r.stderr.decode()[-200:]}")
            continue
    d = np.load(out)
    all_X.append(d["X"])
    all_y.append(d["y"])
    print(f"  seed{seed}: {d['X'].shape[0]} trials, X shape {d['X'].shape}")

X = np.concatenate(all_X)  # (M, 76, T)
y = np.concatenate(all_y)  # (M,)
print(f"\nPooled: X={X.shape}  y={y.shape}")
print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ---------------------------------------------------------------------------
# 3. Define ljoy vs rjoy trial masks
# ---------------------------------------------------------------------------
# symbol_idx 0-7 = ljoy (all 8 directions)
# symbol_idx 8-15 = rjoy (all 8 directions)
# Also define cardinal-only subsets for tighter analysis
ljoy_all = set(range(0, 8))
rjoy_all = set(range(8, 16))
ljoy_card = {0, 2, 4, 6}
rjoy_card = {8, 10, 12, 14}

mask_ljoy = np.isin(y, list(ljoy_all))
mask_rjoy = np.isin(y, list(rjoy_all))
mask_ljoy_card = np.isin(y, list(ljoy_card))
mask_rjoy_card = np.isin(y, list(rjoy_card))

print(f"\nTrial counts:")
print(f"  ljoy all:       {mask_ljoy.sum()}")
print(f"  rjoy all:       {mask_rjoy.sum()}")
print(f"  ljoy cardinals: {mask_ljoy_card.sum()}")
print(f"  rjoy cardinals: {mask_rjoy_card.sum()}")

# ---------------------------------------------------------------------------
# 4. Extract neural channels only (0-63) and split hemispheres
# ---------------------------------------------------------------------------
X_neural = X[:, :64, :]  # (M, 64, T) - neural only, drop controller ch 64-75

def compute_cross_corr(X_subset, label):
    """
    For a set of trials, compute:
    - Mean activation per channel across time (M, 64) -> mean over trials -> (64,)
    - Split into left array (ch 0-31) and right array (ch 32-63)
    - Correlate matched channels (ch_i vs ch_{i+32})
    - Also correlate the full 32-dim activation vectors
    """
    # Mean absolute activation per channel per trial (proxy for firing rate)
    # Using RMS per channel per trial as activation metric
    rms_per_trial = np.sqrt(np.mean(X_subset ** 2, axis=2))  # (M, 64)

    # Mean across trials
    mean_act = np.mean(rms_per_trial, axis=0)  # (64,)

    left = mean_act[:32]   # ch 0-31
    right = mean_act[32:]  # ch 32-63

    # Pearson correlation of the 32-dim activation vectors
    r_vec, p_vec = stats.pearsonr(left, right)

    # Per-channel matched correlation: for each of the 32 channel pairs,
    # correlate their trial-by-trial activations
    left_trials = rms_per_trial[:, :32]   # (M, 32)
    right_trials = rms_per_trial[:, 32:]  # (M, 32)

    per_ch_corrs = []
    for ch in range(32):
        r_ch, _ = stats.pearsonr(left_trials[:, ch], right_trials[:, ch])
        per_ch_corrs.append(r_ch)
    per_ch_corrs = np.array(per_ch_corrs)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  n_trials = {len(X_subset)}")
    print(f"{'='*60}")
    print(f"  Mean activation profile correlation (32-dim vector):")
    print(f"    r = {r_vec:.4f}  p = {p_vec:.2e}")
    print(f"  Per-channel matched correlation (ch_i vs ch_i+32):")
    print(f"    mean  r = {np.nanmean(per_ch_corrs):.4f}")
    print(f"    median r = {np.nanmedian(per_ch_corrs):.4f}")
    print(f"    min   r = {np.nanmin(per_ch_corrs):.4f}")
    print(f"    max   r = {np.nanmax(per_ch_corrs):.4f}")
    print(f"    # channels with r > 0.8: {np.sum(per_ch_corrs > 0.8)}/32")
    print(f"    # channels with r > 0.5: {np.sum(per_ch_corrs > 0.5)}/32")

    return r_vec, per_ch_corrs


# ---------------------------------------------------------------------------
# 5. Run analysis for each trial type
# ---------------------------------------------------------------------------
results = {}

if mask_ljoy.sum() >= 5:
    r_ljoy_vec, r_ljoy_ch = compute_cross_corr(
        X_neural[mask_ljoy], "LJOY-ONLY trials (all directions)")
    results["ljoy_all"] = (r_ljoy_vec, r_ljoy_ch)

if mask_rjoy.sum() >= 5:
    r_rjoy_vec, r_rjoy_ch = compute_cross_corr(
        X_neural[mask_rjoy], "RJOY-ONLY trials (all directions)")
    results["rjoy_all"] = (r_rjoy_vec, r_rjoy_ch)

if mask_ljoy_card.sum() >= 5:
    r_ljoy_card_vec, r_ljoy_card_ch = compute_cross_corr(
        X_neural[mask_ljoy_card], "LJOY CARDINALS ONLY (0,2,4,6)")
    results["ljoy_card"] = (r_ljoy_card_vec, r_ljoy_card_ch)

if mask_rjoy_card.sum() >= 5:
    r_rjoy_card_vec, r_rjoy_card_ch = compute_cross_corr(
        X_neural[mask_rjoy_card], "RJOY CARDINALS ONLY (8,10,12,14)")
    results["rjoy_card"] = (r_rjoy_card_vec, r_rjoy_card_ch)

# ---------------------------------------------------------------------------
# 6. Differential analysis: is the L/R pattern DIFFERENT for ljoy vs rjoy?
# ---------------------------------------------------------------------------
if mask_ljoy.sum() >= 5 and mask_rjoy.sum() >= 5:
    print(f"\n{'='*60}")
    print(f"  DIFFERENTIAL ANALYSIS: ljoy vs rjoy activation profiles")
    print(f"{'='*60}")

    rms_ljoy = np.sqrt(np.mean(X_neural[mask_ljoy] ** 2, axis=2))
    rms_rjoy = np.sqrt(np.mean(X_neural[mask_rjoy] ** 2, axis=2))

    mean_ljoy = np.mean(rms_ljoy, axis=0)  # (64,)
    mean_rjoy = np.mean(rms_rjoy, axis=0)  # (64,)

    # Laterality index per condition: (contralateral - ipsilateral) / (contra + ipsi)
    # For ljoy: left stick -> expect right hemisphere (ch 32-63) dominant
    # For rjoy: right stick -> expect left hemisphere (ch 0-31) dominant
    ljoy_lat = (np.mean(mean_ljoy[32:]) - np.mean(mean_ljoy[:32])) / \
               (np.mean(mean_ljoy[32:]) + np.mean(mean_ljoy[:32]) + 1e-12)
    rjoy_lat = (np.mean(mean_rjoy[:32]) - np.mean(mean_rjoy[32:])) / \
               (np.mean(mean_rjoy[:32]) + np.mean(mean_rjoy[32:]) + 1e-12)

    print(f"  ljoy laterality index (expect >0 if contralateral): {ljoy_lat:.4f}")
    print(f"  rjoy laterality index (expect >0 if contralateral): {rjoy_lat:.4f}")

    # Correlation between ljoy and rjoy mean profiles
    r_profiles, p_profiles = stats.pearsonr(mean_ljoy, mean_rjoy)
    print(f"  Correlation of ljoy vs rjoy 64-ch profiles: r={r_profiles:.4f} p={p_profiles:.2e}")

    # Split by hemisphere
    r_left_hemi, _ = stats.pearsonr(mean_ljoy[:32], mean_rjoy[:32])
    r_right_hemi, _ = stats.pearsonr(mean_ljoy[32:], mean_rjoy[32:])
    print(f"  Left hemisphere (ch 0-31) ljoy vs rjoy: r={r_left_hemi:.4f}")
    print(f"  Right hemisphere (ch 32-63) ljoy vs rjoy: r={r_right_hemi:.4f}")

# ---------------------------------------------------------------------------
# 7. VERDICT
# ---------------------------------------------------------------------------
print(f"\n{'#'*60}")
print(f"#  BIMANUAL VERDICT")
print(f"{'#'*60}")

dead = True  # assume bimanual is dead until proven otherwise
for key in ["ljoy_all", "rjoy_all"]:
    if key in results:
        r_vec, per_ch = results[key]
        frac_high = np.sum(per_ch > 0.8) / 32
        if r_vec > 0.8:
            print(f"  {key}: profile corr r={r_vec:.3f} > 0.8  --> HIGHLY CORRELATED")
        else:
            print(f"  {key}: profile corr r={r_vec:.3f} <= 0.8  --> some differentiation")
            dead = False
        if frac_high > 0.5:
            print(f"  {key}: {np.sum(per_ch > 0.8)}/32 channels have matched r>0.8 --> MOSTLY MIRRORED")
        else:
            print(f"  {key}: {np.sum(per_ch > 0.8)}/32 channels have matched r>0.8 --> NOT mirrored")
            dead = False

if mask_ljoy.sum() >= 5 and mask_rjoy.sum() >= 5:
    if r_profiles > 0.95:
        print(f"  ljoy vs rjoy profile similarity: r={r_profiles:.3f} --> IDENTICAL patterns")
    elif r_profiles > 0.8:
        print(f"  ljoy vs rjoy profile similarity: r={r_profiles:.3f} --> very similar")
    else:
        print(f"  ljoy vs rjoy profile similarity: r={r_profiles:.3f} --> DIFFERENT patterns")
        dead = False

print()
if dead:
    print("  >>> BIMANUAL IS DEAD. Both arrays fire identically regardless of")
    print("  >>> which stick is used. This is a unimanual game.")
else:
    print("  >>> BIMANUAL SHOWS PROMISE. There is some hemispheric differentiation")
    print("  >>> between ljoy and rjoy trials.")
print()

"""Check whether left-stick and right-stick inputs map to disjoint channel subsets."""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_h5_session import align_and_build_session, extract_windows  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "newdata_apr12"

LJOY_IDS = {0, 2, 4, 6}
RJOY_IDS = {8, 10, 12, 14}
N_NEURAL = 64


def main():
    # Collect all seeds except 6
    h5_files = sorted(DATA_DIR.glob("smoke_Ani_seed*.h5"))
    sessions = []
    for h5 in h5_files:
        seed = h5.stem.replace("smoke_Ani_seed", "")
        if seed == "6":
            print(f"SKIP seed 6 (contamination)")
            continue
        cues = DATA_DIR / f"cues_Ani_seed{seed}.json"
        if not cues.exists():
            print(f"SKIP seed {seed}: no cues JSON")
            continue
        print(f"Loading seed {seed} ...")
        try:
            sess = align_and_build_session(h5, cues, n_channels=76)
            X, y, kept = extract_windows(sess)
            # Keep only first 64 neural channels
            X = X[:, :N_NEURAL, :]
            sessions.append((X, y, seed))
            print(f"  X={X.shape}  y={y.shape}  kept={len(kept)}")
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not sessions:
        print("No sessions loaded!")
        return

    # Concatenate
    X_all = np.concatenate([s[0] for s in sessions], axis=0)
    y_all = np.concatenate([s[1] for s in sessions], axis=0)
    print(f"\nConcatenated: X={X_all.shape}  y={y_all.shape}")

    ljoy_mask = np.isin(y_all, list(LJOY_IDS))
    rjoy_mask = np.isin(y_all, list(RJOY_IDS))
    print(f"ljoy trials: {ljoy_mask.sum()},  rjoy trials: {rjoy_mask.sum()}")

    # Step 3: mean activation per channel
    X_ljoy = X_all[ljoy_mask]  # (n_ljoy, 64, T)
    X_rjoy = X_all[rjoy_mask]  # (n_rjoy, 64, T)

    a_ljoy = X_ljoy.mean(axis=(0, 2))  # (64,)
    a_rjoy = X_rjoy.mean(axis=(0, 2))  # (64,)

    # Step 4: Pearson correlation
    r, p = pearsonr(a_ljoy, a_rjoy)
    print(f"\nPearson r(a_ljoy, a_rjoy) = {r:.4f}  (p={p:.2e})")

    # Step 5: per-trial activation, then mean over trials -> top-10 by |mean|
    # v_ljoy[c] = mean over time per trial -> shape (n_ljoy_trials,) per channel
    v_ljoy = X_ljoy.mean(axis=2)  # (n_ljoy, 64)
    v_rjoy = X_rjoy.mean(axis=2)  # (n_rjoy, 64)

    mean_v_ljoy = v_ljoy.mean(axis=0)  # (64,)
    mean_v_rjoy = v_rjoy.mean(axis=0)  # (64,)

    top10_ljoy = np.argsort(np.abs(mean_v_ljoy))[::-1][:10]
    top10_rjoy = np.argsort(np.abs(mean_v_rjoy))[::-1][:10]

    intersection = set(top10_ljoy) & set(top10_rjoy)

    print(f"\nTop-10 ljoy channels (by |mean activation|): {top10_ljoy.tolist()}")
    print(f"  values: {[f'{mean_v_ljoy[c]:.4f}' for c in top10_ljoy]}")
    print(f"Top-10 rjoy channels (by |mean activation|): {top10_rjoy.tolist()}")
    print(f"  values: {[f'{mean_v_rjoy[c]:.4f}' for c in top10_rjoy]}")
    print(f"\nIntersection size: {len(intersection)}")
    if intersection:
        print(f"Shared channels: {sorted(intersection)}")
    else:
        print("No shared channels -- fully disjoint top-10 sets!")


if __name__ == "__main__":
    main()

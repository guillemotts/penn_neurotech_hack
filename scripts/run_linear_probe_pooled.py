"""Pooled 5-fold stratified linear probe across all training sessions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from penn_nh.features.stack import FeatureConfigLowRate, LowRateFeatureStack
from scripts.load_training_csv import SYMBOLS, load_session, extract_trial_windows

N_SYMBOLS = 22
SYM_NAMES = [f"{t}_{v}" for t, v in SYMBOLS]
CHANCE = 1.0 / N_SYMBOLS

PAIRS = [
    ("data/train_ani-1.csv", "data/train_ani_1_seed10.csv"),
    ("data/train_ani-2.csv", "data/train_ani_2_seed11.csv"),
    ("data/train_ani-3.csv", "data/train_ani_3_seed12.csv"),
    ("data/train_andy-1.csv", "data/train_andrew_1-seed13.csv"),
]


def main():
    all_X = []
    all_y = []
    n_sessions = 0

    for neural_path, cue_path in PAIRS:
        label = Path(neural_path).stem
        try:
            session = load_session(Path(neural_path), Path(cue_path))
        except Exception as e:
            print(f"SKIP {label}: {e}")
            continue

        X, y, kept = extract_trial_windows(session, w0_ms=0.0, w1_ms=400.0, onset_offset_ms=50.0)
        if len(kept) == 0:
            print(f"SKIP {label}: no valid trials")
            continue

        print(f"  {label}: fs={session.fs_hz:.1f} trials={X.shape[0]}/{len(session.cue_times)} "
              f"X={X.shape}")

        # Per-session feature stack: fit zscore, transform, flatten
        cfg = FeatureConfigLowRate(fs_hz=session.fs_hz)
        fs = LowRateFeatureStack(cfg)
        fs.fit_zscore([X[i] for i in range(X.shape[0])])
        X_transformed = np.stack([fs.transform(X[i]) for i in range(X.shape[0])])

        all_X.append(X_transformed)
        all_y.append(y)
        n_sessions += 1

    if n_sessions == 0:
        print("No sessions loaded. Aborting.")
        return

    # Truncate to common T_window across sessions (slightly different fs)
    min_T = min(x.shape[2] for x in all_X)
    all_X = [x[:, :, :min_T] for x in all_X]
    X_pool_3d = np.concatenate(all_X, axis=0)
    X_pool = X_pool_3d.reshape(X_pool_3d.shape[0], -1)
    y_pool = np.concatenate(all_y, axis=0)
    n_unique = len(np.unique(y_pool))
    print(f"\nDATA X={X_pool.shape} y={y_pool.shape} n_sessions={n_sessions} classes={n_unique}")

    # Per-class counts
    print("\nPER-CLASS COUNTS:")
    for i in range(N_SYMBOLS):
        cnt = (y_pool == i).sum()
        print(f"  {SYM_NAMES[i]:>14s}: {cnt}")

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_accs = []
    all_preds = np.empty_like(y_pool)
    all_true = np.empty_like(y_pool)
    ptr = 0

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_pool, y_pool)):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=0, solver="lbfgs")),
        ])
        pipe.fit(X_pool[train_idx], y_pool[train_idx])
        pred = pipe.predict(X_pool[test_idx])
        fold_acc = (pred == y_pool[test_idx]).mean()
        fold_accs.append(fold_acc)
        print(f"  Fold {fold_i}: acc={fold_acc:.4f} (test={len(test_idx)})")

        n_te = len(test_idx)
        all_preds[ptr:ptr + n_te] = pred
        all_true[ptr:ptr + n_te] = y_pool[test_idx]
        ptr += n_te

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    ratio = mean_acc / CHANCE
    print(f"\nMEAN ACC: {mean_acc:.4f} +/- {std_acc:.4f}  (chance={CHANCE:.4f}, ratio={ratio:.2f}x)")

    # Confusion matrix
    cm = np.zeros((N_SYMBOLS, N_SYMBOLS), dtype=int)
    for t, p in zip(all_true, all_preds):
        cm[t, p] += 1

    print(f"\nCONFUSION MATRIX ({N_SYMBOLS}x{N_SYMBOLS}):")
    header = "                " + " ".join(f"{i:4d}" for i in range(N_SYMBOLS))
    print(header)
    for i in range(N_SYMBOLS):
        row = f"  {SYM_NAMES[i]:>14s} " + " ".join(f"{cm[i, j]:4d}" for j in range(N_SYMBOLS))
        print(row)

    # Per-symbol recall sorted ascending
    print("\nPER-SYMBOL RECALL (ascending):")
    recalls = []
    for i in range(N_SYMBOLS):
        total = cm[i].sum()
        recall = cm[i, i] / total if total > 0 else 0.0
        recalls.append((SYM_NAMES[i], recall, total))
    recalls.sort(key=lambda x: x[1])
    for name, rec, tot in recalls:
        print(f"  {name:>14s}: {rec:.3f}  (n={tot})")

    # Top 10 confusion pairs
    print("\nTOP 10 CONFUSION PAIRS:")
    pairs = []
    for i in range(N_SYMBOLS):
        for j in range(N_SYMBOLS):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], SYM_NAMES[i], SYM_NAMES[j]))
    pairs.sort(reverse=True)
    for count, true_sym, pred_sym in pairs[:10]:
        print(f"  {true_sym:>14s} -> {pred_sym:<14s}: {count}")


if __name__ == "__main__":
    main()

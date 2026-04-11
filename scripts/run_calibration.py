import argparse, json, sys

import numpy as np
import yaml
from pathlib import Path

from penn_nh.features.stack import FeatureConfig, FeatureStack
from penn_nh.decoder.stage1_cnn import Stage1Config, Stage1Factory
from penn_nh.decoder.linear_probe import LinearProbeFactory
from penn_nh.eval.sweep_n import sweep_n
from penn_nh.collect.labeler import label_batch


def load_trials(path: Path) -> dict:
    """
    Expects .npz with keys: raw (M, C, T), stick_xy (M, 2).
    Returns dict of arrays.
    """
    data = np.load(path)
    return {"raw": data["raw"], "stick_xy": data["stick_xy"]}


def build_factory(arch: str):
    if arch == "eegnet" or arch == "dense":
        return Stage1Factory(Stage1Config(arch=arch, epochs=20))
    if arch == "linear":
        return LinearProbeFactory()
    raise ValueError(f"unknown arch {arch}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output YAML config path")
    p.add_argument("--player", type=str, required=True)
    p.add_argument("--arch", choices=["eegnet", "dense", "linear"], default="eegnet")
    p.add_argument("--n-grid", type=int, nargs="+", default=[6, 8, 10, 12, 16, 20, 24, 32])
    p.add_argument("--theta-n", type=int, default=26)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    data = load_trials(args.trials)
    X, stick = data["raw"], data["stick_xy"]

    # Split by val_frac (last val_frac as val, first as train).
    M = X.shape[0]
    split = int(M * (1.0 - args.val_frac))
    X_feat_train = X[:split]
    stick_train = stick[:split]
    X_feat_val = X[split:]
    stick_val = stick[split:]

    result = sweep_n(
        X_train=X_feat_train,
        stick_xy_train=stick_train,
        X_val=X_feat_val,
        stick_xy_val=stick_val,
        factory=build_factory(args.arch),
        n_grid=np.array(args.n_grid, dtype=int),
        theta_grid=np.linspace(0.0, 0.5, args.theta_n),
    )

    best_idx = int(np.argmax(result["B"]))
    cfg_out = {
        "player": args.player,
        "n": int(result["N"][best_idx]),
        "theta": float(result["theta"][best_idx]),
        "alpha": 0.5,
        "tau": 0.1,
        "window_w0_ms": 0.0,
        "window_w1_ms": 100.0,
        "stage1_arch": args.arch,
        "feature_config": {
            "fs_hz": 32000.0,
            "bandpass_low_hz": 300.0,
            "bandpass_high_hz": 3000.0,
            "bin_ms": 5.0,
            "spike_threshold_k": 3.0,
            "min_isi_ms": 1.0,
            "use_spike_counts": True,
            "use_spike_power": True,
        },
        "sweep_result": {
            "N_grid": result["N"].tolist(),
            "B": [float(x) for x in result["B"]],
            "theta_at_best": [float(x) for x in result["theta"]],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False)
    print(
        f"CALIBRATED player={args.player} N={cfg_out['n']} "
        f"theta={cfg_out['theta']:.3f} B={result['B'][best_idx]:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

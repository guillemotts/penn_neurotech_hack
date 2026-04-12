import argparse
import numpy as np
from pathlib import Path
from penn_nh.decoder.stage1_cnn import Stage1Factory, Stage1Config

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["eegnet", "dense"], default="eegnet")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=20)
    p.add_argument("--n-classes", type=int, default=22)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("artifacts/decoder_dummy.onnx"))
    args = p.parse_args()

    rng = np.random.default_rng(0)
    M = max(100, args.n_classes * 5)
    X = rng.standard_normal((M, args.channels, args.timesteps)).astype(np.float32)
    y = rng.integers(0, args.n_classes, size=M).astype(np.int64)
    for k in range(args.n_classes):
        y[k] = k  # ensure every class present

    factory = Stage1Factory(Stage1Config(
        arch=args.arch, epochs=args.epochs, batch_size=32, device="cpu", seed=0
    ))
    fitted = factory.fit(X, y, args.n_classes)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fitted.export_onnx(str(args.out), example_shape=(1, args.channels, args.timesteps))
    print(f"EXPORTED {args.out}")
    print(f"  arch={args.arch}")
    print(f"  input_shape=(1, {args.channels}, {args.timesteps})")
    print(f"  n_classes={args.n_classes}")

if __name__ == "__main__":
    main()
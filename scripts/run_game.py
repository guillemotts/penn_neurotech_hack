import argparse
import sys

import numpy as np

from penn_nh.game.interface import UnpacedTargetHit
from penn_nh.game.session import Session
from penn_nh.game.ui import NullRenderer, PrintRenderer
from penn_nh.integration.mock_decoder import mock_decoder_stream


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Penn NH game runner")
    parser.add_argument("--mode", choices=["mock-confusion"], default="mock-confusion")
    parser.add_argument(
        "--confusion",
        type=str,
        required=True,
        help="Path to .npz with key 'C' holding (N,N) row-stochastic matrix",
    )
    parser.add_argument("--mean-interval", type=float, default=0.35)
    parser.add_argument("--rejection-rate", type=float, default=0.0)
    parser.add_argument("--session-seconds", type=float, default=60.0)
    parser.add_argument("--time-source", choices=["iterator", "wall"], default="iterator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    data = np.load(args.confusion)
    if "C" not in data.files:
        print("ERROR: confusion .npz must contain key 'C'", file=sys.stderr)
        return 2
    C = data["C"]
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        print("ERROR: C must be square", file=sys.stderr)
        return 2
    n = int(C.shape[0])

    rng = np.random.default_rng(args.seed)
    game = UnpacedTargetHit(n=n, session_seconds=args.session_seconds, rng=rng)
    stream = mock_decoder_stream(
        game=game,
        confusion=C,
        mean_interval_s=args.mean_interval,
        rejection_rate=args.rejection_rate,
        rng=rng,
    )
    renderer = NullRenderer() if args.quiet else PrintRenderer()
    session = Session(
        game=game,
        decoder_stream=stream,
        renderer=renderer,
        time_source=args.time_source,
        return_history=False,
    )
    report = session.run()
    print(
        f"FINAL N={report['N']} S_c={report['S_c']} S_i={report['S_i']} "
        f"rej={report['rejections']} B={report['B']:.4f} duration={report['duration']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

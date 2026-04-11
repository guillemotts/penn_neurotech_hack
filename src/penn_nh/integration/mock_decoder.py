import numpy as np
from typing import Iterator, Optional, Tuple

from penn_nh.game.interface import Game


def mock_decoder_stream(
    game: Game,
    confusion: np.ndarray,
    mean_interval_s: float,
    rejection_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Iterator[Tuple[Optional[int], float, float]]:
    """
    Yields (symbol_or_None, confidence, t) tuples.
    symbol=None denotes rejection.
    t increments by exponential(mean_interval_s) each step.
    """
    confusion = np.asarray(confusion, dtype=float)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be square")
    if not np.allclose(confusion.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("confusion rows must sum to 1.0")
    if not (0 <= rejection_rate <= 1):
        raise ValueError("rejection_rate must be in [0, 1]")
    if mean_interval_s <= 0:
        raise ValueError("mean_interval_s must be > 0")

    if rng is None:
        rng = np.random.default_rng()

    n = confusion.shape[0]
    t = 0.0

    while True:
        dt = rng.exponential(mean_interval_s)
        t += dt
        if t >= game.SESSION_SECONDS:
            return

        if rng.random() < rejection_rate:
            yield (None, 0.0, t)
        else:
            target = game.current_target()
            symbol = int(rng.choice(n, p=confusion[target]))
            yield (symbol, 1.0, t)

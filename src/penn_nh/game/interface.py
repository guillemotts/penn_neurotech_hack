from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from penn_nh.eval.bitrate import shenoy_b


class Game(ABC):
    N: int
    SESSION_SECONDS: float = 60.0
    queue_depth: int = 1

    @abstractmethod
    def current_target(self) -> int: ...

    @abstractmethod
    def upcoming_targets(self) -> list[int]: ...

    @abstractmethod
    def submit(self, symbol: int, t: float) -> dict: ...

    @abstractmethod
    def on_rejection(self, t: float) -> None: ...

    @abstractmethod
    def bitrate(self) -> float: ...


class UnpacedTargetHit(Game):
    """
    iid uniform targets over [0, N). submit() and on_rejection() both
    advance the queue (pop head, push new iid tail). queue_depth controls
    lookahead; default 1.
    """

    def __init__(
        self,
        n: int,
        queue_depth: int = 1,
        session_seconds: float = 60.0,
        rng: Optional[np.random.Generator] = None,
    ):
        if n < 3:
            raise ValueError("n must be >= 3")
        if queue_depth < 1:
            raise ValueError("queue_depth must be >= 1")
        if session_seconds <= 0:
            raise ValueError("session_seconds must be > 0")

        self.N = n
        self.queue_depth = queue_depth
        self.SESSION_SECONDS = session_seconds
        self._rng = rng if rng is not None else np.random.default_rng()
        self._queue = [int(self._rng.integers(0, self.N)) for _ in range(queue_depth)]
        self.S_c = 0
        self.S_i = 0
        self.rejections = 0
        self.history: list[tuple] = []

    def _advance(self) -> None:
        self._queue.pop(0)
        self._queue.append(int(self._rng.integers(0, self.N)))

    def current_target(self) -> int:
        if not self._queue:
            raise RuntimeError("Queue is empty")
        return self._queue[0]

    def upcoming_targets(self) -> list[int]:
        return list(self._queue)

    def submit(self, symbol: int, t: float) -> dict:
        target = self._queue[0]
        if symbol == target:
            outcome = "correct"
            self.S_c += 1
        else:
            outcome = "incorrect"
            self.S_i += 1
        self.history.append((t, "submit", target, symbol, outcome))
        self._advance()
        return {"outcome": outcome, "target": target, "t": t}

    def on_rejection(self, t: float) -> None:
        self.rejections += 1
        self.history.append((t, "reject", self._queue[0], None, None))
        self._advance()

    def bitrate(self) -> float:
        return shenoy_b(self.S_c, self.S_i, self.N, self.SESSION_SECONDS)

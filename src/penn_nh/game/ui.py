import sys
from typing import Protocol


class Renderer(Protocol):
    def show_cue(self, target: int) -> None: ...
    def show_upcoming(self, upcoming: list[int]) -> None: ...
    def show_outcome(self, outcome: str, target: int, decoded: int) -> None: ...
    def show_rejection(self) -> None: ...
    def show_bitrate(self, b: float) -> None: ...
    def show_final(self, report: dict) -> None: ...
    def clear(self) -> None: ...


class NullRenderer:
    """No-op implementation for tests. Implements Renderer protocol."""

    def show_cue(self, target: int) -> None:
        pass

    def show_upcoming(self, upcoming: list[int]) -> None:
        pass

    def show_outcome(self, outcome: str, target: int, decoded: int) -> None:
        pass

    def show_rejection(self) -> None:
        pass

    def show_bitrate(self, b: float) -> None:
        pass

    def show_final(self, report: dict) -> None:
        pass

    def clear(self) -> None:
        pass


class PrintRenderer:
    """Minimal stdout renderer. One line per event. Fixed-width fields."""

    def show_cue(self, target: int) -> None:
        sys.stdout.write(f"CUE  target={target:2d}\n")
        sys.stdout.flush()

    def show_upcoming(self, upcoming: list[int]) -> None:
        pass

    def show_outcome(self, outcome: str, target: int, decoded: int) -> None:
        sys.stdout.write(f"HIT  target={target:2d} decoded={decoded:2d} outcome={outcome:<8s}\n")
        sys.stdout.flush()

    def show_rejection(self) -> None:
        sys.stdout.write("REJ\n")
        sys.stdout.flush()

    def show_bitrate(self, b: float) -> None:
        sys.stdout.write(f"B={b:6.3f} bps\n")
        sys.stdout.flush()

    def show_final(self, report: dict) -> None:
        sys.stdout.write(
            f"N={report['N']} S_c={report['S_c']} S_i={report['S_i']} "
            f"rejections={report['rejections']} B={report['B']:.3f} "
            f"duration={report['duration']:.2f}\n"
        )
        sys.stdout.flush()

    def clear(self) -> None:
        pass

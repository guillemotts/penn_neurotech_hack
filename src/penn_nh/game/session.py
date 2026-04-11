import time
from typing import Iterator, Literal, Optional, Tuple

from penn_nh.game.interface import Game
from penn_nh.game.ui import NullRenderer, Renderer

DecoderEvent = Tuple[Optional[int], float, float]  # (symbol|None, confidence, t)


class Session:
    def __init__(
        self,
        game: Game,
        decoder_stream: Iterator[DecoderEvent],
        renderer: Optional[Renderer] = None,
        time_source: Literal["iterator", "wall"] = "wall",
        bitrate_tick_hz: float = 4.0,
        return_history: bool = False,
    ):
        if bitrate_tick_hz <= 0:
            raise ValueError("bitrate_tick_hz must be > 0")
        if time_source not in ("iterator", "wall"):
            raise ValueError("time_source must be 'iterator' or 'wall'")

        self.game = game
        self.decoder_stream = decoder_stream
        self.renderer: Renderer = renderer if renderer is not None else NullRenderer()
        self.time_source = time_source
        self.bitrate_tick_hz = bitrate_tick_hz
        self.return_history = return_history

    def run(self) -> dict:
        wall = self.time_source == "wall"
        game = self.game
        renderer = self.renderer

        t0 = time.monotonic() if wall else 0.0
        last_tick = t0
        t_elapsed = 0.0

        renderer.show_cue(game.current_target())

        for event in self.decoder_stream:
            symbol, conf, t_iter = event

            if wall:
                t_elapsed = time.monotonic() - t0
            else:
                t_elapsed = t_iter

            if t_elapsed >= game.SESSION_SECONDS:
                break

            if wall and t_iter > t_elapsed:
                time.sleep(t_iter - t_elapsed)
                t_elapsed = time.monotonic() - t0
                if t_elapsed >= game.SESSION_SECONDS:
                    break

            if symbol is None:
                game.on_rejection(t_elapsed)
                renderer.show_rejection()
            else:
                result = game.submit(symbol, t_elapsed)
                renderer.show_outcome(result["outcome"], result["target"], symbol)

            renderer.show_cue(game.current_target())

            now = time.monotonic() if wall else t_iter
            if now - last_tick >= 1.0 / self.bitrate_tick_hz:
                renderer.show_bitrate(game.bitrate())
                last_tick = now

        duration = min(t_elapsed, game.SESSION_SECONDS)

        report = {
            "N": game.N,
            "S_c": game.S_c,
            "S_i": game.S_i,
            "rejections": game.rejections,
            "B": game.bitrate(),
            "duration": duration,
        }
        if self.return_history:
            report["history"] = list(game.history)

        renderer.show_final(report)
        return report

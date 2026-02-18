"""Human-play entrypoint for Bang AI."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bang_ai.config import FPS, SHOW_GAME
from bang_ai.game import HumanGame
from bang_ai.logging_utils import configure_logging, log_run_context


def run_user() -> None:
    configure_logging()
    game = HumanGame()
    log_run_context(
        "play-user",
        {
            "render": SHOW_GAME,
            "fps": FPS if SHOW_GAME else "unlocked",
            "level": game.level,
        },
    )
    try:
        while True:
            game.play_step()
    finally:
        game.close()


if __name__ == "__main__":
    run_user()

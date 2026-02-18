"""Human-play loop for the arena."""

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arcade

from bang_ai.config import (
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_SHOOT,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_WAIT,
    FPS,
    MAX_LEVEL,
    MIN_LEVEL,
    SHOW_GAME,
    STARTING_LEVEL,
)
from bang_ai.core import BaseGame
from bang_ai.runtime import configure_logging, log_run_context


class HumanGame(BaseGame):
    """Allows a human player to control the player agent."""

    def __init__(self):
        self.level = max(MIN_LEVEL, min(STARTING_LEVEL, MAX_LEVEL))
        super().__init__(level=self.level)

    def play_step(self):
        self.frame_count += 1
        self.poll_events()

        if self.window_controller.is_key_down(arcade.key.W):
            action = ACTION_MOVE_FORWARD
        elif self.window_controller.is_key_down(arcade.key.S):
            action = ACTION_MOVE_BACKWARD
        elif self.window_controller.is_key_down(arcade.key.A):
            action = ACTION_TURN_LEFT
        elif self.window_controller.is_key_down(arcade.key.D):
            action = ACTION_TURN_RIGHT
        elif self.window_controller.is_key_down(arcade.key.SPACE):
            action = ACTION_SHOOT
        else:
            action = ACTION_WAIT

        self.apply_player_action(action)
        self._step_enemy()
        self._step_projectiles()

        self.player.tick()
        self.enemy.tick()

        if not self.enemy.is_alive:
            self.p1_score += 1
            self.reset()
        elif not self.player.is_alive:
            self.p2_score += 1
            self.reset()

        self.draw_frame()
        self.frame_clock.tick(FPS if SHOW_GAME else 0)


def run_human() -> None:
    configure_logging()
    game = HumanGame()
    log_run_context(
        "play-human",
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
    run_human()

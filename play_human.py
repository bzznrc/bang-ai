"""Human-play loop for the arena."""

import pygame

from constants import (
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
from game import BaseGame
from utils import log_run_context


class HumanGame(BaseGame):
    """Allows a human player to control the player agent."""

    def __init__(self):
        self.level = max(MIN_LEVEL, min(STARTING_LEVEL, MAX_LEVEL))
        super().__init__(level=self.level)

    def play_step(self):
        self.frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = ACTION_MOVE_FORWARD
        elif keys[pygame.K_s]:
            action = ACTION_MOVE_BACKWARD
        elif keys[pygame.K_a]:
            action = ACTION_TURN_LEFT
        elif keys[pygame.K_d]:
            action = ACTION_TURN_RIGHT
        elif keys[pygame.K_SPACE]:
            action = ACTION_SHOOT
        else:
            action = ACTION_WAIT

        self.apply_player_action(action)
        self._step_enemy()
        self._step_projectiles()

        self.player.tick()
        self.enemy.tick()
        self.ui.render_frame()
        self.clock.tick(FPS if SHOW_GAME else 0)

        if not self.enemy.is_alive:
            self.p1_score += 1
            self.reset()
        elif not self.player.is_alive:
            self.p2_score += 1
            self.reset()


if __name__ == "__main__":
    game = HumanGame()
    log_run_context(
        "play-human",
        {
            "render": SHOW_GAME,
            "fps": FPS if SHOW_GAME else "unlocked",
            "level": game.level,
        },
    )
    while True:
        game.play_step()

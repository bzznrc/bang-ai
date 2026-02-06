"""Training environment wrapping the arena with RL rewards."""

import math

from constants import (
    ACTION_SHOOT,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    AIM_TOLERANCE_DEGREES,
    FPS,
    MAX_EPISODE_STEPS,
    PENALTY_BAD_SHOT,
    PENALTY_GOT_HIT,
    PENALTY_LOSE_ROUND,
    PENALTY_TIME_STEP,
    REWARD_AIMING,
    REWARD_APPROACH_ENEMY_SCALE,
    REWARD_DODGE_PROJECTILE,
    REWARD_GOOD_SHOT,
    REWARD_HIT_ENEMY,
    REWARD_WIN_ROUND,
    SHOW_GAME,
    TRAINING_FPS,
)
from game import BaseGame
from utils import normalize_angle_degrees


class TrainingGame(BaseGame):
    """Environment used by DQN training."""

    def __init__(self, level=1):
        super().__init__(level=level)
        self.previous_enemy_distance = self.player.position.distance_to(self.enemy.position)

    def reset(self):
        super().reset()
        self.previous_enemy_distance = self.player.position.distance_to(self.enemy.position)

    def play_step(self, action):
        self.frame_count += 1
        action_index = action.index(1) if 1 in action else 0

        self.apply_player_action(action_index)
        self._step_enemy()
        projectile_events = self._step_projectiles()

        self.player.tick()
        self.enemy.tick()
        self.ui.render_frame()
        self.clock.tick(FPS if SHOW_GAME else TRAINING_FPS)

        reward = PENALTY_TIME_STEP

        current_enemy_distance = self.player.position.distance_to(self.enemy.position)
        distance_delta = self.previous_enemy_distance - current_enemy_distance
        reward += REWARD_APPROACH_ENEMY_SCALE * distance_delta
        self.previous_enemy_distance = current_enemy_distance

        to_enemy = self.enemy.position - self.player.position
        enemy_angle = math.degrees(math.atan2(to_enemy.y, to_enemy.x))
        aim_error = abs(normalize_angle_degrees(enemy_angle - self.player.angle))
        if aim_error <= AIM_TOLERANCE_DEGREES:
            reward += REWARD_AIMING

        if action_index == ACTION_SHOOT:
            reward += REWARD_GOOD_SHOT if self.has_line_of_sight() else PENALTY_BAD_SHOT

        if self.is_player_in_projectile_trajectory() and action_index in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
            reward += REWARD_DODGE_PROJECTILE

        if projectile_events["enemy_hit"]:
            reward += REWARD_HIT_ENEMY
        if projectile_events["player_hit"]:
            reward += PENALTY_GOT_HIT

        done = False
        if not self.enemy.is_alive:
            reward += REWARD_WIN_ROUND
            done = True
        elif not self.player.is_alive:
            reward += PENALTY_LOSE_ROUND
            done = True
        elif self.frame_count >= MAX_EPISODE_STEPS:
            done = True

        return reward, done

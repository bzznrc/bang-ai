"""Training environment wrapping the arena with RL rewards."""

from bang_ai.config import (
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_SHOOT,
    FPS,
    MAX_EPISODE_STEPS,
    PENALTY_BAD_SHOT,
    PENALTY_BLOCKED_MOVE,
    PENALTY_LOSE,
    PENALTY_TIME_STEP,
    REWARD_HIT_ENEMY,
    REWARD_WIN,
    SHOW_GAME,
    TRAINING_FPS,
)
from bang_ai.core import BaseGame
from bang_ai.runtime import length_squared


class TrainingGame(BaseGame):
    """Environment used by DQN training."""

    def __init__(self, level: int = 1):
        super().__init__(level=level)

    def play_step(self, action):
        self.frame_count += 1
        action_index = action.index(1) if 1 in action else 0
        self.poll_events()

        previous_position = self.player.position
        self.apply_player_action(action_index)
        blocked_move = length_squared(self.player.position - previous_position) == 0
        self._step_enemy()
        projectile_events = self._step_projectiles()

        self.player.tick()
        self.enemy.tick()

        reward = PENALTY_TIME_STEP
        reward_breakdown = {
            "time_step": PENALTY_TIME_STEP,
            "bad_shot": 0.0,
            "blocked_move": 0.0,
            "hit_enemy": 0.0,
            "win": 0.0,
            "lose": 0.0,
        }

        if action_index == ACTION_SHOOT and not self.has_line_of_sight():
            reward += PENALTY_BAD_SHOT
            reward_breakdown["bad_shot"] = PENALTY_BAD_SHOT
        elif action_index in (ACTION_MOVE_FORWARD, ACTION_MOVE_BACKWARD) and blocked_move:
            reward += PENALTY_BLOCKED_MOVE
            reward_breakdown["blocked_move"] = PENALTY_BLOCKED_MOVE

        if projectile_events["enemy_hit"]:
            reward += REWARD_HIT_ENEMY
            reward_breakdown["hit_enemy"] = REWARD_HIT_ENEMY

        done = False
        if not self.enemy.is_alive:
            reward += REWARD_WIN
            reward_breakdown["win"] = REWARD_WIN
            self.p1_score += 1
            done = True
        elif not self.player.is_alive:
            reward += PENALTY_LOSE
            reward_breakdown["lose"] = PENALTY_LOSE
            self.p2_score += 1
            done = True
        elif self.frame_count >= MAX_EPISODE_STEPS:
            done = True

        self.draw_frame()
        self.frame_clock.tick(FPS if SHOW_GAME else TRAINING_FPS)

        return reward, done, reward_breakdown

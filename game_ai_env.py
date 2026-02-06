"""Training environment wrapping the arena with RL rewards."""

from constants import (
    ACTION_SHOOT,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
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
from game import BaseGame


class TrainingGame(BaseGame):
    """Environment used by DQN training."""

    def __init__(self, level=1):
        super().__init__(level=level)

    def play_step(self, action):
        self.frame_count += 1
        action_index = action.index(1) if 1 in action else 0

        previous_position = self.player.position.copy()
        self.apply_player_action(action_index)
        blocked_move = (self.player.position - previous_position).length_squared() == 0
        self._step_enemy()
        projectile_events = self._step_projectiles()

        self.player.tick()
        self.enemy.tick()
        self.ui.render_frame()
        self.clock.tick(FPS if SHOW_GAME else TRAINING_FPS)

        reward = PENALTY_TIME_STEP

        if action_index == ACTION_SHOOT and not self.has_line_of_sight():
            reward += PENALTY_BAD_SHOT
        elif action_index in (ACTION_MOVE_FORWARD, ACTION_MOVE_BACKWARD) and blocked_move:
            reward += PENALTY_BLOCKED_MOVE

        if projectile_events["enemy_hit"]:
            reward += REWARD_HIT_ENEMY

        done = False
        if not self.enemy.is_alive:
            reward += REWARD_WIN
            done = True
        elif not self.player.is_alive:
            reward += PENALTY_LOSE
            done = True
        elif self.frame_count >= MAX_EPISODE_STEPS:
            done = True

        return reward, done

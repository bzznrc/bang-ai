##################################################
# GAME_AI
##################################################

from game import Game
from constants import *
from utils import *
import pygame

class GameAI(Game):
    """AI-controlled version of the Game."""

    def __init__(self, level=STARTING_LEVEL):
        """Initialize the GameAI."""
        super().__init__(level=level)
        self.prev_enemy_distance = self.player.pos.distance_to(self.enemy.pos)
        self.enemy_hit = False
        self.player_hit = False

    def play_step(self, action):
        """
        Execute one game step based on the action taken.
        """
        self.frame_count += 1

        # Parse action
        action_index = action.index(1) if 1 in action else ACTION_WAIT  # Default action is WAIT

        # Apply the action
        self.apply_action(action_index)

        # Handle enemy actions
        self._enemy_actions()

        # Handle projectiles
        self._handle_projectiles()

        # Update UI
        self.ui.update_ui()

        # Adjust the clock tick
        self.clock.tick(FPS if SHOW_GAME else 0)

        # Initialize reward
        reward = 0

        # Apply time penalty
        reward += TIME_PENALTY

        # Calculate distance to enemy
        current_enemy_distance = self.player.pos.distance_to(self.enemy.pos)

        # Reward for moving towards the enemy
        distance_change = self.prev_enemy_distance - current_enemy_distance
        if distance_change > 0:
            reward += APPROACH_REWARD_FACTOR * distance_change

        self.prev_enemy_distance = current_enemy_distance

        # Reward for aiming at the enemy
        vector_to_enemy = self.enemy.pos - self.player.pos
        enemy_angle = math.degrees(math.atan2(vector_to_enemy.y, vector_to_enemy.x)) % 360
        relative_enemy_angle = ((enemy_angle - self.player.angle + 180) % 360) - 180
        if abs(relative_enemy_angle) <= ALIGNMENT_TOLERANCE:
            reward += AIM_REWARD

        # Reward for shooting when enemy is in line of sight
        los = self._is_line_of_sight()
        if los and action_index == ACTION_SHOOT:
            reward += SHOOT_REWARD

        # Check for hits
        if self.enemy_hit:
            reward += WIN_REWARD  # Enemy is defeated
            self.enemy_hit = False  # Reset flag

        if self.player_hit:
            reward += LOSS_PENALTY
            self.player_hit = False  # Reset flag

        # Check for game over conditions
        game_over = False
        if not self.enemy.alive:
            game_over = True
        elif not self.player.alive:
            game_over = True
        elif self.frame_count >= MAX_MATCH_DURATION:
            game_over = True

        return reward, game_over
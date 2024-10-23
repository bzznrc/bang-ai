# game_ai.py

from game import Game, GameAgent
from constants import *

class GameAI(Game):
    """AI-controlled version of the Game."""

    def reset(self):
        """Reset the game state and action reward trackers."""
        super().reset()
        # Initialize incremental action bonuses
        self.inc_closer_bonus = 0.0
        self.inc_away_bonus = 0.0
        self.inc_frame_bonus = 0.0
        self.inc_miss_bonus = 0.0
        # Outcome bonus
        self.outcome_bonus = 0.0
        # Track distance to enemy
        self.prev_distance_to_enemy = self.player.pos.distance_to(self.enemy.pos)
        # Flag for shooting
        self.player_shot = False

    def play_step(self, action):
        """Execute one game step based on the action taken."""
        self.frame_count += 1

        # Compute distance to enemy before action
        distance_before = self.player.pos.distance_to(self.enemy.pos)

        # Parse action
        move_up = move_down = move_left = move_right = rotate_left = rotate_right = shoot = False

        try:
            action_index = action.index(1)
        except ValueError:
            # If no action is selected, treat as ACTION_WAIT
            action_index = ACTION_WAIT

        if action_index == ACTION_MOVE_UP:
            move_up = True
        elif action_index == ACTION_MOVE_DOWN:
            move_down = True
        elif action_index == ACTION_MOVE_LEFT:
            move_left = True
        elif action_index == ACTION_MOVE_RIGHT:
            move_right = True
        elif action_index == ACTION_ROTATE_LEFT:
            rotate_left = True
        elif action_index == ACTION_ROTATE_RIGHT:
            rotate_right = True
        elif action_index == ACTION_SHOOT:
            shoot = True
        elif action_index == ACTION_WAIT:
            pass  # Do nothing

        # Move player
        self._move_agent(self.player, move_up, move_down, move_left, move_right, rotate_left, rotate_right)

        # Compute distance to enemy after action
        distance_after = self.player.pos.distance_to(self.enemy.pos)
        delta_distance = distance_before - distance_after

        # Initialize reward for this step
        reward = 0.0

        # Moving closer to enemy
        if delta_distance > 0:
            if self.inc_closer_bonus < 5:
                incremental_bonus = 0.01 #min(0.01, 5 - self.inc_closer_bonus)
                reward += incremental_bonus
                self.inc_closer_bonus += incremental_bonus
        # Moving away from enemy
        elif delta_distance < 0:
            if self.inc_away_bonus > -5:
                incremental_bonus = -0.01 #max(-0.01, -5 - self.inc_away_bonus)
                reward += incremental_bonus  # Negative bonus
                self.inc_away_bonus += incremental_bonus

        # Update previous distance
        self.prev_distance_to_enemy = distance_after

        # Default frame penalty (negative bonus)
        #if self.inc_frame_bonus > -5:
        #    incremental_bonus = max(-0.005, -5 - self.inc_frame_bonus)
        #    reward += incremental_bonus
        #    self.inc_frame_bonus += incremental_bonus

        # Handle shooting
        if shoot:
            self.player_shot = True
            self._agent_shoot(self.player)

        # Update projectiles
        self._handle_projectiles()

        # Decrement cooldowns
        self._decrement_cooldowns()

        # Handle enemy actions
        self._enemy_actions()

        # Update UI
        self.ui.update_ui()

        # Adjust the clock tick
        if SHOW_GAME:
            self.clock.tick(FPS)
        else:
            # Run as fast as possible when not showing the game
            self.clock.tick(0)

        # Check for missed shots
        if self.player_shot:
            # For now, set the shoot bonus to 0
            # Keep the code for future use
            # Example of applying a penalty for missed shots:
            # if self.inc_miss_bonus > -5:
            #     incremental_bonus = max(-0.01, -5 - self.inc_miss_bonus)
            #     reward += incremental_bonus
            #     self.inc_miss_bonus += incremental_bonus
            pass
        # Reset shot flag
        self.player_shot = False

        # Check for game over conditions
        game_over = False

        # If the enemy is eliminated, give a positive reward
        if not self.enemy.alive:
            reward += 10  # Win bonus
            self.outcome_bonus += 10
            game_over = True
            return reward, game_over

        # If the player is eliminated, give a negative reward
        elif not self.player.alive:
            reward += -5  # Elimination penalty (negative bonus)
            self.outcome_bonus += -5
            game_over = True
            return reward, game_over

        # Penalty for exceeding max match duration
        elif self.frame_count >= MAX_MATCH_DURATION:
            reward += -5  # Timeout penalty (negative bonus)
            self.outcome_bonus += -5
            game_over = True
            return reward, game_over

        return reward, game_over

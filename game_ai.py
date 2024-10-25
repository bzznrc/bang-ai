##################################################
# GAME_AI
##################################################

from game import Game, GameAgent
from constants import *
from utils import *

class GameAI(Game):
    """AI-controlled version of the Game."""

    def __init__(self):
        """Initialize the GameAI."""
        super().__init__()
        self.prev_action = None # To store the previous action

    def reset(self):
        """Reset the game state and action reward trackers."""
        super().reset()
        # Flag for shooting
        self.player_shot = False

    def play_step(self, action):
        """Execute one game step based on the action taken."""
        self.frame_count += 1

        # Initialize bonuses list
        bonuses = []

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

        # Get player's movement vector
        player_movement = get_player_movement_vector(action_index)

        # Move player
        self._move_agent(self.player, move_up, move_down, move_left, move_right, rotate_left, rotate_right)

        # Initialize reward for this step
        reward = 0.0

        # Same Action Bonus (excluding shooting)
        if self.prev_action is not None and action_index != ACTION_SHOOT and self.prev_action != ACTION_SHOOT:
            same_action_bonus = 0.001 if action_index == self.prev_action else 0.000
        else:
            same_action_bonus = 0.000
        reward += same_action_bonus
        bonuses.append(same_action_bonus)
        self.prev_action = action_index

        # Proximity Bonus
        proximity_bonus = self.get_proximity_bonus(bonus=0.010)
        reward += proximity_bonus
        bonuses.append(proximity_bonus)

        # Dodge and Cover Bonus
        dodge_bonus, cover_bonus = self.get_dodge_and_cover_bonus(player_movement, bonus=0.010)
        reward += dodge_bonus
        bonuses.append(dodge_bonus)
        reward += cover_bonus
        bonuses.append(cover_bonus)

        # Handle shooting
        if shoot:
            self.player_shot = True
            self._agent_shoot(self.player)

            shoot_alignment_bonus = self.get_shooting_alignment_bonus(bonus=0.010)
            reward += shoot_alignment_bonus
            bonuses.append(shoot_alignment_bonus)
        else:
            bonuses.append(0) # No shoot alignment bonus

        # Update projectiles and game state
        self._handle_projectiles()
        self._decrement_cooldowns()
        self._enemy_actions()
        self.ui.update_ui()
        self.clock.tick(FPS if SHOW_GAME else 0)
        self.player_shot = False

        # Check for game over conditions
        game_over = False
        outcome_bonus = 0.0
        if not self.enemy.alive:
            outcome_bonus = 10.0
            reward += outcome_bonus
            if self.frame_count <= QUICK_WIN_THRESHOLD:
                reward += 5
            game_over = True
        elif not self.player.alive:
            outcome_bonus = -5.0
            reward += outcome_bonus
            game_over = True
        elif self.frame_count >= MAX_MATCH_DURATION:
            outcome_bonus = -5.0
            reward += outcome_bonus
            game_over = True
        bonuses.append(outcome_bonus)

        return reward, game_over, bonuses
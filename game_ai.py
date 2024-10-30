##################################################
# GAME_AI
##################################################

from game import Game
from constants import *
from utils import *

class GameAI(Game):
    """AI-controlled version of the Game."""

    def __init__(self):
        """Initialize the GameAI."""
        super().__init__()
        self.previous_action = None  # To store the previous movement action

    def reset(self):
        """Reset the game state."""
        super().reset()

    def play_step(self, action):
        """
        Execute one game step based on the action taken.
        :param action: List indicating the action (one-hot encoded)
        :return: Tuple containing the reward and game_over flag
        """
        self.frame_count += 1

        # Parse action
        action_index = action.index(1) if 1 in action else ACTION_WAIT  # Default to waiting if no action

        # Decrement cooldowns
        self._decrement_cooldowns()

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
        reward = 0 #STEP_PENALTY  # Small penalty to encourage faster victories

        # Apply proximity reward if within the set range
        if self.is_player_within_proximity():
            reward += PROXIMITY_REWARD

        # Check for game over conditions
        game_over = False

        if not self.enemy.alive:
            reward += WIN_REWARD
            game_over = True
        elif not self.player.alive or self.frame_count >= MAX_MATCH_DURATION:
            reward += LOSS_PENALTY
            game_over = True

        return reward, game_over

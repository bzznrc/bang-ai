##################################################
# GAME_USER
##################################################

from game import Game
import pygame
from constants import *
from utils import *

class GameUser(Game):
    """User-controlled version of the Game."""

    def __init__(self):
        """Initialize the GameUser."""
        super().__init__()

    def play_step(self):
        """Execute one game step."""
        self.frame_count += 1
        move_forward = False
        move_backward = False
        rotate_left = False
        rotate_right = False
        shoot = False

        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            move_forward = True
        if keys[pygame.K_s]:
            move_backward = True
        if keys[pygame.K_a]:
            rotate_left = True
        if keys[pygame.K_d]:
            rotate_right = True
        if keys[pygame.K_SPACE]:
            shoot = True
        if keys[pygame.K_LCTRL]:
            self.print_state()  # DEBUG

        # Determine current action using the utility function
        current_action = determine_current_action(move_forward, move_backward, rotate_left, rotate_right, shoot)

        # Decrement cooldowns
        self._decrement_cooldowns()

        # Apply the action using the centralized method
        self.apply_action(current_action)

        # Handle enemy actions
        self._enemy_actions()

        # Handle projectiles
        self._handle_projectiles()

        # Update UI
        self.ui.update_ui()

        # Adjust the clock tick
        self.clock.tick(FPS if SHOW_GAME else 0)

        # Check for game over
        if not self.enemy.alive:
            # P1 wins this round
            pygame.time.delay(3000)  # 3-second delay
            self.p1_score += 1  # Increment P1 score
            self.reset()  # Reset the game but keep scores
            return False, (self.p1_score, self.p2_score)
        elif not self.player.alive:
            # P2 wins this round
            pygame.time.delay(3000)  # 3-second delay
            self.p2_score += 1  # Increment P2 score
            self.reset()  # Reset the game but keep scores
            return False, (self.p1_score, self.p2_score)

        # Continue the game
        return False, (self.p1_score, self.p2_score)

if __name__ == '__main__':
    game = GameUser()

    # Game loop
    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    print('Final Score:', score)
    pygame.quit()
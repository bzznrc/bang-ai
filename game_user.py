# human_game.py

from game import Game
import pygame
from constants import *

class GameUser(Game):
    """User-controlled version of the Game."""

    def play_step(self):
        """Execute one game step."""
        self.frame_count += 1
        move_up = False
        move_down = False
        move_left = False
        move_right = False
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
            move_up = True
        if keys[pygame.K_s]:
            move_down = True
        if keys[pygame.K_a]:
            move_left = True
        if keys[pygame.K_d]:
            move_right = True
        if keys[pygame.K_LEFT]:
            rotate_left = True
        if keys[pygame.K_RIGHT]:
            rotate_right = True
        if keys[pygame.K_SPACE]:
            shoot = True
        if keys[pygame.K_LCTRL]:
            self.print_state()  # DEBUG

        # Move player
        self._move_agent(self.player, move_up, move_down, move_left, move_right, rotate_left, rotate_right)

        # Handle shooting
        if shoot:
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
##################################################
# UI
##################################################

import pygame
import math
from constants import *

class GameUI:
    """Class responsible for handling game rendering."""
    def __init__(self, display, game):
        self.display = display
        self.game = game
        self.font = pygame.font.SysFont(None, FONT_SIZE)

    def update_ui(self):
        """Update the game's UI."""
        if SHOW_GAME:
            self.display.fill(COLOR_BACKGROUND)

            # Draw obstacles
            for pt in self.game.obstacles:
                pygame.draw.rect(self.display, COLOR_OBSTACLE_OUTLINE,
                                 pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, COLOR_OBSTACLE_PRIMARY,
                                 pygame.Rect(pt.x + 4, pt.y + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8))

            # Draw enemy
            if self.game.enemy.alive:
                self._draw_agent(self.game.enemy, COLOR_RIGHT_TEAM, COLOR_RIGHT_TEAM_OUTLINE)

            # Draw player
            if self.game.player.alive:
                self._draw_agent(self.game.player, COLOR_LEFT_TEAM, COLOR_LEFT_TEAM_OUTLINE)

            # Draw projectiles
            for proj in self.game.projectiles:
                pygame.draw.circle(self.display, COLOR_PROJECTILE, (int(proj['pos'].x), int(proj['pos'].y)), 5)

            # Draw bottom bar
            pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(0, self.game.h - BB_HEIGHT, self.game.w, BB_HEIGHT))

            # Display P1 and P2 scores on the bottom bar
            score_text = self.font.render(f"P1 Score: {self.game.p1_score}    P2 Score: {self.game.p2_score}", True, COLOR_SCORE)
            self.display.blit(score_text, (BB_MARGIN, self.game.h - BB_HEIGHT + (BB_HEIGHT - FONT_SIZE) // 2))

            pygame.display.flip()
        else:
            # If not showing the game, you might still need to handle events
            # to prevent the program from becoming unresponsive.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

    def _draw_agent(self, agent, color, outline_color):
        """Draw an agent (player or enemy) as a square with an outline and a direction tick."""
        position = agent.pos
        angle = agent.angle

        # Draw the square
        rect = pygame.Rect(position.x - BLOCK_SIZE // 2, position.y - BLOCK_SIZE // 2, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, outline_color, rect)
        inner_rect = pygame.Rect(position.x - BLOCK_SIZE // 2 + 4, position.y - BLOCK_SIZE // 2 + 4, BLOCK_SIZE - 8, BLOCK_SIZE - 8)
        pygame.draw.rect(self.display, color, inner_rect)

        # Draw the direction tick
        tick_length = BLOCK_SIZE // 2
        tick_start = position
        tick_end = position + pygame.Vector2(
            math.cos(math.radians(angle)),
            math.sin(math.radians(angle))
        ) * tick_length
        pygame.draw.line(self.display, (255, 255, 255), tick_start, tick_end, 2)  # White tick

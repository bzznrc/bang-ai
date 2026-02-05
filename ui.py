"""UI rendering for the Bang arena."""

import math
import pygame

from constants import (
    BOTTOM_BAR_HEIGHT,
    BOTTOM_BAR_MARGIN,
    COLOR_BACKGROUND,
    COLOR_ENEMY,
    COLOR_ENEMY_OUTLINE,
    COLOR_OBSTACLE_FILL,
    COLOR_OBSTACLE_OUTLINE,
    COLOR_PLAYER,
    COLOR_PLAYER_OUTLINE,
    COLOR_PROJECTILE,
    COLOR_SCORE,
    FONT_SIZE,
    SHOW_GAME,
    TILE_SIZE,
)


class Renderer:
    """Draw the arena entities and score overlay."""

    def __init__(self, display, game):
        self.display = display
        self.game = game
        self.font = pygame.font.SysFont(None, FONT_SIZE)

    def render_frame(self):
        if not SHOW_GAME:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
            return

        self.display.fill(COLOR_BACKGROUND)
        for obstacle in self.game.obstacles:
            pygame.draw.rect(self.display, COLOR_OBSTACLE_OUTLINE, pygame.Rect(obstacle.x, obstacle.y, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(self.display, COLOR_OBSTACLE_FILL, pygame.Rect(obstacle.x + 4, obstacle.y + 4, TILE_SIZE - 8, TILE_SIZE - 8))

        if self.game.enemy.is_alive:
            self._draw_actor(self.game.enemy, COLOR_ENEMY, COLOR_ENEMY_OUTLINE)
        if self.game.player.is_alive:
            self._draw_actor(self.game.player, COLOR_PLAYER, COLOR_PLAYER_OUTLINE)

        for projectile in self.game.projectiles:
            pygame.draw.circle(self.display, COLOR_PROJECTILE, (int(projectile["pos"].x), int(projectile["pos"].y)), 5)

        pygame.draw.rect(self.display, (0, 0, 0), pygame.Rect(0, self.game.height - BOTTOM_BAR_HEIGHT, self.game.width, BOTTOM_BAR_HEIGHT))
        score_text = self.font.render(f"P1 Score: {self.game.p1_score}    P2 Score: {self.game.p2_score}", True, COLOR_SCORE)
        self.display.blit(score_text, (BOTTOM_BAR_MARGIN, self.game.height - BOTTOM_BAR_HEIGHT + (BOTTOM_BAR_HEIGHT - FONT_SIZE) // 2))
        pygame.display.flip()

    def _draw_actor(self, actor, fill_color, outline_color):
        rect = pygame.Rect(actor.position.x - TILE_SIZE // 2, actor.position.y - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(self.display, outline_color, rect)
        inner_rect = pygame.Rect(actor.position.x - TILE_SIZE // 2 + 4, actor.position.y - TILE_SIZE // 2 + 4, TILE_SIZE - 8, TILE_SIZE - 8)
        pygame.draw.rect(self.display, fill_color, inner_rect)

        tick_end = actor.position + pygame.Vector2(math.cos(math.radians(actor.angle)), math.sin(math.radians(actor.angle))) * (TILE_SIZE // 2)
        pygame.draw.line(self.display, (255, 255, 255), actor.position, tick_end, 2)

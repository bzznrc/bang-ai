"""UI rendering for the Bang arena."""

import math
import pygame

from constants import (
    BB_HEIGHT,
    BOTTOM_BAR_MARGIN,
    COLOR_BACKGROUND,
    COLOR_BOTTOM_BAR,
    COLOR_NEUTRAL_DARK,
    COLOR_NEUTRAL_LIGHT,
    COLOR_P1_DARK,
    COLOR_P1_LIGHT,
    COLOR_P2_DARK,
    COLOR_P2_LIGHT,
    COLOR_PROJECTILE,
    COLOR_SCORE,
    FONT_NAME_BAR,
    FONT_PATH_REGULAR,
    FONT_SIZE_BAR,
    SHOW_GAME,
    TILE_SIZE,
    UI_STATUS_SEPARATOR,
)


class Renderer:
    """Draw the arena entities and score overlay."""

    def __init__(self, display, game):
        self.display = display
        self.game = game
        try:
            self.font = pygame.font.Font(FONT_PATH_REGULAR, FONT_SIZE_BAR)
        except OSError:
            self.font = pygame.font.SysFont(FONT_NAME_BAR or "Roboto", FONT_SIZE_BAR, bold=False, italic=False)
        self.bottom_bar_top = self.game.height - BB_HEIGHT
        self.bottom_bar_rect = pygame.Rect(0, self.bottom_bar_top, self.game.width, BB_HEIGHT)
        self.score_y = self.bottom_bar_top + (BB_HEIGHT - self.font.get_height()) // 2

    def render_frame(self):
        if not SHOW_GAME:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
            return

        self.display.fill(COLOR_BACKGROUND)
        for obstacle in self.game.obstacles:
            pygame.draw.rect(self.display, COLOR_NEUTRAL_LIGHT, pygame.Rect(obstacle.x, obstacle.y, TILE_SIZE, TILE_SIZE))
            pygame.draw.rect(self.display, COLOR_NEUTRAL_DARK, pygame.Rect(obstacle.x + 4, obstacle.y + 4, TILE_SIZE - 8, TILE_SIZE - 8))

        if self.game.enemy.is_alive:
            self._draw_actor(self.game.enemy, COLOR_P2_DARK, COLOR_P2_LIGHT)
        if self.game.player.is_alive:
            self._draw_actor(self.game.player, COLOR_P1_DARK, COLOR_P1_LIGHT)

        for projectile in self.game.projectiles:
            pygame.draw.circle(self.display, COLOR_PROJECTILE, (int(projectile["pos"].x), int(projectile["pos"].y)), 5)

        pygame.draw.rect(self.display, COLOR_BOTTOM_BAR, self.bottom_bar_rect)
        score_text = self.font.render(
            f"P1 Score: {self.game.p1_score}{UI_STATUS_SEPARATOR}P2 Score: {self.game.p2_score}",
            True,
            COLOR_SCORE,
        )
        self.display.blit(score_text, (BOTTOM_BAR_MARGIN, self.score_y))
        pygame.display.flip()

    def _draw_actor(self, actor, fill_color, outline_color):
        rect = pygame.Rect(actor.position.x - TILE_SIZE // 2, actor.position.y - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
        pygame.draw.rect(self.display, outline_color, rect)
        inner_rect = pygame.Rect(actor.position.x - TILE_SIZE // 2 + 4, actor.position.y - TILE_SIZE // 2 + 4, TILE_SIZE - 8, TILE_SIZE - 8)
        pygame.draw.rect(self.display, fill_color, inner_rect)

        tick_end = actor.position + pygame.Vector2(math.cos(math.radians(actor.angle)), math.sin(math.radians(actor.angle))) * (TILE_SIZE // 2)
        pygame.draw.line(self.display, (255, 255, 255), actor.position, tick_end, 2)

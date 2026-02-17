"""UI rendering for the Bang arena."""

import math

import pygame

from config import (
    BB_HEIGHT,
    CELL_INSET,
    FONT_NAME_BAR,
    FONT_PATH_REGULAR,
    FONT_SIZE_BAR,
    SHOW_GAME,
    TILE_SIZE,
    UI_STATUS_SEPARATOR,
)
from bgds.visual.assets import load_font
from bgds.visual.colors import (
    COLOR_AMBER,
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_NEAR_BLACK,
    COLOR_SLATE_GRAY,
    COLOR_SOFT_WHITE,
)
from bgds.visual.statusbar import draw_centered_status_bar
from bgds.visual.square_render import draw_two_tone_cell, draw_two_tone_grid_block

class Renderer:
    """Draw the arena entities and score overlay."""

    def __init__(self, display, game):
        self.display = display
        self.game = game
        self.font = load_font(
            FONT_PATH_REGULAR,
            FONT_SIZE_BAR,
            fallback_family=FONT_NAME_BAR,
        )

    def render_frame(self):
        if not SHOW_GAME:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
            return

        self.display.fill(COLOR_CHARCOAL)
        draw_two_tone_grid_block(
            surface=self.display,
            top_left_points=self.game.obstacles,
            size_px=TILE_SIZE,
            inset_px=CELL_INSET,
            outer_color=COLOR_FOG_GRAY,
            inner_color=COLOR_SLATE_GRAY,
        )

        if self.game.enemy.is_alive:
            self._draw_actor(self.game.enemy, COLOR_BRICK_RED, COLOR_CORAL)
        if self.game.player.is_alive:
            self._draw_actor(self.game.player, COLOR_DEEP_TEAL, COLOR_AQUA)

        for projectile in self.game.projectiles:
            pygame.draw.circle(self.display, COLOR_AMBER, (int(projectile["pos"].x), int(projectile["pos"].y)), 5)

        draw_centered_status_bar(
            surface=self.display,
            font=self.font,
            screen_width_px=self.game.width,
            screen_height_px=self.game.height,
            bar_height_px=BB_HEIGHT,
            items=[
                (f"P1 Score: {self.game.p1_score}", COLOR_DEEP_TEAL),
                (f"P2 Score: {self.game.p2_score}", COLOR_BRICK_RED),
            ],
            background_color=COLOR_NEAR_BLACK,
            default_text_color=COLOR_SOFT_WHITE,
            separator=UI_STATUS_SEPARATOR,
            separator_color=COLOR_SOFT_WHITE,
        )
        pygame.display.flip()

    def _draw_actor(self, actor, fill_color, outline_color):
        top_left = (
            actor.position.x - TILE_SIZE // 2,
            actor.position.y - TILE_SIZE // 2,
        )
        draw_two_tone_cell(
            surface=self.display,
            top_left=top_left,
            size_px=TILE_SIZE,
            inset_px=CELL_INSET,
            outer_color=outline_color,
            inner_color=fill_color,
        )

        tick_end = actor.position + pygame.Vector2(math.cos(math.radians(actor.angle)), math.sin(math.radians(actor.angle))) * (TILE_SIZE // 2)
        pygame.draw.line(self.display, (255, 255, 255), actor.position, tick_end, 2)


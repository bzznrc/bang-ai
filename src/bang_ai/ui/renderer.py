"""Arcade renderer for the Bang arena."""

from __future__ import annotations

import arcade

from bang_ai.config import (
    BB_HEIGHT,
    CELL_INSET,
    FONT_NAME_BAR,
    FONT_PATH_REGULAR,
    FONT_SIZE_BAR,
    SHOW_GAME,
    TILE_SIZE,
    UI_STATUS_SEPARATOR,
)
from bang_ai.runtime import ArcadeWindowController, TextCache, heading_to_vector, load_font_once
from bang_ai.visual import (
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
    resolve_font_path,
)


def _font_name() -> str:
    font_path = resolve_font_path(FONT_PATH_REGULAR)
    load_font_once(font_path)
    return FONT_NAME_BAR or "Roboto"


class Renderer:
    """Draw arena entities and score overlay with Arcade primitives."""

    def __init__(self, game, width: int, height: int, title: str, enabled: bool):
        self.game = game
        self.enabled = bool(enabled and SHOW_GAME)
        self.width = int(width)
        self.height = int(height)
        self.font_name = _font_name()

        self.window_controller = ArcadeWindowController(
            self.width,
            self.height,
            title,
            enabled=self.enabled,
            queue_input_events=False,
            vsync=False,
        )
        self.window = self.window_controller.window
        self.text_cache = TextCache(max_entries=256)

    def close(self):
        self.window_controller.close()
        self.window = None

    def poll_events(self):
        self.window_controller.poll_events_or_raise()

    def is_key_down(self, symbol: int) -> bool:
        return self.window_controller.is_key_down(symbol)

    def draw_frame(self):
        if self.window is None:
            return

        self.window_controller.clear(COLOR_CHARCOAL)

        for obstacle in self.game.obstacles:
            self._draw_two_tone_tile(
                top_left_x=float(obstacle.x),
                top_left_y=float(obstacle.y),
                outer_color=COLOR_FOG_GRAY,
                inner_color=COLOR_SLATE_GRAY,
            )

        if self.game.enemy.is_alive:
            self._draw_actor(self.game.enemy, COLOR_BRICK_RED, COLOR_CORAL)
        if self.game.player.is_alive:
            self._draw_actor(self.game.player, COLOR_DEEP_TEAL, COLOR_AQUA)

        for projectile in self.game.projectiles:
            arcade.draw_circle_filled(
                projectile["pos"].x,
                self.window_controller.to_arcade_y(projectile["pos"].y),
                5,
                COLOR_AMBER,
            )

        self._draw_status_bar()
        self.window_controller.flip()

    def _draw_status_bar(self):
        arcade.draw_lbwh_rectangle_filled(0, 0, self.width, BB_HEIGHT, COLOR_NEAR_BLACK)

        separator = UI_STATUS_SEPARATOR if UI_STATUS_SEPARATOR else " / "
        segments = [
            (f"P1 Score: {self.game.p1_score}", COLOR_DEEP_TEAL),
            (separator, COLOR_SOFT_WHITE),
            (f"P2 Score: {self.game.p2_score}", COLOR_BRICK_RED),
        ]
        text_objects = []
        total_width = 0.0
        for text, color in segments:
            text_obj = self.text_cache.get_text(
                text=text,
                color=color,
                font_size=FONT_SIZE_BAR,
                font_name=self.font_name,
                anchor_x="left",
                anchor_y="center",
            )
            text_objects.append(text_obj)
            total_width += float(text_obj.content_width)

        cursor_x = (self.width - total_width) / 2.0
        center_y = BB_HEIGHT / 2.0
        for text_obj in text_objects:
            text_obj.x = cursor_x
            text_obj.y = center_y
            text_obj.draw()
            cursor_x += float(text_obj.content_width)

    def _draw_two_tone_tile(self, top_left_x: float, top_left_y: float, outer_color, inner_color):
        bottom = self.window_controller.top_left_to_bottom(top_left_y, TILE_SIZE)
        arcade.draw_lbwh_rectangle_filled(top_left_x, bottom, TILE_SIZE, TILE_SIZE, outer_color)
        inner_size = TILE_SIZE - 2 * CELL_INSET
        if inner_size > 0:
            arcade.draw_lbwh_rectangle_filled(
                top_left_x + CELL_INSET,
                bottom + CELL_INSET,
                inner_size,
                inner_size,
                inner_color,
            )

    def _draw_actor(self, actor, fill_color, outline_color):
        self._draw_two_tone_tile(
            top_left_x=actor.position.x - TILE_SIZE / 2,
            top_left_y=actor.position.y - TILE_SIZE / 2,
            outer_color=outline_color,
            inner_color=fill_color,
        )

        facing = heading_to_vector(actor.angle)
        tick_end = actor.position + facing * (TILE_SIZE // 2)
        arcade.draw_line(
            actor.position.x,
            self.window_controller.to_arcade_y(actor.position.y),
            tick_end.x,
            self.window_controller.to_arcade_y(tick_end.y),
            COLOR_SOFT_WHITE,
            2,
        )

"""Bang core gameplay, rendering, and game modes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, TypeVar

import arcade

from bang_ai.assets import resolve_font_path
from bang_ai.config import (
    ACTION_NAMES,
    ACTION_AIM_LEFT,
    ACTION_AIM_RIGHT,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP,
    ACTION_SHOOT,
    ACTION_STOP_MOVE,
    AIM_RATE_PER_STEP,
    BB_HEIGHT,
    CELL_INSET,
    CLEARANCE_HORIZON_STEPS,
    COLOR_AMBER,
    COLOR_AQUA,
    COLOR_BRICK_RED,
    COLOR_CHARCOAL,
    COLOR_CORAL,
    COLOR_DEEP_TEAL,
    COLOR_FOG_GRAY,
    COLOR_NEAR_BLACK,
    COLOR_P3_BLUE,
    COLOR_P3_NAVY,
    COLOR_P4_DEEP_PURPLE,
    COLOR_P4_PURPLE,
    COLOR_SLATE_GRAY,
    COLOR_SOFT_WHITE,
    ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES,
    ENEMY_ESCAPE_FOLLOW_FRAMES,
    ENEMY_SPAWN_X_RATIO,
    ENEMY_STUCK_MOVE_ATTEMPTS,
    EVENT_TIMER_NORMALIZATION_FRAMES,
    FONT_FAMILY_DEFAULT,
    FONT_PATH_ROBOTO_REGULAR,
    FONT_SIZE_BAR,
    FPS,
    INPUT_FEATURE_NAMES,
    LEVEL_SETTINGS,
    MAX_EPISODE_STEPS,
    MAX_LEVEL,
    MAX_OBSTACLE_SECTIONS,
    MIN_LEVEL,
    MIN_OBSTACLE_SECTIONS,
    NUM_ALLIES_OVERRIDE,
    OBSTACLE_START_ATTEMPTS,
    PENALTY_BAD_SHOT,
    PENALTY_BLOCKED_MOVE,
    PENALTY_FRIENDLY_FIRE,
    PENALTY_TIME_STEP,
    PLAYER_MOVE_SPEED,
    PLAYER_SPAWN_X_RATIO,
    PROJECTILE_HITBOX_SIZE,
    PROJECTILE_SPEED,
    PROJECTILE_TRAJECTORY_DOT_THRESHOLD,
    RESULT_REWARD_BY_OUTCOME,
    REWARD_HIT_ENEMY,
    SAFE_RADIUS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SHOOT_COOLDOWN_FRAMES,
    SPAWN_Y_OFFSET,
    STARTING_LEVEL,
    TILE_SIZE,
    TRAIN_NN_CONTROLS_ALLIES,
    TRAINING_FPS,
    UI_STATUS_SEPARATOR,
    WINDOW_TITLE,
)
from bang_ai.runtime import (
    ArcadeFrameClock,
    ArcadeWindowController,
    TextCache,
    Vec2,
    collides_with_square_arena,
    heading_to_vector,
    length_squared,
    load_font_once,
    normalize_angle_degrees,
    rect_from_center,
    rotate_degrees,
    square_obstacle_between_points,
)
from bang_ai.utils import validate_level_settings

T = TypeVar("T")


ALL_PLAYER_ORDER = ("P1", "P2", "P3", "P4")
SUPPORTED_PLAYER_COUNTS = (2, 3, 4)
validate_level_settings(
    min_level=MIN_LEVEL,
    max_level=MAX_LEVEL,
    level_settings=LEVEL_SETTINGS,
    valid_player_counts=SUPPORTED_PLAYER_COUNTS,
)


def _resolve_player_order(num_players: int) -> tuple[str, ...]:
    count = int(num_players)
    if count not in SUPPORTED_PLAYER_COUNTS:
        raise ValueError(f"num_players must be one of {SUPPORTED_PLAYER_COUNTS}, got {count}")
    return ALL_PLAYER_ORDER[:count]


def _num_players_for_level(level: int) -> int:
    return int(LEVEL_SETTINGS[int(level)]["num_players"])


def _num_allies_for_level(level: int) -> int:
    return int(LEVEL_SETTINGS[int(level)]["num_allies"])


PLAYER_STYLES = {
    "P1": {
        "render_fill": COLOR_DEEP_TEAL,
        "render_outline": COLOR_AQUA,
        "status_color": COLOR_DEEP_TEAL,
        "scripted": False,
    },
    "P2": {
        "render_fill": COLOR_BRICK_RED,
        "render_outline": COLOR_CORAL,
        "status_color": COLOR_BRICK_RED,
        "scripted": True,
    },
    "P3": {
        "render_fill": COLOR_P3_NAVY,
        "render_outline": COLOR_P3_BLUE,
        "status_color": COLOR_P3_BLUE,
        "scripted": True,
    },
    "P4": {
        "render_fill": COLOR_P4_DEEP_PURPLE,
        "render_outline": COLOR_P4_PURPLE,
        "status_color": COLOR_P4_PURPLE,
        "scripted": True,
    },
}
SPAWN_AREA_LEFT = "left_column"
SPAWN_AREA_RIGHT = "right_column"
SPAWN_AREA_BOTTOM = "bottom_strip"
SPAWN_AREA_TOP = "top_strip"
SPAWN_AREA_ORDER = (
    SPAWN_AREA_LEFT,
    SPAWN_AREA_RIGHT,
    SPAWN_AREA_BOTTOM,
    SPAWN_AREA_TOP,
)

PLAYER_TARGET_POLICY = {
    "max_lost_frames": 40,
    "switch_distance_ratio": 0.8,
    "random_switch_prob": 0.005,
    "hold_min_frames": 30,
    "hold_max_frames": 75,
}
SCRIPTED_TARGET_POLICY = {
    "max_lost_frames": 45,
    "switch_distance_ratio": 0.85,
    "random_switch_prob": 0.02,
    "hold_min_frames": 18,
    "hold_max_frames": 60,
}
SCRIPTED_MOVE_FALLBACK_OFFSETS = (0.0, 90.0, -90.0, 180.0)
TARGET_LOCK_FRAMES = 10
TARGET_SWITCH_CLOSER_RATIO = 0.8
REWARD_COMPONENT_KEY_ORDER = ("result", "hit", "time", "accuracy", "safety", "obstacle")
REWARD_COMPONENT_KEY_SET = set(REWARD_COMPONENT_KEY_ORDER)


def _grow_connected_random_walk_shape(
    start: T,
    min_sections: int,
    max_sections: int,
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
) -> list[T]:
    target_sections = random.randint(int(min_sections), int(max_sections))
    shape = [start]
    current = start

    for _ in range(target_sections - 1):
        candidates = list(neighbor_candidates_fn(current))
        random.shuffle(candidates)
        for candidate in candidates:
            if is_candidate_valid_fn(candidate, shape):
                shape.append(candidate)
                current = candidate
                break
        else:
            break
    return shape


def spawn_connected_random_walk_shapes(
    shape_count: int,
    min_sections: int,
    max_sections: int,
    sample_start_fn: Callable[[], T | None],
    neighbor_candidates_fn: Callable[[T], list[T]],
    is_candidate_valid_fn: Callable[[T, list[T]], bool],
) -> list[list[T]]:
    shapes: list[list[T]] = []
    for _ in range(int(shape_count)):
        start = sample_start_fn()
        if start is None:
            continue
        shape = _grow_connected_random_walk_shape(
            start=start,
            min_sections=min_sections,
            max_sections=max_sections,
            neighbor_candidates_fn=neighbor_candidates_fn,
            is_candidate_valid_fn=is_candidate_valid_fn,
        )
        if shape:
            shapes.append(shape)
    return shapes


def _font_name() -> str:
    font_path = resolve_font_path(FONT_PATH_ROBOTO_REGULAR)
    load_font_once(font_path)
    return FONT_FAMILY_DEFAULT or "Roboto"


@dataclass
class TargetState:
    target_id: str | None = None
    target_lost_frames: int = 0
    target_switch_cooldown: int = 0
    target_lock_frames_remaining: int = 0
    last_update_frame: int = -1


@dataclass
class ScriptedMovementState:
    blocked_moves: int = 0
    escape_frames_remaining: int = 0
    escape_angle: float = 0.0


@dataclass
class ActorObservationState:
    previous_enemy_distance: float | None = None
    previous_enemy_relative_angle: float | None = None
    previous_projectile_distance: float | None = None
    previous_projectile_relative_angle: float | None = None
    last_action_index: int = 0
    last_blocked_move_action_idx: int | None = None
    last_seen_enemy_frame: int = -EVENT_TIMER_NORMALIZATION_FRAMES


class Actor:
    """A movable actor that can rotate and shoot projectiles."""

    def __init__(self, position: Vec2, angle: float, team: str = "P1", faction: str | None = None) -> None:
        self.position = position
        self.angle = angle
        self.cooldown_frames = 0
        self.max_health = 1
        self.health = self.max_health
        self.is_alive = True
        self.team = team
        self.faction = str(faction) if faction is not None else team

        # Sticky controller state: persists across environment steps.
        self.move_intent_x = 0
        self.move_intent_y = 0
        self.aim_intent = 0

    def step_sticky_intents(self) -> Vec2:
        self.angle = (self.angle + self.aim_intent * AIM_RATE_PER_STEP) % 360
        # Game coordinates are top-left origin, so world +Y maps to screen-up (negative local Y).
        movement = Vec2(float(self.move_intent_x), float(-self.move_intent_y))
        return movement * PLAYER_MOVE_SPEED

    def shoot(self):
        if self.cooldown_frames > 0 or not self.is_alive:
            return None

        direction = heading_to_vector(self.angle)
        self.cooldown_frames = SHOOT_COOLDOWN_FRAMES
        return {
            "pos": self.position + direction * 20,
            "velocity": direction * PROJECTILE_SPEED,
            "owner": self.team,
            "owner_faction": self.faction,
        }

    def take_hit(self, damage: int = 1) -> bool:
        if not self.is_alive:
            return False
        self.health = max(0, self.health - int(damage))
        if self.health <= 0:
            self.is_alive = False
            return True
        return False

    def tick(self) -> None:
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1


class Renderer:
    """Arcade renderer for the Bang arena."""

    def __init__(self, game, width: int, height: int, title: str, enabled: bool) -> None:
        self.game = game
        self.enabled = bool(enabled)
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

    def close(self) -> None:
        self.window_controller.close()
        self.window = None

    def poll_events(self) -> None:
        self.window_controller.poll_events_or_raise()

    def draw_frame(self) -> None:
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

        for actor_id in self.game.actor_order:
            actor = self.game.players_by_id[actor_id]
            if not actor.is_alive:
                continue
            fill_color, outline_color = self.game.actor_render_colors.get(
                actor_id,
                (COLOR_DEEP_TEAL, COLOR_AQUA),
            )
            is_nn_controlled = actor_id in self.game.controlled_actor_ids
            self._draw_actor(actor, fill_color, outline_color, is_nn_controlled=is_nn_controlled)

        for projectile in self.game.projectiles:
            owner_id = str(projectile.get("owner", ""))
            projectile_color = self.game.actor_projectile_colors.get(owner_id, COLOR_AMBER)
            arcade.draw_circle_filled(
                projectile["pos"].x,
                self.window_controller.to_arcade_y(projectile["pos"].y),
                5,
                projectile_color,
            )

        self._draw_status_bar()
        self.window_controller.flip()

    def _draw_status_bar(self) -> None:
        arcade.draw_lbwh_rectangle_filled(0, 0, self.width, BB_HEIGHT, COLOR_NEAR_BLACK)

        separator = UI_STATUS_SEPARATOR if UI_STATUS_SEPARATOR else " / "
        segments: list[tuple[str, tuple[int, int, int]]] = []
        for idx, player_id in enumerate(self.game.player_order):
            if idx > 0:
                segments.append((separator, COLOR_SOFT_WHITE))
            score_color = self.game.player_status_colors.get(player_id, COLOR_SOFT_WHITE)
            label = player_id.upper()
            segments.append((f"{label} Score: {self.game.scores[player_id]}", score_color))
        if segments:
            segments.append((separator, COLOR_SOFT_WHITE))
        elapsed_seconds = self.game.frame_count / max(1, FPS)
        segments.append((f"Time: {self._format_elapsed_time(elapsed_seconds)}", COLOR_SOFT_WHITE))
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

    @staticmethod
    def _format_elapsed_time(elapsed_seconds: float) -> str:
        total_seconds = max(0, int(elapsed_seconds))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _draw_two_tone_tile(self, top_left_x: float, top_left_y: float, outer_color, inner_color) -> None:
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

    def _draw_actor(self, actor: Actor, fill_color, outline_color, is_nn_controlled: bool = False) -> None:
        self._draw_two_tone_tile(
            top_left_x=actor.position.x - TILE_SIZE / 2,
            top_left_y=actor.position.y - TILE_SIZE / 2,
            outer_color=outline_color,
            inner_color=fill_color,
        )
        if is_nn_controlled:
            marker_size = max(4.0, TILE_SIZE * 0.30)
            marker_top_left_x = actor.position.x - marker_size / 2.0
            marker_top_left_y = actor.position.y - marker_size / 2.0
            marker_bottom = self.window_controller.top_left_to_bottom(marker_top_left_y, marker_size)
            arcade.draw_lbwh_rectangle_filled(
                marker_top_left_x,
                marker_bottom,
                marker_size,
                marker_size,
                outline_color,
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


class BaseGame:
    """Top-down arena game logic with faction-based allies."""

    def __init__(self, level: int = 1, show_game: bool = True):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.show_game = bool(show_game)
        self.frame_clock = ArcadeFrameClock()
        self.allies_per_player = 0
        self.aim_tolerance_degrees = 0.0

        initial_level = max(MIN_LEVEL, min(int(level), MAX_LEVEL))
        initial_player_count = _num_players_for_level(initial_level)
        self.player_order = _resolve_player_order(initial_player_count)
        self.player_render_colors: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
        self.actor_order: list[str] = []
        self.actor_factions: dict[str, str] = {}
        self.faction_actor_ids: dict[str, list[str]] = {}
        self.actor_render_colors: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {}
        self.actor_projectile_colors: dict[str, tuple[int, int, int]] = {}
        # Shots use the player's lighter accent color. This already includes reserved styles like P4.
        self.player_projectile_colors: dict[str, tuple[int, int, int]] = {}
        self.player_status_colors: dict[str, tuple[int, int, int]] = {}
        self.scores: dict[str, int] = {}
        self.controlled_actor_ids: list[str] = []
        self._set_player_count(initial_player_count)

        self.players: list[Actor] = []
        self.players_by_id: dict[str, Actor] = {}
        self.scripted_players: list[Actor] = []
        self.scripted_movement_states: dict[str, ScriptedMovementState] = {}
        self.target_states: dict[str, TargetState] = {}
        self.observation_states: dict[str, ActorObservationState] = {}

        self.renderer = Renderer(
            game=self,
            width=self.width,
            height=self.height,
            title=WINDOW_TITLE,
            enabled=self.show_game,
        )
        self.window_controller = self.renderer.window_controller
        self.window = self.renderer.window

        self.level = level
        self.configure_level()
        self.reset()

    def close(self) -> None:
        self.renderer.close()

    def poll_events(self) -> None:
        self.renderer.poll_events()

    def draw_frame(self) -> None:
        self.renderer.draw_frame()

    def configure_level(self) -> None:
        level = max(MIN_LEVEL, min(self.level, MAX_LEVEL))
        self.level = level
        settings = LEVEL_SETTINGS[level]
        configured_allies = _num_allies_for_level(level)
        allies_per_player = configured_allies if NUM_ALLIES_OVERRIDE is None else int(NUM_ALLIES_OVERRIDE)
        allies_per_player = max(0, allies_per_player)
        allies_changed = allies_per_player != self.allies_per_player
        self.allies_per_player = allies_per_player
        self._set_player_count(_num_players_for_level(level), force_rebuild=allies_changed)

        self.num_obstacles = settings["num_obstacles"]
        self.enemy_move_probability = settings["enemy_move_probability"]
        self.enemy_shot_error_choices = settings["enemy_shot_error_choices"]
        self.enemy_shoot_probability = settings["enemy_shoot_probability"]
        self.aim_tolerance_degrees = float(settings["aim_tolerance_degrees"])

    def _actor_ids_for_faction(self, faction_id: str) -> list[str]:
        ally_ids = [f"{faction_id}A{idx + 1}" for idx in range(self.allies_per_player)]
        return [faction_id, *ally_ids]

    def _build_actor_order(self, player_order: tuple[str, ...]) -> list[str]:
        actor_order: list[str] = []
        for faction_id in player_order:
            actor_order.extend(self._actor_ids_for_faction(faction_id))
        return actor_order

    def _set_player_count(self, num_players: int, force_rebuild: bool = False) -> None:
        player_order = _resolve_player_order(num_players)
        if (not force_rebuild) and player_order == self.player_order and self.scores:
            return

        old_scores = dict(self.scores)
        self.player_order = player_order
        self.actor_order = self._build_actor_order(player_order)
        self.player_render_colors = {
            player_id: (
                PLAYER_STYLES[player_id]["render_fill"],
                PLAYER_STYLES[player_id]["render_outline"],
            )
            for player_id in self.player_order
        }
        self.actor_factions = {}
        self.faction_actor_ids = {}
        self.actor_render_colors = {}
        self.actor_projectile_colors = {}
        for faction_id in self.player_order:
            style = PLAYER_STYLES[faction_id]
            actor_ids = self._actor_ids_for_faction(faction_id)
            self.faction_actor_ids[faction_id] = actor_ids
            for actor_id in actor_ids:
                self.actor_factions[actor_id] = faction_id
                self.actor_render_colors[actor_id] = (
                    style["render_fill"],
                    style["render_outline"],
                )
                self.actor_projectile_colors[actor_id] = style["render_outline"]
        # Backward-compatible alias expected by older callers.
        self.player_projectile_colors = dict(self.actor_projectile_colors)
        self.player_status_colors = {
            player_id: PLAYER_STYLES[player_id]["status_color"]
            for player_id in self.player_order
        }
        self.scores = {
            player_id: int(old_scores.get(player_id, 0))
            for player_id in self.player_order
        }
        for player_id in self.player_order:
            setattr(self, f"{player_id}_score", self.scores[player_id])

    def reset(self) -> None:
        spawn_positions = self._spawn_positions_by_actor()

        self.players_by_id = {}
        for actor_id in self.actor_order:
            spawn_pos = spawn_positions[actor_id]
            faction_id = self.actor_factions[actor_id]
            self.players_by_id[actor_id] = Actor(
                spawn_pos,
                angle=self._sample_inner_facing_angle(spawn_pos),
                team=actor_id,
                faction=faction_id,
            )

        self.players = [self.players_by_id[actor_id] for actor_id in self.actor_order]
        self.player = self.players_by_id["P1"]
        # Backward-compatible aliases for older callers.
        self.enemy = self.players_by_id["P2"]
        self.enemy2 = self.players_by_id.get("P3")
        self.enemy3 = self.players_by_id.get("P4")
        self.scripted_players = [
            actor
            for actor in self.players
            if actor.team != self.player.team
        ]
        self.controlled_actor_ids = [self.player.team]

        self.obstacles: list[Vec2] = []
        self.projectiles: list[dict[str, object]] = []
        self.frame_count = 0
        self.target_states = {
            actor.team: TargetState()
            for actor in self.players
        }
        self.scripted_movement_states = {
            actor.team: ScriptedMovementState()
            for actor in self.players
        }
        self.observation_states = {
            actor.team: ActorObservationState()
            for actor in self.players
        }
        self._place_obstacles()

    def _spawn_y_bounds(self) -> tuple[float, float]:
        center_y = self.height / 2 - BB_HEIGHT // 2
        min_y = center_y - SPAWN_Y_OFFSET
        max_y = center_y + SPAWN_Y_OFFSET

        min_actor_y = TILE_SIZE / 2
        max_actor_y = self.height - BB_HEIGHT - TILE_SIZE / 2
        min_y = max(min_y, min_actor_y)
        max_y = min(max_y, max_actor_y)
        return min_y, max_y

    def _spawn_x_bounds(self) -> tuple[float, float]:
        center_x = self.width / 2
        min_x = center_x - SPAWN_Y_OFFSET
        max_x = center_x + SPAWN_Y_OFFSET

        min_actor_x = TILE_SIZE / 2
        max_actor_x = self.width - TILE_SIZE / 2
        min_x = max(min_x, min_actor_x)
        max_x = min(max_x, max_actor_x)
        return min_x, max_x

    def _spawn_bottom_strip_y(self) -> float:
        playable_height = self.height - BB_HEIGHT
        bottom_edge_y = playable_height - TILE_SIZE / 2
        bottom_padding = playable_height * PLAYER_SPAWN_X_RATIO
        return max(TILE_SIZE / 2, bottom_edge_y - bottom_padding)

    def _spawn_top_strip_y(self) -> float:
        top_edge_y = TILE_SIZE / 2
        top_padding = self.height * PLAYER_SPAWN_X_RATIO
        return min(self.height - BB_HEIGHT - TILE_SIZE / 2, top_edge_y + top_padding)

    def _active_spawn_areas(self) -> tuple[str, ...]:
        player_count = len(self.player_order)
        if player_count <= 0:
            return tuple()
        return SPAWN_AREA_ORDER[: min(player_count, len(SPAWN_AREA_ORDER))]

    def _sample_spawn_position(self, area: str) -> Vec2:
        min_y, max_y = self._spawn_y_bounds()
        min_x, max_x = self._spawn_x_bounds()

        if area == SPAWN_AREA_LEFT:
            return Vec2(self.width * PLAYER_SPAWN_X_RATIO, random.uniform(min_y, max_y))
        if area == SPAWN_AREA_RIGHT:
            return Vec2(self.width * ENEMY_SPAWN_X_RATIO, random.uniform(min_y, max_y))
        if area == SPAWN_AREA_BOTTOM:
            return Vec2(random.uniform(min_x, max_x), self._spawn_bottom_strip_y())
        if area == SPAWN_AREA_TOP:
            return Vec2(random.uniform(min_x, max_x), self._spawn_top_strip_y())
        raise ValueError(f"Unknown spawn area: {area}")

    def _spawn_positions_by_actor(self) -> dict[str, Vec2]:
        area_order = list(self._active_spawn_areas())
        random.shuffle(area_order)
        positions: dict[str, Vec2] = {}
        if not area_order:
            return positions

        area_by_faction = {
            faction_id: area_order[idx % len(area_order)]
            for idx, faction_id in enumerate(self.player_order)
        }
        for faction_id in self.player_order:
            area = area_by_faction[faction_id]
            for actor_id in self.faction_actor_ids.get(faction_id, [faction_id]):
                positions[actor_id] = self._sample_spawn_position(area)
        return positions

    def _sample_inner_facing_angle(self, position: Vec2) -> float:
        arena_center = Vec2(self.width / 2.0, (self.height - BB_HEIGHT) / 2.0)
        to_center = arena_center - position
        if length_squared(to_center) == 0:
            base_angle = random.uniform(0.0, 360.0)
        else:
            base_angle = math.degrees(math.atan2(to_center.y, to_center.x))
        return (base_angle + random.uniform(-90.0, 90.0)) % 360.0

    def _apply_action_to_actor_intents(self, actor: Actor, action_index: int) -> bool:
        if action_index == ACTION_MOVE_UP:
            actor.move_intent_x = 0
            actor.move_intent_y = 1
            return False
        if action_index == ACTION_MOVE_DOWN:
            actor.move_intent_x = 0
            actor.move_intent_y = -1
            return False
        if action_index == ACTION_MOVE_LEFT:
            actor.move_intent_x = -1
            actor.move_intent_y = 0
            return False
        if action_index == ACTION_MOVE_RIGHT:
            actor.move_intent_x = 1
            actor.move_intent_y = 0
            return False
        if action_index == ACTION_STOP_MOVE:
            actor.move_intent_x = 0
            actor.move_intent_y = 0
            return False
        if action_index == ACTION_AIM_LEFT:
            actor.aim_intent = -1
            return False
        if action_index == ACTION_AIM_RIGHT:
            actor.aim_intent = 1
            return False
        if action_index == ACTION_SHOOT:
            projectile = actor.shoot()
            if projectile:
                self.projectiles.append(projectile)
                return True
        return False

    def apply_actor_action(self, actor: Actor, action_index: int | None) -> bool:
        if not actor.is_alive:
            return False

        issued_movement_action = False
        if action_index is not None:
            action_value = int(action_index)
            self._apply_action_to_actor_intents(actor, action_value)
            self._observation_state_for(actor).last_action_index = action_value
            issued_movement_action = action_value in {
                ACTION_MOVE_UP,
                ACTION_MOVE_DOWN,
                ACTION_MOVE_LEFT,
                ACTION_MOVE_RIGHT,
            }

        previous_position = actor.position
        movement = actor.step_sticky_intents()
        self._update_actor_position(actor, movement)
        blocked_move = issued_movement_action and length_squared(actor.position - previous_position) == 0
        return blocked_move

    def apply_player_action(self, action_index: int | None) -> bool:
        return self.apply_actor_action(self.player, action_index)

    def _update_actor_position(self, actor: Actor, movement: Vec2) -> None:
        if not actor.is_alive:
            return

        new_position = actor.position + movement
        actor_rect = rect_from_center(new_position, TILE_SIZE)
        if collides_with_square_arena(
            rect=actor_rect,
            obstacles=self.obstacles,
            tile_size=TILE_SIZE,
            arena_width=self.width,
            arena_height=self.height,
            bottom_bar_height=BB_HEIGHT,
        ):
            return

        for other in self.players:
            if other is actor or not other.is_alive:
                continue
            other_rect = rect_from_center(other.position, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                return

        actor.position = new_position

    def _would_collide(self, actor: Actor, movement: Vec2) -> bool:
        if not actor.is_alive:
            return True

        new_position = actor.position + movement
        actor_rect = rect_from_center(new_position, TILE_SIZE)
        if collides_with_square_arena(
            rect=actor_rect,
            obstacles=self.obstacles,
            tile_size=TILE_SIZE,
            arena_width=self.width,
            arena_height=self.height,
            bottom_bar_height=BB_HEIGHT,
        ):
            return True

        for other in self.players:
            if other is actor or not other.is_alive:
                continue
            other_rect = rect_from_center(other.position, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                return True
        return False

    @staticmethod
    def _clip(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
        return max(min_value, min(max_value, value))

    @staticmethod
    def _normalize_elapsed_frames(
        frames: int,
        normalization_frames: int = EVENT_TIMER_NORMALIZATION_FRAMES,
    ) -> float:
        return min(1.0, max(0, frames) / max(1, normalization_frames))

    def _observation_state_for(self, actor: Actor) -> ActorObservationState:
        return self.observation_states.setdefault(actor.team, ActorObservationState())

    def _update_enemy_seen_timer(self, actor: Actor, enemy_in_los: bool) -> float:
        state = self._observation_state_for(actor)
        if enemy_in_los:
            state.last_seen_enemy_frame = self.frame_count
        return self._normalize_elapsed_frames(self.frame_count - state.last_seen_enemy_frame)

    def _normalize_distance(self, distance: float) -> float:
        return self._clip(distance / max(self.width, self.height), 0.0, 1.0)

    @staticmethod
    def _wrapped_angle_delta(current_angle: float, previous_angle: float) -> float:
        return math.atan2(
            math.sin(current_angle - previous_angle),
            math.cos(current_angle - previous_angle),
        )

    @staticmethod
    def _relative_angle(actor: Actor, target_position: Vec2) -> float:
        to_target = target_position - actor.position
        target_angle = math.atan2(to_target.y, to_target.x)
        actor_angle = math.radians(actor.angle)
        return math.atan2(
            math.sin(target_angle - actor_angle),
            math.cos(target_angle - actor_angle),
        )

    def _directional_clearance_norm(self, actor: Actor, direction_vec: Vec2) -> float:
        horizon_steps = max(1, int(CLEARANCE_HORIZON_STEPS))
        for step in range(1, horizon_steps + 1):
            if self._would_collide(actor, direction_vec * step):
                return float(step - 1) / float(horizon_steps)
        return 1.0

    @staticmethod
    def _build_state_vector_from_features(feature_values: dict[str, float]) -> list[float]:
        return [float(feature_values[name]) for name in INPUT_FEATURE_NAMES]

    def _place_obstacles(self) -> None:
        self.obstacles = []
        shapes = spawn_connected_random_walk_shapes(
            shape_count=self.num_obstacles,
            min_sections=MIN_OBSTACLE_SECTIONS,
            max_sections=MAX_OBSTACLE_SECTIONS,
            sample_start_fn=self._sample_valid_obstacle_start,
            neighbor_candidates_fn=self._neighbor_obstacle_candidates,
            is_candidate_valid_fn=self._is_valid_obstacle_tile,
        )
        for shape in shapes:
            self.obstacles.extend(shape)

    def _sample_valid_obstacle_start(self):
        for _ in range(OBSTACLE_START_ATTEMPTS):
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BB_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            point = Vec2(x, y)
            if self._is_valid_obstacle_tile(point, []):
                return point
        return None

    def _is_valid_obstacle_tile(self, tile: Vec2, pending_tiles) -> bool:
        if not (0 <= tile.x < self.width and 0 <= tile.y < self.height - BB_HEIGHT):
            return False
        if any(tile == existing for existing in self.obstacles) or any(tile == existing for existing in pending_tiles):
            return False
        if any(tile.distance(actor.position) < SAFE_RADIUS for actor in self.players if actor.is_alive):
            return False
        return True

    @staticmethod
    def _neighbor_obstacle_candidates(tile: Vec2) -> list[Vec2]:
        return [
            Vec2(tile.x - TILE_SIZE, tile.y),
            Vec2(tile.x + TILE_SIZE, tile.y),
            Vec2(tile.x, tile.y - TILE_SIZE),
            Vec2(tile.x, tile.y + TILE_SIZE),
        ]

    @staticmethod
    def _move_vector_for_angle(angle_degrees: float) -> Vec2:
        return rotate_degrees(Vec2(1, 0), angle_degrees) * PLAYER_MOVE_SPEED

    def _move_actor_in_direction(self, actor: Actor, angle_degrees: float) -> bool:
        previous_position = actor.position
        movement = self._move_vector_for_angle(angle_degrees)
        self._update_actor_position(actor, movement)
        return length_squared(actor.position - previous_position) > 0

    def _attempt_actor_move_with_fallback(self, actor: Actor, desired_angle: float) -> bool:
        candidate_offsets = SCRIPTED_MOVE_FALLBACK_OFFSETS + ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES
        for offset in candidate_offsets:
            candidate_angle = (desired_angle + offset) % 360
            candidate_move = self._move_vector_for_angle(candidate_angle)
            if self._would_collide(actor, candidate_move):
                continue
            self._update_actor_position(actor, candidate_move)
            return True
        return False

    def _alive_allies(self, actor: Actor) -> list[Actor]:
        return [
            other
            for other in self.players
            if other is not actor and other.is_alive and other.faction == actor.faction
        ]

    def _alive_opponents(self, actor: Actor) -> list[Actor]:
        return [
            other
            for other in self.players
            if other is not actor and other.is_alive and other.faction != actor.faction
        ]

    def _resolve_alive_target(self, actor: Actor, target_id: str | None) -> Actor | None:
        if target_id is None:
            return None
        target = self.players_by_id.get(target_id)
        if target is None or target is actor or not target.is_alive or target.faction == actor.faction:
            return None
        return target

    def _has_clear_path_between(self, actor: Actor, target: Actor) -> bool:
        return not square_obstacle_between_points(
            point_a=actor.position,
            point_b=target.position,
            obstacles=self.obstacles,
            tile_size=TILE_SIZE,
        )

    def _is_actor_aimed_at_target(self, actor: Actor, target: Actor) -> bool:
        to_target = target.position - actor.position
        if length_squared(to_target) == 0:
            return True
        target_angle = math.degrees(math.atan2(to_target.y, to_target.x))
        relative = normalize_angle_degrees(target_angle - actor.angle)
        return abs(relative) <= self.aim_tolerance_degrees

    def _nearest_target(self, actor: Actor, candidates: list[Actor], require_clear_path: bool) -> Actor | None:
        filtered = candidates
        if require_clear_path:
            filtered = [candidate for candidate in candidates if self._has_clear_path_between(actor, candidate)]
        if not filtered:
            return None
        return min(filtered, key=lambda candidate: actor.position.distance(candidate.position))

    def _reset_actor_target_tracking(self, actor: Actor, target: Actor | None) -> None:
        observation = self._observation_state_for(actor)
        observation.previous_enemy_distance = None
        observation.previous_enemy_relative_angle = None
        observation.last_seen_enemy_frame = (
            self.frame_count
            if target is not None and self.has_line_of_sight(actor=actor, target=target)
            else -EVENT_TIMER_NORMALIZATION_FRAMES
        )

    def _set_target_state(
        self,
        actor: Actor,
        state: TargetState,
        target: Actor | None,
        _policy: dict[str, float | int],
    ) -> None:
        previous_target_id = state.target_id
        state.target_id = target.team if target is not None else None
        state.target_lost_frames = 0
        state.target_switch_cooldown = 0
        state.target_lock_frames_remaining = TARGET_LOCK_FRAMES if target is not None else 0
        if state.target_id != previous_target_id:
            self._reset_actor_target_tracking(actor, target)

    def _select_target(
        self,
        actor: Actor,
        policy: dict[str, float | int],
        cache_by_frame: bool,
    ) -> Actor | None:
        state = self.target_states.setdefault(actor.team, TargetState())
        if cache_by_frame and state.last_update_frame == self.frame_count:
            return self._resolve_alive_target(actor, state.target_id)
        state.last_update_frame = self.frame_count

        if not actor.is_alive:
            self._set_target_state(actor, state, None, policy)
            return None

        candidates = self._alive_opponents(actor)
        if not candidates:
            self._set_target_state(actor, state, None, policy)
            return None

        nearest = self._nearest_target(actor, candidates, require_clear_path=False)
        current = self._resolve_alive_target(actor, state.target_id)
        if current is None:
            self._set_target_state(actor, state, nearest, policy)
            return nearest

        if nearest is not None and nearest is not current:
            current_distance = actor.position.distance(current.position)
            nearest_distance = actor.position.distance(nearest.position)
            # Allow early switch only when the newly nearest enemy is clearly better.
            if nearest_distance < current_distance * TARGET_SWITCH_CLOSER_RATIO:
                self._set_target_state(actor, state, nearest, policy)
                return nearest

        if state.target_lock_frames_remaining <= 0:
            if nearest is not None and nearest is not current:
                self._set_target_state(actor, state, nearest, policy)
                return nearest
            state.target_lock_frames_remaining = TARGET_LOCK_FRAMES

        state.target_lock_frames_remaining = max(0, state.target_lock_frames_remaining - 1)
        return current

    def _target_policy_for_actor(self, actor: Actor) -> dict[str, float | int]:
        if actor.team in self.controlled_actor_ids:
            return PLAYER_TARGET_POLICY
        return SCRIPTED_TARGET_POLICY

    def _get_actor_target(self, actor: Actor, cache_by_frame: bool = True) -> Actor | None:
        return self._select_target(
            actor=actor,
            policy=self._target_policy_for_actor(actor),
            cache_by_frame=cache_by_frame,
        )

    def _get_player_target(self) -> Actor | None:
        return self._get_actor_target(self.player, cache_by_frame=True)

    def _arena_center(self) -> Vec2:
        return Vec2(self.width / 2.0, (self.height - BB_HEIGHT) / 2.0)

    def _scripted_escape_heading(self, actor: Actor, angle_to_target: float) -> float:
        to_center = self._arena_center() - actor.position
        if length_squared(to_center) == 0:
            center_angle = (angle_to_target + 180.0) % 360.0
        else:
            center_angle = math.degrees(math.atan2(to_center.y, to_center.x)) % 360.0

        candidate_angles = (
            center_angle,
            (center_angle + 35.0) % 360.0,
            (center_angle - 35.0) % 360.0,
            (center_angle + 90.0) % 360.0,
            (center_angle - 90.0) % 360.0,
            (angle_to_target + 180.0) % 360.0,
        )
        for candidate_angle in candidate_angles:
            if not self._would_collide(actor, self._move_vector_for_angle(candidate_angle)):
                return candidate_angle
        return center_angle

    def _scripted_desired_move_angle(
        self,
        actor: Actor,
        target: Actor,
        angle_to_target: float,
        movement_state: ScriptedMovementState,
    ) -> float:
        if movement_state.escape_frames_remaining > 0:
            return movement_state.escape_angle

        distance = actor.position.distance(target.position)
        if distance < SAFE_RADIUS * 0.9:
            return (angle_to_target + 180.0) % 360.0
        if distance > SAFE_RADIUS * 1.8:
            return angle_to_target
        edge_margin = TILE_SIZE * 2.5
        playable_height = self.height - BB_HEIGHT
        near_edge = (
            actor.position.x <= edge_margin
            or actor.position.x >= self.width - edge_margin
            or actor.position.y <= edge_margin
            or actor.position.y >= playable_height - edge_margin
        )
        if near_edge and random.random() < 0.60:
            return self._scripted_escape_heading(actor, angle_to_target)
        if random.random() < 0.35:
            return (angle_to_target + random.choice((90.0, -90.0))) % 360.0
        return angle_to_target

    def _would_hit_ally_before_target(self, actor: Actor, target: Actor) -> bool:
        target_aimed = self._is_actor_aimed_at_target(actor, target)
        target_distance = actor.position.distance(target.position)
        for ally in self._alive_allies(actor):
            if not self._is_actor_aimed_at_target(actor, ally):
                continue
            if not self._has_clear_path_between(actor, ally):
                continue
            ally_distance = actor.position.distance(ally.position)
            if (not target_aimed) or ally_distance < target_distance:
                return True
        return False

    def _step_scripted_actor(self, actor: Actor) -> None:
        if not actor.is_alive:
            return

        movement_state = self.scripted_movement_states.setdefault(actor.team, ScriptedMovementState())
        target = self._get_actor_target(actor, cache_by_frame=False)
        if target is None:
            movement_state.blocked_moves = 0
            movement_state.escape_frames_remaining = 0
            return

        to_target = target.position - actor.position
        if length_squared(to_target) == 0:
            angle_to_target = actor.angle
        else:
            angle_to_target = math.degrees(math.atan2(to_target.y, to_target.x)) % 360

        aim_error = random.choice(self.enemy_shot_error_choices)
        aim_angle = (angle_to_target + aim_error) % 360
        actor.angle = aim_angle

        should_attempt_move = (
            movement_state.escape_frames_remaining > 0
            or random.random() < self.enemy_move_probability
        )
        if should_attempt_move:
            move_angle = self._scripted_desired_move_angle(actor, target, angle_to_target, movement_state)
            moved = self._attempt_actor_move_with_fallback(actor, move_angle)
            if moved:
                movement_state.blocked_moves = 0
                if movement_state.escape_frames_remaining > 0:
                    movement_state.escape_frames_remaining -= 1
            else:
                movement_state.blocked_moves += 1
                if movement_state.blocked_moves >= ENEMY_STUCK_MOVE_ATTEMPTS:
                    movement_state.blocked_moves = 0
                    movement_state.escape_frames_remaining = ENEMY_ESCAPE_FOLLOW_FRAMES
                    movement_state.escape_angle = self._scripted_escape_heading(actor, angle_to_target)

        shoot_probability = self.enemy_shoot_probability
        if self._has_clear_path_between(actor, target):
            shoot_probability = min(1.0, shoot_probability * 1.25)
        if random.random() < shoot_probability and not self._would_hit_ally_before_target(actor, target):
            projectile = actor.shoot()
            if projectile:
                self.projectiles.append(projectile)

    def _step_scripted_players(self) -> None:
        for actor in self.scripted_players:
            self._step_scripted_actor(actor)

    def _step_projectiles(self):
        events = {
            "controlled_hits": 0,
            "controlled_friendly_fire": 0,
            "player_killed_by": None,
        }
        next_projectiles = []
        controlled_actor_id = self.player.team
        controlled_faction = self.player.faction

        for projectile in self.projectiles:
            projectile["pos"] += projectile["velocity"]
            projectile_rect = rect_from_center(projectile["pos"], PROJECTILE_HITBOX_SIZE)
            if collides_with_square_arena(
                rect=projectile_rect,
                obstacles=self.obstacles,
                tile_size=TILE_SIZE,
                arena_width=self.width,
                arena_height=self.height,
                bottom_bar_height=BB_HEIGHT,
            ):
                continue

            owner_id = str(projectile["owner"])
            owner_faction = str(projectile.get("owner_faction", owner_id))
            colliding_targets = []
            for target in self.players:
                if not target.is_alive or target.team == owner_id:
                    continue
                target_rect = rect_from_center(target.position, TILE_SIZE)
                if projectile_rect.colliderect(target_rect):
                    colliding_targets.append(target)

            if colliding_targets:
                target = min(colliding_targets, key=lambda candidate: candidate.position.distance(projectile["pos"]))
                was_killed = target.take_hit(1)
                if owner_id == controlled_actor_id:
                    if target.faction == controlled_faction:
                        events["controlled_friendly_fire"] += 1
                    else:
                        events["controlled_hits"] += 1
                if was_killed and target.team == controlled_actor_id and events["player_killed_by"] is None:
                    if not self.player.is_alive:
                        events["player_killed_by"] = owner_faction
                continue

            next_projectiles.append(projectile)

        self.projectiles = next_projectiles
        return events

    def _distance_to_closest_incoming_projectile(self, actor: Actor):
        incoming_projectiles = [
            projectile
            for projectile in self.projectiles
            if str(projectile.get("owner", "")) != actor.team
        ]
        if not incoming_projectiles:
            return None

        def _projectile_threat_rank(projectile: dict[str, object]) -> tuple[int, float]:
            to_actor = actor.position - projectile["pos"]
            distance = actor.position.distance(projectile["pos"])
            if length_squared(to_actor) == 0:
                return (-1, 0.0)
            projectile_dir = projectile["velocity"].normalize()
            approaching = projectile_dir.dot(to_actor.normalize()) > PROJECTILE_TRAJECTORY_DOT_THRESHOLD
            return (0 if approaching else 1, distance)

        return min(incoming_projectiles, key=_projectile_threat_rank)

    def is_actor_in_projectile_trajectory(self, actor: Actor) -> bool:
        for projectile in self.projectiles:
            owner_id = str(projectile.get("owner", ""))
            if owner_id == actor.team:
                continue
            to_actor = actor.position - projectile["pos"]
            if length_squared(to_actor) == 0:
                return True
            projectile_dir = projectile["velocity"].normalize()
            if projectile_dir.dot(to_actor.normalize()) > PROJECTILE_TRAJECTORY_DOT_THRESHOLD:
                return True
        return False

    def has_line_of_sight(self, actor: Actor | None = None, target: Actor | None = None) -> bool:
        if actor is None:
            actor = self.player
        if target is None:
            target = self._get_actor_target(actor, cache_by_frame=True)
        if target is None:
            return False
        if target.faction == actor.faction:
            return False
        return self._has_clear_path_between(actor, target)

    def get_state_vector_for_actor(self, actor: Actor) -> list[float]:
        observation = self._observation_state_for(actor)
        target = self._get_actor_target(actor, cache_by_frame=True)
        if target is None:
            enemy_distance = 1.0
            enemy_relative_angle = 0.0
            enemy_rel_sin = 0.0
            enemy_rel_cos = 1.0
            enemy_in_los = 0.0
            delta_enemy_distance = 0.0
            delta_enemy_rel_angle = 0.0
            observation.previous_enemy_distance = None
            observation.previous_enemy_relative_angle = None
        else:
            enemy_distance = self._normalize_distance(actor.position.distance(target.position))
            enemy_relative_angle = self._relative_angle(actor, target.position)
            enemy_rel_sin = math.sin(enemy_relative_angle)
            enemy_rel_cos = math.cos(enemy_relative_angle)
            enemy_in_los = 1.0 if self.has_line_of_sight(actor=actor, target=target) else 0.0

            if observation.previous_enemy_distance is None:
                delta_enemy_distance = 0.0
            else:
                delta_enemy_distance = self._clip(
                    observation.previous_enemy_distance - enemy_distance,
                    -1.0,
                    1.0,
                )

            if observation.previous_enemy_relative_angle is None:
                delta_enemy_rel_angle = 0.0
            else:
                delta_enemy_rel_angle = self._clip(
                    self._wrapped_angle_delta(enemy_relative_angle, observation.previous_enemy_relative_angle) / math.pi,
                    -1.0,
                    1.0,
                )

            observation.previous_enemy_distance = enemy_distance
            observation.previous_enemy_relative_angle = enemy_relative_angle

        allies = self._alive_allies(actor)
        if not allies:
            ally_present = 0.0
            ally_distance = 1.0
            ally_rel_sin = 0.0
            ally_rel_cos = 1.0
        else:
            closest_ally = min(allies, key=lambda teammate: actor.position.distance(teammate.position))
            ally_present = 1.0
            ally_distance = self._normalize_distance(actor.position.distance(closest_ally.position))
            ally_relative_angle = self._relative_angle(actor, closest_ally.position)
            ally_rel_sin = math.sin(ally_relative_angle)
            ally_rel_cos = math.cos(ally_relative_angle)

        closest_projectile = self._distance_to_closest_incoming_projectile(actor)
        if closest_projectile is None:
            projectile_present = 0.0
            projectile_distance = 1.0
            projectile_rel_sin = 0.0
            projectile_rel_cos = 1.0
            delta_projectile_distance = 0.0
            delta_projectile_rel_angle = 0.0
            observation.previous_projectile_distance = None
            observation.previous_projectile_relative_angle = None
        else:
            projectile_present = 1.0
            projectile_distance = self._normalize_distance(actor.position.distance(closest_projectile["pos"]))
            projectile_relative_angle = self._relative_angle(actor, closest_projectile["pos"])
            projectile_rel_sin = math.sin(projectile_relative_angle)
            projectile_rel_cos = math.cos(projectile_relative_angle)

            if observation.previous_projectile_distance is None:
                delta_projectile_distance = 0.0
            else:
                delta_projectile_distance = self._clip(
                    observation.previous_projectile_distance - projectile_distance,
                    -1.0,
                    1.0,
                )

            if observation.previous_projectile_relative_angle is None:
                delta_projectile_rel_angle = 0.0
            else:
                delta_projectile_rel_angle = self._clip(
                    self._wrapped_angle_delta(
                        projectile_relative_angle,
                        observation.previous_projectile_relative_angle,
                    )
                    / math.pi,
                    -1.0,
                    1.0,
                )

            observation.previous_projectile_distance = projectile_distance
            observation.previous_projectile_relative_angle = projectile_relative_angle

        player_angle_radians = math.radians(actor.angle)
        player_angle_sin = math.sin(player_angle_radians)
        player_angle_cos = math.cos(player_angle_radians)

        up_clearance_norm = self._directional_clearance_norm(actor, Vec2(0, -PLAYER_MOVE_SPEED))
        down_clearance_norm = self._directional_clearance_norm(actor, Vec2(0, PLAYER_MOVE_SPEED))
        left_clearance_norm = self._directional_clearance_norm(actor, Vec2(-PLAYER_MOVE_SPEED, 0))
        right_clearance_norm = self._directional_clearance_norm(actor, Vec2(PLAYER_MOVE_SPEED, 0))

        cooldown_norm = self._clip(actor.cooldown_frames / max(1, SHOOT_COOLDOWN_FRAMES), 0.0, 1.0)
        num_actions = len(ACTION_NAMES)
        if num_actions <= 1:
            last_action_index_norm = 0.0
        else:
            last_action_index = max(0, min(num_actions - 1, int(observation.last_action_index)))
            last_action_index_norm = (2.0 * last_action_index / float(num_actions - 1)) - 1.0

        feature_values = {
            "enemy_distance": enemy_distance,
            "enemy_rel_sin": enemy_rel_sin,
            "enemy_rel_cos": enemy_rel_cos,
            "enemy_in_los": enemy_in_los,
            "delta_enemy_distance": delta_enemy_distance,
            "delta_enemy_rel_angle": delta_enemy_rel_angle,
            "ally_present": ally_present,
            "ally_distance": ally_distance,
            "ally_rel_sin": ally_rel_sin,
            "ally_rel_cos": ally_rel_cos,
            "projectile_present": projectile_present,
            "projectile_distance": projectile_distance,
            "projectile_rel_sin": projectile_rel_sin,
            "projectile_rel_cos": projectile_rel_cos,
            "delta_projectile_distance": delta_projectile_distance,
            "delta_projectile_rel_angle": delta_projectile_rel_angle,
            "player_angle_sin": player_angle_sin,
            "player_angle_cos": player_angle_cos,
            "up_clearance_norm": up_clearance_norm,
            "down_clearance_norm": down_clearance_norm,
            "left_clearance_norm": left_clearance_norm,
            "right_clearance_norm": right_clearance_norm,
            "cooldown_norm": cooldown_norm,
            "last_action_index_norm": last_action_index_norm,
        }
        state_vector = self._build_state_vector_from_features(feature_values)
        assert len(state_vector) == 24
        return state_vector

    def get_state_vector(self) -> list[float]:
        return self.get_state_vector_for_actor(self.player)

    def get_controlled_state_vectors(self) -> dict[str, list[float]]:
        return {
            actor_id: self.get_state_vector_for_actor(self.players_by_id[actor_id])
            for actor_id in self.controlled_actor_ids
            if actor_id in self.players_by_id and self.players_by_id[actor_id].is_alive
        }

    def _tick_players(self) -> None:
        for actor in self.players:
            actor.tick()

    def _alive_factions(self) -> set[str]:
        return {actor.faction for actor in self.players if actor.is_alive}

    def _last_alive_faction(self) -> str | None:
        alive_factions = self._alive_factions()
        if len(alive_factions) == 1:
            return next(iter(alive_factions))
        return None

    def _last_alive_player(self) -> Actor | None:
        winning_faction = self._last_alive_faction()
        if winning_faction is None:
            return None
        return next(
            (actor for actor in self.players if actor.is_alive and actor.faction == winning_faction),
            None,
        )

    def is_player_last_alive(self) -> bool:
        return self._last_alive_faction() == self.player.faction

    def _increment_score(self, player_id: str) -> None:
        if player_id not in self.scores:
            return
        self.scores[player_id] += 1
        setattr(self, f"{player_id}_score", self.scores[player_id])


class HumanGame(BaseGame):
    """Human-play mode."""

    def __init__(self, show_game: bool = True):
        level = max(MIN_LEVEL, min(STARTING_LEVEL, MAX_LEVEL))
        super().__init__(level=level, show_game=show_game)

    def play_step(self) -> None:
        self.frame_count += 1
        self.poll_events()

        action = None
        if self.player.is_alive:
            move_up = self.window_controller.is_key_down(arcade.key.W)
            move_down = self.window_controller.is_key_down(arcade.key.S)
            move_left = self.window_controller.is_key_down(arcade.key.A)
            move_right = self.window_controller.is_key_down(arcade.key.D)

            if move_up and not move_down:
                self.player.move_intent_x = 0
                self.player.move_intent_y = 1
            elif move_down and not move_up:
                self.player.move_intent_x = 0
                self.player.move_intent_y = -1
            elif move_left and not move_right:
                self.player.move_intent_x = -1
                self.player.move_intent_y = 0
            elif move_right and not move_left:
                self.player.move_intent_x = 1
                self.player.move_intent_y = 0
            else:
                self.player.move_intent_x = 0
                self.player.move_intent_y = 0

            # Prefer mouse/touchpad aiming in human mode.
            mouse_pos = self.window_controller.mouse_position()
            if mouse_pos is not None:
                mouse_x, mouse_y_arcade = mouse_pos
                mouse_y = self.window_controller.to_top_left_y(mouse_y_arcade)
                to_cursor = Vec2(mouse_x, mouse_y) - self.player.position
                if length_squared(to_cursor) > 0:
                    self.player.angle = math.degrees(math.atan2(to_cursor.y, to_cursor.x)) % 360
                self.player.aim_intent = 0
            else:
                aim_left = self.window_controller.is_key_down(arcade.key.Q) or self.window_controller.is_key_down(
                    arcade.key.LEFT
                )
                aim_right = self.window_controller.is_key_down(arcade.key.E) or self.window_controller.is_key_down(
                    arcade.key.RIGHT
                )
                if aim_left and not aim_right:
                    self.player.aim_intent = -1
                elif aim_right and not aim_left:
                    self.player.aim_intent = 1
                else:
                    self.player.aim_intent = 0

            shoot_pressed = self.window_controller.is_key_down(
                arcade.key.SPACE
            ) or self.window_controller.is_mouse_button_down(arcade.MOUSE_BUTTON_LEFT)
            action = ACTION_SHOOT if shoot_pressed else None
        else:
            self.player.move_intent_x = 0
            self.player.move_intent_y = 0
            self.player.aim_intent = 0

        self.apply_player_action(action)
        self._step_scripted_players()
        self._step_projectiles()
        self._tick_players()

        winner = self._last_alive_player()
        if winner is not None:
            self._increment_score(winner.faction)
            self.reset()

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else 0)


class TrainingGame(BaseGame):
    """Environment used by DQN training."""

    def __init__(
        self,
        level: int = 1,
        show_game: bool = True,
        end_on_player_death: bool = True,
        control_allies_with_nn: bool = TRAIN_NN_CONTROLS_ALLIES,
    ):
        self.end_on_player_death = bool(end_on_player_death)
        self.control_allies_with_nn = bool(control_allies_with_nn)
        self.player_loss_recorded = False
        super().__init__(level=level, show_game=show_game)

    def reset(self) -> None:
        super().reset()
        player_faction_actor_ids = list(self.faction_actor_ids.get(self.player.faction, [self.player.team]))
        if self.control_allies_with_nn:
            self.controlled_actor_ids = player_faction_actor_ids
        else:
            self.controlled_actor_ids = [self.player.team]
        controlled_ids = set(self.controlled_actor_ids)
        self.scripted_players = [actor for actor in self.players if actor.team not in controlled_ids]
        self.player_loss_recorded = False

    def _player_win_condition(self, controlled_alive: bool) -> bool:
        if not controlled_alive:
            return False
        if self.allies_per_player <= 0:
            return not any(actor.is_alive and actor.team != self.player.team for actor in self.players)
        return not self._alive_opponents(self.player)

    @staticmethod
    def _action_index_from_one_hot(action_one_hot: list[int]) -> int:
        return action_one_hot.index(1) if 1 in action_one_hot else ACTION_STOP_MOVE

    def _normalize_controlled_actions(self, action: dict[str, list[int]] | list[int]) -> dict[str, int]:
        if isinstance(action, dict):
            action_indices: dict[str, int] = {}
            for actor_id in self.controlled_actor_ids:
                encoded = action.get(actor_id)
                if isinstance(encoded, list):
                    action_indices[actor_id] = self._action_index_from_one_hot(encoded)
                elif isinstance(encoded, int):
                    action_indices[actor_id] = int(encoded)
                else:
                    action_indices[actor_id] = ACTION_STOP_MOVE
            return action_indices

        player_action_index = self._action_index_from_one_hot(action)
        return {
            actor_id: (player_action_index if actor_id == self.player.team else ACTION_STOP_MOVE)
            for actor_id in self.controlled_actor_ids
        }

    @staticmethod
    def _move_delta_for_action(action_index: int) -> Vec2 | None:
        if action_index == ACTION_MOVE_UP:
            return Vec2(0, -PLAYER_MOVE_SPEED)
        if action_index == ACTION_MOVE_DOWN:
            return Vec2(0, PLAYER_MOVE_SPEED)
        if action_index == ACTION_MOVE_LEFT:
            return Vec2(-PLAYER_MOVE_SPEED, 0)
        if action_index == ACTION_MOVE_RIGHT:
            return Vec2(PLAYER_MOVE_SPEED, 0)
        return None

    def compute_reward_components(
        self,
        *,
        bad_shots: int,
        controlled_hits: int,
        controlled_friendly_fire_hits: int,
        obstacle_events: int,
        terminal_outcome: str | None,
    ) -> dict[str, float]:
        components = {
            "result": 0.0,
            "hit": 0.0,
            "time": 0.0,
            "accuracy": 0.0,
            "safety": 0.0,
            "obstacle": 0.0,
        }
        if terminal_outcome is not None:
            if __debug__:
                assert terminal_outcome in RESULT_REWARD_BY_OUTCOME
            components["result"] = float(RESULT_REWARD_BY_OUTCOME[terminal_outcome])
        if controlled_hits > 0:
            components["hit"] += REWARD_HIT_ENEMY * controlled_hits
        components["time"] += PENALTY_TIME_STEP
        if bad_shots > 0:
            components["accuracy"] -= PENALTY_BAD_SHOT * bad_shots
        if controlled_friendly_fire_hits > 0:
            components["safety"] -= PENALTY_FRIENDLY_FIRE * controlled_friendly_fire_hits
        if obstacle_events > 0:
            components["obstacle"] -= PENALTY_BLOCKED_MOVE * obstacle_events

        if __debug__:
            assert set(components.keys()) == REWARD_COMPONENT_KEY_SET
            assert tuple(components.keys()) == REWARD_COMPONENT_KEY_ORDER

        return components

    def play_step(self, action: dict[str, list[int]] | list[int]):
        self.frame_count += 1
        self.poll_events()

        action_indices = self._normalize_controlled_actions(action)
        bad_shots = 0
        obstacle_events = 0
        for actor_id, action_index in action_indices.items():
            actor = self.players_by_id.get(actor_id)
            if actor is None or not actor.is_alive:
                continue
            action_value = int(action_index)
            observation_state = self._observation_state_for(actor)
            move_delta = self._move_delta_for_action(action_value)
            if move_delta is not None and self._would_collide(actor, move_delta):
                if observation_state.last_blocked_move_action_idx != action_value:
                    obstacle_events += 1
                    observation_state.last_blocked_move_action_idx = action_value
            else:
                observation_state.last_blocked_move_action_idx = None
            los_clear_path = self.has_line_of_sight(actor=actor) if action_value == ACTION_SHOOT else False
            enemy_in_los = 1.0 if los_clear_path else 0.0
            if action_value == ACTION_SHOOT and enemy_in_los == 0.0:
                bad_shots += 1
            self.apply_actor_action(actor, action_index)

        self._step_scripted_players()
        projectile_events = self._step_projectiles()

        self._tick_players()

        controlled_hits = int(projectile_events["controlled_hits"])
        controlled_friendly_fire_hits = int(projectile_events["controlled_friendly_fire"])

        controlled_alive = self.player.is_alive
        just_lost = (not controlled_alive) and (not self.player_loss_recorded)

        player_won = self._player_win_condition(controlled_alive)
        timeout_reached = self.frame_count >= MAX_EPISODE_STEPS
        if player_won:
            terminal_outcome = "win"
        elif just_lost:
            terminal_outcome = "lose"
        elif timeout_reached:
            terminal_outcome = "timeout"
        else:
            terminal_outcome = None
        if __debug__:
            assert terminal_outcome in {None, "win", "lose", "timeout"}
        done = terminal_outcome is not None

        reward_breakdown = self.compute_reward_components(
            bad_shots=bad_shots,
            controlled_hits=controlled_hits,
            controlled_friendly_fire_hits=controlled_friendly_fire_hits,
            obstacle_events=obstacle_events,
            terminal_outcome=terminal_outcome,
        )
        reward = sum(reward_breakdown.values())

        if terminal_outcome == "lose":
            self.player_loss_recorded = True

        if done:
            if terminal_outcome == "lose":
                killer_id = projectile_events.get("player_killed_by")
                if isinstance(killer_id, str) and killer_id in self.scores:
                    self._increment_score(killer_id)
            elif terminal_outcome == "win":
                self._increment_score(self.player.team)

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        return reward, done, reward_breakdown

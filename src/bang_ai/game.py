"""Bang core gameplay, rendering, and game modes."""

from __future__ import annotations

import math
import random
from typing import Callable, TypeVar

import arcade

from bang_ai.assets import resolve_font_path
from bang_ai.config import (
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_SHOOT,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_WAIT,
    AIM_TOLERANCE_DEGREES,
    BB_HEIGHT,
    CELL_INSET,
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
    NUM_ACTIONS,
    OBSTACLE_START_ATTEMPTS,
    PENALTY_BAD_SHOT,
    PENALTY_BLOCKED_MOVE,
    PENALTY_LOSE,
    PENALTY_TIME_STEP,
    PLAYER_MOVE_SPEED,
    PLAYER_ROTATION_DEGREES,
    PLAYER_SPAWN_X_RATIO,
    PROJECTILE_DISTANCE_MISSING,
    PROJECTILE_HITBOX_SIZE,
    PROJECTILE_SPEED,
    PROJECTILE_TRAJECTORY_DOT_THRESHOLD,
    REWARD_HIT_ENEMY,
    REWARD_WIN,
    SAFE_RADIUS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SHOOT_COOLDOWN_FRAMES,
    SPAWN_Y_OFFSET,
    STARTING_LEVEL,
    TILE_SIZE,
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

T = TypeVar("T")


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


class Actor:
    """A movable actor that can rotate and shoot projectiles."""

    def __init__(self, position: Vec2, angle: float, team: str = "player") -> None:
        self.position = position
        self.angle = angle
        self.cooldown_frames = 0
        self.is_alive = True
        self.team = team

    def step_movement(self, move_forward: bool, move_backward: bool, rotate_left: bool, rotate_right: bool) -> Vec2:
        if rotate_left:
            self.angle = (self.angle + PLAYER_ROTATION_DEGREES) % 360
        if rotate_right:
            self.angle = (self.angle - PLAYER_ROTATION_DEGREES) % 360

        direction = heading_to_vector(self.angle)
        movement = Vec2(0, 0)
        if move_forward:
            movement += direction * PLAYER_MOVE_SPEED
        if move_backward:
            movement -= direction * PLAYER_MOVE_SPEED
        return movement

    def shoot(self):
        if self.cooldown_frames > 0:
            return None

        direction = heading_to_vector(self.angle)
        self.cooldown_frames = SHOOT_COOLDOWN_FRAMES
        return {
            "pos": self.position + direction * 20,
            "velocity": direction * PROJECTILE_SPEED,
            "owner": self.team,
        }

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

    def _draw_status_bar(self) -> None:
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

    def _draw_actor(self, actor: Actor, fill_color, outline_color) -> None:
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


class BaseGame:
    """Two-player top-down arena game logic."""

    def __init__(self, level: int = 1, show_game: bool = True):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.show_game = bool(show_game)
        self.frame_clock = ArcadeFrameClock()
        self.renderer = Renderer(
            game=self,
            width=self.width,
            height=self.height,
            title=WINDOW_TITLE,
            enabled=self.show_game,
        )
        self.window_controller = self.renderer.window_controller
        self.window = self.renderer.window

        self.p1_score = 0
        self.p2_score = 0
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
        settings = LEVEL_SETTINGS[level]

        self.num_obstacles = settings["num_obstacles"]
        self.enemy_move_probability = settings["enemy_move_probability"]
        self.enemy_shot_error_choices = settings["enemy_shot_error_choices"]
        self.enemy_shoot_probability = settings["enemy_shoot_probability"]

    def reset(self) -> None:
        player_pos = self._sample_spawn_position(PLAYER_SPAWN_X_RATIO)
        enemy_pos = self._sample_spawn_position(ENEMY_SPAWN_X_RATIO)

        self.player = Actor(player_pos, angle=0, team="player")
        self.enemy = Actor(enemy_pos, angle=180, team="enemy")

        self.obstacles: list[Vec2] = []
        self.projectiles: list[dict[str, object]] = []
        self.frame_count = 0
        self.last_action_index = 0
        self.frames_since_last_shot = SHOOT_COOLDOWN_FRAMES
        self.previous_enemy_distance = None
        self.previous_enemy_relative_angle = None
        self.previous_projectile_distance = None
        self.last_seen_enemy_frame = -EVENT_TIMER_NORMALIZATION_FRAMES
        self.last_projectile_seen_frame = -EVENT_TIMER_NORMALIZATION_FRAMES
        self.projectile_in_perception_last_frame = False
        self.last_perception_update_frame = -1
        self.enemy_blocked_move_attempts = 0
        self.enemy_escape_angle = None
        self.enemy_escape_frames_remaining = 0
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

    def _sample_spawn_position(self, x_ratio: float) -> Vec2:
        min_y, max_y = self._spawn_y_bounds()
        return Vec2(self.width * x_ratio, random.uniform(min_y, max_y))

    def apply_player_action(self, action_index: int) -> None:
        self.last_action_index = action_index
        move_forward = action_index == ACTION_MOVE_FORWARD
        move_backward = action_index == ACTION_MOVE_BACKWARD
        rotate_left = action_index == ACTION_TURN_LEFT
        rotate_right = action_index == ACTION_TURN_RIGHT

        movement = self.player.step_movement(move_forward, move_backward, rotate_left, rotate_right)
        self._update_actor_position(self.player, movement)

        if action_index == ACTION_SHOOT:
            projectile = self.player.shoot()
            if projectile:
                self.projectiles.append(projectile)
            self.frames_since_last_shot = 0
        else:
            self.frames_since_last_shot += 1

    def _update_actor_position(self, actor: Actor, movement: Vec2) -> None:
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

        other = self.enemy if actor is self.player else self.player
        if other.is_alive:
            other_rect = rect_from_center(other.position, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                return

        actor.position = new_position

    def _would_collide(self, actor: Actor, movement: Vec2) -> bool:
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

        other = self.enemy if actor is self.player else self.player
        if other.is_alive:
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

    def _update_enemy_seen_timer(self, enemy_in_los: bool) -> float:
        if enemy_in_los:
            self.last_seen_enemy_frame = self.frame_count
        return self._normalize_elapsed_frames(self.frame_count - self.last_seen_enemy_frame)

    def _update_projectile_seen_timer(self, projectile_in_perception: bool) -> float:
        if self.last_perception_update_frame != self.frame_count:
            projectile_entered = projectile_in_perception and not self.projectile_in_perception_last_frame
            if projectile_entered:
                self.last_projectile_seen_frame = self.frame_count
            self.projectile_in_perception_last_frame = projectile_in_perception
            self.last_perception_update_frame = self.frame_count
        return self._normalize_elapsed_frames(self.frame_count - self.last_projectile_seen_frame)

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
        if tile.distance(self.player.position) < SAFE_RADIUS or tile.distance(self.enemy.position) < SAFE_RADIUS:
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

    def _move_enemy_in_direction(self, angle_degrees: float) -> bool:
        previous_position = self.enemy.position
        movement = self._move_vector_for_angle(angle_degrees)
        self._update_actor_position(self.enemy, movement)
        return length_squared(self.enemy.position - previous_position) > 0

    def _choose_enemy_escape_angle(self, reference_angle: float) -> float | None:
        available_angles = []
        for offset in ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES:
            candidate_angle = (reference_angle + offset) % 360
            candidate_move = self._move_vector_for_angle(candidate_angle)
            if not self._would_collide(self.enemy, candidate_move):
                available_angles.append(candidate_angle)
        if not available_angles:
            return None
        return random.choice(available_angles)

    def _start_enemy_escape(self, reference_angle: float) -> None:
        escape_angle = self._choose_enemy_escape_angle(reference_angle)
        if escape_angle is None:
            return
        self.enemy_escape_angle = escape_angle
        self.enemy_escape_frames_remaining = ENEMY_ESCAPE_FOLLOW_FRAMES
        self.enemy_blocked_move_attempts = 0

    def _step_enemy_navigation(self, aim_angle: float) -> None:
        if self.enemy_escape_frames_remaining > 0 and self.enemy_escape_angle is not None:
            moved = self._move_enemy_in_direction(self.enemy_escape_angle)
            self.enemy_escape_frames_remaining -= 1
            if moved:
                self.enemy_blocked_move_attempts = 0
            else:
                replacement_angle = self._choose_enemy_escape_angle(aim_angle)
                if replacement_angle is not None:
                    self.enemy_escape_angle = replacement_angle
            if self.enemy_escape_frames_remaining <= 0:
                self.enemy_escape_angle = None
            return

        moved = self._move_enemy_in_direction(aim_angle)
        if moved:
            self.enemy_blocked_move_attempts = 0
            return

        self.enemy_blocked_move_attempts += 1
        if self.enemy_blocked_move_attempts >= ENEMY_STUCK_MOVE_ATTEMPTS:
            self._start_enemy_escape(aim_angle)

    def _step_enemy(self) -> None:
        if not self.enemy.is_alive:
            return

        to_player = self.player.position - self.enemy.position
        angle_to_player = math.degrees(math.atan2(to_player.y, to_player.x)) % 360
        aim_error = random.choice(self.enemy_shot_error_choices)
        aim_angle = (angle_to_player + aim_error) % 360
        self.enemy.angle = aim_angle

        if random.random() < self.enemy_move_probability:
            self._step_enemy_navigation(aim_angle)

        if random.random() < self.enemy_shoot_probability:
            projectile = self.enemy.shoot()
            if projectile:
                self.projectiles.append(projectile)

    def _step_projectiles(self):
        events = {"enemy_hit": False, "player_hit": False}
        next_projectiles = []

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

            target = self.player if projectile["owner"] == "enemy" else self.enemy
            if target.is_alive:
                target_rect = rect_from_center(target.position, TILE_SIZE)
                if projectile_rect.colliderect(target_rect):
                    target.is_alive = False
                    events["player_hit" if target is self.player else "enemy_hit"] = True
                    continue

            next_projectiles.append(projectile)

        self.projectiles = next_projectiles
        return events

    def _distance_to_closest_enemy_projectile(self):
        enemy_projectiles = [p for p in self.projectiles if p["owner"] == "enemy"]
        if not enemy_projectiles:
            return None
        return min(enemy_projectiles, key=lambda p: self.player.position.distance(p["pos"]))

    def is_player_in_projectile_trajectory(self) -> bool:
        for projectile in self.projectiles:
            if projectile["owner"] != "enemy":
                continue
            to_player = self.player.position - projectile["pos"]
            if length_squared(to_player) == 0:
                return True
            projectile_dir = projectile["velocity"].normalize()
            if projectile_dir.dot(to_player.normalize()) > PROJECTILE_TRAJECTORY_DOT_THRESHOLD:
                return True
        return False

    def has_line_of_sight(self) -> bool:
        to_enemy = self.enemy.position - self.player.position
        if length_squared(to_enemy) == 0:
            return True
        enemy_angle = math.degrees(math.atan2(to_enemy.y, to_enemy.x))
        relative = normalize_angle_degrees(enemy_angle - self.player.angle)
        return abs(relative) <= AIM_TOLERANCE_DEGREES and not square_obstacle_between_points(
            point_a=self.player.position,
            point_b=self.enemy.position,
            obstacles=self.obstacles,
            tile_size=TILE_SIZE,
        )

    def get_state_vector(self) -> list[float]:
        to_enemy = self.enemy.position - self.player.position
        enemy_distance = self.player.position.distance(self.enemy.position) / max(self.width, self.height)
        enemy_angle = math.atan2(to_enemy.y, to_enemy.x)
        enemy_relative_angle = math.radians(normalize_angle_degrees(math.degrees(enemy_angle) - self.player.angle))
        enemy_relative_sin = math.sin(enemy_relative_angle)
        enemy_relative_cos = math.cos(enemy_relative_angle)
        enemy_in_los = self.has_line_of_sight()
        time_since_last_seen_enemy = self._update_enemy_seen_timer(enemy_in_los)

        if self.previous_enemy_distance is None:
            delta_enemy_distance = 0.0
        else:
            delta_enemy_distance = self._clip(enemy_distance - self.previous_enemy_distance)
        self.previous_enemy_distance = enemy_distance

        if self.previous_enemy_relative_angle is None:
            delta_enemy_relative_angle = 0.0
        else:
            delta_enemy_relative_angle = math.atan2(
                math.sin(enemy_relative_angle - self.previous_enemy_relative_angle),
                math.cos(enemy_relative_angle - self.previous_enemy_relative_angle),
            )
            delta_enemy_relative_angle = self._clip(delta_enemy_relative_angle / math.pi)
        self.previous_enemy_relative_angle = enemy_relative_angle

        closest_projectile = self._distance_to_closest_enemy_projectile()
        projectile_in_perception = closest_projectile is not None
        if closest_projectile:
            projectile_distance = self.player.position.distance(closest_projectile["pos"]) / max(self.width, self.height)
            to_projectile = closest_projectile["pos"] - self.player.position
            projectile_angle = math.atan2(to_projectile.y, to_projectile.x)
            projectile_relative_angle = math.radians(
                normalize_angle_degrees(math.degrees(projectile_angle) - self.player.angle)
            )
            projectile_relative_sin = math.sin(projectile_relative_angle)
            projectile_relative_cos = math.cos(projectile_relative_angle)
        else:
            projectile_distance = PROJECTILE_DISTANCE_MISSING
            projectile_relative_sin = 0.0
            projectile_relative_cos = 1.0

        if self.previous_projectile_distance is None or projectile_distance == PROJECTILE_DISTANCE_MISSING:
            delta_projectile_distance = 0.0
        else:
            delta_projectile_distance = self._clip(projectile_distance - self.previous_projectile_distance)
        self.previous_projectile_distance = projectile_distance

        time_since_last_projectile_seen = self._update_projectile_seen_timer(projectile_in_perception)

        forward_dir = rotate_degrees(Vec2(1, 0), self.player.angle).normalize()
        left_dir = rotate_degrees(forward_dir, -90)
        right_dir = rotate_degrees(forward_dir, 90)

        forward_blocked = 1.0 if self._would_collide(self.player, forward_dir * PLAYER_MOVE_SPEED) else 0.0
        left_blocked = 1.0 if self._would_collide(self.player, left_dir * PLAYER_MOVE_SPEED) else 0.0
        right_blocked = 1.0 if self._would_collide(self.player, right_dir * PLAYER_MOVE_SPEED) else 0.0

        last_action = float(self.last_action_index) / max(1, NUM_ACTIONS - 1)
        time_since_last_shot = min(1.0, self.frames_since_last_shot / max(1, SHOOT_COOLDOWN_FRAMES))

        feature_values = {
            "enemy_distance": enemy_distance,
            "enemy_in_los": 1.0 if enemy_in_los else 0.0,
            "enemy_relative_angle_sin": enemy_relative_sin,
            "enemy_relative_angle_cos": enemy_relative_cos,
            "delta_enemy_distance": delta_enemy_distance,
            "delta_enemy_relative_angle": delta_enemy_relative_angle,
            "nearest_projectile_distance": projectile_distance,
            "nearest_projectile_relative_angle_sin": projectile_relative_sin,
            "nearest_projectile_relative_angle_cos": projectile_relative_cos,
            "delta_projectile_distance": delta_projectile_distance,
            "in_projectile_trajectory": 1.0 if self.is_player_in_projectile_trajectory() else 0.0,
            "forward_blocked": forward_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "last_action_index": last_action,
            "time_since_last_shot": time_since_last_shot,
            "time_since_last_seen_enemy": time_since_last_seen_enemy,
            "time_since_last_projectile_seen": time_since_last_projectile_seen,
        }
        return self._build_state_vector_from_features(feature_values)


class HumanGame(BaseGame):
    """Human-play mode."""

    def __init__(self, show_game: bool = True):
        level = max(MIN_LEVEL, min(STARTING_LEVEL, MAX_LEVEL))
        super().__init__(level=level, show_game=show_game)

    def play_step(self) -> None:
        self.frame_count += 1
        self.poll_events()

        if self.window_controller.is_key_down(arcade.key.W):
            action = ACTION_MOVE_FORWARD
        elif self.window_controller.is_key_down(arcade.key.S):
            action = ACTION_MOVE_BACKWARD
        elif self.window_controller.is_key_down(arcade.key.A):
            action = ACTION_TURN_LEFT
        elif self.window_controller.is_key_down(arcade.key.D):
            action = ACTION_TURN_RIGHT
        elif self.window_controller.is_key_down(arcade.key.SPACE):
            action = ACTION_SHOOT
        else:
            action = ACTION_WAIT

        self.apply_player_action(action)
        self._step_enemy()
        self._step_projectiles()

        self.player.tick()
        self.enemy.tick()

        if not self.enemy.is_alive:
            self.p1_score += 1
            self.reset()
        elif not self.player.is_alive:
            self.p2_score += 1
            self.reset()

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else 0)


class TrainingGame(BaseGame):
    """Environment used by DQN training."""

    def play_step(self, action: list[int]):
        self.frame_count += 1
        action_index = action.index(1) if 1 in action else 0
        self.poll_events()

        previous_position = self.player.position
        self.apply_player_action(action_index)
        blocked_move = length_squared(self.player.position - previous_position) == 0
        self._step_enemy()
        projectile_events = self._step_projectiles()

        self.player.tick()
        self.enemy.tick()

        reward = PENALTY_TIME_STEP
        reward_breakdown = {
            "time_step": PENALTY_TIME_STEP,
            "bad_shot": 0.0,
            "blocked_move": 0.0,
            "hit_enemy": 0.0,
            "win": 0.0,
            "lose": 0.0,
        }

        if action_index == ACTION_SHOOT and not self.has_line_of_sight():
            reward += PENALTY_BAD_SHOT
            reward_breakdown["bad_shot"] = PENALTY_BAD_SHOT
        elif action_index in (ACTION_MOVE_FORWARD, ACTION_MOVE_BACKWARD) and blocked_move:
            reward += PENALTY_BLOCKED_MOVE
            reward_breakdown["blocked_move"] = PENALTY_BLOCKED_MOVE

        if projectile_events["enemy_hit"]:
            reward += REWARD_HIT_ENEMY
            reward_breakdown["hit_enemy"] = REWARD_HIT_ENEMY

        done = False
        if not self.enemy.is_alive:
            reward += REWARD_WIN
            reward_breakdown["win"] = REWARD_WIN
            self.p1_score += 1
            done = True
        elif not self.player.is_alive:
            reward += PENALTY_LOSE
            reward_breakdown["lose"] = PENALTY_LOSE
            self.p2_score += 1
            done = True
        elif self.frame_count >= MAX_EPISODE_STEPS:
            done = True

        self.draw_frame()
        self.frame_clock.tick(FPS if self.show_game else TRAINING_FPS)

        return reward, done, reward_breakdown

"""Core arena simulation used by training and human play modes."""

from __future__ import annotations

import math
import random

from bang_ai.boards import spawn_connected_random_walk_shapes
from bang_ai.config import (
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_SHOOT,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    AIM_TOLERANCE_DEGREES,
    BB_HEIGHT,
    ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES,
    ENEMY_ESCAPE_FOLLOW_FRAMES,
    ENEMY_SPAWN_X_RATIO,
    ENEMY_STUCK_MOVE_ATTEMPTS,
    EVENT_TIMER_NORMALIZATION_FRAMES,
    INPUT_FEATURE_NAMES,
    LEVEL_SETTINGS,
    MAX_LEVEL,
    MAX_OBSTACLE_SECTIONS,
    MIN_LEVEL,
    MIN_OBSTACLE_SECTIONS,
    NUM_ACTIONS,
    OBSTACLE_START_ATTEMPTS,
    PLAYER_MOVE_SPEED,
    PLAYER_SPAWN_X_RATIO,
    PROJECTILE_DISTANCE_MISSING,
    PROJECTILE_HITBOX_SIZE,
    PROJECTILE_TRAJECTORY_DOT_THRESHOLD,
    SAFE_RADIUS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SHOOT_COOLDOWN_FRAMES,
    SHOW_GAME,
    SPAWN_Y_OFFSET,
    TILE_SIZE,
    WINDOW_TITLE,
)
from bang_ai.core.actor import Actor
from bang_ai.runtime import (
    ArcadeFrameClock,
    Vec2,
    collides_with_square_arena,
    length_squared,
    normalize_angle_degrees,
    rect_from_center,
    rotate_degrees,
    square_obstacle_between_points,
)
from bang_ai.ui.renderer import Renderer


class BaseGame:
    """Two-player top-down arena game logic."""

    def __init__(self, level: int = 1):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT
        self.frame_clock = ArcadeFrameClock()
        self.renderer = Renderer(
            game=self,
            width=self.width,
            height=self.height,
            title=WINDOW_TITLE,
            enabled=SHOW_GAME,
        )
        self.window_controller = self.renderer.window_controller
        self.window = self.renderer.window

        self.p1_score = 0
        self.p2_score = 0
        self.level = level
        self.configure_level()
        self.reset()

    def close(self):
        self.renderer.close()

    def poll_events(self):
        self.renderer.poll_events()

    def draw_frame(self):
        self.renderer.draw_frame()

    def configure_level(self):
        """Configure enemy complexity and obstacle density."""
        level = max(MIN_LEVEL, min(self.level, MAX_LEVEL))
        settings = LEVEL_SETTINGS[level]

        self.num_obstacles = settings["num_obstacles"]
        self.enemy_move_probability = settings["enemy_move_probability"]
        self.enemy_shot_error_choices = settings["enemy_shot_error_choices"]
        self.enemy_shoot_probability = settings["enemy_shoot_probability"]

    def reset(self):
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

    def apply_player_action(self, action_index: int):
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

    def _update_actor_position(self, actor: Actor, movement: Vec2):
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
            projectile_entered_perception = projectile_in_perception and not self.projectile_in_perception_last_frame
            if projectile_entered_perception:
                self.last_projectile_seen_frame = self.frame_count
            self.projectile_in_perception_last_frame = projectile_in_perception
            self.last_perception_update_frame = self.frame_count
        return self._normalize_elapsed_frames(self.frame_count - self.last_projectile_seen_frame)

    @staticmethod
    def _build_state_vector_from_features(feature_values: dict[str, float]) -> list[float]:
        return [float(feature_values[name]) for name in INPUT_FEATURE_NAMES]

    def _place_obstacles(self):
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

    def _is_valid_obstacle_tile(self, tile: Vec2, pending_tiles):
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

    def _start_enemy_escape(self, reference_angle: float):
        escape_angle = self._choose_enemy_escape_angle(reference_angle)
        if escape_angle is None:
            return
        self.enemy_escape_angle = escape_angle
        self.enemy_escape_frames_remaining = ENEMY_ESCAPE_FOLLOW_FRAMES
        self.enemy_blocked_move_attempts = 0

    def _step_enemy_navigation(self, aim_angle: float):
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

    def _step_enemy(self):
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
        """Advance projectiles and return event flags used by reward logic."""
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

    def get_state_vector(self):
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
            delta_enemy_distance = enemy_distance - self.previous_enemy_distance
            delta_enemy_distance = self._clip(delta_enemy_distance)
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
            delta_projectile_distance = projectile_distance - self.previous_projectile_distance
            delta_projectile_distance = self._clip(delta_projectile_distance)
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

    def play_step(self, action):
        raise NotImplementedError

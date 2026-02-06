"""Core arena simulation used by training and human play modes."""

import math
import random

import pygame

from constants import *
from game_agent import Actor
from ui import Renderer
from utils import is_collision, is_obstacle_between, normalize_angle_degrees


class BaseGame:
    """Two-player top-down arena game logic."""

    def __init__(self, level: int = 1):
        self.width = SCREEN_WIDTH
        self.height = SCREEN_HEIGHT

        pygame.init()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((self.width, self.height)) if SHOW_GAME else pygame.Surface((self.width, self.height))
        if SHOW_GAME:
            pygame.display.set_caption("Bang")

        self.ui = Renderer(self.display, self)

        self.p1_score = 0
        self.p2_score = 0
        self.level = level
        self.configure_level()
        self.reset()

    def configure_level(self):
        """Configure enemy complexity and obstacle density."""
        level = max(MIN_LEVEL, min(self.level, MAX_LEVEL))
        settings = LEVEL_SETTINGS.get(level) or LEVEL_SETTINGS.get(MAX_LEVEL, {})

        self.num_obstacles = settings.get("num_obstacles", DEFAULT_OBSTACLES) or DEFAULT_OBSTACLES
        self.enemy_can_move = settings.get("enemy_can_move", True)
        self.enemy_shot_error_choices = settings.get("enemy_shot_error_choices", [0])
        self.enemy_move_probability = ENEMY_MOVE_PROBABILITY_SCALE * level

    def reset(self):
        player_offset = random.choice([0, SPAWN_Y_OFFSET, -SPAWN_Y_OFFSET])
        enemy_offset = random.choice([0, SPAWN_Y_OFFSET, -SPAWN_Y_OFFSET])

        player_pos = pygame.Vector2(self.width * PLAYER_SPAWN_X_RATIO, (self.height / 2 - BOTTOM_BAR_HEIGHT // 2) + player_offset)
        enemy_pos = pygame.Vector2(self.width * ENEMY_SPAWN_X_RATIO, (self.height / 2 - BOTTOM_BAR_HEIGHT // 2) + enemy_offset)

        self.player = Actor(player_pos, angle=0, team="player")
        self.enemy = Actor(enemy_pos, angle=180, team="enemy")

        self.obstacles = []
        self.projectiles = []
        self.frame_count = 0
        self.last_action_index = 0
        self.frames_since_last_shot = SHOOT_COOLDOWN_FRAMES
        self.previous_enemy_distance = None
        self.previous_enemy_relative_angle = None
        self.previous_projectile_distance = None
        self._place_obstacles()

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

    def _update_actor_position(self, actor, movement: pygame.Vector2):
        new_position = actor.position + movement
        actor_rect = pygame.Rect(new_position.x - TILE_SIZE // 2, new_position.y - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
        if is_collision(self, actor_rect):
            return

        other = self.enemy if actor is self.player else self.player
        if other.is_alive:
            other_rect = pygame.Rect(other.position.x - TILE_SIZE // 2, other.position.y - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                return

        actor.position = new_position

    def _would_collide(self, actor, movement: pygame.Vector2) -> bool:
        new_position = actor.position + movement
        actor_rect = pygame.Rect(new_position.x - TILE_SIZE // 2, new_position.y - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
        if is_collision(self, actor_rect):
            return True

        other = self.enemy if actor is self.player else self.player
        if other.is_alive:
            other_rect = pygame.Rect(other.position.x - TILE_SIZE // 2, other.position.y - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
            if actor_rect.colliderect(other_rect):
                return True
        return False

    def _place_obstacles(self):
        self.obstacles = []
        for _ in range(self.num_obstacles):
            start = self._sample_valid_obstacle_start()
            if start is None:
                continue

            shape = [start]
            current = start
            directions = [(-TILE_SIZE, 0), (TILE_SIZE, 0), (0, -TILE_SIZE), (0, TILE_SIZE)]
            for _ in range(random.randint(MIN_OBSTACLE_SECTIONS, MAX_OBSTACLE_SECTIONS) - 1):
                random.shuffle(directions)
                extended = False
                for dx, dy in directions:
                    candidate = pygame.Vector2(current.x + dx, current.y + dy)
                    if self._is_valid_obstacle_tile(candidate, shape):
                        shape.append(candidate)
                        current = candidate
                        extended = True
                        break
                if not extended:
                    break
            self.obstacles.extend(shape)

    def _sample_valid_obstacle_start(self):
        for _ in range(OBSTACLE_START_ATTEMPTS):
            x = random.randint(0, (self.width - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            y = random.randint(0, (self.height - BOTTOM_BAR_HEIGHT - TILE_SIZE) // TILE_SIZE) * TILE_SIZE
            point = pygame.Vector2(x, y)
            if self._is_valid_obstacle_tile(point, []):
                return point
        return None

    def _is_valid_obstacle_tile(self, tile: pygame.Vector2, pending_tiles):
        if not (0 <= tile.x < self.width and 0 <= tile.y < self.height - BOTTOM_BAR_HEIGHT):
            return False
        if any(tile == existing for existing in self.obstacles) or any(tile == existing for existing in pending_tiles):
            return False
        if tile.distance_to(self.player.position) < SAFE_RADIUS or tile.distance_to(self.enemy.position) < SAFE_RADIUS:
            return False
        return True

    def _step_enemy(self):
        if not self.enemy.is_alive:
            return

        to_player = self.player.position - self.enemy.position
        angle_to_player = math.degrees(math.atan2(to_player.y, to_player.x)) % 360
        aim_error = random.choice(self.enemy_shot_error_choices)
        self.enemy.angle = (angle_to_player + aim_error) % 360

        if self.enemy_can_move and random.random() < self.enemy_move_probability:
            movement = self.enemy.step_movement(True, False, False, False)
            self._update_actor_position(self.enemy, movement)

        if random.random() < ENEMY_SHOOT_PROBABILITY:
            projectile = self.enemy.shoot()
            if projectile:
                self.projectiles.append(projectile)

    def _step_projectiles(self):
        """Advance projectiles and return event flags used by reward logic."""
        events = {"enemy_hit": False, "player_hit": False}
        next_projectiles = []

        for projectile in self.projectiles:
            projectile["pos"] += projectile["velocity"]
            projectile_rect = pygame.Rect(
                projectile["pos"].x - PROJECTILE_HITBOX_HALF,
                projectile["pos"].y - PROJECTILE_HITBOX_HALF,
                PROJECTILE_HITBOX_SIZE,
                PROJECTILE_HITBOX_SIZE,
            )
            if is_collision(self, projectile_rect):
                continue

            target = self.player if projectile["owner"] == "enemy" else self.enemy
            if target.is_alive:
                target_rect = pygame.Rect(target.position.x - TILE_SIZE // 2, target.position.y - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
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
        return min(enemy_projectiles, key=lambda p: self.player.position.distance_to(p["pos"]))

    def is_player_in_projectile_trajectory(self) -> bool:
        for projectile in self.projectiles:
            if projectile["owner"] != "enemy":
                continue
            to_player = self.player.position - projectile["pos"]
            if to_player.length_squared() == 0:
                return True
            projectile_dir = projectile["velocity"].normalize()
            if projectile_dir.dot(to_player.normalize()) > PROJECTILE_TRAJECTORY_DOT_THRESHOLD:
                return True
        return False

    def has_line_of_sight(self) -> bool:
        to_enemy = self.enemy.position - self.player.position
        if to_enemy.length_squared() == 0:
            return True
        enemy_angle = math.degrees(math.atan2(to_enemy.y, to_enemy.x))
        relative = normalize_angle_degrees(enemy_angle - self.player.angle)
        return abs(relative) <= AIM_TOLERANCE_DEGREES and not is_obstacle_between(self, self.player.position, self.enemy.position)

    def get_state_vector(self):
        to_enemy = self.enemy.position - self.player.position
        enemy_distance = self.player.position.distance_to(self.enemy.position) / max(self.width, self.height)
        enemy_angle = math.atan2(to_enemy.y, to_enemy.x)
        enemy_relative_angle = math.radians(normalize_angle_degrees(math.degrees(enemy_angle) - self.player.angle))
        enemy_relative_sin = math.sin(enemy_relative_angle)
        enemy_relative_cos = math.cos(enemy_relative_angle)

        if self.previous_enemy_distance is None:
            delta_enemy_distance = 0.0
        else:
            delta_enemy_distance = enemy_distance - self.previous_enemy_distance
        self.previous_enemy_distance = enemy_distance

        if self.previous_enemy_relative_angle is None:
            delta_enemy_relative_angle = 0.0
        else:
            delta_enemy_relative_angle = math.atan2(
                math.sin(enemy_relative_angle - self.previous_enemy_relative_angle),
                math.cos(enemy_relative_angle - self.previous_enemy_relative_angle),
            )
        self.previous_enemy_relative_angle = enemy_relative_angle

        closest_projectile = self._distance_to_closest_enemy_projectile()
        if closest_projectile:
            projectile_distance = self.player.position.distance_to(closest_projectile["pos"]) / max(self.width, self.height)
            to_projectile = closest_projectile["pos"] - self.player.position
            projectile_angle = math.atan2(to_projectile.y, to_projectile.x)
            projectile_relative_angle = math.radians(normalize_angle_degrees(math.degrees(projectile_angle) - self.player.angle))
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
        self.previous_projectile_distance = projectile_distance

        forward_dir = pygame.Vector2(1, 0).rotate(self.player.angle).normalize()
        left_dir = forward_dir.rotate(-90)
        right_dir = forward_dir.rotate(90)

        forward_blocked = 1.0 if self._would_collide(self.player, forward_dir * PLAYER_MOVE_SPEED) else 0.0
        left_blocked = 1.0 if self._would_collide(self.player, left_dir * PLAYER_MOVE_SPEED) else 0.0
        right_blocked = 1.0 if self._would_collide(self.player, right_dir * PLAYER_MOVE_SPEED) else 0.0

        last_action = float(self.last_action_index) / max(1, NUM_ACTIONS - 1)
        time_since_last_shot = min(1.0, self.frames_since_last_shot / max(1, SHOOT_COOLDOWN_FRAMES))

        state = [
            enemy_distance,
            enemy_relative_sin,
            enemy_relative_cos,
            delta_enemy_distance,
            delta_enemy_relative_angle,
            1.0 if self.has_line_of_sight() else 0.0,
            projectile_distance,
            projectile_relative_sin,
            projectile_relative_cos,
            delta_projectile_distance,
            1.0 if self.is_player_in_projectile_trajectory() else 0.0,
            forward_blocked,
            left_blocked,
            right_blocked,
            last_action,
            time_since_last_shot,
        ]
        return state

    def play_step(self, action):
        raise NotImplementedError

"""Shared helper functions."""

import math
import pygame
import torch

from constants import BOTTOM_BAR_HEIGHT, SHOW_GAME, TILE_SIZE, USE_GPU


def get_device() -> torch.device:
    """Return the torch device configured for the run."""
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def is_collision(game, rect: pygame.Rect) -> bool:
    """Check if rect collides against walls or obstacles."""
    if rect.left < 0 or rect.right > game.width or rect.top < 0 or rect.bottom > game.height - BOTTOM_BAR_HEIGHT:
        return True

    for obstacle in game.obstacles:
        obstacle_rect = pygame.Rect(obstacle.x, obstacle.y, TILE_SIZE, TILE_SIZE)
        if rect.colliderect(obstacle_rect):
            return True
    return False


def line_intersects_rect(p1, p2, rect: pygame.Rect) -> bool:
    """Return True when a line segment intersects a rect."""
    return bool(rect.clipline(p1, p2))


def is_obstacle_between(game, point_a, point_b) -> bool:
    """Return True when any obstacle blocks line of sight."""
    for obstacle in game.obstacles:
        obstacle_rect = pygame.Rect(obstacle.x, obstacle.y, TILE_SIZE, TILE_SIZE)
        if line_intersects_rect(point_a, point_b, obstacle_rect):
            return True
    return False


def normalize_angle_degrees(angle: float) -> float:
    """Normalize degrees into [-180, 180]."""
    return ((angle + 180.0) % 360.0) - 180.0


def heading_to_vector(angle: float) -> pygame.Vector2:
    """Convert heading degrees into a unit vector."""
    return pygame.Vector2(math.cos(math.radians(angle)), math.sin(math.radians(angle)))

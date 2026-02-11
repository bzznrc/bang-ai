"""Shared helper functions."""

import math
from collections import OrderedDict
import pygame
import torch

from constants import BOTTOM_BAR_HEIGHT, SHOW_GAME, TILE_SIZE, USE_GPU


def get_device() -> torch.device:
    """Return the torch device configured for the run."""
    if USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _format_context_value(value) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_context_key(key: str) -> str:
    return key.replace("_", " ").title()


def _format_mode_label(mode: str) -> str:
    words = mode.replace("-", " ").split()
    formatted = []
    for word in words:
        if word.lower() == "ai":
            formatted.append("AI")
        else:
            formatted.append(word.title())
    return " ".join(formatted)


def log_run_context(mode: str, context: dict):
    """Print a compact, consistent startup context line."""
    mode_label = _format_mode_label(mode)
    ordered_context = OrderedDict((key, value) for key, value in context.items() if value is not None)
    segments = [mode_label]
    segments.extend(
        f"{_format_context_key(key)}: {_format_context_value(value)}"
        for key, value in ordered_context.items()
    )
    print(" / ".join(segments))


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

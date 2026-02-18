"""Generic 2D geometry helpers for top-left game spaces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from pyglet.math import Vec2


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in top-left coordinate space."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    def colliderect(self, other: "Rect") -> bool:
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.bottom <= other.top
            or self.top >= other.bottom
        )


def heading_to_vector(angle_degrees: float) -> Vec2:
    radians = math.radians(angle_degrees)
    return Vec2(math.cos(radians), math.sin(radians))


def rotate_degrees(vector: Vec2, angle_degrees: float) -> Vec2:
    return vector.rotate(math.radians(angle_degrees))


def length_squared(vector: Vec2) -> float:
    return vector.dot(vector)


def rect_from_center(position: Vec2, size: int | float) -> Rect:
    half = float(size) / 2.0
    return Rect(position.x - half, position.y - half, float(size), float(size))


def normalize_angle_degrees(angle: float) -> float:
    return ((angle + 180.0) % 360.0) - 180.0


def _obstacle_xy(obstacle: Any) -> tuple[float, float]:
    if hasattr(obstacle, "x") and hasattr(obstacle, "y"):
        return float(obstacle.x), float(obstacle.y)
    if isinstance(obstacle, (tuple, list)) and len(obstacle) >= 2:
        return float(obstacle[0]), float(obstacle[1])
    raise TypeError(f"Unsupported obstacle type: {type(obstacle)!r}")


def collides_with_square_arena(
    rect: Rect,
    obstacles: Iterable[Any],
    tile_size: int,
    arena_width: int,
    arena_height: int,
    bottom_bar_height: int,
) -> bool:
    if (
        rect.left < 0
        or rect.right > arena_width
        or rect.top < 0
        or rect.bottom > arena_height - bottom_bar_height
    ):
        return True

    for obstacle in obstacles:
        x, y = _obstacle_xy(obstacle)
        obstacle_rect = Rect(x, y, tile_size, tile_size)
        if rect.colliderect(obstacle_rect):
            return True
    return False


def _line_intersects_rect(point_a: Vec2, point_b: Vec2, rect: Rect) -> bool:
    x0, y0 = point_a.x, point_a.y
    x1, y1 = point_b.x, point_b.y
    dx = x1 - x0
    dy = y1 - y0

    p = (-dx, dx, -dy, dy)
    q = (x0 - rect.left, rect.right - x0, y0 - rect.top, rect.bottom - y0)

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
            continue

        t = qi / pi
        if pi < 0:
            if t > u2:
                return False
            u1 = max(u1, t)
        else:
            if t < u1:
                return False
            u2 = min(u2, t)

    return True


def square_obstacle_between_points(
    point_a: Vec2,
    point_b: Vec2,
    obstacles: Iterable[Any],
    tile_size: int,
) -> bool:
    for obstacle in obstacles:
        x, y = _obstacle_xy(obstacle)
        obstacle_rect = Rect(x, y, tile_size, tile_size)
        if _line_intersects_rect(point_a, point_b, obstacle_rect):
            return True
    return False

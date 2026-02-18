"""Runtime helpers for Bang AI."""

from .helpers import configure_logging, get_torch_device, log_run_context

try:
    from .arcade_runtime import ArcadeFrameClock, ArcadeWindowController, TextCache, load_font_once
except ModuleNotFoundError:  # optional until runtime dependencies are installed
    pass

try:
    from .geometry import (
        Rect,
        Vec2,
        collides_with_square_arena,
        heading_to_vector,
        length_squared,
        normalize_angle_degrees,
        rect_from_center,
        rotate_degrees,
        square_obstacle_between_points,
    )
except ModuleNotFoundError:  # optional until runtime dependencies are installed
    pass

__all__ = [
    "ArcadeFrameClock",
    "ArcadeWindowController",
    "TextCache",
    "load_font_once",
    "Rect",
    "Vec2",
    "heading_to_vector",
    "rotate_degrees",
    "length_squared",
    "rect_from_center",
    "normalize_angle_degrees",
    "collides_with_square_arena",
    "square_obstacle_between_points",
    "configure_logging",
    "get_torch_device",
    "log_run_context",
]

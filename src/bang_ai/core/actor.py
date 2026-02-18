"""Entity model for player and enemy agents."""

from bang_ai.config import PLAYER_MOVE_SPEED, PLAYER_ROTATION_DEGREES, PROJECTILE_SPEED, SHOOT_COOLDOWN_FRAMES
from bang_ai.runtime import Vec2, heading_to_vector


class Actor:
    """A movable actor that can rotate and shoot projectiles."""

    def __init__(self, position: Vec2, angle: float, team: str = "player"):
        self.position = position
        self.angle = angle
        self.cooldown_frames = 0
        self.is_alive = True
        self.team = team

    def step_movement(self, move_forward: bool, move_backward: bool, rotate_left: bool, rotate_right: bool) -> Vec2:
        """Update heading and return a movement vector for this frame."""
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
        """Return projectile dict when cooldown is ready, else None."""
        if self.cooldown_frames > 0:
            return None

        direction = heading_to_vector(self.angle)
        self.cooldown_frames = SHOOT_COOLDOWN_FRAMES
        return {
            "pos": self.position + direction * 20,
            "velocity": direction * PROJECTILE_SPEED,
            "owner": self.team,
        }

    def tick(self):
        """Advance per-frame actor timers."""
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

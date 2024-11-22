##################################################
# GAME_AGENT
##################################################

import pygame
import math
from constants import *

class GameAgent:
    """Class representing a game agent (player or enemy)."""

    def __init__(self, position, angle, agent_type='player'):
        self.pos = position
        self.angle = angle
        self.cooldown = 0
        self.alive = True
        self.type = agent_type  # 'player' or 'enemy'

    def move(self, move_forward, move_backward, rotate_left, rotate_right):
        """Update the agent's position and angle based on input."""
        # Rotation
        if rotate_left:
            self.angle = (self.angle + ROTATION_SPEED) % 360
        if rotate_right:
            self.angle = (self.angle - ROTATION_SPEED) % 360

        # Movement in the direction the agent is facing
        movement = pygame.Vector2(0, 0)
        if move_forward:
            movement += pygame.Vector2(
                math.cos(math.radians(self.angle)),
                math.sin(math.radians(self.angle))
            ) * PLAYER_SPEED
        if move_backward:
            movement -= pygame.Vector2(
                math.cos(math.radians(self.angle)),
                math.sin(math.radians(self.angle))
            ) * PLAYER_SPEED

        return movement

    def shoot(self):
        """Handle agent shooting."""
        if self.cooldown == 0:
            direction = pygame.Vector2(
                math.cos(math.radians(self.angle)),
                math.sin(math.radians(self.angle))
            )
            projectile = {
                'pos': self.pos + direction * 20,  # Start slightly ahead of the agent
                'velocity': direction * PROJECTILE_SPEED,
                'owner': self.type  # 'player' or 'enemy'
            }
            self.cooldown = SHOOT_COOLDOWN if self.type == 'player' else ENEMY_SHOOT_COOLDOWN
            return projectile
        return None

    def decrement_cooldown(self):
        """Decrement the shooting cooldown."""
        if self.cooldown > 0:
            self.cooldown -= 1
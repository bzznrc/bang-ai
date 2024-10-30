##################################################
# UTILS
##################################################

import torch
import pygame
import math
from constants import *

def get_device():
    """Determine the device to use for torch operations."""
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def is_collision(game, rect):
    """Check if a rectangle collides with any obstacles or boundaries."""
    if rect.left < 0 or rect.right > game.w or rect.top < 0 or rect.bottom > game.h - BB_HEIGHT:
        return True
    for pt in game.obstacles:
        obstacle_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        if rect.colliderect(obstacle_rect):
            return True
    return False

def line_intersects_rect(p1, p2, rect):
    """Check if the line segment between p1 and p2 intersects the rectangle."""
    return rect.clipline(p1, p2)

def determine_current_action(move_forward, move_backward, rotate_left, rotate_right, shoot):
    """Determine the current action based on movement and rotation flags."""
    if shoot:
        return ACTION_SHOOT
    elif rotate_left:
        return ACTION_TURN_LEFT
    elif rotate_right:
        return ACTION_TURN_RIGHT
    elif move_forward:
        return ACTION_MOVE_FORWARD
    elif move_backward:
        return ACTION_MOVE_BACKWARD
    else:
        return ACTION_WAIT  # Default action is to wait/do nothing

def get_player_movement_vector(player_angle, action_index):
    """Get the player's movement vector based on the action index."""
    if action_index == ACTION_MOVE_FORWARD:
        movement = pygame.Vector2(
            math.cos(math.radians(player_angle)),
            math.sin(math.radians(player_angle))
        ) * PLAYER_SPEED
    elif action_index == ACTION_MOVE_BACKWARD:
        movement = -pygame.Vector2(
            math.cos(math.radians(player_angle)),
            math.sin(math.radians(player_angle))
        ) * PLAYER_SPEED
    else:
        return pygame.Vector2(0, 0)  # No movement action

    return movement
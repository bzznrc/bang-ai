##################################################
# UTILS
##################################################
import torch
import pygame
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

# Define the player's movement vector based on the action index
def get_player_movement_vector(action_index):
    """Get the player's movement vector based on the action index."""
    if action_index == ACTION_MOVE_UP:
        movement = pygame.Vector2(0, -1)
    elif action_index == ACTION_MOVE_DOWN:
        movement = pygame.Vector2(0, 1)
    elif action_index == ACTION_MOVE_LEFT:
        movement = pygame.Vector2(-1, 0)
    elif action_index == ACTION_MOVE_RIGHT:
        movement = pygame.Vector2(1, 0)
    else:
        return None  # No movement action

    return movement.normalize()

def is_collision(game, rect):
    """Check if a rectangle collides with any obstacles or boundaries."""
    if rect.left < 0 or rect.right > game.w or rect.top < 0 or rect.bottom > game.h - BB_HEIGHT:
        return True
    for pt in game.obstacles:
        obstacle_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
        if rect.colliderect(obstacle_rect):
            return True
    return False

def set_movement_flags(direction):
    """Set movement flags based on the given direction."""
    move_up = move_down = move_left = move_right = False
    if direction == 'up':
        move_up = True
    elif direction == 'down':
        move_down = True
    elif direction == 'left':
        move_left = True
    elif direction == 'right':
        move_right = True
    return move_up, move_down, move_left, move_right

def line_intersects_rect(p1, p2, rect):
        """Check if the line segment between p1 and p2 intersects the rectangle."""
        return rect.clipline(p1, p2)

def is_point_in_cone(point, cone_origin, cone_direction, cone_angle):
    """Check if a point is within a cone defined by an origin, direction, and angle."""
    vector_to_point = point - cone_origin
    if vector_to_point.length() == 0:
        return True
    vector_to_point = vector_to_point.normalize()
    cone_direction = cone_direction.normalize()
    angle = cone_direction.angle_to(vector_to_point)
    return abs(angle) <= cone_angle / 2
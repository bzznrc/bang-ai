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

def is_obstacle_between(game, point1, point2):
    """Check if there's an obstacle between two points."""
    for obs in game.obstacles:
        obs_rect = pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE)
        if line_intersects_rect(point1, point2, obs_rect):
            return True
    return False
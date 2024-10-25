##################################################
# GAME
##################################################

import pygame
import math
import random
import numpy as np
from constants import *
from utils import *
from ui import GameUI

class GameAgent:
    """Class representing a game agent (player or enemy)."""
    def __init__(self, position, angle, agent_type='player'):
        self.pos = position
        self.angle = angle
        self.cooldown = 0
        self.alive = True
        self.type = agent_type  # 'player' or 'enemy'

class Game:
    """Base class representing the Game."""

    def __init__(self):
        """Initialize the game state."""
        # Game dimensions
        self.w = SCREEN_WIDTH
        self.h = SCREEN_HEIGHT

        # Initialize Pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        if SHOW_GAME:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Bang')
        else:
            # Create a surface to avoid issues when not displaying the game
            self.display = pygame.Surface((self.w, self.h))

        # Initialize UI
        self.ui = GameUI(self.display, self)

        # Initialize scores
        self.p1_score = 0
        self.p2_score = 0

        self.reset()

    def reset(self):
        """Reset the game state."""
        # Initialize player
        player_pos = pygame.Vector2(self.w / 8, self.h / 2 - BB_HEIGHT // 2)
        self.player = GameAgent(player_pos, angle=0, agent_type='player')  # Facing right

        # Initialize enemy
        enemy_pos = pygame.Vector2(7 * self.w / 8, self.h / 2 - BB_HEIGHT // 2)
        self.enemy = GameAgent(enemy_pos, angle=180, agent_type='enemy')  # Facing left

        self.obstacles = []
        self.projectiles = []

        self.frame_count = 0  # Do not reset scores here
        self._place_obstacles()

        # Initialize enemy behavior
        self.enemy_behavior_counter = 0
        self.enemy_move_phase = 0
        self.enemy_move_direction = None

    def _place_obstacles(self):
        """Place multiple obstacles (sections) in random locations."""
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            num_sections = random.randint(MIN_SECTIONS, MAX_SECTIONS)
            # Random starting point
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BB_HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                start_point = pygame.Vector2(x, y)

                # Check distance from player and enemy
                if (start_point.distance_to(self.player.pos) < SAFE_RADIUS or
                    start_point.distance_to(self.enemy.pos) < SAFE_RADIUS):
                    attempts += 1
                    continue

                # Check if start_point overlaps with existing obstacles
                if any(start_point == obs for obs in self.obstacles):
                    attempts += 1
                    continue
                break
            else:
                # Failed to place this obstacle, skip
                continue

            # Generate a connected open shape
            directions = [(-BLOCK_SIZE, 0), (BLOCK_SIZE, 0), (0, -BLOCK_SIZE), (0, BLOCK_SIZE)]
            shape = [start_point]
            current_point = start_point

            for _ in range(num_sections - 1):
                random.shuffle(directions)
                for dx, dy in directions:
                    next_point = pygame.Vector2(current_point.x + dx, current_point.y + dy)
                    # Check boundaries and overlaps
                    if (0 <= next_point.x < self.w and
                        0 <= next_point.y < self.h - BB_HEIGHT and
                        all(not next_point == obs for obs in self.obstacles) and
                        not any(next_point == p for p in shape)):
                        # Ensure it's outside the SAFE_RADIUS
                        if (next_point.distance_to(self.player.pos) < SAFE_RADIUS or
                            next_point.distance_to(self.enemy.pos) < SAFE_RADIUS):
                            continue
                        shape.append(next_point)
                        current_point = next_point
                        break
                else:
                    # Cannot find a valid extension, stop building the shape
                    break

            self.obstacles.extend(shape)

    def _move_agent(self, agent, move_up, move_down, move_left, move_right, rotate_left, rotate_right):
        """Update an agent's position and angle based on input."""
        # Rotation
        if rotate_left:
            agent.angle = (agent.angle + ROTATION_SPEED) % 360
        if rotate_right:
            agent.angle = (agent.angle - ROTATION_SPEED) % 360

        # Movement along the board axes
        movement = pygame.Vector2(0, 0)
        if move_up:
            movement += pygame.Vector2(0, -PLAYER_SPEED)  # Move up along y-axis
        if move_down:
            movement += pygame.Vector2(0, PLAYER_SPEED)   # Move down along y-axis
        if move_left:
            movement += pygame.Vector2(-PLAYER_SPEED, 0)  # Move left along x-axis
        if move_right:
            movement += pygame.Vector2(PLAYER_SPEED, 0)   # Move right along x-axis

        new_position = agent.pos + movement

        # Check collisions with obstacles
        agent_rect = pygame.Rect(new_position.x - BLOCK_SIZE // 2, new_position.y - BLOCK_SIZE // 2, BLOCK_SIZE, BLOCK_SIZE)
        if is_collision(self, agent_rect):
            return

        # Check collisions with other agent
        other_agent = self.enemy if agent == self.player else self.player
        if other_agent.alive:
            other_rect = pygame.Rect(other_agent.pos.x - BLOCK_SIZE // 2, other_agent.pos.y - BLOCK_SIZE // 2, BLOCK_SIZE, BLOCK_SIZE)
            if agent_rect.colliderect(other_rect):
                return

        # No collision detected, move
        agent.pos = new_position

    def _agent_shoot(self, agent):
        """Handle agent shooting."""
        if agent.cooldown == 0:
            direction = pygame.Vector2(
                math.cos(math.radians(agent.angle)),
                math.sin(math.radians(agent.angle))
            )
            projectile = {
                'pos': agent.pos + direction * 20,  # Start slightly ahead of the agent
                'velocity': direction * PROJECTILE_SPEED,
                'owner': agent.type  # 'player' or 'enemy'
            }
            self.projectiles.append(projectile)
            agent.cooldown = SHOOT_COOLDOWN if agent.type == 'player' else ENEMY_SHOOT_COOLDOWN

    def _decrement_cooldowns(self):
        """Decrement shooting cooldowns for agents."""
        for agent in [self.player, self.enemy]:
            if agent.cooldown > 0:
                agent.cooldown -= 1

    def _handle_projectiles(self):
        """Update projectile positions and check for collisions."""
        new_projectiles = []
        for proj in self.projectiles:
            proj['pos'] += proj['velocity']
            proj_rect = pygame.Rect(proj['pos'].x - 5, proj['pos'].y - 5, 10, 10)
            if is_collision(self, proj_rect):
                continue  # Projectile is destroyed upon collision
            # Check if projectile hits an agent
            target_agent = self.player if proj['owner'] == 'enemy' else self.enemy
            if target_agent.alive:
                agent_rect = pygame.Rect(
                    target_agent.pos.x - BLOCK_SIZE // 2,
                    target_agent.pos.y - BLOCK_SIZE // 2,
                    BLOCK_SIZE,
                    BLOCK_SIZE
                )
                if proj_rect.colliderect(agent_rect):
                    if proj['owner'] == 'player':
                        self.p1_score += 1  # Increment P1 score
                    else:
                        self.p2_score += 1  # Increment P2 score
                    target_agent.alive = False  # Agent is eliminated
                    continue  # Projectile is destroyed upon hitting the agent
            new_projectiles.append(proj)
        self.projectiles = new_projectiles

    def _enemy_movement_phase(self, phase_duration):
        """Handle enemy movement during movement phases."""
        if self.enemy_behavior_counter == 1:
            if self.enemy_move_phase == 0:
                # Choose a random initial direction
                self.enemy_move_direction = random.choice(['up', 'down', 'left', 'right'])
            elif self.enemy_move_phase == 1:
                # Choose a perpendicular direction
                if self.enemy_move_direction in ['up', 'down']:
                    self.enemy_move_direction = random.choice(['left', 'right'])
                else:
                    self.enemy_move_direction = random.choice(['up', 'down'])

        # Set movement flags based on direction
        move_up, move_down, move_left, move_right = set_movement_flags(self.enemy_move_direction)
        rotate_left = rotate_right = False

        # Move enemy
        self._move_agent(self.enemy, move_up, move_down, move_left, move_right, rotate_left, rotate_right)

        if self.enemy_behavior_counter >= phase_duration:
            self.enemy_behavior_counter = 0
            self.enemy_move_phase += 1

    def _enemy_actions(self):
        """Handle enemy actions based on the simplified behavior."""
        if self.enemy.alive:
            self.enemy_behavior_counter += 1

            if self.enemy_move_phase in [0, 1]:
                # Movement phases
                self._enemy_movement_phase(25)
            elif self.enemy_move_phase == 2:
                # Align and shoot phase
                aligned = self._enemy_align_to_player()
                if aligned:
                    if self.enemy.cooldown == 0:
                        self._agent_shoot(self.enemy)
                        # Reset behavior after shooting
                        self.enemy_behavior_counter = 0
                        self.enemy_move_phase = 0
                        self.enemy_move_direction = None  # Reset movement direction
                    else:
                        # Wait for cooldown to expire
                        self.enemy.cooldown -= 1

    def _enemy_align_to_player(self):
        """Align the enemy's angle to face the player. Returns True if aligned."""
        # Calculate the angle from enemy to player
        vector_to_player = self.player.pos - self.enemy.pos
        angle_to_player = math.degrees(math.atan2(vector_to_player.y, vector_to_player.x))

        # Normalize angles to be within [0, 360)
        enemy_angle = self.enemy.angle % 360
        angle_to_player = angle_to_player % 360

        # Calculate the minimal signed angle difference between current angle and target angle
        angle_diff = (angle_to_player - enemy_angle + 180) % 360 - 180  # Result is in [-180, 180]

        # Determine rotation direction
        if abs(angle_diff) <= ROTATION_SPEED:
            # Can align in one step
            self.enemy.angle = angle_to_player  # Align exactly
            return True  # Aligned
        else:
            # Need to rotate
            if angle_diff > 0:
                # Need to increase angle (rotate left)
                rotate_left = True   # Rotating left increases the angle
                rotate_right = False
            else:
                # Need to decrease angle (rotate right)
                rotate_left = False
                rotate_right = True  # Rotating right decreases the angle

            # Rotate enemy
            self._move_agent(self.enemy, False, False, False, False, rotate_left, rotate_right)
            return False  # Not yet aligned

    def _get_closest_obstacle(self):
        """Return the position of the closest obstacle."""
        if not self.obstacles:
            return None
        player_pos = self.player.pos
        closest_obstacle = min(self.obstacles, key=lambda obs: player_pos.distance_to(obs))
        return closest_obstacle

    def _get_closest_projectile(self):
        """Return the position of the closest incoming projectile."""
        projectiles = [proj for proj in self.projectiles if proj.get('owner') == 'enemy']
        if not projectiles:
            return None
        player_pos = self.player.pos
        closest_proj = min(projectiles, key=lambda proj: player_pos.distance_to(proj['pos']))
        return closest_proj['pos']
    
    #########################
    # Functions for Rewards
    #########################
    
    #def is_within_engagement_radius(self):
    #    """Check if the player is within the engagement radius of the enemy."""
    #    distance_to_enemy = self.player.pos.distance_to(self.enemy.pos)
    #    return distance_to_enemy <= ENGAGEMENT_RADIUS

    def get_proximity_bonus(self, bonus=0.010):
        """Calculate a continuous bonus based on the player's proximity to the enemy."""
        distance_to_enemy = self.player.pos.distance_to(self.enemy.pos)
        normalized_distance = min(distance_to_enemy / ENGAGEMENT_RADIUS, 1)
        proximity_bonus = bonus * (1 - normalized_distance)
        return proximity_bonus

    #def get_dodge_and_cover_bonus(self, player_movement, bonus = 0.010):
    #    dodge_bonus = 0.0
    #    cover_bonus = 0.0
    #    negative_dodge_bonus = 0.0
#
    #    for proj in self.projectiles:
    #        if proj['owner'] == 'enemy':
    #            proj_pos = proj['pos']
    #            proj_velocity = proj['velocity']
    #            proj_direction = proj_velocity.normalize()
    #            if is_point_in_cone(self.player.pos, proj_pos, proj_direction, 30):
    #                if self.is_player_behind_cover(proj):
    #                    cover_bonus = bonus
    #                else:
    #                    negative_dodge_bonus = -bonus
    #                if player_movement:
    #                    angle_between = abs(player_movement.angle_to(proj_direction))
    #                    if abs(angle_between - 90) <= 15:
    #                        dodge_bonus = bonus
    #    total_dodge_bonus = dodge_bonus + negative_dodge_bonus
    #    return total_dodge_bonus, cover_bonus

    def get_dodge_and_cover_bonus(self, player_velocity, bonus=0.010):
        dodge_bonus = 0.0
        cover_bonus = 0.0
        negative_dodge_bonus = 0.0

        for proj in self.projectiles:
            if proj['owner'] == 'enemy':
                proj_pos = proj['pos']
                proj_velocity = proj['velocity']
                proj_direction = proj_velocity.normalize()
                vector_to_player = self.player.pos - proj_pos

                # Check if the player is behind cover relative to the projectile
                if self._is_player_behind_cover(proj):
                    cover_bonus += bonus

                # Calculate angle between player's movement and projectile's direction
                if player_velocity is not None and player_velocity.length() > 0:
                    angle_between = abs(player_velocity.normalize().angle_to(proj_direction))
                    # Continuous dodge bonus based on deviation from 90 degrees
                    dodge_bonus += bonus * (1 - min(abs(angle_between - 90) / 90, 1))

                    # Penalty if the player is moving directly towards or away from the projectile
                    if abs(angle_between) < 30 or abs(angle_between - 180) < 30:
                        negative_dodge_bonus -= bonus * (1 - min(abs(angle_between - 90) / 90, 1))

        total_dodge_bonus = dodge_bonus + negative_dodge_bonus
        return total_dodge_bonus, cover_bonus

    def _is_player_behind_cover(self, proj):
        """Check if there's an obstacle between the projectile and the player."""
        proj_pos = proj['pos']
        player_pos = self.player.pos
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE)
            if line_intersects_rect(proj_pos, player_pos, obs_rect):
                return True
        return False

    #def is_shooting_aligned_with_enemy(self):
    #    """Check if the player is shooting within a certain angle threshold of the enemy."""
    #    vector_to_enemy = self.enemy.pos - self.player.pos
    #    angle_to_enemy = math.degrees(math.atan2(vector_to_enemy.y, vector_to_enemy.x))
    #    player_angle = self.player.angle % 360
    #    angle_to_enemy = angle_to_enemy % 360
    #    angle_diff = (angle_to_enemy - player_angle + 180) % 360 - 180
    #    return abs(angle_diff) <= ALIGNMENT_ANGLE

    def get_shooting_alignment_bonus(self, bonus=0.010):
        """Calculate a continuous bonus based on how well the player is aligned when shooting."""
        vector_to_enemy = self.enemy.pos - self.player.pos
        angle_to_enemy = math.degrees(math.atan2(vector_to_enemy.y, vector_to_enemy.x))
        player_angle = self.player.angle % 360
        angle_to_enemy = angle_to_enemy % 360
        angle_diff = (angle_to_enemy - player_angle + 180) % 360 - 180
        alignment_bonus = bonus * (1 - abs(angle_diff) / 180)
        return max(alignment_bonus, 0)  # Ensure non-negative

    def get_state(self):
        """Get the current state representation for the AI agent."""
        # Delta X and Y to Enemy (normalized)
        delta_x_enemy = (self.enemy.pos.x - self.player.pos.x) / self.w
        delta_y_enemy = (self.enemy.pos.y - self.player.pos.y) / self.h

        # Delta X and Y to Closest Obstacle (normalized)
        closest_obstacle = self._get_closest_obstacle()
        if closest_obstacle:
            delta_x_obstacle = (closest_obstacle.x - self.player.pos.x) / self.w
            delta_y_obstacle = (closest_obstacle.y - self.player.pos.y) / self.h
        else:
            delta_x_obstacle = 0
            delta_y_obstacle = 0

        # Delta X and Y to Closest Incoming Projectile (normalized)
        closest_projectile = self._get_closest_projectile()
        if closest_projectile:
            delta_x_projectile = (closest_projectile.x - self.player.pos.x) / self.w
            delta_y_projectile = (closest_projectile.y - self.player.pos.y) / self.h
        else:
            delta_x_projectile = 0
            delta_y_projectile = 0

        # Projectile Angle (in degrees)
        projectile_angle = math.degrees(math.atan2(delta_y_projectile, delta_x_projectile)) if closest_projectile else 0

        # Player Velocity (X and Y components)
        player_velocity_x = self.player.pos.x / FPS
        player_velocity_y = self.player.pos.y / FPS

        # Player Angle (normalized between 0 and 360)
        player_angle = self.player.angle

        # Cooldown Status
        cooldown_status = 1 if self.player.cooldown == 0 else 0

        # Distance to Center of Map (normalized)
        center_x = self.w / 2
        center_y = self.h / 2
        distance_to_center = np.linalg.norm([self.player.pos.x - center_x, self.player.pos.y - center_y]) / max(self.w, self.h)

        # Construct the state array
        state = [
            delta_x_enemy,
            delta_y_enemy,
            delta_x_obstacle,
            delta_y_obstacle,
            delta_x_projectile,
            delta_y_projectile,
            projectile_angle,
            player_velocity_x,
            player_velocity_y,
            player_angle,
            cooldown_status,
            distance_to_center
        ]

        return state

    def print_state(self):
        """Print the current state for debugging purposes."""
        print(f"State: {', '.join(f'{x:.2f}' for x in self.get_state())}")

    def play_step(self, action):
        """Execute one game step based on the action taken."""
        # To be implemented in subclasses
        pass
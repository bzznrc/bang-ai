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
from game_agent import GameAgent  # Import the updated GameAgent

class Game:
    """Base class representing the Game."""

    def __init__(self, level=1):
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

        # Initialize previous action
        self.prev_action = None

        # Set game level for curriculum learning
        self.level = level
        self.configure_level()

        self.reset()

    def configure_level(self):
        """Configure game parameters based on the current level."""
        if self.level == 1:
            self.num_obstacles = 4
            self.enemy_stationary = True
            self.enemy_shooting = True
            self.enemy_move_probability = 0.0
            self.enemy_shoot_cooldown = float('inf')  # Shoot without cooldown
            self.enemy_behavior = self._enemy_behavior_level1
        elif self.level == 2:
            self.num_obstacles = 8
            self.enemy_stationary = True
            self.enemy_shooting = True
            self.enemy_move_probability = 0.0
            self.enemy_shoot_cooldown = SHOOT_COOLDOWN * 3
            self.enemy_behavior = self._enemy_behavior_level2
        elif self.level == 3:
            self.num_obstacles = 12
            self.enemy_stationary = False
            self.enemy_shooting = True
            self.enemy_move_probability = 0.05
            self.enemy_shoot_cooldown = SHOOT_COOLDOWN * 3
            self.enemy_behavior = self._enemy_behavior_level3
        else:
            # Default configuration
            self.num_obstacles = NUM_OBSTACLES
            self.enemy_stationary = ENEMY_STATIONARY
            self.enemy_shooting = ENEMY_SHOOTING
            self.enemy_move_probability = ENEMY_MOVE_PROBABILITY
            self.enemy_shoot_cooldown = ENEMY_SHOOT_COOLDOWN
            self.enemy_behavior = self._enemy_behavior_level1  # Default to level 1 behavior

    def reset(self):
        """Reset the game state."""
        # Calculate random Y-offset for player and enemy
        player_y_offset = random.choice([0, SPAWN_Y_OFFSET, -SPAWN_Y_OFFSET])
        enemy_y_offset = random.choice([0, SPAWN_Y_OFFSET, -SPAWN_Y_OFFSET])

        # Initialize player with variable Y position
        player_pos = pygame.Vector2(self.w / 8, (self.h / 2 - BB_HEIGHT // 2) + player_y_offset)
        self.player = GameAgent(player_pos, angle=0, agent_type='player')  # Facing right

        # Initialize enemy with variable Y position
        enemy_pos = pygame.Vector2(7 * self.w / 8, (self.h / 2 - BB_HEIGHT // 2) + enemy_y_offset)
        self.enemy = GameAgent(enemy_pos, angle=180, agent_type='enemy')  # Facing left

        self.obstacles = []
        self.projectiles = []

        self.frame_count = 0  # Do not reset scores here
        self._place_obstacles()

        # Initialize enemy behavior variables
        self.enemy_state = 'idle'  # Enemy state: 'idle', 'move', 'attack'
        self.enemy_state_counter = 0

        # Initialize previous enemy distance for reward calculation
        self.prev_enemy_distance = self.player.pos.distance_to(self.enemy.pos)

    def apply_action(self, action_index):
        """
        Apply the given action to the player.
        """
        move_forward = move_backward = rotate_left = rotate_right = shoot = False

        if action_index == ACTION_MOVE_FORWARD:
            move_forward = True
        elif action_index == ACTION_MOVE_BACKWARD:
            move_backward = True
        elif action_index == ACTION_TURN_LEFT:
            rotate_left = True
        elif action_index == ACTION_TURN_RIGHT:
            rotate_right = True
        elif action_index == ACTION_SHOOT:
            shoot = True
        elif action_index == ACTION_WAIT:
            pass  # Do nothing

        # Move player using GameAgent's move method
        movement = self.player.move(move_forward, move_backward, rotate_left, rotate_right)
        self._update_agent_position(self.player, movement)

        # Handle shooting
        if shoot:
            projectile = self.player.shoot()
            if projectile:
                self.projectiles.append(projectile)

        # Decrement cooldown
        self.player.decrement_cooldown()

    def _update_agent_position(self, agent, movement):
        """Update agent's position if there is no collision."""
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

    def _place_obstacles(self):
        """Place multiple obstacles (sections) in random locations."""
        self.obstacles = []
        for _ in range(self.num_obstacles):
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
                    target_agent.alive = False  # Agent is eliminated
                    continue  # Projectile is destroyed upon hitting the agent
            new_projectiles.append(proj)
        self.projectiles = new_projectiles

    def _enemy_actions(self):
        """Handle enemy actions based on the current level."""
        if not self.enemy.alive:
            return

        # Call the appropriate enemy behavior function
        self.enemy_behavior()

    # Enemy behavior functions for each level

    def _enemy_behavior_level1(self):
        """Level 1 Enemy Behavior: Shoots straight without rotating or moving."""
        self.enemy.angle = 180  # Facing left
        if self.enemy_shooting and self.enemy.cooldown == 0:
            projectile = self.enemy.shoot()
            if projectile:
                self.projectiles.append(projectile)
        else:
            self.enemy.decrement_cooldown()

    def _enemy_behavior_level2(self):
        """Level 2 Enemy Behavior: Rotates towards player (±15° error) and shoots."""
        # Rotate towards player with some error
        vector_to_player = self.player.pos - self.enemy.pos
        angle_to_player = math.degrees(math.atan2(vector_to_player.y, vector_to_player.x)) % 360

        # Introduce aiming error
        aim_error = random.choice([-15, 0, 15])  # 1/3 chance for each
        self.enemy.angle = (angle_to_player + aim_error) % 360

        if self.enemy_shooting and self.enemy.cooldown == 0:
            projectile = self.enemy.shoot()
            if projectile:
                self.projectiles.append(projectile)
        else:
            self.enemy.decrement_cooldown()

    def _enemy_behavior_level3(self):
        """Level 3 Enemy Behavior: Moves towards player, rotates (±15° error), and shoots."""
        if self.enemy_state == 'idle':
            # Decide to move towards player
            self.enemy_state = 'move'
            self.enemy_state_counter = 25  # Move for 25 frames
        elif self.enemy_state == 'move':
            # Move towards player
            move_forward = True
            move_backward = rotate_left = rotate_right = False

            # Calculate angle towards player
            vector_to_player = self.player.pos - self.enemy.pos
            angle_to_player = math.degrees(math.atan2(vector_to_player.y, vector_to_player.x)) % 360
            self.enemy.angle = angle_to_player

            # Move enemy
            movement = self.enemy.move(move_forward, move_backward, rotate_left, rotate_right)
            self._update_agent_position(self.enemy, movement)

            self.enemy_state_counter -= 1
            if self.enemy_state_counter <= 0:
                self.enemy_state = 'attack'
        elif self.enemy_state == 'attack':
            # Rotate towards player and shoot
            vector_to_player = self.player.pos - self.enemy.pos
            angle_to_player = math.degrees(math.atan2(vector_to_player.y, vector_to_player.x)) % 360

            # Introduce aiming error
            aim_error = random.choice([-15, 0, 15])  # 1/3 chance for each
            self.enemy.angle = (angle_to_player + aim_error) % 360

            if self.enemy_shooting and self.enemy.cooldown == 0:
                projectile = self.enemy.shoot()
                if projectile:
                    self.projectiles.append(projectile)
                self.enemy_state = 'idle'  # Reset state after attack
            else:
                self.enemy.decrement_cooldown()

    def _get_closest_projectile(self):
        """Return the closest enemy projectile."""
        projectiles = [proj for proj in self.projectiles if proj['owner'] == 'enemy']
        if not projectiles:
            return None
        player_pos = self.player.pos
        closest_proj = min(projectiles, key=lambda proj: player_pos.distance_to(proj['pos']))
        return closest_proj

    def _is_in_projectile_trajectory(self):
        """Check if the player is in the direct trajectory of any enemy projectile."""
        for proj in self.projectiles:
            if proj['owner'] == 'enemy':
                # Projectile's current position and velocity
                proj_pos = proj['pos']
                proj_vel = proj['velocity']

                # Vector from projectile to player
                vector_to_player = self.player.pos - proj_pos

                # Normalize vectors
                proj_direction = proj_vel.normalize()
                vector_to_player_norm = vector_to_player.normalize()

                # Angle between projectile's direction and vector to player
                angle_between = proj_direction.angle_to(vector_to_player_norm)

                # If the angle is small, the player is in the trajectory
                if abs(angle_between) < 5:  # Threshold angle in degrees
                    # Check if the projectile is moving towards the player
                    if proj_direction.dot(vector_to_player_norm) > 0:
                        return True
        return False

    def _get_obstacle_distances(self):
        """
        Calculate distances to the nearest obstacles in 8 directions (45-degree increments)
        relative to the player's facing direction (self.angle).
        """
        distances = [float('inf')] * 8  # Initialize distances with infinity
        player_pos = self.player.pos
        player_angle = self.player.angle  # The player's current facing angle in degrees

        # Define 8 directional vectors for cones (relative to 0 degrees)
        directions = [
            pygame.Vector2(1, 0),  # 0 degrees
            pygame.Vector2(1, 1).normalize(),  # 45 degrees
            pygame.Vector2(0, 1),  # 90 degrees
            pygame.Vector2(-1, 1).normalize(),  # 135 degrees
            pygame.Vector2(-1, 0),  # 180 degrees
            pygame.Vector2(-1, -1).normalize(),  # 225 degrees
            pygame.Vector2(0, -1),  # 270 degrees
            pygame.Vector2(1, -1).normalize()   # 315 degrees
        ]

        # Rotate the directions based on the player's facing angle
        rotated_directions = []
        for direction in directions:
            rotated_dir = direction.rotate(player_angle)  # Rotate by player angle
            rotated_directions.append(rotated_dir)

        # Iterate over obstacles and find the closest ones in each cone
        for obs in self.obstacles:
            obs_vector = obs - player_pos  # Vector from player to obstacle
            obs_distance = obs_vector.length()  # Calculate distance to the obstacle
            if obs_distance == 0:  # Skip if the obstacle is at the player's position
                continue

            # Calculate the angle between the player and the obstacle
            obs_angle = (math.degrees(math.atan2(obs_vector.y, obs_vector.x)) % 360)

            # Calculate the relative angle difference between the obstacle's angle and the rotated directions
            angle_diffs = []
            for direction in rotated_directions:
                # Get the angle of the direction in degrees
                direction_angle = math.degrees(math.atan2(direction.y, direction.x)) % 360
                angle_diff = min(abs(obs_angle - direction_angle), 360 - abs(obs_angle - direction_angle))
                angle_diffs.append(angle_diff)

            # Find the closest direction by checking the smallest angle difference
            closest_direction_index = angle_diffs.index(min(angle_diffs))
            distances[closest_direction_index] = min(distances[closest_direction_index], obs_distance)

        # Normalize distances to screen diagonal
        max_distance = math.hypot(self.w, self.h)
        return [d / max_distance if d != float('inf') else -1.0 for d in distances]  # Return normalized distances

    def _is_line_of_sight(self):
        """Check if the enemy is within line of sight and aligned with the player's facing direction."""
        # Calculate the vector from the player to the enemy
        vector_to_enemy = self.enemy.pos - self.player.pos
        if vector_to_enemy.length() == 0:
            return False  # Same position, unlikely but handle it

        # Calculate the angle between the player's facing direction and the vector to the enemy
        player_facing_vector = pygame.Vector2(
            math.cos(math.radians(self.player.angle)),
            math.sin(math.radians(self.player.angle))
        )

        angle_between = player_facing_vector.angle_to(vector_to_enemy)

        # Check if the enemy is within the ±ALIGNMENT_TOLERANCE°
        if abs(angle_between) <= ALIGNMENT_TOLERANCE:
            # Check if there are no obstacles between the player and the enemy
            if not is_obstacle_between(self, self.player.pos, self.enemy.pos):
                return True  # LOS is clear and enemy is aligned
        return False  # No LOS or enemy is not aligned

    def get_state(self):
        """Get the current state representation for the AI agent."""
        # Obstacle distances (8 features)
        obstacle_distances = self._get_obstacle_distances()

        # Enemy features
        enemy_distance = self.player.pos.distance_to(self.enemy.pos) / max(self.w, self.h)
        vector_to_enemy = self.enemy.pos - self.player.pos
        enemy_angle = math.degrees(math.atan2(vector_to_enemy.y, vector_to_enemy.x)) % 360
        relative_enemy_angle = ((enemy_angle - self.player.angle + 180) % 360) - 180

        # Projectile features
        closest_projectile = self._get_closest_projectile()
        if closest_projectile:
            proj_distance = self.player.pos.distance_to(closest_projectile['pos']) / max(self.w, self.h)
            vector_to_proj = closest_projectile['pos'] - self.player.pos
            proj_angle = math.degrees(math.atan2(vector_to_proj.y, vector_to_proj.x)) % 360
            relative_proj_angle = ((proj_angle - self.player.angle + 180) % 360) - 180
        else:
            proj_distance = -1.0
            relative_proj_angle = 0.0

        # Position features (2 features)
        x_position = self.player.pos.x / self.w  # Normalize to [0, 1]
        y_position = self.player.pos.y / self.h  # Normalize to [0, 1]

        # Booleans (In Projectile Trajectory, Line of Sight)
        ipt = 1 if self._is_in_projectile_trajectory() else 0
        los = 1 if self._is_line_of_sight() else 0

        # Combine all features
        state = (
            obstacle_distances +                        # 8 features
            [enemy_distance, relative_enemy_angle / 180] +  # 2 features
            [proj_distance, relative_proj_angle / 180] +    # 2 features
            [x_position, y_position] +                      # 2 features (position)
            [ipt, los]                                      # 2 features (booleans)
        )

        return state

    def print_state(self):
        """Print the current state for debugging purposes."""
        print(f"State: {', '.join(f'{x:.2f}' for x in self.get_state())}")

    def play_step(self, action):
        """Execute one game step based on the action taken."""
        pass  # To be implemented in subclasses
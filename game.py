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

        # Initialize previous action
        self.prev_action = None

        self.reset()

    def reset(self):
        """Reset the game state."""

        # Calculate random Y-offset for player and enemy
        player_y_offset = random.choice([0, SPAWN_Y_OFFSET, -SPAWN_Y_OFFSET])
        enemy_y_offset = random.choice([0, SPAWN_Y_OFFSET, -SPAWN_Y_OFFSET])

        # Initialize player with variable Y position
        player_pos = pygame.Vector2(self.w / 8, (self.h / 2 - BB_HEIGHT // 2) + player_y_offset)
        self.player = GameAgent(player_pos, angle=0, agent_type='player')  # Facing right

        # Initialize player with variable Y position
        enemy_pos = pygame.Vector2(7 * self.w / 8, (self.h / 2 - BB_HEIGHT // 2) + enemy_y_offset)
        self.enemy = GameAgent(enemy_pos, angle=180, agent_type='enemy')  # Facing left

        self.obstacles = []
        self.projectiles = []

        self.frame_count = 0  # Do not reset scores here
        self._place_obstacles()

        # Initialize enemy behavior
        self.enemy_behavior_counter = 0
        self.enemy_move_phase = 0
        self.enemy_move_action = None  # Initialize move action
        self.enemy_alignment_choice = None  # For alignment phase
        self.enemy_target_angle = None  # For alignment phase

    def apply_action(self, action_index):
        """
        Apply the given action to the player.
        Sets movement and rotation flags, moves the player, and handles shooting.
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

        # Move player
        self._move_agent(self.player, move_forward, move_backward, rotate_left, rotate_right)

        # Handle shooting
        if shoot:
            self._agent_shoot(self.player)

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

    def _move_agent(self, agent, move_forward, move_backward, rotate_left, rotate_right):
        """Update an agent's position and angle based on input."""
        # Rotation
        if rotate_left:
            agent.angle = (agent.angle + ROTATION_SPEED) % 360
        if rotate_right:
            agent.angle = (agent.angle - ROTATION_SPEED) % 360

        # Movement in the direction the agent is facing
        movement = pygame.Vector2(0, 0)
        if move_forward:
            movement += pygame.Vector2(
                math.cos(math.radians(agent.angle)),
                math.sin(math.radians(agent.angle))
            ) * PLAYER_SPEED
        if move_backward:
            movement -= pygame.Vector2(
                math.cos(math.radians(agent.angle)),
                math.sin(math.radians(agent.angle))
            ) * PLAYER_SPEED

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
                    target_agent.alive = False  # Agent is eliminated
                    continue  # Projectile is destroyed upon hitting the agent
            new_projectiles.append(proj)
        self.projectiles = new_projectiles

    def _enemy_movement_phase(self, phase_duration):
        """Handle enemy movement during movement phases."""
        if self.enemy_behavior_counter == 1:
            if self.enemy_move_phase == 0:
                # Choose a random initial action
                self.enemy_move_action = random.choice(['move_forward', 'move_backward', 'rotate_left', 'rotate_right'])
            elif self.enemy_move_phase == 1:
                # Choose a different action
                possible_actions = ['move_forward', 'move_backward', 'rotate_left', 'rotate_right']
                possible_actions.remove(self.enemy_move_action)  # Ensure it's different
                self.enemy_move_action = random.choice(possible_actions)

        # Set movement flags based on action
        move_forward = move_backward = rotate_left = rotate_right = False
        if self.enemy_move_action == 'move_forward':
            move_forward = True
        elif self.enemy_move_action == 'move_backward':
            move_backward = True
        elif self.enemy_move_action == 'rotate_left':
            rotate_left = True
        elif self.enemy_move_action == 'rotate_right':
            rotate_right = True

        # Move enemy
        self._move_agent(self.enemy, move_forward, move_backward, rotate_left, rotate_right)

        if self.enemy_behavior_counter >= phase_duration:
            self.enemy_behavior_counter = 0
            self.enemy_move_phase += 1

    def _enemy_actions(self):
        """Handle enemy actions based on movement phases."""
        if self.enemy.alive:
            self.enemy_behavior_counter += 1

            if self.enemy_move_phase in [0, 1]:
                # Movement phases
                self._enemy_movement_phase(25)
            elif self.enemy_move_phase == 2:
                # Alignment and shooting phase
                if self.enemy_behavior_counter == 1:
                    # First frame of alignment phase
                    self._enemy_decide_alignment()

                # Now align towards the target angle
                aligned = self._enemy_align_to_target()
                if aligned:
                    if self.enemy.cooldown == 0:
                        self._agent_shoot(self.enemy)
                        # Reset behavior after shooting
                        self.enemy_behavior_counter = 0
                        self.enemy_move_phase = 0
                        self.enemy_move_action = None  # Reset movement action
                        self.enemy_alignment_choice = None  # Reset alignment choice
                        self.enemy_target_angle = None  # Reset target angle
                    else:
                        # Wait for cooldown to expire
                        self.enemy.cooldown -= 1

    def _enemy_decide_alignment(self):
        """Decide the alignment choice and calculate the target angle once per alignment phase."""
        # Randomness in alignment
        self.enemy_alignment_choice = random.choice(['undershoot', 'overshoot', 'exact'])
        
        # Calculate the angle from enemy to player
        vector_to_player = self.player.pos - self.enemy.pos
        angle_to_player = math.degrees(math.atan2(vector_to_player.y, vector_to_player.x)) % 360

        if self.enemy_alignment_choice == 'undershoot':
            # Rotate towards the player but stop ROTATION_SPEED degrees before alignment
            self.enemy_target_angle = (angle_to_player - ROTATION_SPEED) % 360
        elif self.enemy_alignment_choice == 'overshoot':
            # Rotate towards the player but rotate ROTATION_SPEED degrees past alignment
            self.enemy_target_angle = (angle_to_player + ROTATION_SPEED) % 360
        else:
            # Align exactly
            self.enemy_target_angle = angle_to_player % 360

    def _enemy_align_to_target(self):
        """Align the enemy's angle to the stored target angle. Returns True if aligned."""
        if self.enemy_target_angle is None:
            return False  # No target angle set

        # Normalize angles to be within [0, 360)
        enemy_angle = self.enemy.angle % 360
        angle_diff = (self.enemy_target_angle - enemy_angle + 180) % 360 - 180  # Result is in [-180, 180]

        # Determine rotation direction
        if abs(angle_diff) <= ROTATION_SPEED:
            # Can align in one step
            self.enemy.angle = self.enemy_target_angle  # Align to target angle
            return True  # Aligned (or close enough)
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
            self._move_agent(self.enemy, False, False, rotate_left, rotate_right)
            return False  # Not yet aligned

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
    
    def _get_closest_obstacle(self):
        """Return the position of the closest obstacle."""
        if not self.obstacles:
            return None
        player_pos = self.player.pos
        closest_obstacle = min(self.obstacles, key=lambda obs: player_pos.distance_to(obs))
        return closest_obstacle

    def _is_obstacle_between(self, point1, point2):
        """Check if there's an obstacle between two points."""
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE)
            if line_intersects_rect(point1, point2, obs_rect):
                return True
        return False

    def _is_player_behind_cover_from_enemy(self):
        """Check if the player is behind cover from the enemy's perspective."""
        return self._is_obstacle_between(self.enemy.pos, self.player.pos)
    
    def is_player_within_proximity(self):
        """Check if the player is within the proximity range of the enemy."""
        distance = (self.enemy.pos - self.player.pos).magnitude()
        return distance <= ENGAGEMENT_RADIUS

    def get_state(self):
        """Get the current state representation for the AI agent."""
        # Delta X and Y to Enemy (normalized)
        #delta_x_enemy = (self.enemy.pos.x - self.player.pos.x) / self.w
        #delta_y_enemy = (self.enemy.pos.y - self.player.pos.y) / self.h

        # Distance to Enemy (normalized)
        distance_to_enemy = self.player.pos.distance_to(self.enemy.pos) / max(self.w, self.h)

        # Relative Angle to Enemy (normalized)
        vector_to_enemy = self.enemy.pos - self.player.pos
        angle_to_enemy = math.degrees(math.atan2(vector_to_enemy.y, vector_to_enemy.x)) % 360
        relative_angle_to_enemy = ((angle_to_enemy - self.player.angle + 180) % 360) - 180  # Range [-180, 180]
        relative_angle_to_enemy_normalized = relative_angle_to_enemy / 180  # Normalize to [-1, 1]

        # Player Angle (normalized between -1 and 1)
        player_angle_normalized = (self.player.angle % 360) / 180 - 1  # Normalize to [-1, 1]

        # Distance to Center of Map (normalized)
        center_pos = pygame.Vector2(self.w / 2, self.h / 2)
        distance_to_center = self.player.pos.distance_to(center_pos) / max(self.w, self.h)

        # Being Covered (1 if there's an obstacle between player and enemy)
        being_covered = 1 if self._is_player_behind_cover_from_enemy() else 0

        # Angle to Cover (relative angle to nearest obstacle)
        closest_obstacle = self._get_closest_obstacle()
        if closest_obstacle:
            vector_to_obstacle = closest_obstacle - self.player.pos
            angle_to_obstacle = math.degrees(math.atan2(vector_to_obstacle.y, vector_to_obstacle.x)) % 360
            relative_angle_to_obstacle = ((angle_to_obstacle - self.player.angle + 180) % 360) - 180  # Range [-180, 180]
            angle_to_cover = relative_angle_to_obstacle / 180  # Normalize to [-1, 1]
        else:
            angle_to_cover = 0.0  # No obstacle found

        # In Projectile Trajectory (1 if player is in direct path of any enemy projectile)
        in_projectile_trajectory = 1 if self._is_in_projectile_trajectory() else 0

        # Distance to Closest Projectile (normalized)
        closest_projectile = self._get_closest_projectile()
        if closest_projectile:
            distance_to_projectile = self.player.pos.distance_to(closest_projectile['pos']) / max(self.w, self.h)
        else:
            distance_to_projectile = 1.0  # Max distance

        # Line of Sight (1 if no obstacles between player and enemy)
        line_of_sight = 1 if not self._is_obstacle_between(self.player.pos, self.enemy.pos) else 0

        # Cooldown Status
        cooldown_status = 1 if self.player.cooldown == 0 else 0

        # Construct the state array
        state = [ #10 States
            #delta_x_enemy,
            #delta_y_enemy,
            distance_to_enemy,
            relative_angle_to_enemy_normalized,
            player_angle_normalized,
            distance_to_center,
            being_covered,
            angle_to_cover,
            in_projectile_trajectory,
            distance_to_projectile,
            line_of_sight,
            cooldown_status]

        return state

    def print_state(self):
        """Print the current state for debugging purposes."""
        print(f"State: {', '.join(f'{x:.2f}' for x in self.get_state())}")

    def play_step(self, action):
        """Execute one game step based on the action taken."""
        pass  # To be implemented in subclasses
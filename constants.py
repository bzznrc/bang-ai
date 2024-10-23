# constants.py

# Screen and Grid Dimensions
GRID_SIZE_X = 32          # Number of tiles horizontally
GRID_SIZE_Y = 24          # Number of tiles vertically
BLOCK_SIZE = 20           # Size of each grid tile
BB_HEIGHT = 30            # Bottom Bar height
BB_MARGIN = 20            # Margin for the bottom bar
SCREEN_WIDTH = GRID_SIZE_X * BLOCK_SIZE
SCREEN_HEIGHT = GRID_SIZE_Y * BLOCK_SIZE + BB_HEIGHT

# Game Speed
FPS = 60                  # Frames per second for the game
AI_FPS = 180               # Frames per second for the AI game (if applicable)

# Colors
COLOR_LEFT_TEAM = (30, 100, 100)         # Dark Teal for left team (player)
COLOR_LEFT_TEAM_OUTLINE = (50, 215, 200) # Turquoise outline for left team
COLOR_RIGHT_TEAM = (125, 45, 45)         # Dark Red for right team (enemy)
COLOR_RIGHT_TEAM_OUTLINE = (240, 95, 95) # Red Orange outline for right team
COLOR_OBSTACLE_OUTLINE = (125, 125, 125)
COLOR_OBSTACLE_PRIMARY = (45, 45, 45)    # Same as COLOR_BACKGROUND or desired fill color
COLOR_PROJECTILE = (235, 195, 50)        # Yellow for projectiles
COLOR_BACKGROUND = (45, 45, 45)          # Dark Grey
COLOR_SCORE = (255, 255, 255)            # White for score display

# Game Constants
PLAYER_SPEED = 5           # Movement speed of the player (pixels per frame)
ROTATION_SPEED = 30        # Rotation angle in degrees per action
PROJECTILE_SPEED = 10      # Speed of projectiles (pixels per frame)
SHOOT_COOLDOWN = 30        # Frames until the player can shoot again

# Obstacle Constants
NUM_OBSTACLES = 10         # Number of obstacles to spawn in the game

# Add these constants to constants.py if not already present
MIN_SECTIONS = 2          # Minimum number of sections for obstacles
MAX_SECTIONS = 5          # Maximum number of sections for obstacles

SAFE_RADIUS = 100  # Minimum distance from players in pixels

# Font
FONT_SIZE = 24             # Font size for text displays

# Model Constants
MODEL_SAVE_PREFIX = 'bang_32_16_top'     # Prefix tag for saving models
LOAD_PREVIOUS_MODEL = True     # Start training from a saved model if True

# Toggle for training plots
PLOT_TRAIN = False

USE_GPU = False

SHOW_GAME = True  # Set to True to display the game window during training

# Agent Hyperparameters
MAX_MEMORY = 100_000    # Maximum size of the replay memory
BATCH_SIZE = 1000       # Number of samples per training batch
LR = 0.002              # Learning rate for the optimizer
#HIDDEN_LAYERS = [32]    # List defining the size of hidden layers
HIDDEN_LAYERS = [32, 16]
#HIDDEN_LAYERS = [64, 32, 16]
GAMMA = 0.9              # Discount factor for future rewards

# Epsilon Parameters for Exploration
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_DECAY = 0.999995    # Decay rate for exploration
EPSILON_MIN = 0.01          # Minimum exploration rate

MAX_MATCH_DURATION = 2000   # Maximum number of frames per match

# Action Space Indices
ACTION_MOVE_UP = 0
ACTION_MOVE_DOWN = 1
ACTION_MOVE_LEFT = 2
ACTION_MOVE_RIGHT = 3
ACTION_ROTATE_LEFT = 4
ACTION_ROTATE_RIGHT = 5
ACTION_SHOOT = 6
ACTION_WAIT = 7

NUM_ACTIONS = 8  # Total number of actions

# Number of inputs
NUM_INPUTS = 10  # Total number of inputs

ENEMY_SHOOT_COOLDOWN = 180  # Enemy shoots every 3 seconds at 60 FPS
ENEMY_MOVE_PROBABILITY = 0.05  # Adjust as needed (e.g., 0.05 means 5% chance to move each frame)

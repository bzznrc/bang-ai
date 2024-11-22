##################################################
# CONSTANTS
##################################################

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
AI_FPS = 120              # Frames per second for the AI game (if applicable)

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
ROTATION_SPEED = 5         # Rotation angle in degrees per action
PROJECTILE_SPEED = 10      # Speed of projectiles (pixels per frame)
SHOOT_COOLDOWN = 30        # Frames until the player can shoot again
ALIGNMENT_TOLERANCE = 10

# Enemy Constants (Adjusted for Curriculum)
ENEMY_STATIONARY = True  # Initially stationary
ENEMY_SHOOTING = False   # Initially not shooting
ENEMY_MOVE_PROBABILITY = 0.0  # No movement initially
ENEMY_SHOOT_COOLDOWN = SHOOT_COOLDOWN * 3  # Adjusted for curriculum

# Curriculum Levels
STARTING_LEVEL = 1  # Starting level for the game (1, 2, or 3)
LEVEL_UP_GAMES = 2000  # Number of games after which the level increases
MAX_LEVEL = 3  # Maximum level achievable

# Obstacle Constants
NUM_OBSTACLES = 4         # Starting with 4 obstacles
MIN_SECTIONS = 2          # Minimum number of sections for obstacles
MAX_SECTIONS = 5          # Maximum number of sections for obstacles
SAFE_RADIUS = 100         # Minimum distance from players in pixels
SPAWN_Y_OFFSET = 80       # Adjust this value to control the range of vertical displacement

# Font
FONT_SIZE = 24             # Font size for text displays

# Model Constants
MODEL_SAVE_PREFIX = 'model/bang'     # Prefix tag for saving models
LOAD_PREVIOUS_MODEL = False     # Start training from a saved model if True

# Toggle for training plots
PLOT_TRAIN = False
USE_GPU = False
SHOW_GAME = False  # Set to True to display the game window during training

# Agent Hyperparameters
MAX_MEMORY = 100_000    # Maximum size of the replay memory
BATCH_SIZE = 1024       # Number of samples per training batch
LR = 0.0005             # Learning rate for the optimizer
HIDDEN_LAYERS = [128, 64, 32]

GAMMA = 0.95            # Discount factor for future rewards
L2_LAMBDA = 0.001       # Weight Decay / 0 = off
DROPOUT_RATE = 0.2      # Dropout Rate / 0 = off

# Epsilon Parameters for Exploration
EPSILON_START = 1         # Initial exploration rate
EPSILON_DECAY = 0.999995  # Decay rate for exploration
EPSILON_MIN = 0.05        # Minimum exploration rate

# Parameters for RL
MAX_MATCH_DURATION = 1000   # Maximum number of frames per match

# Reward Constants
WIN_REWARD = 100
LOSS_PENALTY = -100
TIME_PENALTY = -0.1              # Time penalty per frame
APPROACH_REWARD_FACTOR = 0.1     # Reward per unit distance closer to enemy
AIM_REWARD = 0.5                  # Reward for aligning aim towards enemy
SHOOT_REWARD = 1                  # Reward for shooting when enemy is in sight

# Action Space Indices
ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_SHOOT = 4
ACTION_WAIT = 5

NUM_ACTIONS = 6  # Total number of actions

# Number of inputs
NUM_INPUTS = 16  # Adjusted total number of inputs (with 8 obstacle distances)

# Activation Function
ACTIVATION_FUNCTION = 'ReLU'  # Use the best activation function
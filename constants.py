"""Configuration values for the Bang RL project."""

# Quick toggles
SHOW_GAME = True
PLOT_TRAINING = False
USE_GPU = False
LOAD_MODEL = "B" # Set to False, "B" (best), or "L" (last checkpoint)
RESUME_LEVEL = None
PLAY_OPPONENT_LEVEL = 3

# Runtime
FPS = 60
TRAINING_FPS = 0  # 0 lets pygame run unlocked for faster headless training

# Arena dimensions
GRID_WIDTH_TILES = 32
GRID_HEIGHT_TILES = 24
TILE_SIZE = 20
BOTTOM_BAR_HEIGHT = 30
BOTTOM_BAR_MARGIN = 20
SCREEN_WIDTH = GRID_WIDTH_TILES * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT_TILES * TILE_SIZE + BOTTOM_BAR_HEIGHT

# Colors (original palette)
COLOR_PLAYER = (30, 100, 100)
COLOR_PLAYER_OUTLINE = (50, 215, 200)
COLOR_ENEMY = (125, 45, 45)
COLOR_ENEMY_OUTLINE = (240, 95, 95)
COLOR_OBSTACLE_OUTLINE = (125, 125, 125)
COLOR_OBSTACLE_FILL = (45, 45, 45)
COLOR_PROJECTILE = (235, 195, 50)
COLOR_BACKGROUND = (45, 45, 45)
COLOR_SCORE = (255, 255, 255)

# Rendering
FONT_SIZE = 24

# Input/output spaces
INPUT_FEATURE_NAMES = [
    "enemy_distance",
    "enemy_relative_angle_sin",
    "enemy_relative_angle_cos",
    "delta_enemy_distance",
    "delta_enemy_relative_angle",
    "enemy_in_los",
    "nearest_projectile_distance",
    "nearest_projectile_relative_angle_sin",
    "nearest_projectile_relative_angle_cos",
    "delta_projectile_distance",
    "in_projectile_trajectory",
    "forward_blocked",
    "left_blocked",
    "right_blocked",
    "last_action_index",
    "time_since_last_shot",
]
ACTION_NAMES = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "shoot",
    "wait",
]
NUM_INPUT_FEATURES = len(INPUT_FEATURE_NAMES)
NUM_ACTIONS = len(ACTION_NAMES)

ACTION_MOVE_FORWARD = 0
ACTION_MOVE_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3
ACTION_SHOOT = 4
ACTION_WAIT = 5

# Gameplay tuning
PLAYER_MOVE_SPEED = 5
PLAYER_ROTATION_DEGREES = 5
PROJECTILE_SPEED = 10
SHOOT_COOLDOWN_FRAMES = 30
AIM_TOLERANCE_DEGREES = 10
MAX_EPISODE_STEPS = 1200
PLAYER_SPAWN_X_RATIO = 1 / 8
ENEMY_SPAWN_X_RATIO = 7 / 8

# Enemy behavior / curriculum
MIN_LEVEL = 1
STARTING_LEVEL = 1
LEVEL_UP_EVERY_GAMES = 1500
MAX_LEVEL = 3
DEFAULT_OBSTACLES = 8
LEVEL_SETTINGS = {
    1: {
        "num_obstacles": 4,
        "enemy_can_move": False,
        "enemy_shot_error_choices": [-20, -10, 0, 10, 20],
    },
    2: {
        "num_obstacles": 8,
        "enemy_can_move": True,
        "enemy_shot_error_choices": [-12, -6, 0, 6, 12],
    },
    3: {
        "num_obstacles": 12,
        "enemy_can_move": True,
        "enemy_shot_error_choices": [-8, -4, 0, 4, 8],
    },
}
ENEMY_MOVE_PROBABILITY_SCALE = 0.05
ENEMY_SHOOT_PROBABILITY = 0.10
SPAWN_Y_OFFSET = 80
SAFE_RADIUS = 100
MIN_OBSTACLE_SECTIONS = 2
MAX_OBSTACLE_SECTIONS = 5

# Collision / sensing
PROJECTILE_TRAJECTORY_DOT_THRESHOLD = 0.98
OBSTACLE_START_ATTEMPTS = 100
PROJECTILE_HITBOX_SIZE = 10
PROJECTILE_HITBOX_HALF = PROJECTILE_HITBOX_SIZE // 2
PROJECTILE_DISTANCE_MISSING = -1.0

# Model and training
MODEL_SUBDIR = "64_32"
MODEL_DIR = f"model/{MODEL_SUBDIR}"
MODEL_CHECKPOINT_PATH = f"{MODEL_DIR}/bang_dqn.pth"
MODEL_BEST_PATH = f"{MODEL_DIR}/bang_dqn_best.pth"

TOTAL_TRAINING_STEPS = 4_000_000
CHECKPOINT_EVERY_STEPS = 100_000

REPLAY_BUFFER_SIZE = 150_000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
GAMMA = 0.98
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = TOTAL_TRAINING_STEPS
PER_EPSILON = 1e-4

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_EPISODES = 1_000
EPSILON_LEVEL_UP_BUMP = 0.25
EPSILON_LEVEL_UP_MAX = EPSILON_START
EPSILON_LEVEL_UP_BUMP_DECAY = 0.85
TARGET_SYNC_EVERY = 500
GRAD_CLIP_NORM = 10.0

HIDDEN_DIMENSIONS = [64, 32]
# HIDDEN_DIMENSIONS = [128, 128, 64, 64]

# Replay-first training cadence
LEARN_START_STEPS = 5_000
TRAIN_EVERY_STEPS = 4
GRADIENT_STEPS_PER_UPDATE = 1

# Reward shaping
REWARD_WIN = 20.0
PENALTY_LOSE = -10.0
REWARD_HIT_ENEMY = 2.0
PENALTY_TIME_STEP = -0.005
PENALTY_BAD_SHOT = -0.1
PENALTY_BLOCKED_MOVE = -0.1

"""Central configuration for Bang AI."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeFlags:
    use_gpu: bool


@dataclass(frozen=True)
class BoardConfig:
    columns: int
    rows: int
    cell_size_px: int
    bottom_bar_height_px: int
    cell_inset_px: int

    @property
    def screen_width_px(self) -> int:
        return self.columns * self.cell_size_px

    @property
    def screen_height_px(self) -> int:
        return self.rows * self.cell_size_px + self.bottom_bar_height_px


FLAGS = RuntimeFlags(
    use_gpu=_env_flag("BANG_USE_GPU", False),
)

BOARD = BoardConfig(
    columns=32,
    rows=24,
    cell_size_px=20,
    bottom_bar_height_px=30,
    cell_inset_px=4,
)

# Quick toggles
SHOW_GAME_OVERRIDE: bool | None = None
USE_GPU = FLAGS.use_gpu
LOAD_MODEL = "B"  # False, "B" (best), or "L" (last)
RESUME_LEVEL = 3
PLAY_OPPONENT_LEVEL = 3


def resolve_show_game(default_value: bool) -> bool:
    if SHOW_GAME_OVERRIDE is None:
        return bool(default_value)
    return SHOW_GAME_OVERRIDE

# Runtime
FPS = 60
TRAINING_FPS = 0
WINDOW_TITLE = "Bang AI"

# Arena dimensions
GRID_WIDTH_TILES = BOARD.columns
GRID_HEIGHT_TILES = BOARD.rows
TILE_SIZE = BOARD.cell_size_px
BB_HEIGHT = BOARD.bottom_bar_height_px
SCREEN_WIDTH = BOARD.screen_width_px
SCREEN_HEIGHT = BOARD.screen_height_px
CELL_INSET = BOARD.cell_inset_px
CELL_INSET_DOUBLE = CELL_INSET * 2

# Rendering
FONT_FAMILY_DEFAULT: str | None = None
FONT_PATH_ROBOTO_REGULAR = "fonts/Roboto-Regular.ttf"
FONT_SIZE_BAR = 18
UI_STATUS_SEPARATOR = "   /   "

# Colors
COLOR_AQUA = (102, 212, 200)
COLOR_DEEP_TEAL = (38, 110, 105)
COLOR_CORAL = (244, 137, 120)
COLOR_BRICK_RED = (150, 62, 54)
COLOR_SLATE_GRAY = (97, 101, 107)
COLOR_FOG_GRAY = (230, 231, 235)
COLOR_CHARCOAL = (28, 30, 36)
COLOR_NEAR_BLACK = (18, 18, 22)
COLOR_SOFT_WHITE = (238, 238, 242)
COLOR_AMBER = (255, 224, 130)

# Input/output spaces
INPUT_FEATURE_NAMES = [
    "enemy_distance",
    "enemy_in_los",
    "enemy_relative_angle_sin",
    "enemy_relative_angle_cos",
    "delta_enemy_distance",
    "delta_enemy_relative_angle",
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
    "time_since_last_seen_enemy",
    "time_since_last_projectile_seen",
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
MODEL_INPUT_SIZE = NUM_INPUT_FEATURES
MODEL_OUTPUT_SIZE = NUM_ACTIONS

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
EVENT_TIMER_NORMALIZATION_FRAMES = MAX_EPISODE_STEPS
PLAYER_SPAWN_X_RATIO = 1 / 8
ENEMY_SPAWN_X_RATIO = 7 / 8

# Enemy behavior / curriculum
MIN_LEVEL = 1
STARTING_LEVEL = 1
MAX_LEVEL = 3
REWARD_ROLLING_WINDOW = 100
CURRICULUM_REWARD_THRESHOLDS = [8.0, 6.0]
CURRICULUM_CONSECUTIVE_CHECKS = 5
CURRICULUM_MIN_EPISODES_PER_LEVEL = 100
LEVEL_SETTINGS = {
    1: {
        "num_obstacles": 4,
        "enemy_move_probability": 0.00,
        "enemy_shot_error_choices": [-20, -10, 0, 10, 20],
        "enemy_shoot_probability": 0.05,
    },
    2: {
        "num_obstacles": 8,
        "enemy_move_probability": 0.20,
        "enemy_shot_error_choices": [-12, -6, 0, 6, 12],
        "enemy_shoot_probability": 0.05,
    },
    3: {
        "num_obstacles": 12,
        "enemy_move_probability": 0.20,
        "enemy_shot_error_choices": [-8, -4, 0, 4, 8],
        "enemy_shoot_probability": 0.10,
    },
}
ENEMY_STUCK_MOVE_ATTEMPTS = 2
ENEMY_ESCAPE_FOLLOW_FRAMES = 16
ENEMY_ESCAPE_ANGLE_OFFSETS_DEGREES = (90, -90, 180)
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
HIDDEN_DIMENSIONS = [64, 48]
MODEL_SUBDIR = "_".join(str(size) for size in HIDDEN_DIMENSIONS)
MODEL_DIR = PROJECT_ROOT / "model" / MODEL_SUBDIR
MODEL_NAME = f"bang_{MODEL_SUBDIR}"
MODEL_CHECKPOINT_PATH = str(MODEL_DIR / f"{MODEL_NAME}.pth")
MODEL_BEST_PATH = str(MODEL_DIR / f"{MODEL_NAME}_best.pth")
MODEL_SAVE_RETRIES = 5
MODEL_SAVE_RETRY_DELAY_SECONDS = 0.2

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

EPSILON_START_SCRATCH = 1.0
EPSILON_START_RESUME = 0.25
EPSILON_MIN = 0.05
EPSILON_DECAY_EPISODES = 1_000
EPSILON_STAGNATION_BOOST = 0.05
EPSILON_EXPLORATION_CAP = 0.5
EPSILON_LEVEL_UP_RESET = 0.25
STAGNATION_WINDOW = REWARD_ROLLING_WINDOW
PATIENCE = 25
STAGNATION_IMPROVEMENT_THRESHOLD = 0.05
EPISODE_CHECKPOINT_EVERY = 50
BEST_MODEL_MIN_EPISODES = REWARD_ROLLING_WINDOW
TARGET_SYNC_EVERY = 500
GRAD_CLIP_NORM = 10.0

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
REWARD_COMPONENTS = {
    "time_step": PENALTY_TIME_STEP,
    "bad_shot": PENALTY_BAD_SHOT,
    "blocked_move": PENALTY_BLOCKED_MOVE,
    "hit_enemy": REWARD_HIT_ENEMY,
    "win": REWARD_WIN,
    "lose": PENALTY_LOSE,
}

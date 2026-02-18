# Bang AI

## Overview
Minimal, local-only top-down arena reinforcement learning project using a Dueling DQN.

## Quickstart
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

## Run
```bash
python -m bang_ai
python -m bang_ai.play_user
python -m bang_ai.play_ai
python -m bang_ai.train_ai
```

## Project Layout
- `src/bang_ai/config.py`: central configuration and constants
- `src/bang_ai/game.py`: arena logic, rendering, and game modes
- `src/bang_ai/model.py`: Dueling DQN model and trainer
- `src/bang_ai/train_ai.py`: prioritized replay training loop
- `src/bang_ai/play_user.py`: human play entrypoint
- `src/bang_ai/play_ai.py`: greedy model play entrypoint
- `src/bang_ai/runtime.py`: arcade runtime and geometry helpers

## RL Inputs/Outputs
- State input size: `18`
- State feature 1: `enemy_distance`
- State feature 2: `enemy_in_los`
- State feature 3: `enemy_relative_angle_sin`
- State feature 4: `enemy_relative_angle_cos`
- State feature 5: `delta_enemy_distance`
- State feature 6: `delta_enemy_relative_angle`
- State feature 7: `nearest_projectile_distance`
- State feature 8: `nearest_projectile_relative_angle_sin`
- State feature 9: `nearest_projectile_relative_angle_cos`
- State feature 10: `delta_projectile_distance`
- State feature 11: `in_projectile_trajectory`
- State feature 12: `forward_blocked`
- State feature 13: `left_blocked`
- State feature 14: `right_blocked`
- State feature 15: `last_action_index`
- State feature 16: `time_since_last_shot`
- State feature 17: `time_since_last_seen_enemy`
- State feature 18: `time_since_last_projectile_seen`
- Action output size: `6` (one-hot)
- Action 1: `move_forward`
- Action 2: `move_backward`
- Action 3: `turn_left`
- Action 4: `turn_right`
- Action 5: `shoot`
- Action 6: `wait`
- Reward component: `time_step = -0.005`
- Reward component: `bad_shot = -0.1`
- Reward component: `blocked_move = -0.1`
- Reward component: `hit_enemy = +2.0`
- Reward component: `win = +20.0`
- Reward component: `lose = -10.0`
- Model input tensor shape: `(..., 18)`
- Model output tensor shape: `(..., 6)`

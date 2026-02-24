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

## Human Controls
- Move (absolute world frame): `W` up, `S` down, `A` left, `D` right
- Aim (sticky rotate): `Q`/`Left Arrow` = aim left, `E`/`Right Arrow` = aim right
- Shoot: left mouse click or `Space`

## Project Layout
- `src/bang_ai/config.py`: central configuration and constants
- `src/bang_ai/game.py`: arena logic, rendering, and game modes
- `src/bang_ai/model.py`: Dueling DQN model and trainer
- `src/bang_ai/train_ai.py`: prioritized replay training loop
- `src/bang_ai/play_user.py`: human play entrypoint
- `src/bang_ai/play_ai.py`: greedy model play entrypoint
- `src/bang_ai/runtime.py`: arcade runtime and geometry helpers

## RL Inputs/Outputs
- State input size: `24`
- State feature 1: `enemy_distance`
- State feature 2: `enemy_in_los`
- State feature 3: `enemy_relative_angle_sin`
- State feature 4: `enemy_relative_angle_cos`
- State feature 5: `delta_enemy_distance`
- State feature 6: `delta_enemy_relative_angle`
- State feature 7: `nearest_projectile_distance`
- State feature 8: `nearest_projectile_relative_angle_sin`
- State feature 9: `nearest_projectile_relative_angle_cos`
- State feature 10: `in_projectile_trajectory`
- State feature 11: `time_since_last_shot`
- State feature 12: `time_since_last_seen_enemy`
- State feature 13: `up_blocked`
- State feature 14: `down_blocked`
- State feature 15: `left_blocked`
- State feature 16: `right_blocked`
- State feature 17: `player_angle_sin`
- State feature 18: `player_angle_cos`
- State feature 19: `move_intent_x`
- State feature 20: `move_intent_y`
- State feature 21: `aim_intent`
- State feature 22: `ally_distance`
- State feature 23: `ally_relative_angle_sin`
- State feature 24: `ally_relative_angle_cos`
- Action output size: `8` (one-hot)
- Action 1: `move_up`
- Action 2: `move_down`
- Action 3: `move_left`
- Action 4: `move_right`
- Action 5: `stop_move`
- Action 6: `aim_left`
- Action 7: `aim_right`
- Action 8: `shoot`
- Reward component: `time_step = -0.005`
- Reward component: `blocked_move = -0.1`
- Reward component: `hit_enemy = +5.0`
- Reward component: `friendly_fire = -5.0`
- Reward component: `win = +20.0`
- Reward component: `lose = -10.0`
- Model input tensor shape: `(..., 24)`
- Model output tensor shape: `(..., 8)`

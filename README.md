# Bang AI - Reinforcement Learning TPS

## Overview
Bang AI is a compact top-down arena shooter used to train and evaluate a DQN-based agent. The project is intentionally small so you can trace the full loop from environment state to action selection to reward shaping.

The environment includes two agents, obstacles, projectiles, and level-based enemy behavior. The RL stack uses a dueling Double-DQN with target network synchronization and prioritized experience replay (PER).

## Run Instructions
Run `train_ai.py` to start training the AI model.

Run `play_ai.py` to evaluate a trained model.

Run `play_human.py` for the user-controlled version.

## Project Layout
- `game.py`: Core arena simulation and physics.
- `game_agent.py`: Player and enemy actor logic.
- `game_ai_env.py`: RL environment wrapper and reward pipeline.
- `rl_model.py`: Dueling Q-network and training utilities.
- `train_ai.py`: Training loop, replay sampling, curriculum, and checkpointing.
- `play_ai.py`: Inference/evaluation runner for trained checkpoints.
- `play_human.py`: Manual control gameplay loop.
- `ui.py`: Rendering and HUD.
- `constants.py`: Central configuration for gameplay and training.

## State Space (18 Inputs)
- enemy distance (normalized)
- enemy in line-of-sight (0/1)
- enemy relative angle (sin)
- enemy relative angle (cos)
- delta enemy distance
- delta enemy relative angle
- nearest projectile distance (normalized, or `-1` if none)
- nearest projectile relative angle (sin)
- nearest projectile relative angle (cos)
- delta projectile distance
- in projectile trajectory (0/1)
- forward blocked (0/1)
- left blocked (0/1)
- right blocked (0/1)
- last action index (normalized)
- time since last shot (normalized)
- time since last seen enemy (normalized [0, 1], resets to `0` when enemy is in LOS)
- time since last projectile seen (normalized [0, 1], resets to `0` when an enemy projectile enters perception)

## Action Space (6 Outputs)
- Move Forward
- Move Backward
- Turn Left
- Turn Right
- Shoot
- Wait

## Rewards
- win: `+20.0`
- lose: `-10.0`
- hit enemy: `+2.0`
- time step: `-0.005`
- bad shot (no LOS): `-0.1`
- blocked move: `-0.1`

## Important Constants
Key configuration are set in `constants.py`. Frequently tuned values include:

- `HIDDEN_DIMENSIONS`: hidden-layer sizes for the Q-network.
- `REPLAY_BUFFER_SIZE`: replay memory capacity.
- `BATCH_SIZE`: number of samples per optimization step.
- `LEARNING_RATE`: optimizer learning rate.
- `GAMMA`: reward discount factor.
- `LEVEL_SETTINGS`: per-level enemy behavior and obstacle density.
- `TRAIN_EVERY_STEPS`: environment steps between training updates.
- `TARGET_SYNC_EVERY`: target network sync interval.
- `EPSILON_*` values: exploration schedule and stagnation controls.
- `LOAD_MODEL`: load mode (`False`, `"B"`, `"L"`).

## Checkpoints
- `model/64_48/bang_dqn.pth`
- `model/64_48/bang_dqn_best.pth`

## Controls (Human Mode)
- `W`: move forward
- `S`: move backward
- `A`: turn left
- `D`: turn right
- `Space`: shoot

## Dependencies
- Python 3.10+
- Pygame
- PyTorch
- Matplotlib

## Notes
- Training runs headless by default. Toggle `SHOW_GAME` and `PLOT_TRAINING` in `constants.py` when needed.
- State and action definitions are in `constants.py` via `INPUT_FEATURE_NAMES` and `ACTION_NAMES`.

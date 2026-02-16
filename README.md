# Bang AI - Reinforcement Learning TPS

Bang AI is a compact top-down arena shooter used to train and evaluate a DQN-based agent. The project is intentionally small so you can trace the full loop from environment state to action selection to reward shaping.

**What this project contains**
- A lightweight 2D arena shooter with obstacles, projectiles, and two agents.
- A training environment that exposes an 18-feature state vector and reward shaping.
- A dueling Double-DQN with a target network and prioritized replay.
- Scripts to train, evaluate, and play manually.

**Project layout**
- `game.py` core arena simulation and physics
- `game_agent.py` player and enemy agent logic
- `game_ai_env.py` RL wrapper with state vector and reward shaping
- `rl_model.py` dueling network and trainer
- `train_ai.py` replay-buffer training loop and checkpoints
- `play_ai.py` run inference using a trained checkpoint
- `play_human.py` manual control for debugging and sanity checks
- `ui.py` rendering utilities
- `constants.py` all hyperparameters and tuning knobs

**State (18 inputs)**
- enemy distance (normalized)
- enemy in line-of-sight (0/1)
- enemy relative angle (sin)
- enemy relative angle (cos)
- Δ enemy distance
- Δ enemy relative angle
- nearest projectile distance (normalized, or -1 if none)
- nearest projectile relative angle (sin)
- nearest projectile relative angle (cos)
- Δ projectile distance
- in projectile trajectory (0/1)
- forward blocked (0/1)
- left blocked (0/1)
- right blocked (0/1)
- last action index (normalized)
- time since last shot (normalized)
- time since last seen enemy (normalized [0, 1], resets to `0` when `enemy_in_los == 1`)
- time since last projectile seen (normalized [0, 1], resets to `0` when an enemy projectile enters perception)

**Action space (6 outputs)**
- Move Forward
- Move Backward
- Turn Left
- Turn Right
- Shoot
- Wait

**Model and training**
The agent uses a dueling architecture with Double-DQN targets, a target network sync, and prioritized experience replay (PER). The network outputs Q-values for each action, and the replay buffer stabilizes learning.
Level progression is performance-based: promotion requires the rolling reward average to stay above per-level thresholds for consecutive checks, with a minimum episode count per level.
Exploration uses epsilon decay plus stagnation-triggered boosts and a level-up epsilon reset.

Key training knobs live in `constants.py`, including:
- `HIDDEN_DIMENSIONS`, `BATCH_SIZE`, `LEARNING_RATE`, `GAMMA`
- `REPLAY_BUFFER_SIZE`, `TARGET_SYNC_EVERY`, epsilon and curriculum values

**How to run**
1. Train a new model
```bash
python train_ai.py
```

Checkpoints:
- `model/64_32/bang_dqn.pth`
- `model/64_32/bang_dqn_best.pth`

2. Run the trained model
```bash
python play_ai.py
```

3. Play manually
```bash
python play_human.py
```

Controls:
- `W` move forward
- `S` move backward
- `A` turn left
- `D` turn right
- `Space` shoot

**Rewards (current)**
- win: `+10`
- lose: `-10`
- hit enemy: `+2`
- time step: `-0.02`
- bad shot (no LOS): `-0.1`
- blocked move: `-0.1`

**Notes**
- Training runs headless by default. Toggle `SHOW_GAME` or `PLOT_TRAINING` in `constants.py` as needed.
- Input/output definitions live in `constants.py` (`INPUT_FEATURE_NAMES`, `ACTION_NAMES`).

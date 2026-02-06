# Bang AI - Reinforcement Learning TPS

A compact third-person-shooter style arena for training and testing an RL bot.

## Naming approach

The project keeps `game_` prefixes for gameplay modules where it helps discovery, while RL-specific modules keep standard names:

- `game.py` → `BaseGame`
- `game_ai_env.py` → `TrainingGame`
- `play_human.py` → `HumanGame`
- `play_ai.py` → `GameModelRunner`
- `game_agent.py` → `Actor`
- `ui.py` → `Renderer`
- `rl_model.py` → `DuelingQNetwork`, `DQNTrainer`
- `train_ai.py` → `DQNAgent`

## Why DQN (dueling + target network) instead of a simple LinearQNet

A plain linear Q model can work for very simple environments, but this TPS setup has rotating aim, moving projectiles, and partial observability from compact inputs. A deeper DQN is a better fit because:

- it learns non-linear relationships between state features (for example, aim + obstacle context + projectile threat),
- it separates state-value and action-advantage estimates (dueling head), which often improves action ranking stability,
- it uses a target network and Double-DQN update to reduce overestimation and unstable learning.

The implementation is still intentionally compact so it remains understandable for RL learners.


### Training schedule details

- Epsilon now decays **per episode** (not per frame), which keeps exploration high for longer early training.
- Curriculum level-up is gated by both:
  - minimum episodes spent at the current level, and
  - rolling `Avg100` reward threshold for the next level.
- On level-up, epsilon gets a small floor boost to encourage brief re-exploration in the harder setting.

## State (16 inputs)

- 8 obstacle distance bins around the player
- enemy distance
- enemy relative angle
- nearest enemy projectile distance
- nearest enemy projectile relative angle
- player normalized X
- player normalized Y
- in projectile trajectory (0/1)
- enemy in line-of-sight (0/1)

## Action space (6 outputs)

- Move Forward
- Move Backward
- Turn Left
- Turn Right
- Shoot
- Wait

## What to launch

### 1) Train a new model

```bash
python train_ai.py
```

Saves periodic checkpoints to:

- `model/bang_dqn.pth`
- `model/bang_dqn_best.pth`

### 2) Try the current trained model

```bash
python play_ai.py
```

Runs greedy inference with the latest checkpoint and prints win rate.

### 3) Play as a human (manual test)

```bash
python play_human.py
```

Controls:

- `W`: move forward
- `S`: move backward
- `A`: turn left
- `D`: turn right
- `Space`: shoot

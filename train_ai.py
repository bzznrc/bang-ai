"""Main training loop and DQN agent implementation."""

import random
from collections import deque

import torch

from constants import *
from game_ai_env import TrainingGame
from rl_model import DQNTrainer, build_loaded_q_network, device
from utils import log_run_context


def plot_training(avg_rewards):
    import matplotlib.pyplot as plt

    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel(f"Avg Reward ({REWARD_ROLLING_WINDOW} Episodes)")
    plt.plot(avg_rewards, label="Avg Reward", color=tuple(c / 255 for c in COLOR_PLAYER_OUTLINE))
    plt.legend()
    plt.pause(0.01)


def resolve_model_load_path():
    if LOAD_MODEL is False:
        return None
    if LOAD_MODEL == "B":
        return MODEL_BEST_PATH
    if LOAD_MODEL == "L":
        return MODEL_CHECKPOINT_PATH
    raise ValueError('Invalid LOAD_MODEL value. Use False, "B", or "L".')


def try_save_model(model, path: str, success_message: str):
    try:
        model.save(path)
    except RuntimeError as exc:
        print(f"----- WARNING: save failed ({path}): {exc} -----")
        return False
    print(success_message)
    return True


class SumTree:
    """Binary sum tree to sample proportional to priority in O(log N)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity)
        self.data = [None] * capacity
        self.write = 0
        self.size = 0

    @property
    def total(self):
        return self.tree[1]

    def add(self, priority, data):
        # Store priority at leaf, then propagate the change up the tree.
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx, priority):
        # Update a leaf, then fix all affected parent sums.
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 1:
            idx //= 2
            self.tree[idx] += change

    def get(self, value):
        # Traverse the tree to find the leaf matching a cumulative sum value.
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """PER with proportional prioritization and importance sampling."""

    def __init__(self, capacity, alpha, beta_start, beta_frames, epsilon):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, beta_frames)
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.size

    def add(self, transition):
        # New samples get max priority so they are replayed at least once.
        priority = self.max_priority ** self.alpha
        return self.tree.add(priority, transition)

    def sample(self, batch_size):
        # Sample proportional to priority using stratified segments.
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total / batch_size
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            idx, priority, data = self.tree.get(value)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        sampling_probabilities = [p / self.tree.total for p in priorities]
        is_weights = [(self.tree.size * p) ** (-beta) for p in sampling_probabilities]
        max_weight = max(is_weights) if is_weights else 1.0
        is_weights = [w / max_weight for w in is_weights]
        return batch, indices, is_weights

    def update_priorities(self, indices, td_errors):
        # Update priorities from latest TD errors so sampling stays informative.
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class DQNAgent:
    """Replay-buffer DQN agent with target network synchronization."""

    def __init__(self):
        self.episodes_played = 0
        self.epsilon = 0.0
        self.epsilon_boost = 0.0
        self.epsilon_start = EPSILON_START_SCRATCH
        self.epsilon_decay_per_episode = (EPSILON_START_SCRATCH - EPSILON_MIN) / max(1, EPSILON_DECAY_EPISODES)
        self.stagnation_rewards = deque(maxlen=STAGNATION_WINDOW)
        self.best_stagnation_average = None
        self.stagnation_episodes = 0
        self.memory = PrioritizedReplayBuffer(
            capacity=REPLAY_BUFFER_SIZE,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES,
            epsilon=PER_EPSILON,
        )
        self.total_env_steps = 0

        load_path = resolve_model_load_path()
        self.requested_model_path = load_path
        self.online_model, self.loaded_model_path = build_loaded_q_network(load_path=load_path, strict=False)
        self.target_model = self.online_model.copy().to(device)
        self.target_model.eval()

        self.epsilon_start = EPSILON_START_RESUME if self.loaded_model_path else EPSILON_START_SCRATCH
        self.epsilon = self.epsilon_start

        self.trainer = DQNTrainer(self.online_model, self.target_model)
        self.training_steps = 0

    def get_state(self, game):
        return [float(value) for value in game.get_state_vector()]

    def remember(self, state, action, reward, next_state, done):
        return self.memory.add((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        batch, indices, is_weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        loss, td_errors = self._train_batch(states, actions, rewards, next_states, dones, is_weights=is_weights)
        self.memory.update_priorities(indices, td_errors)
        return loss

    def _train_batch(self, states, actions, rewards, next_states, dones, is_weights=None):
        loss, td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, is_weights=is_weights)
        self.training_steps += 1
        if self.training_steps % TARGET_SYNC_EVERY == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())
        return loss, td_errors

    def select_action(self, state):
        if random.random() < self.epsilon:
            action_idx = random.randint(0, NUM_ACTIONS - 1)
        else:
            with torch.no_grad():
                q_values = self.online_model(torch.tensor(state, dtype=torch.float32, device=device))
                action_idx = int(torch.argmax(q_values).item())

        action = [0] * NUM_ACTIONS
        action[action_idx] = 1
        return action

    def _base_epsilon(self):
        decayed = self.epsilon_start - self.episodes_played * self.epsilon_decay_per_episode
        return max(EPSILON_MIN, decayed)

    def update_epsilon(self):
        base_epsilon = self._base_epsilon()
        epsilon_upper_bound = max(self.epsilon_start, EPSILON_EXPLORATION_CAP)
        self.epsilon = min(epsilon_upper_bound, max(EPSILON_MIN, base_epsilon + self.epsilon_boost))
        if self.epsilon_boost > 0.0:
            self.epsilon_boost = max(0.0, self.epsilon_boost - self.epsilon_decay_per_episode)

    def _set_exploration_target(self, target_epsilon: float):
        target_epsilon = max(EPSILON_MIN, min(EPSILON_EXPLORATION_CAP, target_epsilon))
        base_epsilon = self._base_epsilon()
        self.epsilon_boost = max(0.0, target_epsilon - base_epsilon)
        self.epsilon = target_epsilon

    def apply_stagnation_boost(self):
        if self.epsilon >= EPSILON_EXPLORATION_CAP:
            return False
        self._set_exploration_target(self.epsilon + EPSILON_STAGNATION_BOOST)
        return True

    def reset_epsilon_for_level_up(self):
        self._set_exploration_target(EPSILON_LEVEL_UP_RESET)
        self.stagnation_rewards.clear()
        self.best_stagnation_average = None
        self.stagnation_episodes = 0

    def update_stagnation_state(self, episode_reward: float) -> bool:
        self.stagnation_rewards.append(episode_reward)
        if len(self.stagnation_rewards) < STAGNATION_WINDOW:
            return False

        moving_avg = sum(self.stagnation_rewards) / len(self.stagnation_rewards)
        if self.best_stagnation_average is None:
            self.best_stagnation_average = moving_avg
            self.stagnation_episodes = 0
            return False

        if moving_avg > self.best_stagnation_average + STAGNATION_IMPROVEMENT_THRESHOLD:
            self.best_stagnation_average = moving_avg
            self.stagnation_episodes = 0
            return False

        self.stagnation_episodes += 1
        if self.stagnation_episodes >= PATIENCE:
            boosted = self.apply_stagnation_boost()
            self.stagnation_episodes = 0
            self.best_stagnation_average = moving_avg
            return boosted
        return False


class PerformanceCurriculum:
    """Progress levels using rolling reward thresholds."""

    def __init__(self, level: int):
        self.level = level
        self.episodes_at_level = 0
        self.consecutive_passes = 0
        expected_transitions = max(0, MAX_LEVEL - MIN_LEVEL)
        if len(CURRICULUM_REWARD_THRESHOLDS) < expected_transitions:
            raise ValueError(
                f"CURRICULUM_REWARD_THRESHOLDS must define at least {expected_transitions} values."
            )

    def threshold_for_current_level(self):
        if self.level >= MAX_LEVEL:
            return None
        idx = self.level - MIN_LEVEL
        if idx < 0 or idx >= len(CURRICULUM_REWARD_THRESHOLDS):
            return None
        return CURRICULUM_REWARD_THRESHOLDS[idx]

    def on_episode_end(self, avg_reward: float, rolling_ready: bool) -> bool:
        self.episodes_at_level += 1
        threshold = self.threshold_for_current_level()
        if threshold is None:
            return False

        if not rolling_ready or self.episodes_at_level < CURRICULUM_MIN_EPISODES_PER_LEVEL:
            self.consecutive_passes = 0
            return False

        if avg_reward > threshold:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        if self.consecutive_passes < CURRICULUM_CONSECUTIVE_CHECKS:
            return False

        self.level += 1
        self.episodes_at_level = 0
        self.consecutive_passes = 0
        return True


def train():
    reward_window = deque(maxlen=REWARD_ROLLING_WINDOW)
    average_rewards = []

    agent = DQNAgent()
    level = RESUME_LEVEL if RESUME_LEVEL is not None else STARTING_LEVEL
    level = max(MIN_LEVEL, min(level, MAX_LEVEL))
    curriculum = PerformanceCurriculum(level=level)
    game = TrainingGame(level=curriculum.level)
    if agent.loaded_model_path:
        model_status = agent.loaded_model_path
    elif agent.requested_model_path:
        model_status = f"missing:{agent.requested_model_path}"
    else:
        model_status = "scratch"
    log_run_context(
        "train",
        {
            "model": model_status,
            "load_mode": LOAD_MODEL,
            "steps": TOTAL_TRAINING_STEPS,
            "eps": f"scratch:{EPSILON_START_SCRATCH} resume:{EPSILON_START_RESUME} -> {EPSILON_MIN}@{EPSILON_DECAY_EPISODES}ep",
            "stagnation": (
                f"w{STAGNATION_WINDOW}/p{PATIENCE}/d{STAGNATION_IMPROVEMENT_THRESHOLD}/"
                f"+{EPSILON_STAGNATION_BOOST}@{EPSILON_EXPLORATION_CAP}"
            ),
            "curriculum": (
                f"window{REWARD_ROLLING_WINDOW}/min{CURRICULUM_MIN_EPISODES_PER_LEVEL}/"
                f"checks{CURRICULUM_CONSECUTIVE_CHECKS}/th{CURRICULUM_REWARD_THRESHOLDS}"
            ),
            "batch": BATCH_SIZE,
            "buffer": REPLAY_BUFFER_SIZE,
            "train_every": TRAIN_EVERY_STEPS,
            "level": curriculum.level,
        },
    )

    best_average_reward = float("-inf")

    while True:
        agent.update_epsilon()
        episode_reward = 0.0
        episode_losses = []
        done = False

        while not done:
            state_old = agent.get_state(game)
            action = agent.select_action(state_old)
            reward, done, _ = game.play_step(action)
            state_new = agent.get_state(game)

            agent.remember(state_old, action, reward, state_new, done)
            episode_reward += reward
            agent.total_env_steps += 1
            if (
                agent.total_env_steps >= LEARN_START_STEPS
                and agent.total_env_steps % TRAIN_EVERY_STEPS == 0
            ):
                for _ in range(GRADIENT_STEPS_PER_UPDATE):
                    loss = agent.train_long_memory()
                    if loss > 0.0:
                        episode_losses.append(loss)

            if agent.total_env_steps % CHECKPOINT_EVERY_STEPS == 0:
                try_save_model(agent.online_model, MODEL_CHECKPOINT_PATH, "----- Model Saved (step checkpoint) -----")

        agent.episodes_played += 1
        mean_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0

        reward_window.append(episode_reward)
        avg_reward = sum(reward_window) / len(reward_window)
        average_rewards.append(avg_reward)
        exploration_boosted = agent.update_stagnation_state(episode_reward)
        if exploration_boosted:
            print(f"----- Exploration Bump: Epsilon -> {agent.epsilon:.3f} -----")

        print(
            f"Episode: {agent.episodes_played}\t"
            f"Level: {curriculum.level}\t"
            f"Frames: {game.frame_count}\t"
            f"Reward: {episode_reward:.2f}\t"
            f"Avg{REWARD_ROLLING_WINDOW}: {avg_reward:.2f}\t"
            f"Best: {best_average_reward:.2f}\t"
            f"Loss: {mean_loss:.4f}\t"
            f"Epsilon: {agent.epsilon:.3f}"
        )

        rolling_ready = len(reward_window) == REWARD_ROLLING_WINDOW
        if curriculum.on_episode_end(avg_reward, rolling_ready):
            game.level = curriculum.level
            game.configure_level()
            agent.reset_epsilon_for_level_up()
            print(f"----- LEVEL UP: {curriculum.level} -----")

        if agent.episodes_played % EPISODE_CHECKPOINT_EVERY == 0:
            try_save_model(agent.online_model, MODEL_CHECKPOINT_PATH, "----- Model Saved -----")

        if agent.episodes_played >= BEST_MODEL_MIN_EPISODES and avg_reward > best_average_reward:
            best_average_reward = avg_reward
            try_save_model(agent.online_model, MODEL_BEST_PATH, "----- New Best Model -----")

        if PLOT_TRAINING:
            plot_training(average_rewards)

        game.reset()

        if agent.total_env_steps >= TOTAL_TRAINING_STEPS:
            print("----- Training step limit reached -----")
            break


if __name__ == "__main__":
    train()

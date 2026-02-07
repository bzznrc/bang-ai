"""Main training loop and DQN agent implementation."""

import random
from collections import deque

import torch

from constants import *
from game_ai_env import TrainingGame
from rl_model import DQNTrainer, DuelingQNetwork, device


def plot_training(avg_rewards):
    import matplotlib.pyplot as plt

    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Reward (100 episodes)")
    plt.plot(avg_rewards, label="avg reward", color=tuple(c / 255 for c in COLOR_PLAYER_OUTLINE))
    plt.legend()
    plt.pause(0.01)


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
        self.epsilon = EPSILON_START
        self.memory = PrioritizedReplayBuffer(
            capacity=REPLAY_BUFFER_SIZE,
            alpha=PER_ALPHA,
            beta_start=PER_BETA_START,
            beta_frames=PER_BETA_FRAMES,
            epsilon=PER_EPSILON,
        )
        self.total_env_steps = 0
        self.plateau_cooldown = 0
        self.bump_remaining = 0

        self.online_model = DuelingQNetwork(NUM_INPUT_FEATURES, HIDDEN_DIMENSIONS, NUM_ACTIONS).to(device)
        self.target_model = self.online_model.copy().to(device)
        self.target_model.eval()

        load_path = None
        if LOAD_BEST_MODEL:
            load_path = MODEL_BEST_PATH
        elif LOAD_PREVIOUS_MODEL:
            load_path = MODEL_CHECKPOINT_PATH

        if load_path:
            self.online_model.load(load_path)
            self.target_model.load_state_dict(self.online_model.state_dict())
            print(f"Loaded model from {load_path}")

        self.trainer = DQNTrainer(self.online_model, self.target_model)
        self.training_steps = 0

    def get_state(self, game):
        return [float(value) for value in game.get_state_vector()]

    def remember(self, state, action, reward, next_state, done):
        return self.memory.add((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done, memory_idx):
        loss, td_errors = self._train_batch([state], [action], [reward], [next_state], [done], is_weights=[1.0])
        self.memory.update_priorities([memory_idx], td_errors)
        return loss

    def train_long_memory(self):
        if len(self.memory) == 0:
            return 0.0
        batch_size = min(BATCH_SIZE, len(self.memory))
        batch, indices, is_weights = self.memory.sample(batch_size)
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

    def update_epsilon(self):
        decay_steps = max(1, int(TOTAL_TRAINING_STEPS * EPSILON_DECAY_FRACTION))
        progress = min(1.0, self.total_env_steps / decay_steps)
        base_epsilon = EPSILON_START + progress * (EPSILON_MIN - EPSILON_START)
        base_epsilon = max(EPSILON_MIN, base_epsilon)

        if self.bump_remaining > 0:
            self.epsilon = max(base_epsilon, EPSILON_BUMP_VALUE)
        else:
            self.epsilon = base_epsilon


def train():
    reward_window = deque(maxlen=100)
    average_rewards = []
    action_window = deque(maxlen=100)

    agent = DQNAgent()
    level = RESUME_LEVEL if RESUME_LEVEL is not None else STARTING_LEVEL
    level = max(MIN_LEVEL, min(level, MAX_LEVEL))
    game = TrainingGame(level=level)

    best_average_reward = float("-inf")
    games_this_level = 0

    while True:
        episode_reward = 0.0
        episode_actions = [0] * NUM_ACTIONS
        reward_breakdown = {
            "time_step": 0.0,
            "bad_shot": 0.0,
            "blocked_move": 0.0,
            "hit_enemy": 0.0,
            "win": 0.0,
            "lose": 0.0,
        }
        done = False

        while not done:
            agent.update_epsilon()
            state_old = agent.get_state(game)
            action = agent.select_action(state_old)
            reward, done, step_breakdown = game.play_step(action)
            state_new = agent.get_state(game)

            memory_idx = agent.remember(state_old, action, reward, state_new, done)
            agent.train_short_memory(state_old, action, reward, state_new, done, memory_idx)
            episode_reward += reward
            action_idx = action.index(1) if 1 in action else 0
            episode_actions[action_idx] += 1
            for key in reward_breakdown:
                reward_breakdown[key] += step_breakdown.get(key, 0.0)
            agent.total_env_steps += 1

            if agent.total_env_steps % CHECKPOINT_EVERY_STEPS == 0:
                agent.online_model.save(MODEL_CHECKPOINT_PATH)
                print("----- Model Saved (step checkpoint) -----")

        agent.episodes_played += 1
        games_this_level += 1
        mean_loss = agent.train_long_memory()

        reward_window.append(episode_reward)
        avg_reward = sum(reward_window) / len(reward_window)
        average_rewards.append(avg_reward)

        print(
            f"Episode: {agent.episodes_played}\t"
            f"Level: {level}\t"
            f"Frames: {game.frame_count}\t"
             f"Reward: {episode_reward:.2f}\t"
            f"Avg100: {avg_reward:.2f}\t"
            f"Best: {best_average_reward:.2f}\t"
            f"Loss: {mean_loss:.4f}\t"
            f"Epsilon: {agent.epsilon:.3f}"
        )
        #print(
        #    "Reward breakdown: "
        #    + ", ".join(f"{key}={value:.2f}" for key, value in reward_breakdown.items())
        #)

        if games_this_level >= LEVEL_UP_EVERY_GAMES and level < MAX_LEVEL:
            level += 1
            game.level = level
            game.configure_level()
            games_this_level = 0
            agent.epsilon = min(EPSILON_LEVEL_UP_MAX, agent.epsilon + EPSILON_LEVEL_UP_BUMP)
            print(f"----- LEVEL UP: {level} -----")

        if agent.episodes_played % 50 == 0:
            agent.online_model.save(MODEL_CHECKPOINT_PATH)
            print("----- Model Saved -----")

        if agent.episodes_played >= 100 and avg_reward > best_average_reward:
            best_average_reward = avg_reward
            agent.online_model.save(MODEL_BEST_PATH)
            print("----- New Best Model -----")

        if agent.episodes_played >= EPSILON_PLATEAU_WINDOW_EPISODES:
            if avg_reward > best_average_reward + EPSILON_PLATEAU_MIN_IMPROVEMENT:
                best_average_reward = avg_reward
            else:
                if agent.plateau_cooldown == 0 and agent.bump_remaining == 0:
                    agent.bump_remaining = EPSILON_BUMP_DURATION_EPISODES
                    agent.plateau_cooldown = EPSILON_BUMP_COOLDOWN_EPISODES

        if agent.bump_remaining > 0:
            agent.bump_remaining -= 1
        if agent.plateau_cooldown > 0:
            agent.plateau_cooldown -= 1

        if PLOT_TRAINING:
            plot_training(average_rewards)

        game.reset()

        if agent.total_env_steps >= TOTAL_TRAINING_STEPS:
            print("----- Training step limit reached -----")
            break


if __name__ == "__main__":
    train()
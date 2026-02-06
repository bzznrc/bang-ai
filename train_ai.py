"""Main training loop and DQN agent implementation."""

import random
from collections import deque

import torch

from constants import (
    BATCH_SIZE,
    COLOR_PLAYER_OUTLINE,
    EPSILON_BOOST_ON_LEVEL_UP,
    EPSILON_DECAY_PER_EPISODE,
    EPSILON_MIN,
    EPSILON_START,
    HIDDEN_DIMENSIONS,
    LEVEL_UP_AVG_REWARD,
    LOAD_PREVIOUS_MODEL,
    MAX_LEVEL,
    MIN_EPISODES_PER_LEVEL,
    MODEL_BEST_PATH,
    MODEL_CHECKPOINT_PATH,
    NUM_ACTIONS,
    NUM_INPUT_FEATURES,
    PLOT_TRAINING,
    REPLAY_BUFFER_SIZE,
    STARTING_LEVEL,
    TARGET_SYNC_EVERY,
)
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


class DQNAgent:
    """Replay-buffer DQN agent with target network synchronization."""

    def __init__(self):
        self.episodes_played = 0
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)

        self.online_model = DuelingQNetwork(NUM_INPUT_FEATURES, HIDDEN_DIMENSIONS, NUM_ACTIONS).to(device)
        self.target_model = self.online_model.copy().to(device)
        self.target_model.eval()

        if LOAD_PREVIOUS_MODEL:
            self.online_model.load(MODEL_CHECKPOINT_PATH)
            self.target_model.load_state_dict(self.online_model.state_dict())
            print(f"Loaded model from {MODEL_CHECKPOINT_PATH}")

        self.trainer = DQNTrainer(self.online_model, self.target_model)
        self.training_steps = 0

    def get_state(self, game):
        return [float(value) for value in game.get_state_vector()]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self._train_batch([state], [action], [reward], [next_state], [done])

    def train_long_memory(self):
        if not self.memory:
            return 0.0
        batch = random.sample(self.memory, min(BATCH_SIZE, len(self.memory)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return self._train_batch(states, actions, rewards, next_states, dones)

    def _train_batch(self, states, actions, rewards, next_states, dones):
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        self.training_steps += 1
        if self.training_steps % TARGET_SYNC_EVERY == 0:
            self.target_model.load_state_dict(self.online_model.state_dict())
        return loss

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

    def on_episode_end(self):
        """Decay epsilon once per episode for smoother exploration scheduling."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY_PER_EPISODE)


def _should_level_up(level: int, games_this_level: int, avg_reward: float) -> bool:
    """Gate level-up by both experience and policy quality."""
    if level >= MAX_LEVEL:
        return False
    if games_this_level < MIN_EPISODES_PER_LEVEL:
        return False
    next_level = level + 1
    return avg_reward >= LEVEL_UP_AVG_REWARD.get(next_level, float("inf"))


def train():
    reward_window = deque(maxlen=100)
    average_rewards = []

    agent = DQNAgent()
    level = STARTING_LEVEL
    game = TrainingGame(level=level)

    best_average_reward = float("-inf")
    games_this_level = 0

    while True:
        episode_reward = 0.0
        done = False

        while not done:
            state_old = agent.get_state(game)
            action = agent.select_action(state_old)
            reward, done = game.play_step(action)
            state_new = agent.get_state(game)

            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            episode_reward += reward

        agent.episodes_played += 1
        games_this_level += 1
        mean_loss = agent.train_long_memory()
        agent.on_episode_end()

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

        if _should_level_up(level, games_this_level, avg_reward):
            level += 1
            game.level = level
            game.configure_level()
            games_this_level = 0
            agent.epsilon = max(agent.epsilon, EPSILON_BOOST_ON_LEVEL_UP)
            print(f"----- LEVEL UP: {level} (epsilon -> {agent.epsilon:.3f}) -----")

        if agent.episodes_played % 50 == 0:
            agent.online_model.save(MODEL_CHECKPOINT_PATH)
            print("----- Model Saved -----")

        if agent.episodes_played >= 100 and avg_reward > best_average_reward:
            best_average_reward = avg_reward
            agent.online_model.save(MODEL_BEST_PATH)
            print("----- New Best Model -----")

        if PLOT_TRAINING:
            plot_training(average_rewards)

        game.reset()


if __name__ == "__main__":
    train()

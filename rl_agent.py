##################################################
# RL_AGENT
##################################################

import torch
import random
import numpy as np
from collections import deque
from game_ai import GameAI
from rl_model import LinearQNet, QTrainer
import matplotlib.pyplot as plt
from constants import *
from utils import get_device

device = get_device()

plt.ion()  # Interactive mode on

def plot(avg_rewards):
    """Plot the average rewards over time."""
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Avg Reward (Last 100 Games)')
    plt.plot(avg_rewards, label='Avg. Reward', color=tuple([c / 255 for c in COLOR_LEFT_TEAM_OUTLINE]))
    plt.legend()
    plt.pause(0.1)

class RL_Agent:
    """RL_Agent that interacts with the environment and learns from it."""

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START  # Initial exploration rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(NUM_INPUTS, HIDDEN_LAYERS, NUM_ACTIONS).to(device)
        if LOAD_PREVIOUS_MODEL:
            self.model.load(f"{MODEL_SAVE_PREFIX}.pth")
            print(f"Loaded model from {MODEL_SAVE_PREFIX}.pth")
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA, l2_lambda = L2_LAMBDA)

    def get_state(self, game):
        """Get the current state representation from the game."""
        state = game.get_state()
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train the model using experiences from memory."""
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train the model using the latest experience."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Decide on an action based on the current state."""
        # Exploration vs Exploitation
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        if random.random() < self.epsilon:
            # Explore: random action
            move = random.randint(0, NUM_ACTIONS - 1)  # 8 possible actions
        else:
            # Exploit: choose the best action based on the model
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        # Convert move to one-hot encoding
        action = [0] * NUM_ACTIONS
        action[move] = 1
        return action
    
    def reset_epsilon(self):
        """Reset epsilon to the initial value."""
        self.epsilon = EPSILON_START / 4
        #print(f"Epsilon reset to {self.epsilon}")

def train():
    """Train the rl_agent."""
    total_rewards = deque(maxlen=100)
    avg_rewards = []
    rl_agent = RL_Agent()
    game = GameAI()
    best_avg_reward = float(1)  # Initialize best average reward

    while True:
        episode_reward = 0
        done = False
        # Initialize the bonuses list for the episode
        episode_bonuses = [0.0] * 6  # Assuming there are 6 bonuses

        while not done:
            # Get the current state
            state_old = rl_agent.get_state(game)

            # Get the action
            action = rl_agent.get_action(state_old)

            # Perform the action and get the new state
            reward, done, bonuses = game.play_step(action)
            episode_reward += reward

            # Accumulate bonuses
            episode_bonuses = [sum(x) for x in zip(episode_bonuses, bonuses)]

            # Get the new state after action
            state_new = rl_agent.get_state(game)

            # Train short memory
            rl_agent.train_short_memory(state_old, action, reward, state_new, done)

            # Remember the experience
            rl_agent.remember(state_old, action, reward, state_new, done)

        # At this point, the game is over
        rl_agent.n_games += 1
        loss = rl_agent.train_long_memory()

        # Record rewards
        total_rewards.append(episode_reward)
        avg_reward = sum(total_rewards) / len(total_rewards)
        avg_rewards.append(avg_reward)

        # Print training progress
        print(f'Game: {rl_agent.n_games}\t\tFrames: {game.frame_count}\t\tReward: {episode_reward:.2f}\t\tAvg L100: {avg_reward:.2f}\t\tEpsilon: {rl_agent.epsilon:.3f}')

        # Unpack the accumulated bonuses
        (same_action_bonus, proximity_bonus, dodge_bonus, cover_bonus, shoot_alignment_bonus, outcome_bonus) = episode_bonuses

        # Print bonuses, including outcome bonuses
        print(f'\033[90mSame Action: {same_action_bonus:.2f}\tProximity: {proximity_bonus:.2f}\tDodge: {dodge_bonus:.2f}\t\tCover: {cover_bonus:.2f}\t\tShoot Al.: {shoot_alignment_bonus:.2f}\tOutcome: {outcome_bonus:.2f}\033[0m')

        # Reset the game after printing
        game.reset()

        # Save the model every 10 games
        if rl_agent.n_games % 10 == 0:
            rl_agent.model.save(f"{MODEL_SAVE_PREFIX}.pth")

        # Save the model if a new best average reward is achieved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            rl_agent.model.save(f"{MODEL_SAVE_PREFIX}_top.pth")

        # Reset epsilon at specified intervals
        if rl_agent.n_games % 1000 == 0:
            rl_agent.reset_epsilon()

        # Plot the average rewards
        if PLOT_TRAIN:
            plot(avg_rewards)

if __name__ == '__main__':
    train()
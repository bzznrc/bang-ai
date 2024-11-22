##################################################
# RL_MODEL
##################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from constants import *
from utils import get_device

device = get_device()

class LinearQNet(nn.Module):
    """Neural network model for Q-learning."""
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if DROPOUT_RATE > 0:
                layers.append(nn.Dropout(DROPOUT_RATE))
            if ACTIVATION_FUNCTION == 'ReLU':
                layers.append(nn.ReLU())
            elif ACTIVATION_FUNCTION == 'LeakyReLU':
                layers.append(nn.LeakyReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        self.load_state_dict(torch.load(file_name, map_location=device))

class QTrainer:
    """Trainer class for the Q-network."""
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)

        if len(state.shape) == 1:
            # Reshape for single sample
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # Predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_next = self.model(next_state[idx])
                Q_new = reward[idx] + GAMMA * torch.max(Q_next)

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
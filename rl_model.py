"""Neural network and trainer utilities for DQN.

This module uses a dueling architecture plus a target network update step
(Double DQN). Compared with a simple linear Q model, this is usually more
stable in environments where action quality depends on multiple interacting
signals.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from constants import DROPOUT_RATE, GAMMA, GRAD_CLIP_NORM, LEARNING_RATE, WEIGHT_DECAY
from utils import get_device


device = get_device()


class DuelingQNetwork(nn.Module):
    """Compact Q-network with separate value and advantage heads."""

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        layers = []
        in_features = input_size
        for hidden in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(DROPOUT_RATE),
            ])
            in_features = hidden

        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_features, 1)
        self.advantage_head = nn.Linear(in_features, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_extractor(x)
        values = self.value_head(features)
        advantages = self.advantage_head(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)

    def copy(self):
        return copy.deepcopy(self)

    def save(self, file_name: str):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name: str):
        self.load_state_dict(torch.load(file_name, map_location=device))


class DQNTrainer:
    """Trains the online network with Double-DQN targets."""

    def __init__(self, online_model: DuelingQNetwork, target_model: DuelingQNetwork):
        self.online_model = online_model
        self.target_model = target_model
        self.optimizer = optim.AdamW(online_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.loss_fn = nn.SmoothL1Loss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=device)
        action = torch.as_tensor(action, dtype=torch.long, device=device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.bool, device=device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        action_indices = action.argmax(dim=1)
        current_q = self.online_model(state).gather(1, action_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.online_model(next_state).argmax(dim=1)
            next_q = self.target_model(next_state).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = reward + (~done).float() * GAMMA * next_q

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        return float(loss.item())

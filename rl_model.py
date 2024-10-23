import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from constants import USE_GPU
from utils import get_device

device = get_device()

class LinearQNet(nn.Module):
    """Feedforward Neural Network with customizable hidden layers."""

    def __init__(self, input_size, hidden_layers, output_size):
        super(LinearQNet, self).__init__()
        layers = []
        in_size = input_size

        # Dynamically create hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def save(self, file_name):
        """Save the model to a file."""
        model_folder_path = Path('./model')
        model_folder_path.mkdir(parents=True, exist_ok=True)
        file_path = model_folder_path / file_name
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_name):
        """Load the model from a file."""
        model_folder_path = Path('./model')
        file_path = model_folder_path / file_name
        if file_path.exists():
            self.load_state_dict(torch.load(file_path, map_location=device))
            self.to(device)  # Ensure the model is on the correct device
            self.eval()
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Starting from scratch.")

class QTrainer:
    """Trainer class for the Q-learning model."""

    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Perform one training step."""
        # Convert to tensors and move to device
        state = torch.tensor(np.array(state), dtype=torch.float).to(device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)  # Changed to torch.long
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)
        done = torch.tensor(np.array(done), dtype=torch.bool).to(device)

        # Reshape if necessary
        if len(state.shape) == 1:
            # Add batch dimension
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # 1. Predict Q values with current state
        pred = self.model(state)  # Shape: (batch_size, num_actions)

        # 2. Compute target Q values
        target = pred.clone().detach()  # Detach to prevent gradient computation on target

        # Compute predicted Q values for next state
        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_pred, _ = torch.max(next_pred, dim=1)

        # Compute Q_new
        Q_new = reward + (1 - done.float()) * self.gamma * max_next_pred

        # Get indices of actions taken
        action_indices = torch.argmax(action, dim=1)

        # Update target Q-values
        target[range(target.size(0)), action_indices] = Q_new

        # 3. Optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()
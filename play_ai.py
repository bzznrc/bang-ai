"""Run a trained model in the TPS arena for quick evaluation."""

import torch

from constants import HIDDEN_DIMENSIONS, MODEL_BEST_PATH, MODEL_CHECKPOINT_PATH, NUM_ACTIONS, NUM_INPUT_FEATURES, STARTING_LEVEL
from game_ai_env import TrainingGame
from rl_model import DuelingQNetwork, device


class GameModelRunner:
    """Loads a trained model and plays episodes greedily."""

    def __init__(self, model_path: str = MODEL_BEST_PATH):
        self.game = TrainingGame(level=STARTING_LEVEL)
        self.model = DuelingQNetwork(input_size=NUM_INPUT_FEATURES, hidden_sizes=HIDDEN_DIMENSIONS, output_size=NUM_ACTIONS).to(device)
        self.model.load(model_path)
        self.model.eval()

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32, device=device))
            action_idx = int(torch.argmax(q_values).item())
        action = [0] * NUM_ACTIONS
        action[action_idx] = 1
        return action

    def run(self, episodes: int = 10):
        wins = 0
        for _ in range(episodes):
            self.game.reset()
            done = False
            while not done:
                state = self.game.get_state_vector()
                action = self.select_action(state)
                _, done, _ = self.game.play_step(action)
            if not self.game.enemy.is_alive and self.game.player.is_alive:
                wins += 1
        print(f"Model win rate: {wins}/{episodes} ({(wins / episodes) * 100:.1f}%)")


if __name__ == "__main__":
    try:
        runner = GameModelRunner(MODEL_BEST_PATH)
    except FileNotFoundError:
        runner = GameModelRunner(MODEL_CHECKPOINT_PATH)
    runner.run()

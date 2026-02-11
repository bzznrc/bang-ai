"""Run a trained model in the TPS arena for quick evaluation."""

import torch

from constants import (
    LOAD_MODEL,
    MAX_LEVEL,
    MIN_LEVEL,
    MODEL_BEST_PATH,
    MODEL_CHECKPOINT_PATH,
    NUM_ACTIONS,
    PLAY_OPPONENT_LEVEL,
)
from game_ai_env import TrainingGame
from rl_model import build_loaded_q_network, device
from utils import log_run_context


class GameModelRunner:
    """Loads a trained model and plays episodes greedily."""

    def __init__(self, model_path: str = MODEL_BEST_PATH):
        self.level = max(MIN_LEVEL, min(PLAY_OPPONENT_LEVEL, MAX_LEVEL))
        self.model_path = model_path
        self.game = TrainingGame(level=self.level)
        self.model, _ = build_loaded_q_network(load_path=model_path, strict=True)
        self.model.eval()

    def select_action(self, state):
        with torch.no_grad():
            q_values = self.model(torch.tensor(state, dtype=torch.float32, device=device))
            action_idx = int(torch.argmax(q_values).item())
        action = [0] * NUM_ACTIONS
        action[action_idx] = 1
        return action

    def run(self, episodes: int = 10):
        log_run_context(
            "play-ai",
            {
                "episodes": episodes,
                "model": self.model_path,
                "load_mode": LOAD_MODEL,
                "level": self.level,
            },
        )
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

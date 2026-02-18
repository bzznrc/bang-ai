# Bang AI

Lightweight top-down arena RL project with a local-only codebase (no shared `bgds` dependency).

## Quickstart
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

Run:
```bash
python -m bang_ai
python -m bang_ai.train_ai
python -m bang_ai.play_ai
```

## Structure
- `src/bang_ai/config.py`: central config
- `src/bang_ai/core/`: game logic
- `src/bang_ai/ui/`: rendering
- `src/bang_ai/train/`: RL env, model, trainer
- `src/bang_ai/runtime/`: local runtime helpers (geometry, window, logging)

## Tests
```bash
pytest
```

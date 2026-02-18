"""Module entrypoint for `python -m bang_ai`."""

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bang_ai.play_human import run_human


if __name__ == "__main__":
    run_human()

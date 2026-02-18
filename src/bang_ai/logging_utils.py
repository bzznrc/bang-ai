"""Logging and runtime utility helpers."""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any


def configure_logging(level: str = "INFO") -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_torch_device(prefer_gpu: bool = False):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _format_context_value(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_mode_label(mode: str) -> str:
    words = mode.replace("-", " ").split()
    formatted: list[str] = []
    for word in words:
        formatted.append("AI" if word.lower() == "ai" else word.title())
    return " ".join(formatted)


def log_run_context(mode: str, context: dict[str, Any]) -> None:
    mode_label = _format_mode_label(mode)
    ordered = OrderedDict((key, value) for key, value in context.items() if value is not None)
    segments = [mode_label]
    segments.extend(
        f"{key.replace('_', ' ').title()}: {_format_context_value(value)}" for key, value in ordered.items()
    )
    logging.getLogger("bang_ai.run").info(" / ".join(segments))

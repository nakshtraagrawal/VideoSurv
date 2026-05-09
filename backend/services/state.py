"""In-memory system mode and operator preferences (prototype)."""

from __future__ import annotations

from typing import Dict

MODE = "harvest"
ACTIVE_MODEL = "hstforu"
THRESHOLDS: Dict[tuple[str, str], float] = {}
THRESHOLD_PERCENTILE = 95
SMOOTH_WINDOW = 15


def set_mode(m: str) -> None:
    global MODE
    MODE = m


def get_mode() -> str:
    return MODE


def set_active_model(name: str) -> None:
    global ACTIVE_MODEL
    ACTIVE_MODEL = name


def get_active_model() -> str:
    return ACTIVE_MODEL


def set_threshold(model_type: str, scene: str, value: float) -> None:
    THRESHOLDS[(model_type, scene)] = value


def get_threshold(model_type: str, scene: str, default: float = 0.015) -> float:
    return THRESHOLDS.get((model_type, scene), default)

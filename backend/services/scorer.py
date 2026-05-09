"""Anomaly scoring helpers (normalization, smoothing)."""
from __future__ import annotations

import numpy as np


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-8)


def smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    k = np.ones(window) / window
    return np.convolve(scores, k, mode="same")


def is_anomaly(raw_score: float, threshold: float) -> bool:
    return raw_score > threshold

"""Detection package providing change detection utilities for continual RL experiments."""

from .reward_trend import RewardTrendDetector
from .prediction_error import PredictionErrorDetector
from .latent_space import LatentSpaceDriftDetector
from .multi_modal import MultiModalDetector
from .multi_modal import WeightedMultiModalDetector
from .base import ChangeDetector, DetectionResult

__all__ = [
    "ChangeDetector",
    "DetectionResult",
    "RewardTrendDetector",
    "PredictionErrorDetector",
    "LatentSpaceDriftDetector",
    "MultiModalDetector",
    "WeightedMultiModalDetector",
]


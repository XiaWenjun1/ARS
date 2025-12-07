from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


@dataclasses.dataclass
class DetectionResult:
    detected: bool
    score: float
    metadata: Dict[str, Any]


class ChangeDetector(ABC):
    """Abstract base for streaming change detectors."""

    def __init__(self, name: str):
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""

    @abstractmethod
    def update(
        self,
        state,
        action,
        reward: float,
        next_state,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> DetectionResult:
        """Update detector with a new transition."""



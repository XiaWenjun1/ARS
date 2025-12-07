from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


@dataclasses.dataclass
class DetectionResult:
    """
    Represents the result of a change detection operation.

    Attributes:
        detected (bool): True if a change was detected, False otherwise.
        score (float): A numerical score indicating the strength or confidence of the detection.
                       Higher scores typically mean a stronger indication of change.
        metadata (Dict[str, Any]): A dictionary for any additional, detector-specific information
                                   relevant to the detection result (e.g., threshold, specific metrics).
    """
    detected: bool
    score: float
    metadata: Dict[str, Any]


class ChangeDetector(ABC):
    """
    Abstract base class for streaming change detectors.
    Subclasses must implement the `reset` and `update` methods to provide
    specific change detection logic. These detectors process environmental
    transitions (state, action, reward, next_state, done) in a streaming fashion
    to identify changes in the environment or agent's behavior.
    """

    def __init__(self, name: str):
        """
        Initializes the ChangeDetector.

        Args:
            name (str): A unique name for this detector instance.
        """
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the internal state of the detector.
        This method should be called when starting a new experiment or a new task
        to clear any accumulated history or statistics.
        """

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
        """
        Updates the detector with a new transition and checks for changes.

        Args:
            state: The current observation from the environment.
            action: The action taken by the agent.
            reward (float): The reward received from the environment.
            next_state: The next observation from the environment.
            done (bool): A boolean indicating if the episode has ended.
            info (Optional[Dict[str, Any]]): Additional information provided by the environment.

        Returns:
            DetectionResult: An object containing whether a change was detected,
                             a detection score, and any relevant metadata.
        """



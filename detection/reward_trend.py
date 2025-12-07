from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from .base import ChangeDetector, DetectionResult


class RewardTrendDetector(ChangeDetector):
    """
    Detects environment shifts by monitoring episode-level reward drift.
    It compares the mean of recent episode rewards to a historical baseline
    and triggers a detection if a significant drop or change is observed.

    Returns metadata with:
      - baseline_mean, recent_mean, diff (absolute), relative_diff
      - confidence: normalized in [0,1]
      - streak, cooldown
    """

    def __init__(
        self,
        window_size: int = 5,       # Number of most recent episodes to consider for current mean.
        baseline_window: int = 30,  # Number of episodes for the historical baseline mean.
        drop_threshold: float = 0.25, # Threshold for relative drop in reward to trigger detection.
        confirm_steps: int = 2,     # Number of consecutive detections needed to confirm a change.
        cooldown_episodes: int = 18, # Number of episodes to wait after a detection before detecting again.
    ):
        """
        Initializes the RewardTrendDetector.

        Args:
            window_size (int): The number of recent episodes to average for comparison.
            baseline_window (int): The number of episodes to average for the baseline.
            drop_threshold (float): The relative reward drop threshold to consider for detection.
            confirm_steps (int): Consecutive detections needed to confirm a change.
            cooldown_episodes (int): Cooldown period in episodes after a detection.
        """
        self.window_size = int(window_size)
        self.baseline_window = int(baseline_window)
        self.drop_threshold = float(drop_threshold)
        self.confirm_steps = int(confirm_steps)
        self.cooldown_episodes = int(cooldown_episodes)
        super().__init__(name="reward_trend")

    def reset(self) -> None:
        """
        Resets the detector's internal state, clearing the episode reward history,
        and resetting streak and cooldown counters.
        """
        # The maxlen ensures that the deque holds enough data for both baseline and recent windows.
        maxlen = self.baseline_window + self.window_size
        self.history: Deque[float] = deque(maxlen=maxlen) # Stores cumulative rewards per episode.
        self._streak = 0       # Counter for consecutive drift detections.
        self._cooldown = 0     # Cooldown counter to prevent immediate re-detection.
        self._episode_accum = 0.0 # Accumulates reward within the current episode.

    def _compute_confidence(self, baseline_mean: float, recent_mean: float, diff: float) -> float:
        """
        Normalizes the reward drop into a confidence score in the range [0, 1].
        Combines relative drop and absolute drop to determine confidence.

        Args:
            baseline_mean (float): The mean reward of the baseline window.
            recent_mean (float): The mean reward of the recent window.
            diff (float): The absolute difference between baseline_mean and recent_mean.

        Returns:
            float: A confidence score between 0.0 and 1.0.
        """
        # Calculate relative drop (how much recent_mean dropped compared to baseline_mean).
        if baseline_mean <= 0: # Avoid division by zero or negative baseline issues.
            rel_drop = 0.0
        else:
            rel_drop = (baseline_mean - recent_mean) / (abs(baseline_mean) + 1e-8) # Add epsilon for stability.
        # Scale relative drop to a confidence score (e.g., a 60% drop gives 1.0 confidence).
        rel_conf = float(np.clip(rel_drop / 0.6, 0.0, 1.0))

        # Scale absolute drop to a confidence score (e.g., a 40-unit drop gives 1.0 confidence).
        abs_conf = float(np.clip(diff / 40.0, 0.0, 1.0))

        # Use the maximum of relative and absolute confidence.
        conf = max(rel_conf, abs_conf)
        return conf

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        """
        Updates the detector with a new state transition, accumulates episode reward,
        and checks for reward trend shifts when an episode ends.

        Args:
            state: The current state observation.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done (bool): Whether the episode has terminated.
            info (Optional[Dict[str, Any]]): Additional information from the environment.

        Returns:
            DetectionResult: The result of the reward trend detection.
        """
        self._episode_accum += float(reward)

        # Reduce cooldown counter at episode boundaries.
        if done and self._cooldown > 0:
            self._cooldown -= 1

        detected = False
        score = 0.0
        metadata: Dict[str, float] = {}

        # If an episode has finished, add its cumulative reward to the history.
        if done:
            self.history.append(float(self._episode_accum))
            self._episode_accum = 0.0 # Reset for the next episode.

        # Perform drift detection only when enough data has been collected in the history buffer.
        if len(self.history) == self.history.maxlen:
            data = np.array(self.history, dtype=float)
            # Baseline data for comparison (older episode rewards).
            baseline = data[: self.baseline_window]
            # Recent data to check for trend shifts (newer episode rewards).
            recent = data[-self.window_size :]
            
            # Calculate the mean reward for the baseline and recent windows.
            baseline_mean = float(np.mean(baseline))
            recent_mean = float(np.mean(recent))
            
            # [Key Modification] Using absolute difference (Diff) instead of just "Drop".
            # This allows detection of significant positive or negative shifts.
            diff = abs(baseline_mean - recent_mean)
            score = float(diff) # The primary score is the absolute difference.

            metadata = {
                "baseline_mean": baseline_mean,
                "recent_mean": recent_mean,
                "diff": diff,
                "streak": int(self._streak),
                "cooldown": int(self._cooldown),
            }

            # [Key Modification] Fix for potential negative values in baseline_mean
            # Calculate relative difference (fractional change).
            if abs(baseline_mean) > 1e-8: # Add epsilon to prevent division by near-zero.
                rel_diff = diff / (abs(baseline_mean) + 1e-8)
            else:
                rel_diff = 0.0
            metadata["relative_diff"] = float(rel_diff)

            # [Key Modification] Trigger condition:
            # Detects if relative change > threshold (0.3) OR absolute difference > 40.0.
            cond = (rel_diff > self.drop_threshold) or (diff > 40.0)

            # Check for drift and manage the detection streak, respecting cooldown.
            if cond and self._cooldown <= 0:
                self._streak += 1
            else:
                self._streak = 0

            # If enough consecutive drifts are detected, signal a change and reset.
            if self._streak >= self.confirm_steps:
                detected = True
                self._streak = 0
                self._cooldown = self.cooldown_episodes # Start cooldown period.

            # Compute normalized confidence using the helper method.
            conf = self._compute_confidence(baseline_mean, recent_mean, diff)
            metadata["confidence"] = float(conf)

        return DetectionResult(detected=bool(detected), score=float(score), metadata=metadata)
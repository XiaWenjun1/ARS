from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from .base import ChangeDetector, DetectionResult


class RewardTrendDetector(ChangeDetector):
    """Detects environment shifts via episode-level reward drift.

    Returns metadata with:
      - baseline_mean, recent_mean, drop (absolute), relative_drop
      - confidence: normalized in [0,1]
      - streak, cooldown
    """

    def __init__(
        self,
        window_size: int = 5,
        baseline_window: int = 30,
        drop_threshold: float = 0.25,
        confirm_steps: int = 2,
        cooldown_episodes: int = 18,
    ):
        self.window_size = int(window_size)
        self.baseline_window = int(baseline_window)
        self.drop_threshold = float(drop_threshold)
        self.confirm_steps = int(confirm_steps)
        self.cooldown_episodes = int(cooldown_episodes)
        super().__init__(name="reward_trend")

    def reset(self) -> None:
        maxlen = self.baseline_window + self.window_size
        self.history: Deque[float] = deque(maxlen=maxlen)
        self._streak = 0
        self._cooldown = 0
        self._episode_accum = 0.0

    def _compute_confidence(self, baseline_mean: float, recent_mean: float, drop: float) -> float:
        """Normalize drop -> confidence in [0,1].

        Heuristics:
         - relative drop (baseline -> recent) mapped with saturate at 0.6 (60% drop -> conf=1)
         - absolute drop mapped with saturate at 40 points -> conf_abs
         - final confidence = max(rel_conf, abs_conf)
        """
        if baseline_mean <= 0:
            rel_drop = 0.0
        else:
            rel_drop = (baseline_mean - recent_mean) / (baseline_mean + 1e-8)
        rel_conf = float(np.clip(rel_drop / 0.6, 0.0, 1.0))

        abs_conf = float(np.clip(drop / 40.0, 0.0, 1.0))

        conf = max(rel_conf, abs_conf)
        return conf

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        self._episode_accum += float(reward)

        # reduce cooldown at episode boundaries
        if done and self._cooldown > 0:
            self._cooldown -= 1

        detected = False
        score = 0.0
        metadata: Dict[str, float] = {}

        if done:
            # append episode-level cumulative reward
            self.history.append(float(self._episode_accum))
            self._episode_accum = 0.0

        if len(self.history) == self.history.maxlen:
            data = np.array(self.history, dtype=float)
            baseline = data[: self.baseline_window]
            recent = data[-self.window_size :]
            baseline_mean = float(np.mean(baseline))
            recent_mean = float(np.mean(recent))
            drop = baseline_mean - recent_mean
            score = float(drop)

            metadata = {
                "baseline_mean": baseline_mean,
                "recent_mean": recent_mean,
                "drop": drop,
                "streak": int(self._streak),
                "cooldown": int(self._cooldown),
            }

            # relative drop
            if baseline_mean > 1e-8:
                rel_drop = (baseline_mean - recent_mean) / (baseline_mean + 1e-8)
            else:
                rel_drop = 0.0
            metadata["relative_drop"] = float(rel_drop)

            # detection condition: either relative drop passes threshold OR absolute drop large
            cond = (rel_drop > self.drop_threshold) or (drop > 20.0)

            if cond and self._cooldown <= 0:
                self._streak += 1
            else:
                self._streak = 0

            if self._streak >= self.confirm_steps:
                detected = True
                self._streak = 0
                self._cooldown = self.cooldown_episodes

            # compute normalized confidence and attach
            conf = self._compute_confidence(baseline_mean, recent_mean, drop)
            metadata["confidence"] = float(conf)

        return DetectionResult(detected=bool(detected), score=float(score), metadata=metadata)
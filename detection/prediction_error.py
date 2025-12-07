from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ChangeDetector, DetectionResult


class _TransitionPredictor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        # state: (B, state_dim), action: (B, action_dim) or (B,)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([state, action.float()], dim=-1)
        return self.net(x)


class PredictionErrorDetector(ChangeDetector):
    """Detects shifts by monitoring transition prediction error.

    Returns metadata including:
      - baseline_error, recent_error, ratio
      - confidence in [0,1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        window_size: int = 20,
        min_samples: int = 12,
        ratio_threshold: float = 2.4,
        confirm_steps: int = 3,
        cooldown_episodes: int = 20,
        ema_alpha: float = 0.01,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self.ratio_threshold = float(ratio_threshold)
        self.confirm_steps = int(confirm_steps)
        self.cooldown_episodes = int(cooldown_episodes)
        self.ema_alpha = float(ema_alpha)
        self.learning_rate = float(learning_rate)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(name="prediction_error")

    def reset(self) -> None:
        # simple online predictor
        self.model = _TransitionPredictor(self.state_dim, 1, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.recent_errors: Deque[float] = deque(maxlen=self.window_size)
        self.baseline_error: Optional[float] = None
        self._streak = 0
        self._cooldown = 0

    def _compute_confidence(self, ratio: float) -> float:
        """Map ratio -> confidence in [0,1]. ratio=1->0, ratio>=3->1 by default."""
        max_ratio = 3.0
        conf = float(np.clip((ratio - 1.0) / (max_ratio - 1.0), 0.0, 1.0))
        return conf

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        # cast tensors
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # action may be int
        action_tensor = torch.tensor([action], dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # train one step online
        self.model.train()
        try:
            prediction = self.model(state_tensor, action_tensor)
            loss = self.loss_fn(prediction, next_state_tensor)
        except Exception:
            # if shape mismatch, fallback gracefully
            loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        error = float(loss.detach().cpu().item())
        self.recent_errors.append(error)

        # initialize baseline after min_samples
        if self.baseline_error is None and len(self.recent_errors) >= self.min_samples:
            self.baseline_error = float(np.mean(self.recent_errors))

        if self.baseline_error is not None:
            # EMA update so baseline slowly follows long-term trend
            self.baseline_error = (1 - self.ema_alpha) * self.baseline_error + self.ema_alpha * error

        # reduce cooldown at episode boundaries
        if done and self._cooldown > 0:
            self._cooldown -= 1

        detected = False
        score = 0.0
        metadata: Dict[str, float] = {}

        if len(self.recent_errors) >= self.min_samples and self.baseline_error is not None:
            recent_mean = float(np.mean(self.recent_errors))
            
            ratio = recent_mean / (max(self.baseline_error, 1e-4))
            
            score = float(ratio)
            metadata = {
                "baseline_error": float(self.baseline_error),
                "recent_error": float(recent_mean),
                "ratio": float(ratio),
                "streak": int(self._streak),
                "cooldown": int(self._cooldown),
            }

            # confidence mapping
            conf = self._compute_confidence(ratio)
            metadata["confidence"] = float(conf)

            if ratio > self.ratio_threshold and self._cooldown <= 0:
                self._streak += 1
            else:
                self._streak = 0

            if self._streak >= self.confirm_steps:
                detected = True
                self._streak = 0
                self._cooldown = self.cooldown_episodes

        return DetectionResult(detected=bool(detected), score=float(score), metadata=metadata)
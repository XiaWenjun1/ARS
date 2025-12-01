from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ChangeDetector, DetectionResult


class _LatentAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon


class LatentSpaceDriftDetector(ChangeDetector):
    """Detects latent distribution drift via an online autoencoder.

    Returns metadata with:
      - drift (L2), baseline_norm, confidence in [0,1]
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 6,
        hidden_dim: int = 128,
        window_size: int = 20,       # [修改] 默认改小
        baseline_window: int = 40,   # [修改] 默认改小
        drift_threshold: float = 1.3,
        confirm_steps: int = 2,
        cooldown_episodes: int = 20,
        learning_rate: float = 5e-4,
        device: Optional[str] = None,
    ):
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.window_size = int(window_size)
        self.baseline_window = int(baseline_window)
        self.drift_threshold = float(drift_threshold)
        self.confirm_steps = int(confirm_steps)
        self.cooldown_episodes = int(cooldown_episodes)
        self.learning_rate = float(learning_rate)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(name="latent_space")

    def reset(self) -> None:
        self.model = _LatentAutoencoder(self.state_dim, self.latent_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        maxlen = self.baseline_window + self.window_size
        self.latent_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._streak = 0
        self._cooldown = 0

    def _compute_confidence(self, drift: float) -> float:
        """
        [关键修改] 动态置信度计算
        不再使用硬编码的 4.0，而是基于 drift_threshold。
        如果 drift 达到阈值的 2 倍，置信度即为 1.0。
        """
        reference = self.drift_threshold * 2.0
        # 避免除以 0
        if reference <= 1e-6:
            reference = 1.0
            
        conf = float(np.clip(drift / reference, 0.0, 1.0))
        return conf

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # online AE update
        self.model.train()
        latent, recon = self.model(state_tensor)
        loss = self.loss_fn(recon, state_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        latent_vec = latent.detach().cpu().numpy()[0]
        self.latent_history.append(latent_vec)

        # reduce cooldown at episode boundaries
        if done and self._cooldown > 0:
            self._cooldown -= 1

        detected = False
        score = 0.0
        metadata: Dict[str, float] = {}

        if len(self.latent_history) == self.latent_history.maxlen:
            data = np.stack(self.latent_history)
            baseline = data[: self.baseline_window]
            recent = data[-self.window_size :]
            baseline_mean = np.mean(baseline, axis=0)
            recent_mean = np.mean(recent, axis=0)
            
            # 计算漂移 (Drift)
            drift = float(np.linalg.norm(recent_mean - baseline_mean))
            score = float(drift)
            
            metadata = {
                "drift": float(drift),
                "baseline_norm": float(np.linalg.norm(baseline_mean)),
                "streak": int(self._streak),
                "cooldown": int(self._cooldown),
            }

            conf = self._compute_confidence(drift)
            metadata["confidence"] = float(conf)

            if drift > self.drift_threshold and self._cooldown <= 0:
                self._streak += 1
            else:
                self._streak = 0

            if self._streak >= self.confirm_steps:
                detected = True
                self._streak = 0
                self._cooldown = self.cooldown_episodes

        return DetectionResult(detected=bool(detected), score=float(score), metadata=metadata)
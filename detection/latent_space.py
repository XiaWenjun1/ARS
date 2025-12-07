from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ChangeDetector, DetectionResult


class _LatentAutoencoder(nn.Module):
    """
    A simple fully-connected Autoencoder used for dimensionality reduction.
    
    It compresses the input state into a lower-dimensional 'latent' representation
    and attempts to reconstruct the original input. This latent space captures
    the essential features of the state distribution.
    """
    def __init__(self, input_dim: int, latent_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        # Encoder: Compresses high-dim state -> low-dim latent vector
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder: Reconstructs latent vector -> original state
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        """
        Returns:
            latent: The compressed representation (used for drift detection).
            recon: The reconstructed state (used for training loss).
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon


class LatentSpaceDriftDetector(ChangeDetector):
    """
    Detects latent distribution drift via an online autoencoder.
    

[Image of autoencoder architecture]


    This detector trains an autoencoder on the fly. It monitors the statistical 
    distance between the current batch of latent vectors and a baseline batch 
    of latent vectors. If the latent representation of the state shifts significantly, 
    it indicates a change in the environment's dynamics.

    Returns metadata with:
      - drift (L2 distance), baseline_norm, confidence in [0,1]
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 6,
        hidden_dim: int = 128,
        window_size: int = 20,       # [Modified] Reduced default value for faster reaction
        baseline_window: int = 40,   # [Modified] Reduced default value
        drift_threshold: float = 1.3,
        confirm_steps: int = 2,
        cooldown_episodes: int = 20,
        learning_rate: float = 5e-4,
        device: Optional[str] = None,
    ):
        """
        Args:
            state_dim: Dimension of environment state.
            latent_dim: Dimension of the bottleneck layer.
            hidden_dim: Hidden layer size for AE.
            window_size: Number of recent samples to compare.
            baseline_window: Number of historical samples to compare against.
            drift_threshold: The L2 distance threshold to trigger detection.
            confirm_steps: Number of consecutive steps above threshold required to trigger.
            cooldown_episodes: Episodes to wait after a detection before detecting again.
        """
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
        """Initialize models, optimizers, and history buffers."""
        self.model = _LatentAutoencoder(self.state_dim, self.latent_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Buffer holds both baseline and recent history
        # [ ... baseline ... | ... recent ... ]
        # 
        maxlen = self.baseline_window + self.window_size
        self.latent_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        
        self._streak = 0
        self._cooldown = 0

    def _compute_confidence(self, drift: float) -> float:
        """
        [Key Modification] Dynamic confidence calculation.
        
        Instead of using a hardcoded scaling factor (e.g., 4.0), this is based 
        on the `drift_threshold`.
        Logic: If drift reaches 2x the threshold, confidence is saturated at 1.0.
        """
        reference = self.drift_threshold * 2.0
        # Avoid division by zero
        if reference <= 1e-6:
            reference = 1.0
            
        conf = float(np.clip(drift / reference, 0.0, 1.0))
        return conf

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # --- 1. Online Autoencoder Update ---
        # We train the AE on every step to minimize reconstruction error on the *current* data.
        self.model.train()
        latent, recon = self.model(state_tensor)
        loss = self.loss_fn(recon, state_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Store the latent representation (detached from graph)
        latent_vec = latent.detach().cpu().numpy()[0]
        self.latent_history.append(latent_vec)

        # --- 2. Cooldown Management ---
        # Reduce cooldown at episode boundaries (assume done=True marks end of episode)
        if done and self._cooldown > 0:
            self._cooldown -= 1

        detected = False
        score = 0.0
        metadata: Dict[str, float] = {}

        # --- 3. Drift Detection Logic ---
        # We only check for drift if we have enough history to fill both baseline and recent windows
        if len(self.latent_history) == self.latent_history.maxlen:
            # Stack deque into numpy array for slicing
            data = np.stack(self.latent_history)
            
            # Split data into baseline (older) and recent (newer)
            baseline = data[: self.baseline_window]
            recent = data[-self.window_size :]
            
            # Compute centroids (mean vectors)
            baseline_mean = np.mean(baseline, axis=0)
            recent_mean = np.mean(recent, axis=0)
            
            # Compute Drift: Euclidean distance (L2 norm) between centroids
            drift = float(np.linalg.norm(recent_mean - baseline_mean))
            score = float(drift)
            
            metadata = {
                "drift": float(drift),
                "baseline_norm": float(np.linalg.norm(baseline_mean)),
                "streak": int(self._streak),
                "cooldown": int(self._cooldown),
            }

            # Calculate confidence score
            conf = self._compute_confidence(drift)
            metadata["confidence"] = float(conf)

            # Check threshold and cooldown
            if drift > self.drift_threshold and self._cooldown <= 0:
                self._streak += 1
            else:
                self._streak = 0

            # Trigger detection only if threshold is exceeded for `confirm_steps` consecutive steps
            if self._streak >= self.confirm_steps:
                detected = True
                self._streak = 0
                self._cooldown = self.cooldown_episodes # Activate cooldown

        return DetectionResult(detected=bool(detected), score=float(score), metadata=metadata)
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ChangeDetector, DetectionResult


class _LatentAutoencoder(nn.Module):
    """
    A simple Autoencoder for learning a latent representation of the input state.
    It consists of an encoder that maps the input to a lower-dimensional latent space
    and a decoder that reconstructs the original input from the latent representation.
    """
    def __init__(self, input_dim: int, latent_dim: int = 6, hidden_dim: int = 128):
        """
        Initializes the Latent Autoencoder.

        Args:
            input_dim (int): The dimension of the input data (e.g., state dimension).
            latent_dim (int, optional): The dimension of the latent space. Defaults to 6.
            hidden_dim (int, optional): The number of neurons in the hidden layers. Defaults to 128.
        """
        super().__init__()
        # Encoder: maps input_dim to latent_dim through a hidden layer.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder: reconstructs input_dim from latent_dim through a hidden layer.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        """
        Defines the forward pass of the autoencoder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing:
                   - latent (torch.Tensor): The latent representation of the input.
                   - recon (torch.Tensor): The reconstructed input.
        """
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon


class LatentSpaceDriftDetector(ChangeDetector):
    """
    Detects environmental changes by monitoring drift in the latent space
    of an online-trained autoencoder. The autoencoder continuously learns
    a compressed representation of the environment's states. Drift is
    measured by comparing the mean of recent latent representations
    to a baseline mean.

    Returns metadata with:
      - drift (L2), baseline_norm, confidence in [0,1]
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 6,
        hidden_dim: int = 128,
        window_size: int = 20,       # Window size for recent latent space mean calculation
        baseline_window: int = 40,   # Window size for baseline latent space mean calculation
        drift_threshold: float = 1.3, # Threshold for detecting significant drift.
        confirm_steps: int = 2,      # Number of consecutive drift detections required to confirm a change.
        cooldown_episodes: int = 20, # Number of episodes to wait after a detection before detecting again.
        learning_rate: float = 5e-4, # Learning rate for the autoencoder.
        device: Optional[str] = None,
    ):
        """
        Initializes the LatentSpaceDriftDetector.

        Args:
            state_dim (int): Dimension of the observation space.
            latent_dim (int): Dimension of the autoencoder's latent space.
            hidden_dim (int): Number of neurons in the autoencoder's hidden layers.
            window_size (int): Size of the window for computing recent latent mean.
            baseline_window (int): Size of the window for computing baseline latent mean.
            drift_threshold (float): Threshold for drift magnitude to trigger a detection.
            confirm_steps (int): Consecutive detections needed to confirm a change.
            cooldown_episodes (int): Cooldown period in episodes after a detection.
            learning_rate (float): Learning rate for the autoencoder's optimizer.
            device (Optional[str]): Device to run the autoencoder on ('cuda' or 'cpu').
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
        """
        Resets the detector's internal state, including re-initializing the autoencoder
        and clearing the latent history. This is called at the start of a new experiment or task.
        """
        self.model = _LatentAutoencoder(self.state_dim, self.latent_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        # The maxlen ensures that the deque holds enough data for both baseline and recent windows.
        maxlen = self.baseline_window + self.window_size
        self.latent_history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._streak = 0       # Counter for consecutive drift detections.
        self._cooldown = 0     # Cooldown counter to prevent immediate re-detection.

    def _compute_confidence(self, drift: float) -> float:
        """
        Dynamically calculates the confidence score for a detected drift.
        The confidence is scaled such that if the drift reaches twice the drift_threshold,
        the confidence is 1.0.

        Args:
            drift (float): The calculated drift magnitude.

        Returns:
            float: A confidence score between 0.0 and 1.0.
        """
        reference = self.drift_threshold * 2.0
        # Avoid division by zero if drift_threshold is extremely small.
        if reference <= 1e-6:
            reference = 1.0
            
        conf = float(np.clip(drift / reference, 0.0, 1.0))
        return conf

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        """
        Updates the detector with a new state transition and checks for latent space drift.

        Args:
            state: The current state observation.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done (bool): Whether the episode has terminated.
            info (Optional[Dict[str, Any]]): Additional information from the environment.

        Returns:
            DetectionResult: The result of the drift detection.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Online Autoencoder Update: The autoencoder is continuously trained on incoming states.
        self.model.train()
        latent, recon = self.model(state_tensor)
        loss = self.loss_fn(recon, state_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Store the detached latent vector in history for drift calculation.
        latent_vec = latent.detach().cpu().numpy()[0]
        self.latent_history.append(latent_vec)

        # Reduce cooldown counter at episode boundaries.
        if done and self._cooldown > 0:
            self._cooldown -= 1

        detected = False
        score = 0.0
        metadata: Dict[str, float] = {}

        # Perform drift detection only when enough data has been collected in the history buffer.
        if len(self.latent_history) == self.latent_history.maxlen:
            data = np.stack(self.latent_history)
            # Baseline data for comparison (older entries).
            baseline = data[: self.baseline_window]
            # Recent data to check for drift (newer entries).
            recent = data[-self.window_size :]
            
            # Calculate the mean latent vector for the baseline and recent windows.
            baseline_mean = np.mean(baseline, axis=0)
            recent_mean = np.mean(recent, axis=0)
            
            # Calculate drift as the L2 norm (Euclidean distance) between the means.
            drift = float(np.linalg.norm(recent_mean - baseline_mean))
            score = float(drift)
            
            # Populate metadata for detailed analysis.
            metadata = {
                "drift": float(drift),
                "baseline_norm": float(np.linalg.norm(baseline_mean)),
                "streak": int(self._streak),
                "cooldown": int(self._cooldown),
            }

            # Compute and add confidence score to metadata.
            conf = self._compute_confidence(drift)
            metadata["confidence"] = float(conf)

            # Check for drift and manage the detection streak, respecting cooldown.
            if drift > self.drift_threshold and self._cooldown <= 0:
                self._streak += 1
            else:
                self._streak = 0

            # If enough consecutive drifts are detected, signal a change and reset.
            if self._streak >= self.confirm_steps:
                detected = True
                self._streak = 0
                self._cooldown = self.cooldown_episodes # Start cooldown period.

        return DetectionResult(detected=bool(detected), score=float(score), metadata=metadata)
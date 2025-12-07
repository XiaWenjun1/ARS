from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .base import ChangeDetector, DetectionResult


class _TransitionPredictor(nn.Module):
    """
    A neural network that predicts the next state given a current state and action.
    Used within the PredictionErrorDetector to monitor for changes in environment dynamics.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initializes the TransitionPredictor network.

        Args:
            state_dim (int): The dimension of the input state.
            action_dim (int): The dimension of the action.
            hidden_dim (int, optional): The number of neurons in the hidden layers. Defaults to 64.
        """
        super().__init__()
        # The network takes a concatenated state and action as input.
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim), # Outputs a prediction for the next state.
        )

    def forward(self, state, action):
        """
        Defines the forward pass of the transition predictor.

        Args:
            state (torch.Tensor): The current state tensor (batch_size, state_dim).
            action (torch.Tensor): The action tensor (batch_size,) or (batch_size, 1).

        Returns:
            torch.Tensor: The predicted next state tensor (batch_size, state_dim).
        """
        # Ensure action tensor has a compatible shape for concatenation.
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        # Concatenate state and action before passing through the network.
        x = torch.cat([state, action.float()], dim=-1)
        return self.net(x)


class PredictionErrorDetector(ChangeDetector):
    """
    Detects environmental changes by monitoring shifts in the prediction error
    of an online-trained transition predictor. If the environment dynamics change,
    the predictor's error for new transitions is expected to increase significantly
    compared to a historical baseline.

    Returns metadata including:
      - baseline_error, recent_error, ratio
      - confidence in [0,1]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        window_size: int = 20,       # Size of the window for computing recent prediction errors.
        min_samples: int = 12,       # Minimum samples required to establish a baseline error.
        ratio_threshold: float = 2.4, # Ratio of recent_error/baseline_error above which drift is suspected.
        confirm_steps: int = 3,      # Number of consecutive drift detections required to confirm a change.
        cooldown_episodes: int = 20, # Number of episodes to wait after a detection before detecting again.
        ema_alpha: float = 0.01,     # Alpha for Exponential Moving Average (EMA) of the baseline error.
        learning_rate: float = 1e-3, # Learning rate for the transition predictor.
        device: Optional[str] = None,
    ):
        """
        Initializes the PredictionErrorDetector.

        Args:
            state_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of neurons in the hidden layers of the predictor network.
            window_size (int): The number of recent prediction errors to consider.
            min_samples (int): The minimum number of samples needed to initialize the baseline.
            ratio_threshold (float): The threshold for the ratio of recent error to baseline error.
            confirm_steps (int): Consecutive detections needed to confirm a change.
            cooldown_episodes (int): Cooldown period in episodes after a detection.
            ema_alpha (float): Smoothing factor for the Exponential Moving Average of the baseline error.
            learning_rate (float): Learning rate for the transition predictor's optimizer.
            device (Optional[str]): Device to run the predictor on ('cuda' or 'cpu').
        """
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
        """
        Resets the detector's internal state, including re-initializing the transition predictor,
        clearing recent errors, and resetting the baseline error and detection streak/cooldown.
        """
        # Initialize a simple online transition predictor.
        # Note: action_dim for _TransitionPredictor refers to the dimension of the action representation,
        # which is 1 for discrete actions.
        self.model = _TransitionPredictor(self.state_dim, 1, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.recent_errors: Deque[float] = deque(maxlen=self.window_size)
        self.baseline_error: Optional[float] = None # Stores the EMA of prediction errors.
        self._streak = 0 # Counter for consecutive drift detections.
        self._cooldown = 0 # Cooldown counter to prevent immediate re-detection.

    def _compute_confidence(self, ratio: float) -> float:
        """
        Maps the prediction error ratio to a confidence score in the range [0, 1].
        A ratio of 1 maps to 0 confidence, while a ratio of 3 or more maps to 1 confidence by default.

        Args:
            ratio (float): The ratio of recent prediction error to baseline error.

        Returns:
            float: A confidence score between 0.0 and 1.0.
        """
        max_ratio = 3.0 # The ratio at which confidence reaches 1.0.
        conf = float(np.clip((ratio - 1.0) / (max_ratio - 1.0), 0.0, 1.0))
        return conf

    def update(self, state, action, reward, next_state, done, info=None) -> DetectionResult:
        """
        Updates the detector with a new state transition, trains the predictor,
        calculates prediction error, and checks for environmental changes.

        Args:
            state: The current state observation.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            done (bool): Whether the episode has terminated.
            info (Optional[Dict[str, Any]]): Additional information from the environment.

        Returns:
            DetectionResult: The result of the prediction error detection.
        """
        # Cast inputs to tensors for the predictor.
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Train the transition predictor online with the current transition.
        self.model.train()
        try:
            prediction = self.model(state_tensor, action_tensor)
            loss = self.loss_fn(prediction, next_state_tensor) # MSE between predicted and actual next state.
        except Exception:
            # Fallback gracefully if there's a shape mismatch or other issue during prediction.
            loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Record the prediction error.
        error = float(loss.detach().cpu().item())
        self.recent_errors.append(error)

        # Initialize the baseline error after a minimum number of samples have been collected.
        if self.baseline_error is None and len(self.recent_errors) >= self.min_samples:
            self.baseline_error = float(np.mean(self.recent_errors))

        # Update the baseline error using Exponential Moving Average (EMA).
        # This allows the baseline to slowly adapt to long-term trends while still being sensitive to sudden changes.
        if self.baseline_error is not None:
            self.baseline_error = (1 - self.ema_alpha) * self.baseline_error + self.ema_alpha * error

        # Reduce cooldown counter at episode boundaries.
        if done and self._cooldown > 0:
            self._cooldown -= 1

        detected = False
        score = 0.0
        metadata: Dict[str, float] = {}

        # Perform detection if enough samples are present and baseline is established.
        if len(self.recent_errors) >= self.min_samples and self.baseline_error is not None:
            recent_mean = float(np.mean(self.recent_errors))
            
            # Calculate the ratio of recent error to baseline error.
            # The denominator has a floor (1e-4) to prevent division by zero or very small numbers,
            # especially relevant for environments like MountainCar with tiny errors.
            ratio = recent_mean / (max(self.baseline_error, 1e-4))
            
            score = float(ratio)
            metadata = {
                "baseline_error": float(self.baseline_error),
                "recent_error": float(recent_mean),
                "ratio": float(ratio),
                "streak": int(self._streak),
                "cooldown": int(self._cooldown),
            }

            # Map the ratio to a confidence score.
            conf = self._compute_confidence(ratio)
            metadata["confidence"] = float(conf)

            # Check for drift and manage the detection streak, respecting cooldown.
            if ratio > self.ratio_threshold and self._cooldown <= 0:
                self._streak += 1
            else:
                self._streak = 0

            # If enough consecutive drifts are detected, signal a change and reset.
            if self._streak >= self.confirm_steps:
                detected = True
                self._streak = 0
                self._cooldown = self.cooldown_episodes # Start cooldown period.

        return DetectionResult(detected=bool(detected), score=float(score), metadata=metadata)
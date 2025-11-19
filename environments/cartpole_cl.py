import gymnasium as gym
import numpy as np
from .base_wrapper import ContinualLearningWrapper

class CartPoleCL(ContinualLearningWrapper):
    # Constants for pole length and wind effects
    DEFAULT_POLE_LENGTH = 0.5
    WIND_POSITION_SCALE = 0.2
    WIND_VELOCITY_SCALE = 0.05

    def __init__(self, task_params, render_mode=None):
        super().__init__('CartPole-v1', task_params, render_mode)

    def _apply_task_parameters(self):
        """Apply CartPole specific parameter modifications"""
        task_config = self.task_params[self.current_task]

        # Get pole length from config
        pole_length = task_config.get('pole_length', self.DEFAULT_POLE_LENGTH)

        # Directly modify pole length (use unwrapped to access base environment)
        env = self.env.unwrapped
        if hasattr(env, 'length'):
            env.length = pole_length

        # Store wind force parameter
        self.wind_force = task_config.get('wind_force', 0.0)

    def step(self, action):
        # Apply wind effects: wind can stochastically affect actions
        if self.wind_force > 0:
            if np.random.random() < min(0.25, self.wind_force * 0.05):
                action = 1 - action

        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.wind_force > 0:
            modified_obs = np.array(observation, dtype=np.float32)
            modified_obs[0] += np.random.normal(0, self.wind_force * self.WIND_POSITION_SCALE)
            modified_obs[1] += np.random.normal(0, self.wind_force * self.WIND_VELOCITY_SCALE)
            observation = modified_obs
        else:
            observation = np.array(observation, dtype=np.float32)

        return observation, reward, terminated, truncated, info

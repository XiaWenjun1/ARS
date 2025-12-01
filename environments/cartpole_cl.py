import gymnasium as gym
import numpy as np
from .base_wrapper import ContinualLearningWrapper

class CartPoleCL(ContinualLearningWrapper):
    # Constants
    DEFAULT_POLE_LENGTH = 0.5
    DEFAULT_MASSCART = 1.0
    WIND_POSITION_SCALE = 0.2
    WIND_VELOCITY_SCALE = 0.05

    def __init__(self, task_params, render_mode=None):
        super().__init__('CartPole-v1', task_params, render_mode)

    def _apply_task_parameters(self):
        """Apply CartPole specific parameter modifications"""
        task_config = self.task_params[self.current_task]
        env = self.env.unwrapped

        # 1. 修改杆长 (Pole Length)
        if hasattr(env, 'length'):
            env.length = task_config.get('pole_length', self.DEFAULT_POLE_LENGTH)

        # 2. [新增] 修改小车质量 (Cart Mass) - 制造惯性差异的关键
        if hasattr(env, 'masscart'):
            env.masscart = task_config.get('masscart', self.DEFAULT_MASSCART)
            # 重新计算总质量，确保物理引擎同步
            if hasattr(env, 'total_mass') and hasattr(env, 'masspole'):
                env.total_mass = env.masspole + env.masscart

        # 3. 设置风力 (Wind Force) - 您的核心逻辑
        self.wind_force = task_config.get('wind_force', 0.0)

    def step(self, action):
        # Apply wind effects: wind can stochastically affect actions
        if self.wind_force > 0:
            # 风力越大，动作被扰动的概率越大
            if np.random.random() < min(0.25, self.wind_force * 0.05):
                action = 1 - action

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Apply wind noise to observation
        if self.wind_force > 0:
            modified_obs = np.array(observation, dtype=np.float32)
            # 位置和速度都会受到风力噪声的影响
            modified_obs[0] += np.random.normal(0, self.wind_force * self.WIND_POSITION_SCALE)
            modified_obs[1] += np.random.normal(0, self.wind_force * self.WIND_VELOCITY_SCALE)
            observation = modified_obs
        else:
            observation = np.array(observation, dtype=np.float32)

        return observation, reward, terminated, truncated, info
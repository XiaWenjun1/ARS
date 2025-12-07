import gymnasium as gym
import numpy as np
from .base_wrapper import ContinualLearningWrapper

class CartPoleCL(ContinualLearningWrapper):
    """
    CartPole Environment Wrapper for Continual Learning.
    
    This wrapper modifies the standard CartPole-v1 environment to simulate different 
    tasks by altering physical properties (pole length, cart mass) and introducing 
    external disturbances (wind) that affect both dynamics and observations.
    """
    
    # Constants
    DEFAULT_POLE_LENGTH = 0.5
    DEFAULT_MASSCART = 1.0
    
    # Scaling factors for wind noise injection
    WIND_POSITION_SCALE = 0.2
    WIND_VELOCITY_SCALE = 0.05

    def __init__(self, task_params, render_mode=None):
        super().__init__('CartPole-v1', task_params, render_mode)

    def _apply_task_parameters(self):
        """
        Apply CartPole specific parameter modifications based on the current task configuration.
        """
        task_config = self.task_params[self.current_task]
        env = self.env.unwrapped

        # 1. Modify Pole Length
        # Changing the length affects the angular physics (moment of inertia).
        # Shorter poles fall faster; longer poles fall slower but are harder to recover.
        if hasattr(env, 'length'):
            env.length = task_config.get('pole_length', self.DEFAULT_POLE_LENGTH)

        # 2. [New] Modify Cart Mass
        # This is key to creating inertia differences. 
        # A heavier cart requires more force (steps) to accelerate and decelerate.
        if hasattr(env, 'masscart'):
            env.masscart = task_config.get('masscart', self.DEFAULT_MASSCART)
            
            # Recalculate total mass to ensure physics engine synchronization
            # The Gym environment uses total_mass for acceleration calculations.
            if hasattr(env, 'total_mass') and hasattr(env, 'masspole'):
                env.total_mass = env.masspole + env.masscart

        # 3. Set Wind Force
        # This is the core logic for environmental disturbance.
        self.wind_force = task_config.get('wind_force', 0.0)

    def step(self, action):
        """
        Perform a step in the environment with potential wind interference.
        """
        # Apply wind effects: wind can stochastically affect actions (Action Noise)
        if self.wind_force > 0:
            # The stronger the wind, the higher the probability of action perturbation.
            # Capped at 25% probability to prevent the task from becoming impossible.
            if np.random.random() < min(0.25, self.wind_force * 0.05):
                action = 1 - action  # Flip the action (0->1 or 1->0)

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Apply wind noise to observation (Observation Noise / Sensor Noise)
        if self.wind_force > 0:
            modified_obs = np.array(observation, dtype=np.float32)
            
            # Position and velocity are both affected by wind noise.
            # We add Gaussian noise scaled by the wind force intensity.
            modified_obs[0] += np.random.normal(0, self.wind_force * self.WIND_POSITION_SCALE)
            modified_obs[1] += np.random.normal(0, self.wind_force * self.WIND_VELOCITY_SCALE)
            observation = modified_obs
        else:
            observation = np.array(observation, dtype=np.float32)

        return observation, reward, terminated, truncated, info
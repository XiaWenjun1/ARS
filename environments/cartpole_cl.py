import gymnasium as gym
import numpy as np
from .base_wrapper import ContinualLearningWrapper

class CartPoleCL(ContinualLearningWrapper):
    """
    A specialized ContinualLearningWrapper for the 'CartPole-v1' environment.
    This wrapper modifies specific physical parameters of the CartPole environment,
    such as pole length, cart mass, and introduces external 'wind' forces,
    to create distinct tasks for continual learning experiments.
    """
    # Constants for default values and wind effect scaling.
    DEFAULT_POLE_LENGTH = 0.5
    DEFAULT_MASSCART = 1.0
    WIND_POSITION_SCALE = 0.2  # Scales the magnitude of wind noise applied to position.
    WIND_VELOCITY_SCALE = 0.05 # Scales the magnitude of wind noise applied to velocity.

    def __init__(self, task_params, render_mode=None):
        """
        Initializes the CartPoleCL environment wrapper.

        Args:
            task_params (list): A list of dictionaries, each defining specific
                                parameters for a CartPole CL task (e.g., pole_length, masscart, wind_force).
            render_mode (str, optional): The render mode for the environment. Defaults to None.
        """
        super().__init__('CartPole-v1', task_params, render_mode)

    def _apply_task_parameters(self):
        """
        Applies CartPole-specific parameter modifications based on the current task configuration.
        This method accesses the underlying Gym environment's unwrapped properties to change them.
        """
        task_config = self.task_params[self.current_task]
        env = self.env.unwrapped

        # 1. Modify Pole Length: Directly changes the 'length' attribute of the pole.
        if hasattr(env, 'length'):
            env.length = task_config.get('pole_length', self.DEFAULT_POLE_LENGTH)

        # 2. [New] Modify Cart Mass: A key factor in creating inertia differences.
        if hasattr(env, 'masscart'):
            env.masscart = task_config.get('masscart', self.DEFAULT_MASSCART)
            # Recalculate total mass to ensure the physics engine is synchronized with the new mass.
            if hasattr(env, 'total_mass') and hasattr(env, 'masspole'):
                env.total_mass = env.masspole + env.masscart

        # 3. Set Wind Force: This is a core logic for introducing external disturbances.
        # This value determines the intensity of the stochastic wind effect.
        self.wind_force = task_config.get('wind_force', 0.0)

    def step(self, action):
        """
        Takes a step in the environment, applying wind effects to actions and observations,
        then returning the modified results.

        Args:
            action (int): The action to take in the environment.

        Returns:
            tuple: (observation, reward, terminated, truncated, info) after applying wind effects.
        """
        # Apply wind effects: wind can stochastically affect actions.
        if self.wind_force > 0:
            # The stronger the wind force, the higher the probability of the action being perturbed.
            # The probability is capped at 0.25 to prevent extreme action flipping.
            if np.random.random() < min(0.25, self.wind_force * 0.05):
                action = 1 - action # Flip the action (e.g., if action is 0, becomes 1, and vice-versa).

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Apply wind noise to observation.
        if self.wind_force > 0:
            modified_obs = np.array(observation, dtype=np.float32)
            # Both position and velocity observations are affected by wind noise.
            # Noise is sampled from a normal distribution, scaled by wind force.
            modified_obs[0] += np.random.normal(0, self.wind_force * self.WIND_POSITION_SCALE)
            modified_obs[1] += np.random.normal(0, self.wind_force * self.WIND_VELOCITY_SCALE)
            observation = modified_obs
        else:
            # Ensure observation is consistently a float32 numpy array even without wind.
            observation = np.array(observation, dtype=np.float32)

        return observation, reward, terminated, truncated, info
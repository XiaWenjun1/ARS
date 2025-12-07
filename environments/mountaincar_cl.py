import gymnasium as gym
import numpy as np
from .base_wrapper import ContinualLearningWrapper

class MountainCarCL(ContinualLearningWrapper):
    """
    A specialized ContinualLearningWrapper for the 'MountainCar-v0' environment.
    This wrapper modifies environment dynamics such as gravity and engine force
    to create distinct tasks for continual learning experiments.
    It preserves the native MountainCar reward structure and termination conditions,
    which simplifies the evaluation of agent adaptation to environmental changes.
    """
    
    def __init__(self, task_params, render_mode=None):
        """
        Initializes the MountainCarCL environment wrapper.

        Args:
            task_params (list): A list of dictionaries, where each dictionary defines
                                specific parameters for a MountainCar CL task (e.g., gravity, force).
            render_mode (str, optional): The render mode for the environment. Defaults to None.
        """
        super().__init__('MountainCar-v0', task_params, render_mode)
    
    def _apply_task_parameters(self):
        """
        Applies MountainCar-specific parameter modifications based on the current task configuration.
        This method accesses the underlying Gym environment's unwrapped properties to change them.
        Specifically, it modifies the 'gravity' and 'force' attributes of the environment.
        """
        task_config = self.task_params[self.current_task]
        
        # Get parameters from config with sensible defaults (native MountainCar values)
        gravity = task_config.get('gravity', 0.0025) # Default gravity value for MountainCar-v0
        force = task_config.get('force', 0.0010)     # Default force value for MountainCar-v0
        
        # Access the base environment's unwrapped object to modify its dynamics.
        env = self.env.unwrapped
        if hasattr(env, 'gravity'):
            env.gravity = gravity
        if hasattr(env, 'force'):
            env.force = force
        
        # Log the applied task parameters for debugging and monitoring.
        task_name = task_config.get('task_name', f'Task_{self.current_task}')
        print(f"[MountainCarCL] Switched to {task_name}: "
              f"gravity={gravity:.4f}, force={force:.4f}")
    
    def step(self, action):
        """
        Executes one step in the environment, returning the observation, reward,
        and termination status. This method uses the native MountainCar reward
        and termination logic without modification.
        
        - Reward: -1 per step (encourages reaching the goal quickly).
        - Termination: position >= 0.5 (car reaches the flag on the right hill).
        - Truncation: max_steps reached (default 200 in native MountainCar-v0).

        Args:
            action (int): The action to take in the environment.

        Returns:
            tuple: (observation, reward, terminated, truncated, info) from the base environment.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Keep native MountainCar behavior - no custom reward shaping or observation modification.
        # This ensures clean task boundaries for change detection and avoids introducing
        # additional complexities that are not directly related to environmental changes.
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """
        Resets the environment to its initial state and applies the parameters
        of the current task. This method calls the base environment's reset
        and then ensures the task parameters are correctly applied.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the base environment's reset method.

        Returns:
            tuple: (observation, info) from the base environment's reset.
        """
        observation, info = self.env.reset(**kwargs)
        # Ensure task parameters are applied after reset, in case the base env reset
        # would revert any changes.
        self._apply_task_parameters() 
        return observation, info
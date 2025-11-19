import gymnasium as gym
import numpy as np
from .base_wrapper import ContinualLearningWrapper

class MountainCarCL(ContinualLearningWrapper):
    """
    MountainCar Continual Learning Wrapper
    
    Modifies environment dynamics (gravity, force) for different tasks.
    Keeps native MountainCar reward structure and termination conditions.
    """
    
    def __init__(self, task_params, render_mode=None):
        super().__init__('MountainCar-v0', task_params, render_mode)
    
    def _apply_task_parameters(self):
        """Apply MountainCar specific parameter modifications"""
        task_config = self.task_params[self.current_task]
        
        # Get parameters from config with defaults (native MountainCar values)
        gravity = task_config.get('gravity', 0.0025)
        force = task_config.get('force', 0.0010)
        
        # Access base environment and modify dynamics
        env = self.env.unwrapped
        if hasattr(env, 'gravity'):
            env.gravity = gravity
        if hasattr(env, 'force'):
            env.force = force
        
        # Log task parameters for debugging
        task_name = task_config.get('task_name', f'Task_{self.current_task}')
        print(f"[MountainCarCL] Switched to {task_name}: "
              f"gravity={gravity:.4f}, force={force:.4f}")
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Uses native MountainCar reward and termination logic:
        - Reward: -1 per step (encourages reaching goal quickly)
        - Termination: position >= 0.5 (right flag)
        - Truncation: max_steps reached (default 200)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Keep native MountainCar behavior - no custom reward shaping
        # This ensures clean task boundaries for change detection
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and apply current task parameters"""
        observation, info = self.env.reset(**kwargs)
        return observation, info
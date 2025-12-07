import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

class ContinualLearningWrapper(gym.Wrapper, ABC):
    def __init__(self, env_name, task_params, render_mode=None):
        env = gym.make(env_name, render_mode=render_mode)
        super().__init__(env)
        self.task_params = task_params
        self.current_task = 0
        self.total_tasks = len(task_params)
        self._render_mode = render_mode
        self._apply_task_parameters()
        
    def change_task(self, task_id):
        """Switch to specified task"""
        if 0 <= task_id < self.total_tasks:
            self.current_task = task_id
            self._apply_task_parameters()
            self.reset()
        else:
            raise ValueError(f"Task ID {task_id} out of range")
    
    @abstractmethod
    def _apply_task_parameters(self):
        """Apply task parameters - subclasses must implement this method"""
        pass
    
    def get_current_task_info(self):
        """Return current task information"""
        return {
            'task_id': self.current_task,
            'task_params': self.task_params[self.current_task],
            'total_tasks': self.total_tasks
        }
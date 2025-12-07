import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

class ContinualLearningWrapper(gym.Wrapper, ABC):
    """
    Abstract base class for a Gymnasium environment wrapper designed for Continual Learning (CL) setups.
    This wrapper allows for dynamically changing task parameters within an environment,
    simulating task shifts in CL scenarios. Subclasses must implement the
    `_apply_task_parameters` method to define how specific task configurations
    modify the underlying environment.
    """
    def __init__(self, env_name, task_params, render_mode=None):
        """
        Initializes the ContinualLearningWrapper.

        Args:
            env_name (str): The name of the Gymnasium environment to wrap (e.g., "CartPole-v1").
            task_params (list): A list of dictionaries, where each dictionary defines
                                specific parameters for a continuous learning task.
            render_mode (str, optional): The render mode for the environment. Defaults to None.
        """
        env = gym.make(env_name, render_mode=render_mode)
        super().__init__(env)
        self.task_params = task_params
        self.current_task = 0 # Index of the currently active task.
        self.total_tasks = len(task_params)
        self._render_mode = render_mode
        self._apply_task_parameters() # Apply initial task parameters.
        
    def change_task(self, task_id):
        """
        Switches the environment to a different task defined in `task_params`.

        Args:
            task_id (int): The index of the new task to switch to.

        Raises:
            ValueError: If the provided `task_id` is out of the valid range.
        """
        if 0 <= task_id < self.total_tasks:
            self.current_task = task_id
            self._apply_task_parameters() # Apply the parameters for the new task.
            self.reset() # Reset the environment after changing tasks.
        else:
            raise ValueError(f"Task ID {task_id} out of range (0 to {self.total_tasks - 1})")
    
    @abstractmethod
    def _apply_task_parameters(self):
        """
        Abstract method to apply the parameters of the current task to the underlying environment.
        Subclasses must implement this method, accessing `self.task_params[self.current_task]`
        to configure the environment specific to the active task.
        """
        pass
    
    def get_current_task_info(self):
        """
        Returns a dictionary containing information about the currently active task.

        Returns:
            dict: A dictionary with 'task_id', 'task_params' (for the current task),
                  and 'total_tasks'.
        """
        return {
            'task_id': self.current_task,
            'task_params': self.task_params[self.current_task],
            'total_tasks': self.total_tasks
        }
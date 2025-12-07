import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract Base Class (ABC) for Reinforcement Learning agents.
    
    This class defines the standard interface that all specific agent implementations
    (e.g., DQN, PPO, SAC) must adhere to. It ensures consistency across different
    algorithms.
    """
    
    def __init__(self, state_dim, action_dim, config):
        """
        Initialize the agent with environment dimensions and configuration.

        Args:
            state_dim (int): The dimension size of the state space (observation).
            action_dim (int): The dimension size of the action space.
            config (object): A configuration object containing hyperparameters 
                             and settings (e.g., learning rate, batch size, device).
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Set the computation device (CPU or CUDA/GPU) based on the config
        self.device = config.DEVICE
        
    @abstractmethod
    def select_action(self, state, training=True):
        """
        Select an action given the current state.

        Args:
            state (np.array or torch.Tensor): The current observation from the environment.
            training (bool): Flag to indicate mode. 
                             - If True: Enable exploration (e.g., epsilon-greedy, stochastic sampling).
                             - If False: Run in evaluation mode (deterministic/greedy action).

        Returns:
            action: The action chosen by the agent.
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        Perform a learning update step.
        
        This method typically involves:
        1. Sampling a batch of experiences from a replay buffer.
        2. Calculating the loss (e.g., Bellman error, policy gradient).
        3. Updating the neural network parameters via backpropagation.
        """
        pass
    
    @abstractmethod
    def save(self, filepath):
        """
        Save the agent's internal state (model weights, optimizers) to disk.

        Args:
            filepath (str): The destination path for the checkpoint file.
        """
        pass
    
    @abstractmethod
    def load(self, filepath):
        """
        Load the agent's internal state from a file on disk.

        Args:
            filepath (str): The path to the checkpoint file to load.
        """
        pass
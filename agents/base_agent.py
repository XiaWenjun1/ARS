import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all RL agents"""
    
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.DEVICE
        
    @abstractmethod
    def select_action(self, state, training=True):
        pass
    
    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def save(self, filepath):
        pass
    
    @abstractmethod
    def load(self, filepath):
        pass
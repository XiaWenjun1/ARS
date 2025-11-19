import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store experience"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample batch"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {len(self.buffer)}")
        
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
    
    def __len__(self):
        return len(self.buffer)
import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """
    Experience Replay Buffer.
    
    A First-In-First-Out (FIFO) buffer that stores transitions (experiences) 
    collected by the agent. It allows for off-policy learning by sampling 
    experiences randomly, which breaks the temporal correlation between 
    consecutive samples and stabilizes training.
    """
    
    def __init__(self, capacity):
        """
        Initialize the buffer.

        Args:
            capacity (int): The maximum number of transitions the buffer can hold. 
                            When full, new transitions overwrite the oldest ones.
        """
        # distinct_id: Use deque with a maxlen to automatically handle FIFO removal
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a new experience transition in the buffer.

        Args:
            state: Current state observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observation.
            done: Boolean flag indicating if the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[torch.Tensor]: A tuple containing stacked tensors for 
                                 (state, action, reward, next_state, done).
        """
        # Ensure there are enough samples in the buffer
        if len(self.buffer) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {len(self.buffer)}")
        
        # Randomly select 'batch_size' transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Transpose the batch from a list of tuples to a tuple of lists
        # structure changes from: [(s1, a1...), (s2, a2...)] -> ([s1, s2], [a1, a2]...)
        # map(np.stack, ...) converts these lists into numpy arrays with a batch dimension
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        # Convert numpy arrays to PyTorch tensors with appropriate data types
        return (
            torch.FloatTensor(state),      # Float for continuous state values
            torch.LongTensor(action),      # Long (int64) for discrete action indices
            torch.FloatTensor(reward),     # Float for reward values
            torch.FloatTensor(next_state), # Float for next state values
            torch.BoolTensor(done)         # Bool for terminal flags
        )
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
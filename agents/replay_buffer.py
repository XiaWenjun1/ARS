import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """
    A fixed-size buffer to store experience tuples (state, action, reward, next_state, done).
    It allows for uniform random sampling of experiences, which helps to break correlations
    in the training data and improve stability of reinforcement learning algorithms like DQN.
    """
    
    def __init__(self, capacity):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of experiences to store in the buffer.
                            When the buffer is full, older experiences are discarded.
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Stores a single experience tuple into the buffer.

        Args:
            state (np.array): The state observed at time t.
            action (int): The action taken at time t.
            reward (float): The reward received after taking action at time t.
            next_state (np.array): The state observed at time t+1.
            done (bool): A boolean indicating if the episode terminated after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Randomly samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing five torch.Tensors:
                   - states (torch.FloatTensor): Batch of states.
                   - actions (torch.LongTensor): Batch of actions.
                   - rewards (torch.FloatTensor): Batch of rewards.
                   - next_states (torch.FloatTensor): Batch of next states.
                   - dones (torch.BoolTensor): Batch of 'done' flags.
        
        Raises:
            ValueError: If the buffer contains fewer experiences than the requested batch_size.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} items from buffer of size {len(self.buffer)}")
        
        batch = random.sample(self.buffer, batch_size)
        # Stacks the individual components of the sampled experiences into numpy arrays.
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        # Converts numpy arrays to PyTorch tensors with appropriate data types.
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.BoolTensor(done)
        )
    
    def __len__(self):
        """
        Returns the current number of experiences stored in the buffer.

        Returns:
            int: The number of experiences in the buffer.
        """
        return len(self.buffer)
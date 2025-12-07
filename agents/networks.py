import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """
    DQN Neural Network.
    
    This class defines a simple fully connected (dense) neural network used to 
    approximate the Q-value function Q(s, a). It maps an input state to Q-values 
    for each possible action.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the network architecture.

        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Dimension of the output action space (number of actions).
            hidden_dim (int): Number of neurons in the hidden layers. Default is 128.
        """
        super(DQNNetwork, self).__init__()
        
        # First fully connected layer: maps state_dim -> hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Second fully connected layer: maps hidden_dim -> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer: maps hidden_dim -> action_dim
        # The output represents the estimated Q-values for every action in the current state.
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Apply first linear layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        
        # Apply second linear layer followed by ReLU activation
        x = F.relu(self.fc2(x))
        
        # Apply output layer (no activation function, as Q-values can be arbitrary real numbers)
        return self.fc3(x)
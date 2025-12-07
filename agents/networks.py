import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """
    DQN Neural Network (Q-Network) to approximate the action-value function.
    This network takes a state as input and outputs Q-values for each possible action.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initializes the DQNNetwork architecture.

        Args:
            state_dim (int): The dimension of the input state space.
            action_dim (int): The dimension of the output action space (number of possible actions).
            hidden_dim (int, optional): The number of neurons in the hidden layers. Defaults to 128.
        """
        super(DQNNetwork, self).__init__()
        # First fully connected layer: maps state input to hidden_dim.
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Second fully connected layer: maps hidden_dim to hidden_dim.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Third fully connected layer: maps hidden_dim to action_dim (Q-values for each action).
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor representing the state.

        Returns:
            torch.Tensor: A tensor of Q-values for each action in the given state.
        """
        # Apply ReLU activation function after the first hidden layer.
        x = F.relu(self.fc1(x))
        # Apply ReLU activation function after the second hidden layer.
        x = F.relu(self.fc2(x))
        # No activation after the output layer, as we are predicting Q-values directly.
        return self.fc3(x)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .networks import DQNNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    """DQN Agent Implementation"""
    
    def __init__(self, state_dim, action_dim, config):
        """
        Initializes the DQN Agent.

        Args:
            state_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            config (object): Configuration object containing hyperparameters.
        """
        super().__init__(state_dim, action_dim, config)
        
        # Policy network (Q-network) to learn the optimal Q-values.
        self.policy_net = DQNNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        # Target network, a delayed copy of the policy network, used to stabilize training.
        self.target_net = DQNNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        # Initialize target network with the same weights as the policy network.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set target network to evaluation mode (no gradient computation).
        self.target_net.eval()
        
        # Optimizer for updating the policy network's weights.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # Experience replay buffer to store and sample transitions for training stability.
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        
        # Training parameters
        # Epsilon for epsilon-greedy action selection.
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        # Control whether to decay epsilon inside the agent (trainer may control it itself)
        # Defaults to False if not specified in config, allowing external control of epsilon decay.
        self.auto_decay_epsilon = getattr(config, 'AUTO_DECAY_EPSILON', False)
        # Discount factor for future rewards.
        self.gamma = config.GAMMA
        # Number of samples to draw from the replay buffer for each training update.
        self.batch_size = config.BATCH_SIZE
        # Frequency (in terms of training steps) at which the target network is updated to match the policy network.
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        Args:
            state (np.array): The current state of the environment.
            training (bool): If True, applies epsilon-greedy exploration; otherwise, selects the best action.

        Returns:
            int: The chosen action.
        """
        # Epsilon-greedy exploration: with probability epsilon, choose a random action.
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            # Otherwise, choose the action with the highest predicted Q-value from the policy network.
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update(self):
        """
        Performs one step of optimization on the policy network.

        Returns:
            float: The loss value after the update, or 0 if not enough experience.
        """
        # Ensure enough experiences are in the replay buffer before starting to train.
        if len(self.memory) < self.batch_size:
            return 0  # Not enough experience
        
        # Sample a batch of transitions from the replay buffer.
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Compute Q(s_t, a) - the Q-value of the current state and chosen action.
        # We use gather to select the Q-value corresponding to the action taken.
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states, then multiply by gamma and add reward.
        # This is the target Q-value for the Bellman equation.
        with torch.no_grad():
            # Get the maximum Q-value for the next state from the target network.
            next_q_values = self.target_net(next_state).max(1)[0]
            # Compute the target Q-value: R + gamma * max_a' Q_target(s', a').
            # If 'done' is true, the next state has no future reward, so next_q_values is 0.
            target_q_values = reward + (self.gamma * next_q_values * ~done)
        
        # Compute Huber loss (or MSELoss) between current Q-values and target Q-values.
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize the policy network.
        self.optimizer.zero_grad()  # Clear previous gradients.
        loss.backward()             # Compute gradients.
        # Clip gradients to prevent exploding gradients (optional but common in DQN).
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()       # Update policy network weights.

        self.steps_done += 1
        # Update the target network periodically to synchronize with the policy network.
        # This reduces instability by providing a more stable target for Q-value updates.
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Only decay epsilon inside agent when auto_decay is enabled.
        # This implements a linear decay of epsilon.
        if self.auto_decay_epsilon and self.epsilon > self.epsilon_end:
            # Linear decay calculation.
            decay_amount = (self.config.EPSILON_START - self.epsilon_end) / self.config.EPSILON_DECAY_STEPS
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)
        
        return loss.item()
    
    def push_memory(self, state, action, reward, next_state, done):
        """
        Stores a single transition into the replay buffer.

        Args:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode terminated.
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def save(self, filepath):
        """
        Saves the agent's current state, including network weights, optimizer state,
        epsilon value, and training steps.

        Args:
            filepath (str): The path to save the model to.
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """
        Loads the agent's state from a saved checkpoint.

        Args:
            filepath (str): The path to the saved model.
        """
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
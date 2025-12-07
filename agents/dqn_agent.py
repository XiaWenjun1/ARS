import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .base_agent import BaseAgent
from .networks import DQNNetwork
from .replay_buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    """
    DQN Agent Implementation.
    
    This class implements the Deep Q-Network (DQN) algorithm, utilizing two networks
    (policy and target) to stabilize training and an experience replay buffer to
    break correlation between consecutive samples.
    """
    
    def __init__(self, state_dim, action_dim, config):
        super().__init__(state_dim, action_dim, config)
        
        # --- Networks ---
        # Initialize the Policy Network (used for selecting actions and updating weights)
        self.policy_net = DQNNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        # Initialize the Target Network (used for calculating stable target Q-values)
        self.target_net = DQNNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        
        # distinct_id: Initialize target network weights to match policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set target network to evaluation mode (no gradient calculation needed)
        self.target_net.eval()
        
        # --- Optimizer ---
        # We only optimize the policy network's parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # --- Experience Replay ---
        # Buffer to store transitions (state, action, reward, next_state, done)
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        
        # --- Training Parameters ---
        # Epsilon for epsilon-greedy exploration
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        
        # Control whether to decay epsilon inside the agent (trainer may control it itself)
        self.auto_decay_epsilon = getattr(config, 'AUTO_DECAY_EPSILON', False)
        
        self.gamma = config.GAMMA  # Discount factor
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ  # How often to update target net
        
        self.steps_done = 0  # Counter for total steps taken
    
    def select_action(self, state, training=True):
        """
        Select an action using the Epsilon-Greedy strategy.
        
        Args:
            state: The current state observation.
            training (bool): If True, use epsilon-greedy (explore). 
                             If False, strictly greedy (exploit).
        """
        # Exploration: choose random action with probability epsilon
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        # Exploitation: choose action with highest Q-value
        else:
            with torch.no_grad():
                # Convert state to tensor and add batch dimension (1, state_dim)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                # Forward pass to get Q-values for all actions
                q_values = self.policy_net(state_tensor)
                # Return the index of the max Q-value
                return q_values.argmax().item()
    
    def update(self):
        """
        Perform one step of training (gradient descent) on the policy network.
        """
        # Check if we have enough samples in memory to form a batch
        if len(self.memory) < self.batch_size:
            return 0  # Not enough experience yet
        
        # Sample a random batch of transitions from replay buffer
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        
        # Move tensors to the configured device (CPU or GPU)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # --- Compute Current Q-values ---
        # Get Q(s) for all actions, then select the Q-value for the action actually taken.
        # gather(1, ...) selects columns based on action indices.
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        # --- Compute Target Q-values ---
        with torch.no_grad():
            # Get max Q(s', a') from Target Network for the next state
            next_q_values = self.target_net(next_state).max(1)[0]
            
            # Bellman equation: Target = R + gamma * max(Q(s'))
            # If done is True, the future value is 0 (terminal state), so we multiply by ~done.
            target_q_values = reward + (self.gamma * next_q_values * ~done)
        
        # --- Compute Loss ---
        # Calculate MSE loss between predicted Q and target Q
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # --- Optimize ---
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update weights

        self.steps_done += 1
        
        # --- Sync Target Network ---
        # Periodically copy policy net weights to target net to stabilize training
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # --- Epsilon Decay ---
        # Only decay epsilon inside agent when auto_decay is enabled
        if self.auto_decay_epsilon and self.epsilon > self.epsilon_end:
            # Linear decay calculation
            decay_amount = (self.config.EPSILON_START - self.epsilon_end) / self.config.EPSILON_DECAY_STEPS
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)
        
        return loss.item()
    
    def push_memory(self, state, action, reward, next_state, done):
        """Store a single transition tuple in the replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def save(self, filepath):
        """Save the entire agent state (networks, optimizer, training progress)"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """Load the agent state from a checkpoint file"""
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
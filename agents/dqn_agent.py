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
        super().__init__(state_dim, action_dim, config)
        
        # Networks
        self.policy_net = DQNNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        
        # Experience replay
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        
        # Training parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        # Control whether to decay epsilon inside the agent (trainer may control it itself)
        self.auto_decay_epsilon = getattr(config, 'AUTO_DECAY_EPSILON', False)
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        
        self.steps_done = 0
    
    def select_action(self, state, training=True):
        """Select action"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update(self):
        """Update networks"""
        if len(self.memory) < self.batch_size:
            return 0  # Not enough experience
        
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        current_q_values = self.policy_net(state).gather(1, action.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0]
            target_q_values = reward + (self.gamma * next_q_values * ~done)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Only decay epsilon inside agent when auto_decay is enabled
        if self.auto_decay_epsilon and self.epsilon > self.epsilon_end:
            # Linear decay
            decay_amount = (self.config.EPSILON_START - self.epsilon_end) / self.config.EPSILON_DECAY_STEPS
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_amount)
        
        return loss.item()
    
    def push_memory(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
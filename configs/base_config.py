import torch

class BaseConfig:
    """
    Base configuration class containing all shared hyperparameters and settings
    for reinforcement learning agents and training processes.
    This class serves as a foundation for specific environment configurations,
    ensuring consistency across different experiments.
    """
    
    # Device configuration
    # Automatically selects CUDA (GPU) if available, otherwise defaults to CPU.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training parameters
    # Learning rate for the optimizer (e.g., Adam optimizer).
    LEARNING_RATE = 1e-3
    # Discount factor for future rewards.
    GAMMA = 0.99
    # Number of samples to draw from the replay buffer for each training update.
    BATCH_SIZE = 32
    # Maximum capacity of the replay buffer.
    BUFFER_SIZE = 10000
    # Maximum number of steps an agent can take within a single episode.
    MAX_STEPS_PER_EPISODE = 500
    
    # DQN-specific parameters
    # Frequency (in terms of training steps) at which the target network is updated
    # to match the policy network.
    TARGET_UPDATE_FREQ = 100
    # Starting value for the epsilon in epsilon-greedy action selection.
    EPSILON_START = 1.0
    # Minimum (ending) value for the epsilon in epsilon-greedy action selection.
    EPSILON_END = 0.01
    # Number of steps over which epsilon will decay from EPSILON_START to EPSILON_END.
    EPSILON_DECAY_STEPS = 10000 # Added this line as it was used in dqn_agent.py but not defined here.
    
    # Network architecture
    # Number of neurons in the hidden layers of the neural networks.
    HIDDEN_DIM = 128
    
    # Logging settings
    # Interval (in episodes or steps) at which training progress and metrics are logged.
    LOG_INTERVAL = 100
import torch

class BaseConfig:
    """Base configuration class containing all shared hyperparameters and settings"""
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training parameters
    LEARNING_RATE = 1e-3
    GAMMA = 0.99
    BATCH_SIZE = 32
    BUFFER_SIZE = 10000
    MAX_STEPS_PER_EPISODE = 500
    
    # DQN parameters
    TARGET_UPDATE_FREQ = 100
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    
    # Network architecture
    HIDDEN_DIM = 128
    
    # Logging settings
    LOG_INTERVAL = 100
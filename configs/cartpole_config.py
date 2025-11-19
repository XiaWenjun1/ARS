from .base_config import BaseConfig

class CartPoleConfig(BaseConfig):
    """Specific configuration for CartPole environment"""
    
    # Environment name
    ENV_NAME = "CartPole-v1"
    
    # Epsilon control parameters
    EPSILON_DECAY_RATE = 0.995  # Standard decay rate
    AUTO_DECAY_EPSILON = False  # Let trainer control epsilon, don't use agent's internal auto-decay
    
    # CartPole specific continual learning parameters
    EPISODES_PER_TASK = 300
    CONVERGENCE_THRESHOLD = 50  # CartPole success threshold (used to calculate convergence speed)
    
    # CL task definition (number of tasks automatically obtained from TASKS list length)
    TASKS = [
        # T0
        {
            'pole_length': 0.5,
            'wind_force': 0.0,
            'task_name': 'normal'
        },
        # T1
        {
            'pole_length': 0.3,
            'wind_force': 10.0,
            'task_name': 'short_strong_wind'
        },
        # T2
        {
            'pole_length': 1.5,
            'wind_force': 12.0,
            'task_name': 'long_strong_wind'
        },
        # T3
        {
            'pole_length': 0.3,
            'wind_force': 15.0,
            'task_name': 'extreme'
        }
    ]
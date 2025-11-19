from .base_config import BaseConfig

class MountainCarConfig(BaseConfig):
    """Specific configuration for MountainCar environment"""
    
    # Environment name
    ENV_NAME = "MountainCar-v0"
    
    # Epsilon control parameters - MountainCar needs more exploration
    # Note: Epsilon decay controlled by trainer, not config
    AUTO_DECAY_EPSILON = False  # Let trainer control epsilon, don't use agent's internal auto-decay
    
    # MountainCar specific continual learning parameters
    EPISODES_PER_TASK = 150  # MountainCar needs more training
    CONVERGENCE_THRESHOLD = -110  # MountainCar success threshold (used to calculate convergence speed)
    MAX_STEPS_PER_EPISODE = 200
    
    # CL task definition (number of tasks automatically obtained from TASKS list length)
    TASKS = [
        # T0
        {
            'gravity': 0.0025,
            'force': 0.001,
            'task_name': 'normal'
        },
        # T1
        {
            'gravity': 0.0045,
            'force': 0.001,
            'task_name': 'strong_gravity'
        },
        # T2
        {
            'gravity': 0.0025,
            'force': 0.0006,
            'task_name': 'weak_force'
        },
        # T3
        {
            'gravity': 0.0035,
            'force': 0.0007,
            'task_name': 'combined_hard'
        }
    ]

    RQ1_WARMUP_EPISODES = 50

    RQ2_WM_WARMUP_EPISODES = 80
from .base_config import BaseConfig

class MountainCarConfig(BaseConfig):
    """
    Specific configuration for the MountainCar environment.
    
    This configuration defines a Continual Learning (CL) curriculum where the 
    physical properties of the car and mountain (gravity, engine force) change 
    over time to test the agent's adaptability.
    """
    
    ENV_NAME = "MountainCar-v0"
    AUTO_DECAY_EPSILON = False 
    
    # [Modification] Slightly increased episode limit to give the Agent more chances to learn.
    EPISODES_PER_TASK = 150 
    
    # Score threshold to consider the task converged/solved.
    CONVERGENCE_THRESHOLD = -110
    
    # Maximum steps allowed per episode before truncation.
    MAX_STEPS_PER_EPISODE = 200
    
    # [Key Modification] Gradual progression of task difficulty.
    TASKS = [
        # T0: Easy Mode (Benchmark)
        # Low gravity, high force. It is very easy to drive up the hill.
        # Expected score range: -90 ~ -110
        {
            'gravity': 0.0015,  # Default is 0.0025
            'force': 0.0015,    # Default is 0.0010
            'task_name': 'easy_start'
        },
        
        # T1: Standard Mode (Harder)
        # Returns to default Gym settings.
        # Expected score range: -120 ~ -150
        {
            'gravity': 0.0025,
            'force': 0.0010,
            'task_name': 'normal'
        },
        
        # T2: Weak Force Mode
        # Requires more swinging (momentum building).
        # Expected score range: -140 ~ -170
        # Note: Force set to 0.0008 (avoided reducing strictly to 0.0006).
        {
            'gravity': 0.0025,
            'force': 0.0008,    
            'task_name': 'weak_force'
        },
        
        # T3: Heavy Gravity Mode
        # The hardest task, but still mathematically solvable.
        # Note: Gravity adjusted (avoided increasing to extreme 0.0045).
        {
            'gravity': 0.0020,  
            'force': 0.0012,
            'task_name': 'heavy_gravity'
        }
    ]

    RQ1_WARMUP_EPISODES = 50
    RQ2_WM_WARMUP_EPISODES = 80
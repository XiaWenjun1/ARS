from .base_config import BaseConfig

class CartPoleConfig(BaseConfig):
    """
    Specific configuration for the CartPole environment in a Continual Learning setup.
    
    This configuration defines a sequence of tasks with varying physical properties
    (pole length, cart mass, wind force) to test the agent's ability to adapt 
    to changing environmental dynamics.
    """
    
    ENV_NAME = "CartPole-v1"
    
    # Disable automatic epsilon decay within the agent; let the Trainer class manage exploration.
    AUTO_DECAY_EPSILON = False
    
    # Short-cycle configuration: 100 episodes per task.
    EPISODES_PER_TASK = 100
    
    # Set the limit higher (500) to distinguish clearly between optimal play and sub-optimal play.
    MAX_STEPS_PER_EPISODE = 500 
    
    # Score threshold to consider the task "solved" or converged.
    CONVERGENCE_THRESHOLD = 450
    
    # --- CL Task Definitions ---
    # Strategy: Create significant performance gaps by manipulating:
    # 1. Pole Length (affects reaction speed required/instability).
    # 2. Mass (affects inertia/force required).
    # 3. Wind Force (adds noise/external disturbance).
    TASKS = [
        # T0: Standard Environment (Easy)
        # Goal: Allow the Agent to learn quickly and reach a high score (400+).
        {
            'pole_length': 0.5,
            'masscart': 1.0,
            'wind_force': 0.0,
            'task_name': 'standard'
        },
        
        # T1: [Fast Storm] Extremely Short Pole + Moderate Wind (Hard)
        # Variation: Short pole (0.25) makes it fall extremely fast. Combined with wind,
        # previous strategies will be too slow to react.
        # Expectation: Performance/Score will plummet.
        {
            'pole_length': 0.25, 
            'masscart': 1.0,
            'wind_force': 5.0,
            'task_name': 'fast_windy'
        },
        
        # T2: [Heavy Headwind] Heavy Cart + Long Pole + Strong Wind (Very Hard)
        # Variation: Heavy cart (2.0) has high inertia (hard to start/stop). 
        # Long pole (0.8) falls slowly but is very hard to recover once tipped.
        # Strong wind (10.0) provides constant interference.
        # Expectation: Previous policies won't apply enough force to maintain balance.
        {
            'pole_length': 0.8, 
            'masscart': 2.0,
            'wind_force': 10.0,
            'task_name': 'heavy_strong_wind'
        },
        
        # T3: [Extreme Chaos] Short Pole + Extreme Wind (Disaster)
        # A combination of high instability and massive external disturbance.
        {
            'pole_length': 0.3,
            'masscart': 1.0,
            'wind_force': 15.0,
            'task_name': 'extreme_chaos'
        }
    ]

    # Warm-up episodes for Research Question 1 (RQ1)
    RQ1_WARMUP_EPISODES = 50
    
    # Warm-up episodes for World Model in Research Question 2 (RQ2)
    RQ2_WM_WARMUP_EPISODES = 80
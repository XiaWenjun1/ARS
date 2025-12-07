class CartPoleConfig(BaseConfig):
    """
    Specific configuration for the CartPole environment, extending the BaseConfig.
    Defines environment-specific parameters, curriculum learning tasks, and
    hyperparameters tailored for CartPole experiments.
    """
    
    # Environment name for OpenAI Gym.
    ENV_NAME = "CartPole-v1"
    
    # Disable Epsilon auto-decay within the agent; Trainer will handle epsilon decay.
    AUTO_DECAY_EPSILON = False
    
    # Curriculum learning task parameters
    # Number of episodes to train on each individual task before transitioning.
    EPISODES_PER_TASK = 100
    # Maximum number of steps an agent can take within a single episode.
    # Set higher to allow for greater distinction between good and bad performance.
    MAX_STEPS_PER_EPISODE = 500  
    # Score threshold for considering a task "converged" or "solved".
    CONVERGENCE_THRESHOLD = 450
    
    # CL task definition: A list of dictionaries, each defining a specific environment configuration (task).
    # Strategy: Create significant gaps in difficulty by manipulating 'pole_length', 'masscart', and 'wind_force'.
    TASKS = [
        # T0: Standard Environment (Easy)
        # Goal: Agent should learn quickly and achieve scores above 400.
        {
            'pole_length': 0.5,
            'masscart': 1.0,
            'wind_force': 0.0,
            'task_name': 'standard'
        },
        # T1: [Extreme Gale] Very short pole + Medium wind (Difficult)
        # Change: Shorter pole (0.25) causes very fast fall speed, combined with wind interference,
        #         making the old policy unable to react effectively.
        # Expected: Significant drop in scores.
        {
            'pole_length': 0.25, 
            'masscart': 1.0,
            'wind_force': 5.0,
            'task_name': 'fast_windy'
        },
        # T2: [Heavy Headwind] Heavy cart + Long pole + Strong wind (Very Difficult)
        # Change: Heavier cart (2.0) makes it harder to push (high inertia),
        #         a long pole (0.8) falls slower but is harder to recover,
        #         and strong wind (10.0) provides continuous interference.
        # Expected: Old policy lacks sufficient force to maintain balance.
        {
            'pole_length': 0.8, 
            'masscart': 2.0,
            'wind_force': 10.0,
            'task_name': 'heavy_strong_wind'
        },
        # T3: [Extreme Chaos] Short pole + Extreme wind (Catastrophic)
        # Change: Short pole (0.3) combined with extreme wind (15.0) creates a highly unstable environment.
        # Expected: Agent struggles significantly to balance.
        {
            'pole_length': 0.3,
            'masscart': 1.0,
            'wind_force': 15.0,
            'task_name': 'extreme_chaos'
        }
    ]

    # Warm-up episodes for specific research questions (RQ1).
    RQ1_WARMUP_EPISODES = 50
    # Warm-up episodes for the World Model in research question (RQ2).
    RQ2_WM_WARMUP_EPISODES = 80
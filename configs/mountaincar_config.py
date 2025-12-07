class MountainCarConfig(BaseConfig):
    """
    Specific configuration for the MountainCar environment, extending the BaseConfig.
    Defines environment-specific parameters, curriculum learning tasks, and
    hyperparameters tailored for MountainCar experiments.
    """
    
    ENV_NAME = "MountainCar-v0"
    # Disable Epsilon auto-decay within the agent; Trainer will handle epsilon decay.
    AUTO_DECAY_EPSILON = False 
    
    # [Modification] Slightly increased step limit to give the Agent more opportunities.
    EPISODES_PER_TASK = 150 
    # Score threshold for considering a task "converged" or "solved".
    CONVERGENCE_THRESHOLD = -110
    # Maximum number of steps an agent can take within a single episode.
    MAX_STEPS_PER_EPISODE = 200
    
    # [Key Modification] Task difficulty tiered degradation.
    TASKS = [
        # T0: Easy Mode (Benchmark) - Low gravity, high force, very easy to climb (-90 to -110).
        {
            'gravity': 0.0015,  # Default is 0.0025
            'force': 0.0015,    # Default is 0.0010
            'task_name': 'easy_start'
        },
        # T1: Standard Mode (Slightly harder) - Reverts to default settings, score drops to -120 to -150.
        {
            'gravity': 0.0025,
            'force': 0.0010,
            'task_name': 'normal'
        },
        # T2: Weak Force Mode - Requires more swinging, score -140 to -170.
        {
            'gravity': 0.0025,
            'force': 0.0008,    # Avoid reducing it too much to 0.0006
            'task_name': 'weak_force'
        },
        # T3: Heavy Gravity Mode - Most difficult, but still solvable.
        {
            'gravity': 0.0020,  # Avoid increasing it to 0.0045
            'force': 0.0012,
            'task_name': 'heavy_gravity'
        }
    ]

    # Warm-up episodes for specific research questions (RQ1).
    RQ1_WARMUP_EPISODES = 50
    # Warm-up episodes for the World Model in research question (RQ2).
    RQ2_WM_WARMUP_EPISODES = 80
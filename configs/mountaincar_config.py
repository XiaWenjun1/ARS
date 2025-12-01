from .base_config import BaseConfig

class MountainCarConfig(BaseConfig):
    """Specific configuration for MountainCar environment"""
    
    ENV_NAME = "MountainCar-v0"
    AUTO_DECAY_EPSILON = False 
    
    # [修改] 稍微增加步数限制，给 Agent 更多机会
    EPISODES_PER_TASK = 150 
    CONVERGENCE_THRESHOLD = -110
    MAX_STEPS_PER_EPISODE = 200
    
    # [关键修改] 任务难度阶梯化降级
    TASKS = [
        # T0: 简单模式 (基准) - 重力小，推力大，很容易冲上去 (-90 ~ -110)
        {
            'gravity': 0.0015,  # 默认是 0.0025
            'force': 0.0015,    # 默认是 0.0010
            'task_name': 'easy_start'
        },
        # T1: 标准模式 (变难一点) - 回归默认设置，分数会下降到 -120 ~ -150
        {
            'gravity': 0.0025,
            'force': 0.0010,
            'task_name': 'normal'
        },
        # T2: 弱推力模式 - 需要更多摆动，分数 -140 ~ -170
        {
            'gravity': 0.0025,
            'force': 0.0008,    # 不要减到 0.0006 那么狠
            'task_name': 'weak_force'
        },
        # T3: 重重力模式 - 最难，但仍可解
        {
            'gravity': 0.0020,  # 不要加到 0.0045
            'force': 0.0012,
            'task_name': 'heavy_gravity'
        }
    ]

    RQ1_WARMUP_EPISODES = 50
    RQ2_WM_WARMUP_EPISODES = 80
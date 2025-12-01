from .base_config import BaseConfig

class CartPoleConfig(BaseConfig):
    """Specific configuration for CartPole environment"""
    
    ENV_NAME = "CartPole-v1"
    
    # 禁用 Epsilon 自动衰减，由 Trainer 接管
    AUTO_DECAY_EPSILON = False
    
    # CartPole 100 eps 短周期配置
    EPISODES_PER_TASK = 100
    MAX_STEPS_PER_EPISODE = 500  # 上限设高一点，让满分和低分差距拉大
    CONVERGENCE_THRESHOLD = 450
    
    # CL task definition
    # 策略：利用“杆长(反应速度)”、“质量(惯性)”和“风力(噪声)”制造巨大的 Gap
    TASKS = [
        # T0: 标准环境 (容易)
        # 目标：让 Agent 快速学会，拿到 400+ 分
        {
            'pole_length': 0.5,
            'masscart': 1.0,
            'wind_force': 0.0,
            'task_name': 'standard'
        },
        # T1: 【极速风暴】极短杆 + 中等风 (难)
        # 变化：杆子变短(0.25)导致倒得极快，加上风力干扰，旧策略反应不过来。
        # 预期：分数暴跌。
        {
            'pole_length': 0.25, 
            'masscart': 1.0,
            'wind_force': 5.0,
            'task_name': 'fast_windy'
        },
        # T2: 【重载逆风】重车 + 长杆 + 强风 (很难)
        # 变化：车变重(2.0)导致推不动(惯性大)，长杆(0.8)虽然倒得慢但很难救回来，强风(10.0)持续干扰。
        # 预期：旧策略力度不够，无法平衡。
        {
            'pole_length': 0.8, 
            'masscart': 2.0,
            'wind_force': 10.0,
            'task_name': 'heavy_strong_wind'
        },
        # T3: 【极限混乱】短杆 + 极端风 (灾难)
        {
            'pole_length': 0.3,
            'masscart': 1.0,
            'wind_force': 15.0,
            'task_name': 'extreme_chaos'
        }
    ]

    RQ1_WARMUP_EPISODES = 50
    RQ2_WM_WARMUP_EPISODES = 80
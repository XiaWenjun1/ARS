import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import uuid
import glob

# 可识别的环境前缀（全小写）
KNOWN_ENVS = ['cartpole', 'mountaincar']

def ensure_visualization_dir():
    """确保visualizations目录存在（在experiments同级目录）"""
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    vis_dir = os.path.join(parent_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    return vis_dir

def get_unique_filepath(prefix: str, vis_dir: str, ext: str = "png") -> str:
    """生成唯一文件路径，格式: {prefix}_{YYYYmmddTHHMMSS}_{6hex}.ext"""
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    filename = f"{prefix}_{ts}_{short_id}.{ext}"
    return os.path.join(vis_dir, filename)

def infer_env_from_cfg(cfg) -> str:
    """尝试从 cfg 推断环境名，返回小写前缀，找不到则返回 'env'"""
    if cfg is None:
        return "env"
    cls_name = cfg.__class__.__name__.lower()
    for env in KNOWN_ENVS:
        if env in cls_name:
            return env
    try:
        tasks = getattr(cfg, 'TASKS', None)
        if tasks:
            combined = " ".join([t.get('task_name', '') for t in tasks]).lower()
            for env in KNOWN_ENVS:
                if env in combined:
                    return env
    except Exception:
        pass
    return "env"

def infer_env_from_latest_visual(vis_dir: str) -> str:
    """当没有 cfg 时，尝试从 visualizations 目录中最新文件的前缀推断环境"""
    if not os.path.exists(vis_dir):
        return "env"
    pattern = os.path.join(vis_dir, "*")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return "env"
    latest = os.path.basename(files[0])
    token = latest.split("_", 1)[0].lower()
    if token in KNOWN_ENVS:
        return token
    for env in KNOWN_ENVS:
        if env in token:
            return env
    return "env"

def plot_task_performance_heatmap(summary_data, cfg, save_dir=None):
    """显示每个检测器在不同任务上的表现，文件名带环境前缀"""
    # 如果未提供 save_dir，则使用默认路径
    if save_dir is None:
        save_dir = ensure_visualization_dir()
    else:
        os.makedirs(save_dir, exist_ok=True)
        
    env_prefix = infer_env_from_cfg(cfg)

    # 准备数据
    task_names = [f"T{i}\n({cfg.TASKS[i]['task_name']})" for i in range(len(cfg.TASKS))]
    detector_names = [data['name'] for data in summary_data]

    # 创建性能矩阵
    performance_matrix = np.zeros((len(detector_names), len(task_names)))
    for i, data in enumerate(summary_data):
        for j in range(len(task_names)):
            # 保护性索引（避免少数 seed 缺失某个任务数据导致 KeyError）
            task_rewards = [r['eval_rewards'].get(j, 0.0) for r in data['all_results']]
            performance_matrix[i, j] = np.mean(task_rewards)

    # 绘制热图
    plt.figure(figsize=(12, 8))
    sns.heatmap(performance_matrix,
                xticklabels=task_names,
                yticklabels=detector_names,
                annot=True, fmt=".1f", cmap="YlGnBu",
                cbar_kws={'label': 'Average Reward'})
    plt.title("Task Performance Heatmap by Detector")
    plt.xlabel("Tasks")
    plt.ylabel("Detectors")
    plt.tight_layout()

    save_path = get_unique_filepath(f"{env_prefix}_task_performance_heatmap", save_dir, ext='png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {save_path}")
    plt.close()

def plot_detector_comparison(summary_data, save_dir=None):
    """对比检测器的综合性能（奖励+检测指标），尽量带环境前缀（从已有可视化推断）"""
    # 如果未提供 save_dir，则使用默认路径
    if save_dir is None:
        vis_dir = ensure_visualization_dir()
    else:
        vis_dir = save_dir
        os.makedirs(vis_dir, exist_ok=True)

    env_prefix = infer_env_from_latest_visual(vis_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：平均评估奖励（按性能排序）
    summary_data_sorted = sorted(summary_data, key=lambda x: x['mean_eval'], reverse=True)
    names = [data['name'] for data in summary_data_sorted]
    evals = [data['mean_eval'] for data in summary_data_sorted]
    eval_err = [data['std_eval'] for data in summary_data_sorted]

    colors = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(names))) if len(names) > 0 else []

    bars1 = ax1.bar(range(len(names)), evals, yerr=eval_err,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Average Evaluation Reward')
    ax1.set_title('Performance Comparison (Sorted by Reward)')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=10)

    max_eval = max(evals) if evals else 1.0
    text_offset = max(1.0, 0.02 * max_eval)
    for bar, val in zip(bars1, evals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset,
                 f'{val:.0f}', ha='center', va='bottom', fontweight='bold')

    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：检测指标
    precisions = [data['mean_prec'] if not np.isnan(data['mean_prec']) else 0 for data in summary_data_sorted]
    recalls = [data['mean_rec'] if not np.isnan(data['mean_rec']) else 0 for data in summary_data_sorted]

    x = np.arange(len(names))
    width = 0.35

    bars2 = ax2.bar(x - width / 2, precisions, width, label='Precision',
                    color='lightcoral', alpha=0.8, edgecolor='darkred')
    bars3 = ax2.bar(x + width / 2, recalls, width, label='Recall',
                    color='lightblue', alpha=0.8, edgecolor='darkblue')

    ax2.set_ylabel('Score')
    ax2.set_title('Detection Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (prec, rec) in enumerate(zip(precisions, recalls)):
        if prec > 0:
            ax2.text(i - width / 2, prec + 0.02, f'{prec:.2f}',
                     ha='center', va='bottom', fontsize=8)
        if rec > 0:
            ax2.text(i + width / 2, rec + 0.02, f'{rec:.2f}',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    save_path = get_unique_filepath(f"{env_prefix}_detector_comparison", vis_dir, ext='png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved detector comparison: {save_path}")
    plt.close()

def plot_learning_curves(summary_data, cfg=None, save_dir=None):
    """
    RQ3风格的学习曲线 (Learning Curves Comparison)
    修改：不再绘制平均值和标准差，而是绘制所有Seed的独立曲线。
    这样可以更直观地看到每个实验的适应过程和稳定性。
    """
    # 如果未提供 save_dir，则使用默认路径
    if save_dir is None:
        save_dir = ensure_visualization_dir()
    else:
        os.makedirs(save_dir, exist_ok=True)
        
    env_prefix = infer_env_from_cfg(cfg)
    plot_data = summary_data
    
    plt.figure(figsize=(14, 7))
    
    # 获取任务切换点 (假设所有 run 的切换点一致，取第一个有数据的)
    change_points = []
    for data in summary_data:
        if data['all_results'] and 'change_points' in data['all_results'][0]:
            change_points = data['all_results'][0]['change_points']
            break
            
    # 绘制曲线
    colors = sns.color_palette("husl", len(plot_data))
    
    for idx, data in enumerate(plot_data):
        name = data['name']
        all_rewards = []
        
        # 收集该检测器所有 Seed 的 episode_rewards
        for res in data['all_results']:
            if 'episode_rewards' in res:
                all_rewards.append(res['episode_rewards'])
        
        if not all_rewards:
            continue
            
        # 遍历每个种子并独立绘制
        window_size = 20
        kernel = np.ones(window_size) / window_size
        
        for i, rewards in enumerate(all_rewards):
            rewards_arr = np.array(rewards)
            
            # 平滑处理 (Window smoothing)
            if len(rewards_arr) > window_size:
                smoothed = np.convolve(rewards_arr, kernel, mode='valid')
                x_axis = np.arange(len(smoothed)) + window_size // 2
            else:
                smoothed = rewards_arr
                x_axis = np.arange(len(rewards_arr))
            
            # 绘图：同一个检测器的不同种子使用相同的颜色
            # 只有第一个种子带 Label，避免图例重复
            label = name if i == 0 else None
            
            # 设置 alpha 透明度，让重叠部分更深
            plt.plot(x_axis, smoothed, label=label, color=colors[idx], linewidth=1.5, alpha=0.6)

    # 绘制任务切换竖线
    for i, cp in enumerate(change_points):
        plt.axvline(x=cp, color='gray', linestyle='--', alpha=0.6, 
                   label='Task Change' if i == 0 else "")
        plt.text(cp + 5, plt.ylim()[1] * 0.95, f'T{i+1}', color='gray', fontsize=8)

    plt.xlabel('Episode')
    plt.ylabel(f'Reward (Smoothed, w={window_size})')
    plt.title(f'RQ1 Learning Curves: {env_prefix.capitalize()} (Individual Seeds)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = get_unique_filepath(f"{env_prefix}_learning_curves", save_dir, ext='png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves: {save_path}")
    plt.close()
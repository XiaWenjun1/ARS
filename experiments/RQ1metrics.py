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
    # 优先用 cfg 类名，比如 CartPoleConfig / MountainCarConfig
    cls_name = cfg.__class__.__name__.lower()
    for env in KNOWN_ENVS:
        if env in cls_name:
            return env
    # 其次检查 TASKS 里可能包含的字符串
    try:
        tasks = getattr(cfg, 'TASKS', None)
        if tasks:
            # 把所有 task_name 合并成字符串检索关键字
            combined = " ".join([t.get('task_name', '') for t in tasks]).lower()
            for env in KNOWN_ENVS:
                if env in combined:
                    return env
    except Exception:
        pass
    return "env"

def infer_env_from_latest_visual(vis_dir: str) -> str:
    """当没有 cfg 时，尝试从 visualizations 目录中最新文件的前缀推断环境"""
    pattern = os.path.join(vis_dir, "*")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return "env"
    latest = os.path.basename(files[0])
    # 假设文件名格式为 <prefix>_YYYY... 以 '_' 分割并取第一个 token
    token = latest.split("_", 1)[0].lower()
    if token in KNOWN_ENVS:
        return token
    # 也尝试 token 中包含已知 env
    for env in KNOWN_ENVS:
        if env in token:
            return env
    return "env"

def plot_task_performance_heatmap(summary_data, cfg):
    """显示每个检测器在不同任务上的表现，文件名带环境前缀"""
    vis_dir = ensure_visualization_dir()
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

    # 保存到visualizations目录，文件名前加环境前缀
    save_path = get_unique_filepath(f"{env_prefix}_task_performance_heatmap", vis_dir, ext='png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {save_path}")
    plt.close()

def plot_detector_comparison(summary_data):
    """对比检测器的综合性能（奖励+检测指标），尽量带环境前缀（从已有可视化推断）"""
    vis_dir = ensure_visualization_dir()
    env_prefix = infer_env_from_latest_visual(vis_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：平均评估奖励（按性能排序）
    summary_data_sorted = sorted(summary_data, key=lambda x: x['mean_eval'], reverse=True)
    names = [data['name'] for data in summary_data_sorted]
    evals = [data['mean_eval'] for data in summary_data_sorted]
    eval_err = [data['std_eval'] for data in summary_data_sorted]

    # 使用颜色渐变：性能越高颜色越深
    colors = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(names))) if len(names) > 0 else []

    bars1 = ax1.bar(range(len(names)), evals, yerr=eval_err,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Average Evaluation Reward')
    ax1.set_title('Performance Comparison (Sorted by Reward)')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=10)

    # 在柱子上添加数值（根据数据范围自适应偏移）
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

    # 在柱子上添加数值
    for i, (prec, rec) in enumerate(zip(precisions, recalls)):
        if prec > 0:
            ax2.text(i - width / 2, prec + 0.02, f'{prec:.2f}',
                     ha='center', va='bottom', fontsize=8)
        if rec > 0:
            ax2.text(i + width / 2, rec + 0.02, f'{rec:.2f}',
                     ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # 保存到visualizations目录，文件名前加环境前缀（若无法推断则为 env）
    save_path = get_unique_filepath(f"{env_prefix}_detector_comparison", vis_dir, ext='png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved detector comparison: {save_path}")
    plt.close()

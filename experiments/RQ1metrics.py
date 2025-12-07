import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import uuid
import glob

# Recognizable environment prefixes (all lowercase)
KNOWN_ENVS = ['cartpole', 'mountaincar']

def ensure_visualization_dir():
    """
    Ensure the 'visualizations' directory exists (at the same level as experiments).
    """
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    vis_dir = os.path.join(parent_dir, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    return vis_dir

def get_unique_filepath(prefix: str, vis_dir: str, ext: str = "png") -> str:
    """
    Generate a unique filepath in the format: {prefix}_{YYYYmmddTHHMMSS}_{6hex}.ext
    """
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    filename = f"{prefix}_{ts}_{short_id}.{ext}"
    return os.path.join(vis_dir, filename)

def infer_env_from_cfg(cfg) -> str:
    """
    Attempt to infer the environment name from the configuration object (cfg).
    Returns a lowercase prefix, or 'env' if not found.
    """
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
    """
    When no cfg is available, try to infer the environment from the prefix of the 
    latest file in the visualizations directory.
    """
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
    """
    Display the performance of each detector across different tasks as a heatmap.
    The filename will include the environment prefix.
    """
    # 
    # If save_dir is not provided, use the default path
    if save_dir is None:
        save_dir = ensure_visualization_dir()
    else:
        os.makedirs(save_dir, exist_ok=True)
        
    env_prefix = infer_env_from_cfg(cfg)

    # Prepare data
    task_names = [f"T{i}\n({cfg.TASKS[i]['task_name']})" for i in range(len(cfg.TASKS))]
    detector_names = [data['name'] for data in summary_data]

    # Create performance matrix
    performance_matrix = np.zeros((len(detector_names), len(task_names)))
    for i, data in enumerate(summary_data):
        for j in range(len(task_names)):
            # Defensive indexing (avoids KeyError if a few seeds are missing data for a task)
            task_rewards = [r['eval_rewards'].get(j, 0.0) for r in data['all_results']]
            performance_matrix[i, j] = np.mean(task_rewards)

    # Plot Heatmap
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
    """
    Compare the comprehensive performance (Reward + Detection Metrics) of detectors.
    Tries to include the environment prefix in the filename (inferred from existing visuals).
    """
    # 
    # If save_dir is not provided, use the default path
    if save_dir is None:
        vis_dir = ensure_visualization_dir()
    else:
        vis_dir = save_dir
        os.makedirs(vis_dir, exist_ok=True)

    env_prefix = infer_env_from_latest_visual(vis_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left Plot: Average Evaluation Reward (Sorted by performance)
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

    # Right Plot: Detection Metrics (Precision & Recall)
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
    RQ3-style Learning Curves Comparison.
    

[Image of reinforcement learning curve plot]

    Modification: Instead of plotting Mean +/- Std, this plots individual curves for ALL seeds.
    This allows for a more intuitive visualization of the adaptation process and stability of each experiment.
    """
    # If save_dir is not provided, use the default path
    if save_dir is None:
        save_dir = ensure_visualization_dir()
    else:
        os.makedirs(save_dir, exist_ok=True)
        
    env_prefix = infer_env_from_cfg(cfg)
    plot_data = summary_data
    
    plt.figure(figsize=(14, 7))
    
    # Get task change points (Assume change points are consistent across runs, take the first valid one)
    change_points = []
    for data in summary_data:
        if data['all_results'] and 'change_points' in data['all_results'][0]:
            change_points = data['all_results'][0]['change_points']
            break
            
    # Plot curves
    colors = sns.color_palette("husl", len(plot_data))
    
    for idx, data in enumerate(plot_data):
        name = data['name']
        all_rewards = []
        
        # Collect episode_rewards for all seeds of this detector
        for res in data['all_results']:
            if 'episode_rewards' in res:
                all_rewards.append(res['episode_rewards'])
        
        if not all_rewards:
            continue
            
        # Iterate through each seed and plot independently
        window_size = 20
        kernel = np.ones(window_size) / window_size
        
        for i, rewards in enumerate(all_rewards):
            rewards_arr = np.array(rewards)
            
            # Smoothing (Window convolution)
            if len(rewards_arr) > window_size:
                smoothed = np.convolve(rewards_arr, kernel, mode='valid')
                x_axis = np.arange(len(smoothed)) + window_size // 2
            else:
                smoothed = rewards_arr
                x_axis = np.arange(len(rewards_arr))
            
            # Plotting: Different seeds of the same detector use the same color
            # Only the first seed gets a Label to avoid legend duplication
            label = name if i == 0 else None
            
            # Set alpha transparency so overlapping lines appear darker
            plt.plot(x_axis, smoothed, label=label, color=colors[idx], linewidth=1.5, alpha=0.6)

    # Draw vertical lines for task changes
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
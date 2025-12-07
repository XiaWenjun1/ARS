"""
MountainCar Continual Learning Training Script
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set fonts to avoid display issues
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs import MountainCarConfig
from environments import MountainCarCL
from agents import DQNAgent
from training import CLTrainer

def main():
    print("=== MountainCar Continual Learning Training ===")
    print("Focus: Convergence Speed | Average Reward | Catastrophic Forgetting")
    print("=" * 60)
    
    # Create configuration, environment and agent
    config = MountainCarConfig()
    env = MountainCarCL(config.TASKS)
    
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config
    )
    
    # Create trainer
    trainer = CLTrainer(agent, env, config)
    
    print(f"\nStarting continual learning training...")
    print(f"Total tasks: {env.total_tasks}")
    print(f"Episodes per task: {config.EPISODES_PER_TASK}")
    print("=" * 60)
    
    # Run continual learning training
    task_performances = trainer.run_continual_learning()
    
    # Use methods from metrics to calculate core metrics
    metrics_summary = trainer.metrics.get_core_metrics_summary(
        total_tasks=env.total_tasks,
        convergence_threshold=getattr(config, 'CONVERGENCE_THRESHOLD', -110)
    )
    
    # Output core metrics
    print(f"\n{'='*60}")
    print("TASK 2 CORE PERFORMANCE METRICS")
    print(f"{'='*60}")
    
    # 1. Convergence Speed Analysis
    print("\n1. CONVERGENCE SPEED ANALYSIS:")
    print("-" * 40)
    
    convergence_data = metrics_summary['convergence_data']
    for task_id, convergence_episode in convergence_data.items():
        print(f"Task {task_id}: {convergence_episode} episodes")
    
    print(f"Average convergence speed: {metrics_summary['avg_convergence']:.1f} episodes")
    
    # 2. Average Reward Analysis
    print(f"\n2. AVERAGE REWARD ANALYSIS:")
    print("-" * 40)
    
    # Use task performance data recorded in metrics
    task_rewards = []
    for task_id in range(env.total_tasks):
        if task_id in trainer.metrics.task_performance:
            rewards = trainer.metrics.task_performance[task_id]
            if rewards:
                avg_reward = np.mean(rewards)
                task_rewards.append(avg_reward)
                print(f"Task {task_id}: {avg_reward:.2f}")
    
    overall_avg_reward = np.mean(task_rewards) if task_rewards else 0
    print(f"Overall average reward: {overall_avg_reward:.2f}")
    
    # 3. Catastrophic Forgetting Analysis
    print(f"\n3. CATASTROPHIC FORGETTING ANALYSIS:")
    print("-" * 40)
    
    cf_matrix = metrics_summary['cf_matrix']
    
    # Generate forgetting matrix visualization
    generate_forgetting_matrix_plot(cf_matrix, env.total_tasks, "MountainCar")
    
    # Training summary
    summary = trainer.get_training_summary()
    print(f"\nTRAINING SUMMARY:")
    print(f"Total episodes: {summary['total_episodes']}")
    print(f"Total steps: {summary['total_steps']}")
    print(f"Final epsilon: {summary['final_epsilon']:.3f}")
    
    # Generate core metrics plots
    if convergence_data and task_rewards:
        generate_core_metrics_plots(convergence_data, task_rewards, env.total_tasks)

def generate_forgetting_matrix_plot(cf_matrix, total_tasks, env_name):
    """Generate forgetting matrix visualization window"""
    print(f"\nGenerating {env_name} forgetting matrix visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create matrix data for visualization
    display_matrix = np.full((total_tasks, total_tasks), np.nan)
    for i in range(total_tasks):
        for j in range(total_tasks):
            if j > i:  # Only upper triangle
                display_matrix[i, j] = cf_matrix[i, j]
    
    # Create heatmap
    im = ax.imshow(display_matrix, cmap='Reds', aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(total_tasks))
    ax.set_yticks(range(total_tasks))
    ax.set_xticklabels([f'T{i}' for i in range(total_tasks)])
    ax.set_yticklabels([f'T{i}' for i in range(total_tasks)])
    
    # Add text annotations
    for i in range(total_tasks):
        for j in range(total_tasks):
            if j <= i:
                text = ax.text(j, i, 'n/a', ha="center", va="center", 
                             color="black", fontsize=10, weight='bold')
            else:
                value = cf_matrix[i, j]
                if value == 0:
                    text = ax.text(j, i, '0.0', ha="center", va="center", 
                                 color="black", fontsize=10)
                else:
                    text = ax.text(j, i, f'{value:.1f}', ha="center", va="center", 
                                 color="white" if value > 0.5 else "black", fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Drop', rotation=270, labelpad=20)
    
    # Set title and labels
    ax.set_title(f'{env_name} Catastrophic Forgetting Matrix\n(Values show performance drop after training on subsequent tasks)')
    ax.set_xlabel('Subsequent Tasks')
    ax.set_ylabel('Original Tasks')
    
    # Invert y-axis to show T0 at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(f'visualizations/{env_name.lower()}_forgetting_matrix.png', 
                dpi=300, bbox_inches='tight')
    
    # Show window
    plt.show()
    
    print(f"Forgetting matrix visualization saved to: visualizations/{env_name.lower()}_forgetting_matrix.png")

def generate_core_metrics_plots(convergence_data, task_rewards, total_tasks):
    """Generate plots for the two core metrics"""
    print("\nGenerating core metrics plots...")
    os.makedirs("visualizations", exist_ok=True)
    
    plt.figure(figsize=(10, 4))
    
    # 1. Convergence Speed
    plt.subplot(1, 2, 1)
    task_ids = list(convergence_data.keys())
    conv_speeds = [convergence_data[tid] for tid in task_ids]
    
    bars = plt.bar(task_ids, conv_speeds, alpha=0.7, color='skyblue')
    plt.title('Convergence Speed')
    plt.xlabel('Task ID')
    plt.ylabel('Episodes to Converge')
    plt.xticks(task_ids, [f'Task {tid}' for tid in task_ids])
    plt.grid(True, alpha=0.3)
    
    for bar, speed in zip(bars, conv_speeds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{speed}', ha='center', va='bottom')
    
    # 2. Average Reward
    plt.subplot(1, 2, 2)
    task_ids = list(range(total_tasks))
    
    bars = plt.bar(task_ids, task_rewards, alpha=0.7, color='lightgreen')
    plt.title('Average Reward')
    plt.xlabel('Task ID')
    plt.ylabel('Average Reward')
    plt.xticks(task_ids, [f'Task {tid}' for tid in task_ids])
    plt.grid(True, alpha=0.3)
    
    for bar, reward in zip(bars, task_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{reward:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('visualizations/task2_core_metrics_mountaincar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Core metrics plots saved to: visualizations/task2_core_metrics_mountaincar.png")

if __name__ == "__main__":
    main()
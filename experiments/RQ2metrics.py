import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List

def plot_rq2_results(cartpole_results: Dict, mountaincar_results: Dict, 
                     save_path: str = "rq2_results.png"):
    """
    生成RQ2的完整可视化报告
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 主性能对比（左上）
    ax1 = plt.subplot(2, 3, 1)
    conditions = list(cartpole_results.keys())
    cartpole_means = [np.mean([r['avg_eval'] for r in cartpole_results[c]]) for c in conditions]
    cartpole_stds = [np.std([r['avg_eval'] for r in cartpole_results[c]]) for c in conditions]
    
    x = np.arange(len(conditions))
    ax1.bar(x, cartpole_means, yerr=cartpole_stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Average Evaluation Reward')
    ax1.set_title('CartPole: Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. MountainCar对比（右上）
    ax2 = plt.subplot(2, 3, 2)
    mc_conditions = list(mountaincar_results.keys())
    mc_means = [np.mean([r['avg_eval'] for r in mountaincar_results[c]]) for c in mc_conditions]
    mc_stds = [np.std([r['avg_eval'] for r in mountaincar_results[c]]) for c in mc_conditions]
    
    x2 = np.arange(len(mc_conditions))
    ax2.bar(x2, mc_means, yerr=mc_stds, capsize=5, alpha=0.7, color='coral')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Average Evaluation Reward')
    ax2.set_title('MountainCar: Performance Comparison')
    ax2.set_xticklabels(mc_conditions, rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Imagination效果（中上）
    ax3 = plt.subplot(2, 3, 3)
    # 对比有/无imagination的Dreamer
    dreamer_with = cartpole_means[conditions.index('dreamer_style')]
    dreamer_without = cartpole_means[conditions.index('dreamer_no_imagination')]
    
    bars = ax3.bar(['With\nImagination', 'Without\nImagination'], 
                   [dreamer_with, dreamer_without],
                   color=['green', 'gray'], alpha=0.7)
    ax3.set_ylabel('Performance')
    ax3.set_title('Imagination Benefit')
    ax3.text(0, dreamer_with + 5, f'+{dreamer_with - dreamer_without:.1f}', 
             ha='center', fontweight='bold')
    
    # 4. 参数效率（左下）
    ax4 = plt.subplot(2, 3, 4)
    # 计算参数效率：performance / parameters
    # small_fixed: 64维, large_fixed: 128维, fully_adaptive: 64维policy+72维WM
    params = {
        'small_fixed': 64,
        'large_fixed': 128,
        'fully_adaptive': 64 + 72
    }
    efficiency = {}
    for cond in ['small_fixed', 'large_fixed', 'fully_adaptive']:
        if cond in conditions:
            idx = conditions.index(cond)
            efficiency[cond] = cartpole_means[idx] / params[cond]
    
    ax4.bar(efficiency.keys(), efficiency.values(), color='purple', alpha=0.7)
    ax4.set_ylabel('Performance per Parameter')
    ax4.set_title('Parameter Efficiency')
    ax4.set_xticklabels(efficiency.keys(), rotation=45, ha='right')
    
    # 5. 容量调整历史（中下）
    ax5 = plt.subplot(2, 3, 5)
    # 如果有capacity_history数据
    if 'fully_adaptive' in cartpole_results and len(cartpole_results['fully_adaptive']) > 0:
        run = cartpole_results['fully_adaptive'][0]
        if 'capacity_history' in run:
            cap_hist = run['capacity_history']
            episodes = [c['episode'] for c in cap_hist]
            policy_caps = [c['policy_hidden_dim'] for c in cap_hist]
            wm_caps = [c['world_model_hidden_dim'] for c in cap_hist]
            
            ax5.plot(episodes, policy_caps, label='Policy', linewidth=2)
            ax5.plot(episodes, wm_caps, label='World Model', linewidth=2)
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Hidden Dimension')
            ax5.set_title('Capacity Evolution (Fully Adaptive)')
            ax5.legend()
            ax5.grid(alpha=0.3)
    
    # 6. 关键对比总结（右下）
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 计算关键指标
    imagination_gain = dreamer_with - dreamer_without
    adaptive_gain = cartpole_means[conditions.index('fully_adaptive')] - cartpole_means[conditions.index('small_fixed')]
    
    summary_text = f"""
    Key Findings (CartPole):
    
    ✓ Imagination Benefit
      {imagination_gain:+.1f} points
      ({imagination_gain/dreamer_without*100:.1f}% improvement)
    
    ✓ Adaptive System Gain  
      {adaptive_gain:+.1f} points over baseline
      ({adaptive_gain/cartpole_means[0]*100:.1f}% improvement)
    
    ✓ Parameter Efficiency
      Fully adaptive: {efficiency['fully_adaptive']:.3f}
      Large fixed: {efficiency['large_fixed']:.3f}
      Ratio: {efficiency['fully_adaptive']/efficiency['large_fixed']:.2f}x
    
    ✓ Best Performance
      {max(cartpole_means):.1f} (fully_adaptive)
    """
    
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {save_path}")
    plt.show()


# 使用示例
if __name__ == '__main__':
    # 假设已经运行了实验并有results
    # cartpole_results = main(...)  # 从cartpole实验获取
    # mountaincar_results = main_mountaincar(...)  # 从mountaincar实验获取
    
    # plot_rq2_results(cartpole_results, mountaincar_results)
    pass
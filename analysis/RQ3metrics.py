#!/usr/bin/env python3
"""
RQ3 Visualization and Metrics Analysis - FIXED VERSION
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Any
import pandas as pd


def plot_performance_comparison(all_results: Dict, save_dir: str = "./visualizations/rq3"):
    """Performance comparison bar chart (Core metric)"""
    os.makedirs(save_dir, exist_ok=True)
    
    conditions = list(all_results.keys())
    means = []
    stds = []
    
    for cond in conditions:
        perfs = [r['avg_performance'] for r in all_results[cond]]
        means.append(np.mean(perfs))
        stds.append(np.std(perfs))
    
    plt.figure(figsize=(12, 6))
    colors = ['red', 'blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray']
    colors = colors[:len(conditions)]
    
    bars = plt.bar(range(len(conditions)), means, yerr=stds, capsize=5, 
                   alpha=0.7, color=colors, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Experimental Condition')
    plt.ylabel('Average Performance')
    plt.title('RQ3: Overall Performance Comparison - Fixed Version')
    plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, means)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Performance comparison saved to {save_dir}/performance_comparison.png")


def plot_memory_efficiency(all_results: Dict, save_dir: str = "./visualizations/rq3"):
    """Performance vs Memory Trade-off scatter plot (Core innovation)"""
    os.makedirs(save_dir, exist_ok=True)
    
    conditions = list(all_results.keys())
    
    # Remove baseline_no_replay as it has 0 memory but distorts the scale
    plot_conditions = [c for c in conditions if c != "baseline_no_replay"]
    
    performances = []
    memories = []
    efficiencies = []
    labels = []
    
    for cond in plot_conditions:
        perf_vals = [r['avg_performance'] for r in all_results[cond]]
        mem_vals = [r['final_memory_mb'] for r in all_results[cond]]
        eff_vals = [r['memory_efficiency'] for r in all_results[cond]]
        
        performances.append(np.mean(perf_vals))
        memories.append(np.mean(mem_vals))
        efficiencies.append(np.mean(eff_vals))
        labels.append(cond)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(memories, performances, s=150, c=efficiencies, 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add labels for each point
    for i, (x, y, label) in enumerate(zip(memories, performances, labels)):
        plt.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.8)
    
    plt.colorbar(scatter, label='Memory Efficiency (Performance/MB)')
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Average Performance')
    plt.title('RQ3: Performance vs Memory Trade-off Analysis')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/memory_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Memory efficiency plot saved to {save_dir}/memory_efficiency.png")


def plot_learning_curves(all_results: Dict, save_dir: str = "./visualizations/rq3"):
    """Learning curves comparison (Training dynamics)"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Select key conditions for clarity
    key_conditions = ["limited_replay", "latent_replay", "hybrid_uncertainty", "hybrid_distill"]
    
    plt.figure(figsize=(12, 6))
    
    for condition in key_conditions:
        if condition in all_results:
            # Take first seed for demonstration
            if len(all_results[condition]) > 0:
                episode_rewards = all_results[condition][0]['episode_rewards']
                
                # Smooth the curve for better visualization
                window_size = 20
                smoothed_rewards = np.convolve(episode_rewards, 
                                             np.ones(window_size)/window_size, 
                                             mode='valid')
                
                plt.plot(smoothed_rewards, label=condition, linewidth=2, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (window=20)')
    plt.title('RQ3: Learning Curves Comparison - Key Methods')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Learning curves saved to {save_dir}/learning_curves.png")


def plot_compression_efficiency(all_results: Dict, save_dir: str = "./visualizations/rq3"):
    """Compression efficiency vs performance analysis (Technical contribution)"""
    os.makedirs(save_dir, exist_ok=True)
    
    methods = ["latent_replay", "hybrid_basic", "hybrid_uncertainty"]
    
    # Get limited_replay performance as baseline
    limited_perf = None
    if "limited_replay" in all_results and len(all_results["limited_replay"]) > 0:
        limited_perf = np.mean([r['avg_performance'] for r in all_results["limited_replay"]])
    
    if limited_perf is None or limited_perf == 0:
        print("⚠️ Cannot compute compression efficiency: limited_replay baseline not available")
        return
    
    compression_ratios = []
    performance_ratios = []
    available_methods = []
    
    for method in methods:
        if method in all_results and len(all_results[method]) > 0:
            comp_ratio = np.mean([r['compression_ratio'] for r in all_results[method]])
            perf_vals = [r['avg_performance'] for r in all_results[method]]
            mean_perf = np.mean(perf_vals)
            
            compression_ratios.append(comp_ratio)
            performance_ratios.append(mean_perf / limited_perf)
            available_methods.append(method)
    
    if not available_methods:
        print("⚠️ No compression methods available for analysis")
        return
    
    x = np.arange(len(available_methods))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Compression ratio bars
    bars1 = ax1.bar(x - width/2, compression_ratios, width, 
                   label='Compression Ratio', alpha=0.7, color='blue', edgecolor='black')
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Compression Ratio', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_methods, rotation=45, ha='right')
    
    # Performance ratio bars (secondary axis)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, performance_ratios, width, 
                   label='Performance Ratio (vs Limited)', alpha=0.7, color='orange', edgecolor='black')
    
    ax2.set_ylabel('Performance Ratio (vs Limited Replay)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Add value labels
    for bars, values in [(bars1, compression_ratios), (bars2, performance_ratios)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.title('RQ3: Compression Efficiency vs Performance Preservation')
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/compression_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Compression efficiency analysis saved to {save_dir}/compression_efficiency.png")


def create_comprehensive_analysis(all_results: Dict, save_dir: str = "./visualizations/rq3"):
    """Create all visualizations for comprehensive analysis"""
    print("\n📊 Generating comprehensive RQ3 visualizations...")
    
    # Create all plots
    plot_performance_comparison(all_results, save_dir)
    plot_memory_efficiency(all_results, save_dir)
    plot_learning_curves(all_results, save_dir)
    plot_compression_efficiency(all_results, save_dir)
    
    # Create summary statistics file
    summary = {}
    for condition in all_results.keys():
        if len(all_results[condition]) > 0:
            summary[condition] = {
                'mean_performance': float(np.mean([r['avg_performance'] for r in all_results[condition]])),
                'std_performance': float(np.std([r['avg_performance'] for r in all_results[condition]])),
                'mean_memory_mb': float(np.mean([r['final_memory_mb'] for r in all_results[condition]])),
                'mean_memory_efficiency': float(np.mean([r['memory_efficiency'] for r in all_results[condition]])),
                'mean_compression_ratio': float(np.mean([r['compression_ratio'] for r in all_results[condition]])),
                'mean_forward_transfer': float(np.mean([r['forward_transfer'] for r in all_results[condition]])),
                'mean_backward_transfer': float(np.mean([r['backward_transfer'] for r in all_results[condition]])),
            }
    
    with open(f'{save_dir}/rq3_comprehensive_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Comprehensive analysis saved to {save_dir}/")
    print(f"✅ Summary statistics saved to {save_dir}/rq3_comprehensive_summary.json")
    
    return summary
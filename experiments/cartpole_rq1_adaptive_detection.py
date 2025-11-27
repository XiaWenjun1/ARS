#!/usr/bin/env python3
"""
完全修复版：解决warmup和过度检测问题（已更新：使用 detectors 返回的 confidence、multi-modal 默认按置信度投票、weighted detector 提供权重）
"""

from __future__ import annotations
import argparse, math, random, sys, os, time, json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cartpole_config import CartPoleConfig
from detection import (
    LatentSpaceDriftDetector,
    MultiModalDetector,
    PredictionErrorDetector,
    RewardTrendDetector,
    WeightedMultiModalDetector,
)
from detection.base import DetectionResult
from environments.cartpole_cl import CartPoleCL

def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class ExperimentConfig:
    name: str
    factory: Callable[[], object]
    detection_window: int = 15

def evaluate_detections(
    change_points: List[int], 
    detections: List[int], 
    detection_window: int
) -> Dict[str, float]:
    detections = sorted(detections)
    success_count = 0
    delays = []
    det_idx = 0
    
    for cp in change_points:
        while det_idx < len(detections) and detections[det_idx] < cp:
            det_idx += 1
        if det_idx < len(detections) and detections[det_idx] - cp <= detection_window:
            delays.append(detections[det_idx] - cp)
            success_count += 1
            det_idx += 1
    
    total_detections = len(detections)
    false_positives = total_detections - success_count
    recall = success_count / max(len(change_points), 1) if change_points else 0.0
    precision = success_count / max(total_detections, 1) if total_detections > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12) if (precision + recall) > 0 else 0.0
    avg_delay = float(np.mean(delays)) if delays else float('nan')
    
    return {
        "detections": total_detections,
        "recall": recall,
        "precision": precision,
        "false_positives": false_positives,
        "f1": f1,
        "avg_delay": avg_delay,
    }

def build_detector_configs(state_dim: int, action_dim: int):
    """修复后的检测器配置（注意：MultiModalDetector 使用置信度投票）"""
    
    # Reward Trend
    reward_kwargs = dict(
        window_size=3,
        baseline_window=15,
        drop_threshold=0.22,
        confirm_steps=2,
        cooldown_episodes=15,
    )
    
    # Latent Space
    latent_kwargs = dict(
        drift_threshold=1.4,
        window_size=20,
        baseline_window=50,
        confirm_steps=3,
        cooldown_episodes=18,
    )
    
    # Prediction Error
    prediction_kwargs = dict(
        ratio_threshold=2.0,
        window_size=20,
        confirm_steps=3,
        cooldown_episodes=18,
    )

    # weights for weighted multimodal: [reward, prediction, latent]
    default_weights = [1.4, 1.2, 0.7]
    
    configs = [
        ExperimentConfig(
            name="reward_trend", 
            factory=lambda: RewardTrendDetector(**reward_kwargs)
        ),
        ExperimentConfig(
            name="prediction_error", 
            factory=lambda: PredictionErrorDetector(
                state_dim, action_dim, **prediction_kwargs
            )
        ),
        ExperimentConfig(
            name="latent_space", 
            factory=lambda: LatentSpaceDriftDetector(state_dim, **latent_kwargs)
        ),
        ExperimentConfig(
            name="reward+prediction", 
            factory=lambda: MultiModalDetector(
                [
                    RewardTrendDetector(**reward_kwargs), 
                    PredictionErrorDetector(state_dim, action_dim, **prediction_kwargs)
                ],
                vote_threshold=0.5,
            )
        ),
        ExperimentConfig(
            name="reward+latent", 
            factory=lambda: MultiModalDetector(
                [
                    RewardTrendDetector(**reward_kwargs), 
                    LatentSpaceDriftDetector(state_dim, **latent_kwargs)
                ],
                vote_threshold=0.5,
            )
        ),
        ExperimentConfig(
            name="prediction+latent", 
            factory=lambda: MultiModalDetector(
                [
                    PredictionErrorDetector(state_dim, action_dim, **prediction_kwargs), 
                    LatentSpaceDriftDetector(state_dim, **latent_kwargs)
                ],
                vote_threshold=0.5,
            )
        ),
        # OR (any)
        ExperimentConfig(
            name="all_three_ANY",
            factory=lambda: MultiModalDetector(
                [
                    RewardTrendDetector(**reward_kwargs),
                    PredictionErrorDetector(state_dim, action_dim, **prediction_kwargs),
                    LatentSpaceDriftDetector(state_dim, **latent_kwargs)
                ],
                vote_threshold=0.33,
            )
        ),
        # MAJORITY
        ExperimentConfig(
            name="all_three_MAJORITY",
            factory=lambda: MultiModalDetector(
                [
                    RewardTrendDetector(**reward_kwargs),
                    PredictionErrorDetector(state_dim, action_dim, **prediction_kwargs),
                    LatentSpaceDriftDetector(state_dim, **latent_kwargs)
                ],
                vote_threshold=0.67,
            )
        ),
        # WEIGHTED
        ExperimentConfig(
            name="all_three_WEIGHTED",
            factory=lambda: WeightedMultiModalDetector(
                [
                    RewardTrendDetector(**reward_kwargs),
                    PredictionErrorDetector(state_dim, action_dim, **prediction_kwargs),
                    LatentSpaceDriftDetector(state_dim, **latent_kwargs)
                ],
                vote_threshold=0.4,
                detector_weights=default_weights,
            )
        ),
    ]

    class NullDetector:
        def __init__(self):
            self.name = "no_detector"
        def reset(self):
            pass
        def update(self, state, action, reward, next_state, done, info=None):
            return DetectionResult(detected=False, score=0.0, metadata={})

    configs.append(ExperimentConfig(name="no_detector", factory=lambda: NullDetector()))
    return configs

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

class WeightedReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.weights = []
        self.max_size = max_size
        self.pos = 0
    
    def push(self, transition, weight=1.0):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
            self.weights.append(weight)
        else:
            self.buffer[self.pos] = transition
            self.weights[self.pos] = weight
            self.pos = (self.pos + 1) % self.max_size
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], [], [], []
        weights = np.clip(np.array(self.weights, dtype=np.float64), 0, None)
        total = np.sum(weights)
        probs = (np.ones(len(weights)) / len(weights) if total <= 1e-8 
                else weights / total)
        batch_size = min(batch_size, len(self.buffer))
        idxs = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)
        batch = [self.buffer[i] for i in idxs]
        return map(np.array, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class DetectionAwareDQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3, gamma=0.99, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.epsilon = 0.05
        self.base_epsilon = 0.05
        self.adapted_epsilon = 0.18
        self.adapt_episodes_total = 10
        self.adapt_episodes_remaining = 0
        
        self.weight_for_new_env = 1.3
        self.current_env_weight = 1.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = WeightedReplayBuffer(max_size=10000)
        
        self.steps_done = 0
        self.update_target_every = 200

    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            qvals = self.policy_net(state_t)
            return int(torch.argmax(qvals).item())

    def push_transition(self, state, action, reward, next_state, done, detector_confidence=1.0):
        weight = self.current_env_weight * (0.3 + 0.7 * detector_confidence)
        self.replay_buffer.push((state, action, reward, next_state, done), weight=weight)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        if len(states) == 0:
            return
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def on_detection(self, detector_name: str, episode: int, metadata: dict):
        if self.adapt_episodes_remaining <= 0:
            confidence = metadata.get("confidence") if isinstance(metadata, dict) else None
            if confidence is None:
                raw_score = metadata.get("score", 1.0) if isinstance(metadata, dict) else 1.0
                if "reward" in detector_name.lower():
                    confidence = min(0.95, max(0.5, raw_score / 40.0))
                elif "prediction" in detector_name.lower():
                    confidence = min(0.95, max(0.5, (raw_score - 1.0) / 2.5))
                elif "latent" in detector_name.lower():
                    confidence = min(0.95, max(0.5, raw_score / 2.5))
                else:
                    confidence = 0.6
            confidence = float(max(0.0, min(1.0, confidence)))

            epsilon_boost = (self.adapted_epsilon - self.base_epsilon) * confidence
            self.epsilon = self.base_epsilon + epsilon_boost
            self.adapt_episodes_remaining = int(self.adapt_episodes_total * confidence)
            self.current_env_weight = 1.0 + (self.weight_for_new_env - 1.0) * confidence

    def post_episode(self):
        if self.adapt_episodes_remaining > 0:
            self.adapt_episodes_remaining -= 1
            if self.adapt_episodes_remaining <= 0:
                self.epsilon = self.base_epsilon
                self.current_env_weight = 1.0

def run_training_with_detector(cfg, detector_factory, seed, episodes_per_task, cycles, warmup_episodes=30):
    set_global_seed(seed)
    env = CartPoleCL(cfg.TASKS)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    detector = detector_factory() if detector_factory is not None else None
    if detector is not None and hasattr(detector, "reset"):
        detector.reset()
    detector_name = getattr(detector, "name", "no_detector") if detector else "no_detector"
    
    agent = DetectionAwareDQNAgent(state_dim, action_dim)
    
    task_sequence = []
    for _ in range(cycles):
        for task_id in range(env.total_tasks):
            task_sequence.extend([task_id] * episodes_per_task)
    
    episode_rewards = []
    detection_episodes = []
    change_points = []
    
    current_task = 0
    env.change_task(current_task)
    episodes_completed = 0

    for episode_idx in range(len(task_sequence)):
        desired_task = task_sequence[episode_idx]
        if desired_task != env.current_task:
            env.change_task(desired_task)
            change_points.append(episode_idx)
            current_task = desired_task
        
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(np.array(state))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            
            confidence = 1.0
            if detector is not None:
                result = detector.update(
                    np.array(state, dtype=np.float32),
                    int(action),
                    float(reward),
                    np.array(next_state, dtype=np.float32),
                    done,
                    info={"task_id": env.current_task}
                )
                md = result.metadata if isinstance(result.metadata, dict) else {}
                confidence = md.get("confidence") if isinstance(md.get("confidence"), (int, float)) else None
                if confidence is None:
                    raw_score = float(getattr(result, "score", 0.0) or 0.0)
                    confidence = float(max(0.0, min(1.0, (raw_score) / (raw_score + 1.0))))

                if episodes_completed >= warmup_episodes:
                    if result.detected:
                        if episode_idx not in detection_episodes:
                            detection_episodes.append(episode_idx)
                            agent.on_detection(detector_name, episode_idx, metadata=md)

            agent.push_transition(state, action, reward, next_state, done, confidence)
            agent.update()
            
            episode_reward += float(reward)
            state = next_state
        
        episode_rewards.append(episode_reward)
        agent.post_episode()
        episodes_completed += 1
    
    eval_rewards = {}
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    for task_id in range(env.total_tasks):
        env.change_task(task_id)
        task_rewards = []
        for _ in range(10):
            s, _ = env.reset()
            total = 0.0
            for _ in range(cfg.MAX_STEPS_PER_EPISODE):
                a = agent.select_action(np.array(s))
                s, r, t, tr, _ = env.step(a)
                total += r
                if t or tr:
                    break
            task_rewards.append(total)
        eval_rewards[task_id] = float(np.mean(task_rewards))
    agent.epsilon = original_epsilon
    
    det_metrics = (evaluate_detections(change_points, detection_episodes, detection_window=15) 
                  if detector else {})
    
    return float(np.mean(episode_rewards)), eval_rewards, detection_episodes, det_metrics

def create_visualizations(summary_data, cfg, save_dir):
    """Generates visualizations for RQ1"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Task Performance Heatmap
    task_names = [f"T{i}\n({cfg.TASKS[i]['task_name']})" for i in range(len(cfg.TASKS))]
    detector_names = [data['name'] for data in summary_data]

    performance_matrix = np.zeros((len(detector_names), len(task_names)))
    for i, data in enumerate(summary_data):
        for j in range(len(task_names)):
            task_rewards = [r['eval_rewards'].get(j, 0.0) for r in data['all_results']]
            performance_matrix[i, j] = np.mean(task_rewards)

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
    plt.savefig(os.path.join(save_dir, "task_performance_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Detector Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    summary_data_sorted = sorted(summary_data, key=lambda x: x['mean_eval'], reverse=True)
    names = [data['name'] for data in summary_data_sorted]
    evals = [data['mean_eval'] for data in summary_data_sorted]
    eval_err = [data['std_eval'] for data in summary_data_sorted]

    colors = plt.cm.YlGnBu(np.linspace(0.3, 0.9, len(names))) if len(names) > 0 else []

    bars1 = ax1.bar(range(len(names)), evals, yerr=eval_err,
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Average Evaluation Reward')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=10)

    max_eval = max(evals) if evals else 1.0
    text_offset = max(1.0, 0.02 * max_eval)
    for bar, val in zip(bars1, evals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + text_offset,
                 f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

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
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (prec, rec) in enumerate(zip(precisions, recalls)):
        if prec > 0:
            ax2.text(i - width / 2, prec + 0.02, f'{prec:.2f}', ha='center', va='bottom', fontsize=8)
        if rec > 0:
            ax2.text(i + width / 2, rec + 0.02, f'{rec:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "detector_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main(seeds=[0, 1, 2], episodes_per_task=50, cycles=1, warmup_episodes=20):
    # === 📂 Path Management ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "rq1_cartpole")
    vis_dir = os.path.join(os.path.dirname(base_dir), "visualizations", "rq1_cartpole")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📂 Results will be saved to: {results_dir}")
    print(f"📊 Visualizations will be saved to: {vis_dir}")
    # ==========================

    cfg = CartPoleConfig()
    dummy_env = CartPoleCL(cfg.TASKS)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    detector_configs = build_detector_configs(state_dim, action_dim)
    
    print("=" * 80)
    print(f"Experimental setup:")
    print(f"  Seeds: {seeds}")
    print(f"  Episodes per task: {episodes_per_task}")
    print(f"  Cycles: {cycles}")
    print(f"  Total episodes: {episodes_per_task * len(cfg.TASKS) * cycles}")
    print(f"  Warmup episodes: {warmup_episodes}")
    print("=" * 80)
    
    all_results = {}
    
    for exp in detector_configs:
        exp_results = []
        
        for seed in seeds:
            set_global_seed(seed)
            print(f"\n[{exp.name}] Seed {seed}...")
            
            avg_train, eval_rewards, detections, det_metrics = run_training_with_detector(
                cfg, exp.factory if exp.name != "no_detector" else None,
                seed=seed,
                episodes_per_task=episodes_per_task,
                cycles=cycles,
                warmup_episodes=warmup_episodes
            )
            
            avg_eval = float(np.mean(list(eval_rewards.values())))
            n_det = len(detections)
            prec = det_metrics.get("precision", float('nan')) if det_metrics else float('nan')
            rec = det_metrics.get("recall", float('nan')) if det_metrics else float('nan')
            
            # === 💾 Save JSON ===
            seed_result = {
                'seed': seed,
                'detector': exp.name,
                'avg_train': avg_train,
                'avg_eval': avg_eval,
                'n_detections': n_det,
                'precision': prec,
                'recall': rec,
                'eval_rewards': eval_rewards,
                'detections': detections,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            exp_results.append(seed_result)
            
            def convert_numpy(obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return obj

            json_filename = f'temp_results_{exp.name}_seed{seed}.json'
            json_path = os.path.join(results_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(seed_result, f, indent=2, default=convert_numpy)
            # ====================
            
            print(f"  → avg_eval={avg_eval:.1f}, detections={n_det}, P={prec:.2f}, R={rec:.2f}")
        
        all_results[exp.name] = exp_results
    
    # 打印汇总表格（保持原样）
    print("\n" + "=" * 80)
    print("Summary Results (mean ± std)")
    print("=" * 80)
    print(f"{'Detector':<25} {'Avg Eval':<15} {'Detections':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    summary_data = []
    for name, results in all_results.items():
        avg_evals = [r['avg_eval'] for r in results]
        n_dets = [r['n_detections'] for r in results]
        precs = [r['precision'] for r in results if not math.isnan(r['precision'])]
        recs = [r['recall'] for r in results if not math.isnan(r['recall'])]
        
        summary_data.append({
            'name': name,
            'mean_eval': np.mean(avg_evals),
            'std_eval': np.std(avg_evals),
            'mean_det': np.mean(n_dets),
            'mean_prec': np.mean(precs) if precs else float('nan'),
            'mean_rec': np.mean(recs) if recs else float('nan'),
            'all_results': results
        })
    
    summary_data.sort(key=lambda x: x['mean_eval'], reverse=True)
    for data in summary_data:
        prec_str = f"{data['mean_prec']:.2f}" if not math.isnan(data['mean_prec']) else "N/A"
        rec_str = f"{data['mean_rec']:.2f}" if not math.isnan(data['mean_rec']) else "N/A"
        print(f"{data['name']:<25} {data['mean_eval']:>6.1f} ± {data['std_eval']:<5.1f} "
              f"{data['mean_det']:>10.0f}  {prec_str:>10}  {rec_str:>10}")
    
    print("=" * 80)
    
    # 统计检验（保持原样）
    if 'no_detector' in all_results:
        baseline_evals = [r['avg_eval'] for r in all_results['no_detector']]
        baseline_mean = np.mean(baseline_evals)
        print("\nStatistical Significance Test (vs Baseline):")
        print("-" * 80)
        from scipy import stats as sp_stats
        for data in summary_data:
            if data['name'] == 'no_detector': continue
            detector_evals = [r['avg_eval'] for r in all_results[data['name']]]
            if len(detector_evals) > 1 and len(baseline_evals) > 1:
                t_stat, p_value = sp_stats.ttest_ind(detector_evals, baseline_evals)
                improvement = data['mean_eval'] - baseline_mean
                sig_marker = "✓" if p_value < 0.05 else " "
                print(f"{data['name']:<25} t={t_stat:>6.2f}, p={p_value:.4f} {sig_marker}  Δ={improvement:+.1f}")

    # === 📊 Visualization ===
    print(f"\n📈 Generating visualizations in {vis_dir}...")
    create_visualizations(summary_data, cfg, vis_dir)
    print("✅ Visualizations saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--episodes-per-task', type=int, default=100)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--warmup-episodes', type=int, default=40)
    parser.add_argument('--quick-test', action='store_true', help="Quick test with 1 seed")
    args = parser.parse_args()
    
    if args.quick_test:
        print("🚀 QUICK TEST MODE")
        main(seeds=[0], episodes_per_task=50, cycles=1, warmup_episodes=20)
    else:
        main(seeds=args.seeds, episodes_per_task=args.episodes_per_task, 
             cycles=args.cycles, warmup_episodes=args.warmup_episodes)
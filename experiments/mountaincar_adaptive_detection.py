#!/usr/bin/env python3
"""
MountainCar RQ1: Change Detection Experiment
测试多种检测器在MountainCar环境动力学变化中的表现
"""

from __future__ import annotations
import argparse, math, random, sys, os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.mountaincar_config import MountainCarConfig
from detection import (
    LatentSpaceDriftDetector,
    MultiModalDetector,
    PredictionErrorDetector,
    RewardTrendDetector,
    WeightedMultiModalDetector,
)
from detection.base import DetectionResult
from environments.mountaincar_cl import MountainCarCL
from RQ1metrics import plot_task_performance_heatmap, plot_detector_comparison


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class ExperimentConfig:
    name: str
    factory: Callable[[], object]
    detection_window: int = 20  # MountainCar需要更长的检测窗口


def evaluate_detections(
    change_points: List[int], 
    detections: List[int], 
    detection_window: int
) -> Dict[str, float]:
    """评估检测性能"""
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
    """构建MountainCar专用的检测器配置
    
    MountainCar特点：
    - 更稀疏的reward信号
    - 环境动力学变化更微妙
    - 需要更多episodes才能稳定
    """
    
    # Reward Trend - MountainCar的reward稀疏，需要更宽松的阈值
    reward_kwargs = dict(
        window_size=5,           # 更长窗口
        baseline_window=20,       # 更长基线
        drop_threshold=0.30,      # 更高阈值（reward变化更大）
        confirm_steps=3,
        cooldown_episodes=20,     # 更长冷却
    )
    
    # Latent Space - 检测状态分布变化
    latent_kwargs = dict(
        drift_threshold=1.6,      # 略高阈值
        window_size=25,
        baseline_window=60,
        confirm_steps=3,
        cooldown_episodes=25,
    )
    
    # Prediction Error - 检测环境动力学变化
    prediction_kwargs = dict(
        ratio_threshold=2.2,      # 略高阈值
        window_size=25,
        confirm_steps=3,
        cooldown_episodes=25,
    )

    # Weighted detector的权重 [reward, prediction, latent]
    default_weights = [1.3, 1.3, 0.8]  # prediction和reward同等重要
    
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
        # OR (any) - 任一检测器触发
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
        # MAJORITY - 多数投票
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
        # WEIGHTED - 加权投票
        ExperimentConfig(
            name="all_three_WEIGHTED",
            factory=lambda: WeightedMultiModalDetector(
                [
                    RewardTrendDetector(**reward_kwargs),
                    PredictionErrorDetector(state_dim, action_dim, **prediction_kwargs),
                    LatentSpaceDriftDetector(state_dim, **latent_kwargs)
                ],
                vote_threshold=0.45,  # 略高于CartPole
                detector_weights=default_weights,
            )
        ),
    ]

    # No detector baseline
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
    """DQN for MountainCar"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):  # 更大的网络
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
    """带权重的replay buffer"""
    def __init__(self, max_size=20000):  # MountainCar需要更大buffer
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
    """Detection-aware DQN agent for MountainCar"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 lr=5e-4, gamma=0.99, batch_size=128):  # MountainCar调优参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # MountainCar需要更多探索
        self.epsilon = 0.15              # 基础探索率更高
        self.base_epsilon = 0.15
        self.adapted_epsilon = 0.35      # 检测后探索率
        self.adapt_episodes_total = 15   # 更长的适应期
        self.adapt_episodes_remaining = 0
        
        # Replay buffer权重调整
        self.weight_for_new_env = 1.5    # 新环境数据权重更高
        self.current_env_weight = 1.0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络初始化
        self.policy_net = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = WeightedReplayBuffer(max_size=20000)
        
        self.steps_done = 0
        self.update_target_every = 300  # 更频繁的target更新

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        self.steps_done += 1
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            qvals = self.policy_net(state_t)
            return int(torch.argmax(qvals).item())

    def push_transition(self, state, action, reward, next_state, done, 
                       detector_confidence=1.0):
        """添加transition到replay buffer，带检测器置信度加权"""
        weight = self.current_env_weight * (0.3 + 0.7 * detector_confidence)
        self.replay_buffer.push(
            (state, action, reward, next_state, done), 
            weight=weight
        )

    def update(self):
        """更新policy network"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        if len(states) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = nn.MSELoss()(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def on_detection(self, detector_name: str, episode: int, metadata: dict):
        """处理检测事件：调整探索率和buffer权重"""
        if self.adapt_episodes_remaining <= 0:
            # 获取检测器置信度
            confidence = metadata.get("confidence") if isinstance(metadata, dict) else None
            
            if confidence is None:
                # Fallback: 从score推断置信度
                raw_score = metadata.get("score", 1.0) if isinstance(metadata, dict) else 1.0
                
                if "reward" in detector_name.lower():
                    confidence = min(0.95, max(0.5, raw_score / 50.0))
                elif "prediction" in detector_name.lower():
                    confidence = min(0.95, max(0.5, (raw_score - 1.0) / 3.0))
                elif "latent" in detector_name.lower():
                    confidence = min(0.95, max(0.5, raw_score / 3.0))
                else:
                    confidence = 0.65
            
            # Clamp confidence
            confidence = float(max(0.0, min(1.0, confidence)))

            # 调整epsilon
            epsilon_boost = (self.adapted_epsilon - self.base_epsilon) * confidence
            self.epsilon = self.base_epsilon + epsilon_boost
            
            # 设置适应期
            self.adapt_episodes_remaining = int(self.adapt_episodes_total * confidence)
            
            # 调整buffer权重
            self.current_env_weight = 1.0 + (self.weight_for_new_env - 1.0) * confidence

    def post_episode(self):
        """Episode结束后的处理"""
        if self.adapt_episodes_remaining > 0:
            self.adapt_episodes_remaining -= 1
            if self.adapt_episodes_remaining <= 0:
                # 恢复正常状态
                self.epsilon = self.base_epsilon
                self.current_env_weight = 1.0


def run_training_with_detector(
    cfg: MountainCarConfig,
    detector_factory: Optional[Callable[[], object]],
    seed: int,
    episodes_per_task: int,
    cycles: int,
    warmup_episodes: int = 50
):
    """运行带检测器的训练"""
    set_global_seed(seed)
    
    # 初始化环境
    env = MountainCarCL(cfg.TASKS)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化检测器
    detector = detector_factory() if detector_factory is not None else None
    if detector is not None and hasattr(detector, "reset"):
        detector.reset()
    detector_name = getattr(detector, "name", "no_detector") if detector else "no_detector"
    
    # 初始化agent
    agent = DetectionAwareDQNAgent(state_dim, action_dim)
    
    # 构建task sequence
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

    # 训练循环
    for episode_idx in range(len(task_sequence)):
        desired_task = task_sequence[episode_idx]
        
        # Task切换
        if desired_task != env.current_task:
            env.change_task(desired_task)
            change_points.append(episode_idx)
            current_task = desired_task
        
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        # Episode循环
        while not done:
            action = agent.select_action(np.array(state))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            
            confidence = 1.0
            
            # 检测器更新
            if detector is not None:
                result = detector.update(
                    np.array(state, dtype=np.float32),
                    int(action),
                    float(reward),
                    np.array(next_state, dtype=np.float32),
                    done,
                    info={"task_id": env.current_task}
                )

                # 获取置信度
                md = result.metadata if isinstance(result.metadata, dict) else {}
                confidence = md.get("confidence")
                if confidence is None:
                    raw_score = float(getattr(result, "score", 0.0) or 0.0)
                    confidence = float(max(0.0, min(1.0, raw_score / (raw_score + 1.0))))

                # Warmup后才触发检测响应
                if episodes_completed >= warmup_episodes:
                    if result.detected:
                        if episode_idx not in detection_episodes:
                            detection_episodes.append(episode_idx)
                            agent.on_detection(detector_name, episode_idx, metadata=md)

            # 存储transition并更新
            agent.push_transition(state, action, reward, next_state, done, confidence)
            agent.update()
            
            episode_reward += float(reward)
            state = next_state
        
        episode_rewards.append(episode_reward)
        agent.post_episode()
        episodes_completed += 1
    
    # 评估阶段
    eval_rewards = {}
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # 关闭探索
    
    for task_id in range(env.total_tasks):
        env.change_task(task_id)
        task_rewards = []
        
        for _ in range(10):  # 每个task评估10次
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
    
    # 计算检测指标
    det_metrics = (evaluate_detections(change_points, detection_episodes, 
                                      detection_window=20) 
                  if detector else {})
    
    return (float(np.mean(episode_rewards)), eval_rewards, 
            detection_episodes, det_metrics)


def main(
    seeds=[0, 1, 2],
    episodes_per_task=150,
    cycles=2,
    warmup_episodes=50
):
    """主实验函数"""
    cfg = MountainCarConfig()
    
    # 获取环境信息
    dummy_env = MountainCarCL(cfg.TASKS)
    state_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.n
    
    # 构建检测器配置
    detector_configs = build_detector_configs(state_dim, action_dim)
    
    print("=" * 80)
    print("MountainCar RQ1: Change Detection Experiment")
    print("=" * 80)
    print(f"Experimental setup:")
    print(f"  Seeds: {seeds}")
    print(f"  Episodes per task: {episodes_per_task}")
    print(f"  Cycles: {cycles}")
    print(f"  Total episodes: {episodes_per_task * len(cfg.TASKS) * cycles}")
    print(f"  Warmup episodes: {warmup_episodes}")
    print(f"  Tasks: {len(cfg.TASKS)}")
    for i, task in enumerate(cfg.TASKS):
        print(f"    T{i} ({task['task_name']}): gravity={task['gravity']}, force={task['force']}")
    print(f"  True boundaries: {[i*episodes_per_task for i in range(1, len(cfg.TASKS)*cycles)]}")
    print("=" * 80)
    
    all_results = {}
    
    # 运行所有检测器配置
    for exp in detector_configs:
        exp_results = []
        
        for seed in seeds:
            set_global_seed(seed)
            print(f"\n[{exp.name}] Seed {seed}...", end=" ")
            
            avg_train, eval_rewards, detections, det_metrics = run_training_with_detector(
                cfg, 
                exp.factory if exp.name != "no_detector" else None,
                seed=seed,
                episodes_per_task=episodes_per_task,
                cycles=cycles,
                warmup_episodes=warmup_episodes
            )
            
            avg_eval = float(np.mean(list(eval_rewards.values())))
            n_det = len(detections)
            prec = det_metrics.get("precision", float('nan')) if det_metrics else float('nan')
            rec = det_metrics.get("recall", float('nan')) if det_metrics else float('nan')
            
            exp_results.append({
                'seed': seed,
                'avg_train': avg_train,
                'avg_eval': avg_eval,
                'n_detections': n_det,
                'precision': prec,
                'recall': rec,
                'eval_rewards': eval_rewards
            })
            
            # 打印每个任务的得分
            print(f"avg_eval={avg_eval:.1f}, det={n_det}, P={prec:.2f}, R={rec:.2f}")
            print(f"       Task rewards: ", end="")
            for task_id, reward in eval_rewards.items():
                task_name = cfg.TASKS[task_id]['task_name']
                print(f"T{task_id}({task_name}):{reward:.1f} ", end="")
            print()
        
        all_results[exp.name] = exp_results
    
    # 汇总结果
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
        
        mean_eval = np.mean(avg_evals)
        std_eval = np.std(avg_evals)
        mean_det = np.mean(n_dets)
        mean_prec = np.mean(precs) if precs else float('nan')
        mean_rec = np.mean(recs) if recs else float('nan')
        
        summary_data.append({
            'name': name,
            'mean_eval': mean_eval,
            'std_eval': std_eval,
            'mean_det': mean_det,
            'mean_prec': mean_prec,
            'mean_rec': mean_rec,
            'all_results': results
        })
    
    # 按性能排序
    summary_data.sort(key=lambda x: x['mean_eval'], reverse=True)
    
    for data in summary_data:
        prec_str = f"{data['mean_prec']:.2f}" if not math.isnan(data['mean_prec']) else "N/A"
        rec_str = f"{data['mean_rec']:.2f}" if not math.isnan(data['mean_rec']) else "N/A"
        print(f"{data['name']:<25} {data['mean_eval']:>6.1f} ± {data['std_eval']:<5.1f} "
              f"{data['mean_det']:>10.0f}  {prec_str:>10}  {rec_str:>10}")
    
    print("=" * 80)
    
    # 详细的任务级别结果
    print("\n" + "=" * 80)
    print("Detailed Task-level Performance (averaged across seeds)")
    print("=" * 80)
    
    task_names = [f"T{i}({cfg.TASKS[i]['task_name']})" for i in range(len(cfg.TASKS))]
    header = f"{'Detector':<25} " + "".join([f"{task_name:<15}" for task_name in task_names])
    print(header)
    print("-" * (25 + 15 * len(task_names)))
    
    for data in summary_data:
        task_rewards_by_id = {}
        for task_id in range(len(cfg.TASKS)):
            task_rewards = []
            for result in data['all_results']:
                if task_id in result['eval_rewards']:
                    task_rewards.append(result['eval_rewards'][task_id])
            task_rewards_by_id[task_id] = np.mean(task_rewards) if task_rewards else 0.0
        
        task_scores = "".join([f"{task_rewards_by_id[i]:>14.1f} " 
                              for i in range(len(cfg.TASKS))])
        print(f"{data['name']:<25} {task_scores}")
    
    print("=" * 80)
    
    # 统计显著性检验
    if 'no_detector' in all_results:
        baseline_evals = [r['avg_eval'] for r in all_results['no_detector']]
        baseline_mean = np.mean(baseline_evals)
        
        print("\nStatistical Significance Test (vs Baseline):")
        print("-" * 80)
        
        from scipy import stats as sp_stats
        
        for data in summary_data:
            if data['name'] == 'no_detector':
                continue
            
            detector_evals = [r['avg_eval'] for r in all_results[data['name']]]
            
            if len(detector_evals) > 1 and len(baseline_evals) > 1:
                t_stat, p_value = sp_stats.ttest_ind(detector_evals, baseline_evals)
                improvement = data['mean_eval'] - baseline_mean
                
                sig_marker = "✓" if p_value < 0.05 else " "
                print(f"{data['name']:<25} t={t_stat:>6.2f}, p={p_value:.4f} {sig_marker}  "
                      f"Δ={improvement:+.1f}")
    
    # 生成可视化
    print("\n📈 Generating visualizations...")
    plot_task_performance_heatmap(summary_data, cfg)
    plot_detector_comparison(summary_data)
    print("✅ Visualizations saved to visualizations/ directory.")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MountainCar RQ1: Change Detection Experiment'
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--episodes-per-task', type=int, default=150)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--warmup-episodes', type=int, default=50)
    parser.add_argument('--quick-test', action='store_true',
                       help="Quick test with 1 seed")
    args = parser.parse_args()
    
    if args.quick_test:
        print("🚀 QUICK TEST MODE")
        main(seeds=[0], episodes_per_task=100, cycles=1, warmup_episodes=30)
    else:
        main(
            seeds=args.seeds,
            episodes_per_task=args.episodes_per_task,
            cycles=args.cycles,
            warmup_episodes=args.warmup_episodes
        )
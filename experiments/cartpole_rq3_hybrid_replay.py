#!/usr/bin/env python3
"""
RQ3 Experimental Framework: Complete Hybrid Latent-Space Replay with World Models - FIXED VERSION
Optimized for CartPole-v1
"""

from __future__ import annotations
import argparse, random, sys, os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import copy
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time

sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cartpole_config import CartPoleConfig
from environments.cartpole_cl import CartPoleCL
from LatentReplayBuffer import LatentReplayBuffer, KnowledgeDistillationLoss
# Import the new visualization module
from RQ3metrics import create_comprehensive_analysis


def set_global_seed(seed: int):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PolicyNetwork(nn.Module):
    """Simple policy network for fair comparison across all conditions."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)


class RQ3Agent:
    """
    Complete agent with all hybrid conditions - FIXED VERSION.
    
    This agent supports various replay strategies including:
    - Standard Replay (Unlimited buffer)
    - Limited Replay (Restricted buffer size)
    - Latent Replay (Storing compressed states)
    - World Model (Generating synthetic samples)
    - Hybrid approaches (Combining Latent + Synthetic + Uncertainty + Distillation)
    """
    # 
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        condition: str,
        device: str = "cpu",
        config: Optional[Dict] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition = condition
        self.device = device
        self.config = config or {}
        
        # Policy networks
        hidden_dim = self.config.get('policy_hidden', 64)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # For knowledge distillation
        self.teacher_net = None
        self.distill_loss = None
        if 'distill' in condition:
            self.teacher_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
            # [OPTIMIZATION]: Reduced alpha from 0.7 to 0.15.
            # Explanation: When environment dynamics change drastically (e.g., wind force),
            # strong distillation can cause negative transfer. 
            # We want to retain a "faint memory" rather than being "stubborn".
            self.distill_loss = KnowledgeDistillationLoss(temperature=2.0, alpha=0.15)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.get('learning_rate', 1e-3)
        )
        
        # Replay mechanism based on condition
        self._setup_replay(condition)
        
        # Training parameters
        self.epsilon = 0.1
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 200
        self.steps_done = 0
        
        # Feature control parameters
        self.force_latent_ratio = 0.0
        self.synthetic_ratio = 0.0
        self.uncertainty_guided = False
        self.use_distillation = ('distill' in condition)
        
        # Initialize based on condition
        self._initialize_condition_features()
        
        # Metrics tracking
        self.memory_usage_history = []
        self.performance_history = []
        self.world_model_errors = []
        self.synthetic_generation_stats = []
        
        # Debugging
        self.debug_counter = 0
    
    def _setup_replay(self, condition: str):
        """Setup replay buffer based on experimental condition."""
        buffer_config = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            # OPTIMIZATION: Increased from 3 to 4. 
            # CartPole state is 4D. 3D loses info. 4D keeps info but allows encoder to learn features.
            'latent_dim': 4, 
            'max_latent_samples': 2000,
            'max_raw_samples': 2000,
            'device': self.device
        }
        
        if condition == "baseline_no_replay":
            self.replay_buffer = None
            self.use_replay = False
            print("🔄 Baseline: No replay")
        
        elif condition == "standard_replay":
            self.replay_buffer = SimpleReplayBuffer(max_size=50000)
            self.use_replay = True
            print("💾 Standard: Unlimited replay")
        
        elif condition == "limited_replay":
            self.replay_buffer = SimpleReplayBuffer(max_size=2000)
            self.use_replay = True
            print("📊 Limited: 2000 samples")
        
        elif condition == "latent_replay":
            buffer_config['max_latent_samples'] = 2000
            buffer_config['max_raw_samples'] = 2000
            self.replay_buffer = LatentReplayBuffer(**buffer_config)
            self.use_replay = True
            print("🧠 Latent: Compression only")
        
        elif condition == "world_model_only":
            buffer_config['use_world_model'] = True
            buffer_config['max_synthetic_samples'] = 2000
            buffer_config['max_raw_samples'] = 1000  # Smaller raw buffer
            self.replay_buffer = LatentReplayBuffer(**buffer_config)
            self.use_replay = True
            print("🤖 WorldModel: Synthetic samples only")
        
        elif condition == "hybrid_basic":
            buffer_config['use_world_model'] = True
            buffer_config['max_synthetic_samples'] = 1000
            self.replay_buffer = LatentReplayBuffer(**buffer_config)
            self.use_replay = True
            print("🧪 Hybrid: Basic combination")
        
        elif condition == "hybrid_uncertainty":
            buffer_config['use_world_model'] = True
            buffer_config['max_synthetic_samples'] = 1000
            self.replay_buffer = LatentReplayBuffer(**buffer_config)
            self.use_replay = True
            print("🎯 Hybrid: Uncertainty-guided")
        
        elif condition == "hybrid_distill":
            buffer_config['use_world_model'] = True
            buffer_config['max_synthetic_samples'] = 1000
            self.replay_buffer = LatentReplayBuffer(**buffer_config)
            self.use_replay = True
            print("📚 Hybrid: + Distillation")
        
        else:
            raise ValueError(f"Unknown condition: {condition}")
    
    def _initialize_condition_features(self):
        """Initialize feature flags based on condition."""
        if self.condition == "latent_replay":
            # FIXED: Allow sampling from latent buffer (LatentReplayBuffer now supports next_state storage)
            self.force_latent_ratio = 0.5 
            self.synthetic_ratio = 0.0
            print("✅ Latent Replay Active: Sampling allowed")
        
        elif self.condition == "world_model_only":
            self.force_latent_ratio = 0.0
            self.synthetic_ratio = 0.3
            self.uncertainty_guided = False
        
        elif self.condition == "hybrid_basic":
            self.force_latent_ratio = 0.1
            self.synthetic_ratio = 0.05
            self.uncertainty_guided = False
        
        elif self.condition == "hybrid_uncertainty":
            self.force_latent_ratio = 0.1
            self.synthetic_ratio = 0.05
            self.uncertainty_guided = True
        
        elif self.condition == "hybrid_distill":
            self.force_latent_ratio = 0.1
            self.synthetic_ratio = 0.05
            self.uncertainty_guided = True
            self.use_distillation = True
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        task_id: int = 0
    ):
        """Store transition with enhanced features (Latent/World Model updates)."""
        if not self.use_replay or self.replay_buffer is None:
            return
        
        self.debug_counter += 1
        
        # Store the transition
        if isinstance(self.replay_buffer, LatentReplayBuffer):
            self.replay_buffer.add_transition(
                state, action, reward, next_state, done,
                task_id=task_id,
                is_synthetic=False,
                world_model_uncertainty=None
            )
            
            # Train world model frequently
            if hasattr(self.replay_buffer, 'world_model') and self.replay_buffer.world_model is not None:
                # [OPTIMIZATION]: Increase training frequency/intensity.
                # CartPole data is simple; frequent updates help adapt to new physics quickly.
                if self.steps_done % 5 == 0:  # Changed from 10 to 5
                    loss, uncertainty = self._train_world_model()
                    if loss > 0 and self.debug_counter % 500 == 0:
                        print(f"🤖 World model training: loss={loss:.4f}, uncertainty={uncertainty:.4f}")
            
            # Train encoder frequently
            if self.steps_done % 100 == 0:
                loss = self._train_latent_encoder()
                if self.debug_counter % 1000 == 0 and loss > 0:
                    print(f"🔧 Encoder training: loss={loss:.4f}")
            
            # Validate world model quality
            if hasattr(self.replay_buffer, 'world_model') and self.replay_buffer.world_model is not None:
                if self.debug_counter % 500 == 0:
                    quality_metrics = self.replay_buffer.validate_world_model_quality()
                    if quality_metrics:
                        self.world_model_errors.append(quality_metrics)
        else:
            self.replay_buffer.push(state, action, reward, next_state, done)
    
    def _train_world_model(self):
        """Train world model on recent experiences."""
        if not isinstance(self.replay_buffer, LatentReplayBuffer):
            return 0.0, 0.0
        
        if len(self.replay_buffer.raw_buffer) < 16:
            return 0.0, 0.0
        
        samples = random.sample(
            self.replay_buffer.raw_buffer[-1000:],
            min(32, len(self.replay_buffer.raw_buffer))
        )
        
        states = torch.FloatTensor(np.array([s[0] for s in samples])).to(self.device)
        actions = torch.LongTensor(np.array([s[1] for s in samples])).to(self.device)
        next_states = torch.FloatTensor(np.array([s[3] for s in samples])).to(self.device)
        rewards = torch.FloatTensor(np.array([s[2] for s in samples])).to(self.device)
        
        loss, uncertainty = self.replay_buffer.train_world_model(
            states, actions, next_states, rewards
        )
        
        return loss, uncertainty
    
    def _train_latent_encoder(self):
        """Train latent encoder using autoencoder loss."""
        if not isinstance(self.replay_buffer, LatentReplayBuffer):
            return 0.0
        
        if len(self.replay_buffer.raw_buffer) < 16:
            return 0.0
        
        samples = random.sample(self.replay_buffer.raw_buffer, min(32, len(self.replay_buffer.raw_buffer)))
        states = torch.FloatTensor(np.array([s[0] for s in samples])).to(self.device)
        actions = torch.LongTensor(np.array([s[1] for s in samples])).to(self.device)
        
        loss = self.replay_buffer.train_encoder(states, actions)
        return loss
    
    def generate_synthetic_samples(self, task_id: int):
        """Generate synthetic samples using the World Model (Dreaming)."""
        # 
        if not isinstance(self.replay_buffer, LatentReplayBuffer):
            return 0
        
        if not hasattr(self.replay_buffer, 'world_model') or self.replay_buffer.world_model is None:
            return 0
        
        n_samples = 30
        
        generated = self.replay_buffer.generate_synthetic_samples(
            n_samples, task_id, high_uncertainty=self.uncertainty_guided
        )
        
        if generated > 0 and self.debug_counter % 200 == 0:
            print(f"🤖 Generated {generated} synthetic samples for task {task_id}")
        
        return generated
    
    def update(self) -> float:
        """Enhanced update with all features (Latent/Synthetic Sampling + Distillation)."""
        if not self.use_replay or self.replay_buffer is None:
            return 0.0
        
        try:
            if isinstance(self.replay_buffer, LatentReplayBuffer):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(
                    self.batch_size,
                    use_latent=True,
                    use_synthetic=(self.synthetic_ratio > 0),
                    force_latent_ratio=self.force_latent_ratio,
                    synthetic_ratio=self.synthetic_ratio,
                    uncertainty_guided=self.uncertainty_guided
                )
            else:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            if len(states) == 0:
                return 0.0
            
            if np.any(np.isnan(states)) or np.any(np.isnan(actions)):
                print("⚠️ NaN detected in sampled batch, skipping update")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error in sampling: {e}")
            return 0.0
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        
        td_loss = nn.MSELoss()(q_values, target)
        
        # Apply Knowledge Distillation if enabled
        if self.use_distillation and self.teacher_net is not None:
            with torch.no_grad():
                teacher_q = self.teacher_net(states)
            student_q = self.policy_net(states)
            
            loss = self.distill_loss(
                student_q,
                teacher_q,
                actions.squeeze(),
                td_loss
            )
        else:
            loss = td_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return float(loss.item())
    
    def snapshot_for_distillation(self):
        """Save current policy as teacher for knowledge distillation."""
        if self.teacher_net is not None:
            self.teacher_net.load_state_dict(self.policy_net.state_dict())
            self.teacher_net.eval()
            print(f"📚 Saved teacher model for distillation (step {self.steps_done})")
    
    def check_and_enable_features(self, episodes_completed: int):
        """Progressive feature enabling based on condition (Warmup periods)."""
        if self.condition == "latent_replay":
            # [OPTIMIZATION]: Added warmup period. Do not use Latent data for first 50 episodes
            # to avoid noise from unconverged Encoder.
            if episodes_completed < 50:
                self.force_latent_ratio = 0.0
            else:
                self.force_latent_ratio = 0.5  # Enable mixing after Encoder stabilizes
        
        elif self.condition in ["world_model_only", "hybrid_basic", "hybrid_uncertainty", "hybrid_distill"]:
            if episodes_completed > 5:
                if episodes_completed % 5 == 0:
                    current_task = 0
                    generated = self.generate_synthetic_samples(current_task)
                    if generated > 0 and episodes_completed % 15 == 0:
                        print(f"🤖 Synthetic generation: {generated} samples (episode {episodes_completed})")
            
            if self.condition == "world_model_only" and episodes_completed > 10:
                if episodes_completed < 20:
                    self.synthetic_ratio = 0.2
                elif episodes_completed < 30:
                    self.synthetic_ratio = 0.25
                else:
                    self.synthetic_ratio = 0.3
    
    def diagnose(self):
        """Enhanced diagnostics for monitoring replay buffer state."""
        print(f"\n=== Diagnostics (step {self.steps_done}) ===")
        
        if isinstance(self.replay_buffer, LatentReplayBuffer):
            stats = self.replay_buffer.get_stats()
            print(f"  Raw samples: {stats['samples_raw']}")
            print(f"  Latent samples: {stats['samples_latent']}")
            print(f"  Synthetic samples: {stats['samples_synthetic']}")
            print(f"  Latent ratio: {self.force_latent_ratio:.1%}")
            print(f"  Synthetic ratio: {self.synthetic_ratio:.1%}")
            print(f"  Compression: {stats.get('compression_ratio', 1.0):.2f}x")
        
        health = self.check_training_health()
        if not health['healthy']:
            print("  ⚠️ Health issues:")
            for issue in health['issues']:
                print(f"    - {issue}")
        
        print("=" * 50)
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics of the replay buffer."""
        if isinstance(self.replay_buffer, LatentReplayBuffer):
            return self.replay_buffer.get_memory_usage()
        elif self.replay_buffer is not None:
            sample_size = (2 * self.state_dim + 3) * 4
            mem = len(self.replay_buffer) * sample_size / (1024 * 1024)
            return {
                'total_mb': mem,
                'samples_raw': len(self.replay_buffer),
                'compression_ratio': 1.0
            }
        else:
            return {'total_mb': 0.0, 'compression_ratio': 1.0}
    
    def check_training_health(self) -> Dict:
        """Check training health status (e.g., gradient norms)."""
        health_status = {
            'healthy': True,
            'issues': []
        }
        
        policy_grad_norm = 0.0
        for param in self.policy_net.parameters():
            if param.grad is not None:
                policy_grad_norm += param.grad.data.norm(2).item()
        
        if policy_grad_norm > 100:
            health_status['healthy'] = False
            health_status['issues'].append(f"Policy gradient too large: {policy_grad_norm:.2f}")
        
        return health_status


class SimpleReplayBuffer:
    """Traditional experience replay buffer (FIFO)."""
    def __init__(self, max_size: int = 10000):
        self.buffer = []
        self.max_size = max_size
        self.pos = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.pos = (self.pos + 1) % self.max_size
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return tuple([np.array([]) for _ in range(5)])
        
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        return tuple(map(np.array, zip(*batch)))
    
    def __len__(self):
        return len(self.buffer)


def run_rq3_experiment(
    cfg: CartPoleConfig,
    condition: str,
    seed: int,
    episodes_per_task: int = 100,
    cycles: int = 2
) -> Dict:
    """Run single experimental condition."""
    print(f"\n{'='*60}")
    print(f"SEED {seed} - Condition: {condition}")
    print(f"{ '='*60}")
    
    set_global_seed(seed)
    
    env = CartPoleCL(cfg.TASKS)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent_config = {
        'policy_hidden': 64,
        'learning_rate': 1e-3,
    }
    agent = RQ3Agent(state_dim, action_dim, condition, config=agent_config)
    
    task_sequence = []
    for _ in range(cycles):
        for task_id in range(env.total_tasks):
            task_sequence.extend([task_id] * episodes_per_task)
    
    episode_rewards = []
    task_performances = {i: [] for i in range(env.total_tasks)}
    memory_usage_log = []
    
    current_task = 0
    env.change_task(current_task)
    episodes_completed = 0
    
    # Main training loop
    for episode_idx in range(len(task_sequence)):
        desired_task = task_sequence[episode_idx]
        
        # Task Change Logic
        if desired_task != env.current_task:
            eval_results = evaluate_all_tasks(agent, env, cfg)
            for tid, perf in eval_results.items():
                task_performances[tid].append(perf)
            
            if 'distill' in condition:
                agent.snapshot_for_distillation()
            
            env.change_task(desired_task)
            current_task = desired_task
            
            if isinstance(agent.replay_buffer, LatentReplayBuffer):
                agent.replay_buffer.update_task_boundary(current_task)
        
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done, current_task)
            loss = agent.update()
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        episodes_completed += 1
        
        agent.check_and_enable_features(episodes_completed)
        
        if episodes_completed % 50 == 0 and episodes_completed > 0:
            agent.diagnose()
        
        if episodes_completed % 20 == 0:
            mem_stats = agent.get_memory_usage()
            mem_stats['episode'] = episodes_completed
            memory_usage_log.append(mem_stats)
    
    final_eval = evaluate_all_tasks(agent, env, cfg)
    
    avg_performance = np.mean(list(final_eval.values()))
    final_memory = agent.get_memory_usage()
    memory_efficiency = avg_performance / (final_memory.get('total_mb', 1.0) + 0.1)
    forward_transfer = compute_forward_transfer(task_performances)
    backward_transfer = compute_backward_transfer(task_performances)
    
    print(f"Seed {seed} final performance: {avg_performance:.2f}")
    print(f"Episode rewards last 50: {np.mean(episode_rewards[-50:]):.2f}")
    
    return {
        'condition': condition,
        'seed': seed,
        'episode_rewards': episode_rewards,
        'task_performances': task_performances,
        'final_eval': final_eval,
        'avg_performance': avg_performance,
        'memory_usage_log': memory_usage_log,
        'final_memory_mb': final_memory.get('total_mb', 0.0),
        'compression_ratio': final_memory.get('compression_ratio', 1.0),
        'memory_efficiency': memory_efficiency,
        'forward_transfer': forward_transfer,
        'backward_transfer': backward_transfer,
    }


def evaluate_all_tasks(agent: RQ3Agent, env: CartPoleCL, cfg: CartPoleConfig, n_episodes: int = 10) -> Dict:
    """Evaluate agent on all tasks."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    results = {}
    for task_id in range(env.total_tasks):
        env.change_task(task_id)
        task_rewards = []
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            
            for _ in range(cfg.MAX_STEPS_PER_EPISODE):
                action = agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            task_rewards.append(total_reward)
        
        results[task_id] = np.mean(task_rewards)
    
    agent.epsilon = original_epsilon
    return results


def compute_forward_transfer(task_performances: Dict) -> float:
    """Measure forward transfer (Performance on new task vs baseline)."""
    if len(task_performances) < 2:
        return 0.0
    
    first_task_curve = task_performances[0][:20] if len(task_performances[0]) > 0 else [0]
    
    transfers = []
    for task_id in range(1, len(task_performances)):
        later_task_curve = task_performances[task_id][:20] if len(task_performances[task_id]) > 0 else [0]
        
        if len(later_task_curve) > 0 and len(first_task_curve) > 0:
            transfer = np.mean(later_task_curve) - np.mean(first_task_curve)
            transfers.append(transfer)
    
    return float(np.mean(transfers)) if transfers else 0.0


def compute_backward_transfer(task_performances: Dict) -> float:
    """Measure backward transfer (Forgeting). Negative means forgetting."""
    if len(task_performances) < 2:
        return 0.0
    
    forgetting_scores = []
    
    for task_id in range(len(task_performances) - 1):
        performance_history = task_performances[task_id]
        
        if len(performance_history) < 2:
            continue
        
        peak_performance = max(performance_history)
        final_performance = np.mean(performance_history[-5:]) if len(performance_history) >= 5 else performance_history[-1]
        
        forgetting = final_performance - peak_performance
        forgetting_scores.append(forgetting)
    
    return float(np.mean(forgetting_scores)) if forgetting_scores else 0.0


def main(seeds: List[int] = [1, 2, 3], episodes_per_task: int = 300, cycles: int = 2):
    """Complete RQ3 experiment with all conditions - FIXED VERSION."""
    print(f"🔬 Experimental configuration: seeds={seeds}, episodes_per_task={episodes_per_task}, cycles={cycles}")
    print(f"🔬 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # === 📂 Path Management: Save like MountainCar ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "rq3_cartpole")
    vis_dir = os.path.join(os.path.dirname(base_dir), "visualizations", "rq3_cartpole")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📂 Results will be saved to: {results_dir}")
    print(f"📊 Visualizations will be saved to: {vis_dir}")
    # ==================================================

    cfg = CartPoleConfig()
    env = CartPoleCL(cfg.TASKS)
    env.reset(seed=seeds[0])
    
    conditions = [
        "baseline_no_replay",
        "standard_replay",        
        "limited_replay", 
        "latent_replay",
        "world_model_only",
        "hybrid_basic", 
        "hybrid_uncertainty",
        "hybrid_distill"
    ]
    
    print("=" * 80)
    print("COMPLETE RQ3 EXPERIMENT: All Hybrid Conditions - CartPole Fixed")
    print("=" * 80)
    
    all_results = {}
    
    for condition in conditions:
        print(f"\n🏃 Running condition: {condition}")
        condition_results = []
        
        for seed in seeds:
            print(f"\n  🔧 Seed {seed} starting...")
            set_global_seed(seed)
            result = run_rq3_experiment(cfg, condition, seed, episodes_per_task, cycles)
            condition_results.append(result)
            
            # === 💾 Save to new path (Same format as MountainCar) ===
            json_filename = f'temp_results_{condition}_seed{seed}.json'
            json_path = os.path.join(results_dir, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump({
                    'seed': seed,
                    'condition': condition,
                    'avg_performance': result['avg_performance'],
                    'final_memory_mb': result['final_memory_mb'],
                    'compression_ratio': result['compression_ratio'],
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
            print(f"  💾 Saved result to: {json_filename}")
            # ==============================================
        
        all_results[condition] = condition_results
    
    print("\n" + "=" * 80)
    print("RQ3 COMPLETE RESULTS SUMMARY - CartPole Fixed")
    print("=" * 80)
    print(f"{ 'Condition':<25} {'Perf':<8} {'Memory':<10} {'Eff':<8} {'Comp':<8} {'FT':<8} {'BT':<8}")
    print("-" * 80)
    
    for condition in conditions:
        if condition in all_results:
            perfs = [r['avg_performance'] for r in all_results[condition]]
            mems = [r['final_memory_mb'] for r in all_results[condition]]
            effs = [r['memory_efficiency'] for r in all_results[condition]]
            comps = [r['compression_ratio'] for r in all_results[condition]]
            fts = [r['forward_transfer'] for r in all_results[condition]]
            bts = [r['backward_transfer'] for r in all_results[condition]]
            
            print(f"{condition:<25} {np.mean(perfs):>5.1f}±{np.std(perfs):<2.1f} {np.mean(mems):>8.3f} "
                  f"{np.mean(effs):>6.1f} {np.mean(comps):>6.2f}x {np.mean(fts):>6.1f} {np.mean(bts):>6.1f}")
    
    # === 📊 Generate viz to new path ===
    print(f"\n📊 Generating comprehensive visualizations in {vis_dir}...")
    create_comprehensive_analysis(all_results, save_dir=vis_dir)
    # ==========================================================
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RQ3: Complete Hybrid Experiments - CartPole Fixed')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                        help='Random seeds for experiments')
    parser.add_argument('--episodes-per-task', type=int, default=300,
                        help='Episodes per task (Increased for better convergence)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of cycles through all tasks')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with fewer episodes and seeds')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("🚀 QUICK TEST MODE")
        main(seeds=[1], episodes_per_task=50, cycles=1)
    else:
        main(seeds=args.seeds, episodes_per_task=args.episodes_per_task, cycles=args.cycles)
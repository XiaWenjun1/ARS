#!/usr/bin/env python3
"""
RQ3 Experimental Framework: Complete Hybrid Latent-Space Replay with World Models - MountainCar Version
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

from configs.mountaincar_config import MountainCarConfig
from environments.mountaincar_cl import MountainCarCL
from LatentReplayBuffer import LatentReplayBuffer, KnowledgeDistillationLoss
# Import the new visualization module
from RQ3metrics import create_comprehensive_analysis


def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PolicyNetwork(nn.Module):
    """Simple policy network for fair comparison"""
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
    Complete agent with all hybrid conditions - FIXED VERSION
    """
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
            self.distill_loss = KnowledgeDistillationLoss(temperature=2.0, alpha=0.2)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.get('learning_rate', 1e-3)
        )
        
        # Replay mechanism based on condition
        self._setup_replay(condition)
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.985
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 200
        self.steps_done = 0
        
        # Feature control parameters - FIXED: More conservative ratios
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
        """Setup replay buffer based on experimental condition"""
        buffer_config = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'latent_dim': 2,  # For MountainCar, state_dim is 2, so latent_dim should be small.
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
        """Initialize feature flags based on condition - FIXED: Critical fix for latent_replay"""
        if self.condition == "latent_replay":
            # FIX: With the buffer now storing next_state, we can enable latent sampling.
            self.force_latent_ratio = 0.5 # Enable sampling from latent buffer
            self.synthetic_ratio = 0.0
            print("✅ INFO: latent_replay is now active, sampling from latent buffer.")
        
        elif self.condition == "world_model_only":
            self.force_latent_ratio = 0.0
            self.synthetic_ratio = 0.3  # FIXED: Reduced from 0.5 to 0.3
            self.uncertainty_guided = False
        
        elif self.condition == "hybrid_basic":
            self.force_latent_ratio = 0.1  # FIXED: Reduced from 0.2 to 0.1
            self.synthetic_ratio = 0.05    # FIXED: Reduced from 0.1 to 0.05
            self.uncertainty_guided = False
        
        elif self.condition == "hybrid_uncertainty":
            self.force_latent_ratio = 0.1  # FIXED: Reduced from 0.2 to 0.1
            self.synthetic_ratio = 0.02    # FIXED: Reduced from 0.1 to 0.05
            self.uncertainty_guided = True
        
        elif self.condition == "hybrid_distill":
            self.force_latent_ratio = 0.1  # FIXED: Reduced from 0.2 to 0.1
            self.synthetic_ratio = 0.02    # FIXED: Reduced from 0.1 to 0.05
            self.uncertainty_guided = True
            self.use_distillation = True
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax().item())

    def update_epsilon(self):
        """Anneal epsilon after each episode"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        task_id: int = 0
    ):
        """Store transition with enhanced features"""
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
            
            # FIXED: Train world model more frequently (every 10 steps instead of 50)
            if hasattr(self.replay_buffer, 'world_model') and self.replay_buffer.world_model is not None:
                if self.steps_done % 10 == 0:  # FIXED: Increased frequency
                    loss, uncertainty = self._train_world_model()
                    if loss > 0 and self.debug_counter % 500 == 0:
                        print(f"🤖 World model training: loss={loss:.4f}, uncertainty={uncertainty:.4f}")
            
            # FIXED: Train encoder more frequently and without step limit
            if self.steps_done % 100 == 0:  # FIXED: Increased frequency, removed step limit
                loss = self._train_latent_encoder()
                if self.debug_counter % 1000 == 0 and loss > 0:
                    print(f"🔧 Encoder training: loss={loss:.4f}")
            
            # # NEW: Validate world model quality periodically
            # if hasattr(self.replay_buffer, 'world_model') and self.replay_buffer.world_model is not None:
            #     if self.debug_counter % 500 == 0:
            #         quality_metrics = self.replay_buffer.validate_world_model_quality()
            #         if quality_metrics:
            #             self.world_model_errors.append(quality_metrics)
        else:
            self.replay_buffer.push(state, action, reward, next_state, done)
    
    def _train_world_model(self):
        """Train world model on recent experiences - FIXED: More frequent training"""
        if not isinstance(self.replay_buffer, LatentReplayBuffer):
            return 0.0, 0.0
        
        if len(self.replay_buffer.raw_buffer) < 16:
            return 0.0, 0.0
        
        # Sample recent experiences for world model training
        samples = random.sample(
            self.replay_buffer.raw_buffer[-1000:],  # Recent samples
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
        """Train latent encoder - FIXED: More frequent training"""
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
        """Generate synthetic samples using world model - FIXED: More frequent generation"""
        if not isinstance(self.replay_buffer, LatentReplayBuffer):
            return 0
        
        if not hasattr(self.replay_buffer, 'world_model') or self.replay_buffer.world_model is None:
            return 0
        
        # Generate synthetic samples
        n_samples = 30  # FIXED: Reduced from 50 to 30 for better quality
        
        generated = self.replay_buffer.generate_synthetic_samples(
            n_samples, task_id, high_uncertainty=self.uncertainty_guided
        )
        
        if generated > 0 and self.debug_counter % 200 == 0:  # FIXED: More frequent logging
            print(f"🤖 Generated {generated} synthetic samples for task {task_id}")
        
        return generated
    
    def update(self) -> float:
        """Enhanced update with all features - FIXED: Better uncertainty debugging"""
        if not self.use_replay or self.replay_buffer is None:
            return 0.0
        
        # Sample batch with appropriate ratios
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
                
                # Enhanced debugging for uncertainty-guided sampling
                if self.uncertainty_guided and self.debug_counter % 800 == 0 and len(states) > 0:
                    stats = self.replay_buffer.get_stats()
                    print(f"🎯 UNCERTAINTY-GUIDED SAMPLING ACTIVE:")
                    print(f"   - Synthetic samples: {stats['samples_synthetic']}")
                    print(f"   - Uncertainty score: {stats['uncertainty_score']:.3f}")
                    print(f"   - Ratios: latent={self.force_latent_ratio:.1%}, synthetic={self.synthetic_ratio:.1%}")
                    
                # Enhanced debugging
                if self.debug_counter % 1000 == 0 and len(states) > 0:
                    stats = self.replay_buffer.get_stats()
                    print(f"🔍 DEBUG [step {self.steps_done}]:")
                    print(f"   - Raw: {stats['samples_raw']}, Latent: {stats['samples_latent']}, Synthetic: {stats['samples_synthetic']}")
                    print(f"   - Ratios: latent={self.force_latent_ratio:.1%}, synthetic={self.synthetic_ratio:.1%}")
                    print(f"   - Batch size: {len(states)}")
                    print(f"   - Compression ratio: {stats.get('compression_ratio', 1.0):.2f}x")
            else:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            
            if len(states) == 0:
                return 0.0
            
            if np.any(np.isnan(states)) or np.any(np.isnan(actions)):
                print("⚠️ NaN detected in sampled batch, skipping update")
                return 0.0
                
        except Exception as e:
            print(f"❌ Error in sampling: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Q-learning update
        q_values = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute base loss
        td_loss = nn.MSELoss()(q_values, target)
        
        # Add distillation loss if applicable
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
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return float(loss.item())
    
    def snapshot_for_distillation(self):
        """Save current policy as teacher for knowledge distillation"""
        if self.teacher_net is not None:
            self.teacher_net.load_state_dict(self.policy_net.state_dict())
            self.teacher_net.eval()
            print(f"📚 Saved teacher model for distillation (step {self.steps_done})")
    
    def check_and_enable_features(self, episodes_completed: int):
        """Progressive feature enabling based on condition - FIXED: More gradual enabling"""
        if self.condition == "latent_replay":
            # ✅ FIX: Do not disable latent replay. The ratio is set during initialization.
            # self.force_latent_ratio = 0.0  # Always 0
            pass
        
        elif self.condition in ["world_model_only", "hybrid_basic", "hybrid_uncertainty", "hybrid_distill"]:
            # Enable world model features earlier and more frequently
            if episodes_completed > 5:  # FIXED: Start earlier (from 20 to 5)
                # Generate synthetic samples periodically
                if episodes_completed % 5 == 0:  # FIXED: More frequent generation (from 10 to 5)
                    current_task = 0
                    generated = self.generate_synthetic_samples(current_task)
                    if generated > 0 and episodes_completed % 15 == 0:
                        print(f"🤖 Synthetic generation: {generated} samples (episode {episodes_completed})")
            
            # Gradually increase synthetic ratio for world_model_only
            if self.condition == "world_model_only" and episodes_completed > 10:
                if episodes_completed < 20:
                    self.synthetic_ratio = 0.2
                elif episodes_completed < 30:
                    self.synthetic_ratio = 0.25
                else:
                    self.synthetic_ratio = 0.3
    
    def diagnose(self):
        """Enhanced diagnostics"""
        print(f"\n=== Diagnostics (step {self.steps_done}) ===")
        
        if isinstance(self.replay_buffer, LatentReplayBuffer):
            stats = self.replay_buffer.get_stats()
            print(f"  Raw samples: {stats['samples_raw']}")
            print(f"  Latent samples: {stats['samples_latent']}")
            print(f"  Synthetic samples: {stats['samples_synthetic']}")
            print(f"  Latent ratio: {self.force_latent_ratio:.1%}")
            print(f"  Synthetic ratio: {self.synthetic_ratio:.1%}")
            print(f"  Compression: {stats.get('compression_ratio', 1.0):.2f}x")
            print(f"  Uncertainty guided: {self.uncertainty_guided}")
            print(f"  Distillation: {self.use_distillation}")
            
            # # NEW: World model quality info
            # if hasattr(self.replay_buffer, 'world_model') and self.replay_buffer.world_model is not None:
            #     quality = self.replay_buffer.validate_world_model_quality()
            #     if quality:
            #         print(f"  World Model Quality: state_mse={quality.get('state_prediction_mse', 0):.4f}")
        
        health = self.check_training_health()
        if not health['healthy']:
            print("  ⚠️ Health issues:")
            for issue in health['issues']:
                print(f"    - {issue}")
        
        print("=" * 50)
    
    def get_memory_usage(self) -> Dict:
        """Get memory usage statistics"""
        if isinstance(self.replay_buffer, LatentReplayBuffer):
            return self.replay_buffer.get_memory_usage()
        elif self.replay_buffer is not None:
            # FIXED: Correct memory calculation for SimpleReplayBuffer
            sample_size = (2 * self.state_dim + 3) * 4  # state, next_state, action, reward, done
            mem = len(self.replay_buffer) * sample_size / (1024 * 1024)
            return {
                'total_mb': mem,
                'samples_raw': len(self.replay_buffer),
                'compression_ratio': 1.0
            }
        else:
            return {'total_mb': 0.0, 'compression_ratio': 1.0}
    
    def check_training_health(self) -> Dict:
        """Check training health status"""
        health_status = {
            'healthy': True,
            'issues': []
        }
        
        # Check policy network gradients
        policy_grad_norm = 0.0
        for param in self.policy_net.parameters():
            if param.grad is not None:
                policy_grad_norm += param.grad.data.norm(2).item()
        
        if policy_grad_norm > 100:
            health_status['healthy'] = False
            health_status['issues'].append(f"Policy gradient too large: {policy_grad_norm:.2f}")
        
        # Check replay buffers
        if isinstance(self.replay_buffer, LatentReplayBuffer):
            if len(self.replay_buffer.raw_buffer) < 20:
                health_status['issues'].append("Raw buffer too small")
            if (
                self.force_latent_ratio > 0 and 
                len(self.replay_buffer.latent_buffer) == 0
            ):
                health_status['issues'].append("Latent buffer empty but sampling enabled")
            if (
                self.synthetic_ratio > 0 and 
                len(self.replay_buffer.synthetic_buffer) == 0
            ):
                health_status['issues'].append("Synthetic buffer empty but sampling enabled")
        
        return health_status


class SimpleReplayBuffer:
    """Traditional experience replay buffer - FIXED: Correct memory calculation"""
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


def run_mountaincar_experiment(
    cfg: MountainCarConfig,
    condition: str,
    seed: int,
    episodes_per_task: int = 100,
    cycles: int = 2
) -> Dict:
    """Run single experimental condition"""
    print(f"\n{'='*60}")
    print(f"SEED {seed} - Condition: {condition}")
    print(f"{'='*60}")
    
    set_global_seed(seed)
    
    env = MountainCarCL(cfg.TASKS)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent_config = {
        'policy_hidden': 64,
        'learning_rate': 1e-3, # Lower learning rate for MountainCar
    }
    agent = RQ3Agent(state_dim, action_dim, condition, config=agent_config)
    
    # Task sequence
    task_sequence = []
    for _ in range(cycles):
        for task_id in range(env.total_tasks):
            task_sequence.extend([task_id] * episodes_per_task)
    
    # Training metrics
    episode_rewards = []
    task_performances = {i: [] for i in range(env.total_tasks)}
    memory_usage_log = []
    
    current_task = 0
    env.change_task(current_task)
    episodes_completed = 0
    
    for episode_idx in range(len(task_sequence)):
        desired_task = task_sequence[episode_idx]
        
        # Task boundary
        if desired_task != env.current_task:
            # Evaluate on all tasks before switching
            eval_results = evaluate_all_tasks(agent, env, cfg)
            for tid, perf in eval_results.items():
                task_performances[tid].append(perf)
            
            # Save teacher for distillation
            if 'distill' in condition:
                agent.snapshot_for_distillation()
            
            # Switch task
            env.change_task(desired_task)
            current_task = desired_task
            
            # Update buffer task boundary
            if isinstance(agent.replay_buffer, LatentReplayBuffer):
                agent.replay_buffer.update_task_boundary(current_task)
        
        # Training episode
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # === IMPROVED Reward Shaping for MountainCar ===
            # Original Logic (FLAWED): height_reward = (pos + 0.5) * 10.0
            # Why flawed? It penalizes going left (backswing), which is necessary for momentum.
            
            pos = next_state[0]
            vel = next_state[1]
            
            # New Logic: Encourage being AWAY from the bottom (-0.5) in EITHER direction
            # This rewards the swinging motion required to climb the hill.
            dist_from_bottom = abs(pos - (-0.5))
            
            # Combine: Base(-1) + Distance Bonus. The velocity bonus was removed
            # as it was encouraging reward hacking (wiggling in the valley).
            modified_reward = reward + (dist_from_bottom * 1.0)
            
            # Huge bonus for reaching goal (overwrites everything)
            if pos >= 0.5:
                modified_reward = 100.0

            # Store the SHAPED reward for learning
            agent.store_transition(state, action, modified_reward, next_state, done, current_task)
            
            # Update policy
            loss = agent.update()
            
            episode_reward += reward
            state = next_state
        
        # End of episode: decay epsilon for exploration-exploitation trade-off
        agent.update_epsilon()

        episode_rewards.append(episode_reward)
        episodes_completed += 1
        
        # Progressive feature enabling
        agent.check_and_enable_features(episodes_completed)
        
        # Periodic diagnostics
        if episodes_completed % 50 == 0 and episodes_completed > 0:
            agent.diagnose()
        
        # Log memory usage
        if episodes_completed % 20 == 0:
            mem_stats = agent.get_memory_usage()
            mem_stats['episode'] = episodes_completed
            memory_usage_log.append(mem_stats)
    
    # Final evaluation
    final_eval = evaluate_all_tasks(agent, env, cfg)
    
    # Compute summary metrics
    avg_performance = np.mean(list(final_eval.values()))
    
    # Memory efficiency
    final_memory = agent.get_memory_usage()
    memory_efficiency = avg_performance / (final_memory.get('total_mb', 1.0) + 0.1)
    
    # Transfer metrics
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


def evaluate_all_tasks(agent: RQ3Agent, env: MountainCarCL, cfg: MountainCarConfig, n_episodes: int = 10) -> Dict:
    """Evaluate agent on all tasks"""
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
    """Measure forward transfer"""
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
    """Measure backward transfer (forgetting)"""
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


def main(seeds: List[int] = [1, 2, 3], episodes_per_task: int = 400, cycles: int = 2):
    """Complete RQ3 experiment with all conditions - FIXED: Ensure independence"""
    print(f"🔬 Experimental configuration: seeds={seeds}, episodes_per_task={episodes_per_task}, cycles={cycles}")
    print(f"🔬 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results", "rq3_mountaincar")
    vis_dir = os.path.join(os.path.dirname(base_dir), "visualizations", "rq3_mountaincar")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📂 Results will be saved to: {results_dir}")
    print(f"📊 Visualizations will be saved to: {vis_dir}")
    # ==================================================

    cfg = MountainCarConfig()
    
    # Validate environment setup
    env = MountainCarCL(cfg.TASKS)
    env.reset(seed=seeds[0])
    print(f"🔬 Environment validation: state_dim={env.observation_space.shape[0]}, action_dim={env.action_space.n}, tasks={env.total_tasks}")
    
    # All experimental conditions
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
    print("COMPLETE RQ3 EXPERIMENT: All Hybrid Conditions - MountainCar Version")
    print("=" * 80)
    
    all_results = {}
    
    for condition in conditions:
        print(f"\n🏃 Running condition: {condition}")
        condition_results = []
        
        for seed in seeds:
            print(f"\n  🔧 Seed {seed} starting...")
            start_time = time.time()
            
            # Set different global seeds for each run
            set_global_seed(seed)
            
            result = run_mountaincar_experiment(cfg, condition, seed, episodes_per_task, cycles)
            condition_results.append(result)
            
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
    
    # Comprehensive results analysis
    print("\n" + "=" * 80)
    print("RQ3 COMPLETE RESULTS SUMMARY - MountainCar VERSION")
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
            
            mean_perf = np.mean(perfs)
            std_perf = np.std(perfs)
            mean_mem = np.mean(mems)
            mean_eff = np.mean(effs)
            mean_comp = np.mean(comps)
            mean_ft = np.mean(fts)
            mean_bt = np.mean(bts)
            
            print(f"{condition:<25} {mean_perf:>5.1f}±{std_perf:<2.1f} {mean_mem:>8.3f} "
                  f"{mean_eff:>6.1f} {mean_comp:>6.2f}x {mean_ft:>6.1f} {mean_bt:>6.1f}")
    
    print("=" * 80)
    
    # Key insights
    print("\n🔍 KEY INSIGHTS:")
    
    # Compare hybrid methods
    hybrid_conditions = ["hybrid_basic", "hybrid_uncertainty", "hybrid_distill"]
    best_hybrid = None
    best_performance = -float('inf')
    
    for cond in hybrid_conditions:
        if cond in all_results:
            perf = np.mean([r['avg_performance'] for r in all_results[cond]])
            if perf > best_performance:
                best_performance = perf
                best_hybrid = cond
    
    if best_hybrid:
        print(f"🏆 Best hybrid method: {best_hybrid} (performance: {best_performance:.1f})")
    
    # Memory efficiency champion
    best_efficiency = -float('inf')
    best_eff_condition = None
    
    for cond in conditions:
        if cond in all_results and cond != "baseline_no_replay":
            eff = np.mean([r['memory_efficiency'] for r in all_results[cond]])
            if eff > best_efficiency:
                best_efficiency = eff
                best_eff_condition = cond
    
    if best_eff_condition:
        print(f"💾 Most memory efficient: {best_eff_condition} ({best_efficiency:.1f} perf/MB)")

    print(f"\n📊 Generating comprehensive visualizations in {vis_dir}...")
    create_comprehensive_analysis(all_results, save_dir=vis_dir)
    
    print(f"🔬 End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RQ3: Complete Hybrid Experiments - MountainCar Version')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3],
                        help='Random seeds for experiments')
    parser.add_argument('--episodes-per-task', type=int, default=400,
                        help='Episodes per task')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of cycles through all tasks')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with fewer episodes and seeds')
    
    args = parser.parse_args()
    
    if args.quick_test:
        print("🚀 QUICK TEST MODE - MountainCar VERSION")
        main(seeds=[1], episodes_per_task=50, cycles=1)
    else:
        main(seeds=args.seeds, episodes_per_task=args.episodes_per_task, cycles=args.cycles)

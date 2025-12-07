#!/usr/bin/env python3
"""
RQ2 Experiment Complete Fix: Balanced Imagination Strategy

Key Fixes:
1. More reasonable Dreamer imagination threshold (0.3 instead of 0.12).
2. More conservative expansion strategy for Fully Adaptive mode.
3. Longer warmup and cooldown periods to stabilize training.
"""

from __future__ import annotations
import argparse, math, random, sys, os, time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import copy
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cartpole_config import CartPoleConfig
from detection import (
    LatentSpaceDriftDetector,
    PredictionErrorDetector,
    RewardTrendDetector,
    WeightedMultiModalDetector,
)
from detection.base import DetectionResult
from environments.cartpole_cl import CartPoleCL
from AdaptiveWorldModel import AdaptiveWorldModel, SmartDynamicDQNetwork
from AdaptiveExplorationController import AdaptiveExplorationController

def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FixedDQNetwork(nn.Module):
    """
    Standard fixed-size Deep Q-Network.
    Used for baselines and non-adaptive components.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.reset_count = 0
        self.reset_weights(init_print=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def reset_weights(self, init_print: bool = True):
        """Re-initialize network weights using Kaiming Uniform initialization."""
        for m in (self.fc1, self.fc2, self.fc3):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
        if init_print:
            self.reset_count += 1


class RQ2WorldModelAgent:
    """
    The core agent class for RQ2 experiments.
    Supports various modes:
    - Fixed Baselines (Small/Large)
    - Dreamer-style (Model-Based RL with Imagination)
    - Adaptive Architectures (Policy-only, WM-only, Fully Adaptive)
    """
    # 
    def __init__(self, state_dim: int, action_dim: int,
                 policy_hidden: int = 64, world_hidden: int = 64,
                 condition: str = "adaptive",
                 agent_overrides: Optional[Dict] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.condition = condition
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default configuration
        self.config = {
            'policy_lr': 1e-3,
            'world_model_lr': 1e-3,
            'skip_world_model_training': False,
            'skip_imagination': False,
            'world_model_train_frequency': 1,
            'wm_warmup_episodes': 0,
            'policy_expansion_cooldown_episodes': 0,
        }
        if agent_overrides:
            self.config.update(agent_overrides)
        
        # Initialize networks based on condition
        self._setup_networks(condition, state_dim, action_dim, policy_hidden, world_hidden)

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config['policy_lr'])
        if hasattr(self, 'world_model'):
            self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.config['world_model_lr'])

        self.epsilon = 0.1
        self.base_epsilon = 0.1
        self.adaptive_exploration = condition in ["adaptive_world_model", "fully_adaptive"]

        self.replay_buffer = []
        self.batch_size = 64
        self.gamma = 0.99
        self.update_target_every = 200
        self.steps_done = 0

        self.world_model_errors = []
        self.capacity_history = []
        self.last_policy_expansion_episode = -999

    def _setup_networks(self, condition, state_dim, action_dim, policy_hidden, world_hidden):
        """Initialize the specific network architecture based on the experimental condition."""
        if condition == "small_fixed":
            self.policy_net = FixedDQNetwork(state_dim, action_dim, 64).to(self.device)
            self.has_world_model = False
            self.policy_adaptive = False
            self.world_model_adaptive = False
        elif condition == "large_fixed":
            self.policy_net = FixedDQNetwork(state_dim, action_dim, 128).to(self.device)
            self.has_world_model = False
            self.policy_adaptive = False
            self.world_model_adaptive = False
        elif condition.startswith("dreamer_"):
            self.policy_net = FixedDQNetwork(state_dim, action_dim, policy_hidden).to(self.device)
            self.world_model = AdaptiveWorldModel(state_dim, action_dim, world_hidden).to(self.device)
            self.has_world_model = True
            self.policy_adaptive = False
            self.world_model_adaptive = False
        elif condition == "adaptive_policy_only":
            self.policy_net = SmartDynamicDQNetwork(state_dim, action_dim, policy_hidden).to(self.device)
            self.has_world_model = False
            self.policy_adaptive = True
            self.world_model_adaptive = False
        elif condition == "adaptive_world_model":
            self.policy_net = FixedDQNetwork(state_dim, action_dim, policy_hidden).to(self.device)
            self.world_model = AdaptiveWorldModel(state_dim, action_dim, world_hidden).to(self.device)
            self.has_world_model = True
            self.policy_adaptive = False
            self.world_model_adaptive = True
        elif condition == "fully_adaptive":
            self.policy_net = SmartDynamicDQNetwork(state_dim, action_dim, policy_hidden).to(self.device)
            self.world_model = AdaptiveWorldModel(state_dim, action_dim, world_hidden).to(self.device)
            self.has_world_model = True
            self.policy_adaptive = True
            self.world_model_adaptive = True
            self.adaptive_exploration = True
        else:
            raise ValueError(f"Unknown condition: {condition}")

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def update_world_model(self, state, action, next_state, reward):
        """Train the World Model to predict next state and reward."""
        if not self.has_world_model:
            return 0.0
        if self.config.get('skip_world_model_training', False):
            return 0.0
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([[reward]]).to(self.device)

        next_state_pred, reward_pred = self.world_model(state_t, action_t)
        state_loss = nn.MSELoss()(next_state_pred, next_state_t)
        reward_loss = nn.MSELoss()(reward_pred, reward_t)
        total_loss = state_loss + reward_loss

        self.world_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_optimizer.step()

        err = float(total_loss.item())
        self.world_model_errors.append(err)
        return err

    def should_expand_world_model(self, window_size: int = 50) -> bool:
        """Heuristic: Check if World Model error has spiked significantly compared to baseline."""
        if not self.has_world_model or not self.world_model_adaptive:
            return False
        if len(self.world_model_errors) < window_size:
            return False
        recent = self.world_model_errors[-window_size:]
        baseline = self.world_model_errors[-2*window_size:-window_size] if len(self.world_model_errors) >= 2*window_size else recent
        recent_mean = float(np.mean(recent))
        baseline_mean = float(np.mean(baseline))
        if baseline_mean <= 1e-6:
            return False
        ratio = recent_mean / (baseline_mean + 1e-8)
        return ratio > 2.0

    def expand_world_model_capacity(self, delta: int = 8):
        """Increase the hidden dimension of the World Model."""
        if self.has_world_model and self.world_model_adaptive:
            self.world_model.expand_capacity(delta)
            self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.config['world_model_lr'])

    def expand_policy_capacity(self, delta: int = 8, episode: int = -1):
        """Increase the hidden dimension of the Policy Network (DQN)."""
        if self.policy_adaptive and hasattr(self.policy_net, 'expand_capacity'):
            self.policy_net.expand_capacity(delta)
            if hasattr(self.target_net, 'expand_capacity'):
                self.target_net.expand_capacity(delta)
            self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
            try:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            except Exception:
                pass
            if episode >= 0:
                self.last_policy_expansion_episode = episode

    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            qvals = self.policy_net(state_t)
            return int(torch.argmax(qvals).item())

    def push_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.append((np.array(state, dtype=np.float32),
                                   int(action), float(reward),
                                   np.array(next_state, dtype=np.float32),
                                   bool(done)))
        if len(self.replay_buffer) > 20000:
            self.replay_buffer.pop(0)

    def update(self):
        """Standard DQN update step."""
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def imagine_and_push(self, n_rollouts: Optional[int] = None, rollout_length: Optional[int] = None, policy_noise_eps: float = 0.05):
        """
        Balanced Imagination: Conservatively effective.
        Generates synthetic trajectories using the World Model and adds them to the replay buffer.
        """
        # 
        if not self.has_world_model:
            return
        
        # Key Fix 1: Lower minimum sample requirement (from 500 down to 200)
        min_real_samples = self.config.get('min_real_samples_for_imagination', 200)
        if len(self.replay_buffer) < min_real_samples:
            return
        
        # Key Fix 2: Relax WM quality check (threshold increased from 0.15 to 0.3)
        if len(self.world_model_errors) >= 20:
            recent_wm_error = np.mean(self.world_model_errors[-20:])
            error_threshold = self.config.get('imagination_error_threshold', 0.3)
            if recent_wm_error > error_threshold:
                return
        
        # Use balanced defaults
        n_rollouts = n_rollouts if n_rollouts is not None else self.config.get('imagination_n_rollouts', 4)
        rollout_length = rollout_length if rollout_length is not None else self.config.get('imagination_rollout_length', 3)
        
        # Limit synthetic data ratio (20% instead of 15%)
        current_buffer_size = len(self.replay_buffer)
        max_synthetic_ratio = self.config.get('max_synthetic_ratio', 0.20)
        max_synthetic = int(current_buffer_size * max_synthetic_ratio)
        synthetic_added = 0
        
        for _ in range(n_rollouts):
            if synthetic_added >= max_synthetic:
                break
            
            # Start imagination from a real state sampled from buffer
            s0, _, _, _, _ = random.choice(self.replay_buffer)
            s_t = np.array(s0, dtype=np.float32)
            s_tensor = torch.FloatTensor(s_t).unsqueeze(0).to(self.device)
            
            for step in range(rollout_length):
                if synthetic_added >= max_synthetic:
                    break
                
                with torch.no_grad():
                    # Select action using current policy
                    q = self.policy_net(s_tensor).cpu().numpy()[0]
                    
                    if random.random() < policy_noise_eps:
                        a = random.randint(0, self.action_dim - 1)
                    else:
                        a = int(np.argmax(q))
                    
                    # Predict next state and reward using World Model
                    a_t = torch.LongTensor([a]).to(self.device)
                    ns_pred, r_pred = self.world_model(s_tensor, a_t)
                    
                    ns = ns_pred.cpu().numpy()[0]
                    r = float(r_pred.cpu().numpy()[0][0]) if r_pred.dim() >= 2 else float(r_pred.cpu().numpy()[0])
                
                done = False
                self.push_transition(s_t, a, r, ns, done)
                synthetic_added += 1
                
                s_t = np.array(ns, dtype=np.float32)
                s_tensor = torch.FloatTensor(s_t).unsqueeze(0).to(self.device)

    def get_policy_capacity(self) -> int:
        if hasattr(self.policy_net, 'hidden_dim'):
            return self.policy_net.hidden_dim
        return self.policy_net.fc1.out_features

    def get_world_model_capacity(self) -> int:
        if self.has_world_model:
            return self.world_model.hidden_dim
        return 0

    def record_architecture_metrics(self, episode: int):
        """Log the current capacity and parameter count."""
        policy_capacity = self.get_policy_capacity()
        world_capacity = self.get_world_model_capacity()
        policy_params = self.policy_net.get_parameter_count()
        world_params = self.world_model.get_parameter_count() if self.has_world_model else 0
        self.capacity_history.append({
            'episode': episode,
            'policy_hidden_dim': policy_capacity,
            'world_model_hidden_dim': world_capacity,
            'epsilon': self.epsilon,
            'total_parameters': policy_params + world_params
        })


class SmartMetaController:
    """
    Decides when to trigger architecture expansion based on detection signals and performance.
    """
    def __init__(self, agent, min_capacity: int = 64, max_capacity: int = 128):
        self.agent = agent
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.exploration_controller = AdaptiveExplorationController(
            base_epsilon=0.1, max_epsilon=0.3, min_epsilon=0.01
        )
        self.capacity_cooldown = 0
        self.reset_cooldown = 0
        self.performance_window = []
        self.expansion_count = 0
        self.last_adjustment_episode = -999
        self.episode_count = 0
        self.performance_baseline = None
        self.adjustment_log = []

    def should_expand_policy(self, detector_confidence: float, current_reward: float, world_model_error: float) -> bool:
        """Conservative logic: reduce the frequency of expansion."""
        if self.capacity_cooldown > 0:
            return False
        if len(self.performance_window) < 20:  # Need more observations
            return False
        if self.expansion_count >= 2:  # Limit to max 2 expansions (reduced from 3)
            return False
        
        recent_perf = np.mean(self.performance_window[-10:])
        baseline_perf = np.mean(self.performance_window[-20:-10]) if len(self.performance_window) >= 20 else recent_perf
        
        perf_drop_ratio = (baseline_perf - recent_perf) / (baseline_perf + 1e-8)
        
        # Stricter expansion condition: requires stronger signal
        strong_signal = (detector_confidence > 0.8 and perf_drop_ratio > 0.20)
        
        return strong_signal and self.agent.get_policy_capacity() < self.max_capacity

    def should_expand_world_model(self, world_model_error: float) -> bool:
        if self.capacity_cooldown > 0:
            return False
        if not self.agent.has_world_model or not self.agent.world_model_adaptive:
            return False
        if world_model_error < 0.02:
            return False
        return self.agent.should_expand_world_model()

    def step(self, detector_confidence: float, current_reward: float,
             world_model_error: float, task_change_detected: bool, episode: int,
             world_model_uncertainty: Optional[float] = None) -> dict:
        
        self.performance_window.append(current_reward)
        self.episode_count += 1

        if len(self.performance_window) >= 25 and self.performance_baseline is None:
            self.performance_baseline = np.mean(self.performance_window[:25])
        elif len(self.performance_window) >= 50:
            self.performance_baseline = 0.995 * self.performance_baseline + 0.005 * np.mean(self.performance_window[-10:])

        if len(self.performance_window) > 120:
            self.performance_window.pop(0)

        try:
            # Update exploration rate using the controller
            new_epsilon = self.exploration_controller.update(
                episode_reward=current_reward,
                world_model_error=world_model_error,
                task_change_detected=task_change_detected,
                world_model_uncertainty=world_model_uncertainty
            )
        except Exception:
            new_epsilon = 0.1

        self.agent.epsilon = new_epsilon

        decisions = {
            'epsilon_adjusted': new_epsilon,
            'policy_capacity_changed': False,
            'world_model_capacity_changed': False,
            'action_taken': 'none'
        }

        # Try to expand WM first (Higher priority)
        if self.should_expand_world_model(world_model_error):
            if world_model_uncertainty is not None and world_model_uncertainty > 0.45:
                pass  # postpone if uncertain
            else:
                self.agent.expand_world_model_capacity(delta=8)
                decisions['world_model_capacity_changed'] = True
                decisions['action_taken'] = 'expand_world_model'
                self.capacity_cooldown = 60
                self.adjustment_log.append({
                    'episode': episode,
                    'type': 'world_model_expansion',
                    'new_capacity': self.agent.get_world_model_capacity(),
                })
        # Then check Policy expansion
        elif self.should_expand_policy(detector_confidence, current_reward, world_model_error):
            if hasattr(self.agent.policy_net, 'expand_capacity'):
                delta = min(8, self.max_capacity - self.agent.get_policy_capacity())
                if delta > 0:
                    old = self.agent.get_policy_capacity()
                    self.agent.expand_policy_capacity(delta, episode=episode)
                    new = self.agent.get_policy_capacity()
                    decisions['policy_capacity_changed'] = True
                    decisions['action_taken'] = 'expand_policy'
                    self.capacity_cooldown = 150  # Longer cooldown (increased from 120 to 150)
                    self.expansion_count += 1
                    self.adjustment_log.append({
                        'episode': episode,
                        'type': 'policy_expansion',
                        'new_capacity': new,
                    })

        if self.capacity_cooldown > 0:
            self.capacity_cooldown -= 1

        try:
            decisions.update(self.exploration_controller.get_exploration_stats())
        except Exception:
            pass

        return decisions

    def get_adjustment_summary(self) -> dict:
        return {
            'total_adjustments': len(self.adjustment_log),
            'policy_expansions': sum(1 for a in self.adjustment_log if a['type'] == 'policy_expansion'),
            'world_model_expansions': sum(1 for a in self.adjustment_log if a['type'] == 'world_model_expansion'),
            'adjustment_log': self.adjustment_log
        }


def get_dreamer_config(condition: str) -> Dict:
    """Balanced Dreamer configuration: Conservatively effective."""
    
    config = {
        'skip_world_model_training': False,
        'skip_imagination': False,
        'world_model_lr': 1e-3,
        'world_model_train_frequency': 1,
        
        # Key Balance Point: Increased from 0.12 to 0.3 (allows more imagination)
        'imagination_threshold': 0.3,  # Uncertainty threshold
        'imagination_error_threshold': 0.3,  # Error threshold
        'imagination_n_rollouts': 4,  # Moderate
        'imagination_rollout_length': 3,  # Moderate
        
        # Moderate Warmup (Reduced from 100 to 60)
        'wm_warmup_episodes': 60,
        
        # Lower minimum sample requirement (Reduced from 500 to 200)
        'min_real_samples_for_imagination': 200,
        'max_synthetic_ratio': 0.20,  # 20% synthetic data
    }
    
    if condition == "dreamer_style":
        pass  # Use default balanced config
    elif condition == "dreamer_no_imagination":
        config['skip_imagination'] = True
    elif condition == "dreamer_no_wm_training":
        config['skip_world_model_training'] = True
        config['skip_imagination'] = True
    
    return config


def build_detector_for_rq2_smart(state_dim: int, action_dim: int):
    """Builds the ensemble drift detector."""
    # 
    reward_kwargs = dict(window_size=6, baseline_window=25, drop_threshold=0.35, confirm_steps=3, cooldown_episodes=20)
    latent_kwargs = dict(drift_threshold=1.8, window_size=25, baseline_window=60, confirm_steps=3, cooldown_episodes=25)
    prediction_kwargs = dict(ratio_threshold=2.6, window_size=25, confirm_steps=3, cooldown_episodes=25)
    detector = WeightedMultiModalDetector([
        RewardTrendDetector(**reward_kwargs),
        PredictionErrorDetector(state_dim, action_dim, **prediction_kwargs),
        LatentSpaceDriftDetector(state_dim, **latent_kwargs)
    ], vote_threshold=0.5, detector_weights=[1.4, 1.1, 0.7])
    return detector


def _extract_wm_error_and_uncert(agent, state, action, next_state, reward):
    """Helper to extract World Model error and uncertainty metrics."""
    wm_err = 0.0
    wm_unc = None
    try:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        action_t = torch.LongTensor([action]).to(agent.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
        reward_t = torch.FloatTensor([[reward]]).to(agent.device)

        if hasattr(agent.world_model, 'compute_prediction_error'):
            info = agent.world_model.compute_prediction_error(state_t, action_t, next_state_t, reward_t)
            if isinstance(info, dict):
                wm_err = float(info.get('total_error', 0.0))
                if 'next_std_mean' in info:
                    wm_unc = float(info['next_std_mean'])
    except Exception:
        wm_err = float(np.mean(agent.world_model_errors[-10:])) if len(agent.world_model_errors) > 0 else 0.0

    if wm_unc is None:
        try:
            arr = np.array(agent.world_model_errors[-40:]) if len(agent.world_model_errors) > 0 else np.array([0.0])
            if arr.size > 1:
                std = float(np.std(arr))
                mean = float(np.mean(arr)) + 1e-8
                wm_unc = float(np.clip(std / (mean + 1e-8), 0.0, 2.0))
        except Exception:
            wm_unc = None

    return wm_err, wm_unc


def run_rq2_experiment(cfg, condition: str, seed: int,
                       episodes_per_task: int = 100, cycles: int = 2, warmup_episodes: int = 50,
                       condition_overrides: Optional[Dict] = None):
    set_global_seed(seed)
    env = CartPoleCL(cfg.TASKS)
    env.reset(seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    use_detector = condition in ["adaptive_world_model", "fully_adaptive", "adaptive_policy_only"]
    detector = build_detector_for_rq2_smart(state_dim, action_dim) if use_detector else None
    if detector:
        detector.reset()

    if condition.startswith("dreamer_"):
        dreamer_cfg = get_dreamer_config(condition)
        if condition_overrides is None:
            condition_overrides = {}
        condition_overrides[condition] = dreamer_cfg
    elif condition == "fully_adaptive":
        if condition_overrides is None:
            condition_overrides = {}
        condition_overrides[condition] = {
            'world_model_train_frequency': 1,
            'world_model_lr': 1e-3,
            
            # Key Fix: Relaxed imagination (threshold raised from 0.10 to 0.25)
            'imagination_threshold': 0.25,
            'imagination_error_threshold': 0.28,
            'imagination_n_rollouts': 3,
            'imagination_rollout_length': 3,
            'wm_warmup_episodes': 60,
            
            # Key Fix: Ultra-long cooldown to prevent over-expansion
            'policy_expansion_cooldown_episodes': 100,  # Increased from 60 to 100
            
            'min_real_samples_for_imagination': 200,
            'max_synthetic_ratio': 0.18,  # More conservative than dreamer (18% vs 20%)
        }
    
    agent_override_cfg = condition_overrides.get(condition, {}) if condition_overrides else None
    agent = RQ2WorldModelAgent(state_dim, action_dim, condition=condition, agent_overrides=agent_override_cfg)

    use_meta_controller = condition in ["adaptive_world_model", "fully_adaptive"]
    meta_controller = SmartMetaController(agent) if use_meta_controller else None

    task_sequence = []
    for _ in range(cycles):
        for task_id in range(env.total_tasks):
            task_sequence.extend([task_id] * episodes_per_task)

    episode_rewards = []
    detection_episodes = []
    architecture_changes = []
    meta_decisions_log = []

    current_task = 0
    env.change_task(current_task)
    episodes_completed = 0

    # Main Training Loop
    for episode_idx in range(len(task_sequence)):
        desired_task = task_sequence[episode_idx]
        if desired_task != env.current_task:
            env.change_task(desired_task)

        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        detector_result = None
        task_change_detected = False
        world_model_error = 0.0
        world_model_uncert = None

        while not done:
            action = agent.select_action(np.array(state))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # World model training
            cooldown_episodes = agent.config.get('policy_expansion_cooldown_episodes', 0)
            in_policy_cooldown = (episodes_completed - agent.last_policy_expansion_episode) < cooldown_episodes
            
            train_freq = agent.config.get('world_model_train_frequency', 1)
            should_train_wm = (agent.steps_done % train_freq == 0) if train_freq > 0 else False
            
            if agent.config.get('skip_world_model_training', False) or not should_train_wm or in_policy_cooldown:
                world_model_error = 0.0
            else:
                try:
                    world_model_error = agent.update_world_model(state, action, next_state, reward)
                except Exception:
                    world_model_error = 0.0

            # Extract WM error and uncertainty
            try:
                wm_err, wm_unc = _extract_wm_error_and_uncert(agent, state, action, next_state, reward)
                if wm_err is not None:
                    world_model_error = float(wm_err)
                if wm_unc is not None:
                    world_model_uncert = float(wm_unc)
            except Exception:
                world_model_uncert = None

            confidence = 1.0
            if detector is not None:
                try:
                    detector_result = detector.update(np.array(state, dtype=np.float32),
                                                      int(action), float(reward),
                                                      np.array(next_state, dtype=np.float32),
                                                      done, info={"task_id": env.current_task})
                except Exception:
                    detector_result = DetectionResult(detected=False, score=0.0, metadata={})
                md = detector_result.metadata if isinstance(detector_result.metadata, dict) else {}
                confidence = md.get("confidence", md.get("score", 1.0))
                task_change_detected = bool(detector_result.detected)
                if episodes_completed >= warmup_episodes and detector_result.detected:
                    if episode_idx not in detection_episodes:
                        detection_episodes.append(episode_idx)

            agent.push_transition(state, action, reward, next_state, done)
            try:
                agent.update()
            except Exception:
                pass

            episode_reward += float(reward)
            state = next_state

        # Imagination gating: Decide whether to generate synthetic data
        if agent.has_world_model and (condition.startswith("dreamer_") or condition == "fully_adaptive"):
            warmup_ep = agent.config.get('wm_warmup_episodes', 0)
            in_warmup = episodes_completed < warmup_ep
            cooldown_ep = agent.config.get('policy_expansion_cooldown_episodes', 0)
            in_policy_cooldown = (episodes_completed - agent.last_policy_expansion_episode) < cooldown_ep
            
            if agent.config.get('skip_imagination', False) or in_warmup or in_policy_cooldown:
                pass  # skip imagination
            else:
                uncert_threshold = agent.config.get('imagination_threshold', 0.3)
                error_threshold = agent.config.get('imagination_error_threshold', 0.3)
                
                do_imagine = False
                try:
                    if world_model_uncert is not None:
                        do_imagine = (world_model_uncert < uncert_threshold)
                    else:
                        recent_err = np.mean(agent.world_model_errors[-30:]) if len(agent.world_model_errors) > 0 else 0.0
                        do_imagine = (recent_err < error_threshold)
                except Exception:
                    do_imagine = False

                if do_imagine:
                    agent.imagine_and_push(policy_noise_eps=0.06)

        episode_rewards.append(episode_reward)

        # Meta Controller Step (Update epsilon/Capacity)
        if meta_controller is not None and episodes_completed >= warmup_episodes:
            try:
                decisions = meta_controller.step(confidence, episode_reward, world_model_error, 
                                                 task_change_detected, episode_idx, 
                                                 world_model_uncertainty=world_model_uncert)
            except Exception:
                decisions = {}
            meta_decisions_log.append(decisions)
            if decisions.get('action_taken', 'none') != 'none':
                architecture_changes.append({
                    'episode': episode_idx,
                    'action': decisions.get('action_taken', 'none'),
                    'epsilon': decisions.get('epsilon_adjusted', agent.epsilon)
                })

        agent.record_architecture_metrics(episode_idx)
        episodes_completed += 1

    # Evaluation
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

    avg_eval = float(np.mean(list(eval_rewards.values())))
    adjustment_summary = meta_controller.get_adjustment_summary() if meta_controller else {}
    efficiency_metrics = {
        'parameters_per_reward': (agent.policy_net.get_parameter_count() + 
                                 (agent.world_model.get_parameter_count() if agent.has_world_model else 0)) / (avg_eval + 1e-8),
        'adjustment_frequency': len(architecture_changes) / len(task_sequence),
        'final_policy_capacity': agent.get_policy_capacity(),
        'final_world_model_capacity': agent.get_world_model_capacity(),
        'final_epsilon': agent.epsilon,
        'avg_eval': avg_eval,
        'reset_count': getattr(agent.policy_net, 'reset_count', 0),
        'has_world_model': agent.has_world_model,
        'meta_adjustments': adjustment_summary
    }

    return (float(np.mean(episode_rewards)), eval_rewards, agent.capacity_history, 
            architecture_changes, efficiency_metrics, meta_decisions_log, episode_rewards)


def create_visualizations(all_results, save_dir="./visualizations"):
    """Generate visualization plots for RQ2 experiment results."""
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    conditions = list(all_results.keys())
    
    # 1. Performance Comparison Plot
    plt.figure(figsize=(12, 8))
    
    means = []
    stds = []
    for condition in conditions:
        evals = [r['avg_eval'] for r in all_results[condition]]
        means.append(np.mean(evals))
        stds.append(np.std(evals))
    
    x_pos = np.arange(len(conditions))
    bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 2, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Experimental Conditions')
    plt.ylabel('Average Evaluation Reward')
    plt.title('RQ2: Performance Comparison Across Conditions')
    plt.xticks(x_pos, conditions, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rq2_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning Curves Comparison
    plt.figure(figsize=(14, 8))
    key_conditions = ['small_fixed', 'large_fixed', 'dreamer_style', 'fully_adaptive']
    colors = ['red', 'blue', 'green', 'purple']
    
    for i, condition in enumerate(key_conditions):
        if condition in all_results:
            if len(all_results[condition]) > 0:
                episode_rewards = all_results[condition][0]['episode_rewards']
                # Smoothing
                window_size = 20
                smoothed = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(len(smoothed)), smoothed, 
                        label=condition, color=colors[i], linewidth=2, alpha=0.8)
    
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward (window=20)')
    plt.title('RQ2: Learning Curves Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rq2_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Capacity Evolution (For adaptive methods)
    adaptive_conditions = ['fully_adaptive', 'adaptive_world_model', 'adaptive_policy_only']
    
    for condition in adaptive_conditions:
        if condition in all_results and len(all_results[condition]) > 0:
            plt.figure(figsize=(12, 6))
            
            run_data = all_results[condition][0]
            if 'capacity_history' in run_data and run_data['capacity_history']:
                cap_hist = run_data['capacity_history']
                episodes = [c['episode'] for c in cap_hist]
                
                # Plot Policy Capacity
                if 'policy_hidden_dim' in cap_hist[0]:
                    policy_caps = [c['policy_hidden_dim'] for c in cap_hist]
                    plt.plot(episodes, policy_caps, label='Policy Capacity', linewidth=3, marker='o', markersize=3)
                
                # Plot WM Capacity
                if 'world_model_hidden_dim' in cap_hist[0]:
                    wm_caps = [c['world_model_hidden_dim'] for c in cap_hist]
                    plt.plot(episodes, wm_caps, label='World Model Capacity', linewidth=3, marker='s', markersize=3)
                
                plt.xlabel('Episode')
                plt.ylabel('Hidden Dimension')
                plt.title(f'RQ2: Capacity Evolution - {condition}')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/rq2_capacity_evolution_{condition}.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    # 4. Parameter Efficiency
    plt.figure(figsize=(10, 6))
    efficiency_data = {}
    for condition in conditions:
        if condition in all_results:
            evals = [r['avg_eval'] for r in all_results[condition]]
            # Estimate parameter counts
            if condition == 'small_fixed':
                params = 64 * 64 * 3 
            elif condition == 'large_fixed':
                params = 128 * 128 * 3
            elif condition == 'dreamer_style':
                params = 64 * 64 * 3 + 64 * 64 * 3
            elif 'adaptive' in condition:
                if len(all_results[condition]) > 0:
                    params = all_results[condition][0]['efficiency_metrics'].get('total_parameters', 10000)
                else:
                    params = 10000
            else:
                params = 10000
            
            efficiency = np.mean(evals) / params
            efficiency_data[condition] = efficiency
    
    plt.bar(efficiency_data.keys(), efficiency_data.values(), 
            color='lightcoral', alpha=0.7)
    plt.xlabel('Condition')
    plt.ylabel('Performance per Parameter')
    plt.title('RQ2: Parameter Efficiency Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rq2_parameter_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Architecture Adjustment Stats
    adaptive_with_meta = ['fully_adaptive', 'adaptive_world_model']
    adjustments_data = {}
    
    for condition in adaptive_with_meta:
        if condition in all_results:
            total_adjustments = []
            for run in all_results[condition]:
                if 'meta_adjustments' in run['efficiency_metrics']:
                    total_adjustments.append(
                        run['efficiency_metrics']['meta_adjustments'].get('total_adjustments', 0)
                    )
            if total_adjustments:
                adjustments_data[condition] = np.mean(total_adjustments)
    
    if adjustments_data:
        plt.figure(figsize=(8, 6))
        plt.bar(adjustments_data.keys(), adjustments_data.values(), 
                color='goldenrod', alpha=0.7)
        plt.xlabel('Condition')
        plt.ylabel('Average Number of Adjustments')
        plt.title('RQ2: Architecture Adjustment Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/rq2_adjustment_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Save Detailed Summary to JSON
    summary_data = {}
    for condition in conditions:
        if condition in all_results:
            evals = [r['avg_eval'] for r in all_results[condition]]
            summary_data[condition] = {
                'mean_performance': float(np.mean(evals)),
                'std_performance': float(np.std(evals)),
                'min_performance': float(np.min(evals)),
                'max_performance': float(np.max(evals)),
                'n_runs': len(evals)
            }
    
    with open(f'{save_dir}/rq2_results_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"✅ Visualizations saved to {save_dir}/")
    print(f"📊 Generated {len(conditions)} conditions analysis")
    print(f"📈 Created performance comparison chart")
    print(f"📉 Created learning curves for key conditions") 
    print(f"🔄 Created capacity evolution charts for adaptive methods")
    print(f"📐 Created parameter efficiency analysis")
    print(f"📋 Saved detailed results to JSON")

def main(seeds: List[int] = [0, 1, 2], episodes_per_task: int = 100, cycles: int = 2, 
         warmup_episodes: int = 50, condition_overrides: Optional[Dict] = None):
    
    # === 📂 Path Management: Consistent with RQ3 ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Results JSON: experiments/results/rq2_cartpole/
    results_dir = os.path.join(base_dir, "results", "rq2_cartpole")
    # Visualizations: visualizations/rq2_cartpole/
    vis_dir = os.path.join(os.path.dirname(base_dir), "visualizations", "rq2_cartpole")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📂 Results will be saved to: {results_dir}")
    print(f"📊 Visualizations will be saved to: {vis_dir}")
    # ===============================================

    cfg = CartPoleConfig()
    
    # Simplified list of critical conditions
    conditions = [
        "small_fixed",              # Baseline 1
        "large_fixed",              # Baseline 2
        "dreamer_style",            # Dreamer (Fixed)
        "dreamer_no_imagination",   # Ablation: No imagination
        "adaptive_policy_only",     # Adaptive policy
        "adaptive_world_model",     # Adaptive WM
        "fully_adaptive",           # Full system (Fixed)
    ]
    
    print("=" * 80)
    print("RQ2 EXPERIMENT: FINAL FIXED VERSION")
    print("=" * 80)
    print(f"Conditions: {len(conditions)} critical conditions")
    print(f"Seeds: {seeds}")
    print(f"Episodes per task: {episodes_per_task}")
    print(f"Total episodes: {episodes_per_task * len(cfg.TASKS) * cycles}")
    print("=" * 80)

    all_results = {}

    for condition in conditions:
        print(f"\n🏃 Running condition: {condition}")
        condition_results = []
        for seed in seeds:
            set_global_seed(seed)
            print(f"  Seed {seed}...", end=" ")
            
            result = run_rq2_experiment(cfg, condition, seed, episodes_per_task, 
                                        cycles, warmup_episodes, condition_overrides)
            
            avg_train, eval_rewards, capacity_history, architecture_changes, efficiency_metrics, meta_decisions, episode_rewards = result
            
            # Construct result dictionary for single seed
            seed_result = {
                'seed': seed,
                'condition': condition,
                'avg_train': avg_train,
                'avg_eval': efficiency_metrics['avg_eval'],
                'eval_rewards': eval_rewards,
                'episode_rewards': episode_rewards,
                'capacity_history': capacity_history,
                'architecture_changes': architecture_changes,
                'efficiency_metrics': efficiency_metrics,
                'meta_decisions': meta_decisions,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            condition_results.append(seed_result)
            
            # === 💾 Fix 1: Save individual run results to JSON ===
            json_filename = f'temp_results_{condition}_seed{seed}.json'
            json_path = os.path.join(results_dir, json_filename)
            
            # Helper: Handle Numpy types for JSON
            def convert_numpy(obj):
                if isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return obj

            with open(json_path, 'w') as f:
                json.dump(seed_result, f, indent=2, default=convert_numpy)
            # ========================================
            
            wm_info = f", WM={efficiency_metrics['final_world_model_capacity']}" if efficiency_metrics['has_world_model'] else ""
            eps_info = f", ε={efficiency_metrics['final_epsilon']:.3f}" if condition in ["adaptive_world_model", "fully_adaptive"] else ""
            adj_info = f", adj={efficiency_metrics['meta_adjustments'].get('total_adjustments', 0)}" if condition in ["adaptive_world_model", "fully_adaptive"] else ""
            print(f"avg_eval={efficiency_metrics['avg_eval']:.1f}{wm_info}{eps_info}{adj_info}")
        
        all_results[condition] = condition_results

    # Summary
    print("\n" + "=" * 80)
    print("RQ2 FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<25} {'Avg Eval':<12} {'Policy':<8} {'WM':<8} {'Adj':<6}")
    print("-" * 80)
    
    for condition in conditions:
        if condition in all_results and len(all_results[condition]) > 0:
            evals = [r['avg_eval'] for r in all_results[condition]]
            policy_caps = [r['efficiency_metrics']['final_policy_capacity'] for r in all_results[condition]]
            wm_caps = [r['efficiency_metrics']['final_world_model_capacity'] for r in all_results[condition]]
            adjustments = [r['efficiency_metrics']['meta_adjustments'].get('total_adjustments', 0) for r in all_results[condition]]
            
            mean_eval = np.mean(evals)
            std_eval = np.std(evals) if len(evals) > 1 else 0.0
            mean_policy = np.mean(policy_caps)
            mean_wm = np.mean(wm_caps)
            mean_adj = np.mean(adjustments)
            
            print(f"{condition:<25} {mean_eval:>6.1f}±{std_eval:<4.1f} {mean_policy:>6.1f} {mean_wm:>6.1f} {mean_adj:>5.1f}")
    
    print("=" * 80)
    
    # Key Comparisons
    print("\n📊 Key Comparisons:")
    print("-" * 80)
    
    dreamer_perf = np.mean([r['avg_eval'] for r in all_results.get('dreamer_style', [])])
    dreamer_no_img = np.mean([r['avg_eval'] for r in all_results.get('dreamer_no_imagination', [])])
    fully_perf = np.mean([r['avg_eval'] for r in all_results.get('fully_adaptive', [])])
    adaptive_wm_perf = np.mean([r['avg_eval'] for r in all_results.get('adaptive_world_model', [])])
    
    if dreamer_perf > 0 and dreamer_no_img > 0:
        imagination_gain = dreamer_perf - dreamer_no_img
        print(f"Imagination benefit: {imagination_gain:+.1f} (dreamer_style vs no_imagination)")
    
    if fully_perf > 0 and adaptive_wm_perf > 0:
        full_adaptive_gain = fully_perf - adaptive_wm_perf
        print(f"Full adaptive gain: {full_adaptive_gain:+.1f} (fully_adaptive vs adaptive_world_model)")
    
    print("\n✅ Validation:")
    if dreamer_perf > 145:
        print("✅ Dreamer imagination is working (>145)")
    else:
        print(f"⚠️  Dreamer may still have issues ({dreamer_perf:.1f})")
    
    if 155 < fully_perf < 185:
        print("✅ Fully adaptive is balanced (155-185)")
    elif fully_perf < 100:
        print(f"❌ Fully adaptive is broken ({fully_perf:.1f})")
    else:
        print(f"⚠️  Fully adaptive needs tuning ({fully_perf:.1f})")
    
    print("=" * 80)
    
    # === 📊 Fix 2: Generate visualizations to new path ===
    print(f"\n📈 Generating visualizations in {vis_dir}...")
    create_visualizations(all_results, save_dir=vis_dir)  # 👈 Passed save_dir here
    # ========================================
    
    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--episodes-per-task', type=int, default=100)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--warmup-episodes', type=int, default=50)
    parser.add_argument('--quick-test', action='store_true',
                        help="Quick test with 1 seed and fewer episodes")
    args = parser.parse_args()
    
    if args.quick_test:
        print("🚀 QUICK TEST MODE")
        main(seeds=[0], episodes_per_task=100, cycles=2, warmup_episodes=50)
    else:
        main(seeds=args.seeds, episodes_per_task=args.episodes_per_task, 
             cycles=args.cycles, warmup_episodes=args.warmup_episodes)
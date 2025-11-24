#!/usr/bin/env python3
"""
Enhanced Latent-Space Replay Buffer with World Model Integration - FIXED VERSION
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Optional, Tuple
import heapq
import random
import time

class WorldModel(nn.Module):
    """Simple world model for next state prediction - FIXED: Better architecture"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * 2)  # Predict mean and log_std
        )
        
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state and reward"""
        # Convert action to one-hot if needed
        if action.dim() == 1:
            action_onehot = torch.nn.functional.one_hot(
                action.long(), num_classes=self.action_dim
            ).float()
        else:
            action_onehot = action.float()
        
        x = torch.cat([state, action_onehot], dim=-1)
        
        # Predict next state distribution
        state_output = self.transition_net(x)
        state_mean = state_output[..., :self.state_dim]
        state_log_std = state_output[..., self.state_dim:]
        state_std = torch.exp(state_log_std)
        
        # Predict reward
        reward = self.reward_net(x)
        
        return state_mean, state_std, reward
    
    def predict_with_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next state with uncertainty estimation"""
        state_mean, state_std, reward = self.forward(state, action)
        
        # Uncertainty is the average standard deviation across state dimensions
        uncertainty = state_std.mean(dim=-1)
        
        return state_mean, reward, uncertainty


class LatentEncoder(nn.Module):
    """Compress state-action pairs into compact latent representations"""
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = 3):  # FIXED: Reduced latent_dim for real compression
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim + action_dim)
        )
        
    def encode(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Encode state-action to latent (action as one-hot)"""
        if action.dim() == 1:
            action_onehot = torch.nn.functional.one_hot(
                action.long(), 
                num_classes=self.action_dim
            ).float()
        else:
            action_onehot = action.float()
        
        x = torch.cat([state, action_onehot], dim=-1)
        return self.encoder(x)
    
    def decode(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent back to state-action"""
        reconstruction = self.decoder(latent)
        state = reconstruction[..., :self.state_dim]
        action = reconstruction[..., self.state_dim:]
        return state, action
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass with reconstruction"""
        latent = self.encode(state, action)
        state_recon, action_recon = self.decode(latent)
        return latent, state_recon, action_recon


class UncertaintyEstimator:
    """Estimate epistemic uncertainty for prioritized sampling"""
    def __init__(self, window_size: int = 100):
        self.prediction_errors = deque(maxlen=window_size)
        self.world_model_stds = deque(maxlen=window_size)
        
    def update(self, prediction_error: float, world_model_std: Optional[float] = None):
        """Update uncertainty estimates"""
        self.prediction_errors.append(float(prediction_error))
        if world_model_std is not None:
            self.world_model_stds.append(float(world_model_std))
    
    def get_uncertainty_score(self) -> float:
        """Compute combined uncertainty score"""
        if len(self.prediction_errors) == 0:
            return 1.0
        
        recent_error = np.mean(list(self.prediction_errors)[-20:])
        baseline_error = np.mean(list(self.prediction_errors)) + 1e-8
        error_ratio = recent_error / baseline_error
        
        if len(self.world_model_stds) > 0:
            wm_uncertainty = np.mean(list(self.world_model_stds)[-10:])
        else:
            wm_uncertainty = 0.5
        
        combined = 0.6 * error_ratio + 0.4 * wm_uncertainty
        return float(np.clip(combined, 0.1, 2.0))


class LatentReplayBuffer:
    """
    Enhanced version with world model and hybrid sampling - FIXED VERSION
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 3,  # FIXED: Reduced from 8 to 3 for real compression
        max_latent_samples: int = 2000,
        max_raw_samples: int = 1000,
        max_synthetic_samples: int = 1000,
        device: str = "cpu",
        use_world_model: bool = False
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_latent_samples = max_latent_samples
        self.max_raw_samples = max_raw_samples
        self.max_synthetic_samples = max_synthetic_samples
        self.device = device
        self.use_world_model = use_world_model
        
        # Latent encoder/decoder
        self.encoder = LatentEncoder(state_dim, action_dim, latent_dim).to(device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        
        # World model
        if use_world_model:
            self.world_model = WorldModel(state_dim, action_dim).to(device)
            self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
            self.synthetic_buffer = []  # Store synthetic transitions
        else:
            self.world_model = None
            self.synthetic_buffer = []
        
        # Storage structures
        self.latent_buffer = []
        self.raw_buffer = []
        
        # Uncertainty tracking
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # Task boundaries
        self.task_boundaries = {}
        self.current_task_samples = 0
        
        # Statistics
        self.compression_ratio = 1.0
        self.total_samples_added = 0
        self.synthetic_samples_generated = 0
        
        # Debugging
        self.debug_counter = 0
        
    def train_encoder(self, batch_states: torch.Tensor, batch_actions: torch.Tensor):
        """Train latent encoder"""
        if len(batch_states) < 8:
            return 0.0
            
        self.encoder.train()
        latent, state_recon, action_recon = self.encoder(batch_states, batch_actions)
        
        state_loss = nn.MSELoss()(state_recon, batch_states)
        
        action_onehot = torch.nn.functional.one_hot(
            batch_actions.long(), 
            num_classes=self.action_dim
        ).float()
        action_loss = nn.MSELoss()(action_recon, action_onehot)
        
        loss = state_loss + 0.5 * action_loss
        
        self.encoder_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.encoder_optimizer.step()
        
        return float(loss.item())
    
    def train_world_model(self, batch_states: torch.Tensor, batch_actions: torch.Tensor, 
                         batch_next_states: torch.Tensor, batch_rewards: torch.Tensor):
        """Train world model on real transitions"""
        if not self.use_world_model or len(batch_states) < 8:
            return 0.0, 0.0
        
        self.world_model.train()
        
        # Predict next state and reward
        state_mean, state_std, pred_rewards = self.world_model(batch_states, batch_actions)
        
        # State prediction loss (Gaussian negative log likelihood)
        state_dist = torch.distributions.Normal(state_mean, state_std)
        state_loss = -state_dist.log_prob(batch_next_states).mean()
        
        # Reward prediction loss
        reward_loss = nn.MSELoss()(pred_rewards.squeeze(), batch_rewards)
        
        # Combined loss
        total_loss = state_loss + reward_loss
        
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_model_optimizer.step()
        
        # Calculate uncertainty for this batch
        with torch.no_grad():
            _, _, uncertainty = self.world_model.predict_with_uncertainty(batch_states, batch_actions)
            avg_uncertainty = uncertainty.mean().item()
        
        return float(total_loss.item()), avg_uncertainty
    
    def generate_synthetic_samples(self, n_samples: int, task_id: int, 
                                 high_uncertainty: bool = False):
        """Generate synthetic transitions using world model - FIXED: Better noise and constraints"""
        if not self.use_world_model or len(self.raw_buffer) < 10:
            return 0
        
        self.world_model.eval()
        
        generated = 0
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample a real transition as starting point
                if len(self.raw_buffer) == 0:
                    continue
                    
                real_sample = random.choice(self.raw_buffer)
                state, action, _, _, _, _ = real_sample
                
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_t = torch.LongTensor([action]).to(self.device)
                
                # Predict next state and reward
                next_state_mean, reward_pred, uncertainty = self.world_model.predict_with_uncertainty(
                    state_t, action_t
                )
                
                # FIXED: Reduced noise and added CartPole-specific constraints
                noise_scale = 0.05  # FIXED: Reduced from 0.1 to 0.05
                noise = torch.randn_like(next_state_mean) * noise_scale
                next_state = next_state_mean + noise
                
                # FIXED: Better done prediction with environment-specific bounds
                next_state_np = next_state.cpu().numpy()[0]
                
                # Check state dimension to determine environment logic
                if len(next_state_np) == 4: # CartPole (x, x_dot, theta, theta_dot)
                    # CartPole bounds: position ±2.4, angle ±0.2 rad
                    out_of_bounds = (abs(next_state_np[0]) > 2.4 or abs(next_state_np[2]) > 0.2)
                    done = out_of_bounds
                elif len(next_state_np) == 2: # MountainCar (position, velocity)
                    # MountainCar done: position >= 0.5 (goal reached)
                    done = (next_state_np[0] >= 0.5)
                else:
                    # Generic fallback
                    done = False
                
                # Store synthetic transition
                synthetic_sample = (
                    state.copy(),
                    action,
                    float(reward_pred.item()),
                    next_state.cpu().numpy()[0],
                    done,
                    task_id,
                    True,  # is_synthetic
                    float(uncertainty.item())  # uncertainty
                )
                
                if len(self.synthetic_buffer) < self.max_synthetic_samples:
                    self.synthetic_buffer.append(synthetic_sample)
                else:
                    # Replace oldest synthetic sample
                    self.synthetic_buffer.pop(0)
                    self.synthetic_buffer.append(synthetic_sample)
                
                generated += 1
        
        self.synthetic_samples_generated += generated
        
        if self.debug_counter % 500 == 0 and generated > 0:
            print(f"🤖 Generated {generated} synthetic samples (total: {len(self.synthetic_buffer)})")
        
        return generated
    
    def add_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        task_id: int,
        is_synthetic: bool = False,
        world_model_uncertainty: Optional[float] = None
    ):
        """Store transition in appropriate buffers"""
        self.total_samples_added += 1
        self.debug_counter += 1
        
        if self.debug_counter % 1000 == 0:
            print(f"🔍 LatentBuffer [sample {self.total_samples_added}]: "
                  f"reward={reward:.3f}, done={done}, synthetic={is_synthetic}")

        # Store to raw buffer (if real sample)
        if not is_synthetic:
            raw_sample = (state.copy(), action, reward, next_state.copy(), done, task_id)
            
            if len(self.raw_buffer) < self.max_raw_samples:
                self.raw_buffer.append(raw_sample)
            else:
                self.raw_buffer.pop(0)
                self.raw_buffer.append(raw_sample)

            # Store to latent buffer
            if self.total_samples_added > 50:  # FIXED: Start latent storage earlier
                self._add_to_latent_buffer(state, action, reward, next_state, done, task_id)
        
        # Store synthetic samples directly to synthetic buffer
        else:
            synthetic_sample = (state.copy(), action, reward, next_state.copy(), done, task_id)
            if len(self.synthetic_buffer) < self.max_synthetic_samples:
                self.synthetic_buffer.append(synthetic_sample)
            else:
                self.synthetic_buffer.pop(0)
                self.synthetic_buffer.append(synthetic_sample)
        
        if self.debug_counter % 1000 == 0:
            print(f"📊 Buffer Stats: raw={len(self.raw_buffer)}, "
                  f"latent={len(self.latent_buffer)}, synthetic={len(self.synthetic_buffer)}")
    
    def _add_to_latent_buffer(self, state, action, reward, next_state, done, task_id):
        """Add compressed version to latent buffer - FIXED: True compression"""
        try:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_t = torch.LongTensor([action]).to(self.device)
            
            with torch.no_grad():
                latent = self.encoder.encode(state_t, action_t)
                latent_np = latent.cpu().numpy()[0]
            
            # FIXED: Only store latent vector, not the full next_state
            # This is true compression - we reconstruct next_state from world model when needed
            latent_sample = (latent_np, action, reward, done, task_id)  # Removed next_state
            
            if len(self.latent_buffer) < self.max_latent_samples:
                self.latent_buffer.append(latent_sample)
            else:
                self.latent_buffer.pop(0)
                self.latent_buffer.append(latent_sample)
                
        except Exception as e:
            if self.debug_counter % 500 == 0:
                print(f"⚠️ Latent encoding failed: {e}")

    def sample_batch(
        self,
        batch_size: int,
        task_id: Optional[int] = None,
        use_latent: bool = True,
        use_synthetic: bool = False,
        temperature: float = 1.0,
        force_latent_ratio: float = 0.0,
        synthetic_ratio: float = 0.0,
        uncertainty_guided: bool = False
    ) -> Tuple:
        """
        Enhanced sampling with multiple sources - FIXED: Uncertainty-guided sampling
        """
        # Calculate sample counts from each source
        n_synthetic = int(batch_size * synthetic_ratio) if use_synthetic else 0
        n_latent = int(batch_size * force_latent_ratio) if use_latent else 0
        n_raw = batch_size - n_synthetic - n_latent
        
        # Ensure we have enough samples from each source
        n_synthetic = min(n_synthetic, len(self.synthetic_buffer))
        n_latent = min(n_latent, len(self.latent_buffer))
        n_raw = min(n_raw, len(self.raw_buffer))
        
        # Adjust if any source doesn't have enough samples
        total_available = n_raw + n_latent + n_synthetic
        if total_available < batch_size:
            # Redistribute remaining slots to available sources
            remaining = batch_size - total_available
            if len(self.raw_buffer) > n_raw:
                n_raw += min(remaining, len(self.raw_buffer) - n_raw)
            elif len(self.latent_buffer) > n_latent:
                n_latent += min(remaining, len(self.latent_buffer) - n_latent)
            elif len(self.synthetic_buffer) > n_synthetic:
                n_synthetic += min(remaining, len(self.synthetic_buffer) - n_synthetic)
        
        # Sample from each source
        all_samples = []
        
        # Sample raw (always uniform)
        if n_raw > 0:
            raw_samples = self._sample_from_buffer(self.raw_buffer, n_raw, task_id)
            all_samples.extend(raw_samples)
        
        # Sample latent (with decoding) - FIX: Add uncertainty guidance for latent
        if n_latent > 0:
            if uncertainty_guided and len(self.latent_buffer) > 0:
                latent_samples = self._sample_latent_with_uncertainty(n_latent, task_id)
            else:
                latent_samples = self._sample_latent_with_decoding(n_latent, task_id)
            all_samples.extend(latent_samples)
        
        # Sample synthetic - FIX: Add uncertainty guidance
        if n_synthetic > 0:
            if uncertainty_guided and len(self.synthetic_buffer) > 0:
                synthetic_samples = self._sample_synthetic_with_uncertainty(n_synthetic, task_id)
            else:
                synthetic_samples = self._sample_from_buffer(self.synthetic_buffer, n_synthetic, task_id)
            all_samples.extend(synthetic_samples)
        
        if len(all_samples) == 0:
            return self._empty_batch()
        
        # Debug info
        if self.debug_counter % 2000 == 0 and len(all_samples) > 0:
            rewards = [s[2] for s in all_samples]
            synthetic_count = sum(1 for s in all_samples if len(s) > 6 and s[6])  # is_synthetic flag
            latent_count = n_latent
            raw_count = n_raw
            
            uncertainty_info = ""
            if uncertainty_guided:
                # Calculate average uncertainty of sampled synthetic samples
                synth_uncertainties = [s[7] for s in all_samples if len(s) > 7]
                if synth_uncertainties:
                    avg_uncertainty = np.mean(synth_uncertainties)
                    uncertainty_info = f", avg_uncertainty={avg_uncertainty:.3f}"
            
            print(f"🎯 Hybrid batch: {raw_count} raw + {latent_count} latent + {synthetic_count} synthetic{uncertainty_info}, "
                  f"avg_reward={np.mean(rewards):.3f}")
        
        return tuple(map(np.array, zip(*[s[:5] for s in all_samples])))
    
    def _sample_from_buffer(self, buffer: List, n_samples: int, task_id: Optional[int] = None):
        """Sample from a given buffer"""
        if len(buffer) == 0:
            return []
        
        buffer_to_sample = buffer
        if task_id is not None:
            buffer_to_sample = [s for s in buffer if s[5] == task_id]
            if len(buffer_to_sample) == 0:
                buffer_to_sample = buffer
        
        if len(buffer_to_sample) < n_samples:
            indices = np.random.choice(len(buffer_to_sample), n_samples, replace=True)
        else:
            indices = np.random.choice(len(buffer_to_sample), n_samples, replace=False)
        
        return [buffer_to_sample[i] for i in indices]
    
    def _sample_synthetic_with_uncertainty(self, n_samples: int, task_id: Optional[int] = None):
        """FIXED: Improved uncertainty sampling with diversity"""
        if len(self.synthetic_buffer) == 0:
            return []
        
        buffer_to_sample = self.synthetic_buffer
        if task_id is not None:
            buffer_to_sample = [s for s in self.synthetic_buffer if s[5] == task_id]
            if len(buffer_to_sample) == 0:
                buffer_to_sample = self.synthetic_buffer
        
        if len(buffer_to_sample) < n_samples:
            # Not enough samples, use all with replacement
            indices = np.random.choice(len(buffer_to_sample), n_samples, replace=True)
            return [buffer_to_sample[i] for i in indices]
        
        # Extract uncertainties (index 7 in synthetic samples)
        uncertainties = np.array([s[7] for s in buffer_to_sample])
        
        # FIXED: Improved diversity sampling - mix high, medium, low uncertainty
        sorted_indices = np.argsort(uncertainties)
        n_high = int(n_samples * 0.3)      # 30% high uncertainty
        n_medium = int(n_samples * 0.5)    # 50% medium uncertainty  
        n_low = n_samples - n_high - n_medium  # 20% low uncertainty
        
        # Sample from different uncertainty ranges
        high_indices = sorted_indices[-n_high:] if n_high > 0 else []
        medium_start = len(sorted_indices) // 2 - n_medium // 2
        medium_indices = sorted_indices[medium_start:medium_start + n_medium] if n_medium > 0 else []
        low_indices = sorted_indices[:n_low] if n_low > 0 else []
        
        indices = list(high_indices) + list(medium_indices) + list(low_indices)
        
        if self.debug_counter % 2000 == 0:
            sampled_uncertainties = uncertainties[indices]
            high_mean = np.mean(sampled_uncertainties[:n_high]) if n_high > 0 else 0
            medium_mean = np.mean(sampled_uncertainties[n_high:n_high+n_medium]) if n_medium > 0 else 0
            low_mean = np.mean(sampled_uncertainties[-n_low:]) if n_low > 0 else 0
            
            print(f"🎯 Diversity uncertainty sampling: high={high_mean:.3f}, medium={medium_mean:.3f}, low={low_mean:.3f}")
        
        return [buffer_to_sample[i] for i in indices]
    
    def _sample_latent_with_uncertainty(self, n_samples: int, task_id: Optional[int] = None):
        """Sample latent transitions with uncertainty-based prioritization"""
        if len(self.latent_buffer) == 0:
            return []
        
        buffer_to_sample = self.latent_buffer
        if task_id is not None:
            buffer_to_sample = [s for s in self.latent_buffer if s[5] == task_id]
            if len(buffer_to_sample) == 0:
                buffer_to_sample = self.latent_buffer
        
        if len(buffer_to_sample) < n_samples:
            indices = np.random.choice(len(buffer_to_sample), n_samples, replace=True)
        else:
            # For latent samples, we don't have stored uncertainty
            # Use random sampling as fallback, or implement reconstruction-based uncertainty
            indices = np.random.choice(len(buffer_to_sample), n_samples, replace=False)
        
        decoded_samples = []
        for idx in indices:
            latent_sample = buffer_to_sample[idx]
            latent_vec, action, reward, done, sample_task_id = latent_sample  # FIXED: removed next_state
            
            try:
                # FIXED: Use world model to predict next_state
                if self.world_model is None:
                    continue  # Skip if no world model available
                    
                with torch.no_grad():
                    latent_t = torch.FloatTensor(latent_vec).unsqueeze(0).to(self.device)
                    state_recon, action_recon = self.encoder.decode(latent_t)
                    state_recon = state_recon.cpu().numpy()[0]
                    
                    # Use world model to predict next_state
                    state_t = torch.FloatTensor(state_recon).unsqueeze(0).to(self.device)
                    action_t = torch.LongTensor([action]).to(self.device)
                    next_state_pred, _, _ = self.world_model.predict_with_uncertainty(state_t, action_t)
                    next_state = next_state_pred.cpu().numpy()[0]
                
                decoded_sample = (state_recon, action, reward, next_state, done, sample_task_id)
                decoded_samples.append(decoded_sample)
                
            except Exception as e:
                continue
        
        return decoded_samples
    
    def _sample_latent_with_decoding(self, n_samples: int, task_id: Optional[int] = None):
        """FIXED: Properly handle latent samples with world model prediction"""
        if len(self.latent_buffer) == 0:
            return []
        
        buffer_to_sample = self.latent_buffer
        if task_id is not None:
            buffer_to_sample = [s for s in self.latent_buffer if s[5] == task_id]
            if len(buffer_to_sample) == 0:
                buffer_to_sample = self.latent_buffer
        
        if len(buffer_to_sample) < n_samples:
            indices = np.random.choice(len(buffer_to_sample), n_samples, replace=True)
        else:
            indices = np.random.choice(len(buffer_to_sample), n_samples, replace=False)
        
        decoded_samples = []
        for idx in indices:
            latent_sample = buffer_to_sample[idx]
            latent_vec, action, reward, done, sample_task_id = latent_sample  # FIXED: removed next_state
            
            try:
                # FIXED: Critical fix - only use latent samples if we have world model
                if self.world_model is None:
                    continue  # Skip if no world model to predict next_state
                    
                with torch.no_grad():
                    latent_t = torch.FloatTensor(latent_vec).unsqueeze(0).to(self.device)
                    state_recon, action_recon = self.encoder.decode(latent_t)
                    state_recon = state_recon.cpu().numpy()[0]
                    
                    # Use world model to predict next_state (CRITICAL FIX)
                    state_t = torch.FloatTensor(state_recon).unsqueeze(0).to(self.device)
                    action_t = torch.LongTensor([action]).to(self.device)
                    next_state_pred, _, _ = self.world_model.predict_with_uncertainty(state_t, action_t)
                    next_state = next_state_pred.cpu().numpy()[0]
                
                decoded_sample = (state_recon, action, reward, next_state, done, sample_task_id)
                decoded_samples.append(decoded_sample)
                
            except Exception as e:
                if self.debug_counter % 1000 == 0:
                    print(f"⚠️ Latent decoding failed: {e}")
                continue
        
        return decoded_samples
    
    def _empty_batch(self):
        """Return empty batch"""
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )
    
    def update_task_boundary(self, task_id: int):
        """Mark task boundary for task-aware sampling"""
        self.task_boundaries[task_id] = (
            self.total_samples_added - self.current_task_samples,
            self.total_samples_added
        )
        self.current_task_samples = 0
    
    def get_memory_usage(self) -> Dict[str, float]:
        """FIXED: Correct memory calculation and compression ratio"""
        # CartPole state_dim=4, action_dim=2, latent_dim=3
        # Each float32 = 4 bytes
        
        # Raw buffer: (state[4], action[1], reward[1], next_state[4], done[1], task_id[1]) = 12 elements
        raw_sample_size = (self.state_dim + 1 + 1 + self.state_dim + 1 + 1) * 4  # 12 * 4 = 48 bytes
        raw_memory = len(self.raw_buffer) * raw_sample_size / (1024 * 1024)  # Convert to MB
        
        # FIXED: Latent buffer: (latent[3], action[1], reward[1], done[1], task_id[1]) = 7 elements
        latent_sample_size = (self.latent_dim + 1 + 1 + 1 + 1) * 4  # 7 * 4 = 28 bytes
        latent_memory = len(self.latent_buffer) * latent_sample_size / (1024 * 1024)
        
        # Synthetic buffer: same as raw buffer
        synthetic_memory = len(self.synthetic_buffer) * raw_sample_size / (1024 * 1024)
        
        total_memory = raw_memory + latent_memory + synthetic_memory
        
        # FIXED: Calculate actual compression ratio
        # Compare what we would need if all samples were stored in raw format
        total_samples = len(self.raw_buffer) + len(self.latent_buffer) + len(self.synthetic_buffer)
        if total_samples > 0:
            equivalent_raw_memory = total_samples * raw_sample_size / (1024 * 1024)
            self.compression_ratio = equivalent_raw_memory / total_memory if total_memory > 0 else 1.0
        else:
            self.compression_ratio = 1.0
        
        # DEBUG: Print detailed memory breakdown
        if self.debug_counter % 2000 == 0:
            print(f"🧮 Memory Debug: raw={len(self.raw_buffer)}*{raw_sample_size}B={raw_memory:.3f}MB, "
                  f"latent={len(self.latent_buffer)}*{latent_sample_size}B={latent_memory:.3f}MB, "
                  f"total={total_memory:.3f}MB, compression={self.compression_ratio:.2f}x")
        
        return {
            'total_mb': total_memory,
            'raw_mb': raw_memory,
            'latent_mb': latent_memory,
            'synthetic_mb': synthetic_memory,
            'compression_ratio': self.compression_ratio,
            'samples_raw': len(self.raw_buffer),
            'samples_latent': len(self.latent_buffer),
            'samples_synthetic': len(self.synthetic_buffer)
        }
    
    def get_stats(self) -> Dict:
        """Return buffer statistics"""
        stats = self.get_memory_usage()
        stats['total_samples_added'] = self.total_samples_added
        stats['synthetic_samples_generated'] = self.synthetic_samples_generated
        stats['uncertainty_score'] = self.uncertainty_estimator.get_uncertainty_score()
        return stats

    def validate_world_model_quality(self, n_samples: int = 50) -> Dict[str, float]:
        """Validate world model prediction quality - NEW: Added quality diagnostics"""
        if not self.use_world_model or len(self.raw_buffer) < n_samples:
            return {}
        
        self.world_model.eval()
        
        # Sample real transitions for validation
        samples = random.sample(self.raw_buffer, min(n_samples, len(self.raw_buffer)))
        states = torch.FloatTensor(np.array([s[0] for s in samples])).to(self.device)
        actions = torch.LongTensor(np.array([s[1] for s in samples])).to(self.device)
        next_states_real = torch.FloatTensor(np.array([s[3] for s in samples])).to(self.device)
        rewards_real = torch.FloatTensor(np.array([s[2] for s in samples])).to(self.device)
        
        with torch.no_grad():
            # World model predictions
            next_states_pred, rewards_pred, uncertainties = self.world_model.predict_with_uncertainty(states, actions)
            
            # Calculate prediction errors
            state_mse = torch.mean((next_states_pred - next_states_real) ** 2).item()
            reward_mse = torch.mean((rewards_pred.squeeze() - rewards_real) ** 2).item()
            
            # Physical constraint violations (CartPole specific)
            position_errors = torch.abs(next_states_pred[:, 0] - next_states_real[:, 0]).mean().item()
            
            # Only check angle for CartPole (dim > 2)
            angle_errors = 0.0
            if states.shape[1] >= 4:
                angle_errors = torch.abs(next_states_pred[:, 2] - next_states_real[:, 2]).mean().item()
        
        quality_metrics = {
            'state_prediction_mse': state_mse,
            'reward_prediction_mse': reward_mse,
            'position_error': position_errors,
            'angle_error': angle_errors,
            'avg_uncertainty': torch.mean(uncertainties).item()
        }
        
        # Quality threshold warnings
        if state_mse > 0.1:  # Empirical threshold
            print(f"🚨 World Model Quality Alert: State MSE = {state_mse:.4f} (too high)")
        
        if self.debug_counter % 1000 == 0:
            print(f"🔬 World Model Quality: state_mse={state_mse:.4f}, reward_mse={reward_mse:.4f}")
        
        return quality_metrics


class KnowledgeDistillationLoss(nn.Module):
    """Distillation loss to preserve old task knowledge"""
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        hard_loss: torch.Tensor
    ) -> torch.Tensor:
        """Compute distillation loss"""
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * distill_loss
        
        return total_loss
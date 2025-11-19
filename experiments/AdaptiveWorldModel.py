# AdaptiveWorldModel.py
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional

class AdaptiveWorldModel(nn.Module):
    """
    Ensemble Adaptive World Model with dynamic capacity adjustment.

    - Ensemble of small transition+reward nets to provide mean + uncertainty.
    - forward(state, action) -> (next_state_mean, reward_mean)  (keeps compatibility)
    - predict_mean_std(state, action) -> (next_mean, reward_mean, next_std, reward_std)
    - compute_prediction_error(state, action, next_state, reward) returns dict with errors and uncertainty.
    - expand_capacity(delta) expands hidden dim for all ensemble members preserving learned weights.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        ensemble_size: int = 3,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.ensemble_size = int(ensemble_size)

        # Each ensemble member: transition_net and reward_net
        self.transition_nets = nn.ModuleList()
        self.reward_nets = nn.ModuleList()
        for _ in range(self.ensemble_size):
            self.transition_nets.append(self._build_transition_net(self.hidden_dim))
            self.reward_nets.append(self._build_reward_net(self.hidden_dim))

        # bookkeeping
        self.architecture_history: List[int] = [self.hidden_dim]
        self.adjustment_count = 0

    def _build_transition_net(self, hidden_dim: int) -> nn.Sequential:
        # input: state_dim + action_dim -> hidden_dim -> hidden_dim -> state_dim
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.state_dim),
        )

    def _build_reward_net(self, hidden_dim: int) -> nn.Sequential:
        # input: state_dim + action_dim -> hidden_dim -> 1
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _action_to_onehot(self, action: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Accepts action as:
          - shape (B,) of ints
          - shape (B,1) of ints
          - shape (B, action_dim) already one-hot / float
        Returns one-hot tensor shape (B, action_dim), float, on same device as state.
        """
        if device is None:
            device = action.device
        if action.dim() == 1 or (action.dim() == 2 and action.size(-1) == 1):
            act_idx = action.view(-1).long().to(device)
            return torch.nn.functional.one_hot(act_idx, num_classes=self.action_dim).float().to(device)
        else:
            # assume already one-hot (float)
            return action.float().to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ensemble mean predictions and return (next_state_mean, reward_mean).
        Keeps compatibility with existing training code that expects two outputs.
        """
        next_mean, reward_mean, _, _ = self.predict_mean_std(state, action)
        return next_mean, reward_mean

    def predict_mean_std(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (next_mean, reward_mean, next_std, reward_std)
        - state: (B, state_dim)
        - action: (B,) or (B,1) or (B, action_dim)
        """
        device = state.device
        action_onehot = self._action_to_onehot(action, device=device)
        x = torch.cat([state, action_onehot], dim=-1)

        preds_next: List[torch.Tensor] = []
        preds_reward: List[torch.Tensor] = []

        for i in range(self.ensemble_size):
            tn = self.transition_nets[i]
            rn = self.reward_nets[i]
            preds_next.append(tn(x).unsqueeze(0))    # (1, B, state_dim)
            preds_reward.append(rn(x).unsqueeze(0))  # (1, B, 1)

        preds_next = torch.cat(preds_next, dim=0)       # (E, B, state_dim)
        preds_reward = torch.cat(preds_reward, dim=0)   # (E, B, 1)

        next_mean = torch.mean(preds_next, dim=0)       # (B, state_dim)
        next_std = torch.std(preds_next, dim=0, unbiased=False)  # (B, state_dim)

        reward_mean = torch.mean(preds_reward, dim=0)   # (B, 1)
        reward_std = torch.std(preds_reward, dim=0, unbiased=False)  # (B, 1)

        return next_mean, reward_mean, next_std, reward_std

    def compute_prediction_error(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor) -> Dict[str, float]:
        """
        Compute prediction error using ensemble mean predictions.
        Returns dictionary with:
          - state_error, reward_error, total_error
          - next_std_mean, reward_std_mean (averaged across batch)
        """
        self.eval()
        with torch.no_grad():
            next_mean, reward_mean, next_std, reward_std = self.predict_mean_std(state, action)

            # ensure shapes
            state_error = torch.nn.functional.mse_loss(next_mean, next_state)
            reward_error = torch.nn.functional.mse_loss(reward_mean, reward)

            total_error = state_error + reward_error

            # summarize uncertainty (mean std over batch and dims)
            next_std_mean = float(next_std.mean().item()) if next_std.numel() > 0 else 0.0
            reward_std_mean = float(reward_std.mean().item()) if reward_std.numel() > 0 else 0.0

        return {
            'state_error': float(state_error.item()),
            'reward_error': float(reward_error.item()),
            'total_error': float(total_error.item()),
            'next_std_mean': next_std_mean,
            'reward_std_mean': reward_std_mean
        }

    def expand_capacity(self, delta: int = 8):
        """
        Expand hidden_dim for all ensemble members.
        Preserves overlapping weights; new parameters initialized small.
        """
        old_dim = self.hidden_dim
        new_dim = old_dim + int(delta)
        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')
        print(f"🌍 World Model: Expanding {old_dim} -> {new_dim} (ensemble size = {self.ensemble_size})")

        # helper to expand a single sequential net
        def _expand_seq(seq_module: nn.Sequential, old_d: int, new_d: int) -> nn.Sequential:
            new_layers: List[nn.Module] = []
            # attempt to find device
            # iterate layers and replace Linear layers with expanded versions
            for layer in seq_module:
                if isinstance(layer, nn.Linear):
                    in_f = layer.in_features
                    out_f = layer.out_features
                    new_in = new_d if in_f == old_d else in_f
                    new_out = new_d if out_f == old_d else out_f

                    new_layer = nn.Linear(new_in, new_out).to(device)
                    with torch.no_grad():
                        # copy overlapping block
                        rows = min(out_f, new_out)
                        cols = min(in_f, new_in)
                        new_layer.weight[:rows, :cols].copy_(layer.weight[:rows, :cols])
                        if layer.bias is not None:
                            new_layer.bias[:rows].copy_(layer.bias[:rows])

                        # initialize added rows/cols if any
                        if new_out > out_f:
                            nn.init.normal_(new_layer.weight[out_f:, :], 0.0, 0.01)
                            nn.init.normal_(new_layer.bias[out_f:], 0.0, 0.01)
                        if new_in > in_f:
                            nn.init.normal_(new_layer.weight[:, in_f:], 0.0, 0.01)
                    new_layers.append(new_layer)
                else:
                    # stateless layers can be reused (ReLU, etc.)
                    new_layers.append(layer)
            return nn.Sequential(*new_layers).to(device)

        # expand each ensemble member nets
        for i in range(self.ensemble_size):
            self.transition_nets[i] = _expand_seq(self.transition_nets[i], old_dim, new_dim)
            self.reward_nets[i] = _expand_seq(self.reward_nets[i], old_dim, new_dim)

        self.hidden_dim = new_dim
        self.architecture_history.append(new_dim)
        self.adjustment_count += 1

    def get_parameter_count(self) -> int:
        """Total parameters across ensemble"""
        return sum(p.numel() for p in self.parameters())

    # Optional helper: convenience function to sample imagined rollouts
    def imagine_step(self, state: torch.Tensor, action: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step imagination: returns (next_state_sample, reward_sample)
        If sample==True, sample from ensemble distribution by adding gaussian noise ~ std
        If sample==False, return mean predictions.
        """
        next_mean, reward_mean, next_std, reward_std = self.predict_mean_std(state, action)

        if sample:
            # sample gaussian noise with predicted std
            eps_state = torch.randn_like(next_std)
            eps_reward = torch.randn_like(reward_std)
            next_sample = next_mean + eps_state * next_std
            reward_sample = reward_mean + eps_reward * reward_std
            return next_sample, reward_sample
        else:
            return next_mean, reward_mean


# ----------------------------
# Light-weight WorldModelBasedAgent (kept for compatibility)
# ----------------------------
class WorldModelBasedAgent:
    """
    (Optional) DQN Agent wrapper that uses AdaptiveWorldModel.
    This is minimal and kept for compatibility with earlier scripts.
    """
    def __init__(self, state_dim: int, action_dim: int, policy_hidden: int = 64, world_hidden: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # policy network placeholders (user may override/replace)
        self.policy_net = SmartDynamicDQNetwork(state_dim, action_dim, policy_hidden).to(self.device)
        self.target_net = SmartDynamicDQNetwork(state_dim, action_dim, policy_hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # world model ensemble
        self.world_model = AdaptiveWorldModel(state_dim, action_dim, world_hidden).to(self.device)

        # optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)

        # simple replay
        self.replay_buffer = []
        self.batch_size = 64

        # exploration
        self.epsilon = 0.1
        self.steps_done = 0

        # world model metrics
        self.world_model_errors: List[float] = []

    def update_world_model(self, state, action, next_state, reward) -> float:
        """
        Train world model on single transition (keeps API).
        Returns scalar loss value (float).
        """
        self.world_model.train()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_t = torch.LongTensor([action]).to(self.device)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward_t = torch.FloatTensor([[reward]]).to(self.device)

        next_pred, reward_pred = self.world_model(state_t, action_t)  # mean preds

        state_loss = nn.MSELoss()(next_pred, next_state_t)
        reward_loss = nn.MSELoss()(reward_pred, reward_t)
        total_loss = state_loss + reward_loss

        self.world_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.world_optimizer.step()

        # compute ensemble-aware prediction error summary
        errors = self.world_model.compute_prediction_error(state_t, action_t, next_state_t, reward_t)
        self.world_model_errors.append(errors['total_error'])

        return float(total_loss.item())

    def should_expand_world_model(self, window_size: int = 50) -> bool:
        if len(self.world_model_errors) < window_size:
            return False
        recent = self.world_model_errors[-window_size:]
        baseline = self.world_model_errors[-2*window_size:-window_size] if len(self.world_model_errors) >= 2*window_size else recent
        recent_mean = float(np.mean(recent))
        baseline_mean = float(np.mean(baseline))
        if baseline_mean <= 0.0:
            return False
        return (recent_mean / (baseline_mean + 1e-8)) > 1.5

    def expand_world_model_capacity(self, delta: int = 8):
        self.world_model.expand_capacity(delta)
        # recreate optimizer to pick up new params
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)


# ----------------------------
# SmartDynamicDQNetwork: kept lightweight, supports expand/reset
# ----------------------------
class SmartDynamicDQNetwork(nn.Module):
    """Dynamic DQN network that supports expand_capacity and reset_weights."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = int(hidden_dim)

        self._build_network(self.hidden_dim)
        self.reset_count = 0

    def _build_network(self, hidden_dim: int):
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)
        self.relu = nn.ReLU()

        # init
        for m in (self.fc1, self.fc2, self.fc3):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def expand_capacity(self, delta: int = 8):
        old = self.hidden_dim
        new = old + int(delta)
        print(f"🧠 POLICY: Expanding policy hidden {old} -> {new}")

        device = next(self.parameters()).device if any(True for _ in self.parameters()) else torch.device('cpu')

        # build new layers
        new_fc1 = nn.Linear(self.input_dim, new).to(device)
        new_fc2 = nn.Linear(new, new).to(device)
        new_fc3 = nn.Linear(new, self.output_dim).to(device)

        with torch.no_grad():
            # fc1 copy
            rows1 = min(self.fc1.out_features, new_fc1.out_features)
            cols1 = min(self.fc1.in_features, new_fc1.in_features)
            new_fc1.weight[:rows1, :cols1].copy_(self.fc1.weight[:rows1, :cols1])
            new_fc1.bias[:rows1].copy_(self.fc1.bias[:rows1])
            if new > old:
                nn.init.normal_(new_fc1.weight[old:, :], 0.0, 0.01)
                nn.init.zeros_(new_fc1.bias[old:])

            # fc2 copy
            rows2 = min(self.fc2.out_features, new_fc2.out_features)
            cols2 = min(self.fc2.in_features, new_fc2.in_features)
            new_fc2.weight[:rows2, :cols2].copy_(self.fc2.weight[:rows2, :cols2])
            new_fc2.bias[:rows2].copy_(self.fc2.bias[:rows2])
            if new > old:
                nn.init.normal_(new_fc2.weight[old:, :], 0.0, 0.01)
                nn.init.normal_(new_fc2.weight[:, old:], 0.0, 0.01)
                nn.init.normal_(new_fc2.bias[old:], 0.0, 0.01)

            # fc3 copy (out x in)
            cols3 = min(self.fc3.in_features, new_fc3.in_features)
            new_fc3.weight[:, :cols3].copy_(self.fc3.weight[:, :cols3])
            new_fc3.bias.copy_(self.fc3.bias)
            if new > old:
                nn.init.normal_(new_fc3.weight[:, old:], 0.0, 0.01)

        # replace
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.fc3 = new_fc3
        self.hidden_dim = new

    def reset_weights(self):
        self._build_network(self.hidden_dim)
        self.reset_count += 1
        print(f"🔄 POLICY: Reset policy weights (count={self.reset_count})")

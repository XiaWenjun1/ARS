# AdaptiveExplorationController.py
import numpy as np
from collections import deque
from typing import Optional

class AdaptiveExplorationController:
    """
    Dynamically adjusts exploration rate using:
      - world model prediction error (high -> increase)
      - world model uncertainty (high std -> increase)
      - task change detection (strong immediate boost)
      - performance plateau (small improvement -> modest boost)

    Backwards-compatible: update(...) still accepts (episode_reward, world_model_error, task_change_detected)
    New optional kwarg: world_model_uncertainty (float)
    """
    def __init__(self,
                 base_epsilon: float = 0.05,
                 max_epsilon: float = 0.45,
                 min_epsilon: float = 0.01):
        self.base_epsilon = float(base_epsilon)
        self.max_epsilon = float(max_epsilon)
        self.min_epsilon = float(min_epsilon)
        self.current_epsilon = float(base_epsilon)

        # history windows
        self.performance_window = deque(maxlen=40)
        self.world_model_error_window = deque(maxlen=60)
        self.world_model_uncert_window = deque(maxlen=60)

        # state flags
        self.task_change_detected = False
        self.change_cooldown = 0

        # smoothing param for epsilon updates
        self.alpha = 0.12

    def update(self,
               episode_reward: float,
               world_model_error: float = 0.0,
               task_change_detected: bool = False,
               world_model_uncertainty: Optional[float] = None) -> float:
        """
        Update exploration rate.

        Args:
            episode_reward: latest episode reward (scalar)
            world_model_error: scalar prediction error (can be 0)
            task_change_detected: True/False
            world_model_uncertainty: optional scalar (e.g. next_std_mean from WM ensemble)
        Returns:
            current_epsilon (float)
        """
        # Safe append (avoid bad types)
        try:
            self.performance_window.append(float(episode_reward))
        except Exception:
            pass
        try:
            self.world_model_error_window.append(float(world_model_error))
        except Exception:
            pass
        if world_model_uncertainty is not None:
            try:
                self.world_model_uncert_window.append(float(world_model_uncertainty))
            except Exception:
                pass

        # Handle task change detection -> strong short boost
        if task_change_detected:
            self.task_change_detected = True
            self.change_cooldown = 30

        if self.change_cooldown > 0:
            self.change_cooldown -= 1
            if self.change_cooldown == 0:
                self.task_change_detected = False

        boost_factors = []

        # 1) Task-change strong immediate boost
        if self.task_change_detected:
            boost_factors.append(2.0)

        # 2) World-model error based boost (use ratio to baseline)
        if len(self.world_model_error_window) >= 20:
            recent = np.mean(list(self.world_model_error_window)[-8:])
            baseline = np.mean(list(self.world_model_error_window)[:-8]) if len(self.world_model_error_window) > 8 else recent + 1e-8
            if baseline > 1e-8:
                ratio = recent / baseline
                if ratio > 1.25:
                    boost_factors.append(1.4 + min(0.8, (ratio - 1.25)))  # mild->strong

        # 3) World-model uncertainty based boost (if provided)
        if len(self.world_model_uncert_window) >= 5:
            recent_unc = np.mean(list(self.world_model_uncert_window)[-5:])
            # normalize uncertainty by a small constant to produce reasonable factor; tuned heuristics
            # if uncertainty is relatively large (>0.2) then we boost
            if recent_unc > 0.15:
                # map [0.15, 1.0+] -> [1.2, 2.0]
                factor = 1.0 + np.clip((recent_unc - 0.15) / 0.85, 0.0, 1.0) * 1.0
                boost_factors.append(1.0 + factor * 0.5)  # softer influence

        # 4) Performance plateau boost
        if len(self.performance_window) >= 20:
            recent_perf = np.mean(list(self.performance_window)[-5:])
            prev_perf = np.mean(list(self.performance_window)[-20:-5]) if len(self.performance_window) >= 20 else recent_perf
            if prev_perf > 1e-8:
                rel_change = (recent_perf - prev_perf) / (abs(prev_perf) + 1e-8)
                if rel_change < 0.02:  # <2% improvement -> plateau
                    boost_factors.append(1.25)

        # Compose target epsilon
        if boost_factors:
            max_boost = float(np.max(boost_factors))
            target_epsilon = float(self.base_epsilon * max_boost)
        else:
            # slowly decay to base
            target_epsilon = float(self.base_epsilon)

        # Smooth update
        self.current_epsilon = float(self.alpha * target_epsilon + (1 - self.alpha) * self.current_epsilon)

        # clamp
        self.current_epsilon = float(np.clip(self.current_epsilon, self.min_epsilon, self.max_epsilon))

        return self.current_epsilon

    def get_exploration_stats(self) -> dict:
        """Return current stats (safe)."""
        recent_perf = 0.0
        recent_wm = 0.0
        recent_unc = 0.0
        try:
            if len(self.performance_window) >= 1:
                recent_perf = float(np.mean(list(self.performance_window)[-5:]))
            if len(self.world_model_error_window) >= 1:
                recent_wm = float(np.mean(list(self.world_model_error_window)[-5:]))
            if len(self.world_model_uncert_window) >= 1:
                recent_unc = float(np.mean(list(self.world_model_uncert_window)[-5:]))
        except Exception:
            recent_perf = recent_wm = recent_unc = 0.0

        return {
            'current_epsilon': float(self.current_epsilon),
            'task_change_active': bool(self.task_change_detected),
            'cooldown_remaining': int(self.change_cooldown),
            'recent_performance': recent_perf,
            'recent_wm_error': recent_wm,
            'recent_wm_uncertainty': recent_unc
        }


class MetaController:
    """
    Meta-controller coordinating capacity & exploration.

    Backwards-compatible step signature:
      step(episode_reward, detector_result, episode, **kwargs)
    Accepts optional kwargs:
      - world_model_uncertainty: float (e.g. next_std_mean)
      - world_model_error_override: float
    """
    def __init__(self, agent, world_model_agent):
        self.agent = agent
        self.world_model_agent = world_model_agent

        self.exploration_controller = AdaptiveExplorationController()

        # capacity cooldowns and tracking
        self.capacity_cooldown = 0
        self.reset_cooldown = 0
        self.episode_count = 0
        self.adjustment_log = []

        # guard rails
        self.max_policy_expansions = 6
        self.policy_expansions = 0

    def step(self,
             episode_reward: float,
             detector_result,
             episode: int,
             **kwargs) -> dict:
        """
        Main coordination step.

        Optional kwargs:
          - world_model_uncertainty: float
          - world_model_error_override: float
        """
        self.episode_count += 1

        wm_unc = kwargs.get('world_model_uncertainty', None)
        wm_err_override = kwargs.get('world_model_error_override', None)

        # try to extract recent world_model_error from agent (fallback)
        try:
            if wm_err_override is not None:
                world_model_error = float(wm_err_override)
            else:
                arr = getattr(self.world_model_agent, 'world_model_errors', None)
                if arr and len(arr) > 0:
                    world_model_error = float(np.mean(arr[-10:]))
                else:
                    world_model_error = 0.0
        except Exception:
            world_model_error = 0.0

        # detection info
        task_change_detected = False
        detector_confidence = 0.0
        try:
            task_change_detected = bool(getattr(detector_result, 'detected', False))
            md = getattr(detector_result, 'metadata', {}) or {}
            detector_confidence = float(md.get('confidence', md.get('score', 0.0)))
        except Exception:
            task_change_detected = False
            detector_confidence = 0.0

        # update exploration controller (pass uncertainty if available)
        new_epsilon = self.exploration_controller.update(
            episode_reward=episode_reward,
            world_model_error=world_model_error,
            task_change_detected=task_change_detected,
            world_model_uncertainty=wm_unc
        )
        # assign to agent
        try:
            self.agent.epsilon = new_epsilon
        except Exception:
            pass

        decisions = {
            'epsilon_adjusted': float(new_epsilon),
            'policy_capacity_changed': False,
            'world_model_capacity_changed': False,
            'action_taken': 'none'
        }

        # COOL: prefer expanding world model only when wm error high and uncertainty low-medium (i.e. model is certain but wrong)
        try:
            wm_unc_val = float(wm_unc) if wm_unc is not None else None
        except Exception:
            wm_unc_val = None

        # World model expansion criterion: recent error large AND not in cooldown
        try:
            if self.capacity_cooldown == 0 and hasattr(self.world_model_agent, 'should_expand_world_model'):
                expand_wm = self.world_model_agent.should_expand_world_model()
                # optional extra check: if expand_wm but uncertainty is very high -> postpone expansion (may want to gather data)
                if expand_wm and wm_unc_val is not None and wm_unc_val > 0.45:
                    # postpone if extremely uncertain
                    expand_wm = False

                if expand_wm:
                    # expand
                    try:
                        self.world_model_agent.expand_world_model_capacity(delta=8)
                        decisions['world_model_capacity_changed'] = True
                        decisions['action_taken'] = 'expand_world_model'
                        self.capacity_cooldown = 18
                        self.adjustment_log.append({
                            'episode': episode,
                            'type': 'world_model_expansion',
                            'new_capacity': getattr(self.world_model_agent.world_model, 'hidden_dim', None),
                            'trigger': 'prediction_error',
                            'wm_error': world_model_error,
                            'wm_uncertainty': wm_unc_val
                        })
                    except Exception:
                        pass
        except Exception:
            pass

        # Policy expansion: be conservative, require detector confidence + either high wm_error or poor performance
        try:
            policy_can_expand = hasattr(self.agent.policy_net, 'expand_capacity') and (self.policy_expansions < self.max_policy_expansions)
            performance_bad = float(episode_reward) < 60.0
            high_wm_err = world_model_error > 0.35
            conf_ok = detector_confidence > 0.55

            if self.capacity_cooldown == 0 and policy_can_expand and conf_ok and (high_wm_err or performance_bad):
                # expand policy
                try:
                    self.agent.expand_policy_capacity(delta=8)
                    if hasattr(self.agent, 'target_net') and hasattr(self.agent.target_net, 'expand_capacity'):
                        try:
                            self.agent.target_net.expand_capacity(delta=8)
                        except Exception:
                            pass
                    decisions['policy_capacity_changed'] = True
                    decisions['action_taken'] = 'expand_policy'
                    self.capacity_cooldown = 30
                    self.policy_expansions += 1
                    self.adjustment_log.append({
                        'episode': episode,
                        'type': 'policy_expansion',
                        'new_capacity': getattr(self.agent.policy_net, 'hidden_dim', None),
                        'trigger': 'detector_confidence_and_perf_or_wmerr',
                        'detector_confidence': detector_confidence,
                        'wm_error': world_model_error
                    })
                except Exception:
                    pass
        except Exception:
            pass

        # reduce cooldowns
        if self.capacity_cooldown > 0:
            self.capacity_cooldown -= 1
        if self.reset_cooldown > 0:
            self.reset_cooldown -= 1

        # attach exploration stats
        try:
            decisions.update(self.exploration_controller.get_exploration_stats())
        except Exception:
            pass

        return decisions

    def get_adjustment_summary(self) -> dict:
        return {
            'total_adjustments': len(self.adjustment_log),
            'policy_expansions': sum(1 for a in self.adjustment_log if a.get('type') == 'policy_expansion'),
            'world_model_expansions': sum(1 for a in self.adjustment_log if a.get('type') == 'world_model_expansion'),
            'adjustment_log': self.adjustment_log
        }

import numpy as np
import copy
import os
import torch
from .metrics import TrainingMetrics

class CLTrainer:
    """Continual Learning Trainer (with safe state management for evaluations)"""
    
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.metrics = TrainingMetrics()
        
        # Training state
        self.current_task = 0
        self.episode_count = 0
        self.global_step = 0
        
        # Store final evaluation reward (average episode reward during evaluation, no exploration)
        self.task_final_performances = {}
        # Store average training reward (average episode reward during training, with exploration)
        self.task_training_rewards = {}
        # Store evaluation reward of previous tasks before training new task (no exploration)
        self.pre_train_performances = {}
        # Store initial baseline evaluation reward for all tasks (no exploration)
        self.initial_performances = {}
        # Store agent states after training each task (in-memory)
        self.task_agent_states = {}
        
        # Environment type for file naming and reward sign handling
        self.env_type = type(env).__name__.lower()
        self.negate_forgetting = 'mountaincar' in self.env_type  # MountainCar needs sign flip
        
        # ensure models dir exists
        os.makedirs("models", exist_ok=True)
        
    def train_single_task(self, task_id, episodes=None):
        """Train on a single task, with adaptive epsilon initialization for sparse-reward environments like MountainCar"""
        if episodes is None:
            episodes = self.config.EPISODES_PER_TASK

        print(f"\nTraining task {task_id}")
        self.env.change_task(task_id)

        # === Adaptive epsilon initialization ===
        if "mountaincar" in self.env_type.lower():
            # MountainCar is sparse-reward: start with full exploration
            self.agent.epsilon = self.config.EPSILON_START
            epsilon_decay = getattr(self.config, "EPSILON_DECAY_RATE", 0.998)  # Use decay rate from config
            epsilon_min = self.config.EPSILON_END
            print(f"Detected MountainCar environment → starting epsilon = {self.agent.epsilon:.2f}, decay rate = {epsilon_decay:.3f}")
        else:
            # Normal environments use config defaults
            self.agent.epsilon = self.config.EPSILON_START
            epsilon_decay = getattr(self.config, "EPSILON_DECAY_RATE", getattr(self.config, "EPSILON_DECAY", 0.995))
            epsilon_min = getattr(self.config, "EPSILON_END", 0.01)

        task_rewards = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_loss = 0
            update_count = 0

            for _ in range(self.config.MAX_STEPS_PER_EPISODE):
                # Select action
                action = self.agent.select_action(state)

                # Execute action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store experience
                self.agent.push_memory(state, action, reward, next_state, done)

                # Update network
                loss = self.agent.update()
                if loss is not None and loss > 0:
                    episode_loss += loss
                    update_count += 1

                state = next_state
                episode_reward += reward
                episode_length += 1
                self.global_step += 1

                if done:
                    break

            # Calculate average loss
            avg_loss = episode_loss / max(update_count, 1)

            # Record metrics
            self.metrics.record_episode(
                episode_reward, episode_length, avg_loss,
                getattr(self.agent, 'epsilon', 0.0), task_id
            )

            task_rewards.append(episode_reward)
            self.episode_count += 1

            # === Adaptive epsilon decay ===
            # gradually reduce exploration rate
            # For MountainCar, keep high exploration longer (sparse reward problem)
            if "mountaincar" in self.env_type.lower():
                # Three-stage decay to balance exploration vs exploitation
                # Stage 1: Keep high exploration for 200 episodes (episode < 200)
                # Stage 2: Medium exploration for 200 episodes (200 <= episode < 400)
                # Stage 3: Low exploration for final 100 episodes (episode >= 400)
                if episode < 200:
                    decay_rate = 0.9995  # Very slow: keep ~90% exploration
                elif episode < 400:
                    decay_rate = 0.995  # Medium: gradual decrease
                else:
                    decay_rate = 0.98  # Fast: exploit learned policy
            else:
                decay_rate = epsilon_decay
            
            self.agent.epsilon = max(epsilon_min, self.agent.epsilon * decay_rate)

            # Optional: Monitor performance but don't reverse epsilon decay
            if "mountaincar" in self.env_type.lower() and np.mean(task_rewards[-10:]) < -180:
                # Just log, but don't increase epsilon
                pass  # Would log if needed

            # Print progress
            if (episode + 1) % self.config.LOG_INTERVAL == 0:
                recent_rewards = task_rewards[-self.config.LOG_INTERVAL:]
                avg_reward = np.mean(recent_rewards)
                print(f"Task {task_id} | Episode {episode + 1}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"ε: {self.agent.epsilon:.3f}")

        # Calculate average reward for all episodes
        mean_task_reward = np.mean(task_rewards) if len(task_rewards) > 0 else 0.0
        print(f"Task {task_id} completed - {episodes}-episode average reward: {mean_task_reward:.2f}")

        return mean_task_reward
    
    def save_task_model(self, task_id):
        """Save model to disk for a specific task and also save in-memory state"""
        # Use environment type to distinguish between different environments
        env_prefix = self.env_type.replace('cl', '')  # Remove 'cl' suffix
        model_path = f"models/{env_prefix}_task_{task_id}_model.pth"
        
        checkpoint = {
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': getattr(self.agent, 'epsilon', 0.0),
            'steps_done': getattr(self.agent, 'steps_done', 0),
            'task_id': task_id,
            'final_performance': self.task_final_performances.get(task_id, 0)
        }
        torch.save(checkpoint, model_path)
        print(f"Task {task_id} model saved to: {model_path}")
        
        # Also save in-memory state for fast restore (deepcopy to avoid mutation)
        self.save_agent_state(task_id)
    
    def load_task_model(self, task_id):
        """Load model for a specific task from disk and place into agent"""
        # Use environment type to distinguish between different environments
        env_prefix = self.env_type.replace('cl', '')  # Remove 'cl' suffix
        model_path = f"models/{env_prefix}_task_{task_id}_model.pth"
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            try:
                self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                # optimizer load might fail if different device / or optimizer state incompatible; ignore but warn
                print("Warning: could not fully restore optimizer state.")
            self.agent.epsilon = checkpoint.get('epsilon', getattr(self.agent, 'epsilon', 0.0))
            self.agent.steps_done = checkpoint.get('steps_done', getattr(self.agent, 'steps_done', 0))
            # also store to in-memory for quicker later use
            self.save_agent_state(task_id)
            print(f"Task {task_id} model loaded from: {model_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    def _capture_agent_state(self):
        """Unified method: capture current agent state (return state dictionary)"""
        try:
            return {
                'policy_net_state': copy.deepcopy(self.agent.policy_net.state_dict()),
                'target_net_state': copy.deepcopy(self.agent.target_net.state_dict()),
                'optimizer_state': copy.deepcopy(self.agent.optimizer.state_dict()),
                'epsilon': getattr(self.agent, 'epsilon', 0.0),
                'steps_done': getattr(self.agent, 'steps_done', 0)
            }
        except Exception as e:
            print(f"Warning: failed to capture agent state: {e}")
            return None
    
    def _restore_agent_state(self, state, context="memory"):
        """Unified method: restore agent state"""
        if state is None:
            print(f"Warning: No state to restore from {context}")
            return
        
        try:
            self.agent.policy_net.load_state_dict(state['policy_net_state'])
            self.agent.target_net.load_state_dict(state['target_net_state'])
            try:
                self.agent.optimizer.load_state_dict(state['optimizer_state'])
            except Exception:
                print(f"Warning: could not fully restore optimizer state from {context}")
            self.agent.epsilon = state.get('epsilon', getattr(self.agent, 'epsilon', 0.0))
            self.agent.steps_done = state.get('steps_done', getattr(self.agent, 'steps_done', 0))
        except Exception as e:
            print(f"Warning: failed to restore agent state from {context}: {e}")
    
    def save_agent_state(self, task_id):
        """Save current agent state to memory dictionary (deep copy to prevent subsequent changes from affecting)"""
        state = self._capture_agent_state()
        if state is not None:
            self.task_agent_states[task_id] = state
    
    def load_agent_state(self, task_id):
        """Load specified task's agent state (from memory)"""
        if task_id in self.task_agent_states:
            self._restore_agent_state(self.task_agent_states[task_id], "memory")
        else:
            print(f"Warning: No saved in-memory state found for task {task_id}")
    
    def _snapshot_current_agent_state(self):
        """Internal use: return deep copy snapshot of current agent state (for temporary save/restore)"""
        return self._capture_agent_state()
    
    def _restore_agent_state_snapshot(self, snapshot):
        """Internal use: restore agent state from snapshot"""
        self._restore_agent_state(snapshot, "snapshot")
    
    def evaluate_task(self, task_id, episodes=50, verbose=True, num_evaluations=3):
        """Evaluate performance on a specific task with multiple evaluations for stability"""
        self.env.change_task(task_id)
        all_rewards = []
        
        # Perform multiple evaluations for stability
        for eval_round in range(num_evaluations):
            total_rewards = []
            
            for episode in range(episodes):
                state, _ = self.env.reset()
                episode_reward = 0
                
                for _ in range(self.config.MAX_STEPS_PER_EPISODE):
                    action = self.agent.select_action(state, training=False)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    
                    state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                
                total_rewards.append(episode_reward)
            
            # Calculate mean reward for this evaluation round
            eval_mean = np.mean(total_rewards) if len(total_rewards) > 0 else 0.0
            all_rewards.append(eval_mean)
        
        # Calculate final mean across all evaluation rounds
        final_mean = np.mean(all_rewards)
        final_std = np.std(all_rewards)
        
        if verbose:
            print(f"Task {task_id} evaluation: {final_mean:.2f} ± {final_std:.2f} (over {num_evaluations} evaluations, {episodes} episodes each)")
        
        return final_mean
    
    def run_continual_learning(self):
        """Run complete continual learning training with robust state management"""
        task_performances = {}
        
        print(f"\n{'='*50}")
        print("Phase 1: Evaluating Initial Baseline Performance")
        print(f"{'='*50}")
        for task_id in range(self.env.total_tasks):
            performance = self.evaluate_task(task_id, episodes=20, verbose=True, num_evaluations=2)
            self.initial_performances[task_id] = performance
        
        # Train each task sequentially
        for task_id in range(self.env.total_tasks):
            print(f"\n{'='*50}")
            print(f"Starting Task {task_id}")
            print(f"{'='*50}")
            
            if task_id == 0:
                print(f"\nPhase 2: Training Task {task_id} from initial state")
                final_reward = self.train_single_task(task_id)
                self.task_training_rewards[task_id] = final_reward
                
                print(f"\nPhase 3: Evaluating Task {task_id} Final Performance")
                final_performance = self.evaluate_task(task_id, episodes=30, verbose=True, num_evaluations=3)
                self.task_final_performances[task_id] = final_performance
                task_performances[task_id] = final_performance
                
                self.save_task_model(task_id)
                
            else:
                print(f"\nPhase 4: Evaluating Previous Tasks Retention BEFORE Training Task {task_id}")
                
                # Snapshot current agent state to avoid pollution
                snapshot = self._snapshot_current_agent_state()
                
                # Evaluate all previous tasks using their saved states.
                # We'll try in-memory first, else load from disk.
                for prev_task in range(task_id):
                    # Prefer in-memory state (faster), else load from disk into agent
                    if prev_task in self.task_agent_states:
                        # restore in-memory state, evaluate, then continue
                        self.load_agent_state(prev_task)
                    else:
                        # load from disk (and it will also save into in-memory via load_task_model)
                        self.load_task_model(prev_task)
                    
                    retention_performance = self.evaluate_task(prev_task, episodes=30, verbose=True, num_evaluations=3)
                    self.pre_train_performances[prev_task] = retention_performance
                    print(f"Task {prev_task} before training T{task_id}: {retention_performance:.2f}")
                
                # Restore the snapshot to ensure we start training from the correct pre-training state
                self._restore_agent_state_snapshot(snapshot)
                
                # Phase 5: Train current task starting from previous task's state (task_id-1)
                print(f"\nPhase 5: Training Task {task_id} from Task {task_id-1}'s State")
                
                # Ensure we load the previous task's final state as training start point
                if (task_id - 1) in self.task_agent_states:
                    self.load_agent_state(task_id - 1)
                else:
                    # If not in memory, try disk (this will also populate in-memory)
                    self.load_task_model(task_id - 1)
                
                # Now train current task
                final_reward = self.train_single_task(task_id)
                self.task_training_rewards[task_id] = final_reward
                
                print(f"\nPhase 6: Evaluating Task {task_id} Final Performance")
                final_performance = self.evaluate_task(task_id, episodes=30, verbose=True, num_evaluations=3)
                self.task_final_performances[task_id] = final_performance
                task_performances[task_id] = final_performance
                
                # Save current task model (disk + in-memory)
                self.save_task_model(task_id)
                
                print(f"\nPhase 7: Evaluating Catastrophic Forgetting After Training Task {task_id}")
                
                for prev_task in range(task_id):
                    current_performance = self.evaluate_task(prev_task, episodes=30, verbose=False, num_evaluations=3)
                    
                    pre_train_performance = self.pre_train_performances.get(prev_task, 0)
                    
                    # Record performance change (will calculate forgetting internally)
                    self.metrics.record_task_performance(
                        f"{prev_task}_after_{task_id}",
                        pre_train_performance,
                        current_performance
                    )
                    
                    # Get forgetting from stored data to ensure consistency
                    task_key = f"{prev_task}_after_{task_id}"
                    stored_forgetting = self.metrics.catastrophic_forgetting.get(task_key, {}).get('forgetting', 0)
                    
                    print(f"Task {prev_task} forgetting after training task {task_id}: {stored_forgetting:.2f}")
                    print(f"  Before training T{task_id}: {pre_train_performance:.2f}")
                    print(f"  After training T{task_id}:  {current_performance:.2f}")
            
            print(f"\nTask {task_id} completed!")
        
        # Final summary
        print(f"\n{'='*50}")
        print("Continual Learning Training Completed!")
        print(f"{'='*50}")
        
        return task_performances
    
    def get_training_summary(self):
        """Get training summary"""
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.global_step,
            'final_epsilon': getattr(self.agent, 'epsilon', 0.0),
            'metrics': self.metrics,
            'initial_performances': self.initial_performances,
            'task_final_performances': self.task_final_performances,
            'task_training_rewards': self.task_training_rewards,
            'pre_train_performances': self.pre_train_performances,
            'task_agent_states': self.task_agent_states
        }
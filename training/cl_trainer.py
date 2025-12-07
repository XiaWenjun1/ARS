import numpy as np
import copy
import os
import torch
from .metrics import TrainingMetrics

class CLTrainer:
    """
    Orchestrates the Continual Learning (CL) training process.
    Manages agent training on sequential tasks, evaluates performance,
    and handles agent state saving/loading for robust experimentation.
    """
    
    def __init__(self, agent, env, config):
        """
        Initializes the CLTrainer.

        Args:
            agent (BaseAgent): The reinforcement learning agent to be trained.
            env (ContinualLearningWrapper): The wrapped environment supporting task changes.
            config (BaseConfig): Configuration object containing hyperparameters and settings.
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.metrics = TrainingMetrics() # Utility to record and process training metrics.
        
        # Training state trackers.
        self.current_task = 0
        self.episode_count = 0
        self.global_step = 0
        
        # Performance storage dictionaries.
        # Stores the final average evaluation reward for each task after it has been trained.
        self.task_final_performances = {}
        # Stores the average training reward (with exploration) achieved during training of each task.
        self.task_training_rewards = {}
        # Stores the evaluation reward of previously learned tasks *before* starting training on a new task.
        # Used to measure forgetting.
        self.pre_train_performances = {}
        # Stores the initial baseline evaluation reward for all tasks, measured before any training.
        self.initial_performances = {}
        # Stores deep copies of agent states (weights, optimizer state, etc.) after training each task, in memory.
        self.task_agent_states = {}
        
        # Environment type for file naming and specific reward handling (e.g., MountainCar).
        self.env_type = type(env).__name__.lower()
        # Flag to negate forgetting calculations for environments where higher negative reward is better (e.g., MountainCar).
        self.negate_forgetting = 'mountaincar' in self.env_type  # MountainCar needs sign flip
        
        # Ensure the directory for saving models exists.
        os.makedirs("models", exist_ok=True)
        
    def train_single_task(self, task_id, episodes=None):
        """
        Trains the agent on a single specified task for a given number of episodes.
        Includes adaptive epsilon initialization and decay, especially for sparse-reward
        environments like MountainCar.

        Args:
            task_id (int): The ID of the task to train on.
            episodes (int, optional): The number of episodes to train for.
                                      Defaults to `self.config.EPISODES_PER_TASK`.

        Returns:
            float: The mean reward achieved over all episodes trained for this task.
        """
        if episodes is None:
            episodes = self.config.EPISODES_PER_TASK

        print(f"\nTraining task {task_id}")
        self.env.change_task(task_id)

        # === Adaptive epsilon initialization ===
        # For sparse-reward environments (e.g., MountainCar), start with full exploration
        # and use a specific decay schedule to maintain exploration longer.
        if "mountaincar" in self.env_type.lower():
            self.agent.epsilon = self.config.EPSILON_START
            epsilon_decay = getattr(self.config, "EPSILON_DECAY_RATE", 0.998)  # Use decay rate from config
            epsilon_min = self.config.EPSILON_END
            print(f"Detected MountainCar environment → starting epsilon = {self.agent.epsilon:.2f}, decay rate = {epsilon_decay:.3f}")
        else:
            # For other environments, use standard configuration defaults.
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
                # Select an action using the agent's policy (epsilon-greedy).
                action = self.agent.select_action(state)

                # Execute the action in the environment.
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store the experience (s, a, r, s', done) in the agent's replay buffer.
                self.agent.push_memory(state, action, reward, next_state, done)

                # Update the agent's network (e.g., DQN update).
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

            # Calculate average loss for the episode.
            avg_loss = episode_loss / max(update_count, 1)

            # Record episode metrics for later analysis.
            self.metrics.record_episode(
                episode_reward, episode_length, avg_loss,
                getattr(self.agent, 'epsilon', 0.0), task_id
            )

            task_rewards.append(episode_reward)
            self.episode_count += 1

            # === Adaptive epsilon decay ===
            # Gradually reduce the exploration rate.
            if "mountaincar" in self.env_type.lower():
                # MountainCar uses a three-stage decay to balance exploration vs. exploitation
                # due to its sparse reward nature.
                if episode < 200:
                    decay_rate = 0.9995  # Very slow decay: maintains high exploration.
                elif episode < 400:
                    decay_rate = 0.995  # Medium decay: gradual decrease in exploration.
                else:
                    decay_rate = 0.98  # Faster decay: encourages exploitation of learned policy.
            else:
                decay_rate = epsilon_decay # Use config-defined decay rate for other environments.
            
            self.agent.epsilon = max(epsilon_min, self.agent.epsilon * decay_rate)

            # Optional: Monitor performance (e.g., for MountainCar) but do not reverse epsilon decay.
            if "mountaincar" in self.env_type.lower() and np.mean(task_rewards[-10:]) < -180:
                pass  # Placeholder for potential logging if needed, without altering epsilon.

            # Print training progress at specified intervals.
            if (episode + 1) % self.config.LOG_INTERVAL == 0:
                recent_rewards = task_rewards[-self.config.LOG_INTERVAL:]
                avg_reward = np.mean(recent_rewards)
                print(f"Task {task_id} | Episode {episode + 1}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"ε: {self.agent.epsilon:.3f}")

        # Calculate and print the mean reward over all episodes for the completed task.
        mean_task_reward = np.mean(task_rewards) if len(task_rewards) > 0 else 0.0
        print(f"Task {task_id} completed - {episodes}-episode average reward: {mean_task_reward:.2f}")

        return mean_task_reward
    
    def save_task_model(self, task_id):
        """
        Saves the agent's current model state (network weights, optimizer, epsilon, etc.)
        to disk and also stores a deep copy in memory for quick retrieval.

        Args:
            task_id (int): The ID of the task for which the model is being saved.
        """
        # Use environment type to create a distinct file path (e.g., "cartpole_task_0_model.pth").
        env_prefix = self.env_type.replace('cl', '')  # Remove 'cl' suffix from environment name.
        model_path = f"models/{env_prefix}_task_{task_id}_model.pth"
        
        # Create a checkpoint dictionary containing essential agent components.
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
        
        # Also save an in-memory state for fast restoration during evaluation phases.
        self.save_agent_state(task_id)
    
    def load_task_model(self, task_id):
        """
        Loads an agent's model state for a specific task from disk and restores it
        to the current agent instance. It also stores this loaded state in memory.

        Args:
            task_id (int): The ID of the task whose model should be loaded.
        """
        # Construct the model file path.
        env_prefix = self.env_type.replace('cl', '')
        model_path = f"models/{env_prefix}_task_{task_id}_model.pth"
        
        if os.path.exists(model_path):
            # Load the checkpoint, mapping to CPU to avoid device issues if saved on GPU.
            checkpoint = torch.load(model_path, map_location='cpu')
            self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict']) # Corrected line
            try:
                # Attempt to load optimizer state, but handle potential incompatibility.
                self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                print("Warning: could not fully restore optimizer state.")
            # Restore epsilon and steps_done, providing defaults if not present in checkpoint.
            self.agent.epsilon = checkpoint.get('epsilon', getattr(self.agent, 'epsilon', 0.0))
            self.agent.steps_done = checkpoint.get('steps_done', getattr(self.agent, 'steps_done', 0))
            # Also store to in-memory for quicker later use.
            self.save_agent_state(task_id)
            print(f"Task {task_id} model loaded from: {model_path}")
        else:
            print(f"Warning: Model file not found: {model_path}")
    
    def _capture_agent_state(self):
        """
        Captures the current state of the agent (network weights, optimizer state, etc.)
        Returns a dictionary containing deep copies of these components.

        Returns:
            dict: A dictionary representing the agent's current state, or None if an error occurs.
        """
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
        """
        Restores the agent's state from a provided state dictionary.

        Args:
            state (dict): The state dictionary to restore.
            context (str): A string indicating the source of the state (for logging purposes).
        """
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
        """
        Captures the current agent's state and stores a deep copy of it in an in-memory dictionary,
        associated with the given task_id. This is useful for quickly restoring agent states
        without disk I/O.

        Args:
            task_id (int): The ID of the task for which the current agent state is being saved.
        """
        state = self._capture_agent_state()
        if state is not None:
            self.task_agent_states[task_id] = state
    
    def load_agent_state(self, task_id):
        """
        Loads a previously saved agent state from the in-memory dictionary and restores it
        to the current agent instance.

        Args:
            task_id (int): The ID of the task whose saved agent state should be loaded.
        """
        if task_id in self.task_agent_states:
            self._restore_agent_state(self.task_agent_states[task_id], "memory")
        else:
            print(f"Warning: No saved in-memory state found for task {task_id}")
    
    def _snapshot_current_agent_state(self):
        """
        Internal utility to create a temporary deep copy snapshot of the current agent's state.
        This is useful when needing to temporarily switch the agent's state (e.g., for evaluation)
        and then revert to the original state.

        Returns:
            dict: A deep copy of the agent's current state.
        """
        return self._capture_agent_state()
    
    def _restore_agent_state_snapshot(self, snapshot):
        """
        Internal utility to restore the agent's state from a previously captured snapshot.

        Args:
            snapshot (dict): The agent state snapshot to restore.
        """
        self._restore_agent_state(snapshot, "snapshot")
    
    def evaluate_task(self, task_id, episodes=50, verbose=True, num_evaluations=3):
        """
        Evaluates the agent's performance on a specific task.
        The evaluation runs over multiple episodes and multiple evaluation rounds
        to provide a more stable and reliable performance metric. Exploration is disabled
        during evaluation (agent acts greedily).

        Args:
            task_id (int): The ID of the task to evaluate.
            episodes (int): The number of episodes to run for each evaluation round.
            verbose (bool): If True, prints detailed evaluation results.
            num_evaluations (int): The number of times to repeat the evaluation process.

        Returns:
            float: The mean reward across all evaluation episodes and rounds.
        """
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
        """
        Executes the complete continual learning training loop across all defined tasks.
        This includes:
        1. Initial baseline performance evaluation on all tasks.
        2. Sequential training on each task.
        3. Evaluating previous tasks' retention before and after training a new task to measure forgetting.
        4. Robust agent state management using in-memory and disk storage.

        Returns:
            dict: A dictionary containing the final performance of the agent on each task.
        """
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
        """
        Generates a summary of the entire continual learning training run.

        Returns:
            dict: A dictionary containing overall training statistics and recorded metrics.
        """
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
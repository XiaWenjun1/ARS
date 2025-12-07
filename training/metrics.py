import numpy as np
import pandas as pd
from collections import defaultdict

class TrainingMetrics:
    """
    A comprehensive tracker for various training and continual learning metrics.
    It records episode-level data (rewards, lengths, losses, epsilon) and
    calculates task-specific performance, convergence speed, and catastrophic forgetting.
    """
    
    def __init__(self):
        """
        Initializes the TrainingMetrics tracker, resetting all internal data structures.
        """
        self.reset()
    
    def reset(self):
        """
        Resets all stored metrics and data. This is typically called at the start
        of a new training run or experiment.
        """
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilons = []
        
        # Continual learning specific metrics
        # Stores episode rewards per task, used for calculating convergence speed and task performance.
        self.task_performance = defaultdict(list)
        # Stores detailed information about catastrophic forgetting for each task pair.
        self.catastrophic_forgetting = {}
    
    def record_episode(self, episode_reward, episode_length, loss, epsilon, task_id=None):
        """
        Records the metrics for a single training episode.

        Args:
            episode_reward (float): The total reward received during the episode.
            episode_length (int): The number of steps taken in the episode.
            loss (float): The average loss incurred during the episode's training updates.
            epsilon (float): The epsilon value used for exploration during the episode.
            task_id (int, optional): The ID of the task this episode belongs to. If provided,
                                     the episode reward is also recorded under task_performance.
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        
        if task_id is not None:
            self.task_performance[task_id].append(episode_reward)
    
    def record_task_performance(self, task_key, performance_before, performance_after, negate_values=False):
        """
        Records the change in performance for a specific task to calculate catastrophic forgetting.

        Args:
            task_key (str): A unique identifier for this forgetting measurement,
                            typically formatted as "prev_task_after_new_task".
            performance_before (float): The agent's performance on `prev_task` before training `new_task`.
            performance_after (float): The agent's performance on `prev_task` after training `new_task`.
            negate_values (bool): If True, negates the performance values before calculating
                                  forgetting. Useful for environments where lower scores are better
                                  (e.g., MountainCar, where -100 is better than -200), ensuring
                                  that a performance 'drop' is correctly interpreted as positive forgetting.
        """
        if negate_values:
            performance_before = -performance_before
            performance_after = -performance_after

        self.catastrophic_forgetting[task_key] = {
            'before': performance_before,
            'after': performance_after,
            'forgetting': performance_before - performance_after
        }
    
    def get_current_stats(self, window=100):
        """
        Calculates and returns current training statistics based on a sliding window of episodes.

        Args:
            window (int): The number of most recent episodes to consider for statistics calculation.

        Returns:
            dict: A dictionary containing mean reward, standard deviation of reward,
                  mean episode length, and total number of episodes. Returns an empty
                  dictionary if no episodes have been recorded.
        """
        if len(self.episode_rewards) == 0:
            return {}
            
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_length': np.mean(recent_lengths),
            'total_episodes': len(self.episode_rewards)
        }
    
    def calculate_convergence_speed(self, threshold, window=10):
        """
        Calculates the convergence speed for each task, defined as the number of episodes
        required to reach and maintain a performance above a specified threshold.

        Args:
            threshold (float): The performance threshold that indicates convergence.
            window (int): The number of consecutive episodes whose average reward must
                          exceed the threshold for convergence to be declared.

        Returns:
            dict: A dictionary where keys are task IDs and values are the number of episodes
                  to convergence, or the total episodes if convergence was not met.
        """
        convergence_data = {}
        
        for task_id, rewards in self.task_performance.items():
            if len(rewards) < window:
                convergence_data[task_id] = len(rewards)
                continue
            
            for i in range(len(rewards) - window + 1):
                window_rewards = rewards[i:i + window]
                if np.mean(window_rewards) >= threshold:
                    convergence_data[task_id] = i + window
                    break
            else:
                # If threshold not reached, record total episodes
                convergence_data[task_id] = len(rewards)
        
        return convergence_data
    
    def calculate_forgetting_matrix(self, total_tasks):
        """
        Calculates the catastrophic forgetting matrix. Each element (i, j) in the matrix
        represents the forgetting experienced on task 'i' after training on task 'j'.
        Forgetting is defined as `performance_before_task_j - performance_after_task_j_trained`.
        Negative values indicate performance improvement.

        Args:
            total_tasks (int): The total number of tasks in the continual learning sequence.

        Returns:
            tuple: A tuple containing:
                   - cf_matrix (np.ndarray): A 2D NumPy array representing the forgetting matrix.
                                            `cf_matrix[i, j]` is forgetting for task `i` after task `j`.
                   - avg_forgetting (float): The average positive forgetting observed across all tasks.
        """
        cf_matrix = np.zeros((total_tasks, total_tasks))
        total_forgetting = 0
        forgetting_count = 0
        
        # Calculate forgetting matrix
        for key, data in self.catastrophic_forgetting.items():
            if 'forgetting' in data:
                forgetting = data['forgetting']
                
                try:
                    # Handle string key format: "X_after_Y" or "TX_after_TY"
                    if isinstance(key, str) and '_after_' in key:
                        parts = key.split('_after_')
                        if len(parts) == 2:
                            # Handle possible "T0" format or direct numbers
                            prev_task_str = parts[0].replace('T', '')  # Remove possible 'T' prefix
                            current_task_str = parts[1].replace('T', '')  # Remove possible 'T' prefix
                            
                            prev_task = int(prev_task_str)
                            current_task = int(current_task_str)
                            
                            if prev_task < total_tasks and current_task < total_tasks:
                                cf_matrix[prev_task, current_task] = forgetting
                                
                                if forgetting > 0:  # Only sum positive forgetting values for average.
                                    total_forgetting += forgetting
                                    forgetting_count += 1
                    
                    # Handle tuple key format: (prev_task, current_task)
                    elif isinstance(key, tuple) and len(key) == 2:
                        prev_task, current_task = key
                        if prev_task < total_tasks and current_task < total_tasks:
                            cf_matrix[prev_task, current_task] = forgetting
                            
                            if forgetting > 0: # Only sum positive forgetting values for average.
                                total_forgetting += forgetting
                                forgetting_count += 1
                                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse forgetting key '{key}': {e}")
                    continue
        
        # Calculate average forgetting degree (only consider positive forgetting values).
        avg_forgetting = total_forgetting / forgetting_count if forgetting_count > 0 else 0
        
        return cf_matrix, avg_forgetting
    
    def get_core_metrics_summary(self, total_tasks, convergence_threshold):
        """
        Compiles a summary of core continual learning metrics, including
        convergence speed and catastrophic forgetting.

        Args:
            total_tasks (int): The total number of tasks in the continual learning sequence.
            convergence_threshold (float): The performance threshold used to determine task convergence.

        Returns:
            dict: A dictionary containing:
                  - 'convergence_data': Dictionary of episodes to convergence per task.
                  - 'cf_matrix': The catastrophic forgetting matrix.
                  - 'avg_convergence': The average convergence speed across all tasks.
                  - 'avg_forgetting': The average positive catastrophic forgetting.
        """
        # Calculate convergence speed and forgetting matrix
        convergence_data = self.calculate_convergence_speed(convergence_threshold)
        cf_matrix, avg_forgetting = self.calculate_forgetting_matrix(total_tasks)
        
        # Calculate average convergence speed
        avg_convergence = np.mean(list(convergence_data.values())) if convergence_data else 0
        
        return {
            'convergence_data': convergence_data,
            'cf_matrix': cf_matrix,
            'avg_convergence': avg_convergence,
            'avg_forgetting': avg_forgetting
        }
    
    def get_task_performance_summary(self):
        """
        Generates a summary of the agent's performance on each individual task.

        Returns:
            dict: A dictionary where keys are task IDs and values are dictionaries
                  containing mean, standard deviation, max, min reward, and total episodes
                  for that task.
        """
        summary = {}
        for task_id, rewards in self.task_performance.items():
            if rewards:
                summary[task_id] = {
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'max_reward': np.max(rewards),
                    'min_reward': np.min(rewards),
                    'episodes': len(rewards)
                }
        return summary
    
    def save_to_csv(self, filename):
        """
        Saves the episode-level training metrics (rewards, lengths, losses, epsilon)
        to a CSV file.

        Args:
            filename (str): The path and name of the CSV file to save.
        """
        df = pd.DataFrame({
            'episode': range(len(self.episode_rewards)),
            'reward': self.episode_rewards,
            'length': self.episode_lengths,
            'loss': self.losses,
            'epsilon': self.epsilons
        })
        df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")
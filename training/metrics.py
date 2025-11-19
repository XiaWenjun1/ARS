import numpy as np
import pandas as pd
from collections import defaultdict

class TrainingMetrics:
    """Training Metrics Tracker"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilons = []
        
        # Continual learning specific metrics
        self.task_performance = defaultdict(list)
        self.catastrophic_forgetting = {}
    
    def record_episode(self, episode_reward, episode_length, loss, epsilon, task_id=None):
        """Record metrics for one episode"""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.losses.append(loss)
        self.epsilons.append(epsilon)
        
        if task_id is not None:
            self.task_performance[task_id].append(episode_reward)
    
    def record_task_performance(self, task_key, performance_before, performance_after):
        """Record task performance change (for calculating catastrophic forgetting)
        
        Args:
            task_key: Task identifier, format "prev_task_after_new_task" or using tuple (prev_task, new_task)
            performance_before: Performance before training new task
            performance_after: Performance after training new task
        """
        self.catastrophic_forgetting[task_key] = {
            'before': performance_before,
            'after': performance_after,
            'forgetting': performance_before - performance_after
        }
    
    def get_current_stats(self, window=100):
        """Get current statistics (sliding window)"""
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
    
    # New: Core metric calculation methods required for the assignment
    
    def calculate_convergence_speed(self, threshold, window=10):
        """Calculate convergence speed - number of episodes required to reach stable performance
        
        Args:
            threshold: Convergence threshold (must be provided, typically from config.CONVERGENCE_THRESHOLD)
            window: Window size for checking convergence (default: 10)
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
        """Calculate catastrophic forgetting matrix (allows negative values)"""
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
                            prev_task_str = parts[0].replace('T', '')  # Remove possible 'T'
                            current_task_str = parts[1].replace('T', '')  # Remove possible 'T'
                            
                            prev_task = int(prev_task_str)
                            current_task = int(current_task_str)
                            
                            if prev_task < total_tasks and current_task < total_tasks:
                                # Allow negative values (performance improvement)
                                cf_matrix[prev_task, current_task] = forgetting
                                
                                if forgetting > 0:  # Only calculate positive forgetting values
                                    total_forgetting += forgetting
                                    forgetting_count += 1
                    
                    # Handle tuple key format: (prev_task, current_task)
                    elif isinstance(key, tuple) and len(key) == 2:
                        prev_task, current_task = key
                        if prev_task < total_tasks and current_task < total_tasks:
                            cf_matrix[prev_task, current_task] = forgetting
                            
                            if forgetting > 0:
                                total_forgetting += forgetting
                                forgetting_count += 1
                                
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse key '{key}': {e}")
                    continue
        
        # Calculate average forgetting degree (only consider positive forgetting values)
        avg_forgetting = total_forgetting / forgetting_count if forgetting_count > 0 else 0
        
        return cf_matrix, avg_forgetting
    
    def get_core_metrics_summary(self, total_tasks, convergence_threshold):
        """Get core metrics summary required for the assignment
        
        Args:
            total_tasks: Total number of tasks
            convergence_threshold: Convergence threshold (must be provided from config.CONVERGENCE_THRESHOLD)
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
        """Get performance summary for each task"""
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
        """Save metrics to CSV file"""
        df = pd.DataFrame({
            'episode': range(len(self.episode_rewards)),
            'reward': self.episode_rewards,
            'length': self.episode_lengths,
            'loss': self.losses,
            'epsilon': self.epsilons
        })
        df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")
#!/usr/bin/env python3
"""
Model demonstration script for both CartPole and MountainCar
"""

import sys
import os
import time
import numpy as np

# Add project root to python search path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def demo_cartpole():
    """CartPole model demonstration - shows all task models"""
    try:
        from configs import CartPoleConfig
        from environments import CartPoleCL
        from agents import DQNAgent
        
        # Setup environment with rendering
        config = CartPoleConfig()
        env = CartPoleCL(config.TASKS, render_mode="human")
        
        # Create agent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config
        )
        
        # Demo all task models (0 to 3)
        for task_id in range(4):
            model_path = f"models/cartpole_task_{task_id}_model.pth"
            
            if not os.path.exists(model_path):
                print(f"Task {task_id} model not found: {model_path}")
                continue
            
            print(f"\n{'='*50}")
            print(f"DEMONSTRATING CART POLE TASK {task_id}")
            print(f"{'='*50}")
            
            # Load task model
            agent.load(model_path)
            
            # Set environment to specific task
            env.change_task(task_id)
            
            # Run single episode
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(500):
                # Get action from trained model (no exploration)
                action = agent.select_action(state, training=False)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Render the environment (fast)
                env.render()
                time.sleep(0.01)  # Fast rendering
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            print(f"Task {task_id} Episode completed:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length} steps")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            
            # Pause between tasks
            input("\nPress Enter to continue to next task...")
        
        env.close()
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        try:
            env.close()
        except:
            pass
    except Exception as e:
        print(f"\nError during CartPole demo: {e}")
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except:
            pass

def demo_mountaincar():
    """MountainCar model demonstration - shows all task models"""
    try:
        from configs import MountainCarConfig
        from environments import MountainCarCL
        from agents import DQNAgent
        
        # Setup environment with rendering
        config = MountainCarConfig()
        env = MountainCarCL(config.TASKS, render_mode="human")
        
        # Create agent
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=config
        )
        
        # Demo all task models (0 to 3)
        for task_id in range(4):
            model_path = f"models/mountaincar_task_{task_id}_model.pth"
            
            if not os.path.exists(model_path):
                print(f"Task {task_id} model not found: {model_path}")
                continue
            
            print(f"\n{'='*50}")
            print(f"DEMONSTRATING MOUNTAIN CAR TASK {task_id}")
            print(f"{'='*50}")
            
            # Load task model
            agent.load(model_path)
            
            # Set environment to specific task
            env.change_task(task_id)
            
            # Run single episode
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(500):
                # Get action from trained model (no exploration)
                action = agent.select_action(state, training=False)
                
                # Execute action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Render the environment (fast)
                env.render()
                time.sleep(0.01)  # Fast rendering
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            print(f"Task {task_id} Episode completed:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length} steps")
            print(f"  Position: {state[0]:.3f}")
            print(f"  Velocity: {state[1]:.3f}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            
            # Pause between tasks
            input("\nPress Enter to continue to next task...")
        
        env.close()
        
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
        try:
            env.close()
        except:
            pass
    except Exception as e:
        print(f"\nError during MountainCar demo: {e}")
        import traceback
        traceback.print_exc()
        try:
            env.close()
        except:
            pass

def main():
    print("=== Model Demonstration ===")
    print("Choose which environment to demonstrate:")
    print("1. CartPole (Tasks 0-3: pole balancing)")
    print("2. MountainCar (Tasks 0-3: hill climbing)")
    print()
    
    # Check available task models
    cartpole_models = []
    mountaincar_models = []
    
    for i in range(4):  # CartPole has tasks 0-3
        if os.path.exists(f"models/cartpole_task_{i}_model.pth"):
            cartpole_models.append(i)
    
    for i in range(4):  # MountainCar has tasks 0-3
        if os.path.exists(f"models/mountaincar_task_{i}_model.pth"):
            mountaincar_models.append(i)
    
    if not cartpole_models and not mountaincar_models:
        print("No trained task models found!")
        print("Please run training first:")
        print("  python experiments/train_cartpole.py")
        print("  python experiments/train_mountaincar.py")
        return
    
    print("Available task models:")
    if cartpole_models:
        print(f"  [OK] CartPole: Tasks {cartpole_models}")
    if mountaincar_models:
        print(f"  [OK] MountainCar: Tasks {mountaincar_models}")
    print()
    
    # Simple choice mechanism
    choice = input("Enter your choice (1 for CartPole, 2 for MountainCar): ").strip()
    
    if choice == "1" and cartpole_models:
        demo_cartpole()
    elif choice == "2" and mountaincar_models:
        demo_mountaincar()
    elif choice == "1" and not cartpole_models:
        print("CartPole task models not found. Please train them first.")
    elif choice == "2" and not mountaincar_models:
        print("MountainCar task models not found. Please train them first.")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
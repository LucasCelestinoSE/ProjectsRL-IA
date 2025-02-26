import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import time
import os
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Import the classes from your original code
from treinmanento_mlp import MCTSMLPPong, RamPreprocessing  

def plot_rewards(episode_rewards, window_size=100):
    """Plot the rewards history with a moving average"""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.plot(range(window_size-1, len(episode_rewards)), moving_avg)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

def evaluate_agent(env, agent, num_episodes=10, render=False):
    """Evaluate the agent's performance"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(state, num_simulations=25)  # Reduced simulations for faster evaluation
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render and episode == 0:  # Render only first episode
                env.render()
                time.sleep(0.02)
            
            if truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}, Steps = {steps}")
    
    return np.mean(episode_rewards), np.std(episode_rewards)

def train_and_evaluate(load_pretrained=False, model_path="pong_mlp_dqn", training_timesteps=100000):
    """Main function to train and evaluate the agent"""
    # Create and wrap the environment
    env = gym.make("Pong-ram-v4")
    env = RamPreprocessing(env)
    
    # Create the agent
    agent = MCTSMLPPong(
        env=env,
        model_path=model_path if load_pretrained else None,
        exploration_weight=1.0
    )
    
    # Training phase
    if not load_pretrained:
        print("Starting DQN training...")
        agent.train_dqn(total_timesteps=training_timesteps)
        agent.save_dqn(model_path)
        print("DQN training completed and model saved.")
    
    # Evaluation phase
    print("\nStarting evaluation...")
    rewards_history = []
    eval_episodes = 50  # Number of episodes to run
    
    for episode in range(eval_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(state)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if truncated:
                break
        
        rewards_history.append(episode_reward)
        
        # Print progress
        print(f"Episode {episode + 1}/{eval_episodes}")
        print(f"Reward: {episode_reward}")
        print(f"Steps: {steps}")
        print("-" * 30)
        
        # Plot rewards every 10 episodes
        if (episode + 1) % 10 == 0:
            plot_rewards(rewards_history)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    mean_reward, std_reward = evaluate_agent(env, agent, num_episodes=10, render=True)
    print(f"\nFinal Evaluation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Clean up
    env.close()
    
    return agent, rewards_history

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # You can either train a new model or load a pretrained one
    LOAD_PRETRAINED = False  # Set to True to load a pretrained model
    
    agent, rewards = train_and_evaluate(
        load_pretrained=LOAD_PRETRAINED,
        model_path="pong_mlp_dqn",
        training_timesteps=100000
    )
    
    # Plot final results
    plot_rewards(rewards)
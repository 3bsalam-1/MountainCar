"""Utility functions for MountainCar Q-Learning."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def discretize_state(state, pos_space, vel_space):
    """
    Discretize continuous state into bins.
    
    Args:
        state (tuple): Continuous state (position, velocity)
        pos_space (np.ndarray): Position bin edges
        vel_space (np.ndarray): Velocity bin edges
    
    Returns:
        tuple: Discretized state (position_bin, velocity_bin)
    """
    pos_bin = np.digitize(state[0], pos_space)
    vel_bin = np.digitize(state[1], vel_space)
    return (pos_bin, vel_bin)


def create_state_spaces(env, num_bins=20):
    """
    Create discretized state spaces for position and velocity.
    
    Args:
        env: Gymnasium environment
        num_bins (int): Number of bins for each dimension
    
    Returns:
        tuple: (pos_space, vel_space) arrays of bin edges
    """
    pos_space = np.linspace(
        env.observation_space.low[0],
        env.observation_space.high[0],
        num_bins
    )
    vel_space = np.linspace(
        env.observation_space.low[1],
        env.observation_space.high[1],
        num_bins
    )
    return pos_space, vel_space


def plot_training_progress(rewards_per_episode, save_path, window=100):
    """
    Plot training progress with rolling mean rewards.
    
    Args:
        rewards_per_episode (np.ndarray): Rewards for each episode
        save_path (str or Path): Path to save the plot
        window (int): Window size for rolling mean
    """
    episodes = len(rewards_per_episode)
    mean_rewards = np.zeros(episodes)
    
    for t in range(episodes):
        mean_rewards[t] = np.mean(
            rewards_per_episode[max(0, t - window):(t + 1)]
        )
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards, linewidth=2)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(f'Mean Reward (over {window} episodes)', fontsize=12)
    plt.title('MountainCar Q-Learning Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_episode_stats(episode, total_episodes, reward, epsilon, interval=100):
    """
    Print episode statistics during training.
    
    Args:
        episode (int): Current episode number
        total_episodes (int): Total number of episodes
        reward (float): Episode reward
        epsilon (float): Current epsilon value
        interval (int): Print interval
    """
    if (episode + 1) % interval == 0 or episode == 0:
        print(f"Episode {episode + 1}/{total_episodes} | "
              f"Reward: {reward:.2f} | "
              f"Epsilon: {epsilon:.4f}")

"""Evaluation module for MountainCar Q-Learning."""

import gymnasium as gym
import numpy as np
from pathlib import Path

from .agent import QLearningAgent
from .utils import create_state_spaces, discretize_state


class Evaluator:
    """
    Evaluator for trained Q-Learning agent on MountainCar environment.
    
    Attributes:
        env: Gymnasium environment
        agent (QLearningAgent): Q-Learning agent
        pos_space (np.ndarray): Position discretization bins
        vel_space (np.ndarray): Velocity discretization bins
    """
    
    def __init__(self, model_path, num_bins=20, render=True):
        """
        Initialize evaluator and load trained model.
        
        Args:
            model_path (str or Path): Path to trained Q-table
            num_bins (int): Number of bins for state discretization
            render (bool): Whether to render environment
        """
        self.env = gym.make(
            'MountainCar-v0',
            render_mode='human' if render else None
        )
        
        self.pos_space, self.vel_space = create_state_spaces(self.env, num_bins)
        
        self.agent = QLearningAgent(
            state_space_shape=(num_bins, num_bins),
            action_space_size=self.env.action_space.n
        )
        
        # Load trained Q-table
        self.agent.load(model_path)
        print(f"Loaded Q-table from: {model_path}")
    
    def evaluate(self, episodes=10, max_steps=1000):
        """
        Evaluate the trained agent.
        
        Args:
            episodes (int): Number of evaluation episodes
            max_steps (int): Maximum steps per episode
        
        Returns:
            dict: Evaluation metrics (mean_reward, std_reward, success_rate)
        """
        rewards = []
        successes = 0
        
        print(f"\nEvaluating agent for {episodes} episodes...")
        print("-" * 60)
        
        for episode in range(episodes):
            state = self.env.reset()[0]
            state = discretize_state(state, self.pos_space, self.vel_space)
            
            episode_reward = 0
            terminated = False
            
            for step in range(max_steps):
                # Select action (no exploration)
                action = self.agent.select_action(state, training=False)
                next_state_raw, reward, terminated, _, _ = self.env.step(action)
                next_state = discretize_state(
                    next_state_raw,
                    self.pos_space,
                    self.vel_space
                )
                
                state = next_state
                episode_reward += reward
                
                if terminated:
                    successes += 1
                    break
            
            rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Steps: {step + 1} | "
                  f"Success: {'✓' if terminated else '✗'}")
        
        self.env.close()
        
        # Calculate metrics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        success_rate = (successes / episodes) * 100
        
        print("-" * 60)
        print(f"Evaluation Results:")
        print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  Success Rate: {success_rate:.1f}% ({successes}/{episodes})")
        print("-" * 60)
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'success_rate': success_rate,
            'rewards': rewards
        }

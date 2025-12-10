"""Training module for MountainCar Q-Learning."""

import gymnasium as gym
import numpy as np
from pathlib import Path

from .agent import QLearningAgent
from .utils import create_state_spaces, discretize_state, plot_training_progress, print_episode_stats


class Trainer:
    """
    Trainer for Q-Learning agent on MountainCar environment.
    
    Attributes:
        env: Gymnasium environment
        agent (QLearningAgent): Q-Learning agent
        pos_space (np.ndarray): Position discretization bins
        vel_space (np.ndarray): Velocity discretization bins
    """
    
    def __init__(
        self,
        num_bins=20,
        learning_rate=0.9,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=None,
        render=False
    ):
        """
        Initialize trainer.
        
        Args:
            num_bins (int): Number of bins for state discretization
            learning_rate (float): Learning rate (alpha)
            discount_factor (float): Discount factor (gamma)
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate (auto-calculated if None)
            render (bool): Whether to render environment
        """
        self.env = gym.make(
            'MountainCar-v0',
            render_mode='human' if render else None
        )
        
        self.pos_space, self.vel_space = create_state_spaces(self.env, num_bins)
        
        # Auto-calculate epsilon decay if not provided
        if epsilon_decay is None:
            epsilon_decay = 0.0004  # Will be updated based on episodes
        
        self.agent = QLearningAgent(
            state_space_shape=(num_bins, num_bins),
            action_space_size=self.env.action_space.n,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay
        )
    
    def train(
        self,
        episodes=5000,
        max_steps=1000,
        save_path='models/mountain_car.pkl',
        plot_path='outputs/plots/training_progress.png',
        print_interval=100
    ):
        """
        Train the Q-Learning agent.
        
        Args:
            episodes (int): Number of training episodes
            max_steps (int): Maximum steps per episode
            save_path (str): Path to save trained Q-table
            plot_path (str): Path to save training plot
            print_interval (int): Episode interval for printing stats
        
        Returns:
            np.ndarray: Rewards per episode
        """
        # Update epsilon decay based on episodes
        self.agent.epsilon_decay = 2 / episodes
        
        rewards_per_episode = np.zeros(episodes)
        
        print(f"Starting training for {episodes} episodes...")
        print(f"Learning Rate: {self.agent.learning_rate}")
        print(f"Discount Factor: {self.agent.discount_factor}")
        print(f"Initial Epsilon: {self.agent.epsilon}")
        print("-" * 60)
        
        for episode in range(episodes):
            state = self.env.reset()[0]
            state = discretize_state(state, self.pos_space, self.vel_space)
            
            episode_reward = 0
            terminated = False
            
            for step in range(max_steps):
                # Select and execute action
                action = self.agent.select_action(state, training=True)
                next_state_raw, reward, terminated, _, _ = self.env.step(action)
                next_state = discretize_state(
                    next_state_raw,
                    self.pos_space,
                    self.vel_space
                )
                
                # Update Q-table
                self.agent.update(state, action, reward, next_state)
                
                state = next_state
                episode_reward += reward
                
                if terminated:
                    break
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            rewards_per_episode[episode] = episode_reward
            
            # Print progress
            print_episode_stats(
                episode,
                episodes,
                episode_reward,
                self.agent.epsilon,
                print_interval
            )
        
        print("-" * 60)
        print("Training completed!")
        
        # Save Q-table
        self.agent.save(save_path)
        print(f"Q-table saved to: {save_path}")
        
        # Plot training progress
        plot_training_progress(rewards_per_episode, plot_path)
        print(f"Training plot saved to: {plot_path}")
        
        self.env.close()
        
        return rewards_per_episode

"""Q-Learning Agent for MountainCar environment."""

import numpy as np
import pickle
from pathlib import Path


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy policy.
    
    Attributes:
        q_table (np.ndarray): Q-value table for state-action pairs
        learning_rate (float): Learning rate (alpha)
        discount_factor (float): Discount factor (gamma)
        epsilon (float): Exploration rate
        epsilon_decay (float): Epsilon decay rate per episode
        min_epsilon (float): Minimum epsilon value
    """
    
    def __init__(
        self,
        state_space_shape,
        action_space_size,
        learning_rate=0.9,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.0004,
        min_epsilon=0.0
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_space_shape (tuple): Shape of discretized state space (pos_bins, vel_bins)
            action_space_size (int): Number of possible actions
            learning_rate (float): Learning rate (alpha)
            discount_factor (float): Discount factor (gamma)
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate per episode
            min_epsilon (float): Minimum epsilon value
        """
        self.q_table = np.zeros((*state_space_shape, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng()
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (tuple): Discretized state (position_bin, velocity_bin)
            training (bool): Whether in training mode (exploration enabled)
        
        Returns:
            int: Selected action
        """
        if training and self.rng.random() < self.epsilon:
            # Explore: random action
            return self.rng.integers(0, self.q_table.shape[2])
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state[0], state[1], :])
    
    def update(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        
        Args:
            state (tuple): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (tuple): Next state
        """
        current_q = self.q_table[state[0], state[1], action]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1], :])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state[0], state[1], action] = new_q
    
    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)
    
    def save(self, filepath):
        """
        Save Q-table to file.
        
        Args:
            filepath (str or Path): Path to save Q-table
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load(self, filepath):
        """
        Load Q-table from file.
        
        Args:
            filepath (str or Path): Path to load Q-table from
        """
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)

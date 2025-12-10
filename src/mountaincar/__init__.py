"""MountainCar Q-Learning Package."""

from .agent import QLearningAgent
from .trainer import Trainer
from .evaluator import Evaluator
from .utils import discretize_state, create_state_spaces, plot_training_progress

__version__ = '1.0.0'
__all__ = [
    'QLearningAgent',
    'Trainer',
    'Evaluator',
    'discretize_state',
    'create_state_spaces',
    'plot_training_progress'
]

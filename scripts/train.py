"""Training script for MountainCar Q-Learning."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mountaincar import Trainer


def main():
    parser = argparse.ArgumentParser(
        description='Train Q-Learning agent on MountainCar environment'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=5000,
        help='Number of training episodes (default: 5000)'
    )
    
    parser.add_argument(
        '--bins',
        type=int,
        default=20,
        help='Number of bins for state discretization (default: 20)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.9,
        help='Learning rate alpha (default: 0.9)'
    )
    
    parser.add_argument(
        '--discount-factor',
        type=float,
        default=0.9,
        help='Discount factor gamma (default: 0.9)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1.0,
        help='Initial exploration rate (default: 1.0)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during training'
    )
    
    parser.add_argument(
        '--no-render',
        dest='render',
        action='store_false',
        help='Do not render environment (default)'
    )
    
    parser.add_argument(
        '--save-path',
        type=str,
        default='models/mountain_car.pkl',
        help='Path to save trained Q-table (default: models/mountain_car.pkl)'
    )
    
    parser.add_argument(
        '--plot-path',
        type=str,
        default='outputs/plots/training_progress.png',
        help='Path to save training plot (default: outputs/plots/training_progress.png)'
    )
    
    parser.set_defaults(render=False)
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(
        num_bins=args.bins,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        render=args.render
    )
    
    # Train agent
    trainer.train(
        episodes=args.episodes,
        save_path=args.save_path,
        plot_path=args.plot_path
    )


if __name__ == '__main__':
    main()

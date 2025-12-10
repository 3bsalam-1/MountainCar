"""Evaluation script for MountainCar Q-Learning."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mountaincar import Evaluator


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained Q-Learning agent on MountainCar environment'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/mountain_car.pkl',
        help='Path to trained Q-table (default: models/mountain_car.pkl)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    
    parser.add_argument(
        '--bins',
        type=int,
        default=20,
        help='Number of bins for state discretization (default: 20)'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during evaluation (default)'
    )
    
    parser.add_argument(
        '--no-render',
        dest='render',
        action='store_false',
        help='Do not render environment'
    )
    
    parser.set_defaults(render=True)
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train a model first using: python scripts/train.py")
        sys.exit(1)
    
    # Create evaluator
    evaluator = Evaluator(
        model_path=args.model_path,
        num_bins=args.bins,
        render=args.render
    )
    
    # Evaluate agent
    evaluator.evaluate(episodes=args.episodes)


if __name__ == '__main__':
    main()

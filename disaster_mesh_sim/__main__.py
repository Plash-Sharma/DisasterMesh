"""
DisasterMesh — Main entry point.

Usage:
    python -m disaster_mesh_sim train [--config path]
    python -m disaster_mesh_sim evaluate [--config path]
    python -m disaster_mesh_sim generate-data [--config path]
    python -m disaster_mesh_sim plot
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="DisasterMesh — AI-Driven Bluetooth D2D Mesh Network"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train
    train_p = subparsers.add_parser("train", help="Train RL agent")
    train_p.add_argument("--config", type=str, default=None)

    # Evaluate
    eval_p = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    eval_p.add_argument("--config", type=str, default=None)

    # Generate data
    gen_p = subparsers.add_parser("generate-data", help="Generate synthetic dataset")
    gen_p.add_argument("--config", type=str, default=None)

    # Plot
    subparsers.add_parser("plot", help="Generate visualization plots")

    args = parser.parse_args()

    if args.command == "train":
        from disaster_mesh_sim.train import train
        train(config_path=args.config)

    elif args.command == "evaluate":
        from disaster_mesh_sim.evaluation.benchmark import run_benchmark
        run_benchmark(config_path=args.config)

    elif args.command == "generate-data":
        from disaster_mesh_sim.data.generator import generate_dataset
        generate_dataset()

    elif args.command == "plot":
        from disaster_mesh_sim.visualization.plotter import generate_all_plots
        import os
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results"
        )
        generate_all_plots(results_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

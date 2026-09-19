import sys
import os

from disaster_mesh_sim.evaluation.benchmark import run_benchmark
from disaster_mesh.visualization.benchmark_plots import generate_all_plots

if __name__ == "__main__":
    results = run_benchmark()
    generate_all_plots(results)
    print("Benchmark completed.")

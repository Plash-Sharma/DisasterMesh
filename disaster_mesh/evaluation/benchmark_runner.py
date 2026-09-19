"""
Benchmark runner wrapper.
Ref: Mega-Prompt Benchmark Runner
"""
from disaster_mesh_sim.evaluation.benchmark import run_benchmark

class BenchmarkRunner:
    def __init__(self, n_nodes=100, seeds=5, output_dir='results/'):
        self.n_nodes = n_nodes
        self.seeds = seeds
        self.output_dir = output_dir

    def run(self):
        return run_benchmark()

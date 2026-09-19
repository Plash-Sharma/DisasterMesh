"""
HMRFCO — Hybrid Manta-Ray Foraging + Chef-Based Optimization.

Implements joint relay selection and resource allocation from:
    Ramesh Babu & Nandakumar, "Towards Energy-Efficient Joint Relay Selection
    and Resource Allocation for D2D Communication Using Hybrid Heuristic-Based
    Deep Learning", Scientific Reports, Vol. 15, 2025.
    DOI: 10.1038/s41598-025-08290-x

Key Equations:
    Eq. 26: Hybrid switching rule (CBOA if j > crft/wrft, else MRFO)
    Eq. 46: Multi-objective: argmax{SE + EE + TP + NC + 1/DY}

Also includes AResGRU relay predictor (Eq. 41).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class NetworkState:
    """Network state for HMRFCO optimization."""
    num_nodes: int
    active_nodes: int
    avg_battery: float
    avg_reputation: float
    connectivity_ratio: float
    avg_throughput: float
    avg_delay: float
    spectral_efficiency: float


class HMRFCOOptimizer:
    """
    Hybrid Manta-Ray Foraging + Chef-Based Optimization.

    From Ramesh Babu & Nandakumar, Scientific Reports 2025:
    - HMRFCO Eq. 26: if j > crft/wrft → CBOA update, else → MRFO update
    - Multi-objective Eq. 46: maximize SE + EE + TP + NC + 1/DY
    - Converges 95% optimal within 180 iterations (13.35% faster than MRFO)
    """

    def __init__(self, population_size: int = 30, max_iterations: int = 250,
                 dim: int = 10):
        self.pop_size = population_size
        self.max_iter = max_iterations
        self.dim = dim

    def multi_objective_fitness(self, solution: np.ndarray,
                                 network_state: NetworkState) -> float:
        """
        HMRFCO Eq. 46: Multi-objective optimization.

        ob4 = argmax{SE + EE + TP + NC + 1/DY}
        """
        # Decode solution into tx_power and resource_block allocation
        tx_power = np.clip(solution[:self.dim // 2], 0.01, 1.0)
        resource_alloc = np.clip(solution[self.dim // 2:], 0.01, 1.0)

        # Spectral Efficiency (simplified)
        se = np.sum(np.log2(1 + tx_power * 10))  # Shannon-like
        # Energy Efficiency
        ee = np.sum(np.log2(1 + tx_power * 10)) / (np.sum(tx_power) + 1e-6)
        # Throughput (proportional to resource allocation)
        tp = np.sum(resource_alloc * np.log2(1 + tx_power * 10))
        # Network Capacity
        nc = network_state.active_nodes * np.mean(resource_alloc)
        # Delay (inverse)
        dy = max(network_state.avg_delay, 0.001)

        return se + ee + tp + nc + (1.0 / dy)

    def optimize(self, network_state: NetworkState) -> np.ndarray:
        """
        Run HMRFCO optimization.

        Eq. 26: Hybrid switching rule between CBOA and MRFO updates.

        Returns:
            Best solution (tx_power + resource allocation vector).
        """
        # Initialize population
        population = np.random.uniform(0.01, 1.0, (self.pop_size, self.dim))
        fitness = np.array([self.multi_objective_fitness(p, network_state)
                           for p in population])

        best_idx = np.argmax(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            worst_fitness = np.min(fitness)

            for m in range(self.pop_size):
                j = np.random.uniform(0, 1)
                crft = fitness[m]
                wrft = worst_fitness if worst_fitness > 0 else 1e-6

                if j > (crft / (wrft + 1e-6)):
                    # CBOA update (Chef-Based Optimization Algorithm)
                    population[m] = self._cboa_update(
                        population[m], best_solution, iteration
                    )
                else:
                    # MRFO update (Manta Ray Foraging Optimization)
                    population[m] = self._mrfo_update(
                        population[m], best_solution, population, iteration
                    )

                # Clip to bounds
                population[m] = np.clip(population[m], 0.01, 1.0)
                fitness[m] = self.multi_objective_fitness(population[m], network_state)

                if fitness[m] > best_fitness:
                    best_fitness = fitness[m]
                    best_solution = population[m].copy()

        return best_solution

    def _cboa_update(self, current: np.ndarray, best: np.ndarray,
                     iteration: int) -> np.ndarray:
        """CBOA: Chef-Based Optimization update."""
        r1 = np.random.uniform(0, 1, self.dim)
        r2 = np.random.uniform(0, 1)
        # Inspired by cooking process: mix with best solution
        new = current + r1 * (best - current) * r2
        return new

    def _mrfo_update(self, current: np.ndarray, best: np.ndarray,
                     population: np.ndarray, iteration: int) -> np.ndarray:
        """MRFO: Manta Ray Foraging Optimization update."""
        r = np.random.uniform(0, 1)
        # Chain foraging: follow best with spiral movement
        S = 2 * r * np.cos(2 * np.pi * r) * (best - current)
        new = current + S + r * (best - current)
        return new


class AResGRU(nn.Module):
    """
    Adaptive Residual GRU for relay count and resource prediction.

    From Ramesh Babu & Nandakumar, Scientific Reports 2025:
    Eq. 41: z_R = ReLU(BN(GRU_last + residual(y)))
    Predicts: [optimal_relay_count, resource_blocks]
    """

    def __init__(self, input_dim: int = 18, hidden_dim: int = 64,
                 output_dim: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.residual = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection (Eq. 41).

        Args:
            x: Input tensor [batch, seq_len, input_dim].

        Returns:
            Predictions [batch, 2] = [relay_count, resource_blocks].
        """
        gru_out, _ = self.gru(x)
        # Residual connection: Eq. 41
        residual_out = self.residual(x[:, -1, :])
        z = F.relu(self.bn(gru_out[:, -1, :] + residual_out))
        return self.output(z)

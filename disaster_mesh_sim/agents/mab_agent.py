"""
Multi-Armed Bandit (UCB1) Agent for relay selection.

Handles short-term exploration of relay candidates while Q-Learning
handles long-term exploitation.

Reference:
    Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem",
    Machine Learning, 2002
"""

import numpy as np
from typing import Dict, List, Optional
from disaster_mesh_sim.utils.config import MABConfig


class MABAgent:
    """
    UCB1-based Multi-Armed Bandit for relay node selection.

    UCB score: μ_i + sqrt(c · ln(t) / n_i)

    Each relay candidate is treated as an arm with unknown reward
    distribution.
    """

    def __init__(self, config: MABConfig):
        self.config = config
        self.pull_count: Dict[int, int] = {}
        self.total_reward: Dict[int, float] = {}
        self.total_pulls = 0

    def select_relay(self, candidates: List[int]) -> int:
        """
        Select relay using UCB1 strategy.

        Args:
            candidates: List of candidate relay node IDs.

        Returns:
            Selected relay node ID.
        """
        if not candidates:
            return -1

        self.total_pulls += 1
        t = self.total_pulls

        # Explore any untried arms first
        for c in candidates:
            if self.pull_count.get(c, 0) == 0:
                return c

        # UCB1 selection
        ucb_scores = []
        for c in candidates:
            n_i = self.pull_count[c]
            mu_i = self.total_reward[c] / n_i
            ucb = mu_i + np.sqrt(self.config.exploration_constant * np.log(t) / n_i)
            ucb_scores.append((c, ucb))

        return max(ucb_scores, key=lambda x: x[1])[0]

    def update(self, relay_id: int, reward: float):
        """Update arm statistics after observing reward."""
        self.pull_count[relay_id] = self.pull_count.get(relay_id, 0) + 1
        self.total_reward[relay_id] = self.total_reward.get(relay_id, 0.0) + reward

    def get_average_reward(self, relay_id: int) -> float:
        """Get average reward for a relay."""
        n = self.pull_count.get(relay_id, 0)
        if n == 0:
            return 0.0
        return self.total_reward.get(relay_id, 0.0) / n

    def reset(self):
        """Reset all arm statistics."""
        self.pull_count.clear()
        self.total_reward.clear()
        self.total_pulls = 0

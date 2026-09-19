"""
Tabular Q-Routing Agent for DisasterMesh.

Lightweight agent using a dictionary-based Q-table, suitable for
resource-constrained Relay and Leaf nodes.

Reference:
    Boyan & Littman, "Packet Routing in Dynamically Changing Networks:
    A Reinforcement Learning Approach", NeurIPS 1994
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from disaster_mesh_sim.utils.config import QRoutingConfig


class QRoutingAgent:
    """
    Tabular Q-Routing agent.

    Q(s, a) ← Q(s, a) + α [R + γ · max_a' Q(s', a') − Q(s, a)]

    State is discretized to (node_id, destination_id).
    Action is the neighbor id chosen for forwarding.
    """

    def __init__(self, config: QRoutingConfig):
        self.config = config
        self.q_table: Dict[Tuple, float] = {}
        self.epsilon = config.epsilon_start

    def _key(self, state: Tuple, action: int) -> Tuple:
        return (state, action)

    def get_q(self, state: Tuple, action: int) -> float:
        return self.q_table.get(self._key(state, action), 0.0)

    def select_action(self, state: Tuple, candidates: List[int]) -> int:
        """ε-greedy action selection from candidate set."""
        if not candidates:
            return -1

        if np.random.random() < self.epsilon:
            return int(np.random.choice(candidates))

        q_vals = {c: self.get_q(state, c) for c in candidates}
        return max(q_vals, key=q_vals.get)

    def update(self, state: Tuple, action: int, reward: float,
               next_state: Tuple, next_candidates: List[int]):
        """Standard Q-learning update rule."""
        old_q = self.get_q(state, action)

        if next_candidates:
            next_max_q = max(
                self.get_q(next_state, a) for a in next_candidates
            )
        else:
            next_max_q = 0.0

        new_q = old_q + self.config.learning_rate * (
            reward + self.config.discount_factor * next_max_q - old_q
        )
        self.q_table[self._key(state, action)] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )

    def get_table_size(self) -> int:
        """Number of entries in Q-table."""
        return len(self.q_table)

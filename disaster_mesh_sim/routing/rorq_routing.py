"""
RORQ — Reputation-Based Opportunistic Routing via Q-Learning.

Implements the core routing algorithm from:
    Ryu & Kim, "Reputation-Based Opportunistic Routing Protocol Using
    Q-Learning for MANET Attacked by Malicious Nodes",
    IEEE Access, Vol. 11, Feb 2023.
    DOI: 10.1109/ACCESS.2023.3242608

Key Equations:
    Eq. 4 & 5: Reputation update (proximity + speed scoring)
    Eq. 10: Q-value update (energy + reputation + buffer)
    Eq. 11: Visit-count adaptive epsilon decay
    Algorithm 1: CNODE candidate selection with isolation filter
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class RORQNodeState:
    """Per-node RORQ state visible to routing decisions."""
    node_id: int
    energy_level: float       # e_res / e_init [0,1] (Eq. 3)
    reputation: float          # Rep(u) [0,1]
    buffer_ratio: float        # B_idle / B_max [0,1]
    position: np.ndarray       # 3D position
    avg_speed: float           # EDMBOPR Eq. 5 computed speed
    neighbor_count: int        # number of 1-hop neighbors


class RORQRouter:
    """
    RORQ Q-Routing engine with reputation feedback.

    Maintains per-node Q-table and reputation table.
    Q-values are HIGHER = BETTER (unlike classic Q-routing where lower=better).

    References:
        Eq. 10: Q_x(d,y) ← (1-η)Q + η(e_level + Rep + B_idle/B_max)
        Eq. 4:  Rep(u) = α·Rep_old + (1-α)·Rep_new
        Eq. 5:  Rep_new = β·(D(u,v)/D(dest,v)) + (1-β)·(TTL/T_uv)
        Eq. 11: ε_t(u) = c / n_t(u)
    """

    def __init__(self, eta: float = 0.1, gamma: float = 0.9,
                 alpha_rep: float = 0.5, beta_rep: float = 0.5,
                 rep_init: float = 0.5, rep_threshold: float = 0.3,
                 c_constant: float = 1.0, packet_ttl: int = 10):
        # RORQ hyperparameters
        self.eta = eta                      # Q-learning rate (Eq. 10)
        self.gamma = gamma                  # Discount factor
        self.alpha_rep = alpha_rep          # Reputation EMA (Eq. 4)
        self.beta_rep = beta_rep            # Proximity vs speed (Eq. 5)
        self.rep_init = rep_init            # Initial reputation
        self.rep_threshold = rep_threshold  # Min rep for CNODE (Alg. 1)
        self.c_constant = c_constant       # Epsilon decay (Eq. 11)
        self.packet_ttl = packet_ttl

        # Per-node Q-tables: {(src_id, dest_id, relay_id): q_value}
        self.q_tables: Dict[int, Dict[Tuple, float]] = {}
        # Visit counts per node: {node_id: count} (for Eq. 11)
        self.visit_counts: Dict[int, int] = {}

    def get_q_value(self, src_id: int, dest_id: int, relay_id: int) -> float:
        """Get Q-value Q_x(d,y) for source x, destination d, relay y."""
        if src_id not in self.q_tables:
            self.q_tables[src_id] = {}
        return self.q_tables[src_id].get((dest_id, relay_id), 0.0)

    def update_q_table(self, src_id: int, dest_id: int, relay_id: int,
                       relay_state: RORQNodeState) -> float:
        """
        RORQ Eq. 10: Q-value update incorporating energy + reputation + buffer.

        Q_x(d,y) ← (1-η)Q_x(d,y) + η(e_level(y) + Rep(y) + B_idle(y)/B_max(y))

        Higher Q-value = better relay choice.
        """
        if src_id not in self.q_tables:
            self.q_tables[src_id] = {}

        key = (dest_id, relay_id)
        old_q = self.q_tables[src_id].get(key, 0.0)

        # RORQ Eq. 10 components
        e_level = relay_state.energy_level
        rep_y = relay_state.reputation
        buffer_ratio = relay_state.buffer_ratio

        new_q = (1 - self.eta) * old_q + self.eta * (e_level + rep_y + buffer_ratio)
        self.q_tables[src_id][key] = new_q
        return new_q

    def update_reputation(self, node_id: int, relay_state: RORQNodeState,
                          dest_state: RORQNodeState,
                          success: bool, T_uv: float,
                          current_reputation: float) -> float:
        """
        RORQ Eq. 4 & 5: Reputation update.

        Rep(u) = α·Rep_old(u) + (1-α)·Rep_new(u)
        Rep_new = β·(D(u,v)/D(dest,v)) + (1-β)·(TTL/T_uv)  [if success]
        Rep_new = 0                                            [if failure]

        Args:
            node_id: The forwarding node whose reputation is updated.
            relay_state: State of the relay node.
            dest_state: State of the destination node.
            success: Whether the packet was delivered.
            T_uv: Transmission time from u to v.
            current_reputation: Current reputation of the node.

        Returns:
            Updated reputation value.
        """
        if success:
            d_uv = np.linalg.norm(relay_state.position - dest_state.position)
            d_dest_v = np.linalg.norm(dest_state.position - relay_state.position)
            proximity_score = d_uv / max(d_dest_v, 1e-6)
            speed_score = self.packet_ttl / max(T_uv, 0.001)
            rep_new = self.beta_rep * proximity_score + (1 - self.beta_rep) * speed_score
            # Clamp to [0, 1]
            rep_new = min(1.0, max(0.0, rep_new))
        else:
            rep_new = 0.0  # RORQ: failure → zero rep_new

        updated = self.alpha_rep * current_reputation + (1 - self.alpha_rep) * rep_new
        return min(1.0, max(0.0, updated))

    def select_relay_epsilon_greedy(self, src_id: int, dest_id: int,
                                     cfl: List[int]) -> int:
        """
        RORQ Eq. 11: Visit-count adaptive epsilon-greedy relay selection.

        ε_t(u) = c / n_t(u) — decreases as node is visited more.

        Args:
            src_id: Source node ID.
            dest_id: Destination node ID.
            cfl: Candidate Forwarding List (node IDs).

        Returns:
            Selected relay node ID.
        """
        if not cfl:
            return -1

        # Update visit count
        self.visit_counts[src_id] = self.visit_counts.get(src_id, 0) + 1
        epsilon = self.c_constant / max(1, self.visit_counts[src_id])

        if np.random.random() < epsilon:
            return int(np.random.choice(cfl))  # Explore

        # Exploit: choose highest Q-value neighbor
        q_values = {n: self.get_q_value(src_id, dest_id, n) for n in cfl}
        return max(q_values, key=q_values.get)

    def filter_cnode(self, neighbors: Dict[int, RORQNodeState]) -> List[int]:
        """
        RORQ Algorithm 1: CNODE candidate selection.

        Filters out:
        - Nodes with reputation < rep_threshold
        - Isolated nodes (neighbor_count <= 1)

        Returns:
            List of qualifying neighbor IDs.
        """
        cnode = []
        for nid, state in neighbors.items():
            if state.reputation >= self.rep_threshold and state.neighbor_count > 1:
                cnode.append(nid)
        return cnode

    def get_exploration_rate(self, node_id: int) -> float:
        """Get current epsilon for a node (Eq. 11)."""
        count = self.visit_counts.get(node_id, 0)
        return self.c_constant / max(1, count)

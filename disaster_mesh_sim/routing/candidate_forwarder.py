"""
ExOR-style Candidate Forwarding with Q-Score weighting.

Novel contribution: replaces geographic ranking with Q-value + PROPHET
combined scoring for candidate relay selection.

References:
    [1] Biswas & Morris, "ExOR: Opportunistic Multi-Hop Routing", SIGCOMM 2005
    [2] Chachulski et al., "MORE: Trading Structure for Randomness", SIGCOMM 2007
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class CandidateForwarder:
    """
    Q-Score weighted candidate forwarding module.

    Ranks candidate relay nodes by a combined metric:
        combined = w_geo · geo_progress + w_q · q_score + w_prophet · prophet_score

    Higher-priority candidates suppress lower-priority duplicates.
    """

    def __init__(self, k: int = 3, w_geo: float = 0.4,
                 w_q: float = 0.4, w_prophet: float = 0.2):
        self.k = k
        self.w_geo = w_geo
        self.w_q = w_q
        self.w_prophet = w_prophet

    def compute_candidate_set(
        self,
        current_pos: np.ndarray,
        dest_pos: np.ndarray,
        neighbors: Dict[int, dict],
        q_table: Dict[Tuple, float],
        prophet_table: Dict[int, float],
        state_key: Tuple,
    ) -> List[Tuple[int, float]]:
        """
        Compute ordered candidate list ranked by combined score.

        Args:
            current_pos: Current node position (x, y).
            dest_pos: Destination node position (x, y).
            neighbors: {node_id: {'position': np.ndarray, 'alive': bool, ...}}
            q_table: Q-table from the node's RL agent.
            prophet_table: PROPHET delivery probabilities.
            state_key: (current_node_id, destination_id) for Q-lookup.

        Returns:
            List of (neighbor_id, combined_score), top-k, descending.
        """
        scored = []
        d_curr = np.linalg.norm(current_pos - dest_pos)

        for nid, ninfo in neighbors.items():
            if not ninfo.get('alive', True):
                continue

            # Geographic progress: how much closer does this neighbor get us?
            n_pos = ninfo['position']
            d_neigh = np.linalg.norm(n_pos - dest_pos)
            geo = max(0.0, (d_curr - d_neigh) / d_curr) if d_curr > 1e-6 else 0.5

            # Q-value (sigmoid-normalized)
            q_raw = q_table.get((state_key, nid), 0.0)
            q_norm = 1.0 / (1.0 + np.exp(-q_raw))

            # PROPHET delivery probability
            prophet = prophet_table.get(nid, 0.1)

            combined = self.w_geo * geo + self.w_q * q_norm + self.w_prophet * prophet
            scored.append((nid, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.k]

    def should_suppress(self, my_priority: int, forwarded_by_priority: int) -> bool:
        """
        ExOR suppression rule: lower-priority nodes suppress their copy
        if a higher-priority node already forwarded the packet.

        Args:
            my_priority: This node's priority rank (0 = highest).
            forwarded_by_priority: Priority rank of the node that forwarded.

        Returns:
            True if this node should suppress (not forward).
        """
        return my_priority > forwarded_by_priority

    @staticmethod
    def geographic_progress(current_pos: np.ndarray, neighbor_pos: np.ndarray,
                            dest_pos: np.ndarray) -> float:
        """Compute normalized geographic progress toward destination."""
        d_curr = np.linalg.norm(current_pos - dest_pos)
        d_neigh = np.linalg.norm(neighbor_pos - dest_pos)
        if d_curr < 1e-6:
            return 0.5
        return max(0.0, (d_curr - d_neigh) / d_curr)

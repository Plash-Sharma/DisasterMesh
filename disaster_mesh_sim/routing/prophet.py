"""
PROPHET — Probabilistic Routing Protocol for Intermittently Connected Networks.

Maintains delivery probability estimates that are updated on contact events,
decayed over time, and propagated transitively through intermediate nodes.

Reference:
    Lindgren et al., "Probabilistic Routing Protocol for Intermittently
    Connected Networks", RFC 6693, 2012
"""

import time
from typing import Dict, Optional
from disaster_mesh_sim.utils.config import PROPHETConfig


class PROPHETProtocol:
    """
    PROPHET delivery probability manager.

    P(A,B) update on contact:
        P(A,B) = P_init + (1 - P_init) · P(A,B)_old

    Transitivity update (when A meets B, for each C known to B):
        P(A,C) = max(P(A,C), P(A,B) · P(B,C) · β)

    Aging (decay over time):
        P(A,B) = P(A,B) · γ^k   (k = intervals elapsed)
    """

    def __init__(self, node_id: int, config: PROPHETConfig):
        self.node_id = node_id
        self.config = config
        self.delivery_prob: Dict[int, float] = {}
        self.last_aging_time: float = 0.0

    def update_on_contact(self, other_id: int):
        """Update delivery probability when encountering another node."""
        p_old = self.delivery_prob.get(other_id, 0.0)
        p_new = self.config.p_init + (1 - self.config.p_init) * p_old
        self.delivery_prob[other_id] = min(1.0, p_new)

    def update_transitivity(self, other_id: int,
                            other_probs: Dict[int, float]):
        """
        Transitive update: if I meet B, and B knows C,
        then P(me, C) can increase.
        """
        p_ab = self.delivery_prob.get(other_id, 0.0)
        beta = self.config.beta_transitivity

        for c_id, p_bc in other_probs.items():
            if c_id == self.node_id:
                continue
            p_ac_old = self.delivery_prob.get(c_id, 0.0)
            p_ac_new = p_ac_old + (1 - p_ac_old) * p_ab * p_bc * beta
            self.delivery_prob[c_id] = min(1.0, p_ac_new)

    def age(self, current_time: float):
        """Apply aging/decay to all delivery probabilities."""
        elapsed = current_time - self.last_aging_time
        if elapsed < self.config.aging_interval:
            return

        k = int(elapsed / self.config.aging_interval)
        decay = self.config.gamma_aging ** k

        for nid in list(self.delivery_prob.keys()):
            self.delivery_prob[nid] *= decay
            # Remove negligible entries
            if self.delivery_prob[nid] < 0.001:
                del self.delivery_prob[nid]

        self.last_aging_time = current_time

    def get_probability(self, dest_id: int) -> float:
        """Get delivery probability for a destination."""
        return self.delivery_prob.get(dest_id, 0.0)

    def get_all_probabilities(self) -> Dict[int, float]:
        """Get copy of all delivery probabilities (for transitivity sharing)."""
        return dict(self.delivery_prob)

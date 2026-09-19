"""
Dynamic Feedback Learning (DFL) — RORQ + EDMBOPR dual-feedback loop.

Combines RORQ reputation system (Ryu & Kim, IEEE Access 2023)
with EDMBOPR mobility tracking (Das & Devi, P2P 2025).

On every packet delivery/failure:
    1. RORQ reputation update (Eq. 4 & 5)
    2. RORQ Q-table update (Eq. 10)
    3. EDMBOPR EMA link quality tracking
    4. Enhanced reward = base_reward + reputation_bonus
"""

import numpy as np
from typing import Dict, Tuple, Optional
from disaster_mesh_sim.routing.rorq_routing import RORQRouter, RORQNodeState


class DynamicFeedbackLearner:
    """
    Dual feedback learner combining RORQ reputation (Ryu & Kim, 2023)
    with EDMBOPR mobility tracking (Das & Devi, 2025).

    Creates a synergistic feedback loop:
    - Reputation enriches Q-value updates
    - Q-scores improve relay selection
    - EMA link quality tracks connection reliability
    """

    def __init__(self, rorq_router: RORQRouter,
                 ema_alpha: float = 0.2,
                 rep_reward_bonus: float = 1.5):
        self.rorq = rorq_router
        self.ema_alpha = ema_alpha          # EDMBOPR EMA weight
        self.rep_bonus = rep_reward_bonus   # reputation bonus multiplier
        self.link_quality: Dict[Tuple[int, int], float] = {}

    def update(self, src_id: int, relay_id: int, dest_id: int,
               relay_state: RORQNodeState, dest_state: RORQNodeState,
               delivery_success: bool, T_uv: float,
               base_reward: float, current_reputation: float) -> Tuple[float, float]:
        """
        Full dual-feedback update on routing outcome.

        Args:
            src_id: Source node ID.
            relay_id: Relay node ID that was used.
            dest_id: Destination node ID.
            relay_state: RORQ state of the relay.
            dest_state: RORQ state of the destination.
            delivery_success: Whether delivery succeeded.
            T_uv: Transmission time from u to v.
            base_reward: Base reward from EDMBOPR Eq. 20.
            current_reputation: Current reputation of the forwarding node.

        Returns:
            Tuple of (enhanced_reward, updated_reputation).
        """
        # 1. RORQ reputation update (Eq. 4 & 5)
        new_rep = self.rorq.update_reputation(
            src_id, relay_state, dest_state,
            delivery_success, T_uv, current_reputation
        )

        # 2. RORQ Q-table update (Eq. 10)
        self.rorq.update_q_table(src_id, dest_id, relay_id, relay_state)

        # 3. EDMBOPR EMA link quality
        pair_key = (src_id, relay_id)
        outcome = 1.0 if delivery_success else 0.0
        old_lq = self.link_quality.get(pair_key, 0.5)
        self.link_quality[pair_key] = (
            self.ema_alpha * outcome + (1 - self.ema_alpha) * old_lq
        )

        # 4. Enhanced reward = base_reward + reputation bonus
        rep_bonus = relay_state.reputation * self.rep_bonus
        enhanced_reward = base_reward + rep_bonus

        return enhanced_reward, new_rep

    def get_link_quality(self, src_id: int, relay_id: int) -> float:
        """Get EMA-smoothed link quality estimate."""
        return self.link_quality.get((src_id, relay_id), 0.5)

"""
Dynamic Feedback Learner (RORQ+EDMBOPR)
"""
from algorithms.rorq_routing import RORQAgent

class DynamicFeedbackLearner:
    def __init__(self):
        self.alpha = 0.5
        self.alpha_ema = 0.2
        self.link_quality = {}
        self.rorq_agent = RORQAgent()

    def update(self, src, relay, dest, delivery_success, T_uv, packet_ttl):
        # 1. RORQ Reputation update
        self.rorq_agent.update_reputation(src, relay, dest, delivery_success, T_uv, packet_ttl)
        
        # 2. RORQ Q-table update
        self.rorq_agent.update_q_value(src, dest.node_id, relay, delivery_success)
        
        # 3. EDMBOPR EMA Link quality
        pair_key = (src.node_id, relay.node_id)
        outcome = 1.0 if delivery_success else 0.0
        self.link_quality[pair_key] = (
            self.alpha_ema * outcome +
            (1 - self.alpha_ema) * self.link_quality.get(pair_key, 0.5)
        )

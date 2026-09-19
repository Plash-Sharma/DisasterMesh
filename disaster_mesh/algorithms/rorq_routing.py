"""
RORQ Q-routing (Eq. 4, 5, 10, 11)
Ref: Ryu & Kim, IEEE Access 2023
"""
import numpy as np

class RORQAgent:
    def __init__(self, eta=0.1, alpha=0.5, beta=0.5):
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        
    def update_q_value(self, node, dest_id, relay, outcome_success):
        # Eq. 10
        old_q = node.get_q_value(dest_id, relay.node_id)
        e_level = relay.battery / 100.0
        rep_y = relay.reputation
        buffer_ratio = relay.idle_buffer / relay.max_buffer
        
        if outcome_success:
            new_q = (1 - self.eta) * old_q + self.eta * (e_level + rep_y + buffer_ratio)
        else:
            new_q = (1 - self.eta) * old_q + self.eta * (-1.0)
            
        node.set_q_value(dest_id, relay.node_id, new_q)
        return new_q
        
    def update_reputation(self, src, relay, dest, success, T_uv, packet_ttl):
        # Eq. 4 & 5
        if success:
            d_uv = np.linalg.norm(src.position - relay.position)
            d_dest_v = np.linalg.norm(dest.position - relay.position)
            proximity_score = d_uv / (d_dest_v + 1e-6)
            speed_score = packet_ttl / (T_uv + 1e-6)
            rep_new = self.beta * proximity_score + (1 - self.beta) * speed_score
        else:
            rep_new = 0.0
            
        rep_old = src.reputation
        src.reputation = self.alpha * rep_old + (1 - self.alpha) * rep_new

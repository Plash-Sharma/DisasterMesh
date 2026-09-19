"""
Balanced epsilon-greedy (Eq. 18, eps_i=0.15)
"""
import random
import numpy as np

def select_relay(node, cfl, dest_id):
    """EDMBOPR Eq. 18"""
    EPSILON_I = 0.15
    
    if not cfl:
        return None
        
    for idx, candidate in enumerate(cfl):
        dist = np.linalg.norm(node.position - candidate.position)
        T_ij = dist + candidate.avg_speed
        choice_selector = (T_ij < 250.0)
        
        rc = random.uniform(0, 1)
        if rc < EPSILON_I and choice_selector:
            return idx # Explore
        elif rc >= EPSILON_I and not choice_selector:
            continue
        else:
            break # Exploit
            
    # Exploit from Q-table
    best_idx = 0
    best_q = -float('inf')
    for i, candidate in enumerate(cfl):
        q = node.get_q_value(dest_id, candidate.node_id)
        if q > best_q:
            best_q = q
            best_idx = i
            
    return best_idx

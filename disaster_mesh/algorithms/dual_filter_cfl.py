"""
Combined RORQ+EDMBOPR CFL (Eq. 14 + Alg.1)
"""
import numpy as np

def build_cfl(node, ffl, k=3):
    """
    EDMBOPR mobility gate (Eq. 14) + RORQ reputation gate (Alg 1)
    """
    DIST_THRESHOLD = 250.0
    REP_THRESHOLD = 0.3
    
    cfl = []
    for neighbor in ffl:
        # 1. EDMBOPR mobility filter (Eq. 14)
        dist = np.linalg.norm(node.position - neighbor.position)
        T_ij = dist + neighbor.avg_speed
        mobility_ok = (T_ij < DIST_THRESHOLD)
        
        # 2. RORQ reputation filter
        rep_ok = (neighbor.reputation >= REP_THRESHOLD)
        
        # 3. RORQ isolation filter
        not_isolated = (len(neighbor.neighbors) > 1)
        
        if mobility_ok and rep_ok and not_isolated:
            cfl.append(neighbor)
            
    # Sort by (reputation * energy_level)
    cfl.sort(key=lambda n: n.reputation * (n.battery / 100.0), reverse=True)
    return cfl[:k]

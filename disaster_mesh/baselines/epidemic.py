"""
Epidemic DTN routing.
Ref: Mega-Prompt Baseline B
"""
import numpy as np

class EpidemicAgent:
    """Epidemic: flood to all candidates (simulated as random)."""
    def select_action(self, obs, k):
        return np.random.randint(0, max(k, 1)) if k > 0 else 0

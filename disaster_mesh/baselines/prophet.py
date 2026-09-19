"""
PRoPHET probabilistic routing.
Ref: Mega-Prompt Baseline F
"""
import numpy as np

class ProphetAgent:
    """PRoPHET probabilistic routing: selects relay probabilistically."""
    def select_action(self, obs, k):
        return np.random.randint(0, max(k, 1)) if k > 0 else 0

"""
AODV baseline routing.
Ref: Mega-Prompt Baseline A
"""
class AODVAgent:
    """AODV-style greedy geographic baseline: always picks first candidate (highest geo score)."""
    def select_action(self, obs, k):
        return 0

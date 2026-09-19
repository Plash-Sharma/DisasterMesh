"""
Spray & Wait DTN routing.
Ref: Mega-Prompt Baseline C
"""
class SprayAndWaitAgent:
    """Spray-and-Wait: prefer first encounter."""
    def select_action(self, obs, k):
        return 0

"""
Energy drain model. Ref: EDMBOPR Eq. 15
"""
class EnergyModel:
    def __init__(self):
        self.max_battery = 100.0
        self.drain_per_refresh = 0.05
        
    def drain(self, current_battery: float) -> float:
        """EDMBOPR Eq. 15 energy drain per FFL refresh"""
        return max(0.0, current_battery - self.drain_per_refresh)
        
    def transmit_drain(self, current_battery: float, tx_power_mw: float) -> float:
        # Approximate drain based on transmission power
        return max(0.0, current_battery - (tx_power_mw * 0.001))

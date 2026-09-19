"""
DisasterNode class (RORQ + EDMBOPR state).
"""
from typing import List, Dict, Tuple
from core.mobility import MobilityModel
from core.energy import EnergyModel

class DisasterNode:
    def __init__(self, node_id: int, node_type: str = 'survivor'):
        self.node_id = node_id
        self.mobility = MobilityModel(node_type)
        self.energy_model = EnergyModel()
        self.battery = 100.0
        self.reputation = 0.5 # RORQ REP_INIT
        self.q_table: Dict[Tuple[int, int], float] = {} # (dest_id, relay_id) -> q_value
        self.ffl: List['DisasterNode'] = []
        self.cfl: List['DisasterNode'] = []
        self.role = 'LEAF'
        self.neighbors: List['DisasterNode'] = []
        self.max_buffer = 100
        self.buffer = []
        self.n_t: Dict[int, int] = {} # Visit count for epsilon decay (Eq. 11)
        self.tx_power_mw = 1.0
        self.active = True
        
    @property
    def position(self):
        return self.mobility.position
        
    @property
    def avg_speed(self):
        return self.mobility.get_avg_speed()
        
    @property
    def idle_buffer(self):
        return self.max_buffer - len(self.buffer)
        
    def step(self, current_time: float, dt: float):
        if not self.active:
            return
            
        self.mobility.step(current_time, dt)
        
        if self.battery <= 0:
            self.active = False
            self.battery = 0.0
            
    def compute_role(self):
        """NodeScore = 0.35*(battery/100) + 0.35*reputation + 0.30*(1-speed/50)"""
        if not self.active:
            self.role = 'DEAD'
            return
            
        speed_factor = max(0.0, 1.0 - (self.avg_speed / 50.0))
        score = 0.35 * (self.battery / 100.0) + 0.35 * self.reputation + 0.30 * speed_factor
        
        if score > 0.75:
            self.role = 'ANCHOR'
        elif score > 0.40 and self.reputation >= 0.3 and self.battery > 25.0:
            self.role = 'RELAY'
        else:
            self.role = 'LEAF'
            
    def get_q_value(self, dest_id: int, relay_id: int) -> float:
        return self.q_table.get((dest_id, relay_id), 0.0)
        
    def set_q_value(self, dest_id: int, relay_id: int, val: float):
        self.q_table[(dest_id, relay_id)] = val

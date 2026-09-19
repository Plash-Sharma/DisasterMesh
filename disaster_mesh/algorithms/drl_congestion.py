"""
DRL-MANET congestion fallback (Pillai 2025)
"""
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class DRLCongestionDQN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DRLCongestionFallback:
    def __init__(self):
        self.dqn = DRLCongestionDQN()
        self.memory = deque(maxlen=5000)
        self.epsilon = 0.1
        self.spray_copies = 6
        
    def fallback_route(self, packet, network_state, connectivity_ratio):
        state = [network_state.get('throughput', 0.0), 
                 network_state.get('avg_latency', 0.0),
                 network_state.get('energy_consumed', 0.0),
                 connectivity_ratio]
                 
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.dqn(state_tensor)
            action = torch.argmax(q_values).item()
            
        if action == 0:   
            return 'boost_power'
        elif action == 1: 
            return 'reduce_rate'
        elif action == 2: 
            return 'store_carry_forward'
        else:
            L = max(2, int(self.spray_copies * connectivity_ratio))
            return f'spray_and_wait_{L}'

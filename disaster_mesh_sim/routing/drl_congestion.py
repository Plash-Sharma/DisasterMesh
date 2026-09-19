"""
DRL-MANET — DRL-Based Congestion Management Fallback (DTN Mode).

Implements congestion control from:
    Pillai et al., "Optimizing Resource Allocation and Congestion Management
    in MANETs: A DRL Approach for Scalable and Secure Network Operations",
    IEEE GIET 2025.
    DOI: 10.1109/GIET65294.2025.11234842

Triggered when connectivity_ratio < 0.3 (heavily fragmented network).
Uses DQN to adaptively select routing strategy under congestion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Optional, Tuple


class CongestionDQN(nn.Module):
    """
    DQN for congestion management (DRL-MANET 2025).

    Architecture: [4] → Dense(64, ReLU) → Dense(32, ReLU) → Dense(4)
    State:  [throughput, latency, energy_consumed, active_ratio]
    Action: [boost_power, reduce_rate, store_carry_forward, spray_copies]
    """

    def __init__(self, state_dim: int = 4, hidden1: int = 64,
                 hidden2: int = 32, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DRLCongestionFallback:
    """
    DRL-based fallback routing for heavily fragmented networks.

    From Pillai et al. IEEE GIET 2025:
    - State: [throughput, latency, energy_consumed, scalability_metric]
    - Actions: boost_power | reduce_rate | store_carry_forward | spray_copies
    - L_spray = max(2, int(6 × connectivity_ratio × battery/100))
    """

    def __init__(self, state_dim: int = 4, hidden1: int = 64,
                 hidden2: int = 32, action_dim: int = 4,
                 lr: float = 0.001, memory_size: int = 5000,
                 batch_size: int = 32, gamma: float = 0.9,
                 spray_copies: int = 6,
                 connectivity_threshold: float = 0.3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.spray_copies = spray_copies
        self.connectivity_threshold = connectivity_threshold
        self.epsilon = 0.3
        self.batch_size = batch_size

        self.device = torch.device("cpu")
        self.dqn = CongestionDQN(state_dim, hidden1, hidden2, action_dim).to(self.device)
        self.target_net = CongestionDQN(state_dim, hidden1, hidden2, action_dim).to(self.device)
        self.target_net.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def should_activate(self, connectivity_ratio: float) -> bool:
        """Check if connectivity is low enough to activate DTN fallback."""
        return connectivity_ratio < self.connectivity_threshold

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection for congestion management."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.dqn(state_t).cpu().numpy()[0]
            return int(np.argmax(q_values))

    def get_action_name(self, action: int) -> str:
        """Human-readable action name."""
        names = ["boost_power", "reduce_rate", "store_carry_forward", "spray_copies"]
        return names[action] if action < len(names) else "unknown"

    def compute_spray_L(self, connectivity_ratio: float, battery_pct: float) -> int:
        """Compute spray parameter L based on network state."""
        return max(2, int(self.spray_copies * connectivity_ratio * battery_pct / 100))

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition for experience replay."""
        self.memory.append((state, action, reward, next_state, done))

    def learn(self) -> Optional[float]:
        """Sample mini-batch and perform one gradient step."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.dqn(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self, tau: float = 0.005):
        """Soft update target network."""
        for tp, pp in zip(self.target_net.parameters(), self.dqn.parameters()):
            tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)

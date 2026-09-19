"""
Q-Learning D2D Resource Allocation.

Implements D2D power allocation from:
    Ji, Li, Yang, Wang, "Resource Allocation and Optimization in Low-altitude
    D2D Communication Networks", IEEE TrustCom 2025.
    DOI: 10.1109/Trustcom66490.2025.00396

Key Equations:
    Eq. 13: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') - Q(s,a)]
    Eq. 15: r_t = η·log2(1+SINR) - μ·P_D2D
    Table I: α=0.1, γ=0.9, exploration=0.1

Result: 11-15% throughput improvement over baselines.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class QD2DAllocator:
    """
    Q-Learning based D2D power allocation.

    From Ji et al. IEEE TrustCom 2025:
    - State: discretized (location, CQI, interference, channel_gain)
    - Action: 5 power allocation levels
    - Reward: η·log2(1+SINR) - μ·P_D2D
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 exploration_rate: float = 0.1, power_levels: int = 5,
                 eta_throughput: float = 1.0, mu_power: float = 0.5,
                 bandwidth: float = 500e6):
        self.alpha = alpha                 # Ji et al. Table I
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.power_levels = power_levels
        self.eta = eta_throughput
        self.mu = mu_power
        self.bandwidth = bandwidth

        # Q-table: {(state_tuple, action): q_value}
        self.q_table: Dict[Tuple, float] = {}

        # Power level mapping (normalized)
        self.power_values = np.linspace(0.1, 1.0, power_levels)

    def _discretize_state(self, location_norm: float, cqi: float,
                           interference: float, channel_gain: float) -> Tuple:
        """Discretize continuous state into Q-table keys."""
        loc_bin = int(np.clip(location_norm * 10, 0, 9))
        cqi_bin = int(np.clip(cqi * 5, 0, 4))
        intf_bin = int(np.clip(interference * 5, 0, 4))
        gain_bin = int(np.clip(channel_gain * 5, 0, 4))
        return (loc_bin, cqi_bin, intf_bin, gain_bin)

    def select_power_level(self, state: Tuple) -> int:
        """Epsilon-greedy power level selection."""
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.power_levels)

        q_values = [self.q_table.get((state, a), 0.0)
                    for a in range(self.power_levels)]
        return int(np.argmax(q_values))

    def compute_reward(self, sinr: float, tx_power_norm: float) -> float:
        """
        Ji et al. Eq. 15: Throughput-energy reward.

        r_t = η·log2(1 + SINR) - μ·P_D2D
        """
        throughput_term = self.eta * np.log2(1 + max(sinr, 1e-6))
        power_cost = self.mu * tx_power_norm
        return throughput_term - power_cost

    def update(self, state: Tuple, action: int, reward: float,
               next_state: Tuple) -> float:
        """
        Ji et al. Eq. 13: Standard Q-learning update.

        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s') - Q(s,a)]
        """
        key = (state, action)
        current_q = self.q_table.get(key, 0.0)

        next_max_q = max(
            self.q_table.get((next_state, a), 0.0)
            for a in range(self.power_levels)
        )

        new_q = current_q + self.alpha * (
            reward + self.gamma * next_max_q - current_q
        )
        self.q_table[key] = new_q
        return new_q

    def get_power_value(self, action: int) -> float:
        """Get actual power value for an action index."""
        return float(self.power_values[min(action, len(self.power_values) - 1)])

    def allocate_for_pair(self, distance: float, channel_noise: float = 0.1,
                          interference: float = 0.01) -> Tuple[float, float]:
        """
        Allocate power for a D2D pair.

        Returns:
            Tuple of (power_level, estimated_sinr).
        """
        # Compute channel gain (simplified path-loss)
        channel_gain = 1.0 / max(distance ** 2, 1.0)

        # Discretize state
        loc_norm = min(distance / 250.0, 1.0)
        cqi = min(channel_gain * 10, 1.0)
        intf_norm = min(interference * 10, 1.0)
        gain_norm = min(channel_gain, 1.0)
        state = self._discretize_state(loc_norm, cqi, intf_norm, gain_norm)

        # Select action
        action = self.select_power_level(state)
        power = self.get_power_value(action)

        # Compute SINR
        sinr = (power * channel_gain) / (interference + channel_noise + 1e-10)

        # Compute reward and update
        reward = self.compute_reward(sinr, power)
        self.update(state, action, reward, state)

        return power, sinr

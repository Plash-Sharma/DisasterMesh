"""
Deep Q-Network (DQN) Agent for DisasterMesh routing.

Architecture: [8] → Dense(64, ReLU) → Dense(32, ReLU) → Dense(k)
Features: Experience replay, target network with soft updates, ε-greedy.

References:
    [1] Mnih et al., "Human-level control through deep RL", Nature 2015
    [2] Boyan & Littman, "Packet Routing in Dynamically Changing Networks", NeurIPS 1994
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Optional
from dataclasses import dataclass

from disaster_mesh_sim.utils.config import DQNConfig, QRoutingConfig


@dataclass
class Experience:
    """Single transition for experience replay."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class QNetwork(nn.Module):
    """
    Q-value approximator neural network.
    
    Architecture: Input(state_dim) → FC(64, ReLU) → FC(32, ReLU) → Output(action_dim)
    """

    def __init__(self, state_dim: int, hidden1: int, hidden2: int, action_dim: int):
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


class ReplayBuffer:
    """Fixed-size circular experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, exp: Experience):
        self.buffer.append(exp)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.int64)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for mesh routing decisions.

    Uses:
    - ε-greedy exploration with decay
    - Experience replay buffer
    - Target network with soft (Polyak) updates
    """

    def __init__(self, dqn_config: DQNConfig, q_config: QRoutingConfig,
                 device: Optional[str] = None):
        self.config = dqn_config
        self.q_config = q_config

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Networks
        self.policy_net = QNetwork(
            dqn_config.state_dim, dqn_config.hidden1,
            dqn_config.hidden2, dqn_config.action_dim
        ).to(self.device)

        self.target_net = QNetwork(
            dqn_config.state_dim, dqn_config.hidden1,
            dqn_config.hidden2, dqn_config.action_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=dqn_config.learning_rate
        )
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(dqn_config.memory_size)

        # Exploration
        self.epsilon = q_config.epsilon_start
        self.epsilon_decay = q_config.epsilon_decay
        self.epsilon_min = q_config.epsilon_min

        # Tracking
        self.training_steps = 0
        self.losses = []

    def select_action(self, state: np.ndarray,
                      num_candidates: Optional[int] = None) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Observation vector.
            num_candidates: Actual number of available candidates
                            (may be < action_dim).
        Returns:
            Action index.
        """
        k = num_candidates or self.config.action_dim

        if random.random() < self.epsilon:
            return random.randint(0, k - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).cpu().numpy()[0]
            return int(np.argmax(q_values[:k]))

    def store(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool):
        """Store a transition in replay buffer."""
        self.memory.push(Experience(state, action, reward, next_state, done))

    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.

        Returns:
            Loss value, or None if buffer too small.
        """
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.memory.sample(self.config.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for chosen actions
        current_q = self.policy_net(states_t).gather(
            1, actions_t.unsqueeze(1)
        ).squeeze(1)

        # Target Q-values: R + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1 - dones_t) * self.q_config.discount_factor * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1
        loss_val = loss.item()
        self.losses.append(loss_val)

        # Soft update target network
        self._soft_update()

        return loss_val

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def _soft_update(self):
        """Polyak averaging: θ_target = τ·θ_policy + (1-τ)·θ_target"""
        tau = self.config.tau
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                tau * policy_param.data + (1 - tau) * target_param.data
            )

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.training_steps = checkpoint.get('training_steps', 0)

    def get_model_size_bytes(self) -> int:
        """Get approximate model size for edge computing budget check."""
        total = 0
        for param in self.policy_net.parameters():
            total += param.nelement() * param.element_size()
        return total

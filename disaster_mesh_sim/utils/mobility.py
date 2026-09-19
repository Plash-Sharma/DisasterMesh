"""
3D Mobility Models for DisasterMesh Simulation.

Implements EDMBOPR 3D Steady-State Random Waypoint mobility model
with position history tracking for average speed calculation.

References:
    [2] Das & Devi, "EDMBOPR", Peer-to-Peer Netw Apps, 2025
        Section 3.2: 3D Steady-State Random Waypoint
        Eq. 5: Instantaneous Average Mobility
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional, List
from collections import deque
from dataclasses import dataclass


class NodeType(Enum):
    """Types of nodes in a disaster scenario."""
    SURVIVOR = "survivor"
    RESCUE_WORKER = "rescue_worker"
    TRAPPED = "trapped"


class NodeRole(Enum):
    """Dynamic roles assigned based on node capabilities."""
    LEAF = "leaf"
    RELAY = "relay"
    ANCHOR = "anchor"
    SUPER = "super"


@dataclass
class PositionRecord:
    """A timestamped 3D position for mobility tracking (EDMBOPR Eq. 5)."""
    pos: np.ndarray
    time: float


class MobilityModel:
    """
    3D Steady-State Random Waypoint mobility model (EDMBOPR Section 3.2).

    Each node type has different movement characteristics:
    - Survivors: Slow, unpredictable movement (0.5-2 m/s)
    - Rescue Workers: Faster, more directed (2-5 m/s)
    - Trapped: Completely stationary

    Supports 3D positions for EDMBOPR's 3D DBSCAN clustering.
    """

    def __init__(
        self,
        area_width: float,
        area_height: float,
        rng: np.random.Generator,
        area_depth: float = 1000.0,
        use_3d: bool = True,
    ):
        self.area_width = area_width
        self.area_height = area_height
        self.area_depth = area_depth
        self.rng = rng
        self.use_3d = use_3d

    def init_position(self) -> np.ndarray:
        """Generate a random initial position within the simulation area."""
        if self.use_3d:
            return np.array([
                self.rng.uniform(0, self.area_width),
                self.rng.uniform(0, self.area_height),
                self.rng.uniform(0, self.area_depth * 0.1),  # mostly ground-level
            ])
        return np.array([
            self.rng.uniform(0, self.area_width),
            self.rng.uniform(0, self.area_height),
            0.0,
        ])

    def generate_waypoint(
        self,
        current_pos: np.ndarray,
        node_type: NodeType,
        speed_range: Tuple[float, float],
        pause_range: Tuple[float, float],
    ) -> dict:
        """
        Generate next waypoint for 3D Random Waypoint model.

        Returns:
            dict with keys: target_pos, speed, pause_time
        """
        if node_type == NodeType.TRAPPED:
            return {
                'target_pos': current_pos.copy(),
                'speed': 0.0,
                'pause_time': float('inf'),
            }

        target = np.array([
            self.rng.uniform(0, self.area_width),
            self.rng.uniform(0, self.area_height),
            self.rng.uniform(0, self.area_depth * 0.1) if self.use_3d else 0.0,
        ])

        speed = self.rng.uniform(speed_range[0], speed_range[1])
        pause = self.rng.uniform(pause_range[0], pause_range[1])

        return {
            'target_pos': target,
            'speed': speed,
            'pause_time': pause,
        }

    def move_toward(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        speed: float,
        dt: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        Move node toward target position in 3D space.

        Returns:
            Tuple of (new_position, reached_target).
        """
        if speed <= 0:
            return current_pos.copy(), True

        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return current_pos.copy(), True

        step_distance = speed * dt

        if step_distance >= distance:
            return target_pos.copy(), True

        unit_dir = direction / distance
        new_pos = current_pos + unit_dir * step_distance

        # Clamp to area bounds
        new_pos[0] = np.clip(new_pos[0], 0, self.area_width)
        new_pos[1] = np.clip(new_pos[1], 0, self.area_height)
        if self.use_3d:
            new_pos[2] = np.clip(new_pos[2], 0, self.area_depth)

        return new_pos, False

    @staticmethod
    def compute_avg_speed(position_history: List[PositionRecord], n: int = 10) -> float:
        """
        EDMBOPR Eq. 5: Instantaneous average mobility.

        Computes average speed over the last n position records.

        Args:
            position_history: List of PositionRecord(pos, time).
            n: Number of recent positions to use (default 10).

        Returns:
            Average speed in m/s.
        """
        if len(position_history) < 2:
            return 0.0
        recent = list(position_history)[-n:]
        speeds = []
        for i in range(1, len(recent)):
            d = np.linalg.norm(recent[i].pos - recent[i - 1].pos)
            dt = recent[i].time - recent[i - 1].time
            if dt > 0:
                speeds.append(d / dt)
        return float(np.mean(speeds)) if speeds else 0.0

    @staticmethod
    def euclidean_3d(a: np.ndarray, b: np.ndarray) -> float:
        """3D Euclidean distance between two position vectors."""
        return float(np.linalg.norm(a - b))


class GaussMarkovMobilityModel(MobilityModel):
    """
    Gauss-Markov mobility model for smoother, more realistic trajectories.

    v(t) = α·v(t-1) + (1-α)·v_mean + σ·sqrt(1-α²)·N(0,1)
    d(t) = α·d(t-1) + (1-α)·d_mean + σ_d·sqrt(1-α²)·N(0,1)
    """

    def __init__(
        self,
        area_width: float,
        area_height: float,
        rng: np.random.Generator,
        area_depth: float = 1000.0,
        alpha: float = 0.75,
        speed_sigma: float = 0.5,
        direction_sigma: float = 0.3,
    ):
        super().__init__(area_width, area_height, rng, area_depth)
        self.alpha = alpha
        self.speed_sigma = speed_sigma
        self.direction_sigma = direction_sigma

    def update_velocity(
        self,
        current_speed: float,
        current_direction: float,
        mean_speed: float,
        mean_direction: float,
    ) -> Tuple[float, float]:
        """Update speed and direction using Gauss-Markov process."""
        a = self.alpha
        new_speed = (
            a * current_speed
            + (1 - a) * mean_speed
            + self.speed_sigma * np.sqrt(1 - a**2) * self.rng.normal()
        )
        new_speed = max(0, new_speed)
        new_direction = (
            a * current_direction
            + (1 - a) * mean_direction
            + self.direction_sigma * np.sqrt(1 - a**2) * self.rng.normal()
        )
        return new_speed, new_direction

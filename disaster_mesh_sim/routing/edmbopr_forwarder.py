"""
EDMBOPR — Energy-Aware DBSCAN and Mobility-Aware Balanced Q-Learning
Based Opportunistic Routing Protocol.

Implements node discovery and candidate forwarding from:
    Das & Devi, "Energy Aware DBSCAN and Mobility Aware Balanced Q-Learning
    Based Opportunistic Routing Protocol in MANET",
    Peer-to-Peer Networking and Applications, Vol. 18, May 2025.
    DOI: 10.1007/s12083-025-02004-w

Key Equations:
    Eq. 5:  Instantaneous average mobility
    Eq. 12: DBSCAN First Forwarder List (FFL)
    Eq. 14: Threshold distance T_ij = d_ij + v_ij
    Eq. 18: Balanced epsilon-greedy with mobility gate
    Eq. 20: Mobility-aware reward function
"""

import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EDMBOPRNeighborInfo:
    """Information about a neighbor for EDMBOPR processing."""
    node_id: int
    position: np.ndarray      # 3D
    energy_level: float        # [0,1] = e_res / e_init
    avg_speed: float           # EDMBOPR Eq. 5
    reputation: float          # RORQ reputation
    neighbor_count: int        # 1-hop neighbor count
    alive: bool = True


class EDMBOPRForwarder:
    """
    EDMBOPR Candidate Forwarder combining DBSCAN discovery with
    mobility-threshold filtering and balanced exploration.

    Stages:
        1. DBSCAN-based FFL construction (Eq. 12)
        2. Dual-filter CFL: mobility gate (Eq. 14) + RORQ reputation
        3. Balanced epsilon-greedy relay selection (Eq. 18)
        4. Mobility-aware reward computation (Eq. 20)
    """

    def __init__(self, dbscan_eps: float = 250.0, dbscan_min_samples: int = 2,
                 energy_threshold: float = 0.25, dist_threshold: float = 250.0,
                 epsilon_i: float = 0.15, k: int = 3,
                 rep_threshold: float = 0.3,
                 mobility_penalty_w: float = 0.1,
                 energy_penalty_w: float = 0.2,
                 dest_bonus: float = 100.0):
        # EDMBOPR parameters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.energy_threshold = energy_threshold
        self.dist_threshold = dist_threshold
        self.epsilon_i = epsilon_i          # Section 4.1 optimal
        self.k = k                           # top-k CFL
        # RORQ parameters for dual filter
        self.rep_threshold = rep_threshold
        # Reward parameters (Eq. 20)
        self.mobility_penalty_w = mobility_penalty_w
        self.energy_penalty_w = energy_penalty_w
        self.dest_bonus = dest_bonus

    def build_ffl(self, node_pos: np.ndarray, node_energy: float,
                  neighbors: Dict[int, EDMBOPRNeighborInfo]) -> List[int]:
        """
        EDMBOPR Algorithm 2 — DBSCAN-based First Forwarder List.

        Eq. 12: FFL = nodes with energy > 25% AND within DBSCAN cluster.

        Args:
            node_pos: Current node's 3D position.
            node_energy: Current node's energy level [0,1].
            neighbors: Dictionary of neighbor info.

        Returns:
            List of neighbor IDs in the FFL.
        """
        if not neighbors:
            return []

        ffl = []
        for nid, ninfo in neighbors.items():
            if not ninfo.alive:
                continue

            # Energy filter: > 25% residual energy
            if ninfo.energy_level <= self.energy_threshold:
                continue

            # Distance filter: within DBSCAN eps
            dist = np.linalg.norm(node_pos - ninfo.position)
            if dist > self.dbscan_eps:
                continue

            # DBSCAN clustering check: are we in the same cluster?
            points = np.array([node_pos, ninfo.position])
            labels = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric='euclidean'
            ).fit(points).labels_

            if labels[0] == labels[1] and labels[0] != -1:
                ffl.append(nid)

        return ffl

    def build_ffl_fast(self, node_pos: np.ndarray,
                       neighbors: Dict[int, EDMBOPRNeighborInfo]) -> List[int]:
        """
        Optimized FFL construction without per-pair DBSCAN.

        For large neighbor sets, cluster all positions at once and filter
        by energy threshold.
        """
        if not neighbors:
            return []

        # Collect alive neighbors with sufficient energy
        candidates = []
        for nid, ninfo in neighbors.items():
            if ninfo.alive and ninfo.energy_level > self.energy_threshold:
                dist = np.linalg.norm(node_pos - ninfo.position)
                if dist <= self.dbscan_eps:
                    candidates.append(nid)

        if not candidates:
            return []

        # Build point cloud: self + all candidates
        points = [node_pos]
        id_map = [-1]  # index 0 = self
        for nid in candidates:
            points.append(neighbors[nid].position)
            id_map.append(nid)

        points_arr = np.array(points)
        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric='euclidean'
        ).fit(points_arr).labels_

        # FFL = neighbors in the same cluster as self
        self_label = labels[0]
        if self_label == -1:
            # Self is noise; include all candidates as fallback
            return candidates

        ffl = []
        for i, nid in enumerate(id_map):
            if i == 0:
                continue
            if labels[i] == self_label:
                ffl.append(nid)

        return ffl

    def build_cfl(self, ffl: List[int], node_pos: np.ndarray,
                  neighbors: Dict[int, EDMBOPRNeighborInfo]) -> List[int]:
        """
        EDMBOPR Eq. 14 + RORQ Algorithm 1: Dual-filter CFL.

        Filter 1 — Mobility threshold (Eq. 14): T_ij = d_ij + v_ij < 250m
        Filter 2 — Reputation gate (RORQ): Rep(v) >= rep_threshold
        Filter 3 — Isolation filter (RORQ Alg. 1): neighbor_count > 1

        Returns top-k candidates sorted by (reputation × energy).
        """
        cfl = []
        for nid in ffl:
            ninfo = neighbors[nid]

            # EDMBOPR Eq. 14: threshold distance
            dist = np.linalg.norm(node_pos - ninfo.position)
            T_ij = dist + ninfo.avg_speed
            if T_ij >= self.dist_threshold:
                continue  # Too mobile or too far

            # RORQ reputation filter
            if ninfo.reputation < self.rep_threshold:
                continue  # Unreliable forwarder

            # RORQ isolated node filter
            if ninfo.neighbor_count <= 1:
                continue  # Isolated node

            cfl.append(nid)

        # Sort by combined score: reputation × energy_level (descending)
        cfl.sort(
            key=lambda nid: neighbors[nid].reputation * neighbors[nid].energy_level,
            reverse=True
        )
        return cfl[:self.k]

    def select_relay_balanced(self, ffl: List[int], node_pos: np.ndarray,
                               neighbors: Dict[int, EDMBOPRNeighborInfo],
                               q_table: Dict[Tuple, float],
                               dest_id: int, src_id: int) -> Optional[int]:
        """
        EDMBOPR Eq. 18: Balanced epsilon-greedy with mobility-threshold gate.

        na_i = a_ex if (rc < εi AND c = (Tij < 250))
               else calculate Tij for next j
        ε_i = 0.15 (optimal from Section 4.1: exploration/exploitation ≈ 1.033)

        Args:
            ffl: First Forwarder List.
            node_pos: Current node position.
            neighbors: Neighbor info dict.
            q_table: Q-value table.
            dest_id: Destination node ID.
            src_id: Source node ID.

        Returns:
            Selected relay ID, or None if no valid candidate.
        """
        for nid in ffl:
            if nid not in neighbors:
                continue
            ninfo = neighbors[nid]

            dist = np.linalg.norm(node_pos - ninfo.position)
            T_ij = dist + ninfo.avg_speed  # EDMBOPR Eq. 14
            choice_selector = (T_ij < self.dist_threshold)

            rc = np.random.uniform(0, 1)

            if rc < self.epsilon_i and choice_selector:
                # Explore this candidate (passes mobility gate)
                return nid
            elif rc >= self.epsilon_i and not choice_selector:
                # Discard: too mobile/far, try next
                continue
            else:
                # Exploit: use Q-table
                return self._exploit_q_table(ffl, neighbors, q_table, dest_id, src_id)

        # Fallback: exploit Q-table
        return self._exploit_q_table(ffl, neighbors, q_table, dest_id, src_id)

    def _exploit_q_table(self, candidates: List[int],
                          neighbors: Dict[int, EDMBOPRNeighborInfo],
                          q_table: Dict[Tuple, float],
                          dest_id: int, src_id: int) -> Optional[int]:
        """Exploit Q-table to pick best relay (RORQ Eq. 10 values)."""
        if not candidates:
            return None
        q_values = {}
        for nid in candidates:
            if nid in neighbors:
                q_values[nid] = q_table.get((dest_id, nid), 0.0)
        if not q_values:
            return candidates[0] if candidates else None
        return max(q_values, key=q_values.get)

    def compute_reward(self, neighbor_info: EDMBOPRNeighborInfo,
                       delivery_success: bool, reached_dest: bool) -> float:
        """
        EDMBOPR Eq. 20: Mobility-aware reward function.

        r_i^(j) = -1 - (v_ij × 0.1) + energy_penalty
        energy_penalty = (e_ij - 25) × 0.2  if e_ij < 25%
        destination_reached: +100

        Returns:
            Reward value.
        """
        if reached_dest:
            return self.dest_bonus

        # Base intermediate penalty
        reward = -1.0

        # Mobility penalty (Eq. 20)
        mobility_penalty = neighbor_info.avg_speed * self.mobility_penalty_w
        reward -= mobility_penalty

        # Energy penalty (Eq. 20)
        energy_percent = neighbor_info.energy_level * 100
        if energy_percent < 25:
            energy_penalty = (energy_percent - 25) * self.energy_penalty_w
            reward += energy_penalty  # negative since (percent - 25) < 0

        return reward

    def compute_node_score(self, battery_pct: float, reputation: float,
                            speed: float, max_speed: float = 50.0) -> float:
        """
        Combined NodeScore from RORQ + EDMBOPR for role assignment.

        NodeScore = 0.35·(battery/100) + 0.35·reputation + 0.30·(1-speed/MAX_SPEED)

        Returns:
            Score in [0, 1].
        """
        battery_score = battery_pct / 100.0
        stability_score = 1.0 - min(speed / max_speed, 1.0)
        return 0.35 * battery_score + 0.35 * reputation + 0.30 * stability_score

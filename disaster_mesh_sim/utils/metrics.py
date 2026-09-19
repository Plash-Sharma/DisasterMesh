"""
Metrics tracking and computation for DisasterMesh simulation.

Tracks all key performance indicators:
- Packet Delivery Ratio (PDR)
- Average End-to-End Latency
- Energy Efficiency (packets per mAh)
- Network Formation Time
- RL Convergence metrics
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class PacketRecord:
    """Record of a single packet's journey through the mesh."""
    packet_id: int
    source_id: int
    destination_id: int
    creation_time: float
    delivery_time: Optional[float] = None
    delivered: bool = False
    hop_count: int = 0
    path: List[int] = field(default_factory=list)
    total_delay: float = 0.0
    battery_consumed: float = 0.0


class MetricsTracker:
    """
    Comprehensive metrics tracker for DisasterMesh evaluation.
    
    Collects per-packet, per-node, and per-episode statistics for
    comparison against baselines and convergence analysis.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new episode."""
        self.packets: Dict[int, PacketRecord] = {}
        self.episode_rewards: List[float] = []
        self.step_rewards: List[float] = []
        
        # Per-episode aggregates
        self.total_packets_sent = 0
        self.total_packets_delivered = 0
        self.total_packets_dropped = 0
        self.total_latency = 0.0
        self.total_hops = 0
        self.total_energy_consumed = 0.0
        
        # Node-level tracking
        self.node_packets_forwarded: Dict[int, int] = defaultdict(int)
        self.node_packets_originated: Dict[int, int] = defaultdict(int)
        self.node_energy_used: Dict[int, float] = defaultdict(float)
        
        # Network-level
        self.connectivity_history: List[float] = []
        self.active_nodes_history: List[int] = []
        self.network_diameter_history: List[int] = []
    
    def register_packet(self, packet: PacketRecord):
        """Register a new packet being sent."""
        self.packets[packet.packet_id] = packet
        self.total_packets_sent += 1
        self.node_packets_originated[packet.source_id] += 1
    
    def record_delivery(self, packet_id: int, delivery_time: float,
                        hop_count: int, path: List[int], battery_consumed: float):
        """Record successful packet delivery."""
        if packet_id not in self.packets:
            return
        
        pkt = self.packets[packet_id]
        pkt.delivered = True
        pkt.delivery_time = delivery_time
        pkt.hop_count = hop_count
        pkt.path = path
        pkt.total_delay = delivery_time - pkt.creation_time
        pkt.battery_consumed = battery_consumed
        
        self.total_packets_delivered += 1
        self.total_latency += pkt.total_delay
        self.total_hops += hop_count
        self.total_energy_consumed += battery_consumed
        
        for node_id in path[1:-1]:  # Exclude source and destination
            self.node_packets_forwarded[node_id] += 1
    
    def record_drop(self, packet_id: int):
        """Record a dropped/expired packet."""
        if packet_id in self.packets:
            self.packets[packet_id].delivered = False
            self.total_packets_dropped += 1
    
    def record_step_reward(self, reward: float):
        """Record reward for a single step."""
        self.step_rewards.append(reward)
    
    def record_episode_reward(self, total_reward: float):
        """Record total reward for an episode."""
        self.episode_rewards.append(total_reward)
    
    def record_network_state(self, connectivity_ratio: float,
                              active_nodes: int, diameter: int):
        """Record network-level metrics at a time step."""
        self.connectivity_history.append(connectivity_ratio)
        self.active_nodes_history.append(active_nodes)
        self.network_diameter_history.append(diameter)
    
    # ---- Computed Metrics ----
    
    @property
    def delivery_ratio(self) -> float:
        """Packet Delivery Ratio (PDR)."""
        if self.total_packets_sent == 0:
            return 0.0
        return self.total_packets_delivered / self.total_packets_sent
    
    @property
    def average_latency(self) -> float:
        """Average end-to-end latency in seconds."""
        if self.total_packets_delivered == 0:
            return float('inf')
        return self.total_latency / self.total_packets_delivered
    
    @property
    def average_hop_count(self) -> float:
        """Average hop count for delivered packets."""
        if self.total_packets_delivered == 0:
            return 0.0
        return self.total_hops / self.total_packets_delivered
    
    @property
    def energy_efficiency(self) -> float:
        """Packets delivered per unit energy consumed."""
        if self.total_energy_consumed <= 0:
            return 0.0
        return self.total_packets_delivered / self.total_energy_consumed
    
    @property
    def average_connectivity(self) -> float:
        """Average network connectivity ratio over time."""
        if not self.connectivity_history:
            return 0.0
        return float(np.mean(self.connectivity_history))
    
    def get_latency_distribution(self) -> np.ndarray:
        """Get array of all delivery latencies for CDF plotting."""
        latencies = [
            p.total_delay for p in self.packets.values() if p.delivered
        ]
        return np.array(latencies) if latencies else np.array([])
    
    def get_summary(self) -> Dict:
        """Get a summary dictionary of all key metrics."""
        return {
            'delivery_ratio': self.delivery_ratio,
            'average_latency': self.average_latency,
            'average_hop_count': self.average_hop_count,
            'energy_efficiency': self.energy_efficiency,
            'total_packets_sent': self.total_packets_sent,
            'total_packets_delivered': self.total_packets_delivered,
            'total_packets_dropped': self.total_packets_dropped,
            'average_connectivity': self.average_connectivity,
            'total_episode_reward': sum(self.step_rewards),
        }
    
    def __repr__(self) -> str:
        s = self.get_summary()
        return (
            f"Metrics(PDR={s['delivery_ratio']:.3f}, "
            f"Latency={s['average_latency']:.2f}s, "
            f"Hops={s['average_hop_count']:.1f}, "
            f"Energy_Eff={s['energy_efficiency']:.3f}, "
            f"Sent={s['total_packets_sent']}, "
            f"Delivered={s['total_packets_delivered']})"
        )

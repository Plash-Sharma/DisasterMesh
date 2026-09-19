"""
Gymnasium-compatible DisasterMeshEnv
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from core.node import DisasterNode
from core.packet import Packet
from algorithms.dbscan_ffl import build_ffl
from algorithms.dual_filter_cfl import build_cfl
from algorithms.edmbopr_exploration import select_relay

class DisasterMeshEnv(gym.Env):
    def __init__(self, n_nodes=100, area_size=(1000,1000,1000)):
        super().__init__()
        self.n_nodes = n_nodes
        self.area_size = area_size
        self.dt = 0.1
        self.current_time = 0.0
        
        # State: [energy_level, reputation, buffer_ratio, link_quality, avg_speed, hop_count_norm, network_density_norm, threshold_T_ij_norm]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3) # CFL size is max 3
        
        self.nodes: List[DisasterNode] = []
        self.packets_in_flight: List[Packet] = []
        self.delivered_packets: List[Packet] = []
        self.dropped_packets: List[Packet] = []
        
        # Metrics
        self.sent_count = 0
        self.delivered_count = 0
        self.dropped_count = 0
        self.total_delay = 0.0
        self.total_energy_consumed = 0.0
        
        self.pending_routing_decisions = [] # List of (packet, current_node, cfl)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.current_time = 0.0
        self.nodes = []
        
        types = ['survivor'] * int(self.n_nodes * 0.6) + ['rescue'] * int(self.n_nodes * 0.1) + ['trapped'] * int(self.n_nodes * 0.3)
        if len(types) < self.n_nodes:
            types += ['survivor'] * (self.n_nodes - len(types))
        random.shuffle(types)
        
        for i in range(self.n_nodes):
            self.nodes.append(DisasterNode(node_id=i, node_type=types[i]))
            
        self.packets_in_flight = []
        self.delivered_packets = []
        self.dropped_packets = []
        self.sent_count = 0
        self.delivered_count = 0
        self.dropped_count = 0
        self.total_delay = 0.0
        self.total_energy_consumed = 0.0
        self.pending_routing_decisions = []
        
        self._update_network_topology()
        self._generate_packets()
        
        return self._get_next_state(), {}
        
    def _update_network_topology(self):
        active_nodes = [n for n in self.nodes if n.active]
        for n in active_nodes:
            n.neighbors = []
            for other in active_nodes:
                if n != other:
                    dist = np.linalg.norm(n.position - other.position)
                    if dist <= 250.0:
                        n.neighbors.append(other)
            
            n.battery = n.energy_model.drain(n.battery)
            self.total_energy_consumed += n.energy_model.drain_per_refresh
            
            if n.battery <= 0:
                n.active = False
            else:
                n.compute_role()
                n.ffl = build_ffl(n, n.neighbors)
                n.cfl = build_cfl(n, n.ffl)
                
    def _generate_packets(self):
        active_nodes = [n for n in self.nodes if n.active]
        if len(active_nodes) >= 2:
            src, dest = random.sample(active_nodes, 2)
            priority = random.choices([3, 2, 1], weights=[0.1, 0.3, 0.6])[0]
            pkt = Packet(src.node_id, dest.node_id, priority)
            pkt.creation_time = self.current_time
            self.packets_in_flight.append(pkt)
            self.sent_count += 1
            
            src.buffer.append(pkt)
            self.pending_routing_decisions.append((pkt, src, src.cfl))

    def _get_next_state(self):
        if not self.pending_routing_decisions:
            return np.zeros(8, dtype=np.float32)
            
        pkt, current_node, cfl = self.pending_routing_decisions[0]
        if not cfl:
            return np.zeros(8, dtype=np.float32)
            
        candidate = cfl[0]
        
        energy_level = candidate.battery / 100.0
        reputation = candidate.reputation
        buffer_ratio = candidate.idle_buffer / candidate.max_buffer
        link_quality = 1.0
        avg_speed = min(1.0, candidate.avg_speed / 50.0)
        hop_count_norm = min(1.0, len(pkt.path) / 10.0)
        network_density_norm = min(1.0, len(candidate.neighbors) / 20.0)
        
        dist = np.linalg.norm(current_node.position - candidate.position)
        t_ij = dist + candidate.avg_speed
        threshold_norm = min(1.0, t_ij / 250.0)
        
        return np.array([energy_level, reputation, buffer_ratio, link_quality, 
                         avg_speed, hop_count_norm, network_density_norm, threshold_norm], dtype=np.float32)

    def step(self, action: int):
        reward = 0.0
        done = False
        
        if not self.pending_routing_decisions:
            self._advance_time()
            return self._get_next_state(), 0.0, done, False, self._get_info()
            
        pkt, current_node, cfl = self.pending_routing_decisions.pop(0)
        
        if not cfl or action >= len(cfl):
            pkt.dropped = True
            self.dropped_packets.append(pkt)
            self.dropped_count += 1
            if pkt in current_node.buffer:
                current_node.buffer.remove(pkt)
            reward = -5.0
        else:
            relay = cfl[action]
            
            if pkt in current_node.buffer:
                current_node.buffer.remove(pkt)
            pkt.path.append(relay.node_id)
            
            if relay.node_id == pkt.dest_id:
                pkt.delivered = True
                pkt.delivery_time = self.current_time
                self.delivered_packets.append(pkt)
                self.delivered_count += 1
                self.total_delay += (self.current_time - pkt.creation_time)
                reward = 100.0
            else:
                relay.buffer.append(pkt)
                self.pending_routing_decisions.append((pkt, relay, relay.cfl))
                v_ij = relay.avg_speed
                reward = -1.0 - (v_ij * 0.1) - max(0, (25.0 - relay.battery)*0.2)
                
        if not self.pending_routing_decisions:
            self._advance_time()
            done = (self.current_time >= 3600 * 24)
            
        return self._get_next_state(), reward, done, False, self._get_info()
        
    def _advance_time(self):
        self.current_time += self.dt
        for n in self.nodes:
            n.step(self.current_time, self.dt)
            
        if int(self.current_time * 10) % 10 == 0:
            self._update_network_topology()
            
        if int(self.current_time * 10) % 20 == 0:
            self._generate_packets()
            
        for pkt in list(self.packets_in_flight):
            if not pkt.delivered and not pkt.dropped:
                if self.current_time - pkt.creation_time > pkt.ttl:
                    pkt.dropped = True
                    self.dropped_packets.append(pkt)
                    self.dropped_count += 1
                    for n in self.nodes:
                        if pkt in n.buffer:
                            n.buffer.remove(pkt)

    def get_connectivity_ratio(self):
        active_nodes = sum(1 for n in self.nodes if n.active)
        if active_nodes == 0:
            return 0.0
        reachable = 0
        for n in self.nodes:
            if n.active and len(n.neighbors) > 0:
                reachable += 1
        return reachable / active_nodes

    def _get_info(self):
        pdr = self.delivered_count / max(1, self.sent_count)
        avg_delay = self.total_delay / max(1, self.delivered_count)
        active_nodes = sum(1 for n in self.nodes if n.active)
        energy_pp = self.total_energy_consumed / max(1, self.delivered_count)
        return {
            'pdr': pdr,
            'avg_delay': avg_delay,
            'energy_per_packet': energy_pp,
            'active_nodes': active_nodes,
            'timestamp': self.current_time
        }

"""
DisasterMeshEnv — Gymnasium environment with 7-stage algorithm pipeline.

Integrates all 5 papers:
    [1] RORQ (Ryu & Kim, IEEE Access 2023) — Q-routing + reputation
    [2] EDMBOPR (Das & Devi, P2P 2025) — DBSCAN FFL + mobility CFL
    [3] DRL-MANET (Pillai et al., IEEE GIET 2025) — congestion fallback
    [4] Q-D2D (Ji et al., TrustCom 2025) — resource allocation
    [5] HMRFCO (Ramesh Babu, Sci Reports 2025) — relay optimization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque

from disaster_mesh_sim.utils.config import DisasterMeshConfig, load_config
from disaster_mesh_sim.utils.mobility import (
    MobilityModel, NodeType, NodeRole, GaussMarkovMobilityModel, PositionRecord
)
from disaster_mesh_sim.utils.metrics import MetricsTracker, PacketRecord
from disaster_mesh_sim.routing.rorq_routing import RORQRouter, RORQNodeState
from disaster_mesh_sim.routing.edmbopr_forwarder import EDMBOPRForwarder, EDMBOPRNeighborInfo


@dataclass
class LinkInfo:
    neighbor_id: int
    rssi: float = -50.0
    link_quality: float = 1.0
    last_contact: float = 0.0
    contact_count: int = 0


@dataclass
class Packet:
    packet_id: int
    source_id: int
    destination_id: int
    creation_time: float
    ttl: int = 10
    hop_count: int = 0
    path: List[int] = field(default_factory=list)
    copies: int = 1
    priority: int = 0


class DisasterNode:
    """Node combining RORQ reputation + EDMBOPR mobility models."""
    MAX_SPEED = 50.0
    MAX_DEGREE = 20

    def __init__(self, node_id, position, node_type, battery, config):
        self.node_id = node_id
        self.position = position.copy()
        self.prev_position = position.copy()
        self.node_type = node_type
        self.battery = battery
        self.initial_battery = battery
        self.config = config
        self.alive = True
        self.target_pos = None
        self.speed = 0.0
        self.pause_remaining = 0.0
        self.direction = 0.0
        self.neighbors: Dict[int, LinkInfo] = {}
        self.message_queue: List[Packet] = []
        self.delivered_packets: List[int] = []
        self.role = NodeRole.LEAF
        self.q_table: Dict[Tuple, float] = {}
        self.reputation = config.rorq.rep_init
        self.position_history: deque = deque(maxlen=config.edmbopr.mobility_history_n)
        self.visit_count = 0
        self.link_quality_ema: Dict[int, float] = {}

    def compute_energy_level(self):
        """RORQ Eq. 3: e_level = e_res / e_init"""
        return self.battery / max(self.initial_battery, 1.0)

    def get_avg_speed(self):
        """EDMBOPR Eq. 5: average speed over last N positions."""
        return MobilityModel.compute_avg_speed(list(self.position_history))

    def compute_node_score(self):
        """NodeScore = 0.35*battery + 0.35*rep + 0.30*(1-mobility)"""
        cfg = self.config.node_scoring
        bat = self.battery / 100.0
        stab = 1.0 - min(self.speed / self.MAX_SPEED, 1.0)
        return cfg.w_battery * bat + cfg.w_reputation * self.reputation + cfg.w_stability * stab

    def assign_role(self):
        score = self.compute_node_score()
        cfg = self.config.node_scoring
        if score >= cfg.anchor_threshold and self.battery > 70:
            self.role = NodeRole.ANCHOR
        elif score >= cfg.relay_threshold and self.battery > 30:
            self.role = NodeRole.RELAY
        else:
            self.role = NodeRole.LEAF

    def drain_battery(self, activity, dt):
        rates = {
            'idle': self.config.battery.idle_drain_rate,
            'transmit': self.config.battery.transmit_drain_rate,
            'receive': self.config.battery.receive_drain_rate,
            'discovery': self.config.battery.discovery_drain_rate,
            'ffl_refresh': self.config.battery.ffl_refresh_drain,
        }
        self.battery = max(0.0, self.battery - rates.get(activity, 0.001) * dt)
        if self.battery <= self.config.battery.death_threshold:
            self.alive = False

    def queue_full(self):
        return len(self.message_queue) >= self.config.simulation.max_queue_size

    def get_rorq_state(self):
        return RORQNodeState(
            node_id=self.node_id, energy_level=self.compute_energy_level(),
            reputation=self.reputation,
            buffer_ratio=1.0 - len(self.message_queue) / max(self.config.simulation.max_queue_size, 1),
            position=self.position, avg_speed=self.get_avg_speed(),
            neighbor_count=len(self.neighbors)
        )

    def get_edmbopr_info(self):
        return EDMBOPRNeighborInfo(
            node_id=self.node_id, position=self.position.copy(),
            energy_level=self.compute_energy_level(), avg_speed=self.get_avg_speed(),
            reputation=self.reputation, neighbor_count=len(self.neighbors),
            alive=self.alive
        )

    def to_vis_dict(self):
        """Export node state for live visualization."""
        return {
            'id': self.node_id, 'x': float(self.position[0]),
            'y': float(self.position[1]),
            'z': float(self.position[2]) if len(self.position) > 2 else 0,
            'battery': round(self.battery, 1),
            'reputation': round(self.reputation, 3),
            'speed': round(self.speed, 2),
            'role': self.role.value, 'type': self.node_type.value,
            'alive': self.alive, 'neighbors': len(self.neighbors),
            'queue': len(self.message_queue),
            'score': round(self.compute_node_score(), 3),
        }


class DisasterMeshEnv(gym.Env):
    """
    Gymnasium environment implementing the 7-stage DisasterMesh pipeline.

    Observation: 8-dim [energy, reputation, buffer, link_quality, speed, hops, density, threshold_T]
    Action: index into CFL (size k=3)
    """
    metadata = {"render_modes": ["human", "none"], "render_fps": 1}

    def __init__(self, config=None, render_mode="none"):
        super().__init__()
        self.config = config or load_config()
        self.render_mode = render_mode
        self.rng = np.random.default_rng(self.config.simulation.seed)
        cfg = self.config
        k = cfg.edmbopr.candidate_set_size
        self.observation_space = spaces.Box(0.0, 1.0, (cfg.dqn.state_dim,), np.float32)
        self.action_space = spaces.Discrete(k)
        self.nodes: List[DisasterNode] = []
        self.graph = nx.Graph()
        self.mobility_model = None
        self.metrics = MetricsTracker()
        self.current_time = 0.0
        self.packet_counter = 0
        self.step_count = 0
        self._current_packet = None
        self._current_node_id = 0
        self._candidate_set: List[int] = []
        # Paper-grounded routing modules
        self.rorq = RORQRouter(
            eta=cfg.rorq.eta, gamma=cfg.rorq.gamma,
            alpha_rep=cfg.rorq.alpha_rep, beta_rep=cfg.rorq.beta_rep,
            rep_init=cfg.rorq.rep_init, rep_threshold=cfg.rorq.rep_threshold,
            c_constant=cfg.rorq.c_constant, packet_ttl=cfg.rorq.packet_ttl
        )
        self.edmbopr = EDMBOPRForwarder(
            dbscan_eps=cfg.edmbopr.dbscan_eps,
            dbscan_min_samples=cfg.edmbopr.dbscan_min_samples,
            energy_threshold=cfg.edmbopr.energy_threshold,
            dist_threshold=cfg.edmbopr.dist_threshold,
            epsilon_i=cfg.edmbopr.epsilon_i, k=cfg.edmbopr.candidate_set_size,
            rep_threshold=cfg.rorq.rep_threshold,
            mobility_penalty_w=cfg.edmbopr.mobility_penalty_weight,
            energy_penalty_w=cfg.edmbopr.energy_penalty_weight,
            dest_bonus=cfg.edmbopr.dest_reached_bonus
        )
        # Visualization state
        self._vis_edges: List[dict] = []
        self._vis_packets: List[dict] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.config
        self.current_time = 0.0
        self.step_count = 0
        self.packet_counter = 0
        self.metrics.reset()
        self._vis_packets = []
        self.mobility_model = MobilityModel(
            cfg.simulation.area_width, cfg.simulation.area_height,
            self.rng, cfg.simulation.area_depth
        )
        self._create_nodes()
        self._update_topology()
        for n in self.nodes:
            n.assign_role()
        obs = self._generate_routing_decision()
        return obs, self._get_info()

    def step(self, action):
        cfg = self.config
        reward = 0.0
        dt = cfg.simulation.time_step
        # 1. Execute RL routing action
        if self._current_packet and self._candidate_set:
            idx = min(action, len(self._candidate_set) - 1)
            relay = self._candidate_set[idx]
            reward = self._forward_packet(self._current_packet, self._current_node_id, relay)
        # 2. Background multi-hop forwarding
        bg = self._process_all_queued_packets()
        reward += bg * 0.1
        # 3. Advance time
        self.current_time += dt
        self.step_count += 1
        # 4. Update mobility
        self._update_mobility(dt)
        # 5. Record position history (EDMBOPR Eq. 5)
        for n in self.nodes:
            if n.alive:
                n.position_history.append(PositionRecord(n.position.copy(), self.current_time))
        # 6. Drain batteries
        for n in self.nodes:
            if n.alive:
                n.drain_battery('idle', dt)
        # 7. Update topology + DBSCAN FFL
        self._update_topology()
        # 8. Re-assign roles
        for n in self.nodes:
            if n.alive:
                n.assign_role()
        # 9. Generate packets
        self._generate_packets()
        # 10. Record metrics
        self.metrics.record_step_reward(reward)
        active = sum(1 for n in self.nodes if n.alive)
        self.metrics.record_network_state(self._compute_connectivity(), active, self._compute_diameter())
        # 11. Termination
        terminated = self.current_time >= cfg.simulation.simulation_duration
        truncated = self.step_count >= cfg.training.max_steps_per_episode
        if active / len(self.nodes) < 0.1:
            terminated = True
        obs = self._generate_routing_decision()
        return obs, float(reward), terminated, truncated, self._get_info()

    # ---- Node creation ----
    def _create_nodes(self):
        cfg = self.config
        n = cfg.simulation.num_nodes
        self.nodes = []
        n_surv = int(n * cfg.node_types.survivor_ratio)
        n_resc = int(n * cfg.node_types.rescue_worker_ratio)
        n_trap = n - n_surv - n_resc
        types = [NodeType.SURVIVOR]*n_surv + [NodeType.RESCUE_WORKER]*n_resc + [NodeType.TRAPPED]*n_trap
        self.rng.shuffle(types)
        for i, nt in enumerate(types):
            pos = self.mobility_model.init_position()
            bat = self.rng.uniform(cfg.battery.initial_min, cfg.battery.initial_max)
            node = DisasterNode(i, pos, nt, bat, cfg)
            sr, pr = self._speed_pause(nt)
            wp = self.mobility_model.generate_waypoint(pos, nt, sr, pr)
            node.target_pos = wp['target_pos']
            node.speed = wp['speed']
            node.pause_remaining = wp['pause_time']
            node.position_history.append(PositionRecord(pos.copy(), 0.0))
            self.nodes.append(node)

    def _speed_pause(self, nt):
        c = self.config.mobility
        if nt == NodeType.SURVIVOR:
            return (c.survivor_speed_min, c.survivor_speed_max), (c.pause_time_min, c.pause_time_max)
        elif nt == NodeType.RESCUE_WORKER:
            return (c.rescue_speed_min, c.rescue_speed_max), (c.pause_time_min, c.pause_time_max)
        return (0.0, 0.0), (1e9, 1e9)

    # ---- Mobility ----
    def _update_mobility(self, dt):
        for node in self.nodes:
            if not node.alive or node.node_type == NodeType.TRAPPED:
                continue
            node.prev_position = node.position.copy()
            if node.pause_remaining > 0:
                node.pause_remaining -= dt
                node.speed = 0.0
                continue
            if node.target_pos is None:
                sr, pr = self._speed_pause(node.node_type)
                wp = self.mobility_model.generate_waypoint(node.position, node.node_type, sr, pr)
                node.target_pos, node.speed, node.pause_remaining = wp['target_pos'], wp['speed'], wp['pause_time']
            new_pos, reached = self.mobility_model.move_toward(node.position, node.target_pos, node.speed, dt)
            node.position = new_pos
            if reached:
                sr, pr = self._speed_pause(node.node_type)
                wp = self.mobility_model.generate_waypoint(node.position, node.node_type, sr, pr)
                node.target_pos, node.speed, node.pause_remaining = wp['target_pos'], wp['speed'], wp['pause_time']

    # ---- Topology ----
    def _update_topology(self):
        bt_range = self.config.simulation.bt_range
        timeout = self.config.edmbopr.hello_interval * 10
        for node in self.nodes:
            expired = [nid for nid, l in node.neighbors.items() if self.current_time - l.last_contact > timeout]
            for nid in expired:
                del node.neighbors[nid]
        self.graph = nx.Graph()
        alive = [n for n in self.nodes if n.alive]
        for n in alive:
            self.graph.add_node(n.node_id)
        self._vis_edges = []
        for i, na in enumerate(alive):
            for nb in alive[i+1:]:
                dist = np.linalg.norm(na.position - nb.position)
                if dist <= bt_range:
                    rssi = -40 - 25 * np.log10(max(dist, 1.0))
                    lq = max(0.0, 1.0 - dist / bt_range)
                    for s, d in [(na, nb), (nb, na)]:
                        if d.node_id in s.neighbors:
                            link = s.neighbors[d.node_id]
                            link.rssi, link.link_quality, link.last_contact = rssi, lq, self.current_time
                            link.contact_count += 1
                        else:
                            s.neighbors[d.node_id] = LinkInfo(d.node_id, rssi, lq, self.current_time, 1)
                    self.graph.add_edge(na.node_id, nb.node_id, weight=lq)
                    self._vis_edges.append({'from': na.node_id, 'to': nb.node_id, 'quality': round(lq, 3)})

    def _compute_connectivity(self):
        alive = [n.node_id for n in self.nodes if n.alive]
        if len(alive) < 2: return 0.0
        r, t = 0, len(alive)*(len(alive)-1)/2
        for c in nx.connected_components(self.graph):
            s = len(c); r += s*(s-1)/2
        return r/t if t > 0 else 0.0

    def _compute_diameter(self):
        if self.graph.number_of_nodes() < 2: return 0
        try:
            largest = max(nx.connected_components(self.graph), key=len)
            sg = self.graph.subgraph(largest)
            return nx.diameter(sg) if sg.number_of_nodes() >= 2 else 0
        except: return 0

    # ---- Packet handling ----
    def _generate_packets(self):
        rate = self.config.simulation.packet_generation_rate
        alive = [n for n in self.nodes if n.alive and not n.queue_full()]
        comps = {}
        for comp in nx.connected_components(self.graph):
            for nid in comp: comps[nid] = comp
        for node in alive:
            if self.rng.random() < rate * self.config.simulation.time_step:
                my_comp = comps.get(node.node_id, set())
                reach = [m for m in self.nodes if m.alive and m.node_id != node.node_id and m.node_id in my_comp]
                others = [m for m in self.nodes if m.alive and m.node_id != node.node_id]
                if reach and self.rng.random() < 0.8: dest = self.rng.choice(reach)
                elif others: dest = self.rng.choice(others)
                else: continue
                pkt = Packet(self.packet_counter, node.node_id, dest.node_id, self.current_time)
                pkt.path.append(node.node_id)
                node.message_queue.append(pkt)
                self.packet_counter += 1
                self.metrics.register_packet(PacketRecord(pkt.packet_id, pkt.source_id, pkt.destination_id, pkt.creation_time))

    def _process_all_queued_packets(self):
        total = 0.0
        cfg = self.config
        fwd = 0
        for node in self.nodes:
            if not node.alive or not node.message_queue: continue
            for pkt in list(node.message_queue[:3]):
                if fwd >= 50: break
                if pkt.destination_id == node.node_id:
                    self.metrics.record_delivery(pkt.packet_id, self.current_time, pkt.hop_count, list(pkt.path), 0.0)
                    node.message_queue.remove(pkt)
                    total += cfg.reward.delivery_success; continue
                if pkt.destination_id in node.neighbors:
                    dn = self.nodes[pkt.destination_id]
                    if dn.alive:
                        pkt.hop_count += 1; pkt.path.append(pkt.destination_id)
                        node.drain_battery('transmit', 0.5); dn.drain_battery('receive', 0.5)
                        self.metrics.record_delivery(pkt.packet_id, self.current_time, pkt.hop_count, list(pkt.path), cfg.battery.transmit_drain_rate)
                        node.message_queue.remove(pkt)
                        total += cfg.reward.delivery_success; fwd += 1; continue
                dn = self.nodes[pkt.destination_id] if pkt.destination_id < len(self.nodes) else None
                if dn and dn.alive:
                    dc = np.linalg.norm(node.position - dn.position)
                    best, bp = None, -1.0
                    for nid in node.neighbors:
                        nb = self.nodes[nid]
                        if not nb.alive or nb.queue_full() or nid in pkt.path: continue
                        p = dc - np.linalg.norm(nb.position - dn.position)
                        if p > bp: bp, best = p, nid
                    if best is not None and bp > 0:
                        pkt.hop_count += 1; pkt.path.append(best)
                        node.drain_battery('transmit', 0.3); self.nodes[best].drain_battery('receive', 0.3)
                        self.nodes[best].message_queue.append(pkt); node.message_queue.remove(pkt); fwd += 1
                        if pkt.hop_count >= pkt.ttl:
                            self.nodes[best].message_queue.remove(pkt); self.metrics.record_drop(pkt.packet_id)
        return total

    def _forward_packet(self, pkt, from_id, to_id):
        cfg = self.config
        fn, tn = self.nodes[from_id], self.nodes[to_id]
        fn.drain_battery('transmit', 1.0); tn.drain_battery('receive', 1.0)
        bd = cfg.battery.transmit_drain_rate + cfg.battery.receive_drain_rate
        if not tn.alive or to_id not in fn.neighbors:
            return cfg.reward.delivery_failure
        pkt.hop_count += 1; pkt.path.append(to_id)
        # RORQ: update Q-table + reputation
        relay_state = tn.get_rorq_state()
        self.rorq.update_q_table(from_id, pkt.destination_id, to_id, relay_state)
        reached = (to_id == pkt.destination_id)
        if reached:
            self.metrics.record_delivery(pkt.packet_id, self.current_time, pkt.hop_count, list(pkt.path), bd)
            if pkt in fn.message_queue: fn.message_queue.remove(pkt)
            # Update reputation on success
            dest_state = self.nodes[pkt.destination_id].get_rorq_state()
            fn.reputation = self.rorq.update_reputation(from_id, relay_state, dest_state, True, 1.0, fn.reputation)
            self._vis_packets.append({'from': from_id, 'to': to_id, 'delivered': True})
            return self.edmbopr.compute_reward(tn.get_edmbopr_info(), True, True) + cfg.reward.reputation_bonus_weight * tn.reputation
        if not tn.queue_full():
            tn.message_queue.append(pkt)
            if pkt in fn.message_queue: fn.message_queue.remove(pkt)
        self._vis_packets.append({'from': from_id, 'to': to_id, 'delivered': False})
        dn = self.nodes[pkt.destination_id]
        df = np.linalg.norm(fn.position - dn.position)
        dt_ = np.linalg.norm(tn.position - dn.position)
        progress = 2.0 * (df - dt_) / max(df, 1e-6) if df > 1e-6 else 0.0
        return progress + self.edmbopr.compute_reward(tn.get_edmbopr_info(), False, False) * 0.5

    # ---- Observation ----
    def _generate_routing_decision(self):
        k = self.config.edmbopr.candidate_set_size
        for node in self.nodes:
            if not node.alive or not node.message_queue or not node.neighbors: continue
            pkt = node.message_queue[0]
            # Stage 1+2: EDMBOPR FFL → dual-filter CFL
            nb_info = {nid: self.nodes[nid].get_edmbopr_info() for nid in node.neighbors if self.nodes[nid].alive}
            ffl = self.edmbopr.build_ffl_fast(node.position, nb_info)
            cfl = self.edmbopr.build_cfl(ffl, node.position, nb_info) if ffl else []
            if not cfl:
                # Fallback: geographic candidates
                cfl = self._geo_candidates(node, pkt.destination_id, k)
            if not cfl: continue
            self._current_packet = pkt
            self._current_node_id = node.node_id
            self._candidate_set = cfl
            return self._build_obs(node, cfl, pkt)
        self._current_packet = None
        self._candidate_set = list(range(k))
        return np.zeros(self.config.dqn.state_dim, dtype=np.float32)

    def _geo_candidates(self, node, dest_id, k):
        dn = self.nodes[dest_id] if dest_id < len(self.nodes) else None
        scored = []
        for nid in node.neighbors:
            nb = self.nodes[nid]
            if not nb.alive: continue
            geo = 0.5
            if dn and dn.alive:
                dc = np.linalg.norm(node.position - dn.position)
                dg = np.linalg.norm(nb.position - dn.position)
                geo = max(0, (dc - dg) / max(dc, 1e-6))
            scored.append((nid, geo))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:k]]

    def _build_obs(self, node, cands, pkt):
        cfg = self.config
        lqs, qs, bs, spds = [], [], [], []
        for cid in cands:
            cn = self.nodes[cid]
            lqs.append(node.neighbors.get(cid, LinkInfo(cid)).link_quality)
            qs.append(1.0 - len(cn.message_queue) / max(cfg.simulation.max_queue_size, 1))
            bs.append(cn.battery / 100.0)
            spds.append(min(cn.get_avg_speed() / self.config.mobility.max_speed, 1.0))
        # 8-dim: [energy, rep, buffer, link_q, speed, hops, density, threshold_T]
        avg_T = np.mean([np.linalg.norm(node.position - self.nodes[c].position) + self.nodes[c].get_avg_speed() for c in cands]) / cfg.edmbopr.dist_threshold if cands else 0
        return np.array([
            float(np.mean(bs)), node.reputation, float(np.mean(qs)),
            float(np.mean(lqs)), float(np.mean(spds)),
            min(pkt.hop_count / 10.0, 1.0),
            min(len(node.neighbors) / DisasterNode.MAX_DEGREE, 1.0),
            min(avg_T, 1.0),
        ], dtype=np.float32)

    def _get_info(self):
        s = self.metrics.get_summary()
        s['current_time'] = self.current_time
        s['active_nodes'] = sum(1 for n in self.nodes if n.alive)
        s['total_nodes'] = len(self.nodes)
        s['candidate_set'] = list(self._candidate_set)
        return s

    # ---- Visualization state export ----
    def get_visualization_state(self):
        """Export full simulation state for live dashboard."""
        nodes = [n.to_vis_dict() for n in self.nodes]
        edges = list(self._vis_edges)
        packets = list(self._vis_packets)
        self._vis_packets = []
        metrics = self.metrics.get_summary()
        metrics['time'] = round(self.current_time, 1)
        metrics['connectivity'] = round(self._compute_connectivity(), 3)
        rep_dist = [round(n.reputation, 3) for n in self.nodes if n.alive]
        return {'nodes': nodes, 'edges': edges, 'packets': packets, 'metrics': metrics, 'reputation_distribution': rep_dist}

    def render(self): pass
    def close(self): self.nodes.clear(); self.graph.clear()

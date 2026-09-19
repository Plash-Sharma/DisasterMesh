"""
Configuration loader for DisasterMesh.

Loads YAML configuration files and provides typed access to all hyperparameters
grounded in the 5 research papers (2023-2025):
    [1] RORQ  — Ryu & Kim, IEEE Access, 2023
    [2] EDMBOPR — Das & Devi, Peer-to-Peer Netw Apps, 2025
    [3] DRL-MANET — Pillai et al., IEEE GIET, 2025
    [4] Q-Learning D2D — Ji et al., IEEE TrustCom, 2025
    [5] HMRFCO — Ramesh Babu & Nandakumar, Sci Reports, 2025
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SimulationConfig:
    """Simulation environment parameters."""
    area_width: float = 1000.0       # meters (EDMBOPR Section 4)
    area_height: float = 1000.0
    area_depth: float = 1000.0       # 3D space (EDMBOPR Section 3.2)
    num_nodes: int = 100             # EDMBOPR validated up to 1000
    bt_range: float = 250.0          # IEEE 802.11ax (EDMBOPR)
    simulation_duration: float = 600.0
    time_step: float = 1.0
    packet_generation_rate: float = 0.05
    max_queue_size: int = 50
    max_buffer_mb: int = 100         # per node (RORQ)
    packet_size_mb: int = 1          # per packet (RORQ)
    seed: int = 42


@dataclass
class NodeTypesConfig:
    """Distribution of node types in the simulation."""
    survivor_ratio: float = 0.60
    rescue_worker_ratio: float = 0.25
    trapped_ratio: float = 0.15


@dataclass
class MobilityConfig:
    """Mobility model parameters (EDMBOPR Section 3.2)."""
    model: str = "3d_random_waypoint"
    survivor_speed_min: float = 0.5
    survivor_speed_max: float = 2.0
    rescue_speed_min: float = 2.0
    rescue_speed_max: float = 5.0
    trapped_speed: float = 0.0
    max_speed: float = 50.0          # EDMBOPR validation range max
    pause_time_min: float = 5.0
    pause_time_max: float = 30.0
    position_history_len: int = 10   # EDMBOPR Eq. 5


@dataclass
class BatteryConfig:
    """Battery consumption model parameters."""
    initial_min: float = 30.0
    initial_max: float = 100.0
    idle_drain_rate: float = 0.001
    transmit_drain_rate: float = 0.005
    receive_drain_rate: float = 0.003
    discovery_drain_rate: float = 0.002
    ffl_refresh_drain: float = 0.05   # EDMBOPR Eq. 15
    death_threshold: float = 5.0


@dataclass
class EDMBOPRConfig:
    """EDMBOPR parameters (Das & Devi, P2P 2025)."""
    dbscan_eps: float = 250.0            # EDMBOPR Eq. 12
    dbscan_min_samples: int = 2
    energy_threshold: float = 0.25       # 25% min energy for FFL
    hello_interval: float = 1.0
    mobility_history_n: int = 10         # Eq. 5
    dist_threshold: float = 250.0        # Eq. 14
    epsilon_i: float = 0.15             # Section 4.1 optimal
    candidate_set_size: int = 3
    mobility_penalty_weight: float = 0.1 # Eq. 20
    energy_penalty_weight: float = 0.2   # Eq. 20
    dest_reached_bonus: float = 100.0    # Eq. 20


@dataclass
class RORQConfig:
    """RORQ parameters (Ryu & Kim, IEEE Access 2023)."""
    eta: float = 0.1                # Q-learning rate (Eq. 10)
    gamma: float = 0.9             # Discount factor
    alpha_rep: float = 0.5         # Reputation EMA weight (Eq. 4)
    beta_rep: float = 0.5          # Proximity vs speed (Eq. 5)
    rep_init: float = 0.5          # Initial reputation
    rep_threshold: float = 0.3     # Min reputation for CFL (Alg. 1)
    c_constant: float = 1.0       # Epsilon decay (Eq. 11)
    packet_ttl: int = 10           # hops


@dataclass
class NodeScoringConfig:
    """Combined RORQ + EDMBOPR node scoring."""
    w_battery: float = 0.35
    w_reputation: float = 0.35
    w_stability: float = 0.30
    anchor_threshold: float = 0.75
    relay_threshold: float = 0.40


@dataclass
class DQNConfig:
    """Deep Q-Network parameters (DRL-MANET, Pillai et al. 2025)."""
    state_dim: int = 8
    hidden1: int = 64
    hidden2: int = 32
    action_dim: int = 3
    learning_rate: float = 0.001
    tau: float = 0.005
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 10


@dataclass
class QRoutingConfig:
    """Tabular Q-Routing parameters."""
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon_start: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05


@dataclass
class RewardConfig:
    """Reward function (EDMBOPR Eq. 20 + RORQ)."""
    delivery_success: float = 100.0       # EDMBOPR: +100 destination
    delivery_failure: float = -5.0
    intermediate_base: float = -1.0       # EDMBOPR Eq. 20: -1/hop
    delay_penalty: float = -0.1
    hop_penalty: float = -0.5
    battery_penalty: float = -2.0
    congestion_penalty: float = -1.0
    reputation_bonus_weight: float = 1.5  # relay.rep × 1.5


@dataclass
class DFLConfig:
    """Dynamic Feedback Learning parameters."""
    ema_alpha: float = 0.2
    rep_reward_bonus: float = 1.5


@dataclass
class DRLCongestionConfig:
    """DRL-MANET congestion fallback (Pillai et al. 2025)."""
    enabled: bool = True
    connectivity_threshold: float = 0.3
    spray_copies: int = 6
    state_dim: int = 4
    hidden1: int = 64
    hidden2: int = 32
    action_dim: int = 4
    learning_rate: float = 0.001
    memory_size: int = 5000
    batch_size: int = 32


@dataclass
class QD2DConfig:
    """Q-Learning D2D resource allocation (Ji et al. TrustCom 2025)."""
    alpha: float = 0.1
    gamma: float = 0.9
    exploration_rate: float = 0.1
    power_levels: int = 5
    eta_throughput: float = 1.0
    mu_power: float = 0.5
    bandwidth: float = 500000000.0


@dataclass
class HMRFCOConfig:
    """HMRFCO optimizer (Ramesh Babu & Nandakumar, Sci Reports 2025)."""
    population_size: int = 30
    max_iterations: int = 250
    aresgru_input_dim: int = 18
    aresgru_hidden_dim: int = 64
    aresgru_output_dim: int = 2


@dataclass
class PROPHETConfig:
    """PROPHET routing protocol (legacy baseline)."""
    p_init: float = 0.75
    beta_transitivity: float = 0.25
    gamma_aging: float = 0.98
    aging_interval: float = 30.0


@dataclass
class MABConfig:
    """Multi-Armed Bandit (UCB1) parameters."""
    exploration_constant: float = 2.0


@dataclass
class FederatedConfig:
    """Federated Learning (WAFL) parameters."""
    local_epochs: int = 5
    fl_round_interval: float = 60.0
    gradient_quantization: int = 8
    sparse_top_k_percent: float = 30.0
    min_nodes_for_fl: int = 3


@dataclass
class SprayAndWaitConfig:
    """DTN Spray and Wait fallback parameters."""
    spray_copies: int = 6
    connectivity_threshold: float = 0.3
    packet_ttl: float = 300.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    eval_interval: int = 50
    save_interval: int = 100
    log_interval: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation parameters."""
    num_eval_episodes: int = 50
    baselines: List[str] = field(default_factory=lambda: [
        "rorq_only", "edmbopr_only", "epidemic",
        "spray_and_wait", "aodv", "disaster_mesh"
    ])


@dataclass
class DisasterMeshConfig:
    """Master configuration aggregating all sub-configs."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    node_types: NodeTypesConfig = field(default_factory=NodeTypesConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    edmbopr: EDMBOPRConfig = field(default_factory=EDMBOPRConfig)
    rorq: RORQConfig = field(default_factory=RORQConfig)
    node_scoring: NodeScoringConfig = field(default_factory=NodeScoringConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    q_routing: QRoutingConfig = field(default_factory=QRoutingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dfl: DFLConfig = field(default_factory=DFLConfig)
    drl_congestion: DRLCongestionConfig = field(default_factory=DRLCongestionConfig)
    q_d2d: QD2DConfig = field(default_factory=QD2DConfig)
    hmrfco: HMRFCOConfig = field(default_factory=HMRFCOConfig)
    prophet: PROPHETConfig = field(default_factory=PROPHETConfig)
    mab: MABConfig = field(default_factory=MABConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    spray_and_wait: SprayAndWaitConfig = field(default_factory=SprayAndWaitConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _fill_dataclass(dc_class, data: dict):
    """Fill a dataclass from a dictionary, ignoring unknown keys."""
    if data is None:
        return dc_class()
    field_names = {f.name for f in dc_class.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return dc_class(**filtered)


def load_config(config_path: Optional[str] = None) -> DisasterMeshConfig:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        DisasterMeshConfig: Fully populated configuration object.
    """
    if config_path is None:
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "configs", "default_config.yaml"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "configs", "default_config.yaml"),
            os.path.join(os.getcwd(), "configs", "default_config.yaml"),
        ]
        config_path = next((p for p in candidates if os.path.exists(p)),
                           candidates[0])

    if not os.path.exists(config_path):
        print(f"[Config] Config file not found at {config_path}, using defaults.")
        return DisasterMeshConfig()

    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return DisasterMeshConfig()

    return DisasterMeshConfig(
        simulation=_fill_dataclass(SimulationConfig, raw.get('simulation')),
        node_types=_fill_dataclass(NodeTypesConfig, raw.get('node_types')),
        mobility=_fill_dataclass(MobilityConfig, raw.get('mobility')),
        battery=_fill_dataclass(BatteryConfig, raw.get('battery')),
        edmbopr=_fill_dataclass(EDMBOPRConfig, raw.get('edmbopr')),
        rorq=_fill_dataclass(RORQConfig, raw.get('rorq')),
        node_scoring=_fill_dataclass(NodeScoringConfig, raw.get('node_scoring')),
        dqn=_fill_dataclass(DQNConfig, raw.get('dqn')),
        q_routing=_fill_dataclass(QRoutingConfig, raw.get('q_routing')),
        reward=_fill_dataclass(RewardConfig, raw.get('reward')),
        dfl=_fill_dataclass(DFLConfig, raw.get('dfl')),
        drl_congestion=_fill_dataclass(DRLCongestionConfig, raw.get('drl_congestion')),
        q_d2d=_fill_dataclass(QD2DConfig, raw.get('q_d2d')),
        hmrfco=_fill_dataclass(HMRFCOConfig, raw.get('hmrfco')),
        prophet=_fill_dataclass(PROPHETConfig, raw.get('prophet')),
        mab=_fill_dataclass(MABConfig, raw.get('mab')),
        federated=_fill_dataclass(FederatedConfig, raw.get('federated')),
        spray_and_wait=_fill_dataclass(SprayAndWaitConfig, raw.get('spray_and_wait')),
        training=_fill_dataclass(TrainingConfig, raw.get('training')),
        evaluation=_fill_dataclass(EvaluationConfig, raw.get('evaluation')),
    )

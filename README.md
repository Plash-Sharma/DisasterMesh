# DisasterMesh 🌐⚡

> **When towers fall, the mesh rises.**
> Built on RL. Powered by Bluetooth. Driven by the need to save lives.

## Overview

DisasterMesh is a self-organizing, AI-driven Bluetooth D2D (Device-to-Device) mesh network
that autonomously forms in disaster-struck areas where telecom infrastructure is destroyed.
Phones become nodes. RL agents become routers. Edge compute becomes the brain.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│   SOS Broadcast │ Location Sharing │ Text Mesh │ File Drop   │
├─────────────────────────────────────────────────────────────┤
│                  INTELLIGENCE LAYER (Edge AI)                │
│   Q-Learning Agent │ DFL (Dynamic Feedback) │ Node Scoring  │
├─────────────────────────────────────────────────────────────┤
│                    ROUTING LAYER                             │
│   Candidate Forwarding │ Q-Routing │ PROPHET │ Multi-path   │
├─────────────────────────────────────────────────────────────┤
│                    NETWORK LAYER                             │
│   Node Discovery │ Topology Mapping │ Link Quality Estimation│
├─────────────────────────────────────────────────────────────┤
│                   PHYSICAL/MAC LAYER                         │
│       Bluetooth Classic │ BLE 5.0 │ WiFi Direct (fallback)  │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```text
DisasterMesh/
├── configs/                # Hyperparameter configurations
├── disaster_mesh/          # Core implementation (Algorithms, Baselines, Visualizer)
│   ├── algorithms/         # DRL congestion management, routing algorithms
│   ├── baselines/          # Baseline protocols for benchmarking
│   ├── core/               # Core data structures and message types
│   ├── environment/        # Mesh environment logic
│   ├── evaluation/         # Benchmark runner and metrics
│   ├── training/           # RL Training loops
│   └── visualization/      # Real-time Plotly Dash visualizer
├── disaster_mesh_sim/      # Simulation components (Gymnasium env, RL Agents)
│   ├── agents/             # DQN, Q-Routing, MAB agents
│   ├── data/               # Datasets and synthetic generators
│   ├── envs/               # Gymnasium-compatible simulation environment
│   ├── evaluation/         # Evaluation scripts
│   ├── federated/          # WAFL Federated Learning module
│   ├── optimization/       # HMRFCO resource optimization
│   ├── results/            # Results from simulations
│   ├── routing/            # Candidate forwarding, PROPHET, Spray-and-Wait
│   ├── utils/              # Mobility models, metrics, helpers
│   └── visualization/      # Visualization tools
├── docs/                   # Research papers and project documentation
├── results/                # Training results and plots
├── create_folders.py       # Script to setup project structure
├── run_benchmark.py        # One-command benchmark runner
├── run_simulation.py       # One-command mesh visualizer launcher
└── requirements.txt        # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation training
python -m disaster_mesh_sim.train

# Launch real-time visualizer
python run_simulation.py

# Evaluate against baselines
python run_benchmark.py

# Generate synthetic disaster dataset
python -m disaster_mesh_sim.data.generator
```

## Algorithm Stack

| Stage | Algorithm | Purpose |
|-------|-----------|---------|
| 1 | Modified BATMAN-Adv | Node Discovery & Network Formation |
| 2 | ExOR + Q-Score Weighting | Opportunistic Candidate Forwarding |
| 3 | DQN Q-Routing | Reinforcement Learning Core |
| 4 | PROPHET + DFL | Dynamic Feedback Learning |
| 5 | UCB1 MAB | Efficient Relay Selection |
| 6 | FedAvg + WAFL | Distributed Federated Learning |
| 7 | Spray and Wait | DTN Fallback for Fragmented Networks |

## Key Metrics

| Metric | Target |
|--------|--------|
| Delivery Ratio | > 85% (connected), > 60% (fragmented) |
| End-to-End Latency | < 30 seconds |
| Network Formation | < 60 seconds |
| RL Convergence | < 500 episodes |
| Scalability | 50-200 nodes |

## Novel Research Contributions

1. **Q-Score Weighted ExOR Candidate Forwarding** — replaces geographic ranking with RL values
2. **Battery-Aware Multi-Objective RL Reward** — delivery + latency + energy + congestion
3. **WAFL over Bluetooth** with 8-bit gradient quantization
4. **Dynamic Node Role Assignment** via real-time NodeScore metric
5. **Dual Feedback Loop** — PROPHET probabilistic + Q-Learning RL simultaneously

## License

MIT License — For academic research and humanitarian purposes.

# DisasterMesh 🌐⚡

> **When towers fall, the mesh rises.**
> Built on RL. Powered by Bluetooth. Driven by the need to save lives.

## Overview

DisasterMesh is a self-organizing, AI-driven Bluetooth D2D (Device-to-Device) mesh network that autonomously forms in disaster-struck areas where telecom infrastructure is destroyed. Phones become nodes, RL agents become routers, and edge compute becomes the brain.

* **Implemented a 7-stage AI routing pipeline** (DBSCAN FFL, RORQ, EDMBOPR, HMRFCO, Q-D2D, DRL, Spray-and-Wait) achieving a 90% packet delivery ratio in connected topologies; visualized live metrics via a Plotly Dash dashboard with topology map, KPI cards, and scrolling timelines.
* **Simulated a self-organizing Bluetooth mesh** over a 500m×500m disaster zone, with mobile nodes dynamically classified into ANCHOR / RELAY / LEAF / DEAD roles via a weighted NodeScore formula (battery + reputation + mobility), links recalculated every tick based on a 120m BT range.

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
│   RORQ │ EDMBOPR │ HMRFCO │ Q-D2D │ Spray & Wait            │
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
├── disaster_mesh_sim/      # Simulation components (Gymnasium env, RL Agents)
├── papers/                 # Research papers referenced in the project
├── plots/                  # Visualized results and graphs
├── results/                # Training results, saved models, JSON logs
├── create_folders.py       # Script to setup project structure
├── dashboard_explanation.md # Explanation of visualizer components
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
```

## Algorithm Stack

The core pipeline integrates multiple advanced routing protocols and resource allocators derived from state-of-the-art research:

| Stage | Algorithm | Source Paper | Purpose |
|:---:|---|---|---|
| 1 | **DBSCAN FFL** | Das & Devi (P2P 2025) | Clusters nearby nodes using DBSCAN to build the First Forwarder List |
| 2 | **RORQ** | Ryu & Kim (IEEE Access 2023) | Reputation-based Q-routing — updates Q-values based on relay success/failure |
| 3 | **EDMBOPR** | Das & Devi (P2P 2025) | Epsilon-greedy relay selection from the Candidate Forwarder List |
| 4 | **HMRFCO** | Hybrid optimizer | Resource allocation — balances power, channels, and buffer usage |
| 5 | **Q-D2D** | D2D power allocation | Q-learning for optimal transmission power in Device-to-Device links |
| 6 | **DRL** | Pillai (2025) | Deep RL congestion management — triggers fallback actions when network is overloaded |
| 7 | **S&W** | Spray-and-Wait DTN | Store-carry-forward fallback for completely fragmented (disconnected) networks |

## Key Metrics

| Metric | Target |
|--------|--------|
| Delivery Ratio | > 85% (connected), > 60% (fragmented) |
| End-to-End Latency | < 30 seconds |
| Network Formation | < 60 seconds |
| RL Convergence | < 500 episodes |
| Scalability | 50-200 nodes |

## Novel Research Contributions

1. **Dual Feedback Loop (DFL)** — Integrates EDMBOPR energy/mobility tracking with RORQ reputation-based RL.
2. **Battery-Aware Multi-Objective Optimization** — Combines HMRFCO for resource allocation and Q-D2D for transmission power control.
3. **Dynamic Node Role Assignment** — Roles (Anchor, Relay, Leaf) adapt via real-time NodeScore metrics incorporating speed, battery, and reputation.
4. **Resilient Hybrid Fallback** — Deep RL (DRL) for congestion management paired with Spray-and-Wait (S&W) DTN strategies for highly fragmented networks.

## License

MIT License — For academic research and humanitarian purposes.

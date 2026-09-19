# DisasterMesh Dashboard — Detailed Explanation

This document explains every component of the **DisasterMesh Live Mesh Topology Simulation** dashboard.

---

## 🔝 Top Row — Key Performance Indicators (5 Cards)

| Card | Example Value | What It Measures |
|------|:---:|---|
| **📦 Delivery Ratio** | 90.0% | Percentage of all generated packets that successfully reached their destination node via multi-hop relay. Research target: >85% in connected networks, >60% in fragmented ones. |
| **⏱️ Avg Latency** | 3.7s | Mean time (in simulation ticks) between a packet being created and successfully delivered. Lower = better routing. |
| **🔋 Avg Battery** | 63.6% | Average remaining battery across all **active** nodes. Decreases over time as nodes transmit, relay, and discover neighbors. |
| **📡 Active Nodes** | 30 | Count of nodes still alive (battery > 0%). When nodes die, this number drops. |
| **🔗 Active Links** | 130 | Number of Bluetooth connections — pairs of active nodes within 120m of each other. |

---

## 🗺️ Center Left — Live Mesh Topology Map

This is the **primary visualization**: a bird's-eye view of a **500m × 500m disaster area**.

### Node Types (by shape & color)
| Symbol | Role | Meaning |
|:---:|---|---|
| 🟢 Diamond | **ANCHOR** | High-stability nodes — high battery, high reputation, low mobility. Act as backbone routers. |
| 🔵 Filled Circle | **RELAY** | Mid-tier forwarding nodes — decent battery & reputation. Route packets between anchors and leaves. |
| 🟡 Open Circle | **LEAF** | Weak or highly mobile nodes — low NodeScore. Can send/receive but don't relay well. |
| 🔴 ✕ | **DEAD** | Battery depleted (0%). No longer participates in the mesh. |

### Links (thin blue lines)
Each line represents an **active Bluetooth connection** between two nodes within the 120m BT range. Links form and break dynamically as nodes move.

### Node Movement
Nodes follow **Random Waypoint Mobility** (EDMBOPR Section 3.2):
- **Survivors** move slowly (0.5–2.0 m/s) — wandering through debris
- **Rescue teams** move faster (2.0–5.0 m/s) — searching the area
- **Trapped people** are nearly stationary (0.0–0.1 m/s)

### How Roles are Computed
Each node's role is determined by the **NodeScore** formula:

```
NodeScore = 0.35 × (battery/100) + 0.35 × reputation + 0.30 × (1 - speed/5)
```

| Score Range | Assigned Role |
|---|---|
| > 0.75 | ANCHOR |
| > 0.40 (+ rep ≥ 0.3, battery > 25%) | RELAY |
| Everything else | LEAF |
| Battery = 0 | DEAD |

---

## 🍩 Center Right (Top) — Node Roles Pie Chart

A **donut chart** showing the real-time distribution of roles across all 30 nodes. As the simulation progresses and batteries drain, you'll see:
- ANCHOR count shrinking
- RELAY → LEAF transitions
- DEAD slice appearing and growing

---

## 📊 Center Right (Bottom) — Battery Distribution

A **bar chart** grouping all nodes by battery level:

| Bucket | Color | Meaning |
|---|:---:|---|
| 0–25% | 🔴 Red | Critically low — node death imminent |
| 25–50% | 🟠 Orange | Low energy — role may downgrade |
| 50–75% | 🔵 Blue | Moderate — stable operation |
| 75–100% | 🟢 Green | Healthy — can serve as ANCHOR |

Over time, the distribution shifts leftward (toward red/orange) as the network degrades without charging.

---

## 📈 Bottom Left — Packet Delivery Ratio Timeline

A **scrolling area chart** showing PDR (%) over the last 80 simulation ticks.

- **Stable high line (~90%)** = mesh is well-connected, routing is effective
- **Dips** = network fragmentation, node deaths, or topology changes breaking routes
- **Decline over time** = expected as nodes die and the mesh degrades

---

## 📈 Bottom Right — Avg Network Battery Timeline

A **scrolling area chart** tracking mean battery level over time.

- **Gradual downward slope** = normal energy drain from discovery, relaying, and transmission
- **Sharp drops** = heavy relaying activity or multiple node deaths
- **Flat regions** = low network activity (e.g., many nodes paused)

---

## ⚡ Bottom — Algorithm Pipeline

Shows the **7-stage AI routing pipeline** from the research papers. The currently "active" stage is highlighted in purple and rotates each tick.

| Stage | Algorithm | Source Paper | Purpose |
|:---:|---|---|---|
| 1 | **DBSCAN FFL** | Das & Devi (P2P 2025) | Clusters nearby nodes using DBSCAN to build the First Forwarder List |
| 2 | **RORQ** | Ryu & Kim (IEEE Access 2023) | Reputation-based Q-routing — updates Q-values based on relay success/failure |
| 3 | **EDMBOPR** | Das & Devi (P2P 2025) | Epsilon-greedy relay selection from the Candidate Forwarder List |
| 4 | **HMRFCO** | Hybrid optimizer | Resource allocation — balances power, channels, and buffer usage |
| 5 | **Q-D2D** | D2D power allocation | Q-learning for optimal transmission power in Device-to-Device links |
| 6 | **DRL** | Pillai (2025) | Deep RL congestion management — triggers fallback actions when network is overloaded |
| 7 | **S&W** | Spray-and-Wait DTN | Store-carry-forward fallback for completely fragmented (disconnected) networks |

---

## 🔄 How the Simulation Works (Per Tick)

```
1. All nodes MOVE according to Random Waypoint mobility
2. Battery DRAINS slightly per tick (0.01–0.08%)
3. NodeScore is RECOMPUTED → roles may change
4. Bluetooth LINKS are recalculated (distance < 120m)
5. A new PACKET is generated every 3 ticks (random src → dst)
6. Packets attempt DELIVERY via direct link or probabilistic multi-hop
7. Undelivered packets are DROPPED after 15 hops
8. All METRICS are updated and pushed to the dashboard
```

> **Refresh rate:** Dashboard updates every **800ms** (~1.25 FPS)

"""
Synthetic Disaster Dataset Generator.

Generates CRAWDAD-compatible contact traces for disaster scenarios
with configurable node types, mobility patterns, and failure models.

Output format:
- contact_traces.csv: pairwise Bluetooth contact events
- node_metadata.csv: per-node properties
- network_snapshots.csv: periodic network statistics
"""

import os
import csv
import numpy as np
from typing import List
from disaster_mesh_sim.utils.config import load_config, DisasterMeshConfig
from disaster_mesh_sim.utils.mobility import MobilityModel, NodeType


def generate_dataset(config: DisasterMeshConfig = None, output_dir: str = None):
    """Generate synthetic disaster scenario dataset."""
    if config is None:
        config = load_config()
        # OVERRIDE: Increase dataset size for Excel/CSV files, but keep it
        # reasonable to avoid pure-Python O(N^2) bottlenecks taking 10+ minutes!
        config.simulation.num_nodes = 80
        config.simulation.simulation_duration = 3600  # 1 hour is plenty
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "disaster_mesh", "data", "synthetic"
        )
    os.makedirs(output_dir, exist_ok=True)

    rng = np.random.default_rng(config.simulation.seed)
    mob = MobilityModel(config.simulation.area_width,
                        config.simulation.area_height, rng)

    n = config.simulation.num_nodes
    bt_range = config.simulation.bt_range
    duration = config.simulation.simulation_duration
    dt = config.simulation.time_step

    # --- Create nodes ---
    n_surv = int(n * config.node_types.survivor_ratio)
    n_resc = int(n * config.node_types.rescue_worker_ratio)
    n_trap = n - n_surv - n_resc

    types = ([NodeType.SURVIVOR] * n_surv
             + [NodeType.RESCUE_WORKER] * n_resc
             + [NodeType.TRAPPED] * n_trap)
    rng.shuffle(types)

    positions = np.array([mob.init_position() for _ in range(n)])
    batteries = rng.uniform(config.battery.initial_min,
                            config.battery.initial_max, n)
    speeds = np.zeros(n)
    targets = positions.copy()

    for i, t in enumerate(types):
        if t == NodeType.SURVIVOR:
            speeds[i] = rng.uniform(config.mobility.survivor_speed_min,
                                    config.mobility.survivor_speed_max)
        elif t == NodeType.RESCUE_WORKER:
            speeds[i] = rng.uniform(config.mobility.rescue_speed_min,
                                    config.mobility.rescue_speed_max)
        targets[i] = mob.init_position()

    # --- Save node metadata ---
    meta_path = os.path.join(output_dir, "node_metadata.csv")
    with open(meta_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'type', 'init_x', 'init_y', 'battery', 'speed'])
        for i in range(n):
            w.writerow([i, types[i].value, f"{positions[i][0]:.2f}",
                        f"{positions[i][1]:.2f}", f"{batteries[i]:.1f}",
                        f"{speeds[i]:.2f}"])

    # --- Simulate and record contacts ---
    contacts_path = os.path.join(output_dir, "contact_traces.csv")
    snapshots_path = os.path.join(output_dir, "network_snapshots.csv")

    active_contacts = {}  # (i,j) → start_time

    with open(contacts_path, 'w', newline='') as cf, \
         open(snapshots_path, 'w', newline='') as sf:

        cw = csv.writer(cf)
        cw.writerow(['node_a', 'node_b', 'start_time', 'end_time',
                      'duration', 'avg_rssi', 'battery_a', 'battery_b'])

        sw = csv.writer(sf)
        sw.writerow(['time', 'active_nodes', 'total_contacts',
                      'avg_degree', 'connectivity'])

        t = 0.0
        step = 0
        total_contacts_logged = 0

        while t < duration:
            # Move nodes
            for i in range(n):
                if batteries[i] <= config.battery.death_threshold:
                    continue
                if types[i] == NodeType.TRAPPED:
                    continue

                direction = targets[i] - positions[i]
                dist = np.linalg.norm(direction)
                if dist < 1.0:
                    targets[i] = mob.init_position()
                    continue

                unit = direction / dist
                step_d = min(speeds[i] * dt, dist)
                positions[i] += unit * step_d
                positions[i] = np.clip(positions[i],
                                       [0, 0, 0],
                                       [config.simulation.area_width,
                                        config.simulation.area_height,
                                        config.simulation.area_depth])

            # Drain battery
            batteries -= config.battery.idle_drain_rate * dt

            # Detect contacts
            current_in_range = set()
            degrees = np.zeros(n)

            alive = [i for i in range(n)
                     if batteries[i] > config.battery.death_threshold]

            for idx_a, i in enumerate(alive):
                for j in alive[idx_a + 1:]:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= bt_range:
                        pair = (min(i, j), max(i, j))
                        current_in_range.add(pair)
                        degrees[i] += 1
                        degrees[j] += 1

                        if pair not in active_contacts:
                            active_contacts[pair] = t

            # End contacts no longer in range
            ended = [p for p in active_contacts if p not in current_in_range]
            for pair in ended:
                start = active_contacts.pop(pair)
                dur = t - start
                if dur > 0:
                    i, j = pair
                    dist = np.linalg.norm(positions[i] - positions[j])
                    rssi = -40 - 25 * np.log10(max(dist, 1.0))
                    cw.writerow([i, j, f"{start:.1f}", f"{t:.1f}",
                                 f"{dur:.1f}", f"{rssi:.1f}",
                                 f"{batteries[i]:.1f}", f"{batteries[j]:.1f}"])
                    total_contacts_logged += 1

            # Network snapshot every 60s
            if step % int(60 / dt) == 0:
                n_active = len(alive)
                avg_deg = float(np.mean(degrees[alive])) if alive else 0
                conn = len(current_in_range) / max(1, n_active * (n_active - 1) / 2)
                sw.writerow([f"{t:.1f}", n_active, len(current_in_range),
                             f"{avg_deg:.2f}", f"{conn:.4f}"])

            t += dt
            step += 1

    print(f"[DataGen] Generated {total_contacts_logged} contact events")
    print(f"[DataGen] Files saved to: {output_dir}")
    print(f"  - {meta_path}")
    print(f"  - {contacts_path}")
    print(f"  - {snapshots_path}")


if __name__ == "__main__":
    generate_dataset()

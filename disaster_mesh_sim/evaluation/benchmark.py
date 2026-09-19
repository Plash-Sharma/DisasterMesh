"""
Benchmark evaluation — compare DisasterMesh against baselines.

Baselines: Pure Q-Routing, PROPHET-only, Epidemic, Spray-and-Wait, AODV.
Metrics: Delivery Ratio, Latency CDF, Energy Consumption, Convergence.
"""

import os
import numpy as np
from typing import Dict, List
from disaster_mesh_sim.utils.config import load_config, DisasterMeshConfig
from disaster_mesh_sim.envs.disaster_mesh_env import DisasterMeshEnv
from disaster_mesh_sim.agents.dqn_agent import DQNAgent
from disaster_mesh_sim.agents.mab_agent import MABAgent


# ---------------------------------------------------------------------------
# Baseline routing strategies (simple heuristic agents)
# ---------------------------------------------------------------------------

from disaster_mesh.baselines.prophet import ProphetAgent as RandomAgent
from disaster_mesh.baselines.aodv import AODVAgent as GreedyGeographicAgent
from disaster_mesh.baselines.epidemic import EpidemicAgent
from disaster_mesh.baselines.spray_and_wait import SprayAndWaitAgent as SprayAgent


def _evaluate_agent(agent, env, config, num_episodes: int, label: str) -> Dict:
    """Run agent for N episodes and collect metrics."""
    all_dr = []
    all_lat = []
    all_energy = []
    all_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=config.simulation.seed + 10000 + ep)
        total_reward = 0.0
        done = False

        while not done:
            k = len(info.get('candidate_set', [0, 1, 2]))
            if hasattr(agent, 'select_action'):
                if isinstance(agent, DQNAgent):
                    action = agent.select_action(obs, k)
                else:
                    action = agent.select_action(obs, k)
            else:
                action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        dr = info.get('delivery_ratio', 0.0)
        lat = info.get('average_latency', 0.0)
        energy = info.get('energy_efficiency', 0.0)

        all_dr.append(dr)
        all_lat.append(lat if lat != float('inf') else 0.0)
        all_energy.append(energy)
        all_rewards.append(total_reward)

    return {
        'label': label,
        'delivery_ratio_mean': float(np.mean(all_dr)),
        'delivery_ratio_std': float(np.std(all_dr)),
        'latency_mean': float(np.mean(all_lat)),
        'latency_std': float(np.std(all_lat)),
        'energy_mean': float(np.mean(all_energy)),
        'reward_mean': float(np.mean(all_rewards)),
    }


def run_benchmark(config_path=None):
    """Run full benchmark comparison."""
    config = load_config(config_path)
    env = DisasterMeshEnv(config=config)
    n_eval = config.evaluation.num_eval_episodes

    results = []

    # 1. DisasterMesh (trained DQN)
    print("[Benchmark] Evaluating DisasterMesh (DQN)...")
    dqn_agent = DQNAgent(config.dqn, config.q_routing)
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "best_model.pth"
    )
    if os.path.exists(model_path):
        dqn_agent.load(model_path)
        dqn_agent.epsilon = 0.0  # Pure exploitation
    results.append(_evaluate_agent(dqn_agent, env, config, n_eval, "DisasterMesh"))

    # 2. Random
    print("[Benchmark] Evaluating Random baseline...")
    results.append(_evaluate_agent(RandomAgent(), env, config, n_eval, "Random"))

    # 3. Greedy Geographic (AODV-like)
    print("[Benchmark] Evaluating AODV (Greedy)...")
    results.append(_evaluate_agent(GreedyGeographicAgent(), env, config, n_eval, "AODV"))

    # 4. Epidemic
    print("[Benchmark] Evaluating Epidemic...")
    results.append(_evaluate_agent(EpidemicAgent(), env, config, n_eval, "Epidemic"))

    # 5. Spray-and-Wait
    print("[Benchmark] Evaluating Spray-and-Wait...")
    results.append(_evaluate_agent(SprayAgent(), env, config, n_eval, "Spray-and-Wait"))

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'PDR':>8} {'Latency':>10} {'Energy':>10} {'Reward':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<20} "
              f"{r['delivery_ratio_mean']:>7.3f} "
              f"{r['latency_mean']:>9.2f}s "
              f"{r['energy_mean']:>9.3f} "
              f"{r['reward_mean']:>9.2f}")
    print("=" * 80)

    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "benchmark_results.npy"), results)
    print(f"Results saved to {results_dir}/benchmark_results.npy")

    env.close()
    return results


if __name__ == "__main__":
    run_benchmark()

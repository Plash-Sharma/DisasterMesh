"""
DisasterMesh Training Script.

Trains the DQN agent in the DisasterMeshEnv simulation environment
with full algorithm stack: Q-Routing + Candidate Forwarding + PROPHET DFL
+ MAB relay selection + Spray-and-Wait fallback.

Usage:
    python -m disaster_mesh_sim.train [--config path/to/config.yaml]
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from disaster_mesh_sim.utils.config import load_config, DisasterMeshConfig
from disaster_mesh_sim.envs.disaster_mesh_env import DisasterMeshEnv
from disaster_mesh_sim.agents.dqn_agent import DQNAgent
from disaster_mesh_sim.agents.mab_agent import MABAgent


def train(config: Optional[DisasterMeshConfig] = None,
          config_path: Optional[str] = None):
    """
    Main training loop for DisasterMesh.

    Args:
        config: Pre-loaded config object.
        config_path: Path to YAML config (used if config is None).
    """
    if config is None:
        config = load_config(config_path)

    # ---- Setup ----
    env = DisasterMeshEnv(config=config)
    agent = DQNAgent(config.dqn, config.q_routing)
    mab = MABAgent(config.mab)

    # Results tracking
    episode_rewards = []
    episode_delivery_ratios = []
    episode_latencies = []
    best_delivery_ratio = 0.0

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("  DisasterMesh Training")
    print("=" * 60)
    print(f"  Nodes: {config.simulation.num_nodes}")
    print(f"  BT Range: {config.simulation.bt_range}m")
    print(f"  Area: {config.simulation.area_width}x{config.simulation.area_height}m")
    print(f"  Episodes: {config.training.num_episodes}")
    print(f"  Device: {agent.device}")
    model_kb = agent.get_model_size_bytes() / 1024
    print(f"  Model Size: {model_kb:.1f} KB")
    print("=" * 60)

    # ---- Training Loop ----
    for episode in tqdm(range(config.training.num_episodes),
                        desc="Training", unit="ep"):
        obs, info = env.reset(seed=config.simulation.seed + episode)
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Select action: combine DQN with MAB exploration
            candidates = info.get('candidate_set', [0, 1, 2])
            num_cand = len(candidates)

            # Use MAB for exploration at per-packet level
            if np.random.random() < 0.3 and num_cand > 0:
                action_idx = candidates.index(
                    mab.select_relay(candidates)
                ) if mab.select_relay(candidates) in candidates else 0
            else:
                action_idx = agent.select_action(obs, num_cand)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

            # Store transition
            agent.store(obs, action_idx, reward, next_obs, done)

            # Learn from replay
            agent.learn()

            # Update MAB statistics
            if num_cand > 0 and action_idx < num_cand:
                mab.update(candidates[action_idx], reward)

            obs = next_obs
            total_reward += reward
            steps += 1

        # Decay exploration
        agent.decay_epsilon()

        # Record episode metrics
        episode_rewards.append(total_reward)
        dr = info.get('delivery_ratio', 0.0)
        lat = info.get('average_latency', float('inf'))
        episode_delivery_ratios.append(dr)
        episode_latencies.append(lat if lat != float('inf') else 0.0)

        # Logging
        if (episode + 1) % config.training.log_interval == 0:
            avg_r = np.mean(episode_rewards[-config.training.log_interval:])
            avg_dr = np.mean(episode_delivery_ratios[-config.training.log_interval:])
            tqdm.write(
                f"  Ep {episode+1:4d} | "
                f"Reward: {avg_r:8.2f} | "
                f"PDR: {avg_dr:.3f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Steps: {steps}"
            )

        # Save best model
        if dr > best_delivery_ratio:
            best_delivery_ratio = dr
            agent.save(os.path.join(results_dir, "best_model.pth"))

        # Periodic save
        if (episode + 1) % config.training.save_interval == 0:
            agent.save(os.path.join(
                results_dir, f"checkpoint_ep{episode+1}.pth"
            ))

    # ---- Save final results ----
    agent.save(os.path.join(results_dir, "final_model.pth"))
    np.savez(
        os.path.join(results_dir, "training_history.npz"),
        rewards=np.array(episode_rewards),
        delivery_ratios=np.array(episode_delivery_ratios),
        latencies=np.array(episode_latencies),
        losses=np.array(agent.losses) if agent.losses else np.array([]),
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best Delivery Ratio: {best_delivery_ratio:.3f}")
    print(f"  Final ε: {agent.epsilon:.4f}")
    print(f"  Results saved to: {results_dir}")
    print("=" * 60)

    env.close()
    return agent, episode_rewards, episode_delivery_ratios


def main():
    parser = argparse.ArgumentParser(description="Train DisasterMesh RL Agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    args = parser.parse_args()
    train(config_path=args.config)


if __name__ == "__main__":
    main()

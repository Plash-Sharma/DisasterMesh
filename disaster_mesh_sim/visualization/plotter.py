"""
Visualization module for DisasterMesh results.

Generates publication-quality plots:
- Training convergence curves
- Delivery ratio comparison bar chart
- Latency CDF
- Network topology snapshots
- Energy consumption comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style setup for publication quality
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']


def plot_training_curves(history_path: str, save_dir: str):
    """Plot training reward and delivery ratio convergence."""
    data = np.load(history_path)
    rewards = data['rewards']
    delivery_ratios = data['delivery_ratios']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Smooth with moving average
    window = 20
    if len(rewards) >= window:
        smoothed_r = np.convolve(rewards, np.ones(window)/window, mode='valid')
        smoothed_dr = np.convolve(delivery_ratios, np.ones(window)/window, mode='valid')
    else:
        smoothed_r = rewards
        smoothed_dr = delivery_ratios

    ax1.plot(smoothed_r, color=COLORS[0], linewidth=1.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Reward Convergence')

    ax2.plot(smoothed_dr, color=COLORS[1], linewidth=1.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Packet Delivery Ratio')
    ax2.set_title('Delivery Ratio Convergence')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_convergence.png'), dpi=150)
    plt.close()
    print(f"  Saved: training_convergence.png")


def plot_benchmark_comparison(results: list, save_dir: str):
    """Plot bar chart comparing methods on delivery ratio, latency, energy."""
    labels = [r['label'] for r in results]
    pdr = [r['delivery_ratio_mean'] for r in results]
    lat = [r['latency_mean'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels))
    width = 0.5

    bars1 = ax1.bar(x, pdr, width, color=COLORS[:len(labels)], edgecolor='white')
    ax1.set_ylabel('Packet Delivery Ratio')
    ax1.set_title('Delivery Ratio Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars1, pdr):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(x, lat, width, color=COLORS[:len(labels)], edgecolor='white')
    ax2.set_ylabel('Average Latency (s)')
    ax2.set_title('Latency Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'benchmark_comparison.png'), dpi=150)
    plt.close()
    print(f"  Saved: benchmark_comparison.png")


def plot_loss_curve(losses: np.ndarray, save_dir: str):
    """Plot DQN training loss curve."""
    if len(losses) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    window = min(100, len(losses) // 2) if len(losses) > 10 else 1
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color=COLORS[4], linewidth=1, alpha=0.8)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('DQN Training Loss')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved: training_loss.png")


def generate_all_plots(results_dir: str):
    """Generate all available plots from training results."""
    os.makedirs(results_dir, exist_ok=True)
    print("[Visualization] Generating plots...")

    history_path = os.path.join(results_dir, 'training_history.npz')
    if os.path.exists(history_path):
        plot_training_curves(history_path, results_dir)
        data = np.load(history_path)
        if 'losses' in data and len(data['losses']) > 0:
            plot_loss_curve(data['losses'], results_dir)

    bench_path = os.path.join(results_dir, 'benchmark_results.npy')
    if os.path.exists(bench_path):
        results = np.load(bench_path, allow_pickle=True).tolist()
        plot_benchmark_comparison(results, results_dir)

    print("[Visualization] Done.")


if __name__ == "__main__":
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    generate_all_plots(results_dir)

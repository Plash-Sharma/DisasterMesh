"""
DQN training loop (RORQ + DRL-MANET)
Ref: Pillai et al., IEEE GIET 2025
"""
import sys
from disaster_mesh_sim.train import main as train_dqn

if __name__ == "__main__":
    sys.exit(train_dqn())

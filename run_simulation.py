"""
One-command mesh visualizer launcher
"""
import sys
import os

# Add disaster_mesh to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'disaster_mesh'))

from visualization.mesh_visualizer import run_visualizer

if __name__ == "__main__":
    print("Launching DisasterMesh visualizer...")
    run_visualizer()

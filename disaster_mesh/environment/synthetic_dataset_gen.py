"""
Generate disaster scenarios using EDMBOPR 3D mobility model
"""
import os
import sys
import csv
import random

# Add root project dir to path so core.mobility can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from disaster_mesh.core.mobility import MobilityModel

def generate_disaster_scenario(n_nodes=100, duration_hours=24, scenario_type='earthquake', output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic')
    os.makedirs(output_dir, exist_ok=True)
    
    if scenario_type == 'earthquake':
        distribution = {'trapped': 0.6, 'survivor': 0.3, 'rescue': 0.1}
    else:
        distribution = {'trapped': 0.1, 'survivor': 0.7, 'rescue': 0.2}
        
    types = []
    for k, v in distribution.items():
        types.extend([k] * int(n_nodes * v))
    if len(types) < n_nodes:
        types.extend(['survivor'] * (n_nodes - len(types)))
        
    nodes = [{'id': i, 'type': types[i], 'mobility': MobilityModel(types[i])} for i in range(n_nodes)]
    
    with open(os.path.join(output_dir, 'node_states.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'node_id', 'x', 'y', 'z', 'speed', 'type'])
        
        # We don't simulate full contacts here, just generate simple states 
        # because the full simulation handles this in DisasterMeshEnv
        for t in range(0, int(duration_hours * 3600), 100):
            for n in nodes:
                n['mobility'].step(t, 100)
                pos = n['mobility'].position
                writer.writerow([t, n['id'], pos[0], pos[1], pos[2], n['mobility'].speed, n['type']])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--duration', type=int, default=24)
    args = parser.parse_args()
    
    generate_disaster_scenario(args.nodes, args.duration, 'earthquake', None)

"""
DisasterMesh Live Visual Dashboard — Real-time mesh simulation visualization.

Flask + WebSocket server that runs the simulation and broadcasts state
to an HTML5 Canvas frontend showing:
    - Animated moving nodes (color = role, size = battery)
    - Dynamic mesh edges (opacity = link quality)
    - Per-node labels (ID, battery%, reputation, speed)
    - Packet routing trails
    - Real-time metrics sidebar
"""

import os
import json
import time
import threading
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

from disaster_mesh_sim.utils.config import load_config
from disaster_mesh_sim.envs.disaster_mesh_env import DisasterMeshEnv
from disaster_mesh_sim.agents.dqn_agent import DQNAgent


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Globals
sim_thread = None
sim_running = False
sim_speed = 1.0  # multiplier


@app.route('/')
def index():
    return render_template('dashboard.html')


def run_simulation():
    """Background thread running the simulation and emitting state."""
    global sim_running
    config = load_config()
    config.simulation.num_nodes = 50  # manageable for viz
    config.simulation.area_width = 800
    config.simulation.area_height = 600
    config.simulation.area_depth = 100
    config.simulation.bt_range = 150
    config.simulation.simulation_duration = 600
    config.training.max_steps_per_episode = 600

    env = DisasterMeshEnv(config=config)
    agent = DQNAgent(config.dqn, config.q_routing)

    # Try loading trained model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "results")
    model_path = os.path.join(results_dir, "best_model.pth")
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            agent.epsilon = 0.05
        except Exception:
            pass

    obs, info = env.reset()
    sim_running = True

    while sim_running:
        # Select action
        k = len(info.get('candidate_set', [0, 1, 2]))
        action = agent.select_action(obs, k)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)

        # Get visualization state
        vis_state = env.get_visualization_state()
        vis_state['reward'] = round(reward, 3)

        # Emit to all connected clients
        socketio.emit('sim_update', vis_state)

        if terminated or truncated:
            obs, info = env.reset()
            socketio.emit('sim_reset', {'message': 'Episode reset'})

        time.sleep(max(0.05, 0.2 / sim_speed))

    env.close()


@socketio.on('connect')
def on_connect():
    print('[Dashboard] Client connected')


@socketio.on('start_sim')
def on_start(data=None):
    global sim_thread, sim_running, sim_speed
    if data and 'speed' in data:
        sim_speed = float(data['speed'])
    if sim_thread is None or not sim_thread.is_alive():
        sim_thread = threading.Thread(target=run_simulation, daemon=True)
        sim_thread.start()


@socketio.on('stop_sim')
def on_stop():
    global sim_running
    sim_running = False


@socketio.on('set_speed')
def on_speed(data):
    global sim_speed
    sim_speed = max(0.1, min(10.0, float(data.get('speed', 1.0))))


def launch_dashboard(host='127.0.0.1', port=5050):
    """Launch the live dashboard server."""
    print(f"\n{'='*60}")
    print(f"  DisasterMesh Live Dashboard")
    print(f"  Open http://{host}:{port} in your browser")
    print(f"{'='*60}\n")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    launch_dashboard()

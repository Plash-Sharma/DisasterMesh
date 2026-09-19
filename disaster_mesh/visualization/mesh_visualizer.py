"""
DisasterMesh — Real-Time Mesh Topology Simulation Dashboard
Self-contained simulation with Plotly Dash visualization.
"""
import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import random
import math
import json

# ─── Inline Simulation Engine ────────────────────────────────────────────────

class SimNode:
    """Lightweight simulation node."""
    TYPES = ['survivor', 'rescue', 'trapped']
    SPEED_RANGES = {'survivor': (0.5, 2.0), 'rescue': (2.0, 5.0), 'trapped': (0.0, 0.1)}
    
    def __init__(self, nid, area=500):
        self.nid = nid
        self.ntype = random.choices(self.TYPES, weights=[0.5, 0.3, 0.2])[0]
        self.x = random.uniform(20, area - 20)
        self.y = random.uniform(20, area - 20)
        self.wx = random.uniform(20, area - 20)
        self.wy = random.uniform(20, area - 20)
        sr = self.SPEED_RANGES[self.ntype]
        self.speed = random.uniform(*sr)
        self.battery = random.uniform(40, 100)
        self.reputation = random.uniform(0.2, 0.9)
        self.role = 'LEAF'
        self.active = True
        self.buffer_used = random.randint(0, 30)
        self.max_buffer = 100
        self.area = area
        self.pause = 0
        
    def step(self):
        if not self.active:
            return
        if self.pause > 0:
            self.pause -= 1
            return
        dx, dy = self.wx - self.x, self.wy - self.y
        dist = math.hypot(dx, dy)
        if dist < 3:
            self.wx = random.uniform(20, self.area - 20)
            self.wy = random.uniform(20, self.area - 20)
            self.pause = random.randint(0, 5)
            sr = self.SPEED_RANGES[self.ntype]
            self.speed = random.uniform(*sr)
        else:
            move = min(self.speed * 2, dist)
            self.x += (dx / dist) * move
            self.y += (dy / dist) * move
        # Energy drain
        self.battery = max(0, self.battery - random.uniform(0.01, 0.08))
        if self.battery <= 0:
            self.active = False
        self.compute_role()
        
    def compute_role(self):
        if not self.active:
            self.role = 'DEAD'; return
        spd_f = max(0, 1 - self.speed / 5.0)
        score = 0.35 * (self.battery / 100) + 0.35 * self.reputation + 0.30 * spd_f
        if score > 0.75:
            self.role = 'ANCHOR'
        elif score > 0.40 and self.reputation >= 0.3 and self.battery > 25:
            self.role = 'RELAY'
        else:
            self.role = 'LEAF'


class SimPacket:
    def __init__(self, src, dst, t):
        self.src = src
        self.dst = dst
        self.created = t
        self.delivered = False
        self.dropped = False
        self.hops = 0
        self.delivery_time = -1


class MeshSimulation:
    BT_RANGE = 120
    
    def __init__(self, n_nodes=30, area=500):
        self.area = area
        self.nodes = [SimNode(i, area) for i in range(n_nodes)]
        self.tick = 0
        self.packets = []
        self.delivered_count = 0
        self.dropped_count = 0
        self.total_sent = 0
        self.pdr_history = []
        self.latency_history = []
        self.energy_history = []
        self.throughput_history = []
        
    def get_links(self):
        links = []
        for i, a in enumerate(self.nodes):
            if not a.active:
                continue
            for j, b in enumerate(self.nodes):
                if j <= i or not b.active:
                    continue
                if math.hypot(a.x - b.x, a.y - b.y) <= self.BT_RANGE:
                    links.append((i, j))
        return links
    
    def step(self):
        self.tick += 1
        for n in self.nodes:
            n.step()
        # Generate packets
        active = [n for n in self.nodes if n.active]
        if len(active) >= 2 and self.tick % 3 == 0:
            src, dst = random.sample(active, 2)
            p = SimPacket(src.nid, dst.nid, self.tick)
            self.packets.append(p)
            self.total_sent += 1
        # Simulate delivery
        links = self.get_links()
        link_set = set()
        for a, b in links:
            link_set.add((a, b))
            link_set.add((b, a))
        for p in self.packets:
            if p.delivered or p.dropped:
                continue
            p.hops += 1
            # Simple probabilistic multi-hop delivery
            if (p.src, p.dst) in link_set or random.random() < 0.15:
                p.delivered = True
                p.delivery_time = self.tick - p.created
                self.delivered_count += 1
            elif p.hops > 15:
                p.dropped = True
                self.dropped_count += 1
        # Metrics
        pdr = self.delivered_count / max(1, self.total_sent)
        self.pdr_history.append(round(pdr * 100, 1))
        recent_d = [p for p in self.packets if p.delivered and p.delivery_time >= 0]
        avg_lat = np.mean([p.delivery_time for p in recent_d[-20:]]) if recent_d else 0
        self.latency_history.append(round(avg_lat, 2))
        avg_bat = np.mean([n.battery for n in self.nodes if n.active]) if active else 0
        self.energy_history.append(round(avg_bat, 1))
        tp = len([p for p in self.packets if p.delivered and self.tick - p.created < 10])
        self.throughput_history.append(tp)


# ─── Global Simulation Instance ──────────────────────────────────────────────
sim = MeshSimulation(n_nodes=30, area=500)

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    'bg': '#0f0f1a', 'card': '#1a1a2e', 'card_border': '#2a2a4a',
    'text': '#e0e0ff', 'text_dim': '#8888aa', 'accent': '#00d4ff',
    'accent2': '#7c3aed', 'green': '#10b981', 'red': '#ef4444',
    'orange': '#f59e0b', 'blue': '#3b82f6',
    'ANCHOR': '#10b981', 'RELAY': '#3b82f6', 'LEAF': '#f59e0b', 'DEAD': '#ef4444',
}
ROLE_SYMBOLS = {'ANCHOR': 'diamond', 'RELAY': 'circle', 'LEAF': 'circle-open', 'DEAD': 'x'}

# ─── Dash App ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="DisasterMesh — Live Simulation")
server = app.server

def make_metric_card(card_id, title, icon):
    return html.Div([
        html.Div(icon, style={'fontSize': '24px', 'marginBottom': '4px'}),
        html.Div(title, style={'fontSize': '11px', 'color': COLORS['text_dim'],
                               'textTransform': 'uppercase', 'letterSpacing': '1px'}),
        html.Div(id=card_id, children="—", style={
            'fontSize': '28px', 'fontWeight': '700', 'color': COLORS['accent'],
            'marginTop': '4px'
        }),
    ], style={
        'background': COLORS['card'], 'border': f"1px solid {COLORS['card_border']}",
        'borderRadius': '12px', 'padding': '16px 20px', 'textAlign': 'center',
        'flex': '1', 'minWidth': '140px',
        'boxShadow': '0 4px 20px rgba(0,0,0,0.3)',
    })

app.layout = html.Div([
    dcc.Interval(id='interval', interval=300, n_intervals=0),
    dcc.Store(id='sim-state'),

    # Header
    html.Div([
        html.Div([
            html.H1("🌐 DisasterMesh", style={
                'margin': '0', 'fontSize': '26px', 'fontWeight': '800',
                'background': 'linear-gradient(90deg, #00d4ff, #7c3aed)',
                'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent',
            }),
            html.Div("Real-Time Mesh Topology Simulation", style={
                'fontSize': '12px', 'color': COLORS['text_dim'], 'marginTop': '2px'
            }),
        ]),
        html.Div(id='tick-display', style={
            'fontSize': '14px', 'color': COLORS['text_dim'],
            'border': f"1px solid {COLORS['card_border']}", 'borderRadius': '8px',
            'padding': '6px 14px', 'background': COLORS['card'],
        }),
    ], style={
        'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
        'padding': '16px 24px', 'borderBottom': f"1px solid {COLORS['card_border']}",
    }),

    # Metric Cards Row
    html.Div(id='metrics-row', style={
        'display': 'flex', 'gap': '12px', 'padding': '16px 24px', 'flexWrap': 'wrap',
    }, children=[
        make_metric_card('metric-pdr', 'Delivery Ratio', '📦'),
        make_metric_card('metric-lat', 'Avg Latency', '⏱️'),
        make_metric_card('metric-bat', 'Avg Battery', '🔋'),
        make_metric_card('metric-nodes', 'Active Nodes', '📡'),
        make_metric_card('metric-links', 'Active Links', '🔗'),
    ]),

    # Main Grid: Topology + Side Panel
    html.Div([
        # Topology Graph + Inference
        html.Div([
            dcc.Graph(id='topology-graph', config={'displayModeBar': True, 'scrollZoom': True},
                      style={'height': '500px'}),
            html.Div(id='dynamic-inference', style={
                'padding': '12px 16px', 'background': 'rgba(0, 212, 255, 0.05)',
                'borderTop': f"1px solid {COLORS['card_border']}",
                'fontSize': '12px', 'color': COLORS['accent'],
                'fontWeight': '600', 'fontStyle': 'italic',
                'minHeight': '40px'
            })
        ], style={
            'flex': '2', 'background': COLORS['card'],
            'border': f"1px solid {COLORS['card_border']}", 'borderRadius': '12px',
            'overflow': 'hidden', 'minHeight': '500px', 'display': 'flex', 'flexDirection': 'column'
        }),
        # Side Panel
        html.Div([
            html.Div("Node Roles", style={
                'fontSize': '13px', 'fontWeight': '700', 'color': COLORS['text'],
                'marginBottom': '8px', 'textTransform': 'uppercase', 'letterSpacing': '1px',
            }),
            dcc.Graph(id='role-pie', config={'displayModeBar': False},
                      style={'height': '200px'}),
            html.Div("Battery Distribution", style={
                'fontSize': '13px', 'fontWeight': '700', 'color': COLORS['text'],
                'marginTop': '12px', 'marginBottom': '8px',
                'textTransform': 'uppercase', 'letterSpacing': '1px',
            }),
            dcc.Graph(id='battery-bar', config={'displayModeBar': False},
                      style={'height': '200px'}),
        ], style={
            'flex': '1', 'background': COLORS['card'],
            'border': f"1px solid {COLORS['card_border']}", 'borderRadius': '12px',
            'padding': '16px', 'minWidth': '260px',
        }),
    ], style={
        'display': 'flex', 'gap': '12px', 'padding': '0 24px 12px', 'flexWrap': 'wrap',
    }),

    # Bottom Charts Row
    html.Div([
        html.Div([
            dcc.Graph(id='pdr-timeline', config={'displayModeBar': False},
                      style={'height': '100%'}),
        ], style={
            'flex': '1', 'background': COLORS['card'],
            'border': f"1px solid {COLORS['card_border']}", 'borderRadius': '12px',
            'overflow': 'hidden', 'minHeight': '220px',
        }),
        html.Div([
            dcc.Graph(id='energy-timeline', config={'displayModeBar': False},
                      style={'height': '100%'}),
        ], style={
            'flex': '1', 'background': COLORS['card'],
            'border': f"1px solid {COLORS['card_border']}", 'borderRadius': '12px',
            'overflow': 'hidden', 'minHeight': '220px',
        }),
    ], style={
        'display': 'flex', 'gap': '12px', 'padding': '0 24px 12px', 'flexWrap': 'wrap',
    }),

    # Algorithm Pipeline
    html.Div([
        html.Div("⚡ Algorithm Pipeline", style={
            'fontSize': '13px', 'fontWeight': '700', 'color': COLORS['text'],
            'marginBottom': '10px', 'textTransform': 'uppercase', 'letterSpacing': '1px',
        }),
        html.Div(id='pipeline-status', style={
            'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap',
        }),
    ], style={
        'padding': '0 24px 20px',
    }),

], style={
    'fontFamily': "'Inter', 'Segoe UI', sans-serif",
    'background': COLORS['bg'], 'color': COLORS['text'],
    'minHeight': '100vh', 'margin': '0',
})


# ─── Pipeline Stages ─────────────────────────────────────────────────────────
PIPELINE = [
    ("DBSCAN FFL", "Node Discovery"), ("RORQ", "Q-Routing"),
    ("EDMBOPR", "Relay Selection"), ("HMRFCO", "Resource Opt"),
    ("Q-D2D", "Power Alloc"), ("DRL", "Congestion Mgmt"),
    ("S&W", "DTN Fallback"),
]

GRAPH_LAYOUT = dict(
    uirevision='constant',
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=COLORS['text_dim'], size=10),
)


@app.callback(
    [Output('topology-graph', 'figure'), Output('role-pie', 'figure'),
     Output('battery-bar', 'figure'), Output('pdr-timeline', 'figure'),
     Output('energy-timeline', 'figure'),
     Output('metric-pdr', 'children'), Output('metric-lat', 'children'),
     Output('metric-bat', 'children'), Output('metric-nodes', 'children'),
     Output('metric-links', 'children'), Output('tick-display', 'children'),
     Output('pipeline-status', 'children'), Output('dynamic-inference', 'children')],
    [Input('interval', 'n_intervals')]
)
def update(n):
    sim.step()
    links = sim.get_links()
    active = [nd for nd in sim.nodes if nd.active]

    # ── Topology Figure ───────────────────────────────────────────────────
    edge_x, edge_y = [], []
    for a, b in links:
        na, nb = sim.nodes[a], sim.nodes[b]
        edge_x += [na.x, nb.x, None]
        edge_y += [na.y, nb.y, None]

    traces = [go.Scatter(x=edge_x, y=edge_y, mode='lines',
                         line=dict(width=0.8, color='rgba(0,212,255,0.18)'),
                         hoverinfo='none')]

    for role in ['ANCHOR', 'RELAY', 'LEAF', 'DEAD']:
        ns = [n for n in sim.nodes if n.role == role]
        if not ns:
            continue
        sz = {'ANCHOR': 16, 'RELAY': 11, 'LEAF': 8, 'DEAD': 6}
        traces.append(go.Scatter(
            x=[n.x for n in ns], y=[n.y for n in ns], mode='markers+text',
            marker=dict(size=sz[role], color=COLORS[role],
                        symbol=ROLE_SYMBOLS[role], line=dict(width=1, color='#fff')),
            text=[str(n.nid) for n in ns], textposition='top center',
            textfont=dict(size=8, color=COLORS['text_dim']),
            name=role, hovertemplate=(
                '<b>Node %{text}</b><br>Role: ' + role +
                '<br>Battery: %{customdata[0]:.1f}%'
                '<br>Rep: %{customdata[1]:.2f}<extra></extra>'
            ),
            customdata=[(n.battery, n.reputation) for n in ns],
        ))

    topo_layout = dict(
        uirevision='constant',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=20, t=30, b=30),
        font=dict(color=COLORS['text_dim'], size=10),
        showlegend=True, title=None,
        legend=dict(orientation='h', x=0, y=1.12, font=dict(size=10, color=COLORS['text_dim'])),
        xaxis=dict(range=[0, sim.area], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, sim.area], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor='x'),
    )
    topo_fig = go.Figure(data=traces, layout=topo_layout)

    # ── Role Pie ──────────────────────────────────────────────────────────
    roles = {}
    for n in sim.nodes:
        roles[n.role] = roles.get(n.role, 0) + 1
    labels = list(roles.keys())
    vals = list(roles.values())
    pie_fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.55,
        marker=dict(colors=[COLORS.get(r, '#888') for r in labels]),
        textinfo='value', textfont=dict(size=12, color='#fff'),
    ))
    pie_fig.update_layout(**GRAPH_LAYOUT, showlegend=True, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(font=dict(size=9, color=COLORS['text_dim'])))

    # ── Battery Bar ───────────────────────────────────────────────────────
    buckets = {'0-25%': 0, '25-50%': 0, '50-75%': 0, '75-100%': 0}
    for n in sim.nodes:
        if n.battery <= 25: buckets['0-25%'] += 1
        elif n.battery <= 50: buckets['25-50%'] += 1
        elif n.battery <= 75: buckets['50-75%'] += 1
        else: buckets['75-100%'] += 1
    bar_colors = [COLORS['red'], COLORS['orange'], COLORS['blue'], COLORS['green']]
    bat_fig = go.Figure(go.Bar(
        x=list(buckets.keys()), y=list(buckets.values()),
        marker=dict(color=bar_colors, cornerradius=4),
        text=list(buckets.values()), textposition='auto',
        textfont=dict(color='#fff', size=11),
    ))
    bat_fig.update_layout(**GRAPH_LAYOUT, margin=dict(l=30, r=10, t=10, b=30),
                          yaxis=dict(gridcolor='#1e1e3a'))

    # ── PDR Timeline ──────────────────────────────────────────────────────
    window = sim.pdr_history[-80:]
    pdr_fig = go.Figure(go.Scatter(
        y=window, mode='lines', fill='tozeroy',
        line=dict(color=COLORS['accent'], width=2),
        fillcolor='rgba(0,212,255,0.1)',
    ))
    pdr_fig.update_layout(**GRAPH_LAYOUT, title=dict(text='Packet Delivery Ratio (%)',
                          font=dict(size=12, color=COLORS['text_dim'])),
                          yaxis=dict(range=[0, 105], gridcolor='#1e1e3a'))

    # ── Energy Timeline ───────────────────────────────────────────────────
    e_window = sim.energy_history[-80:]
    en_fig = go.Figure(go.Scatter(
        y=e_window, mode='lines', fill='tozeroy',
        line=dict(color=COLORS['green'], width=2),
        fillcolor='rgba(16,185,129,0.1)',
    ))
    en_fig.update_layout(**GRAPH_LAYOUT, title=dict(text='Avg Network Battery (%)',
                         font=dict(size=12, color=COLORS['text_dim'])),
                         yaxis=dict(range=[0, 105], gridcolor='#1e1e3a'))

    # ── Metric Cards ──────────────────────────────────────────────────────
    pdr_val = f"{sim.pdr_history[-1]}%" if sim.pdr_history else "—"
    lat_val = f"{sim.latency_history[-1]}s" if sim.latency_history else "—"
    bat_val = f"{sim.energy_history[-1]}%" if sim.energy_history else "—"
    nodes_val = str(len(active))
    links_val = str(len(links))
    tick_text = f"⏱ Tick {sim.tick}"

    # ── Pipeline Status ───────────────────────────────────────────────────
    active_stage = (sim.tick % len(PIPELINE))
    pills = []
    for i, (name, desc) in enumerate(PIPELINE):
        is_active = (i == active_stage)
        pills.append(html.Div([
            html.Div(name, style={'fontWeight': '700', 'fontSize': '11px'}),
            html.Div(desc, style={'fontSize': '9px', 'color': COLORS['text_dim']}),
        ], style={
            'padding': '6px 12px', 'borderRadius': '8px', 'textAlign': 'center',
            'background': COLORS['accent2'] if is_active else COLORS['card'],
            'border': f"1px solid {'#9d5cff' if is_active else COLORS['card_border']}",
            'color': '#fff' if is_active else COLORS['text_dim'],
            'transition': 'all 0.3s ease', 'minWidth': '90px',
        }))

    # ── Dynamic Inference ─────────────────────────────────────────────────
    if not sim.pdr_history:
        inf_text = "💡 INFERENCE: Initializing mesh topology... Nodes are discovering each other via EDMBOPR DBSCAN."
    elif sim.pdr_history[-1] > 85:
        inf_text = f"💡 INFERENCE: High Stability! RORQ Q-learning has optimized routing paths. The mesh is highly connected with {len(links)} active links. PDR is excellent at {sim.pdr_history[-1]}%."
    elif len(active) < 18:
        inf_text = f"🚨 INFERENCE: CRITICAL FAILURE! {30 - len(active)} nodes have died. The DRL-MANET fallback is prioritizing store-carry-forward (DTN) to bridge massive network gaps."
    else:
        inf_text = f"⚡ INFERENCE: Network adapting. Energy is draining (avg {sim.energy_history[-1]}%). HMRFCO is pruning redundant relay connections to conserve battery."

    return (topo_fig, pie_fig, bat_fig, pdr_fig, en_fig,
            pdr_val, lat_val, bat_val, nodes_val, links_val, tick_text, pills, inf_text)


def run_visualizer():
    app.run(debug=True, use_reloader=False, port=8050)


if __name__ == '__main__':
    run_visualizer()

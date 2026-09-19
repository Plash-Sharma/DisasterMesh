"""
Network topology generation and management
"""
import networkx as nx

class NetworkTopology:
    def __init__(self):
        self.graph = nx.Graph()
        
    def update_edges(self, nodes):
        self.graph.clear()
        for node in nodes:
            if node.active:
                self.graph.add_node(node.node_id, pos=node.position)
                
        for node in nodes:
            if not node.active: continue
            for neighbor in node.neighbors:
                if neighbor.active:
                    self.graph.add_edge(node.node_id, neighbor.node_id)

"""
EDMBOPR DBSCAN FFL (Eq. 12, Algorithm 2)
Ref: Das & Devi, P2P Netw Apps 2025
"""
from sklearn.cluster import DBSCAN
import numpy as np

def build_ffl(node, neighbors):
    """
    Build First Forwarder List using DBSCAN.
    Includes only neighbors with energy > 25% and in the same cluster.
    """
    eligible_neighbors = [n for n in neighbors if n.battery > 25.0]
    if not eligible_neighbors:
        return []
        
    points = [node.position] + [n.position for n in eligible_neighbors]
    
    clustering = DBSCAN(eps=250.0, min_samples=2, metric='euclidean').fit(points)
    
    node_label = clustering.labels_[0]
    
    ffl = []
    if node_label != -1:
        for i, n in enumerate(eligible_neighbors):
            if clustering.labels_[i+1] == node_label:
                ffl.append(n)
                
    return ffl

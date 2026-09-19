"""
WAFL — Wireless Ad-hoc Federated Learning for DisasterMesh.

Enables collective intelligence by aggregating local DQN model updates
across nodes without sharing raw data, using compressed gradients over BT.

References:
    [1] McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017
    [2] Yamashita et al., "Detection of Global Anomalies on Distributed
        IoT Edges with D2D Communication", UbiComp 2024
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from disaster_mesh_sim.utils.config import FederatedConfig


class WirelessFederatedLearner:
    """
    WAFL implementation for DisasterMesh.

    Protocol per FL round:
    1. Each node trains local DQN for E local epochs
    2. Node computes compressed gradient delta: ΔW = W_local − W_global
    3. Super Node aggregates via FedAvg: W_global = Σ(n_i/N · W_i)
    4. Super Node broadcasts W_global
    5. Each node updates its local model

    Bandwidth optimizations:
    - 8-bit gradient quantization (4x compression)
    - Sparse top-k% gradient transmission
    """

    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model: Optional[nn.Module] = None
        self.round_num = 0

    def init_global_model(self, model: nn.Module):
        """Initialize the global model (typically done by Super Node)."""
        self.global_model = copy.deepcopy(model)

    def compute_gradient_delta(self, local_model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute compressed gradient delta between local and global model.

        Returns:
            Dictionary of quantized gradient deltas per parameter.
        """
        if self.global_model is None:
            return {}

        delta = {}
        global_params = dict(self.global_model.named_parameters())

        for name, local_param in local_model.named_parameters():
            diff = local_param.data - global_params[name].data

            # Sparse top-k: zero out small gradients
            diff_flat = diff.flatten()
            k = max(1, int(len(diff_flat) * self.config.sparse_top_k_percent / 100))
            topk_vals, topk_idx = torch.topk(diff_flat.abs(), k)
            sparse_diff = torch.zeros_like(diff_flat)
            sparse_diff[topk_idx] = diff_flat[topk_idx]
            sparse_diff = sparse_diff.view(diff.shape)

            # Quantize to 8-bit
            delta[name] = self._quantize(sparse_diff, self.config.gradient_quantization)

        return delta

    def aggregate_updates(self, deltas: List[Dict[str, torch.Tensor]],
                          weights: Optional[List[float]] = None) -> bool:
        """
        FedAvg aggregation of gradient deltas from multiple nodes.

        Args:
            deltas: List of gradient delta dicts from each participating node.
            weights: Optional per-node weighting (default: uniform).

        Returns:
            True if aggregation succeeded.
        """
        if self.global_model is None or not deltas:
            return False

        n = len(deltas)
        if n < self.config.min_nodes_for_fl:
            return False

        if weights is None:
            weights = [1.0 / n] * n

        for name, param in self.global_model.named_parameters():
            avg_delta = torch.zeros_like(param.data)
            for i, d in enumerate(deltas):
                if name in d:
                    # Dequantize
                    dequant = self._dequantize(d[name])
                    avg_delta += weights[i] * dequant
            param.data += avg_delta

        self.round_num += 1
        return True

    def distribute_global_model(self, local_model: nn.Module):
        """Update a local model with the current global model weights."""
        if self.global_model is not None:
            local_model.load_state_dict(self.global_model.state_dict())

    @staticmethod
    def _quantize(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """Quantize tensor to n-bit representation."""
        if bits >= 32:
            return tensor

        t_min = tensor.min()
        t_max = tensor.max()
        t_range = t_max - t_min

        if t_range < 1e-10:
            return torch.zeros_like(tensor)

        levels = 2 ** bits - 1
        normalized = (tensor - t_min) / t_range
        quantized = torch.round(normalized * levels) / levels
        return quantized * t_range + t_min

    @staticmethod
    def _dequantize(tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize (identity for our simplified scheme)."""
        return tensor

    def get_communication_cost(self, model: nn.Module) -> int:
        """Estimate bytes needed to transmit gradient delta over BT."""
        total_params = sum(p.numel() for p in model.parameters())
        active_params = int(total_params * self.config.sparse_top_k_percent / 100)
        bytes_per_param = self.config.gradient_quantization / 8
        return int(active_params * bytes_per_param)

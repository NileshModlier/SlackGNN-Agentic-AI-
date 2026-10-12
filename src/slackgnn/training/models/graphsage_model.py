"""Two-layer GraphSAGE network for node-level timing prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(nn.Module):
    """Predict one value per node with mean neighborhood aggregation."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(x, edge_index))
        hidden = torch.nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        return self.conv2(hidden, edge_index).view(-1)

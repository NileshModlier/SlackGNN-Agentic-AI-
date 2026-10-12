"""Two-layer graph isomorphism network for node-level timing prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv


class GINModel(nn.Module):
    """Predict one value per node with sum-aggregated MLP updates."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv1 = GINConv(_mlp(in_channels, hidden_channels))
        self.conv2 = GINConv(_mlp(hidden_channels, hidden_channels))
        self.readout = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.conv1(x, edge_index))
        hidden = torch.nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        hidden = torch.relu(self.conv2(hidden, edge_index))
        hidden = torch.nn.functional.dropout(hidden, p=self.dropout, training=self.training)
        return self.readout(hidden).view(-1)


def _mlp(in_channels: int, hidden_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_channels, hidden_channels),
        nn.ReLU(),
        nn.Linear(hidden_channels, hidden_channels),
    )

"""Timing-aware graph network for slack and violation prediction.

Message passing follows the signal-flow direction. Each load aggregates its
drivers with a learned attention weight. The weight sees timing criticality,
reported-path membership, and a learned slack-magnitude gate. Ground-truth
slack is reserved for the loss.
"""

from __future__ import annotations

import torch
from torch import nn


class SlackGNN(nn.Module):
    """Encode structure and timing context, then predict a violation logit and a slack."""

    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        timing_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.25,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("SlackGNN needs at least one message-passing layer")
        self.node_encoder = nn.Sequential(nn.Linear(node_channels, hidden_channels), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_channels, hidden_channels), nn.ReLU())
        self.timing_encoder = nn.Sequential(nn.Linear(timing_channels, hidden_channels), nn.ReLU())
        self.layers = nn.ModuleList(
            TimingSAGELayer(hidden_channels, hidden_channels, hidden_channels) for _ in range(num_layers)
        )
        self.classification_head = nn.Linear(hidden_channels, 1)
        self.regression_head = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        timing_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return violation logits, slack, incoming attention, and node embeddings."""
        timing = self.timing_encoder(timing_features)
        hidden = self.node_encoder(node_features) + timing
        encoded_edges = self.edge_encoder(edge_attr)
        attention = node_features.new_zeros(edge_index.size(1))
        for layer in self.layers:
            hidden = torch.nn.functional.dropout(hidden, p=self.dropout, training=self.training)
            hidden, attention = layer(hidden, edge_index, encoded_edges, timing)
        return (
            self.classification_head(hidden).view(-1),
            self.regression_head(hidden).view(-1),
            attention,
            hidden,
        )


class TimingSAGELayer(nn.Module):
    """Mean-style GraphSAGE update with a timing attention weight on each incoming driver."""

    def __init__(self, hidden_channels: int, edge_channels: int, timing_channels: int) -> None:
        super().__init__()
        self.message = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.root = nn.Linear(hidden_channels, hidden_channels)
        self.slack_gate = nn.Linear(hidden_channels, 1)
        score_channels = hidden_channels * 2 + edge_channels + timing_channels * 2 + 2
        self.score = nn.Linear(score_channels, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        timing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        slack_magnitude = self.slack_gate(hidden).abs()
        pair = torch.cat(
            [
                hidden[source],
                hidden[target],
                edge_attr,
                timing[source],
                timing[target],
                slack_magnitude[source],
                slack_magnitude[target],
            ],
            dim=1,
        )
        attention = grouped_softmax(self.score(pair).view(-1), target, hidden.size(0))
        messages = self.message(hidden)[source] * attention.unsqueeze(1)
        aggregated = hidden.new_zeros(hidden.shape)
        aggregated.index_add_(0, target, messages)
        updated = torch.relu(self.root(hidden) + aggregated)
        return updated, attention


def grouped_softmax(scores: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Softmax of ``scores`` inside the group of edges that share a target node."""
    maximum = torch.full((size,), torch.finfo(scores.dtype).min, device=scores.device)
    maximum.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    shifted = (scores - maximum[index]).exp()
    total = scores.new_zeros(size)
    total.index_add_(0, index, shifted)
    return shifted / total[index].clamp_min(1e-12)

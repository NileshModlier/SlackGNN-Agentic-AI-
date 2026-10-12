"""SlackGNN ablations that wrap the published model without editing it."""

from __future__ import annotations

import torch
from torch import nn

from slackgnn.training.models.slackgnn import SlackGNN


class AblatedSlackGNN(SlackGNN):
    """SlackGNN with optional timing, edge, and attention removals.

    ``drop_timing`` zeros the timing-context encoding. ``drop_edges`` zeros
    the edge encoding. ``uniform_attention`` replaces the learned incoming
    weights with the mean of the unique driver edges into each load.
    """

    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        timing_channels: int,
        hidden_channels: int = 32,
        dropout: float = 0.25,
        num_layers: int = 2,
        *,
        drop_timing: bool = False,
        drop_edges: bool = False,
        uniform_attention: bool = False,
    ) -> None:
        super().__init__(
            node_channels,
            edge_channels,
            timing_channels,
            hidden_channels,
            dropout,
            num_layers,
        )
        self.drop_timing = drop_timing
        self.drop_edges = drop_edges
        self.uniform_attention = uniform_attention

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        timing_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        timing = self.timing_encoder(timing_features)
        if self.drop_timing:
            timing = torch.zeros_like(timing)
        hidden = self.node_encoder(node_features) + timing
        encoded_edges = self.edge_encoder(edge_attr)
        if self.drop_edges:
            encoded_edges = torch.zeros_like(encoded_edges)
        attention = node_features.new_zeros(edge_index.size(1))
        for layer in self.layers:
            hidden = torch.nn.functional.dropout(hidden, p=self.dropout, training=self.training)
            if self.uniform_attention:
                hidden, attention = uniform_neighbor_mean(layer, hidden, edge_index)
            else:
                hidden, attention = layer(hidden, edge_index, encoded_edges, timing)
        return (
            self.classification_head(hidden).view(-1),
            self.regression_head(hidden).view(-1),
            attention,
            hidden,
        )


def uniform_neighbor_mean(
    layer: nn.Module,
    hidden: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average incoming driver messages with equal weight per edge."""
    source, target = edge_index
    degree = torch.bincount(target, minlength=hidden.size(0)).to(hidden.dtype).clamp_min(1)
    attention = 1.0 / degree[target]
    messages = layer.message(hidden)[source] * attention.unsqueeze(1)
    aggregated = hidden.new_zeros(hidden.shape)
    aggregated.index_add_(0, target, messages)
    updated = torch.relu(layer.root(hidden) + aggregated)
    return updated, attention


ABLATIONS = (
    {
        "name": "full",
        "description": "Timing context, edge features, learned attention, and both losses.",
        "drop_timing": False,
        "drop_edges": False,
        "uniform_attention": False,
        "task": "multitask",
    },
    {
        "name": "no_timing_context",
        "description": "Timing-context channels are zeroed before message passing.",
        "drop_timing": True,
        "drop_edges": False,
        "uniform_attention": False,
        "task": "multitask",
    },
    {
        "name": "no_edge_features",
        "description": "Edge channels are zeroed. Attention still uses node and timing state.",
        "drop_timing": False,
        "drop_edges": True,
        "uniform_attention": False,
        "task": "multitask",
    },
    {
        "name": "no_attention",
        "description": "Incoming drivers are averaged with equal weight.",
        "drop_timing": False,
        "drop_edges": False,
        "uniform_attention": True,
        "task": "multitask",
    },
    {
        "name": "classification_only",
        "description": "Only the violation loss is trained. The slack head is not supervised.",
        "drop_timing": False,
        "drop_edges": False,
        "uniform_attention": False,
        "task": "classification",
    },
    {
        "name": "regression_only",
        "description": "Only the slack loss is trained. The violation head is not supervised.",
        "drop_timing": False,
        "drop_edges": False,
        "uniform_attention": False,
        "task": "regression",
    },
)

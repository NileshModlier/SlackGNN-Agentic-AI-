"""Baseline node-prediction models."""

from slackgnn.training.models.gcn_model import GCNModel
from slackgnn.training.models.gin_model import GINModel
from slackgnn.training.models.graphsage_model import GraphSAGEModel
from slackgnn.training.models.slackgnn import SlackGNN

MODEL_NAMES = ("gcn", "graphsage", "gin")


def build_model(
    name: str,
    in_channels: int,
    hidden_channels: int = 32,
    dropout: float = 0.5,
    edge_channels: int | None = None,
    timing_channels: int | None = None,
):
    """Return a baseline model, or SlackGNN when edge and timing widths are provided."""
    key = name.lower()
    if key == "gcn":
        return GCNModel(in_channels, hidden_channels, dropout)
    if key == "graphsage":
        return GraphSAGEModel(in_channels, hidden_channels, dropout)
    if key == "gin":
        return GINModel(in_channels, hidden_channels, dropout)
    if key == "slackgnn":
        if edge_channels is None or timing_channels is None:
            raise ValueError("SlackGNN requires edge_channels and timing_channels")
        return SlackGNN(in_channels, edge_channels, timing_channels, hidden_channels, dropout)
    known = ", ".join(MODEL_NAMES + ("slackgnn",))
    raise ValueError(f"Unknown model {name!r}. Expected one of: {known}.")

"""Smoke tests for SlackGNN ablation flags."""

from __future__ import annotations

import torch

from slackgnn.m9.ablation import AblatedSlackGNN


def test_uniform_attention_is_inverse_degree() -> None:
    torch.manual_seed(0)
    model = AblatedSlackGNN(4, 2, 3, hidden_channels=8, dropout=0.0, uniform_attention=True)
    model.eval()
    node_features = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    edge_attr = torch.randn(3, 2)
    timing = torch.randn(4, 3)

    _logits, _slack, attention, _hidden = model(node_features, edge_index, edge_attr, timing)

    assert torch.allclose(attention[0], torch.tensor(1.0))
    assert torch.allclose(attention[1], torch.tensor(0.5))
    assert torch.allclose(attention[2], torch.tensor(0.5))


def test_dropping_timing_context_changes_the_logit() -> None:
    torch.manual_seed(0)
    full = AblatedSlackGNN(4, 2, 3, hidden_channels=8, dropout=0.0)
    ablated = AblatedSlackGNN(4, 2, 3, hidden_channels=8, dropout=0.0, drop_timing=True)
    with torch.no_grad():
        full.timing_encoder[0].bias.fill_(1.0)
    ablated.load_state_dict(full.state_dict())
    full.eval()
    ablated.eval()
    node_features = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_attr = torch.randn(4, 2)
    timing = torch.randn(5, 3)

    full_logits, _, _, _ = full(node_features, edge_index, edge_attr, timing)
    ablated_logits, _, _, _ = ablated(node_features, edge_index, edge_attr, timing)

    assert not torch.allclose(full_logits, ablated_logits)

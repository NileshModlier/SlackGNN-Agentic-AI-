"""Train, validation, and test splits over labeled endpoints."""

from __future__ import annotations

import random

from torch_geometric.data import Data

Pair = tuple[int, int]


def labeled_pairs(graphs: list[Data]) -> list[Pair]:
    """Return ``(graph_index, node_index)`` for every labeled endpoint."""
    pairs: list[Pair] = []
    for graph_index, data in enumerate(graphs):
        nodes = data.label_mask.nonzero(as_tuple=False).view(-1).tolist()
        pairs.extend((graph_index, int(node)) for node in nodes)
    return pairs


def violation_of(graphs: list[Data], pair: Pair) -> int:
    """Return 1 when the endpoint slack is a timing violation."""
    graph_index, node_index = pair
    return int(graphs[graph_index].y_violation[node_index].item())


def random_split(
    graphs: list[Data],
    seed: int = 0,
) -> dict[str, list[Pair]]:
    """Split labeled endpoints into train, validation, and test.

    Each class keeps at least one training example when it has more than one
    labeled endpoint. The leftover examples are divided between validation
    and test.
    """
    rng = random.Random(seed)
    grouped = _group_by_class(graphs, labeled_pairs(graphs), rng)
    train: list[Pair] = []
    validation: list[Pair] = []
    test: list[Pair] = []
    for items in grouped.values():
        if len(items) <= 1:
            train.extend(items)
            continue
        test_count = 1 if len(items) < 4 else max(1, round(len(items) * 0.15))
        validation_count = 1 if len(items) - test_count >= 2 else 0
        test.extend(items[:test_count])
        validation.extend(items[test_count : test_count + validation_count])
        train.extend(items[test_count + validation_count :])
    return {"train": train, "validation": validation, "test": test}


def leave_one_design_out(
    graphs: list[Data],
    seed: int = 0,
) -> list[dict[str, object]]:
    """Rotate each design into the test set and split the others for training."""
    folds: list[dict[str, object]] = []
    for held_out, _data in enumerate(graphs):
        rng = random.Random(seed + held_out)
        remaining = [pair for pair in labeled_pairs(graphs) if pair[0] != held_out]
        grouped = _group_by_class(graphs, remaining, rng)
        train: list[Pair] = []
        validation: list[Pair] = []
        for items in grouped.values():
            if len(items) <= 1:
                train.extend(items)
                continue
            validation.append(items[0])
            train.extend(items[1:])
        test = [pair for pair in labeled_pairs(graphs) if pair[0] == held_out]
        folds.append(
            {
                "held_out": str(graphs[held_out].design_name),
                "train": train,
                "validation": validation,
                "test": test,
            }
        )
    return folds


def _group_by_class(graphs: list[Data], pairs: list[Pair], rng: random.Random) -> dict[int, list[Pair]]:
    grouped: dict[int, list[Pair]] = {0: [], 1: []}
    for pair in pairs:
        grouped[violation_of(graphs, pair)].append(pair)
    for items in grouped.values():
        rng.shuffle(items)
    return grouped

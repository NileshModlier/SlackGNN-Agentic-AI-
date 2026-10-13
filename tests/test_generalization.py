"""Ranking checks for the cross-design generalization report."""

from slackgnn.m9.generalization import rank_generalization, render_generalization_markdown


def _fold(held_out: str, f1: float, mae: float) -> dict[str, object]:
    endpoint = {
        "accuracy": f1,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "roc_auc": f1,
        "mae": mae,
        "mse": mae,
        "rmse": mae,
        "support": 4,
    }
    return {"held_out": held_out, "endpoint": endpoint, "expanded": endpoint}


def test_ranking_names_the_easiest_hardest_and_best_architecture() -> None:
    models = {
        "gcn": {
            "folds": [_fold("aes", 0.2, 1.5), _fold("gcd", 0.8, 0.4)],
            "summary": {"endpoint": {"f1": {"mean": 0.5}, "mae": {"mean": 0.95}}},
        },
        "slackgnn": {
            "folds": [_fold("aes", 0.4, 1.0), _fold("gcd", 1.0, 0.2)],
            "summary": {"endpoint": {"f1": {"mean": 0.7}, "mae": {"mean": 0.6}}},
        },
    }

    ranking = rank_generalization(models)

    assert ranking["easiest_design"] == "gcd"
    assert ranking["hardest_design"] == "aes"
    assert ranking["most_transferable_architecture"] == "slackgnn"
    assert ranking["slackgnn_endpoint_f1_above_every_baseline"] is True
    assert ranking["slackgnn_endpoint_mae_below_every_baseline"] is True


def test_markdown_lists_every_holdout_and_architecture() -> None:
    models = {
        "gcn": {
            "folds": [_fold("aes", 0.2, 1.5)],
            "summary": {"endpoint": {"f1": {"mean": 0.2}, "mae": {"mean": 1.5}}, "expanded": {"f1": {"mean": 0.2}, "mae": {"mean": 1.5}}},
        },
        "graphsage": {
            "folds": [_fold("aes", 0.3, 1.2)],
            "summary": {"endpoint": {"f1": {"mean": 0.3}, "mae": {"mean": 1.2}}, "expanded": {"f1": {"mean": 0.3}, "mae": {"mean": 1.2}}},
        },
        "gin": {
            "folds": [_fold("aes", 0.1, 1.4)],
            "summary": {"endpoint": {"f1": {"mean": 0.1}, "mae": {"mean": 1.4}}, "expanded": {"f1": {"mean": 0.1}, "mae": {"mean": 1.4}}},
        },
        "slackgnn": {
            "folds": [_fold("aes", 0.5, 0.8)],
            "summary": {"endpoint": {"f1": {"mean": 0.5}, "mae": {"mean": 0.8}}, "expanded": {"f1": {"mean": 0.5}, "mae": {"mean": 0.8}}},
        },
    }
    payload = {
        "designs": ["aes", "gcd"],
        "seed": 0,
        "hidden_channels": 32,
        "epochs": 20,
        "note": "Endpoint scores use published slack.",
        "models": models,
        "ranking": rank_generalization(models),
    }

    text = render_generalization_markdown(payload)

    assert "aes" in text
    assert "slackgnn" in text
    assert "Most transferable architecture" in text

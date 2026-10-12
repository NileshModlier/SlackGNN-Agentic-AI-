"""Classification and regression metrics for the labeled endpoints."""

from __future__ import annotations

import math


def classification_metrics(labels: list[int], probabilities: list[float], threshold: float = 0.5) -> dict[str, float | None]:
    """Score binary violation predictions. Class 1 is a timing violation."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be the same non-empty length")
    predictions = [1 if score >= threshold else 0 for score in probabilities]
    tp = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred == 1)
    fp = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred == 0)
    tn = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 0 and pred == 0)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(labels, probabilities),
        "support": len(labels),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    """Score slack predictions in nanoseconds."""
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted slack must be the same non-empty length")
    errors = [pred - value for value, pred in zip(actual, predicted, strict=True)]
    mse = sum(error * error for error in errors) / len(errors)
    mae = sum(abs(error) for error in errors) / len(errors)
    return {"mae": mae, "mse": mse, "rmse": math.sqrt(mse), "support": len(actual)}


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """Return the Mann-Whitney ROC-AUC, or None when a class is missing."""
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def roc_curve(labels: list[int], scores: list[float]) -> tuple[list[float], list[float]]:
    """Return false-positive and true-positive rates from high score to low score."""
    if not labels:
        return [0.0, 1.0], [0.0, 1.0]
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0]
    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0
    for index in order:
        if labels[index] == 1:
            tp += 1
        else:
            fp += 1
        fpr.append(fp / negatives)
        tpr.append(tp / positives)
    return fpr, tpr


def mean_metric(rows: list[dict[str, object]], key: str) -> float | None:
    """Average a metric across folds, skipping folds where it is undefined."""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)

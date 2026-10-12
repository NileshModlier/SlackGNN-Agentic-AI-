"""Training and evaluation for timing-violation prediction.

This package reads the finished PyG graphs. It does not rebuild them.
"""

from slackgnn.training.dataset_builder import build_training_dataset
from slackgnn.training.validate_corpus import validate_corpus

__all__ = ["build_training_dataset", "validate_corpus"]

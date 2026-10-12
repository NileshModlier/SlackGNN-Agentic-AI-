"""Structural and timing features for SlackGNN.

Feature construction reads the published graphs and path tables. It does not
rebuild the dataset pipeline.
"""

from slackgnn.features.feature_builder import enrich_corpus, enrich_design

__all__ = ["enrich_corpus", "enrich_design"]

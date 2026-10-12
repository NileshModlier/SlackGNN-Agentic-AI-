"""Dataset preparation for SlackGNN.

This package copies design artifacts, parses OpenROAD timing reports,
builds a gate-level signal-flow graph, attaches path slack, writes
one PyTorch Geometric graph per design, and collects those graphs
into a multi-design corpus. It does not train models.
"""

from slackgnn.dataset.gcd_import import import_gcd_dataset

__all__ = ["import_gcd_dataset"]

"""Inez's versioned chess data and optional neural model interfaces.

Importing this package does not import PyTorch. The rules engine and the
encoding contract can be used without the optional training dependencies.
"""

from .encoding import CONTRACT_VERSION, FEATURE_COUNT, HISTORY_FRAMES, POLICY_SIZE

__all__ = ["CONTRACT_VERSION", "FEATURE_COUNT", "HISTORY_FRAMES", "POLICY_SIZE"]

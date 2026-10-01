"""Adversarial stress-test generators for HydroJEV."""

from .concealment_ae import (
    ConcealmentAutoencoder,
    ReconstructionDetector,
    coordinate_descent_concealment,
)

__all__ = [
    "ConcealmentAutoencoder",
    "ReconstructionDetector",
    "coordinate_descent_concealment",
]


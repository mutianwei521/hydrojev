"""Hydraulic distance, partitioning, and surrogate-model components."""

from .cpdz_clustering import CPDZPartition, cluster_cpdz
from .distance_metrics import PhysicalDistanceMatrices, build_physical_distances

__all__ = [
    "CPDZPartition",
    "PhysicalDistanceMatrices",
    "build_physical_distances",
    "cluster_cpdz",
]


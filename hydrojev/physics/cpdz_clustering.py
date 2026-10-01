"""Connected single-link hierarchical clustering for cyber-physical zones."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import networkx as nx

from hydrojev.datasets.wdn_loader import NetworkBundle
from hydrojev.physics.distance_metrics import (
    PhysicalDistanceMatrices,
    candidate_external_cut_size,
    combine_normalized_distances,
    network_graph,
)


@dataclass(frozen=True)
class MergeStep:
    left: tuple[str, ...]
    right: tuple[str, ...]
    merged: tuple[str, ...]
    distance: float
    topology: float
    headloss: float
    delay: float


@dataclass(frozen=True)
class CPDZPartition:
    clusters: tuple[tuple[str, ...], ...]
    protected_nodes: tuple[str, ...]
    boundary_links: dict[tuple[str, ...], tuple[str, ...]]
    merge_history: tuple[MergeStep, ...]
    threshold: float


def _clusters_connected(graph: nx.Graph, left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    right_set = set(right)
    return any(neighbor in right_set for node in left for neighbor in graph.neighbors(node))


def _cross_edge_pairs(
    graph: nx.Graph, left: tuple[str, ...], right: tuple[str, ...]
) -> list[tuple[str, str]]:
    right_set = set(right)
    return [
        (node, neighbor)
        for node in left
        for neighbor in graph.neighbors(node)
        if neighbor in right_set
    ]


def _cluster_distance(
    graph: nx.Graph,
    distances: PhysicalDistanceMatrices,
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> tuple[float, float, float, float]:
    if len(left) == 1 and len(right) == 1:
        first, second = left[0], right[0]
        return (
            float(distances.combined.loc[first, second]),
            float(distances.topology.loc[first, second]),
            float(distances.headloss.loc[first, second]),
            float(distances.delay.loc[first, second]),
        )

    pairs = _cross_edge_pairs(graph, left, right)
    if not pairs:
        return math.inf, math.inf, math.inf, math.inf
    raw_cut = float(candidate_external_cut_size(graph, left, right))
    topology = (
        min(raw_cut / distances.topology_scale, 1.0)
        if distances.topology_scale > 0
        else 0.0
    )
    headloss = min(float(distances.headloss.loc[a, b]) for a, b in pairs)
    delay = min(float(distances.delay.loc[a, b]) for a, b in pairs)
    combined = combine_normalized_distances(
        topology,
        headloss,
        delay,
        alpha=distances.alpha,
        beta=distances.beta,
    )
    return combined, topology, headloss, delay


def _edge_identifier(graph: nx.Graph, left: str, right: str) -> list[str]:
    data = graph.get_edge_data(left, right, default={})
    if graph.is_multigraph():
        records = data.values()
    else:
        records = (data,)
    names = []
    for record in records:
        names.append(str(record.get("name", f"{min(left, right)}--{max(left, right)}")))
    return names


def _boundary_links(
    graph: nx.Graph, clusters: Iterable[tuple[str, ...]]
) -> dict[tuple[str, ...], tuple[str, ...]]:
    result: dict[tuple[str, ...], tuple[str, ...]] = {}
    for cluster in clusters:
        members = set(cluster)
        links: set[str] = set()
        for node in cluster:
            for neighbor in graph.neighbors(node):
                if neighbor not in members:
                    links.update(_edge_identifier(graph, node, neighbor))
        result[cluster] = tuple(sorted(links))
    return result


def cluster_cpdz(
    graph: nx.Graph,
    distances: PhysicalDistanceMatrices,
    *,
    threshold: float,
    protected_nodes: Iterable[str] = (),
) -> CPDZPartition:
    """Cluster unprotected nodes with connected, deterministic single-link HAC."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    graph_nodes = set(graph.nodes)
    if graph_nodes != set(distances.nodes):
        raise ValueError("graph and distance matrices must contain identical nodes")
    protected = set(protected_nodes)
    unknown = protected.difference(graph_nodes)
    if unknown:
        raise ValueError(f"protected nodes absent from graph: {sorted(unknown)}")
    clusters = [(node,) for node in sorted(graph_nodes - protected)]
    history: list[MergeStep] = []

    while len(clusters) > 1:
        candidates: list[
            tuple[float, tuple[str, ...], tuple[str, ...], float, float, float]
        ] = []
        for index, left in enumerate(clusters):
            for right in clusters[index + 1 :]:
                if not _clusters_connected(graph, left, right):
                    continue
                combined, topology, headloss, delay = _cluster_distance(
                    graph, distances, left, right
                )
                candidates.append(
                    (combined, tuple(sorted(left)), tuple(sorted(right)), topology, headloss, delay)
                )
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best, left, right, topology, headloss, delay = candidates[0]
        if best > threshold:
            break
        merged = tuple(sorted(set(left) | set(right)))
        history.append(
            MergeStep(
                left=left,
                right=right,
                merged=merged,
                distance=best,
                topology=topology,
                headloss=headloss,
                delay=delay,
            )
        )
        clusters = [cluster for cluster in clusters if cluster not in {left, right}]
        clusters.append(merged)
        clusters.sort()

    final_clusters = tuple(sorted((tuple(sorted(cluster)) for cluster in clusters)))
    return CPDZPartition(
        clusters=final_clusters,
        protected_nodes=tuple(sorted(protected)),
        boundary_links=_boundary_links(graph, final_clusters),
        merge_history=tuple(history),
        threshold=threshold,
    )


def partition_network(
    bundle: NetworkBundle,
    distances: PhysicalDistanceMatrices,
    *,
    threshold: float,
) -> CPDZPartition:
    """Partition a loaded WDN using its source/pump protection set."""

    return cluster_cpdz(
        network_graph(bundle),
        distances,
        threshold=threshold,
        protected_nodes=bundle.protected_nodes,
    )


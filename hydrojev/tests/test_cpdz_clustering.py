import networkx as nx
import pandas as pd

from hydrojev.physics.cpdz_clustering import cluster_cpdz
from hydrojev.physics.distance_metrics import PhysicalDistanceMatrices


def _matrix(nodes: tuple[str, ...], edge_values: dict[tuple[str, str], float]) -> pd.DataFrame:
    frame = pd.DataFrame(1.0, index=nodes, columns=nodes)
    for node in nodes:
        frame.loc[node, node] = 0.0
    for (left, right), value in edge_values.items():
        frame.loc[left, right] = value
        frame.loc[right, left] = value
    return frame


def _distances(nodes: tuple[str, ...]) -> PhysicalDistanceMatrices:
    topology = _matrix(nodes, {("a", "b"): 0.2, ("b", "c"): 0.8, ("c", "d"): 0.9})
    hydraulic = _matrix(nodes, {("a", "b"): 0.1, ("b", "c"): 0.7, ("c", "d"): 0.9})
    delay = _matrix(nodes, {("a", "b"): 0.1, ("b", "c"): 0.7, ("c", "d"): 0.9})
    combined = _matrix(nodes, {("a", "b"): 0.15, ("b", "c"): 0.75, ("c", "d"): 0.9})
    return PhysicalDistanceMatrices(
        nodes=nodes,
        topology_raw=topology,
        headloss_raw=hydraulic,
        delay_raw=delay,
        topology=topology,
        headloss=hydraulic,
        delay=delay,
        combined=combined,
        topology_scale=1.0,
        headloss_scale=1.0,
        delay_scale=1.0,
        alpha=0.5,
        beta=0.25,
    )


def test_single_link_hac_merges_only_connected_clusters_below_threshold() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b", {"name": "ab"}), ("b", "c", {"name": "bc"}), ("c", "d", {"name": "cd"})])

    partition = cluster_cpdz(graph, _distances(("a", "b", "c", "d")), threshold=0.5)

    assert partition.clusters == (("a", "b"), ("c",), ("d",))
    assert len(partition.merge_history) == 1
    assert partition.merge_history[0].distance <= 0.5
    assert partition.boundary_links[("a", "b")] == ("bc",)


def test_protected_nodes_are_excluded_but_retained_as_boundaries() -> None:
    graph = nx.Graph()
    graph.add_edge("source", "a", name="source-a")
    graph.add_edge("a", "b", name="a-b")

    partition = cluster_cpdz(
        graph,
        _distances(("source", "a", "b")),
        threshold=1.0,
        protected_nodes={"source"},
    )

    assert partition.protected_nodes == ("source",)
    assert partition.clusters == (("a", "b"),)
    assert partition.boundary_links[("a", "b")] == ("source-a",)
    covered = set(partition.protected_nodes)
    for cluster in partition.clusters:
        covered.update(cluster)
    assert covered == set(graph.nodes)


def test_ties_are_resolved_deterministically_by_cluster_names() -> None:
    nodes = ("a", "b", "c")
    graph = nx.Graph([("a", "b"), ("a", "c")])
    equal = _matrix(nodes, {("a", "b"): 0.2, ("a", "c"): 0.2})
    distances = PhysicalDistanceMatrices(
        nodes=nodes,
        topology_raw=equal,
        headloss_raw=equal,
        delay_raw=equal,
        topology=equal,
        headloss=equal,
        delay=equal,
        combined=equal,
        topology_scale=1.0,
        headloss_scale=1.0,
        delay_scale=1.0,
        alpha=0.5,
        beta=0.25,
    )

    partition = cluster_cpdz(graph, distances, threshold=0.21)

    assert partition.merge_history[0].left == ("a",)
    assert partition.merge_history[0].right == ("b",)


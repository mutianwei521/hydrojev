"""HydroJEV WDN adaptations of topology, headloss, and wave-delay distances."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd

from hydrojev.datasets.wdn_loader import NetworkBundle


@dataclass(frozen=True)
class PhysicalDistanceMatrices:
    """Raw and normalized pair distances with their normalization provenance."""

    nodes: tuple[str, ...]
    topology_raw: pd.DataFrame
    headloss_raw: pd.DataFrame
    delay_raw: pd.DataFrame
    topology: pd.DataFrame
    headloss: pd.DataFrame
    delay: pd.DataFrame
    combined: pd.DataFrame
    topology_scale: float
    headloss_scale: float
    delay_scale: float
    alpha: float
    beta: float


def _validate_nonnegative(name: str, value: float, *, strictly: bool = False) -> None:
    invalid = value <= 0 if strictly else value < 0
    if not math.isfinite(value) or invalid:
        qualifier = "positive" if strictly else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")


def hazen_williams_headloss(
    length_m: float,
    diameter_m: float,
    roughness_c: float,
    flow_m3_s: float,
) -> float:
    """Return SI Hazen-Williams friction headloss magnitude in metres."""

    _validate_nonnegative("length_m", length_m)
    _validate_nonnegative("diameter_m", diameter_m, strictly=True)
    _validate_nonnegative("roughness_c", roughness_c, strictly=True)
    if not math.isfinite(flow_m3_s):
        raise ValueError("flow_m3_s must be finite")
    return 10.67 * length_m * abs(flow_m3_s) ** 1.852 / (
        roughness_c**1.852 * diameter_m**4.8704
    )


def darcy_weisbach_headloss(
    length_m: float,
    diameter_m: float,
    flow_m3_s: float,
    friction_factor: float = 0.02,
    gravity_m_s2: float = 9.80665,
) -> float:
    """Return Darcy-Weisbach friction headloss magnitude in metres."""

    _validate_nonnegative("length_m", length_m)
    _validate_nonnegative("diameter_m", diameter_m, strictly=True)
    _validate_nonnegative("friction_factor", friction_factor, strictly=True)
    _validate_nonnegative("gravity_m_s2", gravity_m_s2, strictly=True)
    if not math.isfinite(flow_m3_s):
        raise ValueError("flow_m3_s must be finite")
    velocity = 4.0 * abs(flow_m3_s) / (math.pi * diameter_m**2)
    return friction_factor * length_m / diameter_m * velocity**2 / (2.0 * gravity_m_s2)


def water_hammer_delay(length_m: float, acoustic_speed_m_s: float = 1000.0) -> float:
    """Return one-way pressure-wave propagation time, L/a, in seconds."""

    _validate_nonnegative("length_m", length_m)
    _validate_nonnegative("acoustic_speed_m_s", acoustic_speed_m_s, strictly=True)
    return length_m / acoustic_speed_m_s


def candidate_external_cut_size(
    graph: nx.Graph,
    left: Iterable[str],
    right: Iterable[str],
) -> int:
    """Count graph edges crossing the boundary of a proposed merged cluster."""

    merged = set(left) | set(right)
    if not merged:
        raise ValueError("candidate cluster cannot be empty")
    missing = merged.difference(graph.nodes)
    if missing:
        raise ValueError(f"cluster contains nodes absent from graph: {sorted(missing)}")
    count = 0
    for start, end in graph.edges():
        if (start in merged) != (end in merged):
            count += 1
    return count


def normalize_by_finite_maximum(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize finite non-negative values by their finite maximum."""

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if np.any(array[finite] < 0):
        raise ValueError("distance values must be non-negative")
    result = np.full(array.shape, np.nan, dtype=float)
    if not finite.any():
        return result, 0.0
    scale = float(np.max(array[finite]))
    if scale == 0.0:
        result[finite] = 0.0
    else:
        result[finite] = array[finite] / scale
    return result, scale


def combine_normalized_distances(
    topology: float,
    headloss: float,
    delay: float,
    *,
    alpha: float,
    beta: float,
) -> float:
    """Combine normalized components using the HydroJEV WDN weights."""

    if alpha < 0 or beta < 0 or alpha > 1 or beta > 1 or alpha + beta > 1 + 1e-12:
        raise ValueError("alpha and beta must be in [0, 1] and alpha + beta <= 1")
    components = (topology, headloss, delay)
    if any(not math.isfinite(value) or value < 0 for value in components):
        raise ValueError("normalized distances must be finite and non-negative")
    return alpha * topology + beta * headloss + (1.0 - alpha - beta) * delay


def network_graph(bundle: NetworkBundle) -> nx.MultiGraph:
    """Create an undirected multigraph while preserving EPANET link identity."""

    graph = nx.MultiGraph(name=bundle.name)
    graph.add_nodes_from(bundle.model.node_name_list)
    for link_name in bundle.model.link_name_list:
        link = bundle.model.get_link(link_name)
        graph.add_edge(
            link.start_node_name,
            link.end_node_name,
            key=link_name,
            name=link_name,
            link_type=type(link).__name__.lower(),
        )
    return graph


def _empty_raw(nodes: tuple[str, ...]) -> pd.DataFrame:
    values = np.full((len(nodes), len(nodes)), np.nan, dtype=float)
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=nodes, columns=nodes, dtype=float)


def _normalized_matrix(raw: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    normalized_values, scale = normalize_by_finite_maximum(raw.to_numpy(dtype=float))
    values = np.ones(raw.shape, dtype=float)
    finite = np.isfinite(normalized_values)
    values[finite] = normalized_values[finite]
    np.fill_diagonal(values, 0.0)
    normalized = pd.DataFrame(values, index=raw.index, columns=raw.columns, dtype=float)
    return normalized, scale


def _set_pair_minimum(frame: pd.DataFrame, left: str, right: str, value: float) -> None:
    current = frame.loc[left, right]
    selected = value if not math.isfinite(current) else min(float(current), value)
    frame.loc[left, right] = selected
    frame.loc[right, left] = selected


def build_physical_distances(
    bundle: NetworkBundle,
    *,
    alpha: float = 0.35,
    beta: float = 0.45,
    acoustic_speed_m_s: float = 1000.0,
    flow_by_link_m3_s: Mapping[str, float] | None = None,
    default_flow_m3_s: float = 0.01,
    darcy_friction_factor: float = 0.02,
) -> PhysicalDistanceMatrices:
    """Build edge-level matrices; nonadjacent or non-pipe pairs retain distance 1."""

    combine_normalized_distances(0.0, 0.0, 0.0, alpha=alpha, beta=beta)
    _validate_nonnegative("default_flow_m3_s", default_flow_m3_s, strictly=True)
    graph = network_graph(bundle)
    nodes = tuple(bundle.model.node_name_list)
    topology_raw = _empty_raw(nodes)
    headloss_raw = _empty_raw(nodes)
    delay_raw = _empty_raw(nodes)
    flows = flow_by_link_m3_s or {}
    headloss_model = bundle.headloss_model.upper().replace("_", "-")

    for link_name in bundle.model.link_name_list:
        link = bundle.model.get_link(link_name)
        left, right = link.start_node_name, link.end_node_name
        cut = float(candidate_external_cut_size(graph, {left}, {right}))
        _set_pair_minimum(topology_raw, left, right, cut)

        if link_name not in bundle.pipes:
            continue
        length = float(link.length)
        diameter = float(link.diameter)
        flow = float(flows.get(link_name, default_flow_m3_s))
        if headloss_model in {"H-W", "HW", "HAZEN-WILLIAMS"}:
            loss = hazen_williams_headloss(length, diameter, float(link.roughness), flow)
        elif headloss_model in {"D-W", "DW", "DARCY-WEISBACH"}:
            loss = darcy_weisbach_headloss(
                length, diameter, flow, friction_factor=darcy_friction_factor
            )
        else:
            raise ValueError(f"unsupported EPANET headloss model: {bundle.headloss_model}")
        _set_pair_minimum(headloss_raw, left, right, loss)
        _set_pair_minimum(
            delay_raw,
            left,
            right,
            water_hammer_delay(length, acoustic_speed_m_s),
        )

    topology, topology_scale = _normalized_matrix(topology_raw)
    headloss, headloss_scale = _normalized_matrix(headloss_raw)
    delay, delay_scale = _normalized_matrix(delay_raw)
    combined_values = np.ones((len(nodes), len(nodes)), dtype=float)
    np.fill_diagonal(combined_values, 0.0)
    combined = pd.DataFrame(combined_values, index=nodes, columns=nodes, dtype=float)
    for left, right in graph.edges():
        value = combine_normalized_distances(
            float(topology.loc[left, right]),
            float(headloss.loc[left, right]),
            float(delay.loc[left, right]),
            alpha=alpha,
            beta=beta,
        )
        combined.loc[left, right] = min(float(combined.loc[left, right]), value)
        combined.loc[right, left] = combined.loc[left, right]

    return PhysicalDistanceMatrices(
        nodes=nodes,
        topology_raw=topology_raw,
        headloss_raw=headloss_raw,
        delay_raw=delay_raw,
        topology=topology,
        headloss=headloss,
        delay=delay,
        combined=combined,
        topology_scale=topology_scale,
        headloss_scale=headloss_scale,
        delay_scale=delay_scale,
        alpha=alpha,
        beta=beta,
    )

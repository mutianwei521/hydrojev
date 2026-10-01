import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from hydrojev.physics.distance_metrics import (
    build_physical_distances,
    candidate_external_cut_size,
    combine_normalized_distances,
    darcy_weisbach_headloss,
    hazen_williams_headloss,
    normalize_by_finite_maximum,
    water_hammer_delay,
)
from hydrojev.config import load_config
from hydrojev.datasets.wdn_loader import load_network


def test_candidate_cut_counts_only_edges_leaving_merged_cluster() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])

    assert candidate_external_cut_size(graph, {"a"}, {"b"}) == 2
    assert candidate_external_cut_size(graph, {"a", "b"}, {"c"}) == 1


def test_hazen_williams_uses_si_headloss_equation() -> None:
    actual = hazen_williams_headloss(
        length_m=1000.0, diameter_m=0.30, roughness_c=120.0, flow_m3_s=0.05
    )
    expected = 10.67 * 1000.0 * 0.05**1.852 / (120.0**1.852 * 0.30**4.8704)

    assert actual == pytest.approx(expected, rel=1e-12)
    assert hazen_williams_headloss(1000.0, 0.30, 120.0, -0.05) == pytest.approx(
        expected
    )


def test_darcy_weisbach_uses_velocity_head() -> None:
    actual = darcy_weisbach_headloss(
        length_m=100.0,
        diameter_m=0.2,
        flow_m3_s=0.01,
        friction_factor=0.02,
    )
    velocity = 4.0 * 0.01 / (math.pi * 0.2**2)
    expected = 0.02 * 100.0 / 0.2 * velocity**2 / (2.0 * 9.80665)

    assert actual == pytest.approx(expected, rel=1e-12)


def test_water_hammer_delay_is_pipe_length_over_acoustic_speed() -> None:
    assert water_hammer_delay(120.0, acoustic_speed_m_s=1000.0) == pytest.approx(0.12)
    with pytest.raises(ValueError, match="acoustic_speed"):
        water_hammer_delay(120.0, acoustic_speed_m_s=0.0)


def test_normalization_ignores_nonfinite_values_and_preserves_zero() -> None:
    values = np.array([0.0, 2.0, 4.0, np.inf, np.nan])

    normalized, scale = normalize_by_finite_maximum(values)

    assert scale == 4.0
    np.testing.assert_allclose(normalized[:3], [0.0, 0.5, 1.0])
    assert np.isnan(normalized[3:]).all()


def test_weighted_distance_validates_weights_and_combines_components() -> None:
    assert combine_normalized_distances(0.2, 0.5, 0.8, alpha=0.25, beta=0.5) == pytest.approx(
        0.5
    )
    with pytest.raises(ValueError, match=r"alpha \+ beta"):
        combine_normalized_distances(0.2, 0.5, 0.8, alpha=0.8, beta=0.3)


def test_physical_distance_builder_runs_on_real_net1() -> None:
    project_root = Path(__file__).resolve().parents[2]
    bundle = load_network("Net1", config=load_config(project_root=project_root))

    distances = build_physical_distances(bundle)

    assert distances.combined.shape == (11, 11)
    assert np.allclose(np.diag(distances.combined), 0.0)
    assert np.isfinite(distances.combined.to_numpy()).all()

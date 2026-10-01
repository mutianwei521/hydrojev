"""Tests for the JSON-safe Jev state builder."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hydrojev.state_projection.context_builder import (
    UNAVAILABLE,
    UNKNOWN,
    FreshnessInfo,
    ZoneIdentity,
    build_jev_state,
    serialize_state,
    summarize_series,
)
from hydrojev.state_projection.residual_engine import (
    DetectorReading,
    DetectorState,
    cpdz_residual_reading,
    rat_covariance_distortion,
)


def _minimal_state(**overrides):
    kwargs = dict(
        timestamp="2026-09-20T12:00:00Z",
        freshness=FreshnessInfo(state_age_s=30.0, maximum_allowed_age_s=120.0),
        zone=ZoneIdentity(
            network_name="Net1",
            zone_id="cpdz_2",
            junction_count=4,
            boundary_links=("P10", "P11"),
            protected_nodes=("R1", "PU1"),
            headloss_model="H-W",
            hydraulic_timestep_s=900.0,
        ),
        detector_readings=[cpdz_residual_reading([1.0], [0.0], [1.0], threshold=3.0)],
        observed_vs_predicted={
            "J2_head_m": {"observed": 51.0, "predicted": 50.5, "attacked": 62.0}
        },
        hydraulic_consistency={
            "mass_balance_residual_m3_s": 0.001,
            "mass_balance_tolerance_m3_s": 0.01,
        },
        attack_surface={"attackable_measurements": ["P34"], "max_changed_measurements": 4},
    )
    kwargs.update(overrides)
    return build_jev_state(**kwargs)


# --------------------------------------------------------------------------- #
# Freshness
# --------------------------------------------------------------------------- #


def test_freshness_flag() -> None:
    assert FreshnessInfo(30.0, 120.0).is_fresh is True
    assert FreshnessInfo(200.0, 120.0).is_fresh is False


# --------------------------------------------------------------------------- #
# summarize_series
# --------------------------------------------------------------------------- #


def test_summarize_series_reports_stats_and_tail() -> None:
    summary = summarize_series([0.0, 1.0, 2.0, 3.0, 4.0], max_points=8)
    assert summary["count"] == 5
    assert summary["min"] == 0.0
    assert summary["max"] == 4.0
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["last"] == 4.0
    assert summary["sampled"] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_summarize_series_decimates_long_windows() -> None:
    summary = summarize_series(list(range(1000)), max_points=8)
    assert summary["count"] == 1000
    # The decimated sample is bounded and keeps the endpoints.
    assert len(summary["sampled"]) <= 8
    assert summary["sampled"][0] == 0.0
    assert summary["sampled"][-1] == 999.0


def test_summarize_series_marks_empty_unavailable_and_rejects_nan() -> None:
    assert summarize_series([])["state"] == UNAVAILABLE
    with pytest.raises(ValueError):
        summarize_series([1.0, np.nan, 3.0])


# --------------------------------------------------------------------------- #
# build_jev_state structure
# --------------------------------------------------------------------------- #


def test_state_has_required_keys_and_is_json_serializable() -> None:
    state = _minimal_state()
    for key in (
        "schema_version",
        "timestamp",
        "freshness",
        "zone",
        "detectors",
        "observed_vs_predicted",
        "hydraulic_consistency",
        "attack_surface",
        "thresholds",
        "maintenance_context",
        "provenance",
    ):
        assert key in state
    # Must round-trip through JSON with no custom encoder.
    assert json.loads(json.dumps(state)) == state


def test_observed_predicted_attacked_kept_in_distinct_fields() -> None:
    state = _minimal_state()
    fields = state["observed_vs_predicted"]["J2_head_m"]
    assert fields["observed"] == 51.0
    assert fields["predicted"] == 50.5
    assert fields["attacked"] == 62.0


def test_freshness_and_within_tolerance_are_computed() -> None:
    state = _minimal_state()
    assert state["freshness"]["is_fresh"] is True
    assert state["hydraulic_consistency"]["within_tolerance"] is True

    tripped = _minimal_state(
        hydraulic_consistency={
            "mass_balance_residual_m3_s": 5.0,
            "mass_balance_tolerance_m3_s": 0.01,
        }
    )
    assert tripped["hydraulic_consistency"]["within_tolerance"] is False


def test_missing_optional_context_is_marked_not_imputed() -> None:
    state = _minimal_state()
    # Unset zone timing collapses to an explicit unknown token.
    assert state["zone"]["observation_interval_s"] == UNKNOWN
    assert state["zone"]["minimum_wave_propagation_time_s"] == UNKNOWN
    # Maintenance context is present with neutral markers, never assumed benign.
    assert state["maintenance_context"]["weather_event"] == UNKNOWN
    assert state["maintenance_context"]["scheduled_work_orders"] == []
    # Provenance defaults to an explicit unknown note rather than being absent.
    assert state["provenance"] == {"note": UNKNOWN}


def test_unavailable_detector_serializes_as_null_value() -> None:
    unavailable = rat_covariance_distortion(np.zeros((1, 3)), np.eye(3))
    assert unavailable.state is DetectorState.UNAVAILABLE
    state = _minimal_state(
        detector_readings=[unavailable, cpdz_residual_reading([1.0], [0.0], [1.0], threshold=3.0)]
    )
    block = state["detectors"]["rat_covariance_distortion"]
    assert block["value"] is None
    assert block["state"] == "unavailable"
    assert block["source"] == "hydrojev_auxiliary"
    # The primary detector keeps its own field.
    assert state["detectors"]["cpdz_residual_normalized"]["source"] == "primary"


def test_evidence_windows_are_summarized_not_dumped() -> None:
    state = _minimal_state(evidence_windows={"J2_head_m": list(range(500))})
    window = state["evidence_windows"]["J2_head_m"]
    assert window["count"] == 500
    assert len(window["sampled"]) <= 8


# --------------------------------------------------------------------------- #
# NaN/Inf rejection
# --------------------------------------------------------------------------- #


def test_nonfinite_observed_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        _minimal_state(
            observed_vs_predicted={"bad": {"observed": float("nan"), "predicted": 1.0}}
        )


def test_nonfinite_attack_surface_is_rejected() -> None:
    with pytest.raises(ValueError):
        _minimal_state(attack_surface={"score": float("inf")})


# --------------------------------------------------------------------------- #
# Payload budget
# --------------------------------------------------------------------------- #


def test_serialize_state_stays_small_even_with_long_windows() -> None:
    state = _minimal_state(
        evidence_windows={f"series_{i}": list(range(10_000)) for i in range(5)}
    )
    encoded = serialize_state(state, max_bytes=16384)
    # Summaries keep a large amount of evidence well within budget.
    assert len(encoded.encode("utf-8")) < 16384


def test_serialize_state_enforces_budget() -> None:
    state = _minimal_state()
    with pytest.raises(ValueError, match="budget"):
        serialize_state(state, max_bytes=10)

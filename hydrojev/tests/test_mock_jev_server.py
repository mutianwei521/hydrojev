"""Tests for the deterministic, evidence-only mock Jev arbiter."""

from __future__ import annotations

import pytest

from hydrojev.arbiter.decision_primitives import (
    JevSource,
    ThreatCause,
    build_request,
    parse_response,
)
from hydrojev.simulation.mock_jev_server import (
    arbitrate_state,
    create_app,
    decide_locally,
)


def mkstate(
    detectors: dict[str, str] | None = None,
    *,
    within_tolerance: bool = True,
    energy: float = 0.0,
    fresh: bool = True,
    actuator_ok: bool = True,
) -> dict:
    return {
        "detectors": {name: {"state": st} for name, st in (detectors or {}).items()},
        "hydraulic_consistency": {
            "within_tolerance": within_tolerance,
            "extra_energy_fraction": energy,
            "reported_pump_state_is_self_consistent": actuator_ok,
        },
        "freshness": {"is_fresh": fresh},
    }


# Evidence signatures for each cause (detector states only -- no hidden label).
NORMAL = mkstate({"cpdz_residual_normalized": "nominal", "reconstruction_autoencoder": "nominal"})
ADVERSARIAL = mkstate(
    {
        "reconstruction_autoencoder": "nominal",  # learned detector fooled
        "cpdz_residual_normalized": "alerting",  # physics disagrees
        "rat_covariance_distortion": "alerting",  # covariance manipulated
    }
)
FS_FDI = mkstate(
    {"cpdz_residual_normalized": "alerting", "vectorized_cusum": "nominal"},  # SE bypassed
    energy=0.6,  # but pumps over-work
    within_tolerance=True,
)
BURST = mkstate(
    {"reconstruction_autoencoder": "alerting", "cpdz_residual_normalized": "alerting"},
    within_tolerance=False,  # genuine loss of water
    actuator_ok=True,
)


def _selected(state: dict) -> str:
    return arbitrate_state(state)["answers"]["threat_cause"]["choice"]


# --------------------------------------------------------------------------- #
# Determinism and the evidence-only invariant
# --------------------------------------------------------------------------- #


def test_arbitration_is_deterministic() -> None:
    assert arbitrate_state(ADVERSARIAL) == arbitrate_state(ADVERSARIAL)


def test_arbiter_refuses_ground_truth_labels() -> None:
    with pytest.raises(ValueError, match="ground-truth"):
        arbitrate_state({"true_cause": "hydraulic_fs_fdi", **NORMAL})
    with pytest.raises(ValueError, match="ground-truth"):
        arbitrate_state({"zone": {"label": "attack"}, **NORMAL})


# --------------------------------------------------------------------------- #
# Cause discrimination from evidence signatures
# --------------------------------------------------------------------------- #


def test_normal_evidence_selects_normal_transient() -> None:
    assert _selected(NORMAL) == ThreatCause.NORMAL_TRANSIENT.value


def test_ae_evasion_signature_selects_adversarial() -> None:
    assert _selected(ADVERSARIAL) == ThreatCause.ADVERSARIAL_AE_EVASION.value


def test_fs_fdi_signature_selects_hydraulic_fs_fdi() -> None:
    assert _selected(FS_FDI) == ThreatCause.HYDRAULIC_FS_FDI.value


def test_burst_signature_selects_physical_burst() -> None:
    assert _selected(BURST) == ThreatCause.PHYSICAL_BURST.value


def test_choice_probabilities_are_valid_distribution() -> None:
    answer = arbitrate_state(FS_FDI)["answers"]["threat_cause"]
    assert set(answer["probabilities"]) == {c.value for c in ThreatCause}
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Air-gap noul and severity respond to corroboration and consequence
# --------------------------------------------------------------------------- #


def test_airgap_noul_rises_with_corroboration_and_falls_when_stale() -> None:
    normal_noul = arbitrate_state(NORMAL)["answers"]["trip_edge_airgap"]["noul"]
    attack_noul = arbitrate_state(ADVERSARIAL)["answers"]["trip_edge_airgap"]["noul"]
    assert normal_noul < 0.2 < attack_noul

    stale = mkstate(
        {
            "reconstruction_autoencoder": "nominal",
            "cpdz_residual_normalized": "alerting",
            "rat_covariance_distortion": "alerting",
        },
        fresh=False,
    )
    stale_noul = arbitrate_state(stale)["answers"]["trip_edge_airgap"]["noul"]
    assert stale_noul < attack_noul


def test_severity_low_for_normal_and_high_for_energy_catastrophe() -> None:
    normal_score = arbitrate_state(NORMAL)["answers"]["severity_score"]["score"]
    fs_fdi_score = arbitrate_state(FS_FDI)["answers"]["severity_score"]["score"]
    assert normal_score < 1.0
    assert fs_fdi_score > 3.0


# --------------------------------------------------------------------------- #
# The local decision path is validated by the same parser as the live path
# --------------------------------------------------------------------------- #


def test_decide_locally_returns_typed_validated_decision() -> None:
    decision = decide_locally(FS_FDI, source=JevSource.MOCK)
    assert decision.source is JevSource.MOCK
    assert decision.threat_cause.as_threat_cause() is ThreatCause.HYDRAULIC_FS_FDI
    assert 0.0 <= decision.trip_edge_airgap.probability <= 1.0


# --------------------------------------------------------------------------- #
# FastAPI contract
# --------------------------------------------------------------------------- #


def test_fastapi_mock_serves_schema_compliant_response() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.post("/v1/systemone", json=build_request(ADVERSARIAL))
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "mock"
    decision = parse_response(body, source=JevSource.MOCK)
    assert decision.threat_cause.as_threat_cause() is ThreatCause.ADVERSARIAL_AE_EVASION


def test_fastapi_mock_rejects_ground_truth_with_422() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.post("/v1/systemone", json=build_request({"true_cause": "x"}))
    assert response.status_code == 422

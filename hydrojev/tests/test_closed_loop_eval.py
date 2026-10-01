"""Tests for the closed-loop reflex policy and its honest evaluation metrics.

The first group proves the paper's central safety guarantee: a confident cloud
verdict alone cannot cut remote-control authority when the deterministic gates
say no. The rest cover the reflex, recovery hysteresis, provenance auditing, and
the recall/energy/latency metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.arbiter.decision_primitives import (
    JevSource,
    SEVERITY_LEGEND,
    ThreatCause,
    parse_response,
)
from hydrojev.simulation.closed_loop_eval import (
    AuditRecord,
    ClosedLoopEvaluation,
    ControllerAction,
    ControllerState,
    EnergyTrace,
    ReflexController,
    ReflexPolicy,
    ScenarioMisalignedError,
    ScenarioRun,
    StepEvidence,
    distill_evidence,
    energy_mitigation,
    recall_summary,
    summarize_latency,
    wilson_interval,
)

POLICY = ReflexPolicy(
    minimum_evidence_count=3,
    minimum_noul_probability=0.80,
    minimum_attack_persistence=2,
    isolation_hysteresis_steps=2,
)


def decision(
    cause: ThreatCause,
    *,
    noul: float,
    severity_level: int = 0,
    source: JevSource = JevSource.MOCK,
    timing: dict[str, float] | None = None,
    attempts: int = 1,
):
    """Build a schema-valid, typed JevDecision with a chosen cause and noul."""

    others = [c for c in ThreatCause if c is not cause]
    probs = {cause.value: 0.9}
    for c in others:
        probs[c.value] = 0.1 / len(others)
    score_probs = {str(i): (0.9 if i == severity_level else 0.025) for i in range(len(SEVERITY_LEGEND))}
    body = {
        "model": "jev-test",
        "answers": {
            "threat_cause": {
                "type": "choice",
                "choice": cause.value,
                "probabilities": probs,
                "confidence": 0.9,
            },
            "trip_edge_airgap": {"type": "noul", "noul": noul},
            "severity_score": {
                "type": "score",
                "score": float(severity_level),
                "probabilities": score_probs,
                "confidence": 0.9,
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    return parse_response(body, source=source, timing_ms=timing or {}, attempts=attempts)


def evid(
    *,
    fresh: bool = True,
    count: int = 4,
    physics: bool = True,
    severe: bool = False,
    operator: bool = False,
    age: float = 0.0,
) -> StepEvidence:
    return StepEvidence(
        is_fresh=fresh,
        state_age_s=age,
        evidence_count=count,
        local_physics_alert=physics,
        severe_hydraulic_violation=severe,
        operator_confirmed=operator,
    )


def _states(records: list[AuditRecord]) -> set[ControllerState]:
    return {record.state for record in records}


# --------------------------------------------------------------------------- #
# Safety guarantee: Jev alone cannot trip the air gap
# --------------------------------------------------------------------------- #


def test_confident_cloud_cannot_trip_on_stale_data() -> None:
    controller = ReflexController(POLICY)
    d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.99)
    records = controller.run([(d, evid(fresh=False, count=5, physics=True)) for _ in range(6)])
    assert ControllerState.ISOLATED not in _states(records)
    assert ControllerState.ISOLATING not in _states(records)
    assert all(r.action is ControllerAction.HOLD_STALE for r in records)


def test_confident_cloud_cannot_trip_on_weak_evidence() -> None:
    controller = ReflexController(POLICY)
    d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.99)
    # Only one alerting detector: below the minimum corroboration count.
    records = controller.run([(d, evid(count=1, physics=True)) for _ in range(6)])
    assert ControllerState.ISOLATED not in _states(records)
    # It is allowed to *suspect*, just not to isolate.
    assert records[-1].state is ControllerState.SUSPECTED
    assert "insufficient_evidence" in records[-1].reasons


def test_confident_cloud_cannot_trip_without_local_hydraulic_corroboration() -> None:
    controller = ReflexController(POLICY)  # require_local_hydraulic_violation defaults True
    d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.99)
    # Five learned/statistical detectors fire but no physical residual/mass/energy.
    records = controller.run([(d, evid(count=5, physics=False)) for _ in range(6)])
    assert ControllerState.ISOLATED not in _states(records)
    assert ControllerState.ISOLATING not in _states(records)
    assert all(r.state is ControllerState.MONITORING for r in records)


def test_low_noul_blocks_isolation_even_with_strong_evidence() -> None:
    controller = ReflexController(POLICY)
    d = decision(ThreatCause.ADVERSARIAL_AE_EVASION, noul=0.5)  # below 0.80 gate
    records = controller.run([(d, evid(count=5, physics=True)) for _ in range(6)])
    assert ControllerState.ISOLATED not in _states(records)
    assert records[-1].state is ControllerState.SUSPECTED
    assert "noul_below_minimum" in records[-1].reasons


# --------------------------------------------------------------------------- #
# The positive path and the deterministic reflex
# --------------------------------------------------------------------------- #


def test_persistent_corroborated_cyber_attack_reaches_isolated() -> None:
    controller = ReflexController(POLICY)
    d = decision(ThreatCause.ADVERSARIAL_AE_EVASION, noul=0.95)
    records = controller.run([(d, evid(count=4, physics=True)) for _ in range(4)])
    # Debounce: suspect first, assert the gap once persistent, then confirm.
    assert records[0].state is ControllerState.SUSPECTED
    assert records[1].state is ControllerState.ISOLATING
    assert records[1].action is ControllerAction.ASSERT_AIR_GAP
    assert records[2].state is ControllerState.ISOLATED


def test_reflex_trips_safe_fallback_even_on_stale_data() -> None:
    controller = ReflexController(POLICY)
    # A severe local hydraulic violation, no cloud support at all, stale data.
    d = decision(ThreatCause.NORMAL_TRANSIENT, noul=0.0)
    record = controller.step(d, evid(fresh=False, count=0, physics=False, severe=True), step_index=0)
    assert record.state is ControllerState.SAFE_FALLBACK
    assert record.action is ControllerAction.REFLEX_SAFE_FALLBACK
    assert record.reflex_trip is True


def test_reflex_takes_priority_over_cognition() -> None:
    controller = ReflexController(POLICY)
    d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.99)
    record = controller.step(d, evid(count=5, physics=True, severe=True), step_index=0)
    assert record.state is ControllerState.SAFE_FALLBACK


def test_operator_confirmation_gate_blocks_then_allows() -> None:
    policy = ReflexPolicy(
        minimum_evidence_count=3,
        minimum_attack_persistence=2,
        require_operator_confirmation=True,
    )
    controller = ReflexController(policy)
    d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.95)
    # Without operator confirmation, it can only suspect.
    blocked = controller.run([(d, evid(count=4, physics=True, operator=False)) for _ in range(4)])
    assert ControllerState.ISOLATED not in _states(blocked)
    assert "operator_not_confirmed" in blocked[-1].reasons
    # With confirmation, the same evidence isolates.
    controller.reset()
    allowed = controller.run([(d, evid(count=4, physics=True, operator=True)) for _ in range(4)])
    assert ControllerState.ISOLATED in _states(allowed)


def test_recovery_restores_after_sustained_clear() -> None:
    controller = ReflexController(POLICY)
    attack = decision(ThreatCause.ADVERSARIAL_AE_EVASION, noul=0.95)
    clear = decision(ThreatCause.NORMAL_TRANSIENT, noul=0.02)
    steps = [(attack, evid(count=4, physics=True)) for _ in range(3)]
    steps += [(clear, evid(count=0, physics=False)) for _ in range(3)]
    records = controller.run(steps)
    assert records[2].state is ControllerState.ISOLATED
    # After isolation_hysteresis_steps of clear evidence, remote control is restored.
    assert records[-1].state is ControllerState.MONITORING
    assert ControllerAction.RESTORE in {r.action for r in records}


# --------------------------------------------------------------------------- #
# Evidence distillation and provenance auditing
# --------------------------------------------------------------------------- #


def test_distill_evidence_reads_only_physical_fields() -> None:
    state = {
        "freshness": {"is_fresh": True, "state_age_s": 1.0},
        "detectors": {
            "cpdz_residual_normalized": {"state": "alerting"},
            "rat_covariance_distortion": {"state": "alerting"},
            "gepfm_baseline_drift": {"state": "nominal"},
        },
        "hydraulic_consistency": {
            "within_tolerance": True,
            "mass_balance_residual_m3_s": 0.01,
            "mass_balance_tolerance_m3_s": 0.5,
            "extra_energy_fraction": 0.0,
        },
    }
    ev = distill_evidence(state, policy=POLICY)
    assert ev.evidence_count == 2  # two alerting detectors
    assert ev.local_physics_alert is True  # cpdz residual is alerting
    assert ev.severe_hydraulic_violation is False


def test_distill_evidence_flags_severe_burst() -> None:
    state = {
        "freshness": {"is_fresh": True, "state_age_s": 0.5},
        "detectors": {"cpdz_residual_normalized": {"state": "alerting"}},
        "hydraulic_consistency": {
            "within_tolerance": False,
            "mass_balance_residual_m3_s": 5.0,  # 10x tolerance -> burst regime
            "mass_balance_tolerance_m3_s": 0.5,
        },
    }
    ev = distill_evidence(state, policy=POLICY)
    assert ev.severe_hydraulic_violation is True


def test_audit_record_carries_provenance_and_serializes() -> None:
    controller = ReflexController(POLICY)
    d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.95, source=JevSource.LOCAL_FALLBACK)
    record = controller.step(d, evid(count=4, physics=True), step_index=0)
    assert record.jev_source == "local_fallback"
    public = record.to_public_dict()
    assert public["jev_source"] == "local_fallback"
    assert public["gates"]["isolation_permitted"] is False  # persistence not yet met
    assert "state" in public and "reasons" in public


def test_policy_from_safety_config_maps_fields() -> None:
    class _Safety:
        maximum_state_age_s = 3.0
        minimum_evidence_count = 4
        minimum_noul_probability = 0.7
        minimum_attack_persistence = 5
        require_local_hydraulic_violation = False
        require_operator_confirmation = True
        isolation_hysteresis_steps = 6

    policy = ReflexPolicy.from_safety_config(_Safety())
    assert policy.minimum_evidence_count == 4
    assert policy.require_operator_confirmation is True
    assert policy.isolation_hysteresis_steps == 6


# --------------------------------------------------------------------------- #
# Detection metrics: recall, Wilson interval, provenance counts
# --------------------------------------------------------------------------- #


def _run_records(controller: ReflexController, steps) -> tuple[AuditRecord, ...]:
    return tuple(controller.run(steps))


def _attack_run(scenario_id: str, category: str, *, responded: bool) -> ScenarioRun:
    controller = ReflexController(POLICY)
    if responded:
        d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.95)
        steps = [(d, evid(count=4, physics=True)) for _ in range(4)]
    else:
        d = decision(ThreatCause.HYDRAULIC_FS_FDI, noul=0.99)
        steps = [(d, evid(count=1, physics=True)) for _ in range(4)]  # weak -> never isolates
    return ScenarioRun(scenario_id, category, True, 0, 900.0, _run_records(controller, steps))


def test_wilson_interval_bounds_and_guards() -> None:
    low, high = wilson_interval(19, 20)
    assert 0.0 < low < 0.95 < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)
    with pytest.raises(ValueError):
        wilson_interval(5, 3)


def test_recall_summary_counts_tp_fn_fpr_and_categories() -> None:
    runs = [
        _attack_run("a1", "adversarial_ae_evasion", responded=True),
        _attack_run("a2", "adversarial_ae_evasion", responded=True),
        _attack_run("a3", "hydraulic_fs_fdi", responded=False),  # a false negative
    ]
    # A benign scenario that never trips.
    calm = ReflexController(POLICY)
    benign_clear = decision(ThreatCause.NORMAL_TRANSIENT, noul=0.02)
    runs.append(
        ScenarioRun(
            "b1", "normal_transient", False, 0, 900.0,
            _run_records(calm, [(benign_clear, evid(count=0, physics=False)) for _ in range(4)]),
        )
    )
    summary = recall_summary(runs)
    assert summary["attack_scenarios"] == 3
    assert summary["true_positives"] == 2
    assert summary["false_negatives"] == 1
    assert summary["recall"] == pytest.approx(2 / 3)
    assert summary["false_positive_rate"] == pytest.approx(0.0)
    assert summary["per_category"]["adversarial_ae_evasion"]["recall"] == pytest.approx(1.0)
    lo, hi = summary["recall_wilson_95"]
    assert 0.0 <= lo <= 2 / 3 <= hi <= 1.0


def test_evaluation_counts_jev_sources() -> None:
    evaluation = ClosedLoopEvaluation(
        runs=[
            _attack_run("a1", "hydraulic_fs_fdi", responded=True),
            _attack_run("a2", "hydraulic_fs_fdi", responded=True),
        ]
    )
    counts = evaluation.jev_sources()
    assert counts["mock"] == 8  # 2 runs x 4 steps, all mock-sourced
    assert counts["live"] == 0
    public = evaluation.to_public_dict()
    assert public["n_runs"] == 2
    assert public["detection"]["recall"] == pytest.approx(1.0)


def test_time_to_response_measures_from_onset() -> None:
    run = _attack_run("a1", "hydraulic_fs_fdi", responded=True)
    # ISOLATING is reached on step 1 (onset 0), at one 900 s interval.
    assert run.time_to_response_s() == pytest.approx(900.0)


# --------------------------------------------------------------------------- #
# Counterfactual energy mitigation
# --------------------------------------------------------------------------- #


def test_energy_mitigation_quantifies_reduction() -> None:
    t = np.linspace(0.0, 3600.0, 13)
    flow = np.full_like(t, 0.05)
    baseline = EnergyTrace(t, flow, np.full_like(t, 30.0))
    attacked = EnergyTrace(t, flow, np.full_like(t, 48.0))  # +60% head -> +60% energy
    defended = EnergyTrace(t, flow, np.full_like(t, 31.5))  # +5% residual after isolation
    result = energy_mitigation(baseline, attacked, defended)
    assert result["extra_energy_fraction_attacked"] == pytest.approx(0.60, abs=1e-6)
    assert result["extra_energy_fraction_defended"] == pytest.approx(0.05, abs=1e-6)
    assert result["mitigated_fraction"] == pytest.approx(0.55, abs=1e-6)
    assert result["residual_extra_fraction"] < 0.06


def test_energy_mitigation_rejects_misaligned_horizons() -> None:
    t = np.linspace(0.0, 3600.0, 13)
    flow = np.full_like(t, 0.05)
    baseline = EnergyTrace(t, flow, np.full_like(t, 30.0))
    # Different number of samples -> not the same initial condition/horizon.
    t2 = np.linspace(0.0, 3600.0, 12)
    misaligned = EnergyTrace(t2, np.full_like(t2, 0.05), np.full_like(t2, 40.0))
    with pytest.raises(ScenarioMisalignedError):
        energy_mitigation(baseline, misaligned, baseline)
    # Same length, different horizon.
    t3 = np.linspace(0.0, 1800.0, 13)
    misaligned_end = EnergyTrace(t3, flow, np.full_like(t3, 40.0))
    with pytest.raises(ScenarioMisalignedError):
        energy_mitigation(baseline, misaligned_end, baseline)


# --------------------------------------------------------------------------- #
# Latency accounting and the honest water-hammer caveat
# --------------------------------------------------------------------------- #


def test_summarize_latency_meets_target() -> None:
    steps = [{"surrogate_ms": 4.0, "residual_ms": 1.0, "jev_decision_ms": 90.0} for _ in range(20)]
    result = summarize_latency(steps, target_ms=150.0)
    assert result["total_ms"]["p95"] == pytest.approx(95.0, abs=1e-6)
    assert result["meets_target_p95"] is True
    assert set(result["per_stage_ms"]) == {"surrogate_ms", "residual_ms", "jev_decision_ms"}


def test_summarize_latency_flags_missed_target_and_wave_caveat() -> None:
    steps = [{"surrogate_ms": 5.0, "jev_decision_ms": 200.0} for _ in range(20)]
    result = summarize_latency(steps, target_ms=150.0, minimum_wave_propagation_time_s=0.05)
    assert result["meets_target_p95"] is False
    # The cognitive loop is far slower than a 50 ms pipe wave time -- as expected.
    assert result["p95_faster_than_min_wave"] is False
    assert "not the primary water-hammer relay" in result["warning"].lower()


def test_summarize_latency_handles_no_samples() -> None:
    assert summarize_latency([])["count"] == 0

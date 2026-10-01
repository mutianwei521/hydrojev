"""Closed-loop reflex policy and offline evaluation for HydroJEV.

This module realizes the paper's *Cognitive Reflex Paradigm* as executable
control logic and the honest metrics that judge it.

**Reflex vs. cognition.** Two response paths cut a zone's remote-control
authority, and they are deliberately separate:

* The **reflex** (:data:`ControllerAction.REFLEX_SAFE_FALLBACK`) is a deterministic
  *local* interlock. When local hydraulic sensing shows a severe violation --
  the burst / water-hammer regime -- the edge relay drops to a safe fallback
  immediately, independent of the cloud and even on stale data. This is the fast
  path that a 150 ms cloud round-trip must never gate.
* The **cognitive** air-gap (:data:`ControllerAction.ASSERT_AIR_GAP`) is the
  Jev-advised path for *stealthy cyber attacks* (adversarial AE evasion, FS-FDI),
  which by design leave mass balance within tolerance and so never trip the
  reflex. Here the cloud contributes causal discrimination -- but it is advisory.

**The cloud cannot bypass the interlocks.** Every Jev recommendation is filtered
through hard, deterministic gates (:class:`ReflexController._evaluate_gates`):
data freshness, a minimum count of corroborating detectors, local physical
corroboration, attack persistence, a minimum air-gap probability, and optional
operator confirmation. A confident cloud verdict on stale, weak, or
physically-uncorroborated evidence is refused. This is the safety guarantee the
paper claims, enforced in code and proven by the tests.

**Scientific-integrity boundary.** Ground-truth scenario labels (``is_attack``,
``category``) live only in the *evaluation* layer here; they are never part of
the state handed to the arbiter (which :mod:`hydrojev.simulation.mock_jev_server`
enforces). So the recall and isolation numbers measure how well HydroJEV's
orthogonal evidence *plus* these gates neutralize attacks -- not a replay of
hidden answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.arbiter.decision_primitives import JevDecision, JevSource, ThreatCause
from hydrojev.state_projection.residual_engine import integrate_pump_energy

# Cutting remote-control authority stops a *cyber* attacker's channel; it is the
# cognitive, Jev-advised response. A physical burst is handled by the reflex
# fallback instead, so it is not in this set.
AIRGAP_CAUSES: frozenset[ThreatCause] = frozenset(
    {ThreatCause.ADVERSARIAL_AE_EVASION, ThreatCause.HYDRAULIC_FS_FDI}
)

# Canonical detector name whose residual is the deterministic physics check that
# is orthogonal to the (attackable) learned autoencoder.
_PHYSICS_DETECTOR = "cpdz_residual_normalized"


class ControllerState(str, Enum):
    """The five closed-loop states the paper's policy moves through."""

    MONITORING = "monitoring"
    SUSPECTED = "suspected"
    ISOLATING = "isolating"
    ISOLATED = "isolated"
    SAFE_FALLBACK = "safe_fallback"


class ControllerAction(str, Enum):
    """The audited action taken on one step."""

    HOLD = "hold"
    RAISE_SUSPICION = "raise_suspicion"
    ASSERT_AIR_GAP = "assert_air_gap"
    CONFIRM_ISOLATED = "confirm_isolated"
    HOLD_ISOLATED = "hold_isolated"
    REFLEX_SAFE_FALLBACK = "reflex_safe_fallback"
    HOLD_STALE = "hold_stale"
    HOLD_PENDING_RESTORE = "hold_pending_restore"
    RESTORE = "restore"


# States in which remote-control authority is (being) cut -- a protective action
# is in force. Used by the metrics to decide whether an attack was neutralized.
PROTECTIVE_STATES: frozenset[ControllerState] = frozenset(
    {ControllerState.ISOLATING, ControllerState.ISOLATED, ControllerState.SAFE_FALLBACK}
)


# --------------------------------------------------------------------------- #
# Policy and per-step evidence
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReflexPolicy:
    """Deterministic thresholds for the safety gates and the local reflex.

    The gate thresholds mirror :class:`hydrojev.config.SafetyConfig`; the three
    reflex-specific thresholds are local to the controller (the reflex is a local
    interlock, not a cloud policy) and carry conservative defaults.
    """

    maximum_state_age_s: float = 2.0
    minimum_evidence_count: int = 3
    minimum_noul_probability: float = 0.80
    minimum_attack_persistence: int = 2
    require_local_hydraulic_violation: bool = True
    require_operator_confirmation: bool = False
    isolation_hysteresis_steps: int = 2
    # Reflex + corroboration thresholds (deterministic, local).
    material_energy_fraction: float = 0.05  # "material" over-pumping for corroboration
    severe_energy_fraction: float = 0.50  # over-pumping severe enough to trip the reflex
    severe_mass_violation_multiple: float = 3.0  # |residual| >= k * tolerance -> burst regime

    @classmethod
    def from_safety_config(cls, safety: Any) -> "ReflexPolicy":
        """Build from a :class:`SafetyConfig`, keeping the reflex defaults."""

        return cls(
            maximum_state_age_s=float(safety.maximum_state_age_s),
            minimum_evidence_count=int(safety.minimum_evidence_count),
            minimum_noul_probability=float(safety.minimum_noul_probability),
            minimum_attack_persistence=int(safety.minimum_attack_persistence),
            require_local_hydraulic_violation=bool(safety.require_local_hydraulic_violation),
            require_operator_confirmation=bool(safety.require_operator_confirmation),
            isolation_hysteresis_steps=int(safety.isolation_hysteresis_steps),
        )


@dataclass(frozen=True)
class StepEvidence:
    """The deterministic, cloud-independent evidence distilled for one step.

    Kept separate from the Jev decision so the gates can be unit-tested in
    isolation and so the controller never conflates cloud opinion with local
    physical fact.
    """

    is_fresh: bool
    state_age_s: float
    evidence_count: int  # number of *alerting* detectors (corroboration breadth)
    local_physics_alert: bool  # deterministic physics/hydraulic corroboration
    severe_hydraulic_violation: bool  # burst-scale local violation -> reflex
    operator_confirmed: bool = False


def distill_evidence(state: Mapping[str, Any], *, policy: ReflexPolicy) -> StepEvidence:
    """Reduce a Jev *state* dict to the deterministic evidence the gates need.

    Reads only physical/observable fields (freshness, detector alert states,
    mass-balance residual, energy accounting, operator confirmation). It never
    reads a threat label -- causal discrimination is the arbiter's job.
    """

    freshness = state.get("freshness", {})
    if isinstance(freshness, Mapping):
        state_age_s = float(freshness.get("state_age_s", 0.0))
        is_fresh = bool(freshness.get("is_fresh", state_age_s <= policy.maximum_state_age_s))
    else:
        state_age_s, is_fresh = 0.0, True

    detectors = state.get("detectors", {})
    if not isinstance(detectors, Mapping):
        detectors = {}
    evidence_count = 0
    physics_detector_alerting = False
    for name, entry in detectors.items():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("state") == "alerting":
            evidence_count += 1
            if str(name) == _PHYSICS_DETECTOR:
                physics_detector_alerting = True

    hydraulic = state.get("hydraulic_consistency", {})
    if not isinstance(hydraulic, Mapping):
        hydraulic = {}
    within_tolerance = hydraulic.get("within_tolerance", True)
    mass_violated = within_tolerance is False
    residual = _finite_or(hydraulic.get("mass_balance_residual_m3_s"), 0.0)
    tolerance = _finite_or(hydraulic.get("mass_balance_tolerance_m3_s"), 0.0)
    energy_inflation = max(0.0, _finite_or(hydraulic.get("extra_energy_fraction"), 0.0))

    # Deterministic hydraulic corroboration for the isolation gate: an orthogonal
    # physics residual fires, mass balance is broken, or pumps materially over-work.
    local_physics_alert = bool(
        physics_detector_alerting
        or mass_violated
        or energy_inflation > policy.material_energy_fraction
    )

    # Burst-regime reflex: an unambiguous, large local violation. A *stealthy*
    # attack keeps mass balance within tolerance, so it never trips this.
    severe = bool(
        (mass_violated and tolerance > 0.0 and abs(residual) >= policy.severe_mass_violation_multiple * tolerance)
        or energy_inflation >= policy.severe_energy_fraction
    )

    operator_confirmed = bool(state.get("operator_confirmed", False))

    return StepEvidence(
        is_fresh=is_fresh,
        state_age_s=state_age_s,
        evidence_count=evidence_count,
        local_physics_alert=local_physics_alert,
        severe_hydraulic_violation=severe,
        operator_confirmed=operator_confirmed,
    )


def _finite_or(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


# --------------------------------------------------------------------------- #
# Gate evaluation and the per-step audit record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateEvaluation:
    """The outcome of every hard safety gate for one step (the audit basis)."""

    fresh: bool
    cause_actionable: bool
    noul_sufficient: bool
    evidence_sufficient: bool
    persistence_met: bool
    hydraulic_corroborated: bool
    operator_confirmed: bool

    @property
    def isolation_permitted(self) -> bool:
        """Cloud-advised isolation is allowed only if *every* gate passes."""

        return all(
            (
                self.fresh,
                self.cause_actionable,
                self.noul_sufficient,
                self.evidence_sufficient,
                self.persistence_met,
                self.hydraulic_corroborated,
                self.operator_confirmed,
            )
        )

    def blocking_reasons(self) -> tuple[str, ...]:
        """Names of the gates that are currently *failing* (empty if permitted)."""

        checks = {
            "stale_data": not self.fresh,
            "cause_not_actionable": not self.cause_actionable,
            "noul_below_minimum": not self.noul_sufficient,
            "insufficient_evidence": not self.evidence_sufficient,
            "attack_not_persistent": not self.persistence_met,
            "no_local_hydraulic_corroboration": not self.hydraulic_corroborated,
            "operator_not_confirmed": not self.operator_confirmed,
        }
        return tuple(name for name, failing in checks.items() if failing)


@dataclass(frozen=True)
class AuditRecord:
    """A complete, replayable record of one closed-loop step."""

    step_index: int
    timestamp: str | None
    previous_state: ControllerState
    state: ControllerState
    action: ControllerAction
    threat_cause: str
    noul: float
    severity_user_facing: float  # 1-5 operator-facing level
    jev_source: str  # provenance: live / mock / local_fallback
    attempts: int
    reflex_trip: bool
    attack_run: int
    clear_run: int
    gates: GateEvaluation
    reasons: tuple[str, ...]
    latency_ms: dict[str, float]

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "timestamp": self.timestamp,
            "previous_state": self.previous_state.value,
            "state": self.state.value,
            "action": self.action.value,
            "threat_cause": self.threat_cause,
            "noul": self.noul,
            "severity_user_facing": self.severity_user_facing,
            "jev_source": self.jev_source,
            "attempts": self.attempts,
            "reflex_trip": self.reflex_trip,
            "attack_run": self.attack_run,
            "clear_run": self.clear_run,
            "gates": {
                "fresh": self.gates.fresh,
                "cause_actionable": self.gates.cause_actionable,
                "noul_sufficient": self.gates.noul_sufficient,
                "evidence_sufficient": self.gates.evidence_sufficient,
                "persistence_met": self.gates.persistence_met,
                "hydraulic_corroborated": self.gates.hydraulic_corroborated,
                "operator_confirmed": self.gates.operator_confirmed,
                "isolation_permitted": self.gates.isolation_permitted,
            },
            "reasons": list(self.reasons),
            "latency_ms": dict(self.latency_ms),
        }


# --------------------------------------------------------------------------- #
# The reflex controller (state machine with hysteresis)
# --------------------------------------------------------------------------- #


class ReflexController:
    """Deterministic closed-loop policy: reflex interlock + gated cognition.

    Feed it a stream of ``(JevDecision, StepEvidence)`` pairs. Each
    :meth:`step` returns an :class:`AuditRecord` and advances the state machine.
    The controller is deterministic and carries only debounce counters as state.
    """

    def __init__(self, policy: ReflexPolicy | None = None) -> None:
        self.policy = policy or ReflexPolicy()
        self.state = ControllerState.MONITORING
        self._attack_run = 0
        self._clear_run = 0

    def reset(self) -> None:
        self.state = ControllerState.MONITORING
        self._attack_run = 0
        self._clear_run = 0

    def _evaluate_gates(
        self, decision: JevDecision, evidence: StepEvidence, *, attack_run: int
    ) -> GateEvaluation:
        p = self.policy
        cause = decision.threat_cause.as_threat_cause()
        return GateEvaluation(
            fresh=evidence.is_fresh,
            cause_actionable=cause in AIRGAP_CAUSES,
            noul_sufficient=decision.trip_edge_airgap.probability >= p.minimum_noul_probability,
            evidence_sufficient=evidence.evidence_count >= p.minimum_evidence_count,
            persistence_met=attack_run >= p.minimum_attack_persistence,
            hydraulic_corroborated=(not p.require_local_hydraulic_violation)
            or evidence.local_physics_alert,
            operator_confirmed=(not p.require_operator_confirmation) or evidence.operator_confirmed,
        )

    def step(
        self,
        decision: JevDecision,
        evidence: StepEvidence,
        *,
        step_index: int,
        timestamp: str | None = None,
    ) -> AuditRecord:
        prev = self.state
        p = self.policy
        cause = decision.threat_cause.as_threat_cause()

        # A fresh, cause-actionable step with local physical corroboration is the
        # temporal signal that debounces isolation (attack persistence).
        signals_attack = evidence.is_fresh and cause in AIRGAP_CAUSES and evidence.local_physics_alert
        self._attack_run = self._attack_run + 1 if signals_attack else 0

        gates = self._evaluate_gates(decision, evidence, attack_run=self._attack_run)
        reflex_trip = evidence.severe_hydraulic_violation

        if reflex_trip:
            # Priority 1: deterministic local interlock. The cloud cannot veto it,
            # and it fires even on stale data because it reads local sensing.
            new_state = ControllerState.SAFE_FALLBACK
            action = ControllerAction.REFLEX_SAFE_FALLBACK
            self._clear_run = 0
            reasons = ("severe_local_hydraulic_violation",)
        elif not evidence.is_fresh:
            # Priority 2: cloud intel is untrustworthy. Freeze -- no cloud-driven
            # escalation and no recovery -- but a latched protective state holds.
            new_state = prev
            action = ControllerAction.HOLD_STALE
            self._clear_run = 0
            reasons = ("stale_data_freezes_cloud_action",)
        elif gates.isolation_permitted:
            # Priority 3: cognition, fully gated. Assert then confirm the air gap.
            self._clear_run = 0
            if prev in (ControllerState.MONITORING, ControllerState.SUSPECTED):
                new_state = ControllerState.ISOLATING
                action = ControllerAction.ASSERT_AIR_GAP
            elif prev is ControllerState.ISOLATING:
                new_state = ControllerState.ISOLATED
                action = ControllerAction.CONFIRM_ISOLATED
            elif prev is ControllerState.ISOLATED:
                new_state = ControllerState.ISOLATED
                action = ControllerAction.HOLD_ISOLATED
            else:  # SAFE_FALLBACK latched; cognition now concurs
                new_state = ControllerState.SAFE_FALLBACK
                action = ControllerAction.HOLD_ISOLATED
            reasons = ("all_isolation_gates_passed",)
        elif signals_attack:
            # An attack is visible but at least one gate blocks isolation.
            new_state = (
                ControllerState.SUSPECTED
                if prev in (ControllerState.MONITORING, ControllerState.SUSPECTED)
                else prev
            )
            action = ControllerAction.RAISE_SUSPICION
            self._clear_run = 0
            reasons = gates.blocking_reasons()
        else:
            # Clear step: decay suspicion, or restore after the hysteresis window.
            self._clear_run += 1
            if prev is ControllerState.SUSPECTED:
                new_state = ControllerState.MONITORING
                action = ControllerAction.RESTORE
                reasons = ("suspicion_cleared",)
            elif prev in (
                ControllerState.ISOLATING,
                ControllerState.ISOLATED,
                ControllerState.SAFE_FALLBACK,
            ):
                if self._clear_run >= p.isolation_hysteresis_steps:
                    new_state = ControllerState.MONITORING
                    action = ControllerAction.RESTORE
                    reasons = ("sustained_clear_evidence_restored",)
                else:
                    new_state = prev
                    action = ControllerAction.HOLD_PENDING_RESTORE
                    reasons = (f"awaiting_clear_hysteresis_{self._clear_run}/{p.isolation_hysteresis_steps}",)
            else:
                new_state = ControllerState.MONITORING
                action = ControllerAction.HOLD
                reasons = ("nominal",)

        self.state = new_state
        return AuditRecord(
            step_index=step_index,
            timestamp=timestamp,
            previous_state=prev,
            state=new_state,
            action=action,
            threat_cause=cause.value,
            noul=float(decision.trip_edge_airgap.probability),
            severity_user_facing=float(decision.severity.user_facing_level),
            jev_source=decision.source.value,
            attempts=int(decision.attempts),
            reflex_trip=reflex_trip,
            attack_run=self._attack_run,
            clear_run=self._clear_run,
            gates=gates,
            reasons=tuple(reasons),
            latency_ms=dict(decision.timing_ms),
        )

    def run(
        self,
        steps: Sequence[tuple[JevDecision, StepEvidence]],
        *,
        timestamps: Sequence[str] | None = None,
    ) -> list[AuditRecord]:
        """Process a whole scenario from a clean state, returning the audit trail."""

        self.reset()
        records: list[AuditRecord] = []
        for index, (decision, evidence) in enumerate(steps):
            timestamp = timestamps[index] if timestamps is not None else None
            records.append(self.step(decision, evidence, step_index=index, timestamp=timestamp))
        return records


# --------------------------------------------------------------------------- #
# Scenario runs and detection metrics (evaluation layer; uses ground truth)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScenarioRun:
    """One evaluated scenario: its audit trail plus *held-out* ground truth.

    ``is_attack`` and ``category`` are evaluation-only labels. They score the
    controller's behaviour; they are never seen by the arbiter that produced the
    decisions inside ``records``.
    """

    scenario_id: str
    category: str
    is_attack: bool
    onset_step: int
    step_interval_s: float
    records: tuple[AuditRecord, ...]

    def responded(self) -> bool:
        """True if a protective action was in force at or after attack onset."""

        return any(
            record.state in PROTECTIVE_STATES and record.step_index >= self.onset_step
            for record in self.records
        )

    def time_to_response_s(self) -> float | None:
        """Seconds from onset to the first protective action, or ``None``."""

        for record in self.records:
            if record.step_index >= self.onset_step and record.state in PROTECTIVE_STATES:
                return float((record.step_index - self.onset_step) * self.step_interval_s)
        return None


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Preferred over the normal approximation for the small samples and
    near-boundary proportions typical of recall on a handful of scenarios.
    """

    if n <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > n:
        raise ValueError("successes must lie in [0, n]")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * float(np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _percentiles(samples: Sequence[float]) -> dict[str, float]:
    array = np.asarray(list(samples), dtype=float)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def recall_summary(runs: Sequence[ScenarioRun]) -> dict[str, Any]:
    """TP/FN/recall (+ Wilson CI) over the *attack* scenarios in ``runs``.

    Also reports the false-positive rate over the *benign* scenarios, so the
    paper can show the gates neutralize attacks without over-tripping.
    """

    attacks = [run for run in runs if run.is_attack]
    benign = [run for run in runs if not run.is_attack]

    true_positives = sum(1 for run in attacks if run.responded())
    false_negatives = len(attacks) - true_positives
    recall = (true_positives / len(attacks)) if attacks else float("nan")
    ci_low, ci_high = wilson_interval(true_positives, len(attacks)) if attacks else (float("nan"), float("nan"))

    false_positives = sum(1 for run in benign if run.responded())
    fpr = (false_positives / len(benign)) if benign else float("nan")

    per_category: dict[str, dict[str, Any]] = {}
    for run in attacks:
        bucket = per_category.setdefault(run.category, {"n": 0, "tp": 0})
        bucket["n"] += 1
        bucket["tp"] += int(run.responded())
    for bucket in per_category.values():
        bucket["recall"] = bucket["tp"] / bucket["n"] if bucket["n"] else float("nan")

    response_times = [t for run in attacks if (t := run.time_to_response_s()) is not None]

    return {
        "attack_scenarios": len(attacks),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "recall": recall,
        "recall_wilson_95": [ci_low, ci_high],
        "benign_scenarios": len(benign),
        "false_positives": false_positives,
        "false_positive_rate": fpr,
        "per_category": per_category,
        "attack_to_isolation_s": _percentiles(response_times),
    }


# --------------------------------------------------------------------------- #
# Counterfactual pump-energy mitigation
# --------------------------------------------------------------------------- #


class ScenarioMisalignedError(ValueError):
    """Raised when energy counterfactuals do not share an initial condition/horizon."""


@dataclass(frozen=True)
class EnergyTrace:
    """A pump-energy counterfactual sampled on a common time grid."""

    times_s: np.ndarray
    flow_m3_s: np.ndarray
    head_m: np.ndarray


def _require_aligned(baseline: EnergyTrace, other: EnergyTrace, label: str) -> None:
    if baseline.times_s.shape != other.times_s.shape:
        raise ScenarioMisalignedError(f"{label} time grid length differs from baseline")
    if not np.allclose(baseline.times_s, other.times_s):
        raise ScenarioMisalignedError(
            f"{label} must share the baseline time grid (same initial condition and horizon)"
        )


def energy_mitigation(
    baseline: EnergyTrace,
    attacked: EnergyTrace,
    defended: EnergyTrace,
    *,
    efficiency: float = 0.75,
) -> dict[str, float]:
    """Compare baseline/attacked/defended pumping energy on one aligned horizon.

    The counterfactuals must share the same time grid (hence the same initial
    condition and horizon), or the comparison is rejected as invalid -- an
    unaligned energy delta is not a defensible mitigation claim.
    """

    _require_aligned(baseline, attacked, "attacked")
    _require_aligned(baseline, defended, "defended")

    baseline_kwh = integrate_pump_energy(
        baseline.times_s, baseline.flow_m3_s, baseline.head_m, efficiency=efficiency
    )
    attacked_kwh = integrate_pump_energy(
        attacked.times_s, attacked.flow_m3_s, attacked.head_m, efficiency=efficiency
    )
    defended_kwh = integrate_pump_energy(
        defended.times_s, defended.flow_m3_s, defended.head_m, efficiency=efficiency
    )
    if baseline_kwh <= 0:
        raise ScenarioMisalignedError("baseline energy must be positive")

    extra_attacked = (attacked_kwh - baseline_kwh) / baseline_kwh
    extra_defended = (defended_kwh - baseline_kwh) / baseline_kwh
    return {
        "baseline_kwh": baseline_kwh,
        "attacked_kwh": attacked_kwh,
        "defended_kwh": defended_kwh,
        "extra_energy_fraction_attacked": extra_attacked,
        "extra_energy_fraction_defended": extra_defended,
        "mitigated_fraction": extra_attacked - extra_defended,
        "residual_extra_fraction": extra_defended,
    }


# --------------------------------------------------------------------------- #
# Latency accounting
# --------------------------------------------------------------------------- #


def summarize_latency(
    per_step_stage_ms: Sequence[Mapping[str, float]],
    *,
    target_ms: float = 150.0,
    minimum_wave_propagation_time_s: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-stage and total loop latency into p50/p95/p99.

    ``per_step_stage_ms`` is one mapping per step, e.g.
    ``{"surrogate_ms": 4.1, "residual_ms": 0.9, "jev_decision_ms": 88.0}``.
    The total per step is the sum of its stages. The p95 total is compared to the
    research target and, if given, to the minimum pipe wave-propagation time --
    with an explicit warning that cloud arbitration is *not* the primary
    water-hammer relay (that is the local reflex).
    """

    if not per_step_stage_ms:
        return {"count": 0, "warning": "no latency samples"}

    stage_names: list[str] = []
    for step in per_step_stage_ms:
        for name in step:
            if name not in stage_names:
                stage_names.append(name)

    per_stage = {
        name: _percentiles([float(step.get(name, 0.0)) for step in per_step_stage_ms])
        for name in stage_names
    }
    totals = [float(sum(step.values())) for step in per_step_stage_ms]
    total = _percentiles(totals)

    result: dict[str, Any] = {
        "count": len(per_step_stage_ms),
        "per_stage_ms": per_stage,
        "total_ms": total,
        "target_ms": float(target_ms),
        "meets_target_p95": bool(total.get("p95", float("inf")) <= target_ms),
    }
    if minimum_wave_propagation_time_s is not None:
        wave_ms = float(minimum_wave_propagation_time_s) * 1000.0
        result["minimum_wave_propagation_ms"] = wave_ms
        result["p95_faster_than_min_wave"] = bool(total.get("p95", float("inf")) <= wave_ms)
        result["warning"] = (
            "Cloud arbitration is advisory and is NOT the primary water-hammer "
            "relay; the local reflex (safe_fallback) is. This latency compares the "
            "cognitive loop to the wave time for context only."
        )
    return result


# --------------------------------------------------------------------------- #
# Top-level scenario evaluation
# --------------------------------------------------------------------------- #


@dataclass
class ClosedLoopEvaluation:
    """Aggregated, provenance-tagged results for a batch of scenario runs."""

    runs: list[ScenarioRun] = field(default_factory=list)

    def detection(self) -> dict[str, Any]:
        return recall_summary(self.runs)

    def jev_sources(self) -> dict[str, int]:
        """Count decisions by provenance across all runs -- never mixed silently."""

        counts: dict[str, int] = {source.value: 0 for source in JevSource}
        for run in self.runs:
            for record in run.records:
                counts[record.jev_source] = counts.get(record.jev_source, 0) + 1
        return counts

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "n_runs": len(self.runs),
            "detection": self.detection(),
            "jev_source_counts": self.jev_sources(),
        }

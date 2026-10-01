"""A standalone, typed-judgment HydroJEV anomaly detector.

This module deliberately contains no network client and never calls TypeSafe.
It defines the JSON contract, validates a previously received Jev response,
fuses the typed judgment with deterministic evidence-quality gates, and offers
a transparent no-Jev baseline. A live caller can use
``hydrojev.arbiter.jev_client.JevClient`` outside this module, then pass its
validated response here.

The detector is intentionally conservative:

* missing hydraulic quantities are represented as ``"unavailable"``; they are
  never replaced with zero;
* ground-truth labels are rejected from the state;
* stale or unusable evidence produces ``abstain``; and
* contradictory typed answers also produce ``abstain`` instead of silently
  resolving the contradiction in favour of an alarm.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "hydrojev.detection_state/1"
UNKNOWN = "unknown"
UNAVAILABLE = "unavailable"

DETECTOR_STATES = frozenset({"nominal", "alerting", "unavailable"})
EVENT_TYPES = (
    "normal_transient",
    "cyber_attack",
    "physical_fault",
    "insufficient_evidence",
)
DETECTION_QUESTION_IDS = ("alarm", "event_type", "severity")

SEVERITY_LEGEND = (
    "Nominal; observations agree with the healthy prediction and no detector is corroborated.",
    "Minor anomaly; one weak or isolated deviation with no evidence of service impact.",
    "Moderate concern; several signals disagree or a physical consistency check is violated.",
    "Serious incident; corroborated manipulation or fault creates credible service or equipment risk.",
    "Critical emergency; ongoing or imminent damage, contamination, or wide supply loss is plausible.",
)

_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "ground_truth",
        "true_cause",
        "true_label",
        "label",
        "attack_label",
        "scenario_label",
        "expected_cause",
        "y_true",
        "scenario_id",
        "att_flag",
        "attack_id",
        "attack_ids",
        "scenario_ids",
        "severity_label",
        "true_severity",
        "expected_severity",
        "row_index",
        "row_indices",
        "row_id",
        "row_ids",
        "raw_index",
        "raw_indices",
        "grid_index",
        "grid_indices",
        "attack_onset",
        "attack_start",
        "attack_end",
    }
)

EVENT_TYPE_CRITERIA = {
    "normal_transient": (
        "Ordinary demand, scheduled operation, or measurement noise; the evidence has no coherent, "
        "multi-detector signature of manipulation or a physical fault."
    ),
    "cyber_attack": (
        "Telemetry manipulation or false-data injection is the most likely explanation, supported by "
        "disagreement between learned detectors and independent hydraulic or temporal evidence."
    ),
    "physical_fault": (
        "A genuine hydraulic fault such as a burst or large leak is the most likely explanation, "
        "with pressure/flow or mass-balance evidence that is physically coherent."
    ),
    "insufficient_evidence": (
        "The state is stale, incomplete, contradictory, or too weak to distinguish a benign transient, "
        "cyber attack, and physical fault."
    ),
}

_ALARM_CRITERIA = {
    "true": (
        "The current evidence warrants an anomaly event alert. Use this only when the observed "
        "discrepancy is supported by fresh, relevant evidence rather than a single unexplained value."
    ),
    "false": (
        "The evidence does not warrant an anomaly event alert; the state is nominal or adequately "
        "explained by ordinary operation."
    ),
}

_SEVERITY_INSTRUCTIONS = (
    "Rate the operational severity of the current water-distribution event using only the supplied "
    "evidence and the ordered severity criteria."
)


class HydroJEVSchemaError(ValueError):
    """Raised when the detection state, request, or typed answer is invalid."""


class DetectionState(str, Enum):
    """Three-state operational output of HydroJEV."""

    NORMAL = "normal"
    ALARM = "alarm"
    ABSTAIN = "abstain"


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Convert supported scalar/container values to strict JSON values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (Integral, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (Real, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} is NaN or infinity")
        return number
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _assert_no_ground_truth(value: Any, *, path: str = "state") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in _FORBIDDEN_LABEL_KEYS:
                raise HydroJEVSchemaError(
                    f"{path} contains forbidden ground-truth key {key!r}"
                )
            _assert_no_ground_truth(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_ground_truth(child, path=f"{path}[{index}]")


def _optional_numeric(value: Any, *, path: str) -> float | str:
    if value is None or value in {UNKNOWN, UNAVAILABLE}:
        return UNAVAILABLE
    if not _is_finite_number(value):
        raise ValueError(f"{path} must be finite or unavailable")
    return float(value)


def _validate_mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def build_detection_state(
    *,
    timestamp: str,
    freshness: Mapping[str, Any],
    detectors: Mapping[str, Mapping[str, Any]],
    observed_vs_predicted: Mapping[str, Any] | None = None,
    temporal_summary: Mapping[str, Any] | None = None,
    hydraulic_consistency: Mapping[str, Any] | None = None,
    attack_surface: Mapping[str, Any] | None = None,
    maintenance_context: Mapping[str, Any] | None = None,
    routing: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a strict, label-free and JSON-safe detection state.

    ``hydraulic_consistency`` is optional because BATADAL telemetry does not
    contain every hydraulic quantity. Missing values are explicit
    ``"unavailable"`` markers, never numeric placeholders.
    """

    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("timestamp must be a non-empty string")
    freshness_map = _validate_mapping(freshness, path="freshness")
    if not isinstance(freshness_map.get("is_fresh"), bool):
        raise ValueError("freshness.is_fresh must be boolean")
    if not isinstance(detectors, Mapping):
        raise ValueError("detectors must be an object")

    detector_block: dict[str, dict[str, Any]] = {}
    for name, raw in detectors.items():
        entry = _validate_mapping(raw, path=f"detectors.{name}")
        state = entry.get("state")
        if state not in DETECTOR_STATES:
            raise ValueError(
                f"detector state for {name!r} must be one of {sorted(DETECTOR_STATES)}"
            )
        item = {str(key): value for key, value in entry.items()}
        item["state"] = str(state)
        item.setdefault("value", UNAVAILABLE)
        item.setdefault("threshold", UNAVAILABLE)
        item.setdefault("normalized_margin", UNAVAILABLE)
        detector_block[str(name)] = _json_safe(item, path=f"detectors.{name}")

    if hydraulic_consistency is None:
        hydraulic_block: dict[str, Any] = {
            "state": UNAVAILABLE,
            "within_tolerance": UNAVAILABLE,
            "mass_balance_residual_m3_s": UNAVAILABLE,
            "mass_balance_tolerance_m3_s": UNAVAILABLE,
            "extra_energy_fraction": UNAVAILABLE,
        }
    else:
        hydraulic_input = dict(_validate_mapping(hydraulic_consistency, path="hydraulic_consistency"))
        hydraulic_block = {str(key): value for key, value in hydraulic_input.items()}
        hydraulic_block.setdefault("state", "available")
        hydraulic_block.setdefault("within_tolerance", UNAVAILABLE)
        hydraulic_block.setdefault("mass_balance_residual_m3_s", UNAVAILABLE)
        hydraulic_block.setdefault("mass_balance_tolerance_m3_s", UNAVAILABLE)
        hydraulic_block.setdefault("extra_energy_fraction", UNAVAILABLE)
        for key in (
            "mass_balance_residual_m3_s",
            "mass_balance_tolerance_m3_s",
            "extra_energy_fraction",
        ):
            hydraulic_block[key] = _optional_numeric(hydraulic_block[key], path=f"hydraulic_consistency.{key}")

    freshness_block = {str(key): value for key, value in freshness_map.items()}
    freshness_block.setdefault("state_age_s", UNAVAILABLE)
    freshness_block.setdefault("maximum_allowed_age_s", UNAVAILABLE)
    freshness_block = _json_safe(freshness_block, path="freshness")
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "freshness": freshness_block,
        "detectors": detector_block,
        "observed_vs_predicted": _json_safe(
            dict(observed_vs_predicted or {"state": UNAVAILABLE}),
            path="observed_vs_predicted",
        ),
        "temporal_summary": _json_safe(
            dict(temporal_summary or {"state": UNAVAILABLE}),
            path="temporal_summary",
        ),
        "hydraulic_consistency": _json_safe(hydraulic_block, path="hydraulic_consistency"),
        "attack_surface": _json_safe(
            dict(attack_surface or {"state": UNAVAILABLE}), path="attack_surface"
        ),
        "maintenance_context": _json_safe(
            dict(maintenance_context or {"state": UNKNOWN}), path="maintenance_context"
        ),
        "routing": _json_safe(dict(routing or {"candidate": True}), path="routing"),
        "provenance": _json_safe(
            dict(provenance or {"state": UNKNOWN}), path="provenance"
        ),
    }
    _assert_no_ground_truth(state)
    # A final JSON round trip catches accidental non-standard scalar values.
    json.dumps(state, ensure_ascii=False, allow_nan=False)
    return state


def build_detection_questions() -> dict[str, dict[str, Any]]:
    """Return the versioned, detection-specific Jev question contract."""

    return {
        "alarm": {
            "type": "noul",
            "instructions": (
                "Using only the supplied state, should the current causal window be raised as an "
                "anomaly event? Do not infer an answer from an omitted or unavailable quantity."
            ),
            "criteria": dict(_ALARM_CRITERIA),
        },
        "event_type": {
            "type": "choice",
            "instructions": (
                "Using only the supplied state, select the most likely event type. Select "
                "insufficient_evidence when the evidence is stale, incomplete, or contradictory."
            ),
            "criteria": dict(EVENT_TYPE_CRITERIA),
        },
        "severity": {
            "type": "score",
            "instructions": _SEVERITY_INSTRUCTIONS,
            "criteria": list(SEVERITY_LEGEND),
        },
    }


def build_detection_request(
    state: Mapping[str, Any], *, model: str = "jev-latest"
) -> dict[str, Any]:
    """Build a request body without performing any I/O."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    _assert_no_ground_truth(state)
    safe_state = _json_safe(dict(state), path="state")
    request = {"model": model, "state": safe_state, "questions": build_detection_questions()}
    json.dumps(request, ensure_ascii=False, allow_nan=False)
    return request


def _probability(value: Any, *, label: str) -> float:
    if not _is_finite_number(value):
        raise HydroJEVSchemaError(f"{label} must be a finite number")
    number = float(value)
    if number < -1e-9 or number > 1.0 + 1e-9:
        raise HydroJEVSchemaError(f"{label} must lie in [0, 1]")
    return min(max(number, 0.0), 1.0)


def _distribution(
    raw: Any, *, expected_keys: set[Any], label: str
) -> dict[Any, float]:
    if not isinstance(raw, Mapping):
        raise HydroJEVSchemaError(f"{label} must be an object")
    converted: dict[Any, float] = {}
    for key, value in raw.items():
        converted[key] = _probability(value, label=f"{label}[{key}]")
    if set(converted) != expected_keys:
        raise HydroJEVSchemaError(
            f"{label} options {sorted(converted)} must equal {sorted(expected_keys)}"
        )
    # TypeSafe responses are often serialized with two decimal places; a
    # displayed distribution such as 0.07 + 0.11 + 0.81 can therefore total
    # 0.99 even though the underlying probabilities sum to one.  Accept that
    # bounded rounding and renormalize the typed values used downstream.
    total = sum(converted.values())
    if total <= 0.0 or abs(total - 1.0) >= 2.5e-2:
        raise HydroJEVSchemaError(f"{label} must sum to 1")
    return {key: value / total for key, value in converted.items()}


def _source_value(source: Any) -> str:
    value = getattr(source, "value", source)
    if not isinstance(value, str) or not value:
        raise HydroJEVSchemaError("source must be a non-empty string")
    return value


@dataclass(frozen=True)
class JevDetectionDecision:
    """Validated typed answers returned by the detection question contract."""

    alarm_probability: float
    event_type: str
    event_probabilities: dict[str, float]
    event_confidence: float
    severity_score: float
    severity_probabilities: dict[int, float]
    severity_legend: dict[int, str]
    severity_confidence: float
    model: str
    source: str
    usage: dict[str, int]
    timing_ms: dict[str, float]
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "alarm_probability": self.alarm_probability,
            "event_type": self.event_type,
            "event_probabilities": dict(self.event_probabilities),
            "event_confidence": self.event_confidence,
            "severity_score": self.severity_score,
            "severity_probabilities": {str(k): v for k, v in self.severity_probabilities.items()},
            "severity_legend": {str(k): v for k, v in self.severity_legend.items()},
            "severity_confidence": self.severity_confidence,
            "model": self.model,
            "source": self.source,
            "usage": dict(self.usage),
            "timing_ms": dict(self.timing_ms),
            "attempts": self.attempts,
        }


def parse_detection_response(
    body: Mapping[str, Any],
    *,
    source: Any,
    timing_ms: Mapping[str, Any] | None = None,
    attempts: int = 1,
) -> JevDetectionDecision:
    """Validate a raw API response into an immutable typed decision."""

    if not isinstance(body, Mapping):
        raise HydroJEVSchemaError("response body must be an object")
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise HydroJEVSchemaError("response must contain an answers object")
    missing = sorted(set(DETECTION_QUESTION_IDS) - set(answers))
    if missing:
        raise HydroJEVSchemaError(f"response omitted question ids {missing}")

    alarm = answers["alarm"]
    if not isinstance(alarm, Mapping) or alarm.get("type") != "noul":
        raise HydroJEVSchemaError("alarm answer must have type 'noul'")
    if "confidence" in alarm:
        raise HydroJEVSchemaError("noul answers must not carry a confidence field")
    alarm_probability = _probability(alarm.get("noul"), label="alarm.noul")

    event = answers["event_type"]
    if not isinstance(event, Mapping) or event.get("type") != "choice":
        raise HydroJEVSchemaError("event_type answer must have type 'choice'")
    event_probabilities_raw = event.get("probabilities")
    if not isinstance(event_probabilities_raw, Mapping):
        raise HydroJEVSchemaError("event_type answer needs a probabilities map")
    event_probabilities = _distribution(
        {str(key): value for key, value in event_probabilities_raw.items()},
        expected_keys=set(EVENT_TYPES),
        label="event_type.probabilities",
    )
    selected_event = event.get("choice")
    if selected_event not in EVENT_TYPES:
        raise HydroJEVSchemaError("event_type.choice is outside the event taxonomy")
    event_confidence = _probability(event.get("confidence"), label="event_type.confidence")

    severity = answers["severity"]
    if not isinstance(severity, Mapping) or severity.get("type") != "score":
        raise HydroJEVSchemaError("severity answer must have type 'score'")
    raw_score = severity.get("score")
    if not _is_finite_number(raw_score) or not 0.0 <= float(raw_score) <= 4.0:
        raise HydroJEVSchemaError("severity score must lie in [0, 4]")
    raw_probs = severity.get("probabilities")
    if not isinstance(raw_probs, Mapping):
        raise HydroJEVSchemaError("severity answer needs a probabilities map")
    severity_probabilities = _distribution(
        {int(key): value for key, value in raw_probs.items()},
        expected_keys=set(range(5)),
        label="severity.probabilities",
    )
    raw_legend = severity.get("legend", {})
    if not isinstance(raw_legend, Mapping):
        raise HydroJEVSchemaError("severity.legend must be an object")
    severity_legend = {int(key): str(value) for key, value in raw_legend.items()}
    if set(severity_legend) != set(range(5)):
        raise HydroJEVSchemaError("severity.legend must contain levels 0 through 4")
    severity_confidence = _probability(severity.get("confidence"), label="severity.confidence")

    usage_raw = body.get("usage", {})
    if not isinstance(usage_raw, Mapping):
        raise HydroJEVSchemaError("usage must be an object")
    usage: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens"):
        value = usage_raw.get(key, 0)
        if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
            raise HydroJEVSchemaError(f"usage.{key} must be a non-negative integer")
        usage[key] = int(value)
    if not isinstance(attempts, Integral) or isinstance(attempts, bool) or int(attempts) < 1:
        raise HydroJEVSchemaError("attempts must be a positive integer")
    safe_timing = _json_safe(dict(timing_ms or {}), path="timing_ms")
    if not isinstance(safe_timing, dict):  # pragma: no cover - guarded by _json_safe
        raise HydroJEVSchemaError("timing_ms must be an object")

    model = body.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise HydroJEVSchemaError("response model must be a non-empty string")
    return JevDetectionDecision(
        alarm_probability=alarm_probability,
        event_type=str(selected_event),
        event_probabilities=event_probabilities,
        event_confidence=event_confidence,
        severity_score=float(raw_score),
        severity_probabilities=severity_probabilities,
        severity_legend=severity_legend,
        severity_confidence=severity_confidence,
        model=model,
        source=_source_value(source),
        usage=usage,
        timing_ms={str(key): float(value) for key, value in safe_timing.items()},
        attempts=int(attempts),
    )


@dataclass(frozen=True)
class HydroJEVResult:
    """Operational result of one HydroJEV evidence window."""

    state: DetectionState
    alarm_probability: float | None
    event_type: str | None
    event_probabilities: dict[str, float] | None
    severity_score: float | None
    severity_probabilities: dict[int, float] | None
    source: str
    model: str | None
    candidate: bool
    evidence_count: int
    alerting_count: int
    abstain_reason: str | None = None
    timing_ms: dict[str, float] | None = None
    usage: dict[str, int] | None = None
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "alarm_probability": self.alarm_probability,
            "event_type": self.event_type,
            "event_probabilities": (
                dict(self.event_probabilities) if self.event_probabilities is not None else None
            ),
            "severity_score": self.severity_score,
            "severity_probabilities": (
                {str(k): v for k, v in self.severity_probabilities.items()}
                if self.severity_probabilities is not None
                else None
            ),
            "source": self.source,
            "model": self.model,
            "candidate": self.candidate,
            "evidence_count": self.evidence_count,
            "alerting_count": self.alerting_count,
            "abstain_reason": self.abstain_reason,
            "timing_ms": dict(self.timing_ms or {}),
            "usage": dict(self.usage or {}),
            "attempts": self.attempts,
        }


def _state_quality(state: Mapping[str, Any]) -> tuple[bool, bool, int, int]:
    freshness = state.get("freshness", {})
    fresh = bool(freshness.get("is_fresh", False)) if isinstance(freshness, Mapping) else False
    detectors = state.get("detectors", {})
    if not isinstance(detectors, Mapping):
        return fresh, False, 0, 0
    usable = 0
    alerting = 0
    for entry in detectors.values():
        if not isinstance(entry, Mapping):
            continue
        current = entry.get("state")
        if current in {"nominal", "alerting"}:
            usable += 1
        if current == "alerting":
            alerting += 1
    return fresh, usable > 0, usable, alerting


def _candidate(state: Mapping[str, Any]) -> bool:
    routing = state.get("routing", {})
    return bool(routing.get("candidate", True)) if isinstance(routing, Mapping) else True


def _hydraulic_violation(state: Mapping[str, Any]) -> bool:
    hydraulic = state.get("hydraulic_consistency", {})
    return isinstance(hydraulic, Mapping) and hydraulic.get("within_tolerance") is False


def _cyber_signature(state: Mapping[str, Any]) -> bool:
    detectors = state.get("detectors", {})
    if not isinstance(detectors, Mapping):
        return False
    states = {str(name): (entry.get("state") if isinstance(entry, Mapping) else None) for name, entry in detectors.items()}
    return bool(
        states.get("rat_covariance_distortion") == "alerting"
        or (
            states.get("reconstruction_autoencoder") == "nominal"
            and states.get("cpdz_residual_normalized") == "alerting"
        )
        or states.get("state_estimator_residual") == "nominal"
        and states.get("cpdz_residual_normalized") == "alerting"
    )


def _result_from_common(
    *,
    state: DetectionState,
    source: str,
    candidate: bool,
    usable: int,
    alerting: int,
    alarm_probability: float | None = None,
    event_type: str | None = None,
    event_probabilities: dict[str, float] | None = None,
    severity_score: float | None = None,
    severity_probabilities: dict[int, float] | None = None,
    model: str | None = None,
    abstain_reason: str | None = None,
    timing_ms: Mapping[str, float] | None = None,
    usage: Mapping[str, int] | None = None,
    attempts: int = 0,
) -> HydroJEVResult:
    return HydroJEVResult(
        state=state,
        alarm_probability=alarm_probability,
        event_type=event_type,
        event_probabilities=event_probabilities,
        severity_score=severity_score,
        severity_probabilities=severity_probabilities,
        source=source,
        model=model,
        candidate=candidate,
        evidence_count=usable,
        alerting_count=alerting,
        abstain_reason=abstain_reason,
        timing_ms=dict(timing_ms or {}),
        usage=dict(usage or {}),
        attempts=attempts,
    )


def deterministic_no_jev(
    state: Mapping[str, Any], *, min_alerting_detectors: int = 2
) -> HydroJEVResult:
    """Return a transparent local-only decision from detector states."""

    if min_alerting_detectors < 1:
        raise ValueError("min_alerting_detectors must be positive")
    fresh, usable_ok, usable, alerting = _state_quality(state)
    candidate = _candidate(state)
    if not candidate:
        return _result_from_common(
            state=DetectionState.NORMAL, source="no_call", candidate=False, usable=usable, alerting=alerting
        )
    if not fresh:
        return _result_from_common(
            state=DetectionState.ABSTAIN,
            source="no_jev",
            candidate=True,
            usable=usable,
            alerting=alerting,
            abstain_reason="stale_evidence",
        )
    if not usable_ok:
        return _result_from_common(
            state=DetectionState.ABSTAIN,
            source="no_jev",
            candidate=True,
            usable=usable,
            alerting=alerting,
            abstain_reason="no_usable_detector_evidence",
        )
    is_fault = _hydraulic_violation(state)
    is_alarm = alerting >= min_alerting_detectors or is_fault
    return _result_from_common(
        state=DetectionState.ALARM if is_alarm else DetectionState.NORMAL,
        source="no_jev",
        candidate=True,
        usable=usable,
        alerting=alerting,
        event_type=(
            "physical_fault"
            if is_fault
            else ("cyber_attack" if is_alarm and _cyber_signature(state) else None)
        ),
    )


class HydroJEVDetector:
    """Fuse validated Jev judgments with deterministic evidence-quality gates."""

    def __init__(self, *, alarm_threshold: float = 0.80, normal_threshold: float = 0.20) -> None:
        if not 0.0 <= normal_threshold < alarm_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= normal_threshold < alarm_threshold <= 1")
        self.alarm_threshold = float(alarm_threshold)
        self.normal_threshold = float(normal_threshold)

    def predict_no_jev(self, state: Mapping[str, Any]) -> HydroJEVResult:
        return deterministic_no_jev(state)

    def failure_result(
        self, state: Mapping[str, Any], *, reason: str = "live_failure"
    ) -> HydroJEVResult:
        """Represent a failed or deliberately skipped live call as abstention.

        The caller must keep this result in the live provenance table.  It is
        intentionally not converted to a local alarm or a mock judgment.
        """

        _fresh, _usable_ok, usable, alerting = _state_quality(state)
        return _result_from_common(
            state=DetectionState.ABSTAIN,
            source="live_failure",
            candidate=_candidate(state),
            usable=usable,
            alerting=alerting,
            abstain_reason=str(reason),
        )

    def predict_response(
        self,
        state: Mapping[str, Any],
        body: Mapping[str, Any],
        *,
        source: Any = "live",
        timing_ms: Mapping[str, Any] | None = None,
        attempts: int = 1,
    ) -> HydroJEVResult:
        """Validate a raw response and fuse it; no network operation occurs."""

        if not _candidate(state):
            fresh, _, usable, alerting = _state_quality(state)
            del fresh
            return _result_from_common(
                state=DetectionState.NORMAL,
                source="no_call",
                candidate=False,
                usable=usable,
                alerting=alerting,
            )
        decision = parse_detection_response(
            body, source=source, timing_ms=timing_ms, attempts=attempts
        )
        return self.predict_decision(state, decision)

    def predict_decision(
        self, state: Mapping[str, Any], decision: JevDetectionDecision
    ) -> HydroJEVResult:
        """Fuse one already validated typed decision with the state."""

        if not isinstance(decision, JevDetectionDecision):
            raise TypeError("decision must be a JevDetectionDecision")
        fresh, usable_ok, usable, alerting = _state_quality(state)
        candidate = _candidate(state)
        if not candidate:
            return _result_from_common(
                state=DetectionState.NORMAL, source="no_call", candidate=False, usable=usable, alerting=alerting
            )
        common = dict(
            candidate=True,
            usable=usable,
            alerting=alerting,
            alarm_probability=decision.alarm_probability,
            event_type=decision.event_type,
            event_probabilities=decision.event_probabilities,
            severity_score=decision.severity_score,
            severity_probabilities=decision.severity_probabilities,
            model=decision.model,
            source=decision.source,
            timing_ms=decision.timing_ms,
            usage=decision.usage,
            attempts=decision.attempts,
        )
        if not fresh:
            return _result_from_common(
                state=DetectionState.ABSTAIN, abstain_reason="stale_evidence", **common
            )
        if not usable_ok:
            return _result_from_common(
                state=DetectionState.ABSTAIN,
                abstain_reason="no_usable_detector_evidence",
                **common,
            )
        high_alarm = decision.alarm_probability >= self.alarm_threshold
        low_alarm = decision.alarm_probability <= self.normal_threshold
        attack_type = decision.event_type in {"cyber_attack", "physical_fault"}
        contradictory = (high_alarm and decision.event_type == "normal_transient") or (
            low_alarm and attack_type
        )
        if contradictory:
            return _result_from_common(
                state=DetectionState.ABSTAIN,
                abstain_reason="contradictory_jev_answers",
                **common,
            )
        if high_alarm:
            return _result_from_common(state=DetectionState.ALARM, **common)
        if low_alarm:
            return _result_from_common(state=DetectionState.NORMAL, **common)
        if decision.event_type == "insufficient_evidence":
            return _result_from_common(
                state=DetectionState.ABSTAIN,
                abstain_reason="jev_insufficient_evidence",
                **common,
            )
        return _result_from_common(
            state=DetectionState.ABSTAIN,
            abstain_reason="alarm_probability_in_gray_zone",
            **common,
        )


@dataclass(frozen=True)
class DetectionEpisode:
    """Held-out episode labels paired with inference outputs."""

    episode_id: str
    is_attack: bool
    outputs: tuple[HydroJEVResult, ...]
    onset_index: int = 0
    step_interval_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not self.outputs:
            raise ValueError("outputs must not be empty")
        if not 0 <= self.onset_index < len(self.outputs):
            raise ValueError("onset_index must index outputs")
        if not _is_finite_number(self.step_interval_s) or float(self.step_interval_s) <= 0:
            raise ValueError("step_interval_s must be positive and finite")


def episode_metrics(episodes: Sequence[DetectionEpisode]) -> dict[str, Any]:
    """Compute JSON-safe episode recall, FPR, delay, abstention and provenance."""

    episodes = tuple(episodes)
    attacks = [episode for episode in episodes if episode.is_attack]
    benign = [episode for episode in episodes if not episode.is_attack]
    tp = 0
    fp = 0
    delays: list[float] = []
    abstentions = 0
    total_outputs = 0
    source_counts: dict[str, int] = {}
    per_episode: list[dict[str, Any]] = []
    for episode in episodes:
        outputs = episode.outputs[episode.onset_index :] if episode.is_attack else episode.outputs
        first_alarm = next((index for index, output in enumerate(outputs) if output.state is DetectionState.ALARM), None)
        responded = first_alarm is not None
        if episode.is_attack:
            tp += int(responded)
            if responded:
                delays.append(first_alarm * float(episode.step_interval_s))
        else:
            fp += int(responded)
        for output in episode.outputs:
            total_outputs += 1
            abstentions += int(output.state is DetectionState.ABSTAIN)
            source_counts[output.source] = source_counts.get(output.source, 0) + 1
        per_episode.append(
            {
                "episode_id": episode.episode_id,
                "is_attack": episode.is_attack,
                "responded": responded,
                "delay_s": (delays[-1] if episode.is_attack and responded else None),
            }
        )
    attack_count = len(attacks)
    benign_count = len(benign)
    return {
        "attack_scenarios": attack_count,
        "true_positives": tp,
        "false_negatives": attack_count - tp,
        "scenario_recall": (tp / attack_count if attack_count else None),
        "benign_scenarios": benign_count,
        "false_positives": fp,
        "benign_scenario_fp_rate": (fp / benign_count if benign_count else None),
        "mean_detection_delay_s": (float(np.mean(delays)) if delays else None),
        "abstention_rate": (abstentions / total_outputs if total_outputs else None),
        "source_counts": source_counts,
        "per_episode": per_episode,
    }

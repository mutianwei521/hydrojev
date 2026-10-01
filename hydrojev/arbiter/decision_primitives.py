"""Schema-exact TypeSafe Jev primitives for the HydroJEV three-question contract.

This module owns the *wire contract* only: it builds the batched request and
parses/validates the response into typed objects. It performs no I/O, so the
identical request and validation path is exercised by the live client, the
deterministic mock server, and the offline fallback.

The three questions map directly onto the paper's thesis. ``threat_cause``
discriminates the title's threat taxonomy -- stealthy cyber-physical attacks
(``adversarial_ae_evasion`` and ``hydraulic_fs_fdi``) versus hydraulic
catastrophes (``physical_burst``) versus benign variation (``normal_transient``)
-- a causal judgment a threshold detector cannot make. ``trip_edge_airgap`` is
the reflex decision, and ``severity_score`` triages response.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

# Question IDs are for code only; the API does not send them to the model, so
# every question carries its full meaning in instructions and criteria.
THREAT_CAUSE_ID = "threat_cause"
TRIP_AIRGAP_ID = "trip_edge_airgap"
SEVERITY_ID = "severity_score"
QUESTION_IDS = (THREAT_CAUSE_ID, TRIP_AIRGAP_ID, SEVERITY_ID)

DEFAULT_MODEL = "jev-latest"


class JevSchemaError(ValueError):
    """Raised when a response violates the documented Jev schema."""


class JevSource(str, Enum):
    """Provenance of a decision -- kept unmistakable in every result row."""

    LIVE = "live"
    MOCK = "mock"
    LOCAL_FALLBACK = "local_fallback"


class ThreatCause(str, Enum):
    """The HydroJEV threat taxonomy -- the choice options for ``threat_cause``."""

    ADVERSARIAL_AE_EVASION = "adversarial_ae_evasion"
    HYDRAULIC_FS_FDI = "hydraulic_fs_fdi"
    PHYSICAL_BURST = "physical_burst"
    NORMAL_TRANSIENT = "normal_transient"


# Rubrics tie each cause to its distinguishing *evidence signature*, so the live
# model (and the transparent mock) discriminate causes that look identical to a
# single-threshold detector. Edit here to change what the model is asked.
THREAT_CAUSE_CRITERIA: dict[str, str] = {
    ThreatCause.ADVERSARIAL_AE_EVASION.value: (
        "An adversary has crafted sensor measurements to keep the reconstruction "
        "autoencoder's error below its alarm threshold while the true hydraulic "
        "state is anomalous. Indicated when the learned reconstruction detector "
        "reads nominal (below threshold) yet independent physics disagrees -- the "
        "normalized CPDZ residual and/or the residual covariance-structure "
        "detector are alerting and reported measurements are mutually "
        "inconsistent. The learned anomaly detector is being deceived even though "
        "orthogonal physical checks fire."
    ),
    ThreatCause.HYDRAULIC_FS_FDI.value: (
        "A feasible, stealthy false-data-injection attack that satisfies the "
        "operator's state-estimator residual and CUSUM/chi-square checks (so those "
        "read nominal or only marginal) yet is engineered to force a harmful "
        "physical consequence -- typically inflated flow or head that drives "
        "excess pumping energy. Indicated when state-estimation detectors are "
        "bypassed (low) but an independent energy or mass-balance accounting shows "
        "the pumps working harder than the served demand justifies."
    ),
    ThreatCause.PHYSICAL_BURST.value: (
        "A genuine hydraulic fault such as a pipe burst or large leak. Indicated "
        "by a pressure drop with increased inflow that stays physically "
        "self-consistent: the mass-balance residual and actuator states are "
        "coherent with a real loss of water, and the anomaly is large enough that "
        "the reconstruction detector responds rather than being suppressed."
    ),
    ThreatCause.NORMAL_TRANSIENT.value: (
        "Ordinary operational variation -- demand change, scheduled valve or pump "
        "switching, or measurement noise -- with no coherent, multi-detector, "
        "cross-physics evidence of manipulation or fault. Supported when detectors "
        "are within threshold or show only a single weak deviation and, where "
        "relevant, maintenance context explains the change."
    ),
}

AIRGAP_CRITERIA: dict[str, str] = {
    "true": (
        "The evidence justifies immediately air-gapping this zone's edge "
        "controller from the SCADA network -- cutting remote-control authority -- "
        "because manipulation or a severe hydraulic threat is corroborated by "
        "multiple independent detectors on fresh data and the physical cost of "
        "inaction is high."
    ),
    "false": (
        "The evidence does not justify cutting remote-control authority: it is "
        "weak, single-source, stale, or adequately explained by benign operation, "
        "so an unnecessary isolation would cost more than the residual risk. "
        "Assume deterministic hydraulic interlocks, data-freshness gates, and "
        "operator confirmation are enforced separately in code and are not your "
        "responsibility."
    ),
}

# Five ordered levels (0-based on the wire; a 1-5 label is exposed for operators).
SEVERITY_LEGEND: tuple[str, ...] = (
    "Nominal. All detectors within threshold and observations match predictions.",
    "Minor anomaly. A single detector deviates slightly; no physical "
    "inconsistency; service unaffected.",
    "Moderate concern. Multiple detectors deviate or mass balance is violated; "
    "degradation is plausible but not yet damaging.",
    "Serious incident. Corroborated evidence of manipulation or hydraulic fault "
    "with credible risk of service loss or equipment damage.",
    "Critical emergency. Imminent or ongoing damage such as water hammer, pump "
    "destruction, contamination ingress, or wide supply loss.",
)

_SEVERITY_INSTRUCTIONS = (
    "Rate the severity of the current situation for this water distribution zone, "
    "considering potential service disruption, equipment damage, and water-quality "
    "risk implied by the state."
)
_THREAT_INSTRUCTIONS = (
    "Using only the evidence in the state, identify the single most likely cause "
    "of the current discrepancy between observed and surrogate-predicted values "
    "and the alerting detector entries for this water distribution zone."
)
_AIRGAP_INSTRUCTIONS = (
    "Should the edge controller for this zone be physically air-gapped from the "
    "SCADA network right now, cutting remote-control authority? Weigh the evidence "
    "quality across the detectors, the data age in freshness, and the physical "
    "consistency evidence."
)


def build_questions() -> dict[str, dict[str, Any]]:
    """Return the deterministic three-question payload for one request."""

    return {
        THREAT_CAUSE_ID: {
            "type": "choice",
            "instructions": _THREAT_INSTRUCTIONS,
            "criteria": dict(THREAT_CAUSE_CRITERIA),
        },
        TRIP_AIRGAP_ID: {
            "type": "noul",
            "instructions": _AIRGAP_INSTRUCTIONS,
            "criteria": dict(AIRGAP_CRITERIA),
        },
        SEVERITY_ID: {
            "type": "score",
            "instructions": _SEVERITY_INSTRUCTIONS,
            "criteria": list(SEVERITY_LEGEND),
        },
    }


def build_request(state: Mapping[str, Any], *, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Assemble the full ``POST /v1/systemone`` request body."""

    return {"state": dict(state), "model": model, "questions": build_questions()}


# --------------------------------------------------------------------------- #
# Typed answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChoiceAnswer:
    selected: str
    probabilities: dict[str, float]
    confidence: float

    def as_threat_cause(self) -> ThreatCause:
        return ThreatCause(self.selected)


@dataclass(frozen=True)
class NoulAnswer:
    """Probability of *yes*. Noul carries no separate confidence by design."""

    probability: float


@dataclass(frozen=True)
class ScoreAnswer:
    expected_level: float  # 0-based probability-weighted mean
    probabilities: dict[int, float]
    legend: dict[int, str]
    confidence: float

    @property
    def most_likely_level(self) -> int:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def user_facing_level(self) -> float:
        """1-5 presentation of the 0-4 expected level (engineering mapping only)."""

        return self.expected_level + 1.0


@dataclass(frozen=True)
class JevUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class JevDecision:
    threat_cause: ChoiceAnswer
    trip_edge_airgap: NoulAnswer
    severity: ScoreAnswer
    model: str
    usage: JevUsage
    source: JevSource
    timing_ms: dict[str, float]
    attempts: int


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JevSchemaError(message)


def _as_probability(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    _require(0.0 - 1e-9 <= number <= 1.0 + 1e-9, f"{label} must lie in [0, 1], got {number}")
    return min(max(number, 0.0), 1.0)


# The live API serializes probabilities with two decimals, so a valid
# distribution can display as 0.99 or 1.01; accept that bounded rounding
# (same tolerance as hydrojev.methods.hydrojev_detector) and renormalize.
_SUM_TOLERANCE = 2.5e-2


def _require_sums_to_one(probabilities: Mapping[Any, float], label: str) -> None:
    total = sum(probabilities.values())
    _require(total > 0.0 and abs(total - 1.0) < _SUM_TOLERANCE, f"{label} must sum to 1, got {total:.6f}")


def _normalized(probabilities: Mapping[Any, float]) -> dict[Any, float]:
    total = sum(probabilities.values())
    return {k: v / total for k, v in probabilities.items()}


def _parse_choice(answer: Mapping[str, Any]) -> ChoiceAnswer:
    _require(answer.get("type") == "choice", "threat_cause answer must have type 'choice'")
    raw_probabilities = answer.get("probabilities")
    _require(isinstance(raw_probabilities, Mapping), "choice answer needs a probabilities map")
    probabilities = {str(k): _as_probability(v, f"choice probability {k}") for k, v in raw_probabilities.items()}
    expected = {cause.value for cause in ThreatCause}
    _require(
        set(probabilities) == expected,
        f"choice options {sorted(probabilities)} must equal the threat taxonomy {sorted(expected)}",
    )
    _require_sums_to_one(probabilities, "choice probabilities")
    probabilities = _normalized(probabilities)
    selected = answer.get("choice")
    _require(selected in expected, f"selected choice {selected!r} is not in the threat taxonomy")
    confidence = _as_probability(answer.get("confidence"), "choice confidence")
    return ChoiceAnswer(selected=str(selected), probabilities=probabilities, confidence=confidence)


def _parse_noul(answer: Mapping[str, Any]) -> NoulAnswer:
    _require(answer.get("type") == "noul", "trip_edge_airgap answer must have type 'noul'")
    _require("confidence" not in answer, "noul answers must not carry a confidence field")
    return NoulAnswer(probability=_as_probability(answer.get("noul"), "noul"))


def _parse_score(answer: Mapping[str, Any]) -> ScoreAnswer:
    _require(answer.get("type") == "score", "severity_score answer must have type 'score'")
    raw_probabilities = answer.get("probabilities")
    _require(isinstance(raw_probabilities, Mapping), "score answer needs a probabilities map")
    try:
        probabilities = {int(k): _as_probability(v, f"score probability {k}") for k, v in raw_probabilities.items()}
    except (TypeError, ValueError) as exc:
        raise JevSchemaError(f"score probability keys must be integer indices: {exc}") from exc
    expected_indices = set(range(len(SEVERITY_LEGEND)))
    _require(
        set(probabilities) == expected_indices,
        f"score levels {sorted(probabilities)} must be the 0-based range {sorted(expected_indices)}",
    )
    _require_sums_to_one(probabilities, "score probabilities")
    probabilities = _normalized(probabilities)
    raw_legend = answer.get("legend", {})
    _require(isinstance(raw_legend, Mapping), "score legend must be a map")
    legend = {int(k): str(v) for k, v in raw_legend.items()}
    score = answer.get("score")
    _require(isinstance(score, (int, float)) and not isinstance(score, bool), "score must be numeric")
    confidence = _as_probability(answer.get("confidence"), "score confidence")
    return ScoreAnswer(
        expected_level=float(score),
        probabilities=probabilities,
        legend=legend,
        confidence=confidence,
    )


def parse_response(
    body: Mapping[str, Any],
    *,
    source: JevSource,
    timing_ms: Mapping[str, float] | None = None,
    attempts: int = 1,
) -> JevDecision:
    """Validate a raw response body and return a typed, provenance-tagged decision."""

    _require(isinstance(body, Mapping), "response body must be a JSON object")
    answers = body.get("answers")
    _require(isinstance(answers, Mapping), "response must contain an answers object")
    missing = sorted(set(QUESTION_IDS) - set(answers))
    _require(not missing, f"response omitted question ids {missing}")

    usage_raw = body.get("usage", {})
    _require(isinstance(usage_raw, Mapping), "usage must be an object")
    usage = JevUsage(
        input_tokens=int(usage_raw.get("input_tokens", 0)),
        output_tokens=int(usage_raw.get("output_tokens", 0)),
    )

    return JevDecision(
        threat_cause=_parse_choice(answers[THREAT_CAUSE_ID]),
        trip_edge_airgap=_parse_noul(answers[TRIP_AIRGAP_ID]),
        severity=_parse_score(answers[SEVERITY_ID]),
        model=str(body.get("model", "")),
        usage=usage,
        source=source,
        timing_ms=dict(timing_ms or {}),
        attempts=int(attempts),
    )

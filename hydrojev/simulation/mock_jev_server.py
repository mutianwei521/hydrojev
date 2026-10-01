"""Deterministic, evidence-only mock of the TypeSafe Jev arbiter.

Purpose in the paper. The mock is a transparent surrogate System-One arbiter
that lets the entire HydroJEV closed loop run and reproduce offline, with no API
key and no network. It is **not** presented as equal to the live Jev; every
result row records its ``jev_source`` so live, mock, and fallback decisions stay
unmistakable.

Why the results are still meaningful. The mock reasons *only from the evidence
features* in the state -- detector states, hydraulic consistency, energy
accounting, freshness -- exactly as the real model would. It never reads a
ground-truth label; :func:`_assert_evidence_only` enforces this in code. So the
recall and isolation numbers obtained offline measure the *discriminability of
HydroJEV's orthogonal residual features*, which is itself a contribution, rather
than a circular replay of hidden answers.

The scoring is a small, auditable rule set with named weights (module constants).
Every judgment is a deterministic function of the state, so identical evidence
always yields identical decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:  # FastAPI is optional at import time so the decision core runs on minimal installs.
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    _FASTAPI_AVAILABLE = False

from hydrojev.arbiter.decision_primitives import (
    DEFAULT_MODEL,
    SEVERITY_LEGEND,
    JevDecision,
    JevSource,
    ThreatCause,
    parse_response,
)

# Canonical detector names the mock understands (others are ignored gracefully).
_CPDZ_RESIDUAL = "cpdz_residual_normalized"
_RECONSTRUCTION_AE = "reconstruction_autoencoder"
_STATE_ESTIMATOR = ("state_estimator_residual", "vectorized_cusum", "chi_square_residual")
_COVARIANCE = "rat_covariance_distortion"
_DRIFT = "gepfm_baseline_drift"

# Any state key exactly matching one of these would leak a hidden answer.
_FORBIDDEN_LABEL_KEYS = frozenset(
    {"ground_truth", "true_cause", "true_label", "label", "attack_label",
     "scenario_label", "expected_cause", "y_true"}
)

_ENERGY_EPS = 0.05  # extra-energy fraction above which over-pumping is "material"
_SEVERITY_SIGMA = 0.6  # spread of the severity level distribution
_SEVERITY_LEVELS = np.arange(len(SEVERITY_LEGEND), dtype=float)


def _assert_evidence_only(state: Mapping[str, Any]) -> None:
    """Guard the scientific invariant: the arbiter must not see a ground-truth label."""

    def scan(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key).lower() in _FORBIDDEN_LABEL_KEYS:
                    raise ValueError(
                        f"state contains forbidden ground-truth key {key!r}; the "
                        "arbiter must reason from evidence only"
                    )
                scan(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                scan(item)

    scan(state)


@dataclass(frozen=True)
class _Signals:
    physical_alert: bool
    ae_present: bool
    ae_alert: bool
    ae_suppressed: bool
    se_present: bool
    se_alert: bool
    se_bypassed: bool
    cov_alert: bool
    drift_alert: bool
    mass_violated: bool
    actuator_inconsistent: bool
    energy_inflation: float
    is_fresh: bool

    @property
    def n_alerts(self) -> int:
        return sum(
            [
                self.physical_alert,
                self.ae_alert,
                self.se_alert,
                self.cov_alert,
                self.drift_alert,
                self.mass_violated,
            ]
        )

    @property
    def cyber_signature(self) -> bool:
        # A cyber-manipulation signature requires corroboration: a *suppressed*
        # learned detector only counts as evidence when physics disagrees, and a
        # *bypassed* state estimator only when pumps are demonstrably over-working.
        # A nominal detector in a quiet zone (normal transient) is not a signature.
        return (
            self.cov_alert
            or (self.ae_suppressed and self.physical_alert)
            or (self.se_bypassed and self.energy_inflation > _ENERGY_EPS)
        )


def _detector_state(detectors: Mapping[str, Any], name: str) -> str | None:
    entry = detectors.get(name)
    if not isinstance(entry, Mapping):
        return None
    state = entry.get("state")
    return str(state) if state is not None else None


def extract_signals(state: Mapping[str, Any]) -> _Signals:
    """Reduce a defence-zone state to the boolean/continuous evidence signals."""

    detectors = state.get("detectors", {})
    if not isinstance(detectors, Mapping):
        detectors = {}

    ae_state = _detector_state(detectors, _RECONSTRUCTION_AE)
    se_states = [s for s in (_detector_state(detectors, n) for n in _STATE_ESTIMATOR) if s is not None]

    hydraulic = state.get("hydraulic_consistency", {})
    if not isinstance(hydraulic, Mapping):
        hydraulic = {}
    within_tolerance = hydraulic.get("within_tolerance", True)
    energy_inflation = hydraulic.get("extra_energy_fraction", 0.0)
    try:
        energy_inflation = float(energy_inflation) if energy_inflation is not None else 0.0
    except (TypeError, ValueError):
        energy_inflation = 0.0

    freshness = state.get("freshness", {})
    is_fresh = bool(freshness.get("is_fresh", True)) if isinstance(freshness, Mapping) else True

    return _Signals(
        physical_alert=_detector_state(detectors, _CPDZ_RESIDUAL) == "alerting",
        ae_present=ae_state is not None,
        ae_alert=ae_state == "alerting",
        ae_suppressed=ae_state == "nominal",
        se_present=bool(se_states),
        se_alert=any(s == "alerting" for s in se_states),
        se_bypassed=bool(se_states) and not any(s == "alerting" for s in se_states),
        cov_alert=_detector_state(detectors, _COVARIANCE) == "alerting",
        drift_alert=_detector_state(detectors, _DRIFT) == "alerting",
        mass_violated=within_tolerance is False,
        actuator_inconsistent=hydraulic.get("reported_pump_state_is_self_consistent", True) is False,
        energy_inflation=energy_inflation,
        is_fresh=is_fresh,
    )


def _threat_logits(s: _Signals) -> dict[str, float]:
    """Additive, auditable evidence scores per cause (pre-softmax)."""

    adversarial = 0.0
    adversarial += 2.6 if (s.ae_suppressed and s.physical_alert) else 0.0
    adversarial += 1.4 if s.cov_alert else 0.0
    adversarial += 0.6 if s.drift_alert else 0.0
    adversarial += 0.4 if s.mass_violated else 0.0
    adversarial -= 1.2 if s.ae_alert else 0.0

    fs_fdi = 0.0
    fs_fdi += 2.2 if (s.se_bypassed and (s.energy_inflation > _ENERGY_EPS or s.physical_alert)) else 0.0
    fs_fdi += min(2.0, 3.0 * s.energy_inflation) if s.energy_inflation > _ENERGY_EPS else 0.0
    fs_fdi += 0.6 if (not s.mass_violated and s.physical_alert) else 0.0
    fs_fdi -= 1.2 if s.se_alert else 0.0

    burst = 0.0
    burst += 2.2 if (s.mass_violated and s.ae_alert) else 0.0
    burst += 1.0 if (s.physical_alert and s.ae_alert) else 0.0
    burst += 0.5 if (s.mass_violated and not s.actuator_inconsistent) else 0.0
    burst -= 1.6 if s.ae_suppressed else 0.0
    burst -= 0.6 if (s.se_bypassed and s.energy_inflation > _ENERGY_EPS) else 0.0

    normal = 0.0
    if s.n_alerts == 0:
        normal += 2.6
    elif s.n_alerts == 1:
        normal += 0.8
    normal -= 0.9 * s.n_alerts

    return {
        ThreatCause.ADVERSARIAL_AE_EVASION.value: adversarial,
        ThreatCause.HYDRAULIC_FS_FDI.value: fs_fdi,
        ThreatCause.PHYSICAL_BURST.value: burst,
        ThreatCause.NORMAL_TRANSIENT.value: normal,
    }


def _softmax(logits: Mapping[str, float]) -> dict[str, float]:
    keys = list(logits)
    values = np.array([logits[k] for k in keys], dtype=float)
    shifted = np.exp(values - values.max())
    weights = shifted / shifted.sum()
    return {k: float(round(w, 6)) for k, w in zip(keys, weights)}


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _airgap_noul(s: _Signals) -> float:
    cyber = sum([s.ae_suppressed, s.cov_alert, s.se_bypassed and s.energy_inflation > _ENERGY_EPS])
    physical = sum([s.physical_alert, s.mass_violated, s.energy_inflation > 0.2])
    corroboration = cyber + physical
    consequence = min(1.0, max(0.0, s.energy_inflation))
    # Centered so ~2.5 independent corroborations put the probability near 0.5.
    raw = 1.2 * corroboration + 2.0 * consequence - 3.0
    noul = _sigmoid(raw)
    if not s.is_fresh:
        # Lean against isolating on stale evidence; a hard freshness gate in the
        # closed-loop code enforces this regardless of the model's probability.
        noul *= 0.4
    return float(round(noul, 6))


def _severity_distribution(s: _Signals) -> tuple[float, dict[str, float], float]:
    level = 0.0
    level += 1.0 if s.physical_alert else 0.0
    level += 1.0 if s.cyber_signature else 0.0
    level += 1.0 if s.mass_violated else 0.0
    level += 1.0 if s.energy_inflation > 0.2 else 0.0
    level += 1.0 if s.energy_inflation > 0.5 else 0.0
    level = float(np.clip(level, 0.0, len(SEVERITY_LEGEND) - 1))

    weights = np.exp(-0.5 * ((_SEVERITY_LEVELS - level) / _SEVERITY_SIGMA) ** 2)
    probabilities = weights / weights.sum()
    expected = float(np.sum(_SEVERITY_LEVELS * probabilities))
    prob_map = {str(i): float(round(p, 6)) for i, p in enumerate(probabilities)}
    confidence = float(round(probabilities.max(), 6))
    return round(expected, 6), prob_map, confidence


def arbitrate_state(
    state: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    source: str = JevSource.MOCK.value,
) -> dict[str, Any]:
    """Deterministically map a defence-zone state to a raw Jev response body."""

    _assert_evidence_only(state)
    signals = extract_signals(state)

    probabilities = _softmax(_threat_logits(signals))
    selected = max(probabilities, key=probabilities.__getitem__)
    choice_confidence = float(round(probabilities[selected], 6))

    noul = _airgap_noul(signals)
    score, score_probabilities, score_confidence = _severity_distribution(signals)

    # Deterministic, non-secret usage estimate from payload size.
    input_tokens = max(1, len(json.dumps(state, default=str)) // 4)

    return {
        "model": model,
        "source": source,
        "answers": {
            "threat_cause": {
                "type": "choice",
                "choice": selected,
                "probabilities": probabilities,
                "confidence": choice_confidence,
            },
            "trip_edge_airgap": {"type": "noul", "noul": noul},
            "severity_score": {
                "type": "score",
                "score": score,
                "legend": {str(i): SEVERITY_LEGEND[i] for i in range(len(SEVERITY_LEGEND))},
                "probabilities": score_probabilities,
                "confidence": score_confidence,
            },
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 24},
    }


def decide_locally(
    state: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    source: JevSource = JevSource.MOCK,
) -> JevDecision:
    """Run the mock arbiter and return a validated, typed decision.

    Used both by the mock HTTP server's callers and by the client's offline
    fallback, so the local path is validated by the same parser as the live path.
    """

    body = arbitrate_state(state, model=model, source=source.value)
    return parse_response(body, source=source, timing_ms={"local_ms": 0.0}, attempts=1)


def create_app(*, model: str = DEFAULT_MODEL):
    """Build the FastAPI app exposing the mock at ``POST /v1/systemone``.

    FastAPI is an optional dependency: the pure decision functions and the
    offline fallback import this module without it. The FastAPI symbols are
    resolved at module import (see the top-level guarded import) so that the
    endpoint's stringized ``Request`` annotation -- ``from __future__ import
    annotations`` defers all annotations to strings -- binds to the raw request
    body instead of being mistaken for a query parameter.
    """

    if not _FASTAPI_AVAILABLE:  # pragma: no cover - exercised only on minimal installs
        raise RuntimeError(
            "FastAPI is required to serve the mock arbiter; install it with "
            "`pip install fastapi uvicorn`. The offline decision path "
            "(decide_locally) does not need it."
        )

    app = FastAPI(title="HydroJEV Mock Jev", version="1.0")

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:  # pragma: no cover - thin shim
        body = await request.json()
        state = body.get("state", {})
        req_model = body.get("model", model)
        try:
            result = arbitrate_state(state, model=req_model, source=JevSource.MOCK.value)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": {"message": str(exc)}})
        return JSONResponse(status_code=200, content=result)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:  # pragma: no cover - trivial
        return {"status": "ok", "source": JevSource.MOCK.value}

    return app

"""Pre-registered Jev pilot P1: benign false-alarm filtering on BATADAL.

Protocol: ``docs/jev/PILOT_PROTOCOL_v1.md``. The front end (GRU surrogate,
CPDZ/GEpFM/RAT families, Mahalanobis/PCA/AE/Isolation Forest) is the frozen
v2 front end fitted on the attack-free dataset03 training rows. This module
adds the evidence state v2 that the v2 benchmark lacked:

* the five channels with the largest surrogate residuals, with observed and
  predicted values in engineering units;
* physical-consistency checks that SCADA can verify on its own (pump
  status versus flow, tank level change versus pump/valve flows, junction
  pressure versus operation, tank level envelope);
* operating context (hour, weekday, pump states atypical for this hour);
* the detector agreement pattern.

Regression models are fitted on the dataset03 training rows; every check and
residual scale is calibrated on the held-out attack-free dataset03 rows
(7,008 onward). The evaluated streams contribute nothing to fitting except
the R2 comparator, which is trained on dev labels by design (a label
advantage for the comparator, not for Jev).

The candidate gate is loose (any detector alerts in the causal six-hour
window) and Jev holds the veto: final alarm = candidate AND Jev alarm.
Labels never enter a request; they are stored only in ``prepared.json`` for
scoring. Live calls go through ``tools/jev/invoke-jev.ps1`` with the key
passed to the child process environment only, and every call is appended to
a ledger that enforces the pilot's hard budget.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks.hydrojev_detection_benchmark import (
    DETECTOR_ORDER,
    INSTANTANEOUS,
    MIN_CANDIDATE_ALERTS,
    STRIDE,
    TEMPORAL,
    WARMUP,
    FrontEnd,
    _runs_metrics,
    _safe,
    fit_front_end,
    score_stream,
)
from hydrojev.benchmarks.named_family_benchmark import one_step_residual_stream
from hydrojev.config import HydroJEVConfig, load_config
from hydrojev.datasets.wdn_loader import load_batadal
from hydrojev.methods.hydrojev_detector import (
    HydroJEVSchemaError,
    _assert_no_ground_truth,
    _distribution,
    _is_finite_number,
    _json_safe,
    _probability,
)

SCHEMA_VERSION = "hydrojev.detection_state/2"
PROTOCOL = "docs/jev/PILOT_PROTOCOL_v1.md"
CALIBRATION_START = 7008  # dataset03 rows >= this are held-out attack-free rows
CHECK_WINDOW_QUANTILE = 0.99  # each check family: 1% window false-alarm on held-out normal data
RESIDUAL_SIGMA_RUN = 3.0
TOP_RESIDUALS = 5
PILOT_CALL_CAP = 2000  # raised from 400 on 2026-09-24: author authorised the extra cost (protocol v2.1)
DEFAULT_OUTDIR = Path("artifacts/jev_pilot_p1")
DEFAULT_SAMPLE_SEED = 20260924
# dataset03 is published at full precision, dataset04 and test_dataset at two decimals;
# every residual scale is floored at that reporting resolution
EVALUATED_RESOLUTION = 0.01

EVENT_TYPES_V2 = (
    "cyber_attack",
    "physical_fault",
    "normal_transient",
    "sensor_fault",
    "insufficient_evidence",
)
QUESTION_IDS = ("alarm", "event_type", "severity")
DETECTOR_FAMILY = {
    "Mahalanobis": "instantaneous_statistical",
    "PCA residual": "instantaneous_statistical",
    "Isolation Forest": "instantaneous_statistical",
    "HydroJEV AE": "instantaneous_learned",
    "CPDZ residual": "surrogate_one_step_residual",
    "GEpFM drift": "temporal_drift",
    "RAT covariance": "temporal_covariance",
}
PHYSICAL_CHECKS = (
    "pump_status_vs_flow",
    "tank_level_change_vs_flows",
    "junction_pressure_vs_operation",
    "tank_level_envelope",
)
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SYSTEM_DESCRIPTION = (
    "C-Town benchmark water distribution network, hourly SCADA. Channels: L_T1..L_T7 tank levels (m); "
    "F_PU1..F_PU11 pump flows (L/s) and S_PU1..S_PU11 pump on/off status (1 = on); F_V2/S_V2 valve flow "
    "and status; P_J* junction pressures (m). Pumps are switched by programmable logic controllers from "
    "tank-level readings, so a forged level reading can make pumps run or stop without a real change in "
    "water level; each sensor reading and each actuator command passes through a network link that an "
    "attacker could alter. 'predicted' values come from a recurrent surrogate trained only on attack-free "
    "operation; physical checks are regression or rule models also fitted only on attack-free operation."
)

ALARM_CRITERIA_V2 = {
    "true": (
        "Raise an anomaly event: the window shows a coherent departure from normal operation that is "
        "worth an operator's attention, supported by physical-consistency violations, large surrogate "
        "residuals on physically related channels, or several independent detector families agreeing."
    ),
    "false": (
        "Do not raise an event: the window is explained by ordinary operation. Typical signs are a single "
        "detector (or detectors of one family) slightly above threshold, all physical checks consistent, "
        "small or scattered residuals, and pump states that are usual for this hour. Detectors alerting "
        "only marginally is common in normal operation."
    ),
}

# Debug round 2 (protocol section 2, second and last debug round): the alarm is defined
# as the operational consequence of the explanation, so it cannot contradict event_type.
ALARM_INSTRUCTIONS_R2 = (
    "You are reviewing one six-hour window of a water distribution SCADA system that at least one anomaly "
    "detector flagged. In this system most flagged windows come from ordinary operation, and even a single "
    "physical-consistency violation also occurs in ordinary operation (see each check's "
    "benign_flagged_window_violation_rate). Using only the supplied evidence, decide whether the window "
    "should be raised to the operator. Raise it only if control or telemetry manipulation, or a real "
    "hydraulic fault, is more likely than ordinary operation; the answer must agree with your judgement "
    "of the most likely explanation."
)
ALARM_CRITERIA_R2 = {
    "true": (
        "Raise: manipulation or a real hydraulic fault is the most likely explanation. The evidence goes "
        "beyond what flagged ordinary windows show, e.g. several physically related checks violated together, "
        "violations far above their thresholds, sustained structured residuals on channels that drive pump "
        "control, or pump behaviour that contradicts the reported levels."
    ),
    "false": (
        "Do not raise: ordinary operation or a demand transient is the most likely explanation (including "
        "an isolated instrument problem without hydraulic consequence). Typical signs are marginal "
        "exceedances, a single violated check of the kind that flagged ordinary windows also show, "
        "scattered residuals, and usual pump states for this hour."
    ),
}
BENIGN_RATE_NOTE = (
    "benign_flagged_window_violation_rate is the fraction of flagged development windows that were ordinary "
    "operation in which this check was violated; normal_operation_window_violation_rate is measured on "
    "unflagged attack-free calibration data and is therefore much lower"
)

EVENT_CRITERIA_V2 = {
    "cyber_attack": (
        "Telemetry or control manipulation: e.g. a tank-level or pump reading that contradicts the flows, "
        "pump states or pressures that should follow from it; pumps operating in a way inconsistent with "
        "the reported levels; a sustained, structured residual on channels an attacker could forge."
    ),
    "physical_fault": (
        "A real hydraulic fault such as a burst or leak: physically coherent loss of pressure or level "
        "with increased pumping, while telemetry stays mutually consistent."
    ),
    "normal_transient": (
        "Ordinary operation or demand variation: physical checks consistent, residuals small or explained "
        "by a usual pump switch at this hour, and at most marginal detector exceedances."
    ),
    "sensor_fault": (
        "A single instrument is stuck, drifting or spiking: one channel is anomalous while the physically "
        "related channels (flows, pressures, other tanks) behave normally and no actuator acts on it."
    ),
    "insufficient_evidence": (
        "Use only when the supplied evidence is missing, stale or genuinely contradictory so that none of "
        "the other four explanations can be preferred."
    ),
}

SEVERITY_LEGEND_V2 = (
    "Nominal operation; nothing to act on.",
    "Minor anomaly; isolated deviation without service or equipment impact.",
    "Moderate concern; a physical-consistency check is violated or several detector families agree.",
    "Serious incident; corroborated manipulation or fault with credible service or equipment risk.",
    "Critical emergency; ongoing or imminent overflow, supply loss or equipment damage is plausible.",
)


def build_questions_v2(revision: int = 1) -> dict[str, dict[str, Any]]:
    """Question contract v2 (self-contained instructions; ids are not sent to the model).

    ``revision=1`` is the frozen round-1 contract; ``revision=2`` only replaces the alarm question.
    """

    if revision not in (1, 2):
        raise ValueError(f"unknown question revision {revision}")
    questions = {
        "alarm": {
            "type": "noul",
            "instructions": (
                "You are reviewing one six-hour window of a water distribution SCADA system that at least one "
                "anomaly detector flagged. Most such flags in normal operation are false alarms. Using only the "
                "supplied evidence, decide whether this window should be raised to the operator as an anomaly "
                "event."
            ),
            "criteria": dict(ALARM_CRITERIA_V2),
        },
        "event_type": {
            "type": "choice",
            "instructions": (
                "Using only the supplied evidence for this six-hour SCADA window, choose the most likely "
                "explanation of what the detectors and physical checks show."
            ),
            "criteria": dict(EVENT_CRITERIA_V2),
        },
        "severity": {
            "type": "score",
            "instructions": (
                "Rate the operational severity of this six-hour SCADA window using only the supplied evidence."
            ),
            "criteria": list(SEVERITY_LEGEND_V2),
        },
    }
    if revision == 2:
        questions["alarm"] = {"type": "noul", "instructions": ALARM_INSTRUCTIONS_R2, "criteria": dict(ALARM_CRITERIA_R2)}
    return questions


def build_request_v2(state: Mapping[str, Any], *, model: str = "jev-latest", revision: int = 1) -> dict[str, Any]:
    _assert_no_ground_truth(state)
    request = {"model": model, "state": _json_safe(dict(state), path="state"), "questions": build_questions_v2(revision)}
    json.dumps(request, ensure_ascii=False, allow_nan=False)
    return request


def canonical_bytes(request: Mapping[str, Any]) -> bytes:
    return json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class PilotDecision:
    alarm_probability: float
    event_type: str
    event_probabilities: dict[str, float]
    event_confidence: float
    severity_score: float
    model: str
    usage: dict[str, int]


def parse_response_v2(body: Mapping[str, Any]) -> PilotDecision:
    """Validate a raw System One response against question contract v2."""

    if not isinstance(body, Mapping) or not isinstance(body.get("answers"), Mapping):
        raise HydroJEVSchemaError("response must contain an answers object")
    answers = body["answers"]
    missing = sorted(set(QUESTION_IDS) - set(answers))
    if missing:
        raise HydroJEVSchemaError(f"response omitted question ids {missing}")
    alarm = answers["alarm"]
    if not isinstance(alarm, Mapping) or alarm.get("type") != "noul":
        raise HydroJEVSchemaError("alarm answer must have type 'noul'")
    p_alarm = _probability(alarm.get("noul"), label="alarm.noul")
    event = answers["event_type"]
    if not isinstance(event, Mapping) or event.get("type") != "choice":
        raise HydroJEVSchemaError("event_type answer must have type 'choice'")
    probs = _distribution(
        {str(k): v for k, v in dict(event.get("probabilities") or {}).items()},
        expected_keys=set(EVENT_TYPES_V2),
        label="event_type.probabilities",
    )
    choice = event.get("choice")
    if choice not in EVENT_TYPES_V2:
        raise HydroJEVSchemaError("event_type.choice is outside the v2 taxonomy")
    severity = answers["severity"]
    if not isinstance(severity, Mapping) or severity.get("type") != "score":
        raise HydroJEVSchemaError("severity answer must have type 'score'")
    score = severity.get("score")
    if not _is_finite_number(score) or not 0.0 <= float(score) <= 4.0:
        raise HydroJEVSchemaError("severity score must lie in [0, 4]")
    _distribution(
        {int(k): v for k, v in dict(severity.get("probabilities") or {}).items()},
        expected_keys=set(range(5)),
        label="severity.probabilities",
    )
    model = body.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise HydroJEVSchemaError("response model must be a non-empty string")
    usage_raw = body.get("usage", {}) if isinstance(body.get("usage", {}), Mapping) else {}
    usage = {k: int(usage_raw.get(k, 0) or 0) for k in ("input_tokens", "output_tokens")}
    return PilotDecision(
        alarm_probability=p_alarm,
        event_type=str(choice),
        event_probabilities=probs,
        event_confidence=_probability(event.get("confidence"), label="event_type.confidence"),
        severity_score=float(score),
        model=model,
        usage=usage,
    )


# --------------------------------------------------------------------------- #
# Physical reference (fit on dataset03 train rows, calibrated on held-out rows)
# --------------------------------------------------------------------------- #


def _hour_onehot(hours: np.ndarray) -> np.ndarray:
    out = np.zeros((len(hours), 24), dtype=float)
    out[np.arange(len(hours)), hours.astype(int) % 24] = 1.0
    return out


def _window_max(values: np.ndarray, width: int = STRIDE) -> np.ndarray:
    """Causal rolling max over the last ``width`` rows (NaN treated as -inf)."""

    v = np.where(np.isfinite(values), values, -np.inf)
    out = np.full(v.shape, -np.inf)
    for lag in range(width):
        shifted = np.full(v.shape, -np.inf)
        shifted[lag:] = v[: len(v) - lag] if lag else v
        out = np.maximum(out, shifted)
    return out


def _consecutive_run_ending(mask: np.ndarray) -> np.ndarray:
    """For each row, number of consecutive True rows ending at that row."""

    out = np.zeros(mask.shape, dtype=int)
    run = np.zeros(mask.shape[1:], dtype=int)
    for t in range(mask.shape[0]):
        run = np.where(mask[t], run + 1, 0)
        out[t] = run
    return out


class _Ridge:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_Ridge":
        self.mu = x.mean(axis=0)
        self.sd = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
        z = (x - self.mu) / self.sd
        a = z.T @ z + self.alpha * np.eye(z.shape[1])
        self.ymu = y.mean(axis=0)
        self.coef = np.linalg.solve(a, z.T @ (y - self.ymu))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mu) / self.sd) @ self.coef + self.ymu


@dataclass
class PhysicsReference:
    columns: tuple[str, ...]
    tanks: tuple[str, ...]
    pumps: tuple[str, ...]
    junctions: tuple[str, ...]
    flow_cols: tuple[str, ...]
    status_cols: tuple[str, ...]
    level_lo: np.ndarray
    level_hi: np.ndarray
    off_flow_max: np.ndarray
    on_flow_min: np.ndarray
    flow_tol: np.ndarray
    tank_model: _Ridge
    tank_sd: np.ndarray
    pressure_model: _Ridge
    pressure_sd: np.ndarray
    status_freq_by_hour: np.ndarray  # [24, pumps] P(status on | hour)
    residual_sigma: np.ndarray  # per channel, standardized-residual std on held-out normal rows
    residual_q995: np.ndarray  # per channel |residual/sigma| 99.5% quantile, held-out normal rows
    check_thresholds: dict[str, float]
    check_window_rates: dict[str, float]
    schedule_threshold: float
    schedule_window_rate: float
    detector_window_rates: dict[str, float]


def _resolution(values: np.ndarray) -> np.ndarray:
    """Per-column measurement resolution: smallest positive step between distinct readings."""

    out = []
    for col in np.asarray(values, dtype=float).T:
        steps = np.diff(np.unique(np.round(col, 6)))
        steps = steps[steps > 1e-9]
        out.append(float(steps.min()) if steps.size else 1.0)
    return np.maximum(np.asarray(out), EVALUATED_RESOLUTION)


def _idx(columns: Sequence[str], names: Sequence[str]) -> list[int]:
    return [list(columns).index(n) for n in names]


def _tank_features(raw: np.ndarray, ref: PhysicsReference | None, cols: Sequence[str], hours: np.ndarray) -> np.ndarray:
    flow = raw[:, _idx(cols, [c for c in cols if c.startswith("F_")])]
    status = raw[:, _idx(cols, [c for c in cols if c.startswith("S_")])]
    prev_flow = np.vstack([flow[:1], flow[:-1]])
    return np.hstack([flow, prev_flow, status, _hour_onehot(hours)])


def _pressure_features(raw: np.ndarray, cols: Sequence[str], hours: np.ndarray) -> np.ndarray:
    keep = [c for c in cols if c.startswith(("L_T", "F_", "S_"))]
    return np.hstack([raw[:, _idx(cols, keep)], _hour_onehot(hours)])


def _raw_physics(ref: PhysicsReference, raw: np.ndarray, hours: np.ndarray) -> dict[str, np.ndarray]:
    """Per-hour physical-check statistics (larger = less consistent)."""

    cols = ref.columns
    levels = raw[:, _idx(cols, ref.tanks)]
    flows = raw[:, _idx(cols, [f"F_{p}" for p in ref.pumps])]
    status = raw[:, _idx(cols, [f"S_{p}" for p in ref.pumps])]
    press = raw[:, _idx(cols, ref.junctions)]
    # pump status vs flow: violation if flow leaves the training range for that status
    off_bad = (status < 0.5) & (flows > ref.off_flow_max + ref.flow_tol)
    on_bad = (status >= 0.5) & (flows < ref.on_flow_min - ref.flow_tol)
    pump_bad = (off_bad | on_bad).astype(float)
    # tank level change vs flows
    dlevel = np.vstack([np.full((1, levels.shape[1]), np.nan), np.diff(levels, axis=0)])
    pred = ref.tank_model.predict(_tank_features(raw, ref, cols, hours))
    tank_z = np.abs(dlevel - pred) / ref.tank_sd
    tank_z[0] = np.nan
    # junction pressure vs operation
    press_z = np.abs(press - ref.pressure_model.predict(_pressure_features(raw, cols, hours))) / ref.pressure_sd
    # envelope
    span = ref.level_hi - ref.level_lo
    outside = np.maximum(ref.level_lo - levels, levels - ref.level_hi) / np.where(span > 0, span, 1.0)
    # schedule atypicality
    freq_on = ref.status_freq_by_hour[hours.astype(int) % 24]
    freq_obs = np.where(status >= 0.5, freq_on, 1.0 - freq_on)
    atypical = (freq_obs < 0.02).astype(float)
    return {
        "pump_bad": pump_bad,
        "tank_z": tank_z,
        "press_z": press_z,
        "outside": outside,
        "atypical": atypical,
        "flows": flows,
        "status": status,
        "levels": levels,
        "freq_obs": freq_obs,
    }


def _check_stats(phys: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Scalar per-hour statistic per check family (max over channels)."""

    with np.errstate(invalid="ignore"):
        return {
            "pump_status_vs_flow": np.nanmax(phys["pump_bad"], axis=1),
            "tank_level_change_vs_flows": np.nanmax(np.nan_to_num(phys["tank_z"], nan=0.0), axis=1),
            "junction_pressure_vs_operation": np.nanmax(phys["press_z"], axis=1),
            "tank_level_envelope": np.nanmax(phys["outside"], axis=1),
            "schedule": np.sum(phys["atypical"], axis=1),
        }


def fit_physics_reference(front: FrontEnd, config: HydroJEVConfig, detector_stream: Mapping[str, Any]) -> PhysicsReference:
    clean = load_batadal("dataset03", config=config)
    cols = tuple(front.columns)
    raw_all = clean.frame[list(cols)].to_numpy(dtype=float)
    hours_all = clean.frame["DATETIME"].dt.hour.to_numpy()
    tr = slice(0, CALIBRATION_START)
    raw_tr, hours_tr = raw_all[tr], hours_all[tr]
    tanks = tuple(c for c in cols if c.startswith("L_T"))
    pumps = tuple(c[2:] for c in cols if c.startswith("F_PU"))
    junctions = tuple(c for c in cols if c.startswith("P_J"))
    flows_tr = raw_tr[:, _idx(cols, [f"F_{p}" for p in pumps])]
    status_tr = raw_tr[:, _idx(cols, [f"S_{p}" for p in pumps])]
    off_max = np.array([np.max(flows_tr[status_tr[:, k] < 0.5, k]) if np.any(status_tr[:, k] < 0.5) else 0.0 for k in range(len(pumps))])
    on_min = np.array([np.min(flows_tr[status_tr[:, k] >= 0.5, k]) if np.any(status_tr[:, k] >= 0.5) else 0.0 for k in range(len(pumps))])
    tol = 0.02 * np.maximum(flows_tr.max(axis=0), 1.0)
    levels_tr = raw_tr[:, _idx(cols, tanks)]
    span = levels_tr.max(axis=0) - levels_tr.min(axis=0)
    lo, hi = levels_tr.min(axis=0) - 0.02 * span, levels_tr.max(axis=0) + 0.02 * span
    dlevel = np.diff(levels_tr, axis=0)
    x_tank = _tank_features(raw_tr, None, cols, hours_tr)[1:]
    tank_model = _Ridge(1.0).fit(x_tank, dlevel)
    # residual scales never go below the sensor resolution, so a one-step quantized
    # change on a near-constant channel cannot masquerade as a large violation
    tank_sd = np.hypot((dlevel - tank_model.predict(x_tank)).std(axis=0), _resolution(levels_tr))
    press_tr = raw_tr[:, _idx(cols, junctions)]
    x_press = _pressure_features(raw_tr, cols, hours_tr)
    pressure_model = _Ridge(1.0).fit(x_press, press_tr)
    pressure_sd = np.hypot((press_tr - pressure_model.predict(x_press)).std(axis=0), _resolution(press_tr))
    freq = np.zeros((24, len(pumps)))
    for h in range(24):
        sel = hours_tr == h
        freq[h] = (status_tr[sel] >= 0.5).sum(axis=0) / max(1, sel.sum())
    # residual scales from the frozen surrogate on held-out normal rows
    trained = front.trained
    std_all = trained.standardizer.transform(raw_all)
    res, r0 = one_step_residual_stream(trained.predict_fn, std_all, trained.seq_len)
    held = res[CALIBRATION_START - r0 :]
    sigma = np.hypot(held.std(axis=0), _resolution(raw_all[:CALIBRATION_START]) / trained.standardizer.scale)
    q995 = np.quantile(np.abs(held / sigma), 0.995, axis=0)
    ref = PhysicsReference(
        columns=cols, tanks=tanks, pumps=pumps, junctions=junctions,
        flow_cols=tuple(c for c in cols if c.startswith("F_")), status_cols=tuple(c for c in cols if c.startswith("S_")),
        level_lo=lo, level_hi=hi, off_flow_max=off_max, on_flow_min=on_min, flow_tol=tol,
        tank_model=tank_model, tank_sd=tank_sd, pressure_model=pressure_model, pressure_sd=pressure_sd,
        status_freq_by_hour=freq, residual_sigma=sigma, residual_q995=q995,
        check_thresholds={}, check_window_rates={}, schedule_threshold=0.0, schedule_window_rate=0.0,
        detector_window_rates={},
    )
    # calibrate every check family on held-out normal rows (window level)
    phys = _raw_physics(ref, raw_all, hours_all)
    stats = _check_stats(phys)
    cal = slice(CALIBRATION_START + STRIDE, None)
    for name in PHYSICAL_CHECKS:
        w = _window_max(stats[name])[cal]
        if name == "pump_status_vs_flow":
            thr = 0.5  # any mismatch
        elif name == "tank_level_envelope":
            thr = 0.0  # any excursion beyond the training envelope (+2% margin)
        else:
            thr = float(np.quantile(w[np.isfinite(w)], CHECK_WINDOW_QUANTILE))
        ref.check_thresholds[name] = thr
        ref.check_window_rates[name] = float(np.mean(w > thr))
    ws = _window_max(stats["schedule"])[cal]
    ref.schedule_threshold = float(np.quantile(ws, CHECK_WINDOW_QUANTILE))
    ref.schedule_window_rate = float(np.mean(ws > ref.schedule_threshold))
    # detector window alert rates on held-out normal rows
    s0 = int(detector_stream["start"])
    for name, thr in front.thresholds.items():
        v = np.asarray(detector_stream["scores"][name], dtype=float)
        alert = np.where(np.isfinite(v), (v >= thr).astype(float), 0.0)
        w = _window_max(alert)[max(0, CALIBRATION_START + STRIDE - s0):]
        ref.detector_window_rates[name] = float(np.mean(w > 0.5))
    return ref


# --------------------------------------------------------------------------- #
# Evidence state v2 per grid point
# --------------------------------------------------------------------------- #


@dataclass
class StreamEvidence:
    variant: str
    stream: Mapping[str, Any]
    raw: np.ndarray
    hours: np.ndarray
    weekdays: np.ndarray
    residual_sigma: np.ndarray  # [n_raw, C], NaN before surrogate warmup
    predicted_raw: np.ndarray  # [n_raw, C]
    phys: dict[str, np.ndarray]
    stats: dict[str, np.ndarray]


def build_stream_evidence(front: FrontEnd, ref: PhysicsReference, stream: Mapping[str, Any]) -> StreamEvidence:
    bundle = stream["bundle"]
    cols = list(front.columns)
    raw = bundle.frame[cols].to_numpy(dtype=float)
    stamp = bundle.frame["DATETIME"]
    hours, weekdays = stamp.dt.hour.to_numpy(), stamp.dt.weekday.to_numpy()
    trained = front.trained
    std = trained.standardizer.transform(raw)
    res, r0 = one_step_residual_stream(trained.predict_fn, std, trained.seq_len)
    rs = np.full(raw.shape, np.nan)
    rs[r0:] = res / ref.residual_sigma
    pred = np.full(raw.shape, np.nan)
    pred[r0:] = trained.standardizer.inverse_transform(std[r0:] - res)
    phys = _raw_physics(ref, raw, hours)
    return StreamEvidence(stream["variant"], stream, raw, hours, weekdays, rs, pred, phys, _check_stats(phys))


def _r(x: float, nd: int = 3) -> float:
    return float(round(float(x), nd))


def detector_window(front: FrontEnd, stream: Mapping[str, Any], raw_i: int) -> dict[str, dict[str, Any]]:
    j = raw_i - int(stream["start"])
    left, right = max(0, j - STRIDE + 1), j + 1
    out: dict[str, dict[str, Any]] = {}
    for name in DETECTOR_ORDER:
        if name not in front.thresholds or name not in stream["scores"]:
            continue
        v = np.asarray(stream["scores"][name][left:right], dtype=float)
        v = v[np.isfinite(v)]
        thr = front.thresholds[name]
        if v.size == 0:
            out[name] = {"available": False}
            continue
        out[name] = {
            "available": True,
            "alerting": bool(np.any(v >= thr)),
            "peak_ratio": float(np.max(v) / thr),
            "alert_hours": int(np.sum(v >= thr)),
        }
    return out


def build_state_v2(front: FrontEnd, ref: PhysicsReference, ev: StreamEvidence, raw_i: int) -> dict[str, Any]:
    lo, hi = raw_i - STRIDE + 1, raw_i + 1
    cols = list(front.columns)
    det = detector_window(front, ev.stream, raw_i)
    detectors = {
        name: (
            {"family": DETECTOR_FAMILY[name], "state": "unavailable"}
            if not d["available"]
            else {
                "family": DETECTOR_FAMILY[name],
                "state": "alerting" if d["alerting"] else "nominal",
                "peak_score_over_threshold": _r(d["peak_ratio"]),
                "hours_above_threshold_in_window": d["alert_hours"],
                "normal_operation_window_alert_rate": _r(ref.detector_window_rates.get(name, float("nan"))),
            }
        )
        for name, d in det.items()
    }
    alerting = [n for n, d in det.items() if d.get("alerting")]
    families = sorted({DETECTOR_FAMILY[n] for n in alerting})
    # residuals
    rs_win = ev.residual_sigma[lo:hi]
    peak = np.nanmax(np.abs(rs_win), axis=0)
    order = [int(k) for k in np.argsort(-np.nan_to_num(peak, nan=-1.0))[:TOP_RESIDUALS]]
    runs = _consecutive_run_ending(np.abs(np.nan_to_num(ev.residual_sigma[max(0, hi - 48):hi])) > RESIDUAL_SIGMA_RUN)[-1]
    residuals = [
        {
            "channel": cols[k],
            "observed_now": _r(ev.raw[raw_i, k]),
            "predicted_now": _r(ev.predicted_raw[raw_i, k]),
            "residual_sigma_now": _r(ev.residual_sigma[raw_i, k], 2),
            "peak_abs_residual_sigma_in_window": _r(peak[k], 2),
            "normal_operation_abs_residual_sigma_q99_5": _r(ref.residual_q995[k], 2),
            "consecutive_hours_above_3_sigma": int(runs[k]),
        }
        for k in order
    ]
    # physical checks
    p = ev.phys
    checks: dict[str, Any] = {}
    stat_w = {name: float(np.nanmax(ev.stats[name][lo:hi])) if np.any(np.isfinite(ev.stats[name][lo:hi])) else float("nan") for name in PHYSICAL_CHECKS}
    viol_pumps = sorted({ref.pumps[k] for k in np.flatnonzero(p["pump_bad"][lo:hi].max(axis=0) > 0.5)})
    checks["pump_status_vs_flow"] = {
        "status": "violated" if stat_w["pump_status_vs_flow"] > ref.check_thresholds["pump_status_vs_flow"] else "consistent",
        "meaning": "pump reports off but carries flow, or reports on with no flow, compared with the attack-free range",
        "violating_pumps": [
            {"pump": pu, "status_now": int(ev.raw[raw_i, cols.index(f"S_{pu}")] >= 0.5), "flow_now": _r(ev.raw[raw_i, cols.index(f"F_{pu}")])}
            for pu in viol_pumps
        ],
        "normal_operation_window_violation_rate": _r(ref.check_window_rates["pump_status_vs_flow"]),
    }
    tz = np.nan_to_num(p["tank_z"][lo:hi], nan=0.0).max(axis=0)
    checks["tank_level_change_vs_flows"] = {
        "status": "violated" if stat_w["tank_level_change_vs_flows"] > ref.check_thresholds["tank_level_change_vs_flows"] else "consistent",
        "meaning": "hourly level change of each tank compared with the change expected from the reported pump and valve flows",
        "peak_abs_z_by_tank": {ref.tanks[k]: _r(tz[k], 2) for k in range(len(ref.tanks))},
        "violation_threshold_abs_z": _r(ref.check_thresholds["tank_level_change_vs_flows"], 2),
        "normal_operation_window_violation_rate": _r(ref.check_window_rates["tank_level_change_vs_flows"]),
    }
    pz = p["press_z"][lo:hi].max(axis=0)
    worst = [int(k) for k in np.argsort(-pz)[:3]]
    checks["junction_pressure_vs_operation"] = {
        "status": "violated" if stat_w["junction_pressure_vs_operation"] > ref.check_thresholds["junction_pressure_vs_operation"] else "consistent",
        "meaning": "each junction pressure compared with the pressure expected from reported tank levels, pump states and flows",
        "largest_abs_z": {ref.junctions[k]: _r(pz[k], 2) for k in worst},
        "violation_threshold_abs_z": _r(ref.check_thresholds["junction_pressure_vs_operation"], 2),
        "normal_operation_window_violation_rate": _r(ref.check_window_rates["junction_pressure_vs_operation"]),
    }
    out_now = [
        {"tank": ref.tanks[k], "level_now": _r(p["levels"][raw_i, k]), "normal_min": _r(ref.level_lo[k]), "normal_max": _r(ref.level_hi[k])}
        for k in np.flatnonzero(p["outside"][lo:hi].max(axis=0) > 0.0)
    ]
    checks["tank_level_envelope"] = {
        "status": "violated" if stat_w["tank_level_envelope"] > ref.check_thresholds["tank_level_envelope"] else "consistent",
        "meaning": "tank level outside the range seen in attack-free operation",
        "tanks_outside": out_now,
        "normal_operation_window_violation_rate": _r(ref.check_window_rates["tank_level_envelope"]),
    }
    atyp_now = [
        {"pump": ref.pumps[k], "status_now": int(p["status"][raw_i, k] >= 0.5), "normal_frequency_of_this_status_at_this_hour": _r(p["freq_obs"][raw_i, k])}
        for k in np.flatnonzero(p["atypical"][raw_i] > 0.5)
    ]
    sched_peak = float(np.max(ev.stats["schedule"][lo:hi]))
    state = {
        "schema_version": SCHEMA_VERSION,
        "system": SYSTEM_DESCRIPTION,
        "window": {
            "length_hours": STRIDE,
            "hour_of_day_now": int(ev.hours[raw_i]),
            "day_of_week": _WEEKDAYS[int(ev.weekdays[raw_i])],
            "evidence_fresh": True,
        },
        "detector_pattern": {
            "alerting": alerting,
            "nominal": [n for n, d in det.items() if d.get("available") and not d.get("alerting")],
            "alerting_count": len(alerting),
            "alerting_families": families,
            "note": "normal_operation_window_alert_rate is the fraction of attack-free six-hour windows in which that detector alerts",
        },
        "detectors": detectors,
        "largest_surrogate_residuals": residuals,
        "physical_consistency": {
            "violated_checks": [n for n in PHYSICAL_CHECKS if checks[n]["status"] == "violated"],
            **checks,
        },
        "operating_context": {
            "pumps_in_atypical_state_for_this_hour_now": atyp_now,
            "peak_count_atypical_pumps_in_window": int(sched_peak),
            "normal_operation_count_threshold_q99": _r(ref.schedule_threshold, 1),
        },
        "provenance": {
            "observation_source": "hourly SCADA historian",
            "labels_removed": True,
            "episode_identifiers_removed": True,
            "dates_removed": True,
        },
    }
    _assert_no_ground_truth(state)
    return _json_safe(state, path="state")


# --------------------------------------------------------------------------- #
# Comparators and features
# --------------------------------------------------------------------------- #


def grid_points(stream: Mapping[str, Any]) -> np.ndarray:
    start = int(stream["start"])
    n = len(stream["labels"])
    return np.arange(max(WARMUP, start), n + start, STRIDE, dtype=int)


def grid_features(front: FrontEnd, ref: PhysicsReference, ev: StreamEvidence, raw_i: int) -> tuple[dict[str, Any], np.ndarray]:
    """Deterministic summaries shared by the gate, R1 and R2 (same evidence as the state)."""

    lo, hi = raw_i - STRIDE + 1, raw_i + 1
    det = detector_window(front, ev.stream, raw_i)
    alert_count = sum(int(d.get("alerting", False)) for d in det.values())
    violated = []
    ratios = []
    for name in PHYSICAL_CHECKS:
        w = ev.stats[name][lo:hi]
        s = float(np.nanmax(w)) if np.any(np.isfinite(w)) else 0.0
        thr = ref.check_thresholds[name]
        if s > thr:
            violated.append(name)
        ratios.append(s / thr if thr > 0 else float(s > thr))
    peak_res = float(np.nanmax(np.abs(ev.residual_sigma[lo:hi]))) if np.any(np.isfinite(ev.residual_sigma[lo:hi])) else 0.0
    sched = float(np.max(ev.stats["schedule"][lo:hi]))
    vec = []
    for name in DETECTOR_ORDER:
        d = det.get(name, {})
        vec += [float(min(d.get("peak_ratio", 0.0), 10.0)), float(d.get("alerting", False)), float(d.get("alert_hours", 0)) / STRIDE]
    vec += [float(min(r, 10.0)) for r in ratios]
    vec += [min(peak_res, 50.0), sched, float(alert_count)]
    summary = {
        "alert_count": alert_count,
        "alerting": [n for n, d in det.items() if d.get("alerting")],
        "violated_checks": violated,
        "candidate": alert_count >= 1,
        "r1": bool(violated) or alert_count >= MIN_CANDIDATE_ALERTS,
        "v2_rule": alert_count >= MIN_CANDIDATE_ALERTS,
    }
    for name in ("Mahalanobis", "PCA residual", "HydroJEV AE", "Isolation Forest"):
        summary[name] = bool(det.get(name, {}).get("alerting", False))
    return summary, np.asarray(vec, dtype=float)


def hold_to_hourly(grid: np.ndarray, grid_alarm: np.ndarray, *, start: int, n: int) -> np.ndarray:
    held = np.zeros(n, dtype=bool)
    for k, raw_i in enumerate(grid):
        nxt = int(grid[k + 1]) if k + 1 < len(grid) else n + start
        a, b = max(0, int(raw_i - start)), min(n, int(nxt - start))
        held[a:b] = bool(grid_alarm[k])
    return held


def episode_metrics(labels: np.ndarray, grid: np.ndarray, grid_alarm: np.ndarray, *, start: int, interval_s: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    held = hold_to_hourly(grid, np.asarray(grid_alarm, dtype=bool), start=start, n=len(labels))
    m = _runs_metrics(labels, held, interval_s=interval_s, start=start)
    rec, far = m["scenario_recall"], m["benign_scenario_fp_rate"]
    m["episode_balanced_accuracy"] = (rec + 1.0 - far) / 2.0 if rec is not None and far is not None else None
    m["hourly_balanced_accuracy_S_CLF"] = m["batadal_score"].get("S_CLF") if isinstance(m.get("batadal_score"), Mapping) else None
    return m


def window_metrics(grid_labels: np.ndarray, grid_alarm: np.ndarray, *, reps: int = 2000, seed: int = 0) -> dict[str, Any]:
    y = np.asarray(grid_labels, dtype=int)
    a = np.asarray(grid_alarm, dtype=bool)

    def stats(yy: np.ndarray, aa: np.ndarray) -> tuple[float, float, float]:
        tp = np.sum(aa & (yy == 1)); fp = np.sum(aa & (yy == 0))
        pos = np.sum(yy == 1); neg = np.sum(yy == 0)
        rec = tp / pos if pos else float("nan")
        fpr = fp / neg if neg else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        return float(prec), float(rec), float(fpr)

    prec, rec, fpr = stats(y, a)
    rng = np.random.default_rng(seed)
    draws = np.array([stats(y[i], a[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))])

    def ci(col: int) -> list[float]:
        v = draws[:, col]; v = v[np.isfinite(v)]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v.size else [float("nan")] * 2

    return {"precision": prec, "recall": rec, "fpr": fpr, "precision_ci": ci(0), "recall_ci": ci(1), "fpr_ci": ci(2), "n_windows": int(len(y)), "n_attack_windows": int(np.sum(y == 1))}


def fit_r2(x: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return model.fit(x, y)


# --------------------------------------------------------------------------- #
# Prepare: fit, build states/requests (no network), leakage scan
# --------------------------------------------------------------------------- #

_LEAK_TOKENS = ("att_flag", "ground_truth", "true_cause", "scenario_id", "attack_id", "expected_cause", "labels_internal", "batadal", "2014-", "2016-", "2017-")


def scan_request(body: bytes, *, key_fingerprint: str | None = None) -> list[str]:
    text = body.decode("utf-8").lower()
    problems = [t for t in _LEAK_TOKENS if t in text]
    if key_fingerprint and key_fingerprint.lower() in text:
        problems.append("api_key")
    if "bearer" in text or "tsk_" in text or "sk-" in text:
        problems.append("credential_like_token")
    return problems


def prepare(config: HydroJEVConfig, outdir: Path, *, seed: int = 42, gru_epochs: int = 12, ae_epochs: int = 12) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "prepared.json").exists():
        raise FileExistsError(f"{outdir / 'prepared.json'} exists; the prepared evidence is frozen")
    front = fit_front_end(config, seed=seed, gru_epochs=gru_epochs, ae_epochs=ae_epochs)
    if isinstance(front, dict):
        return front
    cal_stream = score_stream(front, config, "dataset03")
    ref = fit_physics_reference(front, config, cal_stream)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "fit": {"seed": seed, "gru_epochs": gru_epochs, "ae_epochs": ae_epochs, "thresholds": front.thresholds},
        "calibration": {
            "check_thresholds": ref.check_thresholds,
            "check_window_rates_heldout_normal": ref.check_window_rates,
            "schedule_threshold": ref.schedule_threshold,
            "schedule_window_rate": ref.schedule_window_rate,
            "detector_window_rates_heldout_normal": ref.detector_window_rates,
        },
        "questions_sha256": hashlib.sha256(canonical_bytes(build_questions_v2())).hexdigest(),
        "variants": {},
    }
    feats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for variant in ("dataset04", "test_dataset"):
        stream = score_stream(front, config, variant)
        ev = build_stream_evidence(front, ref, stream)
        grid = grid_points(stream)
        start = int(stream["start"])
        labels = np.asarray(stream["labels"], dtype=int)
        reqdir = outdir / "requests" / variant
        reqdir.mkdir(parents=True, exist_ok=True)
        rows = []
        xs = []
        for raw_i in grid:
            summary, vec = grid_features(front, ref, ev, int(raw_i))
            row = {"raw_index": int(raw_i), "label_internal": int(labels[raw_i - start]), **summary}
            if summary["candidate"]:
                body = canonical_bytes(build_request_v2(build_state_v2(front, ref, ev, int(raw_i))))
                problems = scan_request(body)
                if problems:
                    raise HydroJEVSchemaError(f"leak scan failed at grid {raw_i}: {problems}")
                path = reqdir / f"grid_{int(raw_i):05d}.request.json"
                path.write_bytes(body)
                row.update({"request_path": str(path), "request_sha256": hashlib.sha256(body).hexdigest(), "request_bytes": len(body)})
            rows.append(row)
            xs.append(vec)
        feats[variant] = (np.asarray(xs), np.asarray([r["label_internal"] for r in rows]), np.asarray([r["candidate"] for r in rows]))
        result["variants"][variant] = {
            "start": start,
            "interval_s": float(stream["interval_s"]),
            "labels_hourly_internal": labels.tolist(),
            "grid": rows,
            "n_candidates": int(sum(r["candidate"] for r in rows)),
            "n_grid": len(rows),
        }
    # R2: trained on dev candidates with dev labels, applied to both splits' candidates
    x_dev, y_dev, c_dev = feats["dataset04"]
    r2 = fit_r2(x_dev[c_dev], y_dev[c_dev])
    for variant, (x, _, c) in feats.items():
        prob = np.zeros(len(x))
        prob[c] = r2.predict_proba(x[c])[:, 1]
        for row, pr in zip(result["variants"][variant]["grid"], prob):
            row["r2_probability"] = float(pr)
            row["r2"] = bool(row["candidate"] and pr >= 0.5)
    result["r2_model"] = {"type": "logistic_regression_balanced", "trained_on": "dataset04 candidate windows with labels", "n_train": int(c_dev.sum())}
    (outdir / "prepared.json").write_text(json.dumps(_safe(result), indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def benign_candidate_rates(prepared: Mapping[str, Any]) -> dict[str, float]:
    """Per-check violation rate among flagged benign development windows (dev labels, like R2)."""

    benign = [r for r in prepared["variants"]["dataset04"]["grid"] if r["candidate"] and r["label_internal"] == 0]
    return {name: float(np.mean([name in r["violated_checks"] for r in benign])) for name in PHYSICAL_CHECKS} | {"n_windows": len(benign)}


def add_benign_rates(state: Mapping[str, Any], rates: Mapping[str, float]) -> dict[str, Any]:
    out = json.loads(json.dumps(state))
    pc = out["physical_consistency"]
    for name in PHYSICAL_CHECKS:
        pc[name]["benign_flagged_window_violation_rate"] = _r(rates[name])
    pc["note"] = BENIGN_RATE_NOTE
    _assert_no_ground_truth(out)
    return out


def prepare_r2(outdir: Path) -> dict[str, Any]:
    """Round-2 requests: the frozen round-1 evidence plus benign base rates, alarm question revision 2."""

    target = outdir / "prepared_r2.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; the round-2 evidence is frozen")
    prepared = json.loads((outdir / "prepared.json").read_text(encoding="utf-8"))
    rates = benign_candidate_rates(prepared)
    result: dict[str, Any] = {
        "protocol": PROTOCOL,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "derived_from": {"prepared_json_sha256": hashlib.sha256((outdir / "prepared.json").read_bytes()).hexdigest()},
        "question_revision": 2,
        "questions_sha256": hashlib.sha256(canonical_bytes(build_questions_v2(2))).hexdigest(),
        "benign_candidate_rates_dev": rates,
        "decision_rule_primary": "final alarm = candidate and alarm noul >= 0.5 (unchanged, pre-registered)",
        "decision_rule_secondary": "candidate and noul >= 0.5 and event_type in {cyber_attack, physical_fault}",
        "caveat": "benign rates use development labels, so development-set results of round 2 are in-sample; the test set stays clean",
        "variants": {},
    }
    for variant, block in prepared["variants"].items():
        reqdir = outdir / "requests_r2" / variant
        reqdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for row in block["grid"]:
            row = dict(row)
            if row["candidate"]:
                old = json.loads(Path(row["request_path"]).read_text(encoding="utf-8"))
                body = canonical_bytes(build_request_v2(add_benign_rates(old["state"], rates), revision=2))
                problems = scan_request(body)
                if problems:
                    raise HydroJEVSchemaError(f"leak scan failed at grid {row['raw_index']}: {problems}")
                path = reqdir / f"grid_{int(row['raw_index']):05d}.request.json"
                path.write_bytes(body)
                row.update({"request_path_r1": row["request_path"], "request_path": str(path), "request_sha256": hashlib.sha256(body).hexdigest(), "request_bytes": len(body)})
            rows.append(row)
        result["variants"][variant] = {**{k: v for k, v in block.items() if k != "grid"}, "grid": rows}
    target.write_text(json.dumps(_safe(result), indent=1, ensure_ascii=False), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# Live calls with ledger and hard budget
# --------------------------------------------------------------------------- #


def _ledger_path(outdir: Path) -> Path:
    return outdir / "ledger.jsonl"


def ledger_entries(outdir: Path) -> list[dict[str, Any]]:
    p = _ledger_path(outdir)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_ledger(outdir: Path, entry: Mapping[str, Any]) -> None:
    with _ledger_path(outdir).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_safe(dict(entry)), ensure_ascii=False) + "\n")


def _key_env() -> dict[str, str]:
    env = dict(os.environ)
    if not env.get("TYPESAFE_API_KEY"):
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as k:
            env["TYPESAFE_API_KEY"] = str(winreg.QueryValueEx(k, "TYPESAFE_API_KEY")[0])
    return env


def _summarize_v2(response: Mapping[str, Any]) -> dict[str, Any]:
    d = parse_response_v2(response)
    return {"model": d.model, "usage": d.usage, "alarm_probability": d.alarm_probability, "event_type": d.event_type, "severity": d.severity_score}


def call_live(outdir: Path, request_path: Path, *, stage: str, tag: str, summarize: Any = None) -> dict[str, Any]:
    """One paid call. Never re-sends a request that already has a ledger entry for ``tag``.

    ``summarize`` parses the raw response into ledger fields (default: the P1 v2 contract); P2 passes
    its own parser so that both pilot stages share this ledger and its hard cap.
    """

    entries = ledger_entries(outdir)
    if any(e.get("tag") == tag for e in entries):
        prior = [e for e in entries if e.get("tag") == tag][-1]
        return {**prior, "reused": True}
    if len(entries) >= PILOT_CALL_CAP:
        raise RuntimeError(f"pilot hard cap of {PILOT_CALL_CAP} calls reached")
    respdir = outdir / "responses" / stage
    respdir.mkdir(parents=True, exist_ok=True)
    resp = respdir / f"{tag}.response.json"
    if resp.exists():
        raise FileExistsError(f"{resp} exists without a ledger entry; refusing to overwrite")
    body = request_path.read_bytes()
    script = Path(__file__).resolve().parents[2] / "tools" / "jev" / "invoke-jev.ps1"
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-RequestPath", str(request_path.resolve()), "-OutputPath", str(resp.resolve())]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False, env=_key_env(), timeout=240)
    except subprocess.TimeoutExpired:
        # the request may have been billed: record it so it is never re-sent blindly
        entry = {"tag": tag, "stage": stage, "request_path": str(request_path), "request_sha256": hashlib.sha256(body).hexdigest(), "response_path": str(resp), "status": "timeout", "wall_seconds": time.perf_counter() - t0, "utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        _append_ledger(outdir, entry)
        return entry
    wall = time.perf_counter() - t0
    entry: dict[str, Any] = {
        "tag": tag, "stage": stage, "request_path": str(request_path), "request_sha256": hashlib.sha256(body).hexdigest(),
        "response_path": str(resp), "returncode": int(p.returncode), "wall_seconds": wall,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if p.returncode != 0 or not resp.exists():
        entry.update({"status": "failed", "stderr_tail": p.stderr[-300:]})
    else:
        try:
            wrapper = json.loads(resp.read_text(encoding="utf-8-sig"))
            fields = (summarize or _summarize_v2)(wrapper["response"])
            entry.update({"status": "ok", **fields, "elapsed_seconds": float(wrapper.get("elapsed_seconds", wall)), "attempts": int(wrapper.get("attempts", 1))})
        except Exception as exc:  # recorded, never retried automatically
            entry.update({"status": "schema_failed", "error": f"{type(exc).__name__}: {exc}"[:300]})
    _append_ledger(outdir, entry)
    return entry


def mock_response(request: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic stand-in used only for the dry run of the pipeline."""

    st = request["state"]
    violated = bool(st["physical_consistency"]["violated_checks"])
    n = int(st["detector_pattern"]["alerting_count"])
    p = 0.85 if violated or n >= 3 else 0.2
    probs = {k: 0.05 for k in EVENT_TYPES_V2}
    top = "cyber_attack" if p > 0.5 else "normal_transient"
    probs[top] = 0.8
    return {
        "model": "mock-v2",
        "answers": {
            "alarm": {"type": "noul", "noul": p},
            "event_type": {"type": "choice", "choice": top, "confidence": 0.8, "probabilities": probs},
            "severity": {"type": "score", "score": 2.0 if p > 0.5 else 0.5, "confidence": 0.7, "probabilities": {str(i): 0.2 for i in range(5)}, "legend": {str(i): s for i, s in enumerate(SEVERITY_LEGEND_V2)}},
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def select_debug(prepared: Mapping[str, Any], n: int, *, seed: int = DEFAULT_SAMPLE_SEED) -> list[int]:
    """Stratified dev sample (labels used only to stratify, never in the request)."""

    rows = [r for r in prepared["variants"]["dataset04"]["grid"] if r["candidate"]]
    rng = np.random.default_rng(seed)
    pos = [r["raw_index"] for r in rows if r["label_internal"] == 1]
    neg = [r["raw_index"] for r in rows if r["label_internal"] == 0]
    k_pos = min(len(pos), n // 2)
    chosen = list(rng.choice(pos, k_pos, replace=False)) + list(rng.choice(neg, min(len(neg), n - k_pos), replace=False))
    return sorted(int(c) for c in chosen)


def summarize_debug(prepared: Mapping[str, Any], decisions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {r["raw_index"]: r for r in prepared["variants"]["dataset04"]["grid"]}
    ok = {k: v for k, v in decisions.items() if v.get("status") == "ok"}
    if not ok:
        return {"n_ok": 0}
    jev = {k: v["alarm_probability"] >= 0.5 for k, v in ok.items()}
    coherent = np.array([v["alarm_probability"] >= 0.5 and v["event_type"] in ("cyber_attack", "physical_fault") for v in ok.values()])
    agree = np.mean([jev[k] == rows[k]["r1"] for k in ok])
    insuff = np.mean([v["event_type"] == "insufficient_evidence" for v in ok.values()])
    y = np.array([rows[k]["label_internal"] for k in ok])
    j = np.array([jev[k] for k in ok])
    r1 = np.array([rows[k]["r1"] for k in ok])
    maha = np.array([rows[k]["Mahalanobis"] for k in ok])
    r2 = np.array([rows[k]["r2"] for k in ok])

    def acc(a: np.ndarray) -> dict[str, float]:
        tpr = float(np.mean(a[y == 1])) if np.any(y == 1) else float("nan")
        tnr = float(np.mean(~a[y == 0])) if np.any(y == 0) else float("nan")
        return {"tpr": tpr, "tnr": tnr, "balanced": (tpr + tnr) / 2}

    event_counts: dict[str, int] = {}
    for v in ok.values():
        event_counts[v["event_type"]] = event_counts.get(v["event_type"], 0) + 1
    return {
        "n_ok": len(ok),
        "n_failed": len(decisions) - len(ok),
        "insufficient_evidence_rate": float(insuff),
        "agreement_with_r1": float(agree),
        "early_stop": bool(insuff > 0.5 or agree > 0.95),
        "agreement_with_r1_secondary_rule": float(np.mean(coherent == r1)),
        "alarm_contradicts_normal_transient": int(sum(v["alarm_probability"] >= 0.5 and v["event_type"] == "normal_transient" for v in ok.values())),
        "event_type_counts": event_counts,
        "alarm_probability_quantiles": [float(q) for q in np.quantile([v["alarm_probability"] for v in ok.values()], [0, 0.25, 0.5, 0.75, 1])],
        "window_accuracy_on_sample": {"jev": acc(j), "jev_secondary_rule": acc(coherent),"r1": acc(r1), "r2": acc(r2), "mahalanobis": acc(maha), "gate_only": acc(np.ones_like(j))},
        "models": sorted({v.get("model", "") for v in ok.values()}),
        "tokens_input": int(sum(v.get("usage", {}).get("input_tokens", 0) for v in ok.values())),
        "tokens_output": int(sum(v.get("usage", {}).get("output_tokens", 0) for v in ok.values())),
        "latency_median_s": float(np.median([v.get("elapsed_seconds", np.nan) for v in ok.values()])),
    }


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "prepare_r2", "dryrun", "debug"))
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args(argv)
    outdir = Path(a.outdir)
    if a.command == "prepare":
        res = prepare(load_config(), outdir)
        print(json.dumps({k: v for k, v in res.items() if k != "variants"} | {"counts": {k: {"grid": v["n_grid"], "candidates": v["n_candidates"]} for k, v in res.get("variants", {}).items()}}, indent=1, default=str))
        return 0
    if a.command == "prepare_r2":
        res = prepare_r2(outdir)
        print(json.dumps({k: v for k, v in res.items() if k != "variants"}, indent=1, default=str))
        return 0
    # the debug sample is always drawn from the round-1 grid so both rounds see the same windows
    chosen = select_debug(json.loads((outdir / "prepared.json").read_text(encoding="utf-8")), a.n)
    prepared = json.loads((outdir / ("prepared.json" if a.round == 1 else "prepared_r2.json")).read_text(encoding="utf-8"))
    stage = f"p1_debug_r{a.round}"
    rows = {r["raw_index"]: r for r in prepared["variants"]["dataset04"]["grid"]}
    decisions: dict[int, dict[str, Any]] = {}
    for raw_i in chosen:
        path = Path(rows[raw_i]["request_path"])
        if a.command == "dryrun" or not a.live:
            d = parse_response_v2(mock_response(json.loads(path.read_text(encoding="utf-8"))))
            decisions[raw_i] = {"status": "ok", "alarm_probability": d.alarm_probability, "event_type": d.event_type, "model": d.model, "usage": d.usage, "elapsed_seconds": 0.0}
        else:
            decisions[raw_i] = call_live(outdir, path, stage=stage, tag=f"{stage}_dataset04_{raw_i:05d}")
            print(raw_i, decisions[raw_i].get("status"), decisions[raw_i].get("alarm_probability"), decisions[raw_i].get("event_type"), flush=True)
    summary = {"chosen": chosen, "round": a.round, "mode": "live" if a.live and a.command == "debug" else "mock", **summarize_debug(prepared, decisions)}
    suffix = "" if a.round == 1 else "_r2"
    name = f"debug_summary_live{suffix}.json" if summary["mode"] == "live" else f"dryrun_summary{suffix}.json"
    (outdir / name).write_text(json.dumps(_safe(summary), indent=1), encoding="utf-8")
    print(json.dumps(_safe(summary), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

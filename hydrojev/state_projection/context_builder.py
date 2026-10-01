"""Assemble a compact, JSON-safe Jev state from zone evidence.

The builder keeps observed, surrogate-predicted, inferred, and attacked values in
distinct fields; marks anything missing with an explicit ``unknown`` /
``unavailable`` token instead of imputing it; rejects NaN/Inf; and bounds the
payload by summarizing long evidence windows rather than dumping every sample.
The output is a plain nested ``dict`` that :func:`json.dumps` accepts directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.state_projection.residual_engine import DetectorReading

SCHEMA_VERSION = "hydrojev.jev_state/1"
UNKNOWN = "unknown"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class FreshnessInfo:
    state_age_s: float
    maximum_allowed_age_s: float

    @property
    def is_fresh(self) -> bool:
        return self.state_age_s <= self.maximum_allowed_age_s


@dataclass(frozen=True)
class ZoneIdentity:
    network_name: str
    zone_id: str
    junction_count: int
    boundary_links: tuple[str, ...]
    protected_nodes: tuple[str, ...]
    headloss_model: str = UNKNOWN
    hydraulic_timestep_s: float | None = None
    observation_interval_s: float | None = None
    minimum_wave_propagation_time_s: float | None = None


def _finite_scalar(value: float, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} is NaN or infinity")
    return number


def _json_safe(obj: Any) -> Any:
    """Coerce to JSON-serializable types, rejecting non-finite floats."""

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, DetectorReading):
        return {
            "value": _json_safe(obj.value),
            "threshold": _json_safe(obj.threshold),
            "state": obj.state.value,
            "detail": obj.detail,
            "source": obj.source,
        }
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _finite_scalar(float(obj), "value")
    if isinstance(obj, np.ndarray):
        return [_json_safe(item) for item in obj.tolist()]
    if isinstance(obj, Mapping):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    raise TypeError(f"cannot serialize object of type {type(obj).__name__}")


def summarize_series(values: Sequence[float], *, max_points: int = 8) -> dict[str, Any]:
    """Summary statistics plus a decimated tail, so long windows stay compact."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] == 0:
        return {"count": 0, "state": UNAVAILABLE}
    if not np.isfinite(array).all():
        raise ValueError("evidence series contains NaN or infinity")
    if max_points < 1:
        raise ValueError("max_points must be positive")
    if array.shape[0] <= max_points:
        sample = array
    else:
        # Even decimation keeps the first and last sample for context.
        index = np.linspace(0, array.shape[0] - 1, max_points).round().astype(int)
        sample = array[np.unique(index)]
    return {
        "count": int(array.shape[0]),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "last": float(array[-1]),
        "sampled": [float(x) for x in sample],
    }


def _detector_block(readings: Sequence[DetectorReading]) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for reading in readings:
        block[reading.name] = _json_safe(reading)
    return block


def _maintenance_block(context: Mapping[str, Any] | None) -> dict[str, Any]:
    # Absent maintenance evidence is marked, never assumed benign.
    default = {
        "scheduled_work_orders": [],
        "recent_valve_operations": [],
        "known_sensor_faults": [],
        "weather_event": UNKNOWN,
    }
    if not context:
        return default
    default.update({str(k): _json_safe(v) for k, v in context.items()})
    return default


def build_jev_state(
    *,
    timestamp: str,
    freshness: FreshnessInfo,
    zone: ZoneIdentity,
    detector_readings: Sequence[DetectorReading],
    observed_vs_predicted: Mapping[str, Mapping[str, float]],
    hydraulic_consistency: Mapping[str, float],
    attack_surface: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
    evidence_windows: Mapping[str, Sequence[float]] | None = None,
    provenance: Mapping[str, Any] | None = None,
    maintenance_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the nested, JSON-safe Jev state for one defence zone at one instant."""

    mass_residual = _finite_scalar(
        hydraulic_consistency.get("mass_balance_residual_m3_s", float("nan")),
        "mass_balance_residual_m3_s",
    )
    tolerance = _finite_scalar(
        hydraulic_consistency.get("mass_balance_tolerance_m3_s", float("nan")),
        "mass_balance_tolerance_m3_s",
    )

    observed_block: dict[str, Any] = {}
    for name, fields in observed_vs_predicted.items():
        observed_block[str(name)] = {
            key: _finite_scalar(val, f"{name}.{key}") for key, val in fields.items()
        }

    windows_block = {
        str(name): summarize_series(series)
        for name, series in (evidence_windows or {}).items()
    }

    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "freshness": {
            "state_age_s": _finite_scalar(freshness.state_age_s, "state_age_s"),
            "maximum_allowed_age_s": _finite_scalar(
                freshness.maximum_allowed_age_s, "maximum_allowed_age_s"
            ),
            "is_fresh": bool(freshness.is_fresh),
        },
        "zone": {
            "network_name": zone.network_name,
            "zone_id": zone.zone_id,
            "junction_count": int(zone.junction_count),
            "boundary_links": list(zone.boundary_links),
            "protected_nodes": list(zone.protected_nodes),
            "headloss_model": zone.headloss_model,
            "hydraulic_timestep_s": (
                zone.hydraulic_timestep_s if zone.hydraulic_timestep_s is not None else UNKNOWN
            ),
            "observation_interval_s": (
                zone.observation_interval_s
                if zone.observation_interval_s is not None
                else UNKNOWN
            ),
            "minimum_wave_propagation_time_s": (
                zone.minimum_wave_propagation_time_s
                if zone.minimum_wave_propagation_time_s is not None
                else UNKNOWN
            ),
        },
        "detectors": _detector_block(detector_readings),
        "observed_vs_predicted": observed_block,
        "evidence_windows": windows_block,
        "hydraulic_consistency": {
            "mass_balance_residual_m3_s": mass_residual,
            "mass_balance_tolerance_m3_s": tolerance,
            "within_tolerance": bool(abs(mass_residual) <= tolerance),
        },
        "attack_surface": _json_safe(dict(attack_surface)),
        "thresholds": _json_safe(dict(thresholds or {})),
        "maintenance_context": _maintenance_block(maintenance_context),
        "provenance": _json_safe(dict(provenance or {"note": UNKNOWN})),
    }
    # A single coercion pass guarantees the whole payload is JSON-safe and finite.
    return _json_safe(state)


def serialize_state(state: Mapping[str, Any], *, max_bytes: int = 16384) -> str:
    """Serialize to compact JSON, enforcing an explicit payload-size budget."""

    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    size = len(encoded.encode("utf-8"))
    if size > max_bytes:
        raise ValueError(
            f"Jev state payload is {size} bytes, exceeding the {max_bytes}-byte budget; "
            "summarize evidence further"
        )
    return encoded

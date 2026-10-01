"""Orthogonal residual detectors for HydroJEV cyber-physical defence zones.

Three detector families answer independent questions about a zone, so a
concealment attack that fools one is unlikely to fool all:

* **Normalized CPDZ residual** -- the instantaneous mismatch between observed
  measurements and the surrogate's prediction, standardized by a robust healthy
  baseline scale. This is the primary detector.
* **GEpFM baseline drift** -- a HydroJEV *auxiliary* slow-drift detector (not
  drawn from the reference papers). It tracks an exponentially weighted moving
  average of the standardized residual and flags when the smoothed level drifts
  away from its healthy mean, catching gradual biasing that an instantaneous
  residual misses.
* **RAT covariance distortion** -- a HydroJEV *auxiliary* detector (again not
  from the reference papers) comparing the residual *covariance structure* over
  a window against a healthy reference covariance. An attacker who matches the
  mean residual can still distort correlations between measurements.

Pump-energy integration supports the excess-pumping-energy metric. Every
statistic rejects NaN/Inf inputs, regularizes singular covariances, and reports
an explicit ``unavailable`` state when the evidence window is too short rather
than fabricating a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Standard gravity and the density of water at ~20 C, in SI units.
GRAVITY_M_S2 = 9.80665
WATER_DENSITY_KG_M3 = 998.2
_JOULES_PER_KWH = 3.6e6
# 1.4826 rescales the median absolute deviation to a Gaussian-consistent stddev.
_MAD_TO_SIGMA = 1.4826


class DetectorState(str, Enum):
    NOMINAL = "nominal"
    ALERTING = "alerting"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DetectorReading:
    """One detector's verdict, kept auditable and JSON-safe."""

    name: str
    value: float | None
    threshold: float | None
    state: DetectorState
    detail: str
    source: str  # "primary" or "hydrojev_auxiliary"


def _require_finite(array: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return values


# --------------------------------------------------------------------------- #
# Robust scaling and the normalized CPDZ residual
# --------------------------------------------------------------------------- #


def robust_feature_scale(baseline_window: np.ndarray, *, floor: float = 1e-9) -> np.ndarray:
    """Per-feature robust scale (MAD-based sigma) from a healthy baseline window.

    ``baseline_window`` is ``[samples, features]``. The median absolute deviation
    is rescaled by 1.4826 to match a Gaussian standard deviation and floored so a
    constant feature never produces a divide-by-zero in standardization.
    """

    window = _require_finite(baseline_window, "baseline_window")
    if window.ndim != 2 or window.shape[0] < 1:
        raise ValueError("baseline_window must be a non-empty 2D array")
    median = np.median(window, axis=0)
    mad = np.median(np.abs(window - median), axis=0)
    scale = _MAD_TO_SIGMA * mad
    return np.maximum(scale, floor)


def normalized_cpdz_residual(
    observed: np.ndarray,
    predicted: np.ndarray,
    scale: np.ndarray,
) -> float:
    """RMS of standardized per-feature residuals: dimensionless and cross-zone comparable."""

    obs = _require_finite(observed, "observed").reshape(-1)
    pred = _require_finite(predicted, "predicted").reshape(-1)
    sig = _require_finite(scale, "scale").reshape(-1)
    if not obs.shape == pred.shape == sig.shape:
        raise ValueError("observed, predicted, and scale must share one shape")
    if np.any(sig <= 0):
        raise ValueError("scale entries must be positive")
    standardized = (obs - pred) / sig
    return float(np.sqrt(np.mean(np.square(standardized))))


def cpdz_residual_reading(
    observed: np.ndarray,
    predicted: np.ndarray,
    scale: np.ndarray,
    *,
    threshold: float,
) -> DetectorReading:
    value = normalized_cpdz_residual(observed, predicted, scale)
    state = DetectorState.ALERTING if value > threshold else DetectorState.NOMINAL
    return DetectorReading(
        name="cpdz_residual_normalized",
        value=value,
        threshold=float(threshold),
        state=state,
        detail="RMS standardized observed-vs-surrogate residual",
        source="primary",
    )


# --------------------------------------------------------------------------- #
# Pump energy integration
# --------------------------------------------------------------------------- #


def integrate_pump_energy(
    times_s: np.ndarray,
    flow_m3_s: np.ndarray,
    head_m: np.ndarray,
    *,
    efficiency: float = 0.75,
    density_kg_m3: float = WATER_DENSITY_KG_M3,
) -> float:
    """Trapezoidal electrical pump energy in kWh over the sampled window.

    Hydraulic power is ``rho * g * Q * H``; dividing by the wire-to-water
    efficiency gives electrical power, which is integrated over ``times_s``.
    """

    t = _require_finite(times_s, "times_s").reshape(-1)
    q = _require_finite(flow_m3_s, "flow_m3_s").reshape(-1)
    h = _require_finite(head_m, "head_m").reshape(-1)
    if not t.shape == q.shape == h.shape:
        raise ValueError("times, flow, and head must share one shape")
    if t.shape[0] < 2:
        raise ValueError("energy integration needs at least two samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times_s must be strictly increasing")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("efficiency must lie in (0, 1]")
    power_w = density_kg_m3 * GRAVITY_M_S2 * q * h / efficiency
    # Trapezoidal rule written out (np.trapz was removed in NumPy 2.0).
    energy_joules = float(np.sum(0.5 * (power_w[1:] + power_w[:-1]) * np.diff(t)))
    return energy_joules / _JOULES_PER_KWH


def extra_energy_fraction(attacked_kwh: float, baseline_kwh: float) -> float:
    """Fractional extra pumping energy relative to a nominal baseline."""

    if not np.isfinite(attacked_kwh) or not np.isfinite(baseline_kwh):
        raise ValueError("energy values must be finite")
    if baseline_kwh <= 0:
        raise ValueError("baseline energy must be positive")
    return (attacked_kwh - baseline_kwh) / baseline_kwh


# --------------------------------------------------------------------------- #
# GEpFM baseline drift (HydroJEV auxiliary)
# --------------------------------------------------------------------------- #


def ewma_level(series: np.ndarray, *, smoothing: float) -> float:
    """Return the final exponentially weighted moving average level."""

    values = _require_finite(series, "series").reshape(-1)
    if values.shape[0] < 1:
        raise ValueError("series must be non-empty")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    level = float(values[0])
    for sample in values[1:]:
        level = smoothing * float(sample) + (1.0 - smoothing) * level
    return level


def gepfm_baseline_drift(
    residual_series: np.ndarray,
    *,
    reference_mean: float,
    reference_scale: float,
    smoothing: float = 0.2,
    threshold: float = 3.0,
) -> DetectorReading:
    """Standardized drift of the smoothed residual level from its healthy mean."""

    if reference_scale <= 0:
        raise ValueError("reference_scale must be positive")
    level = ewma_level(residual_series, smoothing=smoothing)
    drift = abs(level - reference_mean) / reference_scale
    state = DetectorState.ALERTING if drift > threshold else DetectorState.NOMINAL
    return DetectorReading(
        name="gepfm_baseline_drift",
        value=float(drift),
        threshold=float(threshold),
        state=state,
        detail="standardized EWMA drift of the residual level from baseline",
        source="hydrojev_auxiliary",
    )


# --------------------------------------------------------------------------- #
# RAT covariance distortion (HydroJEV auxiliary)
# --------------------------------------------------------------------------- #


def regularize_covariance(covariance: np.ndarray, *, ridge: float) -> np.ndarray:
    """Add a ridge to the diagonal so a singular covariance becomes usable."""

    cov = _require_finite(covariance, "covariance")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    return cov + ridge * np.eye(cov.shape[0])


def rat_covariance_distortion(
    residual_window: np.ndarray,
    reference_covariance: np.ndarray,
    *,
    ridge: float = 1e-6,
    threshold: float = 0.25,
    minimum_samples: int | None = None,
) -> DetectorReading:
    """Normalized Frobenius distance between observed and reference covariances.

    Requires at least ``minimum_samples`` rows (default ``features + 1``) to form
    a meaningful sample covariance; with fewer it returns an ``unavailable``
    reading rather than a spurious value.
    """

    window = _require_finite(residual_window, "residual_window")
    if window.ndim != 2:
        raise ValueError("residual_window must be 2D [samples, features]")
    n_samples, n_features = window.shape
    needed = (n_features + 1) if minimum_samples is None else int(minimum_samples)
    if n_samples < needed:
        return DetectorReading(
            name="rat_covariance_distortion",
            value=None,
            threshold=float(threshold),
            state=DetectorState.UNAVAILABLE,
            detail=f"insufficient window: {n_samples} < {needed} samples required",
            source="hydrojev_auxiliary",
        )
    reference = regularize_covariance(reference_covariance, ridge=ridge)
    if reference.shape[0] != n_features:
        raise ValueError("reference covariance dimension does not match the window")
    observed = regularize_covariance(np.cov(window, rowvar=False), ridge=ridge)
    reference_norm = float(np.linalg.norm(reference, ord="fro"))
    distortion = float(np.linalg.norm(observed - reference, ord="fro") / reference_norm)
    state = DetectorState.ALERTING if distortion > threshold else DetectorState.NOMINAL
    return DetectorReading(
        name="rat_covariance_distortion",
        value=distortion,
        threshold=float(threshold),
        state=state,
        detail="normalized Frobenius distance between residual covariances",
        source="hydrojev_auxiliary",
    )

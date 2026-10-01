"""Tests for the orthogonal residual detectors and pump-energy integration."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.state_projection.residual_engine import (
    DetectorState,
    cpdz_residual_reading,
    ewma_level,
    extra_energy_fraction,
    gepfm_baseline_drift,
    integrate_pump_energy,
    normalized_cpdz_residual,
    rat_covariance_distortion,
    regularize_covariance,
    robust_feature_scale,
)


# --------------------------------------------------------------------------- #
# Robust scaling and normalized CPDZ residual
# --------------------------------------------------------------------------- #


def test_robust_scale_matches_mad_and_floors_constant_features() -> None:
    window = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0], [5.0, 5.0]])
    scale = robust_feature_scale(window, floor=1e-9)
    # Column 0: median 3, MAD 1 -> 1.4826; column 1 is constant -> floored.
    assert scale[0] == pytest.approx(1.4826)
    assert scale[1] == pytest.approx(1e-9)


def test_robust_scale_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        robust_feature_scale(np.array([1.0, 2.0, 3.0]))  # not 2D
    with pytest.raises(ValueError):
        robust_feature_scale(np.array([[1.0, np.nan]]))


def test_normalized_residual_is_zero_when_matched_and_rms_otherwise() -> None:
    assert normalized_cpdz_residual([1.0, 2.0], [1.0, 2.0], [1.0, 1.0]) == 0.0
    # Standardized residuals [2, 0] -> RMS = sqrt(2).
    assert normalized_cpdz_residual([3.0, 2.0], [1.0, 2.0], [1.0, 1.0]) == pytest.approx(
        np.sqrt(2.0)
    )


def test_cpdz_reading_alerts_above_threshold() -> None:
    alerting = cpdz_residual_reading([10.0], [0.0], [1.0], threshold=3.0)
    assert alerting.state is DetectorState.ALERTING
    assert alerting.source == "primary"
    nominal = cpdz_residual_reading([1.0], [0.0], [1.0], threshold=3.0)
    assert nominal.state is DetectorState.NOMINAL


# --------------------------------------------------------------------------- #
# Pump energy
# --------------------------------------------------------------------------- #


def test_pump_energy_matches_hand_computed_constant_power() -> None:
    times = np.array([0.0, 3600.0])
    flow = np.array([0.1, 0.1])
    head = np.array([50.0, 50.0])
    energy = integrate_pump_energy(times, flow, head, efficiency=1.0, density_kg_m3=1000.0)
    # 1000 * 9.80665 * 0.1 * 50 W over 3600 s = 176_519_700 J = 49.033 kWh.
    assert energy == pytest.approx(49.033, rel=1e-3)


def test_pump_energy_rejects_degenerate_time_axis() -> None:
    with pytest.raises(ValueError):
        integrate_pump_energy([0.0], [0.1], [50.0])
    with pytest.raises(ValueError):
        integrate_pump_energy([0.0, 0.0], [0.1, 0.1], [50.0, 50.0])
    with pytest.raises(ValueError):
        integrate_pump_energy([0.0, 1.0], [0.1, 0.1], [50.0, 50.0], efficiency=0.0)


def test_extra_energy_fraction_recovers_reference_percentage() -> None:
    # The reference FS-FDI scenario inflates pumping energy by 62.74%.
    assert extra_energy_fraction(162.74, 100.0) == pytest.approx(0.6274)
    with pytest.raises(ValueError):
        extra_energy_fraction(1.0, 0.0)


# --------------------------------------------------------------------------- #
# GEpFM baseline drift
# --------------------------------------------------------------------------- #


def test_ewma_level_smooths_toward_recent_samples() -> None:
    assert ewma_level([4.0, 4.0, 4.0], smoothing=0.3) == pytest.approx(4.0)
    assert ewma_level([0.0, 10.0], smoothing=0.5) == pytest.approx(5.0)


def test_gepfm_drift_flags_sustained_offset() -> None:
    drifting = gepfm_baseline_drift(
        [5.0, 5.2, 5.1, 5.3], reference_mean=0.0, reference_scale=1.0, threshold=3.0
    )
    assert drifting.state is DetectorState.ALERTING
    assert drifting.source == "hydrojev_auxiliary"
    steady = gepfm_baseline_drift(
        [0.1, -0.1, 0.05], reference_mean=0.0, reference_scale=1.0, threshold=3.0
    )
    assert steady.state is DetectorState.NOMINAL


# --------------------------------------------------------------------------- #
# RAT covariance distortion
# --------------------------------------------------------------------------- #


def test_regularize_covariance_makes_singular_matrix_invertible() -> None:
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.linalg.matrix_rank(singular) == 1
    regularized = regularize_covariance(singular, ridge=0.1)
    assert np.all(np.linalg.eigvalsh(regularized) > 0)


def test_rat_reports_unavailable_for_short_window() -> None:
    reading = rat_covariance_distortion(
        np.zeros((1, 3)), np.eye(3), threshold=0.25
    )
    assert reading.state is DetectorState.UNAVAILABLE
    assert reading.value is None
    assert "insufficient" in reading.detail


def test_rat_is_zero_for_matching_covariance_and_alerts_on_distortion() -> None:
    window = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]]
    )
    reference = np.cov(window, rowvar=False)
    matched = rat_covariance_distortion(window, reference, ridge=1e-9, threshold=0.25)
    assert matched.value == pytest.approx(0.0, abs=1e-6)
    assert matched.state is DetectorState.NOMINAL

    distorted = rat_covariance_distortion(window, 3.0 * reference, ridge=1e-9, threshold=0.25)
    assert distorted.value == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert distorted.state is DetectorState.ALERTING


def test_rat_rejects_nonfinite_window() -> None:
    with pytest.raises(ValueError):
        rat_covariance_distortion(np.array([[np.inf, 0.0], [0.0, 1.0]]), np.eye(2))

"""Unit tests for the anomaly-detection baselines and fair scoring harness."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.detection_baselines import (
    calibrate_threshold,
    contiguous_runs,
    evaluate_scores,
    fit_isolation_forest_detector,
    fit_mahalanobis_detector,
    fit_pca_residual_detector,
)


@pytest.fixture
def normal_train() -> np.ndarray:
    rng = np.random.default_rng(0)
    # Correlated 4-D Gaussian: channel 1 tracks channel 0 (like coupled sensors).
    base = rng.normal(size=(400, 1))
    noise = rng.normal(scale=0.05, size=(400, 4))
    return np.hstack([base, base, -base, base]) + noise


def _labelled_eval(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """200 benign rows, then a contiguous 40-row attack that breaks correlation.

    The shift is large enough to push the affected channels outside their training
    marginals, so it is a clear outlier for every baseline family (subspace,
    covariance, and axis-aligned tree). Subtler correlation-only breaks — where
    Isolation Forest's axis-aligned splits underperform Mahalanobis/PCA — are the
    subject of the real BATADAL benchmark, not this contract test.
    """
    base = rng.normal(size=(240, 1))
    noise = rng.normal(scale=0.05, size=(240, 4))
    data = np.hstack([base, base, -base, base]) + noise
    labels = np.zeros(240, dtype=int)
    # Inject a coordinated shift that violates the learned structure.
    data[200:240, 1] += 6.0
    data[200:240, 2] += 6.0
    labels[200:240] = 1
    return data, labels


# --------------------------------------------------------------------------- #
# Threshold calibration + run extraction
# --------------------------------------------------------------------------- #


def test_threshold_calibrated_at_training_quantile() -> None:
    scores = np.linspace(0.0, 1.0, 1001)
    threshold = calibrate_threshold(scores, 0.995)
    assert abs(threshold - 0.995) < 1e-6
    # ~0.5% of training scores exceed the 99.5th-percentile threshold.
    assert 0.0 <= np.mean(scores >= threshold) <= 0.01


def test_calibrate_threshold_rejects_bad_quantile() -> None:
    with pytest.raises(ValueError):
        calibrate_threshold(np.arange(10.0), 1.5)


def test_contiguous_runs_finds_maximal_runs() -> None:
    flags = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1])
    assert contiguous_runs(flags, 1) == [(1, 3), (5, 6), (7, 10)]
    assert contiguous_runs(flags, 0) == [(0, 1), (3, 5), (6, 7)]
    assert contiguous_runs(np.ones(3), 1) == [(0, 3)]  # trailing run closes


# --------------------------------------------------------------------------- #
# Each baseline: fits on normal data only, then discriminates injected attacks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fitter",
    [fit_pca_residual_detector, fit_mahalanobis_detector, fit_isolation_forest_detector],
)
def test_baseline_obeys_contract_and_ranks(fitter, normal_train) -> None:
    """Every baseline honours the shared contract and *ranks* attacks well.

    ROC-AUC is calibration-independent, so it is the fair discrimination claim
    that must hold for all three families. The operating-point recall at the
    shared quantile threshold is asserted separately, because it does not hold
    uniformly (see the Isolation Forest saturation test below).
    """
    detector = fitter(normal_train, threshold_quantile=0.995)
    # Shared scoring contract.
    assert isinstance(detector.name, str) and detector.name
    assert isinstance(detector.family, str) and detector.family
    assert isinstance(detector.threshold, float)
    assert detector.provenance
    train_scores = detector.score_samples(normal_train)
    assert train_scores.shape == (len(normal_train),)
    # Calibrated threshold trips on ~0.5% of the training data (no leakage).
    assert np.mean(train_scores >= detector.threshold) <= 0.02

    data, labels = _labelled_eval(np.random.default_rng(1))
    metrics = evaluate_scores(detector.score_samples(data), detector.threshold, labels)
    # A coordinated structural break is highly separable for every baseline.
    assert metrics["roc_auc"] > 0.9
    # Calibrated on clean data -> benign false positives stay low.
    assert metrics["point_false_positive_rate"] <= 0.1


@pytest.mark.parametrize("fitter", [fit_pca_residual_detector, fit_mahalanobis_detector])
def test_unbounded_score_baselines_trip_at_operating_point(fitter, normal_train) -> None:
    """Residual/distance scores grow without bound, so a clear outlier trips.

    PCA reconstruction residual and Mahalanobis distance both increase
    monotonically with deviation magnitude, so a decisive outlier always exceeds
    the training-quantile threshold. This is the property Isolation Forest lacks.
    """
    detector = fitter(normal_train, threshold_quantile=0.995)
    data, labels = _labelled_eval(np.random.default_rng(1))
    metrics = evaluate_scores(detector.score_samples(data), detector.threshold, labels)
    assert metrics["scenario_recall"] == 1.0


def test_isolation_forest_ranks_but_saturates_at_operating_point(normal_train) -> None:
    """Document Isolation Forest's score saturation: great ranking, weak operating point.

    IF path-length scores saturate, so a decisive magnitude outlier is scored
    barely above a natural Gaussian tail. Its ROC-AUC stays high (ranking is
    fine), but at the shared 99.5th-percentile-of-training threshold — a policy
    driven by IF's own natural training tails — the attack can fail to trip. This
    is exactly why the paper reports threshold-independent AUC alongside the
    operating point rather than recall alone.
    """
    detector = fit_isolation_forest_detector(normal_train, threshold_quantile=0.995)
    data, labels = _labelled_eval(np.random.default_rng(1))
    metrics = evaluate_scores(detector.score_samples(data), detector.threshold, labels)
    assert metrics["roc_auc"] > 0.9  # ranking is excellent
    # Saturation gap: the calibrated operating point does not achieve full recall.
    assert metrics["scenario_recall"] < 1.0


def test_isolation_forest_is_seed_deterministic(normal_train) -> None:
    a = fit_isolation_forest_detector(normal_train, seed=7, n_estimators=50)
    b = fit_isolation_forest_detector(normal_train, seed=7, n_estimators=50)
    assert np.allclose(a.score_samples(normal_train), b.score_samples(normal_train))


@pytest.mark.parametrize(
    "fitter",
    [fit_pca_residual_detector, fit_mahalanobis_detector, fit_isolation_forest_detector],
)
def test_fitters_reject_degenerate_input(fitter) -> None:
    with pytest.raises(ValueError):
        fitter(np.array([[1.0, np.nan]]))  # non-finite
    with pytest.raises(ValueError):
        fitter(np.array([1.0, 2.0, 3.0]))  # 1D


# --------------------------------------------------------------------------- #
# Fair scoring harness
# --------------------------------------------------------------------------- #


def test_evaluate_scores_exact_operating_point() -> None:
    # labels: benign[0:3], attack[3:6], benign[6:8]  -> 1 attack run, 2 benign runs
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    scores = np.array([0.0, 0.0, 0.0, 0.9, 0.1, 0.1, 0.0, 0.8])
    m = evaluate_scores(scores, threshold=0.5, labels=labels)
    assert m["attack_scenarios"] == 1
    assert m["benign_scenarios"] == 2
    assert m["scenario_recall"] == 1.0            # the attack run has one alarm
    assert m["point_recall"] == pytest.approx(1 / 3)  # 1 of 3 attack points alarms
    assert m["benign_scenario_fp_rate"] == 0.5    # 1 of 2 benign runs trips (idx 7)
    assert m["point_false_positive_rate"] == pytest.approx(1 / 5)


def test_evaluate_scores_perfect_separation_auc_one() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.95, 0.8, 0.85])
    m = evaluate_scores(scores, threshold=0.5, labels=labels)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_evaluate_scores_single_class_is_nan_not_crash() -> None:
    labels = np.zeros(10, dtype=int)
    m = evaluate_scores(np.linspace(0, 1, 10), threshold=0.5, labels=labels)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])
    assert np.isnan(m["scenario_recall"])  # no attack runs

"""Unit tests for the evasion-benchmark aggregation and presentation helpers.

These exercise the pure logic (Wilson interval, target selection, per-detector
summary via a toy scorer, table/figure) without touching BATADAL or torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hydrojev.benchmarks.evasion_benchmark import (
    _format_table,
    _median,
    _select_targets,
    _summarize_detector,
    plot_evasion,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# wilson_interval
# --------------------------------------------------------------------------- #


def test_wilson_interval_brackets_point_estimate() -> None:
    lo, hi = wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # Known Wilson values for 5/10 at 95%: ~[0.237, 0.763].
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)


def test_wilson_interval_extremes_stay_in_unit_interval() -> None:
    lo0, hi0 = wilson_interval(0, 20)
    lo1, hi1 = wilson_interval(20, 20)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.25
    assert 0.75 < lo1 < 1.0 and hi1 == 1.0


def test_wilson_interval_empty_is_nan() -> None:
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


# --------------------------------------------------------------------------- #
# _median / _select_targets
# --------------------------------------------------------------------------- #


def test_median_ignores_nan_and_empty() -> None:
    assert _median([1.0, 2.0, 3.0]) == 2.0
    assert _median([1.0, np.nan, 3.0]) == 2.0
    assert np.isnan(_median([]))
    assert np.isnan(_median([np.nan, np.nan]))


def test_select_targets_picks_detected_attacks_only() -> None:
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.95, 0.5])
    labels = np.array([0, 1, 1, 1, 0, 1])
    # threshold 0.6: detected attacks are indices 1 and 2 (label 1 & score>=0.6);
    # index 4 scores high but is benign; indices 3,5 are attacks but under threshold.
    picked = _select_targets(scores, labels, threshold=0.6, max_targets=10)
    assert picked.tolist() == [1, 2]


def test_select_targets_subsamples_evenly_and_capped() -> None:
    scores = np.ones(100)
    labels = np.ones(100, dtype=int)
    picked = _select_targets(scores, labels, threshold=0.5, max_targets=5)
    assert picked.size <= 5
    assert picked[0] == 0 and picked[-1] == 99  # endpoints retained
    assert np.all(np.diff(picked) > 0)  # strictly increasing / unique


def test_select_targets_empty_when_nothing_detected() -> None:
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 0])
    assert _select_targets(scores, labels, threshold=0.9, max_targets=5).size == 0


# --------------------------------------------------------------------------- #
# _summarize_detector on a toy scorer
# --------------------------------------------------------------------------- #


@dataclass
class _AdditiveScorer:
    """score = sum of features; threshold fixed. Every in-range cut lowers it."""

    threshold: float
    family: str = "toy"

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float).sum(axis=1)


def test_summarize_detector_reports_evasion_and_effort() -> None:
    # 3 features, all modifiable, box [0,1]. Attack rows sum to 3 (>threshold 1.5),
    # so every attack is "detected" and can be driven to 0 by zeroing features.
    scorer = _AdditiveScorer(threshold=1.5)
    features = np.array(
        [
            [0.0, 0.0, 0.0],  # benign, score 0
            [1.0, 1.0, 1.0],  # attack, score 3 -> detected
            [1.0, 1.0, 1.0],  # attack, score 3 -> detected
        ]
    )
    labels = np.array([0, 1, 1])
    low, high = np.zeros(3), np.ones(3)
    mask = np.ones(3, dtype=bool)

    summary = _summarize_detector(
        "toy", scorer, is_hydrojev=False, stochastic=False, provenance="p",
        features=features, labels=labels, feature_low=low, feature_high=high,
        modifiable_mask=mask, max_changed=3, candidate_values=11, budget=500, max_targets=10,
    )

    assert summary["n_detected_attack_points"] == 2
    assert summary["n_targets"] == 2
    assert summary["n_evaded"] == 2
    assert summary["evasion_success_rate"] == 1.0
    assert 0.0 <= summary["evasion_ci95_low"] <= summary["evasion_ci95_high"] <= 1.0
    # Score driven 3 -> below 1.5 by zeroing at least two of three unit features.
    assert summary["median_score_suppression"] > 0.5
    assert summary["median_sensors_changed_evaded"] >= 2


def test_summarize_detector_no_detected_targets() -> None:
    scorer = _AdditiveScorer(threshold=100.0)  # nothing ever alarms
    features = np.array([[1.0, 1.0], [1.0, 1.0]])
    labels = np.array([1, 1])
    low, high = np.zeros(2), np.ones(2)
    mask = np.ones(2, dtype=bool)
    summary = _summarize_detector(
        "toy", scorer, is_hydrojev=False, stochastic=False, provenance="p",
        features=features, labels=labels, feature_low=low, feature_high=high,
        modifiable_mask=mask, max_changed=2, candidate_values=5, budget=100, max_targets=10,
    )
    assert summary["n_targets"] == 0
    assert np.isnan(summary["evasion_success_rate"])
    assert np.isnan(summary["median_queries"])


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _fake_result() -> dict:
    def det(rate, lo, hi, hj=False):
        return {
            "family": "f", "is_hydrojev": hj, "stochastic": hj, "provenance": "p",
            "n_detected_attack_points": 50, "n_targets": 40, "n_evaded": int(rate * 40),
            "evasion_success_rate": rate, "evasion_ci95_low": lo, "evasion_ci95_high": hi,
            "median_score_suppression": 0.4, "median_sensors_changed_evaded": 3.0,
            "median_scaled_l2_evaded": 0.25, "median_queries": 300.0,
        }

    return {
        "status": "ok",
        "evaluate_dataset": "BATADAL:dataset04",
        "max_changed_sensors": 4,
        "detectors": {
            "HydroJEV AE": det(0.3, 0.2, 0.45, hj=True),
            "Mahalanobis": det(0.8, 0.66, 0.9),
        },
    }


def test_format_table_lists_every_detector() -> None:
    text = _format_table(_fake_result())
    assert "HydroJEV AE *" in text
    assert "Mahalanobis" in text
    assert "evasion rate" in text


def test_plot_evasion_writes_file(tmp_path) -> None:
    out = tmp_path / "evasion.png"
    path = plot_evasion(_fake_result(), str(out))
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_evasion_none_when_not_run(tmp_path) -> None:
    assert plot_evasion({"status": "not_run", "reason": "x"}, str(tmp_path / "e.png")) is None

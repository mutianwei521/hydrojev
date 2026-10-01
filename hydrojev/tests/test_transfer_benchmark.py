"""Tests for the cross-detector transfer + ensemble-evasion aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hydrojev.benchmarks.transfer_benchmark import (
    ensemble_all_silent_fraction,
    evaluate_joint_evasion,
    format_transfer_table,
    plot_transfer,
    score_scale,
    still_alerting_fraction,
)


@dataclass
class CoordScorer:
    """score = x[coord]; alerts at or above threshold on that one axis."""

    coord: int
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)[:, self.coord]


@dataclass
class ConstScorer:
    value: float
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(values).shape[0], self.value, dtype=float)


def test_score_scale_uses_std_and_falls_back_to_one() -> None:
    feats = np.array([[0.0], [1.0], [2.0], [3.0]])
    assert score_scale(CoordScorer(0, 0.5), feats) == pytest.approx(np.std([0, 1, 2, 3]))
    # Constant scores -> std 0 -> safe fallback so joint-margin division is finite.
    assert score_scale(ConstScorer(7.0, 1.0), feats) == 1.0


def test_still_alerting_fraction() -> None:
    det = CoordScorer(0, 0.5)
    samples = np.array([[0.6, 0.0], [0.4, 0.0], [0.9, 0.0]])
    assert still_alerting_fraction(det, samples) == pytest.approx(2 / 3)
    # Empty adversarial set -> undefined, reported as nan (no successful evasions).
    assert np.isnan(still_alerting_fraction(det, np.empty((0, 2))))


def test_ensemble_all_silent_fraction() -> None:
    a, b = CoordScorer(0, 0.5), CoordScorer(1, 0.5)
    samples = np.array([[0.1, 0.1], [0.6, 0.1], [0.1, 0.6], [0.6, 0.6]])
    # Only the first row has BOTH detectors silent.
    assert ensemble_all_silent_fraction([a, b], samples) == pytest.approx(0.25)
    assert np.isnan(ensemble_all_silent_fraction([a, b], np.empty((0, 2))))


def _box(n: int):
    return dict(
        feature_low=np.zeros(n),
        feature_high=np.ones(n),
        modifiable_mask=np.ones(n, dtype=bool),
        candidate_values=11,
        budget=2000,
    )


def test_evaluate_joint_evasion_all_evadable() -> None:
    a, b = CoordScorer(0, 0.5), CoordScorer(1, 0.5)
    targets = np.ones((5, 2))  # every row must have both axes pushed down
    out = evaluate_joint_evasion(
        [a, b], targets, scales=[1.0, 1.0], names=["A", "B"], max_changed=2, **_box(2)
    )
    assert out["n_targets"] == 5
    assert out["n_evaded"] == 5
    assert out["joint_evasion_rate"] == pytest.approx(1.0)
    assert out["median_sensors_changed_evaded"] == pytest.approx(2.0)


def test_evaluate_joint_evasion_blocked_by_sparsity() -> None:
    a, b = CoordScorer(0, 0.5), CoordScorer(1, 0.5)
    targets = np.ones((4, 2))
    out = evaluate_joint_evasion(
        [a, b], targets, scales=[1.0, 1.0], names=["A", "B"], max_changed=1, **_box(2)
    )
    # One spoof cannot silence two orthogonal detectors at once.
    assert out["n_evaded"] == 0
    assert out["joint_evasion_rate"] == pytest.approx(0.0)
    assert np.isnan(out["median_sensors_changed_evaded"])


def _fake_result() -> dict:
    order = ["HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest"]

    def srow(vals, n):
        return {"n_detected": n, "n_targets": n, "n_evaded_single": n,
                "still_alerts": dict(zip(order, vals))}

    return {
        "status": "ok",
        "train_dataset": "BATADAL:dataset03",
        "evaluate_dataset": "BATADAL:dataset04",
        "detector_order": order,
        "transfer_matrix": {
            "HydroJEV AE": srow([0.0, 0.9, 0.8, 0.5], 20),
            "Mahalanobis": srow([0.7, 0.0, 0.6, 0.4], 40),
            "PCA residual": srow([0.6, 0.5, 0.0, 0.3], 60),
            "Isolation Forest": srow([0.4, 0.3, 0.2, 0.0], 10),
        },
        "ae_single_evasion_rate": 0.25,
        "ae_single_ci95_low": 0.17,
        "ae_single_ci95_high": 0.35,
        "ae_detected_targets": 60,
        "ensembles": {
            "AE + Mahalanobis": {
                "n_targets": 60, "n_evaded": 3, "joint_evasion_rate": 0.05,
                "evasion_ci95_low": 0.01, "evasion_ci95_high": 0.14,
                "naive_ensemble_evasion_of_ae_samples": 0.10,
            },
            "all four detectors": {
                "n_targets": 60, "n_evaded": 1, "joint_evasion_rate": 0.017,
                "evasion_ci95_low": 0.00, "evasion_ci95_high": 0.09,
                "naive_ensemble_evasion_of_ae_samples": 0.05,
            },
        },
    }


def test_format_transfer_table() -> None:
    text = format_transfer_table(_fake_result())
    assert "Transfer matrix" in text
    assert "AE-alone evasion rate" in text
    assert "AE + Mahalanobis" in text
    # A diagonal cell (AE evaded -> AE still fires) should read ~0.
    assert "0.00" in text


def test_plot_transfer(tmp_path) -> None:
    out = tmp_path / "t.png"
    assert plot_transfer(_fake_result(), str(out)) == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert plot_transfer({"status": "not_run"}, str(tmp_path / "no.png")) is None

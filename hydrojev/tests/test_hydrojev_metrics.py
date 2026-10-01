from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.hydrojev_metrics import (
    binary_grid_metrics,
    episode_metrics,
    forward_hold_grid_predictions,
    official_batadal_score,
    paired_moving_block_bootstrap,
    reconstruct_online_hourly,
    score_grid_and_online,
)


def test_complete_grid_metrics_and_operating_point():
    labels = [0, 0, 1, 1]
    result = binary_grid_metrics(labels, [0.1, 0.2, 0.8, 0.9])
    assert result["complete_grid"] is True
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["threshold_confusion"]["TP"] == 2
    assert result["threshold_confusion"]["TN"] == 2
    assert result["ternary"]["n_alarm"] == 2
    assert result["ternary"]["n_normal"] == 2


def test_missing_grid_probabilities_are_null_and_lower_bound_alarm_false():
    result = binary_grid_metrics([0, 1, 1, 0], [None, 0.9, None, 0.1])
    assert result["complete_grid"] is False
    assert result["roc_auc"] is None
    assert result["n_missing"] == 2
    assert result["available_subset"]["coverage"] == pytest.approx(0.5)
    assert result["available_subset"]["roc_auc"] == pytest.approx(1.0)
    assert result["threshold_confusion"]["TP"] == 1
    assert result["threshold_confusion"]["FP"] == 0
    assert result["threshold_confusion"]["FN"] == 1


def test_forward_hold_is_causal_and_does_not_backfill():
    result = forward_hold_grid_predictions(
        [2, 5, 8], [0.1, 0.9, None], raw_row_count=10, warmup_start=2
    )
    probabilities = result["hourly_probabilities"]
    assert np.isnan(probabilities[:2]).all()
    assert probabilities[2:5].tolist() == [0.1, 0.1, 0.1]
    assert probabilities[5:8].tolist() == [0.9, 0.9, 0.9]
    assert np.isnan(probabilities[8:]).all()
    assert result["grid_failure_count"] == 1
    assert result["excluded_prefix_rows"] == 2


def test_episode_metrics_reconstructs_runs_and_reports_false_alarm_rate():
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    alarms = np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=bool)
    result = episode_metrics(labels, alarms, sample_interval_hours=1.0)
    assert result["attack_scenarios"] == 2
    assert result["detected_attack_scenarios"] == 2
    assert result["episode_recall"] == pytest.approx(1.0)
    assert result["episode_recall_wilson95"][0] > 0.3
    assert result["false_positive_hours"] == 2
    assert result["false_alarm_event_starts"] == 2
    assert result["per_attack"][0]["ttd_hours"] == pytest.approx(1.0)
    assert result["per_attack"][1]["ttd_hours"] == pytest.approx(0.0)


def test_official_batadal_score_matches_known_cases():
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    perfect = official_batadal_score(labels, labels)
    silent = official_batadal_score(labels, np.zeros_like(labels))
    assert perfect["S"] == pytest.approx(1.0)
    assert silent["S_TTD"] == pytest.approx(0.0)
    assert silent["S_CLF"] == pytest.approx(0.5)
    assert silent["S"] == pytest.approx(0.25)


def test_score_wrapper_reports_raw_and_grid_counts():
    raw_labels = np.zeros(20, dtype=int)
    raw_labels[8:12] = 1
    result = score_grid_and_online(
        raw_labels, [2, 8, 14], [0.1, 0.9, 0.1], warmup_start=2
    )
    assert result["alignment"]["forward_hold"] is True
    assert result["alignment"]["backfill"] is False
    assert result["hourly"]["attack_scenarios"] == 1
    assert result["hourly"]["detected_attack_scenarios"] == 1
    assert result["hourly"]["official_batadal"]["S"] is not None


def test_paired_bootstrap_is_reproducible_and_uses_common_rows():
    labels = np.tile([0, 0, 1, 1], 30)
    full = np.where(labels == 1, 0.9, 0.1).astype(float)
    no_jev = np.where(labels == 1, 0.7, 0.3).astype(float)
    full[5] = np.nan
    first = paired_moving_block_bootstrap(labels, full, no_jev, block_len=2, n_boot=100, seed=7)
    second = paired_moving_block_bootstrap(labels, full, no_jev, block_len=2, n_boot=100, seed=7)
    assert first["status"] == "available_subset"
    assert first["n_common_available"] == len(labels) - 1
    assert first["paired"]["brier_score"] == second["paired"]["brier_score"]
    assert first["block_hours"] == pytest.approx(12.0)


def test_invalid_grid_order_is_rejected():
    with pytest.raises(ValueError):
        reconstruct_online_hourly([3, 2], [0.1, 0.9], raw_row_count=5, warmup_start=0)

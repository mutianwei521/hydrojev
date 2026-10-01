"""Tests for multi-seed aggregation of the transfer / ensemble-evasion benchmark.

These exercise the pure aggregation and presentation helpers on synthetic per-seed
result dicts (no BATADAL, no attacks) so the pooling arithmetic, seed spread, matrix
averaging, and ``not_run`` degradation are checked deterministically and fast.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydrojev.benchmarks.multiseed_transfer import (
    _pool,
    aggregate_seed_results,
    format_multiseed_table,
    plot_multiseed,
)


def _seed_result(
    seed: int,
    *,
    ae_rate: float,
    ae_targets: int,
    ens_counts: dict[str, tuple[int, int]],
    matrix: dict[str, dict[str, float]] | None = None,
) -> dict:
    """Build a minimal transfer-benchmark result dict for one seed."""

    order = ["HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest"]
    if matrix is None:
        matrix = {s: {d: 0.5 for d in order} for s in order}
    return {
        "status": "ok",
        "seed": seed,
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "max_changed_sensors": 4,
        "query_budget": 4000,
        "detector_order": order,
        "ae_single_evasion_rate": ae_rate,
        "ae_detected_targets": ae_targets,
        "ensembles": {
            label: {
                "n_evaded": ev,
                "n_targets": tg,
                "joint_evasion_rate": (ev / tg if tg else float("nan")),
                "naive_ensemble_evasion_of_ae_samples": 0.25,
                "members": ["HydroJEV AE", "Mahalanobis"],
            }
            for label, (ev, tg) in ens_counts.items()
        },
        "transfer_matrix": {s: {"still_alerts": matrix[s]} for s in order},
    }


def test_pool_basic_arithmetic():
    p = _pool([2, 4], [10, 10])
    assert p["pooled_evaded"] == 6
    assert p["pooled_targets"] == 20
    assert p["pooled_rate"] == pytest.approx(0.3)
    # mean of per-seed rates 0.2 and 0.4
    assert p["mean"] == pytest.approx(0.3)
    assert p["std"] == pytest.approx(0.1)
    assert 0.0 <= p["pooled_ci95_low"] <= p["pooled_rate"] <= p["pooled_ci95_high"] <= 1.0


def test_pool_ignores_empty_target_seed_in_mean():
    p = _pool([1, 0], [4, 0])
    # second seed has 0 targets -> nan rate -> excluded from mean/std, still pooled
    assert p["n_seeds"] == 1
    assert p["mean"] == pytest.approx(0.25)
    assert p["pooled_targets"] == 4
    assert p["pooled_evaded"] == 1


def test_aggregate_pools_ae_and_ensembles():
    results = [
        _seed_result(42, ae_rate=0.25, ae_targets=60, ens_counts={"AE + Mahalanobis": (9, 60)}),
        _seed_result(7, ae_rate=0.30, ae_targets=60, ens_counts={"AE + Mahalanobis": (12, 60)}),
    ]
    agg = aggregate_seed_results(results)
    assert agg["status"] == "ok"
    assert agg["seeds"] == [42, 7]
    assert agg["n_seeds"] == 2
    # AE evaded recovered as round(rate*targets): 15 and 18 -> pooled 33/120
    assert agg["ae_single"]["pooled_evaded"] == 33
    assert agg["ae_single"]["pooled_targets"] == 120
    assert agg["ae_single"]["pooled_rate"] == pytest.approx(33 / 120)
    ens = agg["ensembles"]["AE + Mahalanobis"]
    assert ens["pooled_evaded"] == 21
    assert ens["pooled_targets"] == 120
    assert ens["pooled_rate"] == pytest.approx(21 / 120)
    assert ens["members"] == ["HydroJEV AE", "Mahalanobis"]


def test_aggregate_averages_transfer_matrix():
    m1 = {s: {d: 0.2 for d in ["HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest"]}
          for s in ["HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest"]}
    m2 = {s: {d: 0.8 for d in ["HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest"]}
          for s in ["HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest"]}
    results = [
        _seed_result(1, ae_rate=0.25, ae_targets=40, ens_counts={"AE + Mahalanobis": (6, 40)}, matrix=m1),
        _seed_result(2, ae_rate=0.25, ae_targets=40, ens_counts={"AE + Mahalanobis": (6, 40)}, matrix=m2),
    ]
    agg = aggregate_seed_results(results)
    assert agg["transfer_off_diagonal_mean"]["HydroJEV AE"]["Mahalanobis"] == pytest.approx(0.5)


def test_aggregate_ignores_nan_matrix_cells():
    order = ["HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest"]
    m1 = {s: {d: float("nan") for d in order} for s in order}
    m2 = {s: {d: 0.6 for d in order} for s in order}
    results = [
        _seed_result(1, ae_rate=0.25, ae_targets=40, ens_counts={"AE + Mahalanobis": (6, 40)}, matrix=m1),
        _seed_result(2, ae_rate=0.25, ae_targets=40, ens_counts={"AE + Mahalanobis": (6, 40)}, matrix=m2),
    ]
    agg = aggregate_seed_results(results)
    # only the finite seed contributes
    assert agg["transfer_off_diagonal_mean"]["HydroJEV AE"]["Mahalanobis"] == pytest.approx(0.6)


def test_aggregate_not_run_when_no_ok_seeds():
    agg = aggregate_seed_results([{"status": "not_run", "reason": "x"}])
    assert agg["status"] == "not_run"


def test_aggregate_skips_failed_seeds():
    results = [
        _seed_result(42, ae_rate=0.25, ae_targets=60, ens_counts={"AE + Mahalanobis": (9, 60)}),
        {"status": "not_run", "reason": "flaky"},
    ]
    agg = aggregate_seed_results(results)
    assert agg["status"] == "ok"
    assert agg["seeds"] == [42]
    assert agg["ae_single"]["pooled_targets"] == 60


def test_format_table_and_plot(tmp_path):
    results = [
        _seed_result(42, ae_rate=0.25, ae_targets=60, ens_counts={"AE + Mahalanobis": (9, 60)}),
        _seed_result(7, ae_rate=0.30, ae_targets=60, ens_counts={"AE + Mahalanobis": (12, 60)}),
    ]
    agg = aggregate_seed_results(results)
    table = format_multiseed_table(agg)
    assert "AE alone" in table
    assert "AE + Mahalanobis" in table
    assert "pooled rate" in table
    out = plot_multiseed(agg, str(tmp_path / "multiseed.png"))
    assert out is not None
    assert (tmp_path / "multiseed.png").exists()


def test_format_table_not_run():
    assert format_multiseed_table({"status": "not_run"}) == "not_run"
    assert plot_multiseed({"status": "not_run"}, "unused.png") is None

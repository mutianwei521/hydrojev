"""Tests for the detection-benchmark aggregation and presentation helpers.

The full runner needs the real BATADAL assets, so it is exercised by the offline
end-to-end run rather than here; these tests lock the seed-aggregation math and
the table/figure shaping that turn its output into paper assets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydrojev.benchmarks.detection_benchmark import (
    _METRIC_KEYS,
    _aggregate,
    _summarize,
    benchmark_to_rows,
    plot_benchmark,
)


def test_summarize_matches_student_t_ci() -> None:
    values = [0.90, 0.92, 0.88, 0.94, 0.86]
    s = _summarize(values)
    arr = np.asarray(values)
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(arr.mean())
    assert s["std"] == pytest.approx(arr.std(ddof=1))
    # 95% half-width from the t(4) critical value.
    from scipy import stats

    half = stats.t.ppf(0.975, 4) * arr.std(ddof=1) / math.sqrt(5)
    assert s["ci95_low"] == pytest.approx(s["mean"] - half)
    assert s["ci95_high"] == pytest.approx(s["mean"] + half)
    assert s["ci95_low"] < s["mean"] < s["ci95_high"]


def test_summarize_single_value_has_zero_width() -> None:
    s = _summarize([0.75])
    assert s == {"mean": 0.75, "std": 0.0, "ci95_low": 0.75, "ci95_high": 0.75, "n": 1}


def test_summarize_ignores_nans_and_handles_empty() -> None:
    s = _summarize([0.5, float("nan"), 0.7])
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.6)
    empty = _summarize([float("nan"), float("nan")])
    assert empty["n"] == 0 and math.isnan(empty["mean"])


def test_aggregate_covers_every_metric_key() -> None:
    run = {k: 0.5 for k in _METRIC_KEYS}
    agg = _aggregate([run, {**run, "roc_auc": 0.7}])
    assert set(agg) == set(_METRIC_KEYS)
    assert agg["roc_auc"]["mean"] == pytest.approx(0.6)
    assert agg["scenario_recall"]["mean"] == pytest.approx(0.5)


def _fake_result() -> dict:
    def det(mean, hj, stochastic):
        return {
            "family": "f",
            "is_hydrojev": hj,
            "stochastic": stochastic,
            "n_runs": 3 if stochastic else 1,
            "provenance": "p",
            "metrics": {k: {"mean": mean, "std": 0.01, "ci95_low": mean - 0.02, "ci95_high": mean + 0.02, "n": 3} for k in _METRIC_KEYS},
            "seed_values": {k: [mean] for k in _METRIC_KEYS},
        }

    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "threshold_quantile": 0.995,
        "seeds": [42, 43, 44],
        "attack_scenarios": 7,
        "benign_scenarios": 8,
        "detectors": {
            "HydroJEV AE": det(0.95, True, True),
            "PCA residual": det(0.80, False, False),
            "Isolation Forest": det(0.70, False, True),
        },
    }


def test_benchmark_to_rows_shapes_one_row_per_detector_metric() -> None:
    rows = benchmark_to_rows(_fake_result())
    # 3 detectors x 5 reported metrics.
    assert len(rows) == 3 * 5
    hydro = [r for r in rows if r["is_hydrojev"]]
    assert {r["metric_key"] for r in hydro} >= {"roc_auc", "scenario_recall", "point_false_positive_rate"}
    assert all("mean" in r and "ci95_low" in r for r in rows)


def test_benchmark_to_rows_empty_when_not_run() -> None:
    assert benchmark_to_rows({"status": "not_run", "reason": "x"}) == []


def test_plot_benchmark_writes_file(tmp_path) -> None:
    out = tmp_path / "bench.png"
    path = plot_benchmark(_fake_result(), str(out))
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_benchmark_none_when_not_run(tmp_path) -> None:
    assert plot_benchmark({"status": "not_run"}, str(tmp_path / "x.png")) is None

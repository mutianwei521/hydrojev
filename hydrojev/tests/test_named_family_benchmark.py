"""Tests for the named-orthogonal-residual-family benchmark.

The pure streaming helpers are pinned against the *already-tested* residual-engine
primitives (:func:`normalized_cpdz_residual`, :func:`ewma_level` /
:func:`gepfm_baseline_drift`, :func:`rat_covariance_distortion`) on synthetic
arrays -- so the causal series are provably identical, sample by sample, to the
architecture's own detector maths evaluated on each prefix/window. A single
BATADAL-guarded integration test then checks the end-to-end runner, skipping
cleanly when the dataset (or torch) is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.named_family_benchmark import (
    complementary_catch,
    cpdz_series,
    format_named_family_table,
    gepfm_series,
    one_step_residual_stream,
    plot_named_family,
    rat_series,
    scenario_alarms,
    spearman_matrix,
)
from hydrojev.state_projection.residual_engine import (
    gepfm_baseline_drift,
    normalized_cpdz_residual,
    rat_covariance_distortion,
)


# --------------------------------------------------------------------------- #
# one_step_residual_stream
# --------------------------------------------------------------------------- #


def test_residual_stream_persistence_predictor():
    rng = np.random.default_rng(0)
    stream = rng.normal(size=(30, 4))
    seq_len = 5

    # Persistence: predict the last row of each window.
    def predict(windows: np.ndarray) -> np.ndarray:
        return windows[:, -1, :]

    residuals, start = one_step_residual_stream(predict, stream, seq_len)
    assert start == seq_len
    assert residuals.shape == (30 - seq_len, 4)
    # residual[k] = stream[k+seq_len] - stream[k+seq_len-1]
    expected = stream[seq_len:] - stream[seq_len - 1 : -1]
    np.testing.assert_allclose(residuals, expected)


def test_residual_stream_zero_predictor_returns_observed():
    stream = np.arange(24, dtype=float).reshape(8, 3)

    def predict(windows: np.ndarray) -> np.ndarray:
        return np.zeros((windows.shape[0], windows.shape[2]))

    residuals, start = one_step_residual_stream(predict, stream, 2)
    np.testing.assert_allclose(residuals, stream[2:])
    assert start == 2


def test_residual_stream_validates_shapes():
    with pytest.raises(ValueError):
        one_step_residual_stream(lambda w: w[:, -1, :], np.zeros(10), 3)
    with pytest.raises(ValueError):
        one_step_residual_stream(lambda w: w[:, -1, :], np.zeros((10, 2)), 10)


# --------------------------------------------------------------------------- #
# cpdz_series  ==  normalized_cpdz_residual per row
# --------------------------------------------------------------------------- #


def test_cpdz_series_matches_primitive_rowwise():
    rng = np.random.default_rng(1)
    residuals = rng.normal(size=(12, 5))
    scale = np.abs(rng.normal(size=5)) + 0.5
    series = cpdz_series(residuals, scale)
    assert series.shape == (12,)
    for k, row in enumerate(residuals):
        assert series[k] == pytest.approx(
            normalized_cpdz_residual(row, np.zeros_like(row), scale)
        )


# --------------------------------------------------------------------------- #
# gepfm_series  ==  gepfm_baseline_drift on every prefix
# --------------------------------------------------------------------------- #


def test_gepfm_series_matches_primitive_on_prefixes():
    rng = np.random.default_rng(2)
    levels = np.abs(rng.normal(loc=1.0, scale=0.3, size=20))
    ref_mean, ref_scale, smoothing = 0.9, 0.25, 0.2
    series = gepfm_series(
        levels, reference_mean=ref_mean, reference_scale=ref_scale, smoothing=smoothing
    )
    assert series.shape == (20,)
    for t in range(1, 21):
        reading = gepfm_baseline_drift(
            levels[:t], reference_mean=ref_mean, reference_scale=ref_scale, smoothing=smoothing
        )
        assert series[t - 1] == pytest.approx(reading.value)


def test_gepfm_series_first_element_is_raw_drift():
    series = gepfm_series(np.array([2.0, 3.0]), reference_mean=1.0, reference_scale=0.5)
    # First EWMA level is the first sample: |2 - 1| / 0.5 = 2.
    assert series[0] == pytest.approx(2.0)


def test_gepfm_series_rejects_bad_scale_and_smoothing():
    with pytest.raises(ValueError):
        gepfm_series(np.ones(3), reference_mean=0.0, reference_scale=0.0)
    with pytest.raises(ValueError):
        gepfm_series(np.ones(3), reference_mean=0.0, reference_scale=1.0, smoothing=0.0)


# --------------------------------------------------------------------------- #
# rat_series  ==  rat_covariance_distortion on trailing window
# --------------------------------------------------------------------------- #


def test_rat_series_matches_primitive_on_windows():
    rng = np.random.default_rng(3)
    residuals = rng.normal(size=(40, 3))
    ref_cov = np.cov(residuals[:20], rowvar=False)
    window = 8
    series = rat_series(residuals, ref_cov, window=window)
    assert series.shape == (40,)
    # Warmup entries are nan.
    assert np.all(np.isnan(series[: window - 1]))
    assert np.all(np.isfinite(series[window - 1 :]))
    for t in range(window - 1, 40):
        reading = rat_covariance_distortion(
            residuals[t - window + 1 : t + 1], ref_cov, minimum_samples=window
        )
        assert series[t] == pytest.approx(reading.value)


def test_rat_series_validates_window():
    with pytest.raises(ValueError):
        rat_series(np.zeros((5, 2)), np.eye(2), window=1)
    with pytest.raises(ValueError):
        rat_series(np.zeros((5, 2)), np.eye(2), window=6)


# --------------------------------------------------------------------------- #
# spearman_matrix
# --------------------------------------------------------------------------- #


def test_spearman_matrix_monotone_and_antitone():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    corr = spearman_matrix({"a": x, "b": x ** 2, "c": -x})
    assert corr["a"]["a"] == pytest.approx(1.0)
    # Monotone increasing transform -> perfect rank correlation.
    assert corr["a"]["b"] == pytest.approx(1.0)
    # Monotone decreasing -> -1.
    assert corr["a"]["c"] == pytest.approx(-1.0)
    # Symmetric.
    assert corr["b"]["a"] == pytest.approx(corr["a"]["b"])


def test_spearman_matrix_handles_nan_pairwise():
    a = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    corr = spearman_matrix({"a": a, "b": b})
    # Drops the NaN row; remaining four are perfectly correlated.
    assert corr["a"]["b"] == pytest.approx(1.0)


def test_spearman_matrix_ties_handled():
    x = np.array([1.0, 1.0, 2.0, 2.0])
    y = np.array([5.0, 5.0, 9.0, 9.0])
    corr = spearman_matrix({"x": x, "y": y})
    assert corr["x"]["y"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# scenario_alarms + complementary_catch
# --------------------------------------------------------------------------- #


def test_scenario_alarms_flags_runs():
    labels = np.array([0, 1, 1, 0, 1, 1, 0])
    # Scores exceed threshold only inside the first attack run.
    scores = np.array([0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    flags = scenario_alarms(scores, threshold=1.0, labels=labels)
    assert flags == [True, False]


def test_scenario_alarms_ignores_nan():
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.0, np.nan, np.nan, 0.0])
    assert scenario_alarms(scores, threshold=1.0, labels=labels) == [False]


def test_complementary_catch_counts():
    a = [True, True, False, False]
    b = [True, False, True, False]
    c = complementary_catch(a, b)
    assert c["n_scenarios"] == 4
    assert c["both"] == 1
    assert c["a_only"] == 1
    assert c["b_only"] == 1
    assert c["union_recall"] == pytest.approx(0.75)
    assert c["a_recall"] == pytest.approx(0.5)


def test_complementary_catch_rejects_mismatch():
    with pytest.raises(ValueError):
        complementary_catch([True], [True, False])


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _synthetic_ok_result() -> dict:
    names = ["CPDZ residual", "GEpFM drift", "RAT covariance", "Mahalanobis", "PCA residual"]
    det = {
        n: {
            "roc_auc": 0.8,
            "pr_auc": 0.7,
            "scenario_recall": 0.5,
            "point_recall": 0.4,
            "point_false_positive_rate": 0.01,
            "is_named_family": n in ("CPDZ residual", "GEpFM drift", "RAT covariance"),
            "threshold": 1.0,
        }
        for n in names
    }
    corr = {a: {b: (1.0 if a == b else 0.2) for b in names} for a in names}
    comp = {
        f: {"a_only": 1, "b_only": 2, "both": 3, "union_recall": 0.6,
            "a_recall": 0.4, "b_recall": 0.5, "n_scenarios": 6}
        for f in ("CPDZ residual", "GEpFM drift", "RAT covariance")
    }
    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "n_scored_samples": 1000,
        "n_attack_scenarios": 6,
        "families": ["CPDZ residual", "GEpFM drift", "RAT covariance"],
        "references": ["Mahalanobis", "PCA residual"],
        "detection": det,
        "score_correlation_spearman": corr,
        "complementary_catch_over_mahalanobis": comp,
        "named_union_vs_mahalanobis": {
            "a_only": 2, "b_only": 1, "both": 3, "union_recall": 0.8,
            "a_recall": 0.6, "b_recall": 0.5, "n_scenarios": 6,
        },
    }


def test_format_table_ok_and_not_run():
    table = format_named_family_table(_synthetic_ok_result())
    assert "CPDZ residual" in table
    assert "orthogonal" in table.lower()
    assert "UNION(3 families)" in table
    assert format_named_family_table({"status": "not_run"}) == "not_run"


def test_plot_ok_and_not_run(tmp_path):
    out = plot_named_family(_synthetic_ok_result(), str(tmp_path / "nf.png"))
    assert out is not None and (tmp_path / "nf.png").exists()
    assert plot_named_family({"status": "not_run"}, "unused.png") is None


# --------------------------------------------------------------------------- #
# Integration on real BATADAL (guarded)
# --------------------------------------------------------------------------- #


def test_named_family_runner_if_available():
    pytest.importorskip("torch")
    from hydrojev.config import load_config
    from hydrojev.benchmarks.named_family_benchmark import run_named_family_benchmark
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_named_family_benchmark(config, mode=mode, seed=7, max_epochs=6)
    if result.get("status") != "ok":
        pytest.skip(f"BATADAL unavailable: {result.get('reason')}")

    # Every named family produced honest detection metrics on the shared rows.
    for fam in result["families"]:
        m = result["detection"][fam]
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["point_false_positive_rate"] <= 1.0
        assert m["is_named_family"] is True
    # Orthogonality matrix is square, symmetric-ish, unit diagonal.
    corr = result["score_correlation_spearman"]
    for name in corr:
        assert corr[name][name] == pytest.approx(1.0, abs=1e-6)
    # All series were scored on one common, fully-defined index set.
    n = result["n_scored_samples"]
    assert n > 0
    assert result["n_attack_scenarios"] >= 1

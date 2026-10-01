"""Tests for the §6.1(a) detection-quality moving-block bootstrap CIs.

The AUC primitive is checked for its single-class / non-finite guard; the bootstrap is
exercised on controllable synthetic (scores, labels) where the ORDERING of detectors is
known a priori (a clearly-stronger detector's paired ROC-AUC difference is positive with
high probability; a statistical twin of the reference yields a difference interval that
contains zero => "on par"); reproducibility, presentation, and a torch-guarded real-data
smoke test round it out. Attacks are laid down as several contiguous runs so moving-block
resamples almost always contain both classes.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.detection_significance import (
    _auc_pair,
    bootstrap_detection_cis,
    format_detection_significance_table,
    plot_detection_significance,
)


# --------------------------------------------------------------------------- #
# AUC primitive
# --------------------------------------------------------------------------- #


def test_auc_pair_guards_single_class_and_non_finite():
    labels = np.array([0, 0, 1, 1])
    roc, pr = _auc_pair(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert roc == pytest.approx(1.0) and pr == pytest.approx(1.0)
    # single-class resample -> NaN (not a fabricated 0.5)
    r2, p2 = _auc_pair(np.array([1, 1, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert np.isnan(r2) and np.isnan(p2)
    # non-finite score -> NaN
    r3, p3 = _auc_pair(labels, np.array([0.1, np.nan, 0.8, 0.9]))
    assert np.isnan(r3) and np.isnan(p3)


# --------------------------------------------------------------------------- #
# synthetic (scores, labels) with a known detector ordering
# --------------------------------------------------------------------------- #


def _labels_multi_run(n: int) -> np.ndarray:
    """Alternating contiguous blocks of length 40 (several attack runs) so moving-block
    resamples reliably see both classes."""

    return ((np.arange(n) // 40) % 2 == 1).astype(int)


def _detectors(rng: np.random.Generator, labels: np.ndarray):
    signal = labels.astype(float)
    n = labels.shape[0]
    return {
        "ref": signal + rng.normal(scale=1.0, size=n),      # moderate separator
        "strong": signal + rng.normal(scale=0.25, size=n),  # clearly better
        "twin": signal + rng.normal(scale=1.0, size=n),      # statistical twin of ref
    }


def test_bootstrap_detection_stronger_detector_beats_reference():
    labels = _labels_multi_run(600)
    scores = _detectors(np.random.default_rng(11), labels)
    res = bootstrap_detection_cis(
        scores, labels, block_len=50, n_boot=400, seed=42, contrast_reference="ref",
    )
    # point estimates respect the construction
    assert res["detectors"]["strong"]["roc_auc"]["point"] > res["detectors"]["ref"]["roc_auc"]["point"]
    strong = res["roc_auc_vs_reference"]["strong"]
    assert strong["roc_auc_diff_point"] > 0.1
    assert strong["frac_ge_zero"] > 0.9  # almost every replicate ranks it above the reference
    # ordered, finite CI
    lo, hi = strong["ci95"]
    assert lo <= strong["roc_auc_diff_point"] <= hi


def test_bootstrap_detection_twin_is_on_par_with_reference():
    labels = _labels_multi_run(600)
    scores = _detectors(np.random.default_rng(12), labels)
    res = bootstrap_detection_cis(
        scores, labels, block_len=50, n_boot=400, seed=42, contrast_reference="ref",
    )
    twin = res["roc_auc_vs_reference"]["twin"]
    assert abs(twin["roc_auc_diff_point"]) < 0.1
    lo, hi = twin["ci95"]
    assert lo < 0.0 < hi           # difference interval contains 0
    assert twin["indistinguishable"] is True  # => reported as "on par"


def test_bootstrap_detection_cis_bracket_points_and_bounded():
    labels = _labels_multi_run(400)
    scores = _detectors(np.random.default_rng(5), labels)
    res = bootstrap_detection_cis(
        scores, labels, block_len=40, n_boot=300, seed=1, contrast_reference="ref",
    )
    for name, d in res["detectors"].items():
        for key in ("roc_auc", "pr_auc"):
            lo, hi = d[key]["ci95"]
            assert 0.0 <= lo <= d[key]["point"] <= hi <= 1.0
        assert 0 < d["n_boot_valid"] <= 300


def test_bootstrap_detection_seed_reproducible():
    labels = _labels_multi_run(400)
    scores = _detectors(np.random.default_rng(7), labels)
    kw = dict(block_len=40, n_boot=200, seed=99, contrast_reference="ref")
    r1 = bootstrap_detection_cis(scores, labels, **kw)
    r2 = bootstrap_detection_cis(scores, labels, **kw)
    assert r1["detectors"]["strong"]["roc_auc"]["ci95"] == r2["detectors"]["strong"]["roc_auc"]["ci95"]
    assert r1["roc_auc_vs_reference"]["twin"]["ci95"] == r2["roc_auc_vs_reference"]["twin"]["ci95"]


def test_bootstrap_detection_rejects_unknown_reference():
    labels = _labels_multi_run(200)
    scores = _detectors(np.random.default_rng(3), labels)
    with pytest.raises(ValueError):
        bootstrap_detection_cis(scores, labels, block_len=20, n_boot=10, seed=0,
                                contrast_reference="nope")


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _synthetic_ok_result() -> dict:
    def det(roc, roc_ci, pr, pr_ci):
        return {"roc_auc": {"point": roc, "ci95": roc_ci},
                "pr_auc": {"point": pr, "ci95": pr_ci}, "n_boot_valid": 2000}

    def con(diff, ci, frac, indist):
        return {"roc_auc_diff_point": diff, "ci95": ci, "frac_ge_zero": frac,
                "indistinguishable": indist}

    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "n_scored_samples": 4106,
        "n_attack_scenarios": 7,
        "n_boot": 2000,
        "primary_block_len": 50,
        "families": ["CPDZ residual", "GEpFM drift", "RAT covariance"],
        "references": ["Mahalanobis", "PCA residual", "HydroJEV AE"],
        "contrast_reference": "Mahalanobis",
        "primary": {
            "block_len": 50,
            "contrast_reference": "Mahalanobis",
            "detectors": {
                "CPDZ residual": det(0.800, [0.62, 0.93], 0.569, [0.35, 0.78]),
                "GEpFM drift": det(0.868, [0.72, 0.96], 0.680, [0.45, 0.86]),
                "RAT covariance": det(0.877, [0.74, 0.96], 0.407, [0.22, 0.63]),
                "Mahalanobis": det(0.821, [0.66, 0.94], 0.648, [0.42, 0.83]),
                "PCA residual": det(0.814, [0.65, 0.93], 0.595, [0.38, 0.80]),
                "HydroJEV AE": det(0.812, [0.65, 0.93], 0.560, [0.34, 0.77]),
            },
            "roc_auc_vs_reference": {
                "CPDZ residual": con(-0.021, [-0.12, 0.09], 0.36, True),
                "GEpFM drift": con(0.047, [-0.05, 0.16], 0.80, True),
                "RAT covariance": con(0.056, [-0.04, 0.17], 0.85, True),
                "PCA residual": con(-0.007, [-0.10, 0.08], 0.44, True),
                "HydroJEV AE": con(-0.009, [-0.11, 0.09], 0.42, True),
            },
        },
    }


def test_format_detection_table_ok_and_not_run():
    text = format_detection_significance_table(_synthetic_ok_result())
    assert "Detection-quality CIs" in text
    assert "ROC-AUC" in text and "PR-AUC" in text
    assert "CPDZ residual" in text and "Mahalanobis" in text
    assert "on par" in text and "0.800" in text
    assert format_detection_significance_table({"status": "not_run"}) == "not_run"


def test_plot_detection_significance(tmp_path):
    out = tmp_path / "det.png"
    path = plot_detection_significance(_synthetic_ok_result(), str(out))
    assert path is not None and out.exists() and out.stat().st_size > 0
    assert plot_detection_significance({"status": "not_run"}, str(tmp_path / "none.png")) is None


# --------------------------------------------------------------------------- #
# guarded real-data smoke test
# --------------------------------------------------------------------------- #


def test_detection_significance_runner_if_available():
    pytest.importorskip("torch")
    from hydrojev.benchmarks.detection_significance import run_detection_significance
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_detection_significance(
        config, mode=mode, seed=7, max_epochs=6, n_boot=120,
        block_lengths=(25, 50), primary_block_len=50,
    )
    if result.get("status") != "ok":
        pytest.skip(f"detection significance not runnable: {result.get('reason')}")

    assert result["n_scored_samples"] > 100
    assert result["n_attack_scenarios"] >= 1
    assert result["contrast_reference"] in result["references"]
    dets = result["primary"]["detectors"]
    assert len(dets) >= 5
    for name, d in dets.items():
        for key in ("roc_auc", "pr_auc"):
            lo, hi = d[key]["ci95"]
            assert 0.0 <= lo <= d[key]["point"] + 1e-9
            assert d[key]["point"] - 1e-9 <= hi <= 1.0
    for name, c in result["primary"]["roc_auc_vs_reference"].items():
        lo, hi = c["ci95"]
        assert lo <= c["roc_auc_diff_point"] <= hi
        assert 0.0 <= c["frac_ge_zero"] <= 1.0

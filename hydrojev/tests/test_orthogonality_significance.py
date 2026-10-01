"""Tests for the §6.1 orthogonality-significance paired moving-block bootstrap.

The correlation primitives are pinned against the already-tested compact rank/Spearman
helpers in ``named_family_benchmark`` (so the bootstrap inherits their correctness), the
moving-block index generator is checked for length/range/contiguity/reproducibility, and
the contrast is exercised on controllable synthetic series where the sign of Delta is
known a priori (correlated instantaneous pair + independent "temporal" series => Delta>0;
three mutually independent series => Delta CI straddles 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.named_family_benchmark import _rankdata, spearman_matrix
from hydrojev.benchmarks.orthogonality_significance import (
    _avg_rank,
    bootstrap_contrast,
    format_significance_table,
    moving_block_indices,
    plot_significance,
    spearman_matrix_fast,
)


# --------------------------------------------------------------------------- #
# rank / correlation primitives pinned to the codebase helpers
# --------------------------------------------------------------------------- #


def test_avg_rank_matches_codebase_rankdata_with_ties():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    x[:40] = 0.0  # a large block of ties to exercise averaging
    x[40:60] = 1.5
    assert np.allclose(_avg_rank(x), _rankdata(x))


def test_spearman_matrix_fast_matches_named_family_benchmark():
    rng = np.random.default_rng(1)
    a = rng.normal(size=400)
    b = 0.6 * a + 0.4 * rng.normal(size=400)
    c = rng.normal(size=400)
    c[:50] = 0.0  # ties in one column
    X = np.column_stack([a, b, c])
    ref = spearman_matrix({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
    fast = spearman_matrix_fast(X)
    names = ["a", "b", "c"]
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            assert fast[i, j] == pytest.approx(ref[ni][nj], abs=1e-9)


# --------------------------------------------------------------------------- #
# moving-block bootstrap index generator
# --------------------------------------------------------------------------- #


def test_moving_block_indices_length_range_and_contiguity():
    n, block_len = 20, 5
    idx = moving_block_indices(n, block_len, np.random.default_rng(3))
    assert idx.shape[0] == n
    assert idx.min() >= 0 and idx.max() < n
    # With n divisible by block_len there is no truncation: every block is contiguous.
    blocks = idx.reshape(n // block_len, block_len)
    assert np.all(np.diff(blocks, axis=1) == 1)
    # Every block start leaves room for a full block (non-circular).
    assert blocks[:, 0].max() <= n - block_len


def test_moving_block_indices_reproducible_and_validated():
    a = moving_block_indices(100, 10, np.random.default_rng(7))
    b = moving_block_indices(100, 10, np.random.default_rng(7))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        moving_block_indices(50, 0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        moving_block_indices(50, 51, np.random.default_rng(0))


# --------------------------------------------------------------------------- #
# paired contrast: sign of Delta is known a priori on synthetic series
# --------------------------------------------------------------------------- #


def _synthetic(rng: np.random.Generator, n: int, *, correlated: bool):
    """Two 'instantaneous' series + one 'temporal' series. When ``correlated`` the two
    instantaneous series share signal (within-corr high) while the temporal one is
    independent (cross-corr ~0) => Delta>0; otherwise all three are independent."""

    base = rng.normal(size=n)
    if correlated:
        a = 0.8 * base + 0.2 * rng.normal(size=n)
        b = 0.8 * base + 0.2 * rng.normal(size=n)
    else:
        a = rng.normal(size=n)
        b = rng.normal(size=n)
    c = rng.normal(size=n)  # independent 'temporal' family
    return np.column_stack([a, b, c])


def test_bootstrap_contrast_positive_delta_when_instantaneous_correlated():
    X = _synthetic(np.random.default_rng(11), 600, correlated=True)
    res = bootstrap_contrast(
        X, temporal_cols=[2], instantaneous_cols=[0, 1],
        block_len=50, n_boot=400, seed=42,
    )
    assert res["delta"]["point"] > 0.3
    lo, hi = res["delta"]["ci95"]
    assert lo > 0.0  # 95% interval excludes 0 => significant
    assert res["delta"]["frac_positive"] > 0.99
    assert res["n_boot_valid"] == 400


def test_bootstrap_contrast_delta_ci_straddles_zero_under_null():
    X = _synthetic(np.random.default_rng(12), 600, correlated=False)
    res = bootstrap_contrast(
        X, temporal_cols=[2], instantaneous_cols=[0, 1],
        block_len=50, n_boot=400, seed=42,
    )
    lo, hi = res["delta"]["ci95"]
    assert lo < 0.0 < hi  # no real gap => 95% interval contains 0 (not significant)
    assert abs(res["delta"]["point"]) < 0.15  # tiny effect (all series independent)
    assert 0.0 < res["delta"]["frac_positive"] < 1.0  # genuinely uncertain, not degenerate


def test_bootstrap_contrast_seed_reproducible():
    X = _synthetic(np.random.default_rng(13), 400, correlated=True)
    kw = dict(temporal_cols=[2], instantaneous_cols=[0, 1], block_len=40, n_boot=200, seed=99)
    r1 = bootstrap_contrast(X, **kw)
    r2 = bootstrap_contrast(X, **kw)
    assert r1["delta"]["ci95"] == r2["delta"]["ci95"]
    assert r1["delta"]["point"] == r2["delta"]["point"]


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _synthetic_ok_result() -> dict:
    contrast = {
        "block_len": 50,
        "n_boot_valid": 2000,
        "within_instantaneous": {"point": 0.477, "ci95": [0.45, 0.50]},
        "temporal_vs_instantaneous": {"point": 0.190, "ci95": [0.16, 0.22]},
        "delta": {"point": 0.287, "ci95": [0.25, 0.33], "frac_positive": 1.0},
    }
    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "n_scored_samples": 4106,
        "n_boot": 2000,
        "primary_block_len": 50,
        "series": ["CPDZ residual", "GEpFM drift", "RAT covariance",
                   "Mahalanobis", "PCA residual", "HydroJEV AE"],
        "temporal_families": ["GEpFM drift", "RAT covariance"],
        "instantaneous_refs": ["Mahalanobis", "PCA residual", "HydroJEV AE"],
        "contrast_by_block_len": [contrast],
        "pairwise_spearman_ci": {
            "GEpFM drift": {
                "Mahalanobis": {"rho": 0.17, "ci95": [0.10, 0.24]},
                "PCA residual": {"rho": 0.19, "ci95": [0.12, 0.26]},
                "HydroJEV AE": {"rho": 0.20, "ci95": [0.13, 0.27]},
            },
            "RAT covariance": {
                "Mahalanobis": {"rho": 0.22, "ci95": [0.15, 0.29]},
                "PCA residual": {"rho": 0.16, "ci95": [0.09, 0.23]},
                "HydroJEV AE": {"rho": 0.20, "ci95": [0.13, 0.27]},
            },
            "Mahalanobis": {
                "PCA residual": {"rho": 0.34, "ci95": [0.27, 0.41]},
                "HydroJEV AE": {"rho": 0.52, "ci95": [0.46, 0.58]},
            },
            "PCA residual": {"HydroJEV AE": {"rho": 0.57, "ci95": [0.51, 0.63]}},
        },
    }


def test_format_table_ok_and_not_run():
    text = format_significance_table(_synthetic_ok_result())
    assert "Orthogonality significance" in text
    assert "Delta" in text and "0.287" in text
    assert "GEpFM drift" in text and "Mahalanobis" in text
    assert "P(D>0)" in text
    assert format_significance_table({"status": "not_run", "reason": "x"}) == "not_run"


def test_plot_significance(tmp_path):
    out = tmp_path / "sig.png"
    path = plot_significance(_synthetic_ok_result(), str(out))
    assert path is not None and out.exists() and out.stat().st_size > 0
    assert plot_significance({"status": "not_run"}, str(tmp_path / "none.png")) is None


# --------------------------------------------------------------------------- #
# guarded real-data smoke test
# --------------------------------------------------------------------------- #


def test_orthogonality_significance_runner_if_available():
    pytest.importorskip("torch")
    from hydrojev.benchmarks.orthogonality_significance import run_orthogonality_significance
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_orthogonality_significance(
        config, mode=mode, seed=7, max_epochs=6, n_boot=120,
        block_lengths=(25, 50), primary_block_len=50,
    )
    if result.get("status") != "ok":
        pytest.skip(f"orthogonality significance not runnable: {result.get('reason')}")

    assert result["n_scored_samples"] > 100
    assert len(result["temporal_families"]) >= 1
    assert len(result["instantaneous_refs"]) >= 2
    # Every pairwise cell has a well-ordered interval bracketing its point estimate.
    for a, row in result["pairwise_spearman_ci"].items():
        for b, cell in row.items():
            lo, hi = cell["ci95"]
            assert lo <= cell["rho"] + 1e-6 and cell["rho"] - 1e-6 <= hi
            assert -1.0 <= lo <= hi <= 1.0
    # Each contrast reports an ordered CI and a probability in [0,1].
    for c in result["contrast_by_block_len"]:
        dlo, dhi = c["delta"]["ci95"]
        assert dlo <= c["delta"]["point"] <= dhi
        assert 0.0 <= c["delta"]["frac_positive"] <= 1.0

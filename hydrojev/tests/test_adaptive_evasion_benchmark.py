"""Tests for the constant-hold vs adaptive per-timestep evasion head-to-head.

The pure aggregation and presentation are pinned on hand-built results; a single
torch/BATADAL-guarded test covers the real-data runner and asserts the design
invariant that makes the comparison meaningful: warm-started from the constant-hold
optimum, the adaptive attacker can only match or beat it (never fewer joint evasions,
never re-firing a family the constant hold had already silenced).
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.adaptive_evasion_benchmark import (
    _aggregate,
    format_adaptive_comparison_table,
    plot_adaptive_comparison,
    run_adaptive_evasion_comparison,
)
from hydrojev.benchmarks.temporal_evasion import FAMILIES


# --------------------------------------------------------------------------- #
# pure aggregation
# --------------------------------------------------------------------------- #


def test_aggregate_rates_and_residual_alarm():
    fired = {"CPDZ residual": 8, "GEpFM drift": 5, "RAT covariance": 0}
    still = {"CPDZ residual": 2, "GEpFM drift": 5, "RAT covariance": 0}
    agg = _aggregate(evaded=3, fired=fired, still=still, n=10)
    assert agg["n_jointly_evaded"] == 3
    assert agg["joint_evasion_rate"] == pytest.approx(0.3)
    lo, hi = agg["joint_evasion_wilson95"]
    assert 0.0 <= lo <= 0.3 <= hi <= 1.0
    ra = agg["per_family_residual_alarm_rate"]
    assert ra["CPDZ residual"] == pytest.approx(0.25)
    assert ra["GEpFM drift"] == pytest.approx(1.0)  # never silenced
    assert np.isnan(ra["RAT covariance"])           # never fired -> nan, not 0


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _ok_result() -> dict:
    fired = {"CPDZ residual": 6, "GEpFM drift": 5, "RAT covariance": 4}
    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "max_changed": 4,
        "max_attack_len": 24,
        "constant_budget": 3000,
        "adaptive_budget": 8000,
        "n_targets": 7,
        "families": list(FAMILIES),
        "per_family_fired": fired,
        "constant": {
            "joint_evasion_rate": 2 / 7, "joint_evasion_wilson95": [0.08, 0.64],
            "n_jointly_evaded": 2,
            "per_family_still_firing_after_joint_attack": {
                "CPDZ residual": 1, "GEpFM drift": 5, "RAT covariance": 3},
            "per_family_residual_alarm_rate": {
                "CPDZ residual": 1 / 6, "GEpFM drift": 1.0, "RAT covariance": 0.75},
        },
        "adaptive": {
            "joint_evasion_rate": 3 / 7, "joint_evasion_wilson95": [0.16, 0.75],
            "n_jointly_evaded": 3,
            "per_family_still_firing_after_joint_attack": {
                "CPDZ residual": 0, "GEpFM drift": 5, "RAT covariance": 2},
            "per_family_residual_alarm_rate": {
                "CPDZ residual": 0.0, "GEpFM drift": 1.0, "RAT covariance": 0.5},
        },
        "adaptive_extra_evasions": {"count": 1, "run_starts": [2337]},
        "gepfm": {"fired": 5, "still_constant": 5, "still_adaptive": 5,
                  "silenced_constant": 0, "silenced_adaptive": 0},
        "mean_queries": {"constant": 900.0, "adaptive_refine": 4200.0, "adaptive_total": 5100.0},
    }


def test_format_table_ok_and_not_run():
    text = format_adaptive_comparison_table(_ok_result())
    assert "constant-hold" in text and "adaptive" in text
    assert "GEpFM drift" in text
    assert "0.286" in text  # constant 2/7
    assert "0.429" in text  # adaptive 3/7
    assert "[2337]" in text  # the extra-evasion window
    assert format_adaptive_comparison_table({"status": "not_run", "reason": "x"}) == "not_run"


def test_plot_ok_and_not_run(tmp_path):
    out = tmp_path / "adaptive.png"
    path = plot_adaptive_comparison(_ok_result(), str(out))
    assert path == str(out) and out.exists() and out.stat().st_size > 0
    assert plot_adaptive_comparison({"status": "not_run"}, str(tmp_path / "none.png")) is None


# --------------------------------------------------------------------------- #
# guarded real-data integration + the monotonicity invariant
# --------------------------------------------------------------------------- #


def test_run_adaptive_comparison_if_available():
    pytest.importorskip("torch")
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_adaptive_evasion_comparison(
        config, mode=mode, seed=7, max_epochs=6, max_targets=3,
        constant_budget=400, adaptive_budget=800,
    )
    if result.get("status") != "ok":
        assert result.get("status") == "not_run" and "reason" in result
        return

    n = result["n_targets"]
    assert n > 0
    c, a = result["constant"], result["adaptive"]
    # The design invariant: adaptive is warm-started from the constant hold, so it can
    # only match or beat it -- never fewer joint evasions, never re-firing a family the
    # constant hold silenced.
    assert a["n_jointly_evaded"] >= c["n_jointly_evaded"]
    for fam in FAMILIES:
        assert (
            a["per_family_still_firing_after_joint_attack"][fam]
            <= c["per_family_still_firing_after_joint_attack"][fam]
        )
    assert result["adaptive_extra_evasions"]["count"] == a["n_jointly_evaded"] - c["n_jointly_evaded"]
    # GEpFM accounting is internally consistent.
    g = result["gepfm"]
    assert g["silenced_constant"] == g["fired"] - g["still_constant"]
    assert g["silenced_adaptive"] == g["fired"] - g["still_adaptive"]
    assert len(result["targets"]) == n

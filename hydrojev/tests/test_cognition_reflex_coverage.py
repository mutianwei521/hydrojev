"""Tests for the cognition-vs-reflex coverage analysis.

The pure helpers (envelope, over-level detection, cognition OR, run first-alarm,
cross-tab) are exercised with hand-built arrays so the division-of-labour logic is
pinned exactly; a single torch/BATADAL-guarded test covers the real-data runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.cognition_reflex_coverage import (
    coverage_crosstab,
    cognition_alarm_series,
    first_true_in_runs,
    format_coverage_table,
    healthy_level_envelope,
    overlevel_ratio_series,
    physical_overlevel_series,
    plot_coverage,
    run_cognition_reflex_coverage,
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_healthy_level_envelope_is_column_max() -> None:
    clean = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]])
    env = healthy_level_envelope(clean)
    assert env.tolist() == [3.0, 5.0]


def test_healthy_level_envelope_validates() -> None:
    with pytest.raises(ValueError):
        healthy_level_envelope(np.array([1.0, 2.0, 3.0]))  # 1D
    with pytest.raises(ValueError):
        healthy_level_envelope(np.zeros((0, 2)))  # no samples
    with pytest.raises(ValueError):
        healthy_level_envelope(np.array([[0.0, -1.0], [0.0, -2.0]]))  # non-positive ceiling


def test_physical_overlevel_series_strict_and_margin() -> None:
    ceiling = np.array([4.0, 5.0])
    levels = np.array(
        [
            [3.9, 4.9],  # both under -> False
            [4.0, 4.0],  # exactly at ceiling (not strictly above) -> False
            [4.1, 1.0],  # tank 0 over -> True
            [1.0, 5.5],  # tank 1 over -> True
        ]
    )
    strict = physical_overlevel_series(levels, ceiling)
    assert strict.tolist() == [False, False, True, True]

    # A 5% margin lifts tank-0's band to 4.2 (drops the 4.1 row) but tank-1's band
    # is only 5.25, so the 5.5 row still fires.
    margined = physical_overlevel_series(levels, ceiling, margin=0.05)
    assert margined.tolist() == [False, False, False, True]

    # A 15% margin (bands 4.6 / 5.75) clears every row.
    wide = physical_overlevel_series(levels, ceiling, margin=0.15)
    assert wide.tolist() == [False, False, False, False]


def test_physical_overlevel_series_validates() -> None:
    with pytest.raises(ValueError):
        physical_overlevel_series(np.zeros(3), np.ones(1))  # 1D levels
    with pytest.raises(ValueError):
        physical_overlevel_series(np.zeros((3, 2)), np.ones(3))  # ceiling shape
    with pytest.raises(ValueError):
        physical_overlevel_series(np.zeros((3, 2)), np.ones(2), margin=-0.1)


def test_overlevel_ratio_series() -> None:
    ceiling = np.array([4.0, 5.0])
    levels = np.array([[2.0, 5.0], [4.4, 1.0]])
    ratio = overlevel_ratio_series(levels, ceiling)
    # row0: max(0.5, 1.0)=1.0 ; row1: max(1.1, 0.2)=1.1
    assert np.allclose(ratio, [1.0, 1.1])


def test_cognition_alarm_series_or_and_nan() -> None:
    scores = {
        "CPDZ residual": np.array([0.1, 0.9, np.nan, 0.2]),
        "GEpFM drift": np.array([0.0, 0.0, 0.0, 0.8]),
    }
    thr = {"CPDZ residual": 0.5, "GEpFM drift": 0.5}
    fired = cognition_alarm_series(scores, thr, ["CPDZ residual", "GEpFM drift"])
    # t0 none; t1 CPDZ; t2 CPDZ nan -> not alarming; t3 GEpFM
    assert fired.tolist() == [False, True, False, True]


def test_cognition_alarm_series_validates() -> None:
    with pytest.raises(ValueError):
        cognition_alarm_series({}, {}, [])  # empty families
    with pytest.raises(ValueError):
        cognition_alarm_series(
            {"a": np.zeros(3), "b": np.zeros(4)}, {"a": 1.0, "b": 1.0}, ["a", "b"]
        )  # length mismatch


def test_first_true_in_runs() -> None:
    flag = np.array([0, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    runs = [(0, 2), (2, 5), (5, 8)]
    # run0: no True -> None; run1: first True at abs 2; run2: first True at abs 7
    assert first_true_in_runs(flag, runs) == [None, 2, 7]


def test_coverage_crosstab_division_of_labor() -> None:
    # Four scenarios of length 5 each; construct the four coverage classes.
    # s0 cognition-only, s1 both (cog leads physical by 2), s2 physical-only, s3 neither.
    runs = [(0, 5), (5, 10), (10, 15), (15, 20)]
    cog = np.zeros(20, dtype=bool)
    phys = np.zeros(20, dtype=bool)
    cog[1] = True                       # s0 cognition-only
    cog[6] = True; phys[8] = True       # s1 both, lead = 8 - 6 = +2
    phys[12] = True                     # s2 physical-only
    # s3 nothing

    ct = coverage_crosstab(cog, phys, runs)
    agg = ct["aggregate"]
    assert agg["n_scenarios"] == 4
    assert agg["cognition_only"] == 1
    assert agg["both"] == 1
    assert agg["physical_only"] == 1
    assert agg["neither"] == 1
    assert agg["physical_abnormal_count"] == 2
    assert agg["cognition_recall"] == pytest.approx(0.5)
    assert agg["union_recall"] == pytest.approx(0.75)

    s1 = ct["scenarios"][1]
    assert s1["cognition_alarms"] and s1["physical_abnormal"]
    assert s1["cognition_first_step"] == 1  # abs 6 - start 5
    assert s1["physical_first_step"] == 3   # abs 8 - start 5
    assert s1["cognition_lead_steps"] == 2  # cognition alarms 2 steps before the breach

    s0 = ct["scenarios"][0]
    assert s0["cognition_alarms"] and not s0["physical_abnormal"]
    assert s0["cognition_lead_steps"] is None


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _ok_result() -> dict:
    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "level_margin": 0.0,
        "aggregate": {
            "n_scenarios": 3,
            "cognition_recall": 2 / 3,
            "physical_abnormal_count": 1,
            "cognition_only": 1,
            "physical_only": 0,
            "both": 1,
            "neither": 1,
            "union_recall": 2 / 3,
        },
        "scenarios": [
            {
                "abs_start": 100, "abs_end": 110, "length": 10,
                "cognition_alarms": True, "cognition_first_step": 3,
                "physical_abnormal": False, "physical_first_step": None,
                "cognition_lead_steps": None, "peak_overlevel_ratio": 0.82,
                "families_alarming": ["CPDZ residual"],
            },
            {
                "abs_start": 200, "abs_end": 215, "length": 15,
                "cognition_alarms": True, "cognition_first_step": 1,
                "physical_abnormal": True, "physical_first_step": 4,
                "cognition_lead_steps": 3, "peak_overlevel_ratio": 1.18,
                "families_alarming": ["CPDZ residual", "GEpFM drift"],
            },
            {
                "abs_start": 300, "abs_end": 305, "length": 5,
                "cognition_alarms": False, "cognition_first_step": None,
                "physical_abnormal": False, "physical_first_step": None,
                "cognition_lead_steps": None, "peak_overlevel_ratio": 0.60,
                "families_alarming": [],
            },
        ],
    }


def test_format_coverage_table_ok_and_not_run() -> None:
    text = format_coverage_table(_ok_result())
    assert "Cognition vs reflex coverage" in text
    assert "cognition-only" in text
    assert "1.18" in text  # the physically-abnormal scenario's peak ratio
    assert "+3" in text     # the lead on the 'both' scenario
    assert format_coverage_table({"status": "not_run", "reason": "x"}) == "not_run"


def test_plot_coverage_ok_and_not_run(tmp_path) -> None:
    out = tmp_path / "coverage.png"
    path = plot_coverage(_ok_result(), str(out))
    assert path == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert plot_coverage({"status": "not_run"}, str(out)) is None


# --------------------------------------------------------------------------- #
# Guarded real-data integration
# --------------------------------------------------------------------------- #


def test_run_cognition_reflex_coverage_guarded() -> None:
    pytest.importorskip("torch")
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_cognition_reflex_coverage(config, mode=mode, seed=7, max_epochs=6)

    if result.get("status") != "ok":
        assert result.get("status") == "not_run" and "reason" in result
        return

    agg = result["aggregate"]
    n = agg["n_scenarios"]
    assert n >= 1
    # The four coverage classes partition the scenarios exactly.
    assert agg["cognition_only"] + agg["both"] + agg["physical_only"] + agg["neither"] == n
    assert 0.0 <= agg["union_recall"] <= 1.0
    assert 0.0 <= agg["cognition_recall"] <= 1.0
    assert agg["physical_abnormal_count"] == agg["both"] + agg["physical_only"]
    assert len(result["scenarios"]) == n
    assert result["n_tanks"] == len(result["tank_columns"]) >= 1
    # Every scenario carries the enriched fields.
    for s in result["scenarios"]:
        assert set(s) >= {
            "abs_start", "abs_end", "peak_overlevel_ratio", "families_alarming",
            "cognition_alarms", "physical_abnormal", "cognition_lead_steps",
        }

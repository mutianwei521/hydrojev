"""Tests for the §7 reflex-mitigation robustness / sensitivity sweep.

The summarisation and safe-window logic are pure and pinned on synthetic
``run_hydraulic_mitigation``-shaped dicts (no EPANET), including the boundary cases that
matter for the paper's claim: a window that is fully bracketed, a window whose lower edge
falls below the swept range (flagged honestly), and a sweep where nothing qualifies. A
wntr/CTOWN-guarded smoke test exercises the real runner on a tiny sweep.
"""

from __future__ import annotations

import math

import pytest

from hydrojev.benchmarks.mitigation_sensitivity import (
    format_sensitivity_table,
    plot_sensitivity,
    safe_window,
    summarize_run,
)


# --------------------------------------------------------------------------- #
# summarize_run
# --------------------------------------------------------------------------- #


def _run_result(op_costs, nuisance_flags, *, mit=1.0, ave=1.0) -> dict:
    attacks = [
        {"reflex_operational_cost_fraction": c, "reflex_nuisance_trip": n}
        for c, n in zip(op_costs, nuisance_flags)
    ]
    return {
        "status": "ok",
        "attacks": attacks,
        "mean_energy_mitigated_fraction": mit,
        "mean_overflow_averted_fraction": ave,
        "mean_reflex_operational_cost_fraction": sum(op_costs) / len(op_costs),
        "any_reflex_nuisance_trip": any(nuisance_flags),
        "total_overflow_attacked_m3": 4558.0,
        "total_overflow_defended_m3": 0.0,
    }


def test_summarize_run_extracts_curve_point():
    res = _run_result([0.0, -0.03, 0.001, 0.0], [False, True, False, False], mit=0.99, ave=1.0)
    pt = summarize_run(res, param_key="reflex_fraction", param_value=0.94)
    assert pt["reflex_fraction"] == pytest.approx(0.94)
    assert pt["mean_energy_mitigated_fraction"] == pytest.approx(0.99)
    assert pt["n_nuisance_attacks"] == 1
    assert pt["any_reflex_nuisance_trip"] is True
    # worst single-attack |op cost| is the 0.03, not the ~0 mean
    assert pt["max_abs_op_cost_fraction"] == pytest.approx(0.03)


def test_summarize_run_handles_all_clean():
    res = _run_result([0.0, 0.0], [False, False])
    pt = summarize_run(res, param_key="reflex_fraction", param_value=0.98)
    assert pt["n_nuisance_attacks"] == 0
    assert pt["any_reflex_nuisance_trip"] is False
    assert pt["max_abs_op_cost_fraction"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# safe_window
# --------------------------------------------------------------------------- #


def _pt(frac, *, mit=1.0, ave=1.0, nuisance=False) -> dict:
    return {
        "reflex_fraction": frac,
        "mean_energy_mitigated_fraction": mit,
        "mean_overflow_averted_fraction": ave,
        "any_reflex_nuisance_trip": nuisance,
    }


def test_safe_window_brackets_the_band():
    # low fractions nuisance-trip; 0.96-0.99 are clean and fully mitigating.
    sweep = [
        _pt(0.90, mit=1.3, ave=1.0, nuisance=True),
        _pt(0.94, mit=1.1, ave=1.0, nuisance=True),
        _pt(0.96, mit=1.0, ave=1.0, nuisance=False),
        _pt(0.98, mit=1.0, ave=1.0, nuisance=False),
        _pt(0.99, mit=0.99, ave=1.0, nuisance=False),
    ]
    w = safe_window(sweep)
    assert w["min_safe_fraction"] == pytest.approx(0.96)
    assert w["max_safe_fraction"] == pytest.approx(0.99)
    assert w["n_safe_points"] == 3
    assert w["lower_boundary_below_sweep"] is False  # 0.90/0.94 tripped => boundary bracketed


def test_safe_window_flags_unbracketed_lower_boundary():
    # every swept fraction is safe -> the true lower edge is below the sweep.
    sweep = [_pt(0.96), _pt(0.97), _pt(0.98), _pt(0.99)]
    w = safe_window(sweep)
    assert w["min_safe_fraction"] == pytest.approx(0.96)
    assert w["lower_boundary_below_sweep"] is True


def test_safe_window_rejects_high_mitigation_with_nuisance():
    # a fraction that mitigates fully but nuisance-trips must NOT count as safe.
    sweep = [_pt(0.90, mit=1.5, ave=1.0, nuisance=True), _pt(0.98, mit=1.0, ave=1.0, nuisance=False)]
    w = safe_window(sweep)
    assert w["n_safe_points"] == 1
    assert w["min_safe_fraction"] == pytest.approx(0.98)


def test_safe_window_empty_when_none_qualify():
    sweep = [_pt(0.90, ave=0.5, nuisance=True), _pt(0.98, mit=0.5, ave=0.8, nuisance=False)]
    w = safe_window(sweep)
    assert math.isnan(w["min_safe_fraction"])
    assert math.isnan(w["max_safe_fraction"])
    assert w["n_safe_points"] == 0


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _ok_result() -> dict:
    fr = [
        summarize_run(_run_result([0.0, -0.05], [False, True], mit=1.2, ave=1.0),
                      param_key="reflex_fraction", param_value=0.90),
        summarize_run(_run_result([0.0, 0.0], [False, False], mit=1.0, ave=1.0),
                      param_key="reflex_fraction", param_value=0.98),
    ]
    hz = [
        summarize_run(_run_result([0.0, 0.0], [False, False], mit=1.0, ave=1.0),
                      param_key="horizon_h", param_value=24),
        summarize_run(_run_result([0.0, 0.0], [False, False], mit=1.0, ave=1.0),
                      param_key="horizon_h", param_value=72),
    ]
    return {
        "status": "ok",
        "network": "CTOWN",
        "n_attacks": 2,
        "efficiency": 0.75,
        "base_reflex_fraction": 0.98,
        "base_horizon_h": 72,
        "reflex_fraction_sweep": fr,
        "horizon_sweep": hz,
        "safe_window": safe_window(fr),
    }


def test_format_sensitivity_table_ok_and_not_run():
    text = format_sensitivity_table(_ok_result())
    assert "sensitivity" in text.lower()
    assert "safe design window" in text
    assert "Horizon sweep" in text
    assert format_sensitivity_table({"status": "not_run"}) == "not_run"


def test_plot_sensitivity_ok_and_not_run(tmp_path):
    out = tmp_path / "sens.png"
    path = plot_sensitivity(_ok_result(), str(out))
    assert path is not None and out.exists() and out.stat().st_size > 0
    assert plot_sensitivity({"status": "not_run"}, str(tmp_path / "none.png")) is None


# --------------------------------------------------------------------------- #
# guarded real-CTOWN smoke test
# --------------------------------------------------------------------------- #


def test_mitigation_sensitivity_runner_if_available():
    pytest.importorskip("wntr")
    from hydrojev.config import load_config
    from hydrojev.simulation.hydraulic_mitigation import OverpumpAttack
    from hydrojev.benchmarks.mitigation_sensitivity import run_mitigation_sensitivity

    config = load_config()
    result = run_mitigation_sensitivity(
        config,
        reflex_fractions=(0.96, 0.98),
        horizons_h=(24,),
        attacks=(OverpumpAttack("PU4", "T3", "control 8", "over-pump T3 (PU4)"),),
        base_horizon_h=24,
    )
    if result.get("status") != "ok":
        pytest.skip(f"CTOWN unavailable: {result.get('reason')}")

    assert len(result["reflex_fraction_sweep"]) == 2
    assert len(result["horizon_sweep"]) == 1
    for p in result["reflex_fraction_sweep"]:
        # each point has ordered, finite structure
        assert "mean_energy_mitigated_fraction" in p
        assert 0.0 <= p["max_abs_op_cost_fraction"] < 1.0 or math.isnan(p["max_abs_op_cost_fraction"])
    # at the 98% trigger the reflex should mitigate substantially and stay operationally
    # neutral (this tiny 1-attack/24h smoke config mitigates less than the 4-attack/72h
    # design point, so the bound is deliberately loose — the full sweep quantifies it).
    hi = next(p for p in result["reflex_fraction_sweep"] if abs(p["reflex_fraction"] - 0.98) < 1e-9)
    assert hi["mean_energy_mitigated_fraction"] > 0.5
    assert hi["mean_overflow_averted_fraction"] > 0.9
    assert hi["any_reflex_nuisance_trip"] is False
    assert "min_safe_fraction" in result["safe_window"]

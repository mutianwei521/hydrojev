"""Tests for the real-hydraulics consequence-mitigation module.

The pure consequence helpers (energy, overflow integration, fraction algebra,
suite aggregation, presentation) are exercised deterministically on synthetic
arrays -- no EPANET needed -- so the physics accounting is pinned down exactly.
A single wntr/CTOWN-guarded integration test then checks the end-to-end runner
on the real network, skipping cleanly when the asset or solver is absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hydrojev.simulation.hydraulic_mitigation import (
    OverpumpAttack,
    aggregate_attacks,
    averted_fraction,
    extra_fraction,
    format_mitigation_table,
    mitigated_fraction,
    overflow_volume_m3,
    plot_mitigation,
    pump_energy_kwh,
)
from hydrojev.state_projection.residual_engine import integrate_pump_energy


# --------------------------------------------------------------------------- #
# Energy
# --------------------------------------------------------------------------- #


def test_pump_energy_matches_integrator():
    t = np.arange(5) * 900.0
    q = np.full(5, 0.05)
    h = np.full(5, 40.0)
    assert pump_energy_kwh(t, q, h) == pytest.approx(integrate_pump_energy(t, q, h))


def test_pump_energy_clamps_reverse_flow_and_head():
    t = np.arange(3) * 900.0
    # Negative flow and negative head must not create negative energy.
    q = np.array([-0.05, -0.05, -0.05])
    h = np.array([-10.0, -10.0, -10.0])
    assert pump_energy_kwh(t, q, h) == pytest.approx(0.0)


def test_pump_energy_zero_when_idle():
    t = np.arange(4) * 900.0
    assert pump_energy_kwh(t, np.zeros(4), np.full(4, 30.0)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Overflow integration
# --------------------------------------------------------------------------- #


def test_overflow_only_counts_at_max_positive_inflow():
    t = np.arange(4) * 900.0
    level = np.array([5.0, 6.75, 6.75, 5.0])
    net = np.array([0.1, 0.2, 0.2, 0.1])
    vol, steps = overflow_volume_m3(net, level, 6.75, t)
    # Only the two at-max samples spill: (0.2 + 0.2) * 900 = 360.
    assert vol == pytest.approx(360.0)
    assert steps == 2


def test_overflow_ignores_draining_at_max():
    t = np.arange(3) * 900.0
    level = np.array([6.75, 6.75, 6.75])
    net = np.array([0.3, -0.5, 0.0])  # only first sample actually spills
    vol, steps = overflow_volume_m3(net, level, 6.75, t)
    assert vol == pytest.approx(0.3 * 900.0)
    assert steps == 1


def test_overflow_zero_when_never_full():
    t = np.arange(5) * 900.0
    level = np.full(5, 3.0)
    net = np.full(5, 0.5)
    vol, steps = overflow_volume_m3(net, level, 6.75, t)
    assert vol == pytest.approx(0.0)
    assert steps == 0


def test_overflow_uniform_grid_equals_rect_sum():
    # Reproduces the validated recipe: sum_{at max} max(0, net) * dt on a uniform grid.
    rng = np.random.default_rng(0)
    t = np.arange(50) * 900.0
    level = np.where(np.arange(50) >= 20, 6.75, 4.0)
    net = rng.uniform(-0.1, 0.4, size=50)
    vol, _ = overflow_volume_m3(net, level, 6.75, t)
    at_max = level >= 6.75 - 1e-3
    expected = float(np.clip(net[at_max], 0.0, None).sum() * 900.0)
    assert vol == pytest.approx(expected)


def test_overflow_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        overflow_volume_m3(np.zeros(3), np.zeros(4), 6.0, np.arange(3) * 900.0)


# --------------------------------------------------------------------------- #
# Fraction algebra
# --------------------------------------------------------------------------- #


def test_extra_fraction_basic_and_guards():
    assert extra_fraction(2039.0, 905.0) == pytest.approx((2039.0 - 905.0) / 905.0)
    assert math.isnan(extra_fraction(10.0, 0.0))
    assert math.isnan(extra_fraction(10.0, -1.0))


def test_mitigated_fraction_full_cancellation():
    # attack adds 125%, defence adds ~0% -> nearly all mitigated.
    assert mitigated_fraction(1.25, 0.0) == pytest.approx(1.0)
    # defence undercuts baseline -> >1.
    assert mitigated_fraction(1.0, -0.02) == pytest.approx(1.02)
    assert math.isnan(mitigated_fraction(0.0, 0.0))
    assert math.isnan(mitigated_fraction(-0.1, 0.0))


def test_averted_fraction():
    assert averted_fraction(4558.0, 0.0) == pytest.approx(1.0)
    assert averted_fraction(4558.0, 2279.0) == pytest.approx(0.5)
    assert averted_fraction(0.0, 0.0) != averted_fraction(0.0, 0.0) or math.isnan(
        averted_fraction(0.0, 0.0)
    )
    # negative (defence worse) clamps to 0.
    assert averted_fraction(100.0, 250.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Suite aggregation + presentation
# --------------------------------------------------------------------------- #


def _attack_dict(tank: str, pump: str, extra_att: float, extra_def: float,
                 ov_att: float, ov_def: float) -> dict:
    return {
        "pump": pump,
        "tank": tank,
        "defeated_control": "control X",
        "label": f"over-pump {tank} ({pump})",
        "baseline": {"overflow_volume_m3": 0.0, "total_energy_kwh": 900.0},
        "attacked": {"overflow_volume_m3": ov_att, "total_energy_kwh": 900.0 * (1 + extra_att)},
        "defended": {"overflow_volume_m3": ov_def, "total_energy_kwh": 900.0 * (1 + extra_def)},
        "baseline_reflex": {"overflow_volume_m3": 0.0, "total_energy_kwh": 900.0},
        "extra_energy_fraction_attacked": extra_att,
        "extra_energy_fraction_defended": extra_def,
        "energy_mitigated_fraction": mitigated_fraction(extra_att, extra_def),
        "overflow_averted_fraction": averted_fraction(ov_att, ov_def),
        "reflex_operational_cost_fraction": 0.0,
        "reflex_nuisance_trip": False,
    }


def test_aggregate_attacks_means():
    attacks = [
        _attack_dict("T3", "PU4", 1.25, -0.02, 4558.0, 0.0),
        _attack_dict("T4", "PU7", 0.60, 0.05, 1000.0, 0.0),
    ]
    agg = aggregate_attacks(attacks)
    assert agg["n_attacks"] == 2
    assert agg["mean_extra_energy_fraction_attacked"] == pytest.approx((1.25 + 0.60) / 2)
    assert agg["mean_overflow_averted_fraction"] == pytest.approx(1.0)
    assert agg["total_overflow_attacked_m3"] == pytest.approx(5558.0)
    assert agg["total_overflow_defended_m3"] == pytest.approx(0.0)
    assert agg["mean_reflex_operational_cost_fraction"] == pytest.approx(0.0)
    assert agg["any_reflex_nuisance_trip"] is False


def test_aggregate_skips_nan_metrics():
    attacks = [
        _attack_dict("T3", "PU4", 1.25, -0.02, 4558.0, 0.0),
        # a no-overflow attack contributes nan averted fraction; must be skipped in mean.
        _attack_dict("T2", "PU2", 0.40, 0.01, 0.0, 0.0),
    ]
    agg = aggregate_attacks(attacks)
    assert agg["mean_overflow_averted_fraction"] == pytest.approx(1.0)  # only T3 counts


def test_format_table_ok_and_not_run():
    result = {
        "status": "ok",
        "horizon_h": 72,
        "attacks": [_attack_dict("T3", "PU4", 1.25, -0.02, 4558.0, 0.0)],
        "mean_extra_energy_fraction_attacked": 1.25,
        "mean_extra_energy_fraction_defended": -0.02,
        "mean_energy_mitigated_fraction": 1.0,
        "mean_overflow_averted_fraction": 1.0,
        "mean_reflex_operational_cost_fraction": 0.0,
        "any_reflex_nuisance_trip": False,
        "total_overflow_attacked_m3": 4558.0,
        "total_overflow_defended_m3": 0.0,
    }
    table = format_mitigation_table(result)
    assert "T3" in table
    assert "overflow" in table
    assert "OPERATIONAL NEUTRALITY" in table
    assert format_mitigation_table({"status": "not_run"}) == "not_run"


def test_plot_ok_and_not_run(tmp_path):
    result = {
        "status": "ok",
        "attacks": [
            _attack_dict("T3", "PU4", 1.25, -0.02, 4558.0, 0.0),
            _attack_dict("T4", "PU7", 0.60, 0.05, 1000.0, 0.0),
        ],
    }
    out = plot_mitigation(result, str(tmp_path / "mit.png"))
    assert out is not None and (tmp_path / "mit.png").exists()
    assert plot_mitigation({"status": "not_run"}, "unused.png") is None


# --------------------------------------------------------------------------- #
# Integration on real CTOWN (guarded)
# --------------------------------------------------------------------------- #


def test_ctown_integration_if_available():
    pytest.importorskip("wntr")
    from hydrojev.config import load_config
    from hydrojev.simulation.hydraulic_mitigation import run_hydraulic_mitigation

    config = load_config()
    result = run_hydraulic_mitigation(
        config,
        attacks=(OverpumpAttack("PU4", "T3", "control 8", "over-pump T3 (PU4)"),),
        horizon_h=48,
    )
    if result.get("status") != "ok":
        pytest.skip(f"CTOWN unavailable: {result.get('reason')}")

    a = result["attacks"][0]
    # The undefended attack inflates energy and overflows the tank.
    assert a["extra_energy_fraction_attacked"] > 0.3
    assert a["attacked"]["overflow_volume_m3"] > 100.0
    # The local reflex removes essentially all of both consequences.
    assert a["extra_energy_fraction_defended"] < 0.10
    assert a["defended"]["overflow_volume_m3"] == pytest.approx(0.0, abs=1.0)
    assert a["overflow_averted_fraction"] > 0.98
    # Operational neutrality: the reflex installed under NORMAL operation must not
    # nuisance-trip -- near-zero energy cost, no spurious overflow.
    assert abs(a["reflex_operational_cost_fraction"]) < 0.01
    assert a["reflex_nuisance_trip"] is False
    assert a["baseline_reflex"]["overflow_volume_m3"] == pytest.approx(0.0, abs=1.0)

"""Tests for the cross-network reflex generalization study (§7.x)."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.hydraulic_generalization import (
    DEFAULT_NETWORKS,
    _overflow_fully_averted,
    cross_network_findings,
    format_generalization_table,
    plot_generalization,
    summarize_network,
)
from hydrojev.simulation.hydraulic_mitigation import NET3_ATTACKS


# --------------------------------------------------------------------------- #
# NET3 attack specification is faithful to the network's control structure
# --------------------------------------------------------------------------- #


def test_net3_attack_defeats_pump_close_and_bypass():
    # Net3 has exactly one level-interlock attack surface: PUMP 335 -> Tank 1.
    assert len(NET3_ATTACKS) == 1
    atk = NET3_ATTACKS[0]
    assert atk.pump == "335" and atk.tank == "1"
    # the pump high-level close ...
    assert atk.close_control == "control 16"
    # ... AND the pressure-zone relief bypass must both be defeated for an overflow.
    assert "control 18" in atk.extra_defeated_controls
    assert "330" in atk.force_closed_links


def test_default_networks_cover_ctown_and_net3():
    names = [n for n, _ in DEFAULT_NETWORKS]
    assert names == ["CTOWN", "Net3"]


# --------------------------------------------------------------------------- #
# summarize_network / _overflow_fully_averted (pure)
# --------------------------------------------------------------------------- #


def test_summarize_network_passes_through_not_run():
    s = summarize_network({"status": "not_run", "reason": "wntr unavailable"})
    assert s["status"] == "not_run"
    assert s["reason"] == "wntr unavailable"


def _fake_network_result(network, *, att, dfn, energy_att, energy_def, mitig, op_cost, nuisance):
    return {
        "status": "ok",
        "network": network,
        "n_attacks": 1,
        "timestep_s": 3600,
        "mean_extra_energy_fraction_attacked": energy_att,
        "mean_extra_energy_fraction_defended": energy_def,
        "mean_energy_mitigated_fraction": mitig,
        "mean_overflow_averted_fraction": 1.0 if dfn <= 1.0 and att > 0 else 0.0,
        "total_overflow_attacked_m3": att,
        "total_overflow_defended_m3": dfn,
        "mean_reflex_operational_cost_fraction": op_cost,
        "any_reflex_nuisance_trip": nuisance,
    }


def test_summarize_network_extracts_fields():
    s = summarize_network(
        _fake_network_result("Net3", att=33000.0, dfn=0.0, energy_att=3.03,
                             energy_def=1.95, mitig=0.36, op_cost=0.0, nuisance=False)
    )
    assert s["status"] == "ok"
    assert s["network"] == "Net3"
    assert s["total_overflow_attacked_m3"] == 33000.0
    assert s["mean_energy_mitigated_fraction"] == pytest.approx(0.36)


def test_overflow_fully_averted_logic():
    # attack spills, defence stops it -> averted
    assert _overflow_fully_averted({"total_overflow_attacked_m3": 33000.0, "total_overflow_defended_m3": 0.0})
    # defence still spills materially -> not averted
    assert not _overflow_fully_averted({"total_overflow_attacked_m3": 33000.0, "total_overflow_defended_m3": 500.0})
    # no attack overflow to begin with -> not a mitigation claim
    assert not _overflow_fully_averted({"total_overflow_attacked_m3": 0.0, "total_overflow_defended_m3": 0.0})


# --------------------------------------------------------------------------- #
# cross_network_findings (pure) — the honest headline logic
# --------------------------------------------------------------------------- #


def test_findings_overflow_averted_and_neutral_all_networks():
    summaries = [
        summarize_network(_fake_network_result("CTOWN", att=10702.0, dfn=0.0, energy_att=0.56,
                                              energy_def=0.0, mitig=1.0, op_cost=0.0, nuisance=False)),
        summarize_network(_fake_network_result("Net3", att=33000.0, dfn=0.0, energy_att=3.03,
                                              energy_def=1.95, mitig=0.36, op_cost=0.0, nuisance=False)),
    ]
    f = cross_network_findings(summaries)
    assert f["n_networks_ok"] == 2
    assert f["overflow_fully_averted_all_networks"] is True
    assert f["operationally_neutral_all_networks"] is True
    # energy recovery reported per-network, NOT averaged away (that difference is the finding)
    assert f["energy_mitigated_fraction_by_network"]["CTOWN"] == pytest.approx(1.0)
    assert f["energy_mitigated_fraction_by_network"]["Net3"] == pytest.approx(0.36)


def test_findings_flag_a_nuisance_trip_and_incomplete_averting():
    summaries = [
        summarize_network(_fake_network_result("CTOWN", att=10702.0, dfn=0.0, energy_att=0.56,
                                              energy_def=0.0, mitig=1.0, op_cost=0.0, nuisance=False)),
        # a network where the reflex nuisance-trips and does not fully avert
        summarize_network(_fake_network_result("BadNet", att=100.0, dfn=50.0, energy_att=1.0,
                                              energy_def=0.5, mitig=0.5, op_cost=0.05, nuisance=True)),
    ]
    f = cross_network_findings(summaries)
    assert f["overflow_fully_averted_all_networks"] is False
    assert f["operationally_neutral_all_networks"] is False


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _fake_ok_result():
    summaries = [
        summarize_network(_fake_network_result("CTOWN", att=10702.0, dfn=0.0, energy_att=0.56,
                                              energy_def=0.0, mitig=1.0, op_cost=0.0, nuisance=False)),
        summarize_network(_fake_network_result("Net3", att=33066.0, dfn=0.0, energy_att=3.03,
                                              energy_def=1.95, mitig=0.36, op_cost=0.0, nuisance=False)),
    ]
    return {
        "status": "ok",
        "horizon_h": 72,
        "reflex_safety_fraction": 0.98,
        "efficiency": 0.75,
        "summaries": summaries,
        "findings": cross_network_findings(summaries),
        "networks_detail": [],
    }


def test_format_table_ok_and_not_run():
    assert format_generalization_table({"status": "not_run"}) == "not_run"
    txt = format_generalization_table(_fake_ok_result())
    assert "Cross-network generalization" in txt
    assert "CTOWN" in txt and "Net3" in txt
    # the honest headline and the network-specific energy caveat are both present
    assert "overflow fully averted on ALL networks = YES" in txt
    assert "network-specific" in txt


def test_plot_ok_and_not_run(tmp_path):
    assert plot_generalization({"status": "not_run"}, str(tmp_path / "x.png")) is None
    out = plot_generalization(_fake_ok_result(), str(tmp_path / "gen.png"))
    assert out is not None
    assert (tmp_path / "gen.png").exists()


# --------------------------------------------------------------------------- #
# guarded end-to-end (wntr + Net3 asset required) — the real generalization
# --------------------------------------------------------------------------- #


def test_net3_generalization_if_available():
    pytest.importorskip("wntr")
    from hydrojev.benchmarks.hydraulic_generalization import run_generalization
    from hydrojev.config import load_config

    # Net3 only, to keep the guarded test fast (CTOWN is covered by §7's own tests).
    result = run_generalization(
        load_config(), networks=(("Net3", NET3_ATTACKS),), horizon_h=48,
    )
    if result.get("status") != "ok":
        pytest.skip(f"Net3/wntr unavailable: {result.get('reason')}")

    s = result["summaries"][0]
    assert s["status"] == "ok" and s["network"] == "Net3"
    # the attack must actually cause an overflow, and the reflex must fully avert it
    assert s["total_overflow_attacked_m3"] > 0.0
    assert s["total_overflow_defended_m3"] <= 1.0
    assert s["mean_overflow_averted_fraction"] == pytest.approx(1.0, abs=1e-6)
    # the paradigm's core claim generalizes: catastrophe averted, no nuisance trips
    f = result["findings"]
    assert f["overflow_fully_averted_all_networks"] is True
    assert f["operationally_neutral_all_networks"] is True
    # operational neutrality: the reflex costs ~nothing under normal operation
    assert abs(s["mean_reflex_operational_cost_fraction"]) <= 0.01

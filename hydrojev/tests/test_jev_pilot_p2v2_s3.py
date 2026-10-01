"""Offline tests for state format 3 (twin-free physics cross-checks) of the P2 v2 pilot."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v

H = 30  # hours of synthetic history; window end at hour 29


def _toy():
    """One pump PU from a reservoir (head 0) through node D to tank T (elevation 60, area 36 m2)."""

    spec = SimpleNamespace(pumps=["PU"], pump_ends={"PU": ("R", "D")}, reservoir_head={"R": 0.0}, elevation={"D": 0.0},
                           tank_area={"T": 36.0}, level_rules=[{"link": "PU", "tank": "T", "open_below": 1.0, "close_above": 3.0}])
    pcal = {"curve": {"PU": [90.0, 0.0, 1.0]}, "curve_median": {"PU": 0.0}, "curve_scale": {"PU": 1.0},
            "stations": {"T": ["PU"]}, "tank_elevation": {"T": 60.0},
            "thresholds": {"pump_operating_point_vs_curve": 2.0, "stopped_station_discharge_vs_tank_head": 0.5,
                           "control_logic_vs_metered_level": 1.0},
            "event_free_violation_rate": {c: 0.025 for c in v.PHYSICS_CHECKS}}
    # running pump on its curve (discharge head 90 m), tank filling 10 L/s = 1 m/h over 36 m2, honestly reported
    rep = {"S_PU": np.ones(H), "F_PU": np.full(H, 10.0), "P_D": np.full(H, 90.0), "Fin_T": np.full(H, 10.0),
           "L_T": np.clip(np.arange(H) - float(H - 1 - p2.SYNC_LAG_H), 0.0, None)}
    return spec, pcal, rep


def _stat(spec, rep, pcal, check):
    return v.physics_statistics(spec, rep, H - 1, pcal)[check][0]


def test_metered_level_integrates_the_inflow_meter() -> None:
    spec, _, rep = _toy()
    lm = v._metered_levels(spec, rep, "T", H - 1)
    assert len(lm) == p2.WINDOW_H and np.allclose(np.diff(lm), 1.0)


def test_running_pump_off_its_curve_is_flagged() -> None:
    spec, pcal, rep = _toy()
    assert _stat(spec, rep, pcal, "pump_operating_point_vs_curve") == pytest.approx(0.0)
    rep["P_D"] = np.full(H, 70.0)  # reported running at 10 L/s but lifting 20 m less than its curve
    s = v.physics_statistics(spec, rep, H - 1, pcal)["pump_operating_point_vs_curve"]
    assert s[0] == pytest.approx(20.0) and s[1]["curve_head_at_reported_flow_m"] - s[1]["measured_head_gain_m"] == pytest.approx(20.0)
    rep["P_D"] = np.full(H, -1e7)  # WNTR head at a disconnected node: statistic stays bounded
    assert _stat(spec, rep, pcal, "pump_operating_point_vs_curve") == v.STAT_CLIP


def test_stopped_station_cannot_lift_above_its_tank() -> None:
    spec, pcal, rep = _toy()
    rep.update({"S_PU": np.zeros(H), "F_PU": np.zeros(H), "L_T": np.full(H, 2.0), "Fin_T": np.zeros(H)})
    rep["P_D"] = np.full(H, 61.0)  # below the tank head 62 m: consistent
    assert _stat(spec, rep, pcal, "stopped_station_discharge_vs_tank_head") == 0.0
    rep["P_D"] = np.full(H, 75.0)  # 13 m above the tank head with every pump of the station stopped
    assert _stat(spec, rep, pcal, "stopped_station_discharge_vs_tank_head") == pytest.approx(13.0)


def test_metered_level_check_needs_the_reported_level_to_hide_the_overfill() -> None:
    spec, pcal, rep = _toy()
    # honest overfill: metered and reported agree (up to 11 m), pump runs above its setpoint but nothing is hidden
    assert _stat(spec, rep, pcal, "control_logic_vs_metered_level") == pytest.approx(0.0)
    rep["L_T"] = np.zeros(H)  # reported level frozen at the sync value 0
    s = _stat(spec, rep, pcal, "control_logic_vs_metered_level")
    assert s == pytest.approx(p2.SYNC_LAG_H - 3.0)  # metered 11 m at the window end, 8 m above the 3 m setpoint


def test_r1b_overrides_r1_only_on_physics_evidence() -> None:
    def state(**viol):
        c = {k: {"status": "violated" if k in viol else "consistent"} for k in v.PHYSICS_CHECKS}
        if "pump_operating_point_vs_curve" in viol:
            c["pump_operating_point_vs_curve"]["detail"] = viol["pump_operating_point_vs_curve"]
        return {"consistency_checks": c}

    assert v.r1b_classify("sensor_fault", state()) == "sensor_fault"
    assert v.r1b_classify("physical_fault", state(stopped_station_discharge_vs_tank_head=1)) == "cyber_attack"
    assert v.r1b_classify("normal_transient", state(control_logic_vs_metered_level=1)) == "cyber_attack"
    small = {"curve_head_at_reported_flow_m": 60.0, "measured_head_gain_m": 55.0}
    big = {"curve_head_at_reported_flow_m": 60.0, "measured_head_gain_m": 40.0}
    assert v.r1b_classify("sensor_fault", state(pump_operating_point_vs_curve=small)) == "sensor_fault"
    assert v.r1b_classify("sensor_fault", state(pump_operating_point_vs_curve=big)) == "cyber_attack"


def test_physics_meanings_are_leak_free() -> None:
    assert v.scan_state({"m": v.PHYSICS_MEANING}) == []
    assert set(v.PHYSICS_MEANING) == set(v.PHYSICS_CHECKS)


def test_format3_states_on_disk_are_leak_free_and_extend_format2() -> None:
    d3 = v.OUTDIR / "requests" / "train_s3"
    if not d3.exists():
        pytest.skip("format-3 train split not prepared")
    f = sorted(d3.glob("*.state.json"))[0]
    s3 = json.loads(f.read_text(encoding="utf-8"))
    s2 = json.loads((v.OUTDIR / "requests" / "train_s2" / f.name).read_text(encoding="utf-8"))
    assert v.scan_state(s3) == []
    assert set(s3["consistency_checks"]) == set(s2["consistency_checks"]) | set(v.PHYSICS_CHECKS)
    for c in s2["consistency_checks"]:
        assert s3["consistency_checks"][c] == s2["consistency_checks"][c]


def test_round3_seed_ladders_never_collide_with_other_splits() -> None:
    def ladder(split: str) -> set[int]:
        _, per_class, base, _ = v.SPLITS[split]
        return {base + 1000 * ci + i + 100_000 * k for ci in range(len(v.CLASSES)) for i in range(per_class) for k in range(41)}

    others = set().union(*(ladder(s) for s in v.SPLITS if not s.endswith("_r3")))
    others |= {2_900_000 + i for i in range(v.N_BALANCE_CAL)} | {3_100_000 + i for i in range(v.N_PHYSICS_CAL)}
    a, b = ladder("eval_id_r3"), ladder("eval_novel3_r3")
    assert not (a & others) and not (b & others) and not (a & b)


def test_round3_sealed_splits_need_their_own_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v, "OUTDIR", tmp_path)
    (tmp_path / "design_freeze_r2.json").write_text("{}", encoding="utf-8")
    for split in ("eval_id_r3", "eval_novel3_r3"):
        with pytest.raises(RuntimeError):
            v.prepare_split(None, split, v.STATE_FORMAT_PHYSICS)  # type: ignore[arg-type]


def test_novel3_family_is_disjoint_and_sampled() -> None:
    old = {s for fam in (v.NOVEL,) for x in fam.values() for s in x} | {
        "overpump_masked_level", "pump_off_spoofed_running", "burst", "demand_rise", "demand_fall", "operator_switch", "stuck", "drift", "offset"}
    new = {s for x in v.NOVEL3.values() for s in x}
    assert not new & old and set(v.NOVEL3) == set(v.CLASSES)
    spec = SimpleNamespace(name="toy", level_rules=[{"link": "PU", "tank": "T", "close_above": 3.0}], pumps=["PU"])
    for cls in v.CLASSES:
        sc = v.sample_scenario(spec, cls, 42, "novel3")
        assert sc["subtype"] in v.NOVEL3[cls] and sc["onset_h"] < sc["window_end_h"]


def test_override_jev_only_overrides_on_physics_evidence() -> None:
    def state(viol: bool):
        c = {k: {"status": "consistent"} for k in v.PHYSICS_CHECKS}
        if viol:
            c["stopped_station_discharge_vs_tank_head"]["status"] = "violated"
        return {"consistency_checks": c}

    pj = np.asarray([[0.1, 0.7, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
    assert v.override_jev(pj, None, [state(False), state(True)]) == ["physical_fault", "cyber_attack"]

"""Offline tests for the Jev pilot P2 contract (no network calls, no full preparation)."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.methods.hydrojev_detector import HydroJEVSchemaError


def _ev(violated, n_anom=3, pattern="none"):
    return {"violated": list(violated), "anomalous_channels": [f"P_{i}" for i in range(n_anom)], "pressure_spread": {"pattern": pattern}}


def test_question_is_single_five_way_choice() -> None:
    q = p2.build_questions()
    assert list(q) == ["event_type"]
    assert set(q["event_type"]["criteria"]) == set(p2.EVENT_TYPES)


def test_parse_accepts_rounded_distribution_and_rejects_bad_choice() -> None:
    body = p2.mock_response({"r1": "sensor_fault"})
    body["answers"]["event_type"]["probabilities"] = {"cyber_attack": 0.1, "physical_fault": 0.1, "normal_transient": 0.1, "sensor_fault": 0.69, "insufficient_evidence": 0.0}
    d = p2.parse_response(body)
    assert d.event_type == "sensor_fault" and sum(d.probabilities.values()) == pytest.approx(1.0)
    body["answers"]["event_type"]["choice"] = "burst"
    with pytest.raises(HydroJEVSchemaError):
        p2.parse_response(body)


def test_state_scan_ignores_fixed_system_text_but_flags_labels() -> None:
    assert p2.scan_state({"system": p2.SYSTEM_DESCRIPTION, "x": 1}) == []
    assert "burst" in p2.scan_state({"system": "", "note": "burst at J1"})
    with pytest.raises(HydroJEVSchemaError):
        p2.build_request({"label": "cyber_attack"})


def test_r1_decision_tree_branches() -> None:
    assert p2.r1_classify(_ev(["system_demand_vs_twin", "pressure_vs_twin"], 20, "widespread")) == "normal_transient"
    assert p2.r1_classify(_ev(["pump_head_gain_vs_twin", "tank_level_vs_twin"])) == "cyber_attack"
    assert p2.r1_classify(_ev(["tank_mass_balance", "control_logic_vs_twin_level"])) == "cyber_attack"
    assert p2.r1_classify(_ev(["tank_mass_balance"])) == "sensor_fault"
    assert p2.r1_classify(_ev(["pump_flow_vs_twin"], 1)) == "sensor_fault"
    assert p2.r1_classify(_ev(["system_demand_vs_twin"], 4, "localized")) == "physical_fault"
    assert p2.r1_classify(_ev(["control_logic_vs_reported_level"], 0)) == "normal_transient"


def test_metrics_count_abstention_as_wrong() -> None:
    m = p2.cause_metrics(["cyber_attack", "sensor_fault"], ["cyber_attack", "insufficient_evidence"])
    assert m["accuracy"] == 0.5 and m["abstention_rate"] == 0.5
    assert m["per_class_recall"]["sensor_fault"] == 0.0


def test_debug_selection_is_stratified_and_deterministic() -> None:
    rows = [{"id": f"train_{i:04d}", "split": "train", "label_internal": p2.CLASSES[i % 4]} for i in range(40)]
    a = p2.select_debug({"rows": rows})
    assert a == p2.select_debug({"rows": rows}) and len(a) == 12
    labels = [rows[int(i[-4:])]["label_internal"] for i in a]
    assert all(labels.count(c) == 3 for c in p2.CLASSES)


def test_window_is_six_hours_ending_at_window_end() -> None:
    w = p2._win(30)
    assert list(range(40))[w] == [25, 26, 27, 28, 29, 30]


def test_sampling_is_deterministic_and_onset_precedes_window_end() -> None:
    class S:
        name = "X"
        level_rules = [{"link": "P1", "tank": "T1", "close_above": 4.0, "open_below": 1.0}]
        pumps = ["P1"]

    for cls in p2.CLASSES:
        a, b = p2.sample_scenario(S, cls, 11, None), p2.sample_scenario(S, cls, 11, None)
        assert a == b
        assert a["window_end_h"] - a["onset_h"] in range(p2.LAG_RANGE_H[0], p2.LAG_RANGE_H[1] + 1)
        assert a["window_end_h"] - p2.SYNC_LAG_H < a["onset_h"]  # twin always synchronised before onset

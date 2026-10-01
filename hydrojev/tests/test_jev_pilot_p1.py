"""Offline tests for the Jev pilot P1 contract (no network, no BATADAL fit)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.methods.hydrojev_detector import HydroJEVSchemaError


def _state() -> dict:
    return {
        "schema_version": p1.SCHEMA_VERSION,
        "detector_pattern": {"alerting": ["Mahalanobis"], "alerting_count": 1},
        "physical_consistency": {"violated_checks": []},
        "provenance": {"labels_removed": True},
    }


def _response(**overrides) -> dict:
    body = p1.mock_response({"state": _state()})
    body["model"] = "jev-1.13.0"
    body["answers"]["event_type"]["probabilities"] = {
        "cyber_attack": 0.07, "physical_fault": 0.05, "normal_transient": 0.81, "sensor_fault": 0.05, "insufficient_evidence": 0.01,
    }
    body["answers"]["event_type"]["choice"] = "normal_transient"
    body.update(overrides)
    return body


def test_questions_are_self_contained_and_five_way() -> None:
    q = p1.build_questions_v2()
    assert set(q) == set(p1.QUESTION_IDS)
    assert set(q["event_type"]["criteria"]) == set(p1.EVENT_TYPES_V2)
    assert "sensor_fault" in q["event_type"]["criteria"]
    assert len(q["severity"]["criteria"]) == 5


def test_parse_accepts_two_decimal_rounding_and_renormalizes() -> None:
    d = p1.parse_response_v2(_response())  # event probabilities sum to 0.99
    assert d.event_type == "normal_transient"
    assert sum(d.event_probabilities.values()) == pytest.approx(1.0)
    assert d.model == "jev-1.13.0"


def test_parse_rejects_old_four_way_taxonomy() -> None:
    body = _response()
    body["answers"]["event_type"]["probabilities"].pop("sensor_fault")
    body["answers"]["event_type"]["probabilities"]["normal_transient"] = 0.86
    with pytest.raises(HydroJEVSchemaError):
        p1.parse_response_v2(body)


def test_request_rejects_label_keys_and_scan_flags_dates() -> None:
    bad = {**_state(), "att_flag": 1}
    with pytest.raises(HydroJEVSchemaError):
        p1.build_request_v2(bad)
    body = p1.canonical_bytes(p1.build_request_v2(_state()))
    assert p1.scan_request(body) == []
    assert "2016-" in p1.scan_request(body + b" 2016-07-04")


def test_window_max_and_run_length_are_causal() -> None:
    x = np.array([0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w = p1._window_max(x, width=3)
    assert w.tolist() == [0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0]
    runs = p1._consecutive_run_ending(np.array([[1], [1], [0], [1]], dtype=bool))
    assert runs[:, 0].tolist() == [1, 2, 0, 1]


def test_resolution_floor_matches_reporting_precision() -> None:
    fine = np.linspace(2.97, 2.99, 500)[:, None]
    assert p1._resolution(fine)[0] == pytest.approx(p1.EVALUATED_RESOLUTION)


def test_hold_to_hourly_forward_holds_grid_decisions() -> None:
    held = p1.hold_to_hourly(np.array([10, 16]), np.array([True, False]), start=8, n=12)
    assert held.tolist() == [False, False, True, True, True, True, True, True, False, False, False, False]


def test_select_debug_is_stratified_and_deterministic() -> None:
    grid = [{"raw_index": i, "candidate": True, "label_internal": int(i % 3 == 0)} for i in range(90)]
    prepared = {"variants": {"dataset04": {"grid": grid}}}
    a = p1.select_debug(prepared, 10)
    assert a == p1.select_debug(prepared, 10)
    labels = [grid[i]["label_internal"] for i in a]
    assert sum(labels) == 5 and len(a) == 10


def test_call_live_never_resends_and_respects_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = tmp_path / "r.request.json"
    req.write_bytes(p1.canonical_bytes(p1.build_request_v2(_state())))
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = Path(cmd[cmd.index("-OutputPath") + 1])
        out.write_text(json.dumps({"elapsed_seconds": 1.0, "attempts": 1, "response": _response()}), encoding="utf-8")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(p1.subprocess, "run", fake_run)
    monkeypatch.setattr(p1, "_key_env", lambda: {})
    first = p1.call_live(tmp_path, req, stage="t", tag="a")
    again = p1.call_live(tmp_path, req, stage="t", tag="a")
    assert first["status"] == "ok" and again["reused"] and len(calls) == 1
    monkeypatch.setattr(p1, "PILOT_CALL_CAP", 1)
    with pytest.raises(RuntimeError):
        p1.call_live(tmp_path, req, stage="t", tag="b")
    assert "TYPESAFE" not in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")


def test_call_live_records_timeout_without_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = tmp_path / "r.request.json"
    req.write_bytes(p1.canonical_bytes(p1.build_request_v2(_state())))

    def slow(cmd, **kwargs):
        raise p1.subprocess.TimeoutExpired(cmd, 240)

    monkeypatch.setattr(p1.subprocess, "run", slow)
    monkeypatch.setattr(p1, "_key_env", lambda: {})
    assert p1.call_live(tmp_path, req, stage="t", tag="x")["status"] == "timeout"
    assert p1.call_live(tmp_path, req, stage="t", tag="x")["reused"]


def test_round1_questions_are_unchanged_and_round2_only_replaces_alarm() -> None:
    r1, r2 = p1.build_questions_v2(), p1.build_questions_v2(2)
    assert r1 == p1.build_questions_v2(1)
    assert r1["event_type"] == r2["event_type"] and r1["severity"] == r2["severity"]
    assert r1["alarm"] != r2["alarm"] and "benign_flagged_window_violation_rate" in r2["alarm"]["instructions"]


def test_benign_rates_use_only_benign_candidates() -> None:
    grid = [
        {"candidate": True, "label_internal": 0, "violated_checks": ["junction_pressure_vs_operation"]},
        {"candidate": True, "label_internal": 0, "violated_checks": []},
        {"candidate": True, "label_internal": 1, "violated_checks": list(p1.PHYSICAL_CHECKS)},
        {"candidate": False, "label_internal": 0, "violated_checks": ["pump_status_vs_flow"]},
    ]
    rates = p1.benign_candidate_rates({"variants": {"dataset04": {"grid": grid}}})
    assert rates["junction_pressure_vs_operation"] == 0.5 and rates["pump_status_vs_flow"] == 0.0 and rates["n_windows"] == 2
    state = {"physical_consistency": {name: {"status": "consistent"} for name in p1.PHYSICAL_CHECKS}}
    out = p1.add_benign_rates(state, rates)
    assert out["physical_consistency"]["junction_pressure_vs_operation"]["benign_flagged_window_violation_rate"] == 0.5
    assert "benign_flagged_window_violation_rate" not in state["physical_consistency"]["pump_status_vs_flow"]

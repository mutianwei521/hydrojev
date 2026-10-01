"""Offline CLI tests; these tests must never call the TypeSafe API."""

from __future__ import annotations

import json

import pytest

from hydrojev.methods.detect import main
from hydrojev.methods.hydrojev_detector import build_detection_request, build_detection_state


def _request():
    state = build_detection_state(
        timestamp="current_causal_window", freshness={"is_fresh": True},
        detectors={"a": {"state": "alerting"}, "b": {"state": "alerting"}},
        routing={"candidate": True},
    )
    return build_detection_request(state)


def _response():
    return {"timestamp_utc": "2026-09-24T00:00:00Z", "elapsed_seconds": 1.0, "attempts": 1,
        "response": {"model": "fixture-jev", "usage": {"input_tokens": 10, "output_tokens": 5},
        "answers": {
            "alarm": {"type": "noul", "noul": 0.65},
            "event_type": {"type": "choice", "choice": "insufficient_evidence", "confidence": 0.7,
                           "probabilities": {"normal_transient": 0.1, "cyber_attack": 0.1, "physical_fault": 0.1, "insufficient_evidence": 0.7}},
            "severity": {"type": "score", "score": 2.0, "confidence": 1.0,
                         "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0},
                         "legend": {str(i): f"level {i}" for i in range(5)}},
        }}}


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_replay_is_labelled_replay_and_uses_benchmark_thresholds(tmp_path, monkeypatch):
    request = tmp_path / "request.json"; response = tmp_path / "response.json"; output = tmp_path / "result.json"
    _write(request, _request()); _write(response, _response())
    def no_network(*args, **kwargs):
        raise AssertionError("Offline replay tried to invoke a process")
    monkeypatch.setattr("hydrojev.methods.detect.subprocess.run", no_network)
    assert main(["--request", str(request), "--mode", "replay", "--response", str(response), "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["result"]["source"] == "replay"
    assert result["result"]["state"] == "alarm"
    assert result["provenance"]["execution_source"] == "replay"
    assert result["provenance"]["replayed"] is True
    assert result["provenance"]["live_api_invoked"] is False
    assert result["method"]["alarm_threshold"] == 0.5
    assert result["method"]["normal_threshold"] == 0.2


def test_no_jev_is_fully_offline_and_has_no_model(tmp_path, monkeypatch):
    request = tmp_path / "request.json"; output = tmp_path / "result.json"
    _write(request, _request())
    monkeypatch.setattr("hydrojev.methods.detect.subprocess.run", lambda *a, **k: pytest.fail("no-Jev invoked a process"))
    assert main(["--request", str(request), "--mode", "no-jev", "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["result"]["source"] == "no_jev"
    assert result["result"]["model"] is None
    assert result["provenance"]["live_api_invoked"] is False


def test_cli_rejects_label_leakage_before_output(tmp_path):
    request = tmp_path / "request.json"; output = tmp_path / "result.json"
    body = _request(); body["state"]["provenance"]["ATT_FLAG"] = 1; _write(request, body)
    assert main(["--request", str(request), "--mode", "no-jev", "--output", str(output)]) == 2
    assert not output.exists()


def test_replay_rejects_stale_question_contract(tmp_path):
    request = tmp_path / "request.json"; response = tmp_path / "response.json"; output = tmp_path / "result.json"
    body = _request(); body["questions"]["alarm"]["instructions"] = "A different scientific question"
    _write(request, body); _write(response, _response())
    assert main(["--request", str(request), "--mode", "replay", "--response", str(response), "--output", str(output)]) == 2
    assert not output.exists()


def test_replay_requires_response(tmp_path):
    request = tmp_path / "request.json"; output = tmp_path / "result.json"; _write(request, _request())
    assert main(["--request", str(request), "--mode", "replay", "--output", str(output)]) == 2
    assert not output.exists()

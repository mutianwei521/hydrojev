"""Offline tests for the Ollama LLM baselines (no network calls)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import llm_baselines as lb
from hydrojev.benchmarks import llm_baselines_eval as le


def test_parse_normalises_and_rounds() -> None:
    a = lb.parse('noise {"choice": "cyber_attack", "probabilities": {"cyber_attack": 2, "physical_fault": 1, "normal_transient": 1}} tail')
    assert a["choice"] == "cyber_attack" and set(a["probabilities"]) == set(v.EVENT_TYPES)
    assert a["probabilities"]["cyber_attack"] == 0.5 and a["probabilities"]["sensor_fault"] == 0.0


def test_parse_falls_back_to_argmax_choice_and_rejects_empty() -> None:
    assert lb.parse('{"choice": "bogus", "probabilities": {"sensor_fault": 0.9, "cyber_attack": 0.1}}')["choice"] == "sensor_fault"
    for bad in ("no json here", '{"choice": "cyber_attack", "probabilities": {}}'):
        with pytest.raises(ValueError):
            lb.parse(bad)


def test_user_message_is_the_jev_request_content() -> None:
    state = json.loads(next((v.OUTDIR / "requests" / "train_s2").glob("*.state.json")).read_text(encoding="utf-8"))
    msg = json.loads(lb.user_message(state))
    req = v.build_request(state, revision=3)
    assert msg["evidence"] == req["state"]
    assert msg["question"]["instructions"] == req["questions"]["event_type"]["instructions"]
    assert msg["question"]["criteria"] == req["questions"]["event_type"]["criteria"]
    assert v.scan_state(msg["evidence"]) == []


def test_schema_covers_every_event_type() -> None:
    s = lb.schema()
    assert s["properties"]["choice"]["enum"] == list(v.EVENT_TYPES)
    assert s["properties"]["probabilities"]["required"] == list(v.EVENT_TYPES)


def test_model_lists() -> None:
    assert len(lb.MODELS) == 15 and set(lb.RETIRED) <= set(lb.MODELS)
    assert len(lb.AVAILABLE) == 9 and not set(lb.AVAILABLE) & set(lb.RETIRED)
    assert set(lb.RETIRED_MIDRUN) <= set(lb.AVAILABLE) and len(lb.COMPLETING) == 7
    assert not set(lb.ADDED) & set(lb.MODELS) and set(lb.ADDED) - set(lb.DROPPED) <= set(lb.COMPLETING)
    assert not set(lb.DROPPED) & set(lb.COMPLETING)


def test_sealed_query_refused_without_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(lb, "OUTDIR", tmp_path)
    for split in lb.SPLITS:
        with pytest.raises(RuntimeError):
            lb.ask("gemma4:cloud", split, "x")


def test_invalid_window_is_scored_as_abstention() -> None:
    pm = np.asarray([[0.7, 0.1, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0]])
    assert le.decide(pm, np.asarray([True, False]), None) == ["cyber_attack", "insufficient_evidence"]


def test_two_sided_bootstrap_is_symmetric() -> None:
    y = np.repeat(np.arange(4), 10)
    a, b = y.copy(), np.roll(y, 5)
    ab = le.two_sided(y, a, b, np.random.default_rng(0))
    ba = le.two_sided(y, b, a, np.random.default_rng(0))
    assert ab["delta"] == -ba["delta"] > 0 and ab["p_two_sided"] == ba["p_two_sided"] < 0.05

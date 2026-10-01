"""Pure tests for the standalone HydroJEV detection contract."""

from __future__ import annotations

import copy

import pytest

from hydrojev.methods.hydrojev_detector import (
    DETECTION_QUESTION_IDS,
    EVENT_TYPES,
    DetectionEpisode,
    DetectionState,
    HydroJEVSchemaError,
    HydroJEVDetector,
    build_detection_questions,
    build_detection_request,
    build_detection_state,
    deterministic_no_jev,
    episode_metrics,
    parse_detection_response,
)


def _state(
    *,
    fresh: bool = True,
    detector_states: dict[str, str] | None = None,
    candidate: bool = True,
    within_tolerance: bool | None = True,
) -> dict:
    detectors = detector_states or {
        "cpdz_residual_normalized": "alerting",
        "gepfm_baseline_drift": "nominal",
    }
    hydraulic = {
        "state": "available" if within_tolerance is not None else "unavailable",
        "within_tolerance": (
            within_tolerance if within_tolerance is not None else "unavailable"
        ),
    }
    return build_detection_state(
        timestamp="2026-09-24T00:00:00Z",
        freshness={"is_fresh": fresh, "state_age_s": 0.5},
        detectors={
            name: {
                "state": state,
                "value": 4.0 if state == "alerting" else 0.2,
                "threshold": 3.0,
            }
            for name, state in detectors.items()
        },
        hydraulic_consistency=hydraulic,
        routing={"candidate": candidate},
    )


def _valid_body(
    *,
    alarm: float = 0.92,
    event_type: str = "cyber_attack",
    severity: float = 3.0,
    model: str = "jev-1.13.0",
) -> dict:
    event_probabilities = {event: 0.05 for event in EVENT_TYPES}
    event_probabilities[event_type] = 0.85
    severity_probabilities = {str(i): 0.05 for i in range(5)}
    severity_probabilities[str(round(severity))] = 0.8
    return {
        "model": model,
        "answers": {
            "alarm": {"type": "noul", "noul": alarm},
            "event_type": {
                "type": "choice",
                "choice": event_type,
                "probabilities": event_probabilities,
                "confidence": 0.85,
            },
            "severity": {
                "type": "score",
                "score": severity,
                "legend": {str(i): f"level {i}" for i in range(5)},
                "probabilities": severity_probabilities,
                "confidence": 0.8,
            },
        },
        "usage": {"input_tokens": 100, "output_tokens": 25},
    }


def test_build_detection_state_is_json_safe_and_marks_missing_hydraulics() -> None:
    state = build_detection_state(
        timestamp="2026-09-24T00:00:00Z",
        freshness={"is_fresh": True},
        detectors={"ae": {"state": "nominal", "value": 0.1}},
    )
    assert state["schema_version"].startswith("hydrojev.detection_state/")
    assert state["hydraulic_consistency"]["state"] == "unavailable"
    assert state["hydraulic_consistency"]["mass_balance_residual_m3_s"] == "unavailable"
    import json

    assert json.loads(json.dumps(state)) == state


def test_build_detection_state_rejects_nonfinite_values_and_unknown_detector_state() -> None:
    with pytest.raises(ValueError, match="NaN|infinity"):
        build_detection_state(
            timestamp="t",
            freshness={"is_fresh": True},
            detectors={"ae": {"state": "nominal", "value": float("nan")}},
        )
    with pytest.raises(ValueError, match="detector state"):
        build_detection_state(
            timestamp="t",
            freshness={"is_fresh": True},
            detectors={"ae": {"state": "maybe"}},
        )


def test_questions_and_request_have_detection_contract() -> None:
    questions = build_detection_questions()
    assert set(questions) == set(DETECTION_QUESTION_IDS)
    assert questions["alarm"]["type"] == "noul"
    assert questions["event_type"]["type"] == "choice"
    assert set(questions["event_type"]["criteria"]) == set(EVENT_TYPES)
    assert questions["severity"]["type"] == "score"
    request = build_detection_request(_state(), model="jev-latest")
    assert request["state"] == _state()
    assert request["model"] == "jev-latest"


@pytest.mark.parametrize("key", [
    "ATT_FLAG", "att_flag", "attack_id", "attack_ids", "scenario_ids",
    "severity_label", "row_index", "row_indices", "row_id", "raw_index",
    "grid_index", "true_severity", "expected_severity", "attack_onset",
])
def test_request_rejects_nested_labels_attack_ids_and_row_metadata(key: str) -> None:
    state = _state()
    state["provenance"]["nested"] = [{key: 1}]
    with pytest.raises(HydroJEVSchemaError, match="forbidden ground-truth key"):
        build_detection_request(state)


def test_request_rejects_whitespace_padded_forbidden_key() -> None:
    state = _state()
    state["provenance"][" ATT_FLAG "] = 1
    with pytest.raises(HydroJEVSchemaError, match="forbidden ground-truth key"):
        build_detection_request(state)


def test_parse_detection_response_is_typed_and_does_not_mutate_input() -> None:
    body = _valid_body()
    snapshot = copy.deepcopy(body)
    decision = parse_detection_response(body, source="live", timing_ms={"http_ms": 10.0})
    assert decision.alarm_probability == pytest.approx(0.92)
    assert decision.event_type == "cyber_attack"
    assert decision.severity_score == pytest.approx(3.0)
    assert decision.source == "live"
    assert decision.usage["input_tokens"] == 100
    assert body == snapshot


@pytest.mark.parametrize(
    "mutator, pattern",
    [
        (lambda body: body["answers"].pop("severity"), "omitted"),
        (lambda body: body["answers"]["event_type"]["probabilities"].pop("normal_transient"), "options"),
        (lambda body: body["answers"]["alarm"].update(confidence=0.8), "confidence"),
        (lambda body: body["answers"]["severity"].update(score=8.0), "severity score"),
        (lambda body: body.update(model=""), "model"),
    ],
)
def test_parse_detection_response_rejects_schema_errors(mutator, pattern: str) -> None:
    body = _valid_body()
    mutator(body)
    with pytest.raises(HydroJEVSchemaError, match=pattern):
        parse_detection_response(body, source="live")


def test_no_jev_baseline_is_deterministic_and_uses_physical_evidence() -> None:
    state = _state(
        detector_states={
            "cpdz_residual_normalized": "alerting",
            "rat_covariance_distortion": "alerting",
        },
        within_tolerance=False,
    )
    first = deterministic_no_jev(state)
    second = deterministic_no_jev(state)
    assert first == second
    assert first.state is DetectionState.ALARM
    assert first.source == "no_jev"
    assert first.event_type == "physical_fault"


def test_no_jev_baseline_abstains_on_stale_or_unavailable_evidence() -> None:
    stale = deterministic_no_jev(_state(fresh=False))
    unavailable = deterministic_no_jev(
        _state(
            detector_states={"rat": "unavailable"},
            within_tolerance=None,
        )
    )
    assert stale.state is DetectionState.ABSTAIN
    assert stale.abstain_reason == "stale_evidence"
    assert unavailable.state is DetectionState.ABSTAIN
    assert unavailable.abstain_reason == "no_usable_detector_evidence"


def test_hydrojev_detector_fuses_typed_live_answer_into_three_states() -> None:
    detector = HydroJEVDetector(alarm_threshold=0.8, normal_threshold=0.2)
    state = _state(detector_states={"cpdz": "alerting", "rat": "alerting"})
    alarm = detector.predict_response(state, _valid_body(alarm=0.92), source="live")
    normal = detector.predict_response(
        state, _valid_body(alarm=0.05, event_type="normal_transient", severity=0.0), source="live"
    )
    uncertain = detector.predict_response(
        state, _valid_body(alarm=0.5, event_type="insufficient_evidence", severity=1.0), source="live"
    )
    assert alarm.state is DetectionState.ALARM
    assert normal.state is DetectionState.NORMAL
    assert uncertain.state is DetectionState.ABSTAIN
    assert alarm.source == normal.source == uncertain.source == "live"


def test_hydrojev_detector_rejects_contradictory_live_answers_as_abstention() -> None:
    detector = HydroJEVDetector(alarm_threshold=0.8, normal_threshold=0.2)
    result = detector.predict_response(
        _state(),
        _valid_body(alarm=0.95, event_type="normal_transient"),
        source="live",
    )
    assert result.state is DetectionState.ABSTAIN
    assert result.abstain_reason == "contradictory_jev_answers"


def test_candidate_false_is_a_no_call_normal_result() -> None:
    detector = HydroJEVDetector()
    result = detector.predict_response(
        _state(candidate=False),
        _valid_body(alarm=0.99),
        source="live",
    )
    assert result.state is DetectionState.NORMAL
    assert result.source == "no_call"
    assert result.alarm_probability is None


def test_episode_metrics_report_recall_fpr_delay_abstention_and_sources() -> None:
    detector = HydroJEVDetector()
    attack_state = _state(detector_states={"cpdz": "alerting", "rat": "alerting"})
    benign_state = _state(detector_states={"cpdz": "nominal", "rat": "nominal"})
    attack = DetectionEpisode(
        "a1",
        True,
        (
            detector.predict_response(attack_state, _valid_body(alarm=0.4), source="live"),
            detector.predict_response(attack_state, _valid_body(alarm=0.95), source="live"),
        ),
        onset_index=0,
        step_interval_s=60.0,
    )
    benign = DetectionEpisode(
        "b1",
        False,
        (
            detector.predict_response(
                benign_state, _valid_body(alarm=0.5, event_type="insufficient_evidence"), source="live"
            ),
            deterministic_no_jev(benign_state),
        ),
    )
    metrics = episode_metrics([attack, benign])
    assert metrics["attack_scenarios"] == 1
    assert metrics["true_positives"] == 1
    assert metrics["scenario_recall"] == pytest.approx(1.0)
    assert metrics["benign_scenario_fp_rate"] == pytest.approx(0.0)
    assert metrics["mean_detection_delay_s"] == pytest.approx(60.0)
    assert metrics["abstention_rate"] > 0.0
    assert metrics["source_counts"]["live"] == 3
    assert metrics["source_counts"]["no_jev"] == 1

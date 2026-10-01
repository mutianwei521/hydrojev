"""Schema-exactness tests for the Jev three-question contract."""

from __future__ import annotations

import copy

import pytest

from hydrojev.arbiter.decision_primitives import (
    QUESTION_IDS,
    SEVERITY_LEGEND,
    JevSchemaError,
    JevSource,
    ThreatCause,
    build_questions,
    build_request,
    parse_response,
)


def _valid_body() -> dict:
    return {
        "model": "jev-latest",
        "answers": {
            "threat_cause": {
                "type": "choice",
                "choice": "hydraulic_fs_fdi",
                "probabilities": {
                    "adversarial_ae_evasion": 0.20,
                    "hydraulic_fs_fdi": 0.55,
                    "physical_burst": 0.15,
                    "normal_transient": 0.10,
                },
                "confidence": 0.55,
            },
            "trip_edge_airgap": {"type": "noul", "noul": 0.82},
            "severity_score": {
                "type": "score",
                "score": 2.9,
                "legend": {str(i): text for i, text in enumerate(SEVERITY_LEGEND)},
                "probabilities": {"0": 0.0, "1": 0.05, "2": 0.20, "3": 0.60, "4": 0.15},
                "confidence": 0.60,
            },
        },
        "usage": {"input_tokens": 321, "output_tokens": 24},
    }


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def test_build_questions_matches_the_three_primitive_contract() -> None:
    questions = build_questions()
    assert set(questions) == set(QUESTION_IDS)
    assert questions["threat_cause"]["type"] == "choice"
    assert set(questions["threat_cause"]["criteria"]) == {c.value for c in ThreatCause}
    assert questions["trip_edge_airgap"]["type"] == "noul"
    assert set(questions["trip_edge_airgap"]["criteria"]) == {"true", "false"}
    assert questions["severity_score"]["type"] == "score"
    assert len(questions["severity_score"]["criteria"]) == 5


def test_build_request_wraps_state_model_questions() -> None:
    request = build_request({"zone": "CPDZ-1"}, model="jev-latest")
    assert request["state"] == {"zone": "CPDZ-1"}
    assert request["model"] == "jev-latest"
    assert set(request["questions"]) == set(QUESTION_IDS)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


def test_parse_valid_response_yields_typed_decision() -> None:
    decision = parse_response(_valid_body(), source=JevSource.LIVE)
    assert decision.threat_cause.as_threat_cause() is ThreatCause.HYDRAULIC_FS_FDI
    assert decision.threat_cause.confidence == pytest.approx(0.55)
    assert decision.trip_edge_airgap.probability == pytest.approx(0.82)
    assert decision.severity.expected_level == pytest.approx(2.9)
    assert decision.severity.user_facing_level == pytest.approx(3.9)
    assert decision.severity.most_likely_level == 3
    assert decision.source is JevSource.LIVE
    assert decision.usage.input_tokens == 321


def test_parse_rejects_missing_question() -> None:
    body = _valid_body()
    del body["answers"]["severity_score"]
    with pytest.raises(JevSchemaError, match="omitted"):
        parse_response(body, source=JevSource.MOCK)


def test_parse_rejects_choice_options_off_taxonomy() -> None:
    body = _valid_body()
    body["answers"]["threat_cause"]["probabilities"] = {"foo": 0.5, "bar": 0.5}
    with pytest.raises(JevSchemaError, match="taxonomy"):
        parse_response(body, source=JevSource.MOCK)


def test_parse_rejects_choice_probabilities_not_summing_to_one() -> None:
    body = _valid_body()
    body["answers"]["threat_cause"]["probabilities"]["hydraulic_fs_fdi"] = 0.9
    with pytest.raises(JevSchemaError, match="sum to 1"):
        parse_response(body, source=JevSource.MOCK)


def test_parse_rejects_noul_out_of_range() -> None:
    body = _valid_body()
    body["answers"]["trip_edge_airgap"]["noul"] = 1.4
    with pytest.raises(JevSchemaError, match=r"\[0, 1\]"):
        parse_response(body, source=JevSource.MOCK)


def test_parse_rejects_noul_carrying_confidence() -> None:
    body = _valid_body()
    body["answers"]["trip_edge_airgap"]["confidence"] = 0.9
    with pytest.raises(JevSchemaError, match="confidence"):
        parse_response(body, source=JevSource.MOCK)


def test_parse_rejects_score_indices_not_zero_based_five_levels() -> None:
    body = _valid_body()
    body["answers"]["severity_score"]["probabilities"] = {"1": 0.5, "2": 0.5}
    with pytest.raises(JevSchemaError, match="0-based range"):
        parse_response(body, source=JevSource.MOCK)


def test_parse_rejects_wrong_answer_type() -> None:
    body = _valid_body()
    body["answers"]["threat_cause"]["type"] = "noul"
    with pytest.raises(JevSchemaError, match="choice"):
        parse_response(body, source=JevSource.MOCK)


def test_parse_is_pure_and_does_not_mutate_input() -> None:
    body = _valid_body()
    snapshot = copy.deepcopy(body)
    parse_response(body, source=JevSource.LIVE)
    assert body == snapshot


def test_parse_accepts_two_decimal_rounding_and_renormalizes() -> None:
    # Live Jev (jev-1.13.0) returned {1: 0.55, 2: 0.40, 3: 0.04} -> total 0.99.
    body = _valid_body()
    body["answers"]["severity_score"]["probabilities"] = {"0": 0.0, "1": 0.55, "2": 0.4, "3": 0.04, "4": 0.0}
    decision = parse_response(body, source=JevSource.LIVE)
    assert abs(sum(decision.severity.probabilities.values()) - 1.0) < 1e-12
    assert abs(decision.severity.probabilities[1] - 0.55 / 0.99) < 1e-12

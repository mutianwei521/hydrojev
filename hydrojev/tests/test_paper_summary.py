"""Checks for denominator, causal alignment and integrity failure modes."""
import json

import numpy as np
import pytest

from hydrojev.benchmarks.hydrojev_paper_provenance import _response_details
from hydrojev.benchmarks.hydrojev_paper_metrics import hold_grid, paired_event_interval, probability_metrics, paired_comparison


def test_forward_hold_does_not_use_future_decisions():
    held = hold_grid(np.array([2, 5]), np.array([True, False]), 8)
    assert held.tolist() == [False, False, True, True, True, False, False, False]


def test_probability_denominator_separates_uncalled_and_failed():
    scored = probability_metrics(np.array([0, 1, 1, 0]), [.1, None, .8, None], np.array([1, 0, 1, 1], bool))
    assert (scored["n_grid"], scored["n_called"], scored["n_available"]) == (4, 3, 2)
    assert (scored["n_not_called"], scored["n_called_missing"]) == (1, 1)
    assert scored["brier_score"] == pytest.approx(.025)
    assert scored["all_grid"]["roc_auc"] is None
    assert scored["all_grid"]["brier_score"] is None


def test_probability_cannot_exist_for_uncalled_point():
    with pytest.raises(ValueError, match="uncalled"):
        probability_metrics(np.array([1]), [.9], np.array([False]))


def test_identical_seven_episode_outcomes_do_not_prove_equivalence():
    interval = paired_event_interval(np.ones(7, bool), np.ones(7, bool))
    assert interval["difference"] == 0
    assert interval["n_event_pairs"] == 7
    assert interval["ci95"][0] < 0 < interval["ci95"][1]
    assert interval["live_only_detected"] == interval["control_only_detected"] == 0


def test_missing_stored_hash_is_not_a_match(tmp_path):
    req = tmp_path / "request.json"
    req.write_text("{}")
    details = _response_details(tmp_path, {"request_path": str(req)})
    assert details["request_file_present"]
    assert details["request_sha256"]
    assert not details["request_hash_matches"]
    assert not details["response_hash_matches"]


def test_incomplete_typed_schema_fails(tmp_path):
    resp = tmp_path / "response.json"
    resp.write_text(json.dumps({"response": {"model": "fixture-model", "answers": {}, "usage": {}},
                                "attempts": 1, "elapsed_seconds": 1, "timestamp_utc": "2026-09-24T00:00:00Z"}))
    details = _response_details(tmp_path, {"response_path": str(resp)})
    assert details["response_file_present"]
    assert not details["response_schema_valid"]


def test_abstention_is_reported_as_state_change_and_lower_bound_alarm_loss():
    labels = np.zeros(90, dtype=int)
    labels[12:24] = 1
    grid = np.arange(0, 90, 6)
    control = np.zeros(len(grid), bool)
    control[2:4] = True
    live = control.copy(); live[2] = False
    cs = np.where(control, "alarm", "normal").astype(object)
    ls = np.where(live, "alarm", "normal").astype(object); ls[2] = "abstain"
    probabilities = [None] * len(grid); probabilities[3] = .9
    compared = paired_comparison(labels, grid, live, control, ls, cs, probabilities, 0, n_boot=20)
    assert compared["decision_changes"]["grid_count"] == 1
    assert compared["decision_changes"]["hourly_count"] == 6
    assert compared["decision_changes"]["new_abstentions"] == 1
    assert compared["delta"]["mean_ttd_hours"] == 6
    assert compared["delta"]["balanced_accuracy"] < 0

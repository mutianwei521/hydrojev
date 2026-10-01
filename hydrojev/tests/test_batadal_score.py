"""Tests for the official-BATADAL-metric external anchor (:mod:`hydrojev.benchmarks.batadal_score`).

The pure scoring primitives are checked against hand-computable cases whose S values
follow directly from the competition definition (Taormina et al., 2018): a perfect
detector scores 1, a silent detector scores 0.25 (it still gets every true-negative),
a paranoid always-on detector scores 0.75, and a half-late single detection scores a
value we compute by hand. A torch/BATADAL-guarded integration test then exercises the
end-to-end runner, skipping cleanly when the assets are unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.batadal_score import (
    BATADAL_REFERENCE,
    batadal_score,
    classification_score,
    format_batadal_score_table,
    plot_batadal_score,
    time_to_detection_score,
    tune_threshold,
)


# --------------------------------------------------------------------------- #
# Pure scoring primitives
# --------------------------------------------------------------------------- #

# two attacks: samples [2,5) (duration 3) and [7,9) (duration 2)
_LABELS = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])


def test_time_to_detection_perfect_is_one():
    s_ttd, per = time_to_detection_score(_LABELS, _LABELS)
    assert s_ttd == pytest.approx(1.0)
    assert [p["ttd_samples"] for p in per] == [0, 0]
    assert all(p["detected"] for p in per)


def test_time_to_detection_silent_is_zero():
    s_ttd, per = time_to_detection_score(_LABELS, np.zeros_like(_LABELS))
    assert s_ttd == pytest.approx(0.0)
    # never detected -> TTD equals each attack's full duration
    assert [p["ttd_samples"] for p in per] == [3, 2]
    assert not any(p["detected"] for p in per)


def test_time_to_detection_half_late_single_attack():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0])   # one attack [2,6), duration 4
    alarms = np.array([0, 0, 0, 0, 1, 1, 0, 0])   # first alarm at offset 2 -> norm 0.5
    s_ttd, per = time_to_detection_score(labels, alarms)
    assert per[0]["ttd_samples"] == 2
    assert per[0]["normalized_ttd"] == pytest.approx(0.5)
    assert s_ttd == pytest.approx(0.5)


def test_time_to_detection_no_attacks_is_nan():
    s_ttd, per = time_to_detection_score(np.zeros(5), np.zeros(5))
    assert np.isnan(s_ttd)
    assert per == []


def test_classification_score_balanced_accuracy():
    # alarms: catch one attack fully, miss the other, one false positive
    alarms = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    s_clf, d = classification_score(_LABELS, alarms)
    # TP=3 (idx2,3,4), FN=2 (idx7,8), FP=1 (idx1), TN=4 (idx0,5,6,9)
    assert (d["TP"], d["FN"], d["FP"], d["TN"]) == (3, 2, 1, 4)
    assert d["TPR"] == pytest.approx(3 / 5)
    assert d["TNR"] == pytest.approx(4 / 5)
    assert s_clf == pytest.approx((3 / 5 + 4 / 5) / 2)


def test_batadal_score_perfect():
    r = batadal_score(_LABELS, _LABELS)
    assert r["S_TTD"] == pytest.approx(1.0)
    assert r["S_CLF"] == pytest.approx(1.0)
    assert r["S"] == pytest.approx(1.0)


def test_batadal_score_silent_detector():
    r = batadal_score(_LABELS, np.zeros_like(_LABELS))
    # silent: S_TTD=0, TPR=0/TNR=1 -> S_CLF=0.5 -> S=0.25
    assert r["S_TTD"] == pytest.approx(0.0)
    assert r["S_CLF"] == pytest.approx(0.5)
    assert r["S"] == pytest.approx(0.25)


def test_batadal_score_always_alarm_detector():
    r = batadal_score(_LABELS, np.ones_like(_LABELS))
    # always on: S_TTD=1 (instant), TPR=1/TNR=0 -> S_CLF=0.5 -> S=0.75
    assert r["S_TTD"] == pytest.approx(1.0)
    assert r["S_CLF"] == pytest.approx(0.5)
    assert r["S"] == pytest.approx(0.75)


def test_batadal_score_half_late_case():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    alarms = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    r = batadal_score(labels, alarms)
    # S_TTD=0.5; TP=2,FN=2,FP=0,TN=4 -> TPR=0.5,TNR=1 -> S_CLF=0.75 -> S=0.625
    assert r["S_TTD"] == pytest.approx(0.5)
    assert r["S_CLF"] == pytest.approx(0.75)
    assert r["S"] == pytest.approx(0.625)


def test_gamma_weights_the_two_subscores():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    alarms = np.array([0, 0, 0, 0, 1, 1, 0, 0])
    # gamma=1 -> S == S_TTD; gamma=0 -> S == S_CLF
    assert batadal_score(labels, alarms, gamma=1.0)["S"] == pytest.approx(0.5)
    assert batadal_score(labels, alarms, gamma=0.0)["S"] == pytest.approx(0.75)


def test_tune_threshold_finds_separating_value():
    labels = _LABELS
    # attack samples score high, normal low; a threshold in (0.2, 0.85] separates perfectly
    scores = np.array([0.10, 0.20, 0.90, 0.95, 0.85, 0.15, 0.05, 0.88, 0.92, 0.10])
    thr, best_s = tune_threshold(scores, labels)
    assert best_s == pytest.approx(1.0)
    alarms = np.isfinite(scores) & (scores >= thr)
    assert np.array_equal(alarms.astype(int), labels)


def test_tune_threshold_prefers_balance_over_trivial_extremes():
    # a detector that can only either over- or under-alarm should still beat 0.25/0.75
    labels = _LABELS
    scores = np.array([0.1, 0.2, 0.9, 0.4, 0.85, 0.15, 0.05, 0.88, 0.3, 0.1])
    thr, best_s = tune_threshold(scores, labels)
    # best achievable here is > the always-on baseline (0.75)
    assert best_s > 0.75


def test_tune_threshold_all_nonfinite_returns_nan():
    scores = np.array([np.nan, np.inf, -np.inf])
    labels = np.array([0, 1, 0])
    thr, best_s = tune_threshold(scores, labels)
    assert thr == float("inf")
    assert np.isnan(best_s)


def test_nonfinite_scores_never_alarm():
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.1, np.nan, 5.0, 0.1])
    # at a low threshold the finite high score alarms but the NaN never does
    alarms = np.isfinite(scores) & (scores >= 0.0)
    assert alarms.tolist() == [True, False, True, True]


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _synthetic_ok_result() -> dict:
    def det(s_ttd, s_clf, tpr, tnr, dev_s):
        s = 0.5 * s_ttd + 0.5 * s_clf
        return {"dev_threshold": 1.23, "dev_S": dev_s,
                "test": {"S": s, "S_TTD": s_ttd, "S_CLF": s_clf, "gamma": 0.5,
                         "classification": {"TP": 5, "FP": 1, "TN": 9, "FN": 2,
                                            "TPR": tpr, "TNR": tnr},
                         "per_attack": []}}
    detectors = {
        "CPDZ residual": det(0.8, 0.85, 0.9, 0.8, 0.82),
        "GEpFM drift": det(0.7, 0.80, 0.8, 0.8, 0.78),
        "RAT covariance": det(0.6, 0.75, 0.7, 0.8, 0.70),
        "Mahalanobis": det(0.5, 0.70, 0.6, 0.8, 0.65),
    }
    families = ["CPDZ residual", "GEpFM drift", "RAT covariance"]
    ranking = sorted(detectors, key=lambda n: detectors[n]["test"]["S"], reverse=True)
    union = {"S": 0.86, "S_TTD": 0.9, "S_CLF": 0.82, "gamma": 0.5,
             "classification": {"TP": 6, "FP": 2, "TN": 8, "FN": 1, "TPR": 0.95, "TNR": 0.8},
             "per_attack": []}
    return {
        "status": "ok",
        "seed": 42,
        "n_test_attacks": 7,
        "n_test_samples": 2018,
        "families": families,
        "references": ["Mahalanobis", "PCA residual", "HydroJEV AE"],
        "detectors": detectors,
        "ranking_by_test_S": ranking,
        "hydrojev_family_union": {"test": union, "dev": union, "members": families},
        "published_reference": BATADAL_REFERENCE,
    }


def test_format_table_ok_and_not_run():
    table = format_batadal_score_table(_synthetic_ok_result())
    assert "BATADAL" in table
    assert "family union" in table
    assert "CPDZ residual" in table
    # the published range is cited for context, clearly not as our own numbers
    assert "context" in table.lower()
    assert format_batadal_score_table({"status": "not_run"}) == "not_run"


def test_plot_ok_and_not_run(tmp_path):
    out = plot_batadal_score(_synthetic_ok_result(), str(tmp_path / "bat.png"))
    assert out is not None and (tmp_path / "bat.png").exists()
    assert plot_batadal_score({"status": "not_run"}, "unused.png") is None


def test_reference_is_citation_not_fabricated_numbers():
    # honesty invariant: we keep a citation + qualitative range, never per-team decimals
    assert "Taormina" in BATADAL_REFERENCE["citation"]
    assert BATADAL_REFERENCE["n_ranked_entries"] == 7
    lo, hi = BATADAL_REFERENCE["published_S_range_approx"]
    assert 0.0 <= lo < hi <= 1.0


# --------------------------------------------------------------------------- #
# Integration on real BATADAL (guarded)
# --------------------------------------------------------------------------- #


def test_batadal_official_runner_if_available():
    pytest.importorskip("torch")
    from hydrojev.config import load_config
    from hydrojev.benchmarks.batadal_score import run_batadal_official_score
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_batadal_official_score(config, mode=mode, seed=7, max_epochs=4)
    if result.get("status") != "ok":
        pytest.skip(f"BATADAL unavailable: {result.get('reason')}")

    # the held-out competition test set has exactly 7 attacks
    assert result["n_test_attacks"] == 7
    # every detector produced in-range official sub-scores on the test set
    for name, d in result["detectors"].items():
        t = d["test"]
        assert 0.0 <= t["S"] <= 1.0
        assert 0.0 <= t["S_TTD"] <= 1.0
        assert 0.0 <= t["S_CLF"] <= 1.0
        assert np.isfinite(d["dev_threshold"])
    # the family union is reported and in range
    u = result["hydrojev_family_union"]["test"]
    assert 0.0 <= u["S"] <= 1.0
    # named families are present among the scored detectors
    assert any(f in result["detectors"] for f in result["families"])

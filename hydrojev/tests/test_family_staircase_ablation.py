"""Tests for the faithful named-family defense-in-depth staircase ablation (§6.2).

The subset-awareness of the shared evasion primitives (``_window_exceedance`` /
``_families_alarming`` / ``temporal_conceal(target_families=...)``) is pinned on
deterministic synthetic cases with a zero predictor (so residuals equal the raw
stream and every family score is controllable). The aggregation and presentation are
unit-tested pure, and the end-to-end staircase invariant (constant denominator +
monotone-non-increasing evasion via the sound cascade) is checked on a torch/BATADAL
guarded smoke run.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.named_family_benchmark import FamilyReference
from hydrojev.benchmarks.temporal_evasion import (
    FAMILIES,
    _families_alarming,
    _window_exceedance,
    temporal_conceal,
)
from hydrojev.benchmarks.family_staircase_ablation import (
    _rung_summary,
    format_staircase_table,
    plot_staircase,
)


def _zero_predict(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=float)
    return np.zeros((windows.shape[0], windows.shape[2]), dtype=float)


def _reference(features: int, *, rat_window: int) -> FamilyReference:
    dummy = np.array([np.nan], dtype=float)
    return FamilyReference(
        scale=np.ones(features),
        gepfm_ref_mean=0.0,
        gepfm_ref_scale=1.0,
        gepfm_smoothing=0.2,
        rat_ref_cov=np.eye(features),
        rat_window=rat_window,
        clean_cpdz=dummy,
        clean_gepfm=dummy,
        clean_rat=dummy,
    )


# --------------------------------------------------------------------------- #
# subset-aware objective helpers
# --------------------------------------------------------------------------- #


def test_window_exceedance_and_alarming_restrict_to_subset():
    thr = {f: 1.0 for f in FAMILIES}
    scales = {f: 1.0 for f in FAMILIES}
    ws = {"CPDZ residual": 3.0, "GEpFM drift": 4.0, "RAT covariance": 2.0}

    # Full stack: all three exceed -> (3-1)+(4-1)+(2-1) = 6; all three alarm.
    assert _window_exceedance(ws, thr, scales) == pytest.approx(6.0)
    assert _families_alarming(ws, thr) == FAMILIES

    # Deployed subset {CPDZ}: only CPDZ counts -> 2.0; only CPDZ alarms.
    only_cpdz = ("CPDZ residual",)
    assert _window_exceedance(ws, thr, scales, only_cpdz) == pytest.approx(2.0)
    assert _families_alarming(ws, thr, only_cpdz) == only_cpdz

    # Deployed subset {CPDZ, GEpFM}: RAT ignored even though it exceeds.
    two = ("CPDZ residual", "GEpFM drift")
    assert _window_exceedance(ws, thr, scales, two) == pytest.approx(5.0)
    assert _families_alarming(ws, thr, two) == two


# --------------------------------------------------------------------------- #
# temporal_conceal(target_families=...)
# --------------------------------------------------------------------------- #


def _conceal(target_families, *, budget=500):
    features, seq_len, rat_window = 3, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((6, features))
    attack[:, 0] = 5.0  # sustained spike -> CPDZ and (via drift) GEpFM both fire
    ref = _reference(features, rat_window=rat_window)
    thr = {"CPDZ residual": 1.0, "GEpFM drift": 1.0, "RAT covariance": 1e9}
    scales = {f: 1.0 for f in FAMILIES}
    return temporal_conceal(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
        modifiable_mask=np.ones(features, dtype=bool), max_changed=2, candidate_values=11,
        budget=budget, target_families=target_families,
    )


def test_conceal_originally_alarming_restricted_to_deployed_subset():
    # With the full stack both CPDZ and GEpFM fire on the sustained spike.
    full = _conceal(None)
    assert "CPDZ residual" in full.originally_alarming
    assert "GEpFM drift" in full.originally_alarming

    # Deploying only CPDZ restricts the objective AND the judged alarm set to CPDZ:
    # GEpFM is a bystander that plays no role, even though it exceeds its threshold.
    cpdz_only = _conceal(("CPDZ residual",))
    assert cpdz_only.originally_alarming == ("CPDZ residual",)
    assert "GEpFM drift" not in cpdz_only.originally_alarming


def test_conceal_target_families_validation():
    with pytest.raises(ValueError):  # empty subset
        _conceal(())
    with pytest.raises(ValueError):  # not a subset of FAMILIES
        _conceal(("not a family",))


def test_conceal_none_matches_full_families_explicitly():
    # target_families=None must be identical to passing the full FAMILIES tuple.
    a = _conceal(None)
    b = _conceal(tuple(FAMILIES))
    assert a.originally_alarming == b.originally_alarming
    assert a.still_alarming == b.still_alarming
    assert a.evaded_all == b.evaded_all


# --------------------------------------------------------------------------- #
# aggregation (pure, fixed denominator)
# --------------------------------------------------------------------------- #


def test_rung_summary_fixed_denominator():
    silent = [True, False, True, False, True]   # 3 of 5 windows silent
    fired = [True, True, True, False, True]      # deployed stack fired on 4 originally
    s = _rung_summary(("CPDZ residual",), silent, fired)
    assert s["n_windows"] == 5           # denominator = ALL windows, not just fired ones
    assert s["n_evaded"] == 3
    assert s["n_fired_originally"] == 4
    assert s["joint_evasion_rate"] == pytest.approx(0.6)
    lo, hi = s["joint_evasion_wilson95"]
    assert 0.0 <= lo <= 0.6 <= hi <= 1.0


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _synthetic_ok() -> dict:
    def rung(dep, n_fired, n_evaded, rate, ci):
        return {
            "deployed": dep, "n_deployed": len(dep), "n_windows": 5,
            "n_fired_originally": n_fired, "n_evaded": n_evaded,
            "joint_evasion_rate": rate, "joint_evasion_wilson95": ci,
        }
    return {
        "status": "ok", "train_dataset": "dataset03", "evaluate_dataset": "dataset04",
        "max_changed": 4, "max_attack_len": 24, "n_targets": 5, "families": list(FAMILIES),
        "deployment_order": [FAMILIES[:1], FAMILIES[:2], FAMILIES[:3]],
        "selection_ceiling": {
            "n_scenarios": 7, "n_selected": 5, "n_dropped": 2,
            "dropped": [
                {"run_start": 2027, "drop_reason": "families_silent", "family_scores": None},
                {"run_start": 3927, "drop_reason": "families_silent", "family_scores": None},
            ],
            "note": "n is a data ceiling",
        },
        "staircase": [
            rung(list(FAMILIES[:1]), 5, 3, 0.6, [0.23, 0.88]),
            rung(list(FAMILIES[:2]), 5, 2, 0.4, [0.12, 0.77]),
            rung(list(FAMILIES[:3]), 5, 2, 0.4, [0.12, 0.77]),
        ],
        "single_family": {
            "CPDZ residual": rung(["CPDZ residual"], 5, 3, 0.6, [0.23, 0.88]),
            "GEpFM drift": rung(["GEpFM drift"], 3, 2, 0.4, [0.12, 0.77]),
            "RAT covariance": rung(["RAT covariance"], 2, 3, 0.6, [0.23, 0.88]),
        },
    }


def test_format_table_ok_and_not_run():
    text = format_staircase_table(_synthetic_ok())
    assert "0.600" in text and "0.400" in text
    assert "CPDZ" in text and "GEpFM" in text and "RAT" in text
    assert "0.600 -> 0.400" in text  # the headline drop line
    # The honest n=5 ceiling line: 7 scenarios, 5 flagged, the 2 misses named.
    assert "7 attack scenarios" in text and "flags 5" in text
    assert "2027" in text and "3927" in text and "data ceiling" in text
    assert format_staircase_table({"status": "not_run", "reason": "x"}) == "not_run"


def test_format_table_without_ceiling_is_backward_compatible():
    # A result lacking selection_ceiling (older artifact) must still render, no crash.
    result = _synthetic_ok()
    del result["selection_ceiling"]
    text = format_staircase_table(result)
    assert "0.600 -> 0.400" in text
    assert "data ceiling" not in text


def test_plot_staircase(tmp_path):
    out = tmp_path / "staircase.png"
    path = plot_staircase(_synthetic_ok(), str(out))
    assert path is not None and out.exists() and out.stat().st_size > 0
    assert plot_staircase({"status": "not_run"}, str(tmp_path / "none.png")) is None


# --------------------------------------------------------------------------- #
# guarded end-to-end invariant: constant denominator + monotone staircase
# --------------------------------------------------------------------------- #


def test_staircase_ablation_if_available():
    pytest.importorskip("torch")
    from hydrojev.benchmarks.family_staircase_ablation import run_family_staircase_ablation
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_family_staircase_ablation(
        config, mode=mode, seed=7, max_epochs=6, max_targets=3, budget=500
    )
    if result.get("status") != "ok":
        pytest.skip(f"staircase ablation not runnable: {result.get('reason')}")

    staircase = result["staircase"]
    assert len(staircase) == len(FAMILIES)
    assert [r["n_deployed"] for r in staircase] == [1, 2, 3]
    assert result["deployment_order"] == [
        list(FAMILIES[:1]), list(FAMILIES[:2]), list(FAMILIES[:3])
    ]

    n = result["n_targets"]
    assert n > 0
    # Fixed denominator: every rung is scored over ALL flagged windows.
    assert all(r["n_windows"] == n for r in staircase)
    # Larger deployed stacks fire on at least as many windows (nested supersets).
    fired = [r["n_fired_originally"] for r in staircase]
    assert all(fired[i] <= fired[i + 1] for i in range(len(fired) - 1))
    # The staircase is monotone NON-INCREASING by the sound cascade.
    rates = [r["joint_evasion_rate"] for r in staircase]
    assert all(rates[i] >= rates[i + 1] - 1e-9 for i in range(len(rates) - 1))
    for r in staircase:
        assert 0.0 <= r["joint_evasion_rate"] <= 1.0
        lo, hi = r["joint_evasion_wilson95"]
        assert 0.0 <= lo <= hi <= 1.0
    # Per-single-family block present with the same fixed denominator.
    for fam in FAMILIES:
        assert result["single_family"][fam]["n_windows"] == n

    # Honest n=5 ceiling: the selection report is the single source of truth and must be
    # self-consistent with the fixture that drove the staircase (drift guard).
    ceiling = result["selection_ceiling"]
    report = result["window_selection"]
    assert ceiling["n_scenarios"] == len(report)
    assert ceiling["n_selected"] == sum(1 for e in report if e["selected"]) == n
    assert ceiling["n_dropped"] == ceiling["n_scenarios"] - ceiling["n_selected"]
    assert ceiling["n_scenarios"] >= ceiling["n_selected"]  # detected <= all scenarios
    # Every dropped scenario carries a reason; every selected one cleared both gates.
    for e in report:
        assert e["history_ok"] is True or e["drop_reason"] == "insufficient_history"
        if e["selected"]:
            assert e["drop_reason"] is None and e["history_ok"] and e["length_ok"]
            assert e["alarming"]  # a selected window must have flagged >=1 family
        else:
            assert e["drop_reason"] in {
                "insufficient_history", "attack_too_short", "families_silent"
            }
    for d in ceiling["dropped"]:
        assert "run_start" in d and "drop_reason" in d

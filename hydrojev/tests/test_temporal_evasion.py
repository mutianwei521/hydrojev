"""Tests for the temporal (trajectory) evasion attack on the named families.

Pure helpers are pinned against the already-tested residual-engine primitives via
the public ``*_series`` functions, and the greedy concealer is exercised on
deterministic synthetic cases (a trivially evadable spike on a modifiable sensor, a
non-evadable spike on a frozen sensor, and the sensor-sparsity cap) using a zero
predictor so residuals equal the raw stream and the objective is fully controllable.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.named_family_benchmark import (
    FamilyReference,
    cpdz_series,
    gepfm_series,
    one_step_residual_stream,
    rat_series,
)
from hydrojev.benchmarks.temporal_evasion import (
    FAMILIES,
    TemporalEvasionResult,
    _families_alarming,
    _window_exceedance,
    format_temporal_evasion_table,
    plot_temporal_evasion,
    temporal_conceal,
    temporal_conceal_adaptive,
    temporal_family_window_scores,
)


def _zero_predict(windows: np.ndarray) -> np.ndarray:
    """Predict the zero vector for every window (residual == observed)."""

    windows = np.asarray(windows, dtype=float)
    return np.zeros((windows.shape[0], windows.shape[2]), dtype=float)


def _reference(features: int, *, rat_window: int) -> FamilyReference:
    """A FamilyReference whose only live fields are the scoring parameters."""

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
# temporal_family_window_scores: wiring pinned to the tested primitives
# --------------------------------------------------------------------------- #


def test_window_scores_match_series_tail_max():
    features, seq_len, rat_window = 3, 2, 3
    context_len = seq_len + rat_window
    rng = np.random.default_rng(0)
    context = rng.normal(size=(context_len, features)) * 0.1
    attack = rng.normal(size=(5, features))
    attack[:, 0] += 4.0  # a clear anomaly on sensor 0
    stream = np.vstack([context, attack])
    ref = _reference(features, rat_window=rat_window)

    residuals, start = one_step_residual_stream(_zero_predict, stream, seq_len)
    exp_cpdz = cpdz_series(residuals, ref.scale)
    exp_gepfm = gepfm_series(
        exp_cpdz, reference_mean=ref.gepfm_ref_mean,
        reference_scale=ref.gepfm_ref_scale, smoothing=ref.gepfm_smoothing,
    )
    exp_rat = rat_series(residuals, ref.rat_ref_cov, window=ref.rat_window)
    tail = slice(context_len - start, None)

    def finite_max(series):
        vals = series[tail]
        vals = vals[np.isfinite(vals)]
        return float(np.max(vals))

    got = temporal_family_window_scores(
        _zero_predict, stream, seq_len=seq_len, reference=ref, window_start=context_len
    )
    assert got["CPDZ residual"] == pytest.approx(finite_max(exp_cpdz))
    assert got["GEpFM drift"] == pytest.approx(finite_max(exp_gepfm))
    assert got["RAT covariance"] == pytest.approx(finite_max(exp_rat))


def test_window_scores_reject_early_window_start():
    ref = _reference(3, rat_window=3)
    stream = np.zeros((9, 3))
    with pytest.raises(ValueError):
        temporal_family_window_scores(
            _zero_predict, stream, seq_len=2, reference=ref, window_start=1  # < start (=2)
        )


# --------------------------------------------------------------------------- #
# objective helpers
# --------------------------------------------------------------------------- #


def test_window_exceedance_and_alarming():
    thr = {f: 1.0 for f in FAMILIES}
    scales = {f: 1.0 for f in FAMILIES}
    ws = {"CPDZ residual": 3.0, "GEpFM drift": 0.5, "RAT covariance": float("nan")}
    # (3-1)/1 from CPDZ; GEpFM below threshold; RAT nan skipped.
    assert _window_exceedance(ws, thr, scales) == pytest.approx(2.0)
    assert _families_alarming(ws, thr) == ("CPDZ residual",)


# --------------------------------------------------------------------------- #
# temporal_conceal greedy behaviour on controllable synthetic cases
# --------------------------------------------------------------------------- #


def _single_family_thresholds():
    # Only CPDZ can fire; GEpFM/RAT are pushed out of range so the objective and
    # the alarm set reduce to CPDZ alone (clean, controllable test).
    thr = {"CPDZ residual": 1.0, "GEpFM drift": 1e9, "RAT covariance": 1e9}
    scales = {f: 1.0 for f in FAMILIES}
    return thr, scales


def test_conceal_silences_modifiable_spike():
    features, seq_len, rat_window = 3, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((4, features))
    attack[:, 0] = 5.0  # anomaly the attacker owns
    ref = _reference(features, rat_window=rat_window)
    thr, scales = _single_family_thresholds()

    res = temporal_conceal(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
        modifiable_mask=np.ones(features, dtype=bool), max_changed=2, candidate_values=11,
        budget=2000,
    )
    assert isinstance(res, TemporalEvasionResult)
    assert res.originally_alarming == ("CPDZ residual",)
    assert res.evaded_all is True
    assert res.still_alarming == ()
    assert res.changed_sensors == (0,)
    assert res.n_changed == 1
    assert res.termination_reason == "threshold_reached"


def test_conceal_cannot_silence_frozen_spike():
    features, seq_len, rat_window = 3, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((4, features))
    attack[:, 0] = 5.0
    ref = _reference(features, rat_window=rat_window)
    thr, scales = _single_family_thresholds()

    frozen_mask = np.array([False, True, True])  # sensor 0 (the anomaly) is immutable
    res = temporal_conceal(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
        modifiable_mask=frozen_mask, max_changed=2, candidate_values=11, budget=2000,
    )
    assert res.evaded_all is False
    assert "CPDZ residual" in res.still_alarming
    assert res.n_changed == 0
    assert res.termination_reason == "no_improvement"


def test_conceal_respects_sensor_sparsity_cap():
    features, seq_len, rat_window = 6, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((4, features))
    attack[:, :5] = 5.0  # five spiked sensors, attacker may hold at most two
    ref = _reference(features, rat_window=rat_window)
    thr, scales = _single_family_thresholds()

    res = temporal_conceal(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
        modifiable_mask=np.ones(features, dtype=bool), max_changed=2, candidate_values=11,
        budget=4000,
    )
    assert res.n_changed <= 2
    # Two sensors cannot silence five spikes, so CPDZ must remain firing.
    assert res.evaded_all is False
    assert "CPDZ residual" in res.still_alarming


def test_conceal_validates_inputs():
    ref = _reference(3, rat_window=3)
    thr, scales = _single_family_thresholds()
    ctx = np.zeros((5, 3))
    atk = np.zeros((4, 3))
    common = dict(
        reference=ref, thresholds=thr, scales=scales, seq_len=2,
        box_low=np.full(3, -5.0), box_high=np.full(3, 5.0),
        modifiable_mask=np.ones(3, dtype=bool),
    )
    with pytest.raises(ValueError):  # feature-count mismatch
        temporal_conceal(_zero_predict, ctx, np.zeros((4, 2)), **common)
    with pytest.raises(ValueError):  # max_changed out of range
        temporal_conceal(_zero_predict, ctx, atk, max_changed=99, **common)
    with pytest.raises(ValueError):  # candidate_values too small
        temporal_conceal(_zero_predict, ctx, atk, candidate_values=1, **common)


# --------------------------------------------------------------------------- #
# temporal_conceal_adaptive: the per-timestep (strictly stronger) attacker
# --------------------------------------------------------------------------- #


def test_adaptive_silences_modifiable_spike():
    # From scratch (warm_start=None) the adaptive attacker internally warm-starts from
    # the constant hold, so it silences a trivially evadable modifiable spike too.
    features, seq_len, rat_window = 3, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((4, features))
    attack[:, 0] = 5.0
    ref = _reference(features, rat_window=rat_window)
    thr, scales = _single_family_thresholds()

    res = temporal_conceal_adaptive(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
        modifiable_mask=np.ones(features, dtype=bool), max_changed=2, candidate_values=11,
        budget=2000,
    )
    assert isinstance(res, TemporalEvasionResult)
    assert res.originally_alarming == ("CPDZ residual",)
    assert res.evaded_all is True
    assert res.still_alarming == ()
    assert res.n_changed == 1  # only the spiked sensor need move
    assert res.adversarial_block is not None and res.adversarial_block.shape == attack.shape


def test_adaptive_never_worse_than_constant_hold():
    # The guarantee we claim in the paper: warm-started from the constant-hold block,
    # the per-timestep attacker's joint exceedance can only match or beat it, and it
    # never re-fires a family the constant hold had silenced or exceeds the budget cap.
    features, seq_len, rat_window = 6, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((4, features))
    attack[:, :5] = 5.0  # five spikes, at most two sensors may move
    ref = _reference(features, rat_window=rat_window)
    thr, scales = _single_family_thresholds()
    box = dict(box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
               modifiable_mask=np.ones(features, dtype=bool))

    const = temporal_conceal(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, max_changed=2, candidate_values=11, budget=4000, **box,
    )
    adapt = temporal_conceal_adaptive(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, max_changed=2, candidate_values=11, budget=4000,
        warm_start=const.adversarial_block, **box,
    )
    const_ex = _window_exceedance(const.adversarial_scores, thr, scales)
    adapt_ex = _window_exceedance(adapt.adversarial_scores, thr, scales)
    assert adapt_ex <= const_ex + 1e-9                      # never worse
    assert set(adapt.still_alarming) <= set(const.still_alarming)
    assert adapt.n_changed <= 2                             # same sensor budget


def test_adaptive_cannot_silence_frozen_spike():
    features, seq_len, rat_window = 3, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((4, features))
    attack[:, 0] = 5.0
    ref = _reference(features, rat_window=rat_window)
    thr, scales = _single_family_thresholds()

    frozen_mask = np.array([False, True, True])  # the anomaly sensor is immutable
    res = temporal_conceal_adaptive(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
        modifiable_mask=frozen_mask, max_changed=2, candidate_values=11, budget=2000,
    )
    assert res.evaded_all is False
    assert "CPDZ residual" in res.still_alarming
    assert res.n_changed == 0


def test_adaptive_respects_sensor_sparsity_cap():
    features, seq_len, rat_window = 6, 2, 3
    context = np.zeros((seq_len + rat_window, features))
    attack = np.zeros((4, features))
    attack[:, :5] = 5.0
    ref = _reference(features, rat_window=rat_window)
    thr, scales = _single_family_thresholds()

    res = temporal_conceal_adaptive(
        _zero_predict, context, attack, reference=ref, thresholds=thr, scales=scales,
        seq_len=seq_len, box_low=np.full(features, -5.0), box_high=np.full(features, 5.0),
        modifiable_mask=np.ones(features, dtype=bool), max_changed=2, candidate_values=11,
        budget=4000,
    )
    assert res.n_changed <= 2
    assert res.evaded_all is False
    assert "CPDZ residual" in res.still_alarming


def test_adaptive_validates_inputs():
    ref = _reference(3, rat_window=3)
    thr, scales = _single_family_thresholds()
    ctx = np.zeros((5, 3))
    atk = np.zeros((4, 3))
    common = dict(
        reference=ref, thresholds=thr, scales=scales, seq_len=2,
        box_low=np.full(3, -5.0), box_high=np.full(3, 5.0),
        modifiable_mask=np.ones(3, dtype=bool),
    )
    with pytest.raises(ValueError):  # feature-count mismatch
        temporal_conceal_adaptive(_zero_predict, ctx, np.zeros((4, 2)), **common)
    with pytest.raises(ValueError):  # warm_start shape mismatch
        temporal_conceal_adaptive(
            _zero_predict, ctx, atk, warm_start=np.zeros((3, 3)), **common
        )


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _synthetic_ok_result() -> dict:
    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "max_changed": 4,
        "max_attack_len": 24,
        "n_targets": 10,
        "n_jointly_evaded": 3,
        "joint_evasion_rate": 0.3,
        "joint_evasion_wilson95": [0.108, 0.603],
        "families": list(FAMILIES),
        "per_family_fired": {"CPDZ residual": 8, "GEpFM drift": 5, "RAT covariance": 4},
        "per_family_still_firing_after_joint_attack": {
            "CPDZ residual": 2, "GEpFM drift": 4, "RAT covariance": 4},
        "per_family_residual_alarm_rate": {
            "CPDZ residual": 0.25, "GEpFM drift": 0.8, "RAT covariance": 1.0},
    }


def test_format_table_ok_and_not_run():
    text = format_temporal_evasion_table(_synthetic_ok_result())
    assert "Joint evasion" in text
    assert "CPDZ residual" in text and "RAT covariance" in text
    assert "0.300" in text
    assert format_temporal_evasion_table({"status": "not_run", "reason": "x"}) == "not_run"


def test_format_table_handles_nan_residual_alarm():
    result = _synthetic_ok_result()
    result["per_family_residual_alarm_rate"]["RAT covariance"] = float("nan")
    text = format_temporal_evasion_table(result)
    assert "n/a" in text


def test_plot_temporal_evasion(tmp_path):
    out = tmp_path / "temporal.png"
    path = plot_temporal_evasion(_synthetic_ok_result(), str(out))
    assert path is not None and out.exists() and out.stat().st_size > 0
    assert plot_temporal_evasion({"status": "not_run"}, str(tmp_path / "none.png")) is None


# --------------------------------------------------------------------------- #
# guarded real-data smoke test
# --------------------------------------------------------------------------- #


def test_temporal_evasion_runner_if_available():
    pytest.importorskip("torch")
    from hydrojev.benchmarks.temporal_evasion import run_temporal_evasion_benchmark
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    result = run_temporal_evasion_benchmark(
        config, mode=mode, seed=7, max_epochs=6, max_targets=3, budget=500
    )
    if result.get("status") != "ok":
        pytest.skip(f"temporal evasion not runnable: {result.get('reason')}")

    assert 0.0 <= result["joint_evasion_rate"] <= 1.0
    assert result["n_targets"] > 0
    lo, hi = result["joint_evasion_wilson95"]
    assert 0.0 <= lo <= hi <= 1.0
    for fam in FAMILIES:
        assert fam in result["per_family_fired"]
        assert fam in result["per_family_still_firing_after_joint_attack"]
    assert len(result["targets"]) == result["n_targets"]
    for target in result["targets"]:
        assert isinstance(target["evaded_all"], bool)
        assert target["n_changed_sensors"] <= result["max_changed"]


def test_fixture_selection_report_is_the_ceiling_source_of_truth():
    """The fixture's selection_report must faithfully account for EVERY dataset04
    attack scenario — the honest n=5 ceiling reported in §6.2. Cross-checked against an
    independently computed scenario count (no magic numbers), with the cap set high
    enough not to bind so the report is complete."""

    pytest.importorskip("torch")
    from hydrojev.benchmarks.temporal_evasion import build_temporal_fixture
    from hydrojev.benchmarks.detection_baselines import contiguous_runs
    from hydrojev.config import load_config
    from hydrojev.datasets.wdn_loader import DatasetUnavailableError, load_batadal
    from hydrojev.run_experiments import RunMode

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=True)
    fixture = build_temporal_fixture(
        config, mode=mode, seed=7, max_epochs=6, max_targets=50
    )
    if isinstance(fixture, dict):
        pytest.skip(f"temporal fixture not runnable: {fixture.get('reason')}")

    # Independent ground truth: how many contiguous attack scenarios does dataset04 have?
    try:
        labelled = load_batadal("dataset04", config=config)
    except DatasetUnavailableError as exc:  # pragma: no cover
        pytest.skip(str(exc))
    labels = labelled.frame[labelled.label_column].to_numpy().astype(int)
    n_scenarios = len(contiguous_runs(labels, 1))

    report = fixture.selection_report
    # Cap did not bind (50 >> scenarios), so the report accounts for EVERY scenario.
    assert len(report) == n_scenarios
    # Selected entries correspond exactly to the fixture's actual targets (drift guard).
    selected_starts = sorted(e["run_start"] for e in report if e["selected"])
    target_starts = sorted(int(t.run_start) for t in fixture.targets)
    assert selected_starts == target_starts
    # On dataset04 every scenario starts far past the context and is >=2 steps long, so
    # the ONLY drop reason is a genuinely already-missed attack (families silent) — never
    # a gate artifact. This is what makes n a data ceiling, not a tunable.
    for e in report:
        assert e["history_ok"] and e["length_ok"]
        if e["selected"]:
            assert e["alarming"] and e["drop_reason"] is None
        else:
            assert e["drop_reason"] == "families_silent" and not e["alarming"]

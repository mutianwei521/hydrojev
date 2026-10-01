"""Tests for the detection-layer measurement-degradation robustness study (§4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.batadal_score import BATADAL_REFERENCE
from hydrojev.benchmarks.detection_robustness import (
    _aggregate,
    _alarms,
    format_detection_robustness_table,
    gaussian_sensor_noise,
    nested_dropout_indices,
    plot_detection_robustness,
    sensor_dropout,
)


# --------------------------------------------------------------------------- #
# gaussian_sensor_noise
# --------------------------------------------------------------------------- #


def test_noise_is_deterministic_in_seed():
    raw = np.arange(30.0).reshape(10, 3)
    scale = np.array([1.0, 2.0, 3.0])
    p = gaussian_sensor_noise(0.2, scale, seed=11)
    a = p(raw, ["a", "b", "c"])
    b = p(raw, ["a", "b", "c"])
    np.testing.assert_array_equal(a, b)  # same seed -> identical draw
    assert a.shape == raw.shape


def test_noise_differs_across_seeds():
    raw = np.zeros((100, 3))
    scale = np.ones(3)
    a = gaussian_sensor_noise(0.3, scale, seed=1)(raw, ["a", "b", "c"])
    b = gaussian_sensor_noise(0.3, scale, seed=2)(raw, ["a", "b", "c"])
    assert not np.allclose(a, b)


def test_noise_zero_sigma_is_identity():
    raw = np.random.default_rng(0).standard_normal((20, 4))
    p = gaussian_sensor_noise(0.0, np.ones(4), seed=5)
    np.testing.assert_array_equal(p(raw, list("abcd")), raw)


def test_noise_scaled_per_channel():
    raw = np.zeros((6000, 3))
    scale = np.array([1.0, 2.0, 3.0])
    out = gaussian_sensor_noise(0.5, scale, seed=7)(raw, ["a", "b", "c"])
    added = out - raw
    # per-channel empirical std approx sigma_frac * scale = [0.5, 1.0, 1.5]
    np.testing.assert_allclose(added.std(axis=0), 0.5 * scale, rtol=0.06)
    np.testing.assert_allclose(added.mean(axis=0), np.zeros(3), atol=0.05)


def test_noise_rejects_scale_mismatch():
    p = gaussian_sensor_noise(0.1, np.ones(5), seed=0)
    with pytest.raises(ValueError):
        p(np.zeros((4, 3)), ["a", "b", "c"])


# --------------------------------------------------------------------------- #
# sensor_dropout / nested_dropout_indices
# --------------------------------------------------------------------------- #


def test_dropout_replaces_only_selected_columns_with_fill():
    raw = np.arange(24.0).reshape(6, 4)
    fill = np.array([-1.0, -2.0, -3.0, -4.0])
    out = sensor_dropout([1, 3], fill)(raw, list("abcd"))
    # dropped columns become the constant fill; others untouched
    np.testing.assert_array_equal(out[:, 1], np.full(6, -2.0))
    np.testing.assert_array_equal(out[:, 3], np.full(6, -4.0))
    np.testing.assert_array_equal(out[:, 0], raw[:, 0])
    np.testing.assert_array_equal(out[:, 2], raw[:, 2])
    assert out.shape == raw.shape


def test_dropout_empty_is_identity():
    raw = np.random.default_rng(1).standard_normal((5, 3))
    out = sensor_dropout([], np.zeros(3))(raw, list("abc"))
    np.testing.assert_array_equal(out, raw)


def test_dropout_rejects_fill_mismatch():
    with pytest.raises(ValueError):
        sensor_dropout([0], np.ones(2))(np.zeros((3, 4)), list("abcd"))


def test_nested_dropout_indices_are_nested_and_sized():
    n = 43
    lo = nested_dropout_indices(n, 0.2, seed=3)
    hi = nested_dropout_indices(n, 0.5, seed=3)
    assert lo.size == round(0.2 * n)
    assert hi.size == round(0.5 * n)
    assert set(lo.tolist()).issubset(set(hi.tolist()))  # same seed -> nested prefix
    assert nested_dropout_indices(n, 0.0, seed=3).size == 0
    assert nested_dropout_indices(n, 1.0, seed=3).size == n


def test_nested_dropout_indices_deterministic():
    a = nested_dropout_indices(20, 0.3, seed=9)
    b = nested_dropout_indices(20, 0.3, seed=9)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# _alarms / _aggregate
# --------------------------------------------------------------------------- #


def test_alarms_threshold_and_nonfinite():
    s = np.array([0.1, 1.0, np.nan, 2.0, np.inf])
    got = _alarms(s, 1.0)
    # finite AND >= thr; every non-finite score (nan, +inf) never alarms, matching
    # the isfinite() convention used throughout batadal_score.py
    np.testing.assert_array_equal(got, np.array([False, True, False, True, False]))


def test_aggregate_mean_std():
    reps = [
        {"d": {"S": 0.8, "S_TTD": 0.9, "S_CLF": 0.7, "classification": {"TPR": 0.6, "TNR": 0.8}}},
        {"d": {"S": 0.6, "S_TTD": 0.7, "S_CLF": 0.5, "classification": {"TPR": 0.4, "TNR": 0.6}}},
    ]
    agg = _aggregate(reps, ["d"])["d"]
    assert agg["S_mean"] == pytest.approx(0.7)
    assert agg["S_std"] == pytest.approx(0.1)
    assert agg["S_TTD_mean"] == pytest.approx(0.8)
    assert agg["TPR_mean"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _fake_ok_result() -> dict:
    def curve(vals):
        return [
            {"level": lv, "S_mean": v, "S_std": 0.01, "S_TTD_mean": 0.95,
             "S_CLF_mean": v - 0.1, "TPR_mean": 0.7, "TNR_mean": 0.8}
            for lv, v in vals
        ]

    lv_noise = [(0.0, 0.88), (0.2, 0.80), (0.5, 0.66)]
    lv_drop = [(0.0, 0.88), (0.3, 0.78)]
    names = ["Mahalanobis", "CPDZ residual", "__union__"]
    return {
        "status": "ok", "seed": 42, "repeats": 3,
        "n_test_attacks": 7, "n_test_samples": 2018, "n_features": 43,
        "detectors": ["Mahalanobis", "CPDZ residual"],
        "families": ["CPDZ residual"],
        "references": ["Mahalanobis"],
        "baseline_S": {"Mahalanobis": 0.88, "CPDZ residual": 0.80, "__union__": 0.85},
        "noise": {"unit": "sigma_frac", "levels": [0.0, 0.2, 0.5],
                  "curves": {n: curve(lv_noise) for n in names}},
        "dropout": {"unit": "fraction", "levels": [0.0, 0.3],
                    "curves": {n: curve(lv_drop) for n in names},
                    "example_dropped_sensors": {"0.00": [], "0.30": ["L_T1"]}},
        "published_reference": BATADAL_REFERENCE,
    }


def test_format_table_ok_and_not_run():
    assert format_detection_robustness_table({"status": "not_run"}) == "not_run"
    txt = format_detection_robustness_table(_fake_ok_result())
    assert "measurement degradation" in txt
    assert "Gaussian noise" in txt
    assert "sensor dropout" in txt
    assert "family union" in txt
    # published range cited, never per-team decimals
    assert "0.50-0.97" in txt


def test_plot_ok_and_not_run(tmp_path):
    assert plot_detection_robustness({"status": "not_run"}, str(tmp_path / "x.png")) is None
    out = plot_detection_robustness(_fake_ok_result(), str(tmp_path / "rob.png"))
    assert out is not None
    assert (tmp_path / "rob.png").exists()


# --------------------------------------------------------------------------- #
# guarded end-to-end smoke test (torch + BATADAL data required)
# --------------------------------------------------------------------------- #


def test_detection_robustness_runner_if_available():
    pytest.importorskip("torch")
    from hydrojev.benchmarks.detection_robustness import run_detection_robustness
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    result = run_detection_robustness(
        load_config(), mode=RunMode.resolve(dry=False, quick=True),
        seed=42, max_epochs=4,
        noise_levels=[0.0, 0.3], dropout_levels=[0.0, 0.4], repeats=2,
    )
    if result.get("status") != "ok":
        pytest.skip(f"BATADAL data unavailable: {result.get('reason')}")

    assert result["n_test_attacks"] == 7  # held-out competition test set
    curve_names = result["detectors"] + ["__union__"]
    assert set(result["families"]).issubset(set(result["detectors"]))

    for block in (result["noise"], result["dropout"]):
        for name in curve_names:
            curve = block["curves"][name]
            assert len(curve) == len(block["levels"])
            for pt in curve:
                assert 0.0 <= pt["S_mean"] <= 1.0
                assert pt["S_std"] >= 0.0
        # level-0 point reproduces the frozen baseline exactly (sanity anchor)
        for name in curve_names:
            assert block["curves"][name][0]["S_mean"] == pytest.approx(
                result["baseline_S"][name], abs=1e-9
            )
        # clean baseline is a real level in the sweep
        assert block["levels"][0] == 0.0

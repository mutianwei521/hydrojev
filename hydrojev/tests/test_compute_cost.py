"""Tests for the computational-cost & real-time feasibility study (§4.3)."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks.compute_cost import (
    LAYER_ARBITER,
    LAYER_DETECTOR,
    LAYER_REFLEX,
    _bench,
    _headroom,
    _representative_attack_state,
    format_compute_cost_table,
    plot_compute_cost,
)


# --------------------------------------------------------------------------- #
# _bench (pure micro-benchmark timer)
# --------------------------------------------------------------------------- #


def test_bench_reports_ordered_quartiles_and_positive_median():
    m = _bench(lambda: sum(range(50)), repeats=8)
    assert m["median_ms"] > 0.0
    assert np.isfinite(m["median_ms"])
    assert m["p25_ms"] <= m["median_ms"] <= m["p75_ms"]
    assert m["inner"] >= 1
    assert m["repeats"] == 8
    # throughput is the reciprocal of the median (per-call, in Hz)
    assert m["throughput_hz"] == pytest.approx(1000.0 / m["median_ms"], rel=1e-6)


def test_bench_orders_slow_above_fast():
    fast = _bench(lambda: None, repeats=12)["median_ms"]
    slow = _bench(lambda: sum(range(20000)), repeats=12)["median_ms"]
    # a clearly heavier callable must measure a larger per-call latency
    assert slow > fast


def test_bench_calibrates_inner_above_timer_resolution():
    # a trivial callable needs many inner reps to exceed the min batch time
    m = _bench(lambda: None, repeats=5, min_batch_s=5e-3)
    assert m["inner"] > 1


# --------------------------------------------------------------------------- #
# _headroom
# --------------------------------------------------------------------------- #


def test_headroom_period_over_latency():
    # 1 ms latency against a 1 s period -> 1000x headroom
    assert _headroom(1.0, 1.0) == pytest.approx(1000.0)
    # 2 ms against 60 s -> 30000x
    assert _headroom(2.0, 60.0) == pytest.approx(30000.0)


def test_headroom_zero_latency_is_infinite():
    assert _headroom(0.0, 1.0) == float("inf")


# --------------------------------------------------------------------------- #
# representative reflex state actually exercises the trip path
# --------------------------------------------------------------------------- #


def test_representative_state_triggers_severe_reflex():
    # No torch needed: the closed-loop evidence distiller is pure Python.
    from hydrojev.simulation.closed_loop_eval import ReflexPolicy, distill_evidence

    ev = distill_evidence(_representative_attack_state(), policy=ReflexPolicy())
    # the benchmark must time the real reflex trip, not a nominal no-op step
    assert ev.severe_hydraulic_violation is True
    assert ev.local_physics_alert is True
    assert ev.is_fresh is True


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #


def _fake_ok_result() -> dict:
    def m(median, layer):
        return {
            "median_ms": median, "p25_ms": median * 0.9, "p75_ms": median * 1.1,
            "inner": 100, "repeats": 10, "throughput_hz": 1000.0 / median, "layer": layer,
        }

    return {
        "status": "ok", "seed": 42, "device": "cpu", "n_features": 43,
        "sequence_length": 12, "rat_window": 60, "evaluate_variant": "test_dataset",
        "repeats": 10,
        "environment": {
            "platform": "TestOS", "processor": "TestCPU", "python": "3.11.0",
            "torch": "2.14.0", "torch_threads": 8, "numpy": "2.0.0",
        },
        "model_size": {"surrogate_params": 123456, "ae_params": 7890},
        "measurements": {
            "reflex interlock": m(0.002, LAYER_REFLEX),
            "GRU surrogate forward": m(0.9, LAYER_DETECTOR),
            "CPDZ residual": m(0.01, LAYER_DETECTOR),
            "mock arbiter (local)": m(0.05, LAYER_ARBITER),
            "full cognition step": m(1.4, LAYER_DETECTOR),
        },
        "reference_periods_s": [1.0, 60.0, 3600.0],
        "headroom": {
            "1": {"reflex": 500000.0, "full_cognition_step": 714.0},
            "60": {"reflex": 30000000.0, "full_cognition_step": 42857.0},
            "3600": {"reflex": 1.8e9, "full_cognition_step": 2571428.0},
        },
        "batadal_sampling_note": "BATADAL SCADA telemetry is sampled hourly (3600 s) ...",
    }


def test_format_table_ok_and_not_run():
    assert format_compute_cost_table({"status": "not_run"}) == "not_run"
    txt = format_compute_cost_table(_fake_ok_result())
    assert "real-time feasibility" in txt
    assert "reflex interlock" in txt
    assert "full cognition step" in txt
    assert "Headroom" in txt
    # model size reported, honestly, from the fake params
    assert "123,456 params" in txt
    # the fastest component (reflex) is ordered before the fused cognition step
    assert txt.index("reflex interlock") < txt.index("full cognition step")


def test_plot_ok_and_not_run(tmp_path):
    assert plot_compute_cost({"status": "not_run"}, str(tmp_path / "x.png")) is None
    out = plot_compute_cost(_fake_ok_result(), str(tmp_path / "cost.png"))
    assert out is not None
    assert (tmp_path / "cost.png").exists()


# --------------------------------------------------------------------------- #
# guarded end-to-end smoke test (torch + BATADAL data required)
# --------------------------------------------------------------------------- #


def test_compute_cost_runner_if_available():
    pytest.importorskip("torch")
    from hydrojev.benchmarks.compute_cost import run_compute_cost
    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    result = run_compute_cost(
        load_config(), mode=RunMode.resolve(dry=False, quick=True),
        seed=42, max_epochs=4, evaluate_variant="dataset04", repeats=5,
    )
    if result.get("status") != "ok":
        pytest.skip(f"BATADAL data unavailable: {result.get('reason')}")

    meas = result["measurements"]
    assert "reflex interlock" in meas and "full cognition step" in meas
    for name, m in meas.items():
        assert m["median_ms"] > 0.0 and np.isfinite(m["median_ms"])
        assert m["p25_ms"] <= m["median_ms"] <= m["p75_ms"]
        assert m["throughput_hz"] > 0.0

    # the paper's core latency claim: the safety-critical local reflex decision is
    # cheaper than the full ML cognition step
    assert meas["reflex interlock"]["median_ms"] < meas["full cognition step"]["median_ms"]

    # headroom is positive and the reflex has strictly more headroom than cognition
    for period in result["reference_periods_s"]:
        h = result["headroom"][f"{period:g}"]
        assert h["reflex"] > 0.0 and h["full_cognition_step"] > 0.0
        assert h["reflex"] > h["full_cognition_step"]

    # model size is measured honestly from the live surrogate
    assert result["model_size"]["surrogate_params"] is not None
    assert result["model_size"]["surrogate_params"] > 0

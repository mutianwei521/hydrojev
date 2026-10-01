"""Offline tests for round 5 of the P2 v2 pilot (Jev-screen cascades on the round-4 seed ladders)."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.benchmarks import jev_cascade as jc
from hydrojev.benchmarks import jev_cascade_eval as ce
from hydrojev.benchmarks import jev_pilot_p2v2 as v

A = np.asarray


def test_rule_cascade_keeps_only_benign_screen_verdicts() -> None:
    jev = A(["normal_transient", "sensor_fault", "cyber_attack", "physical_fault", "insufficient_evidence"], dtype=object)
    r1 = A(["cyber_attack", "normal_transient", "normal_transient", "cyber_attack", "sensor_fault"], dtype=object)
    d, keep = jc.rule_cascade(jev, r1)
    assert keep.tolist() == [True, True, False, False, False]
    assert d.tolist() == ["normal_transient", "sensor_fault", "normal_transient", "cyber_attack", "sensor_fault"]


def test_llm_cascade_needs_benign_agreement_with_the_rule() -> None:
    jev = A(["normal_transient", "normal_transient", "cyber_attack", "cyber_attack"], dtype=object)
    r1 = A(["normal_transient", "sensor_fault", "cyber_attack", "normal_transient"], dtype=object)
    rev = A(["physical_fault"] * 4, dtype=object)
    d, keep = jc.llm_cascade(jev, r1, rev)
    assert keep.tolist() == [True, False, False, False]  # an agreed attack verdict is still reviewed
    assert d.tolist() == ["normal_transient"] + ["physical_fault"] * 3


def test_failed_answers_are_never_benign() -> None:
    pm = A([[0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.7, 0.1]])
    d = jc.decisions(pm, [0.25] * 4, A([True, False]))
    assert d.tolist() == ["normal_transient", "insufficient_evidence"]
    assert not jc.rule_cascade(d, A(["cyber_attack"] * 2, dtype=object))[1][1]


def test_calibration_divides_by_the_floored_prior() -> None:
    pm = A([[0.4, 0.3, 0.2, 0.1]])
    assert jc.decisions(pm, [0.8, 0.1, 0.05, 0.05]).tolist() == ["normal_transient"]
    assert np.allclose(jc.calibrated(pm, [0.0, 0.0, 0.0, 0.0]).sum(1), 1.0)


def test_latency_subset_is_stratified() -> None:
    rows = [{"id": f"x_{k:04d}", "label_internal": v.CLASSES[k // 50]} for k in range(200)]
    orig = v.load_split
    v.load_split = lambda key: {"rows": rows}
    try:
        ids = jc.latency_ids("any")
    finally:
        v.load_split = orig
    assert len(ids) == 20 and all(sum(r["id"] in ids for r in rows[50 * c : 50 * c + 50]) == 5 for c in range(4))


def test_noninferiority_and_superiority_p_values() -> None:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 50)
    same = ce.boot(y, y.copy(), y.copy(), rng)
    assert same["delta"] == 0 and same["p_noninferiority"] == 0.0 and same["p_superiority"] == 1.0
    assert ce.wilson(8, 10)[0] < 0.8 < ce.wilson(8, 10)[1]


def test_cascade_sealed_sets_need_the_cascade_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v, "OUTDIR", tmp_path)
    (tmp_path / "design_freeze_r4.json").write_text("{}", encoding="utf-8")  # an older-round freeze does not unlock them
    for split in jc.SETS.values():
        assert v.FREEZE_FILE[split] == jc.CASCADE_FREEZE
        with pytest.raises(RuntimeError):
            v.prepare_split(None, split, jc.FMT)  # type: ignore[arg-type]

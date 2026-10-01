"""Offline tests for round 4 of the P2 v2 pilot (q5 decomposed questions, novel4 family, sealed round-4 splits)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_pilot_r4 as r4


def _ans(c, s, w, a, probs=(0.25, 0.25, 0.25, 0.25)):
    return {"event_probabilities": dict(zip(v.CLASSES, probs)),
            "noul": {"contradiction": c, "single_instrument": s, "honest_operation": w, "asset_fault": a}}


def _state(viol: bool):
    c = {k: {"status": "consistent"} for k in v.PHYSICS_CHECKS}
    if viol:
        c["stopped_station_discharge_vs_tank_head"]["status"] = "violated"
    return {"consistency_checks": c}


def test_questions_extend_q3_with_four_leak_free_nouls() -> None:
    q = r4.build_questions()
    q3 = v.build_questions(3)
    assert {k: q[k] for k in q3} == q3 and set(q) - set(q3) == set(r4.SUBQ)
    assert all(q[k]["type"] == "noul" for k in r4.SUBQ)
    assert p1.scan_request(p1.canonical_bytes({"model": "jev-latest", "state": {}, "questions": q})) == []


def test_tree_routes_each_class() -> None:
    ans = {"cy": _ans(0.95, 0.02, 0.05, 0.1), "se": _ans(0.5, 0.95, 0.05, 0.1),
           "ph": _ans(0.05, 0.02, 0.1, 0.9), "no": _ans(0.05, 0.02, 0.9, 0.1)}
    m = r4.tree_matrix(ans, list(ans))
    assert np.allclose(m.sum(1), 1.0)
    assert [v.CLASSES[k] for k in m.argmax(1)] == ["cyber_attack", "sensor_fault", "physical_fault", "normal_transient"]


def test_adjudication_needs_both_physics_and_contradiction() -> None:
    ans = {"a": _ans(0.9, 0.1, 0.1, 0.9, (0.1, 0.7, 0.1, 0.1)), "b": _ans(0.3, 0.1, 0.1, 0.9, (0.1, 0.7, 0.1, 0.1))}
    flat = [0.25] * 4
    assert r4.adjudicate(ans, ["a", "b"], [_state(True), _state(True)], flat, flat) == ["cyber_attack", "physical_fault"]
    assert r4.adjudicate(ans, ["b"], [_state(False)], flat, flat) == ["physical_fault"]


def test_novel4_is_disjoint_from_every_earlier_family() -> None:
    old = {s for fam in (v.NOVEL, v.NOVEL3) for x in fam.values() for s in x} | {
        "overpump_masked_level", "pump_off_spoofed_running", "burst", "demand_rise", "demand_fall", "operator_switch",
        "stuck", "drift", "offset"}
    assert not r4._NOVEL4_SUBTYPES & old and set(r4.NOVEL4) == set(v.CLASSES)
    spec = SimpleNamespace(name="toy", level_rules=[{"link": "PU", "tank": "T", "open_below": 1.0}], pumps=["PU"])
    for cls in v.CLASSES:
        sc = v.sample_scenario(spec, cls, 7, "novel4")  # dispatched through the registered sampler
        assert sc["subtype"] in r4.NOVEL4[cls] and sc["onset_h"] < sc["window_end_h"]


def test_round4_seed_ladders_never_collide_with_other_splits() -> None:
    def ladder(split: str) -> set[int]:
        _, per_class, base, _ = v.SPLITS[split]
        return {base + 1000 * ci + i + 100_000 * k for ci in range(len(v.CLASSES)) for i in range(per_class) for k in range(41)}

    others = set().union(*(ladder(s) for s in v.SPLITS if s not in r4.R4_SPLITS))
    others |= {2_900_000 + i for i in range(v.N_BALANCE_CAL)} | {3_100_000 + i for i in range(v.N_PHYSICS_CAL)}
    others |= {9_900_000 + 1000 * ci + i + 100_000 * k for ci in range(4) for i in range(10) for k in range(41)}  # smoke
    a, b = (ladder(s) for s in r4.R4_SPLITS)
    assert not (a & others) and not (b & others) and not (a & b)


def test_round4_sealed_splits_need_their_own_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v, "OUTDIR", tmp_path)
    (tmp_path / "design_freeze_r3.json").write_text("{}", encoding="utf-8")
    for split in r4.R4_SPLITS:
        assert split in v.SEALED
        with pytest.raises(RuntimeError):
            v.prepare_split(None, split, v.STATE_FORMAT_PHYSICS)  # type: ignore[arg-type]

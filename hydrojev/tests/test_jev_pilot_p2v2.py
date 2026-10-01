"""Offline tests for the Jev pilot P2 v2 contract (no network calls, no split preparation).

Simulation smoke tests draw from a seed range (9_000_000+) disjoint from every pilot split and assert only that the code
runs and the state is leak-free; they never inspect the evidence against a label.
"""

from __future__ import annotations

import json
import re

import pytest

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.methods.hydrojev_detector import HydroJEVSchemaError

SUBTYPE_WORDS = ("burst", "leak", "stuck", "drift", "spoof", "replay", "closure", "gain error", "gain_error", "pattern shift", "pattern_shift", "operator_switch", "demand_rise")


@pytest.mark.parametrize("revision", sorted(v.INSTRUCTIONS))
def test_questions_name_no_subtype(revision: int) -> None:
    text = json.dumps(v.build_questions(revision)).lower()
    assert set(v.build_questions(revision)["event_type"]["criteria"]) == set(v.EVENT_TYPES)
    assert [w for w in SUBTYPE_WORDS if re.search(rf"\b{w}\b", text)] == []
    assert len(v.questions_sha256(revision)) == 64


def test_system_description_is_leak_free() -> None:
    assert v.scan_state({"system": v.SYSTEM_DESCRIPTION}) == []


@pytest.mark.parametrize("token", ["replay", "closure", "gain_error", "pattern_shift", "spoofed", "cyber_attack"])
def test_scan_state_flags_label_tokens(token: str) -> None:
    assert v.scan_state({"note": f"x {token} y"})


def test_build_request_rejects_ground_truth() -> None:
    with pytest.raises(HydroJEVSchemaError):
        v.build_request({"label": "cyber_attack"}, revision=3)


def test_split_keys_and_seed_ranges_are_disjoint() -> None:
    assert v.split_key("dev", 1) == "dev" and v.split_key("eval_id", 2) == "eval_id_s2"
    assert v.train_key("eval_novel_s2") == "train_s2" and v.train_key("dev") == "train"
    ranges = sorted((base, base + 4000) for _, _, base, _ in v.SPLITS.values())
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))


def test_sealed_split_refused_without_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v, "OUTDIR", tmp_path)
    for split in v.SEALED:
        with pytest.raises(RuntimeError):
            v.prepare_split(None, split)  # type: ignore[arg-type]


def test_balance_check_hides_detail_when_consistent() -> None:
    import numpy as np

    bcal = {"stations": {"R": ["PA", "PB"]}, "median": {"R": 0.0}, "threshold_abs_ls": {"R": 3.0}}
    ch = {"Fsrc_R": np.full(40, 10.0), "F_PA": np.full(40, 5.0), "F_PB": np.full(40, 5.0)}
    ok = v.balance_check({"reported": ch}, 30, bcal)
    assert ok["status"] == "consistent" and "detail" not in ok
    ch["F_PB"] = np.full(40, 0.0)
    bad = v.balance_check({"reported": ch}, 30, bcal)
    assert bad["status"] == "violated" and bad["statistic"] > 1.0 and "detail" in bad


def test_novel_sampling_is_deterministic() -> None:
    class S:
        name = "X"
        level_rules = [{"link": "P1", "tank": "T1", "close_above": 4.0, "open_below": 1.0}]
        pumps = ["P1"]

    for cls in v.CLASSES:
        a, b = v.sample_scenario(S, cls, 9_000_001, "novel"), v.sample_scenario(S, cls, 9_000_001, "novel")
        assert a == b and a["subtype"] in v.NOVEL[cls]
        assert a["window_end_h"] - p2.SYNC_LAG_H < a["onset_h"] < a["window_end_h"] + 1


@pytest.fixture(scope="module")
def ctown():
    cal_path, bcal_path = v.OUTDIR / "calibration_ctown.json", v.OUTDIR / "balance_calibration_ctown.json"
    if not (cal_path.exists() and bcal_path.exists()):
        pytest.skip("CTOWN calibration caches not present")
    from hydrojev.config import load_config

    cfg = load_config()
    return cfg, p2.net_spec(cfg, "CTOWN"), json.loads(cal_path.read_text(encoding="utf-8")), json.loads(bcal_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("subtype", [s for subs in v.NOVEL.values() for s in subs])
def test_novel_subtype_simulates_to_leak_free_state(ctown, subtype: str) -> None:
    cfg, spec, cal, bcal = ctown
    cls = next(c for c, subs in v.NOVEL.items() if subtype in subs)
    for k in range(60):
        sc = v.sample_scenario(spec, cls, 9_000_000 + 1000 * v.CLASSES.index(cls) + k, "novel")
        if sc["subtype"] != subtype:
            continue
        try:
            sim = v.simulate(cfg, spec, sc, cal["sigma"])
        except ValueError:
            continue
        state = v.build_state(spec, sim, sc["window_end_h"], cal, bcal)
        assert v.scan_state(state) == []
        assert state["schema_version"] == v.SCHEMA_VERSION
        for check in state["consistency_checks"].values():
            if check.get("status") == "consistent":
                assert "detail" not in check
        x, _ = v.features(spec, sim, sc["window_end_h"], cal, bcal)
        assert x.ndim == 1 and bool((x == x).all())
        v.build_request(state, revision=3)
        return
    pytest.fail(f"{subtype} never realised in 60 smoke seeds")


def test_round2_seed_ladders_never_collide_with_other_splits() -> None:
    def ladder(split: str) -> set[int]:
        _, per_class, base, _ = v.SPLITS[split]
        return {base + 1000 * ci + i + 100_000 * k for ci in range(len(v.CLASSES)) for i in range(per_class) for k in range(41)}

    others = set().union(*(ladder(s) for s in v.SPLITS if not s.endswith("_r2")))
    others |= {2_900_000 + i for i in range(v.N_BALANCE_CAL)}
    a, b = ladder("eval_id_r2"), ladder("eval_novel_r2")
    assert not (a & others) and not (b & others) and not (a & b)


def test_round2_sealed_split_needs_its_own_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(v, "OUTDIR", tmp_path)
    (tmp_path / "design_freeze.json").write_text("{}", encoding="utf-8")
    for split in ("eval_id_r2", "eval_novel_r2"):
        with pytest.raises(RuntimeError):
            v.prepare_split(None, split)  # type: ignore[arg-type]

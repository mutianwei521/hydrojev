"""Round 6 of the Jev pilot P2 v2: the round-5 screen and cascades, unchanged, on the two other networks of this project.

Rounds 1-5 simulate every window on C-Town. Round 6 asks whether the results transfer to the other networks used by
earlier versions of HydroJEV: EPANET Net3 (3 tanks, 2 pumps, a level-controlled pressure-zone bypass) and EPANET Net1
(1 tank, 1 pump). Everything that was fitted or chosen is frozen from C-Town and transferred zero-shot:

* Jev: q3 question, state format 2, the C-Town dev_s2 prior (round-5 freeze);
* reviewers glm-5.2:cloud and deepseek-v4-pro:cloud with their C-Town dev_s2 priors;
* R1 (hand-written tree over network-normalised checks) and R2 (logistic regression fitted on C-Town train_s2);
* the rule and LLM cascades of jev_cascade.

Only what is a property of a network is recomputed on it, with the frozen code: the channel set (p2.net_spec), the
residual and check calibration (p2.calibrate) and the station balance calibration (v.balance_calibration). Two
network-dependent choices are fixed here, before any round-6 window exists:

* Net1 is the canonical EPANET Net1 of the WNTR library (pump 9 started/stopped by the level of tank 2). The copy in
  refCode/FDI-WDNs has an empty [CONTROLS] section, so it cannot host control-logic attacks or operator switches.
* Monitored junctions: p2.net_spec picks 8 by farthest-point sampling. Net1 has 9 junctions, so 8 monitors would leave
  no unmonitored junction for a leak. The rule is k = min(8, floor(pool / 2)), pool = junctions with median pressure
  > 15 m that are not pump ends; the greedy choice is prefix-consistent, so C-Town and Net3 keep k = 8.

* Redraw budget: the frozen prepare loop redraws a window up to 40 times (+100_000 per attempt) when it is not
  realisable or when no check flags it. On Net1 (5 pressure sensors, one tank, event-free system-demand sigma ~4.5 % of
  demand) a v1 burst of 6-15 % of demand trips a check in only ~3 % of draws (realisability probe), so 40 redraws cannot
  fill 50 windows. Round 6 uses MAX_ATTEMPTS = 400 on both networks; the ladder is the same, so any window the frozen
  loop realises within 40 redraws is identical, and the per-class redraw counts (the gate pass rate) are reported.

Subtypes a network cannot host (e.g. masked_duty_swap needs two pumps feeding one tank) are redrawn by the frozen
prepare logic, as on C-Town; the realised subtype mix is reported. Seed ladders (10_0x0_000) are disjoint from every
earlier ladder. Native in-network R2 (fitted on a v1 training split of the same network) is a reported reference
only: it needs labelled data from the target network, which is what the transfer question removes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks import jev_cascade as jc  # registers the round-4 splits and the novel4 family
from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_pilot_r4 as r4

FMT = jc.FMT
REVISION = jc.REVISION
R6_FREEZE = "design_freeze_r6.json"
NETWORKS = ("Net3", "Net1")
MAX_MONITORS = p2.N_MONITORS
CAL_SEED = {"Net3": 950_000, "Net1": 960_000}  # Net3 as in p2.CAL_SEED; Net1 was never calibrated before
R6_SPLITS = {  # name: (network, per class, seed base, family)
    "eval_id_net3_r6": ("Net3", 50, 10_010_000, "v1"),
    "eval_novel4_net3_r6": ("Net3", 50, 10_020_000, "novel4"),
    "eval_id_net1_r6": ("Net1", 50, 10_030_000, "v1"),
    "eval_novel4_net1_r6": ("Net1", 50, 10_040_000, "novel4"),
}
R6_TRAIN = {  # reference only (native R2); not sealed, never sent to Jev or an LLM
    "train_net3_r6": ("Net3", 50, 10_050_000, "v1"),
    "train_net1_r6": ("Net1", 50, 10_060_000, "v1"),
}
SETS = {"N3-V1": "eval_id_net3_r6", "N3-N4": "eval_novel4_net3_r6", "N1-V1": "eval_id_net1_r6", "N1-N4": "eval_novel4_net1_r6"}
KEYS = {name: v.split_key(split, FMT) for name, split in SETS.items()}
TRAIN_OF = {"Net3": "train_net3_r6", "Net1": "train_net1_r6"}
PROBE_BASE = 11_070_000  # realisability probe only: simulation and gates, no method output is computed
MAX_ATTEMPTS = 400  # frozen loop: 40; see the module docstring

_orig_load = p2._load_model


def _wntr_library(name: str) -> Path:
    import wntr

    return Path(wntr.__file__).parent / "library" / "networks" / f"{name}.inp"


def _load_model(config: Any, network: str) -> Any:
    if network == "Net1":
        import wntr

        return wntr.network.WaterNetworkModel(str(_wntr_library("Net1")))
    return _orig_load(config, network)


def register() -> None:
    p2._load_model = _load_model
    p2.CAL_SEED.update({k: s for k, s in CAL_SEED.items() if k not in p2.CAL_SEED})
    v.SPLITS.update(R6_SPLITS)
    v.SPLITS.update(R6_TRAIN)
    v.SEALED = tuple(dict.fromkeys(v.SEALED + tuple(R6_SPLITS)))
    v.FREEZE_FILE.update({s: R6_FREEZE for s in R6_SPLITS})


register()


def net_spec(config: Any, network: str) -> p2.NetSpec:
    """p2.net_spec with the monitor count k = min(8, floor(pool / 2)) (see the module docstring)."""

    saved = p2.N_MONITORS
    try:
        p2.N_MONITORS = 10**6
        full = p2.net_spec(config, network)
        pumps_ends = {n for ends in full.pump_ends.values() for n in ends}
        pool = [n for n in full.pressure_nodes if n not in pumps_ends]
        p2.N_MONITORS = min(MAX_MONITORS, len(pool) // 2)
        spec = p2.net_spec(config, network)
    finally:
        p2.N_MONITORS = saved
    return spec


def spec_cal(config: Any, network: str) -> tuple[p2.NetSpec, dict[str, Any]]:
    spec = net_spec(config, network)
    cache = v.OUTDIR / f"calibration_{network.lower()}.json"
    if cache.exists():
        return spec, json.loads(cache.read_text(encoding="utf-8"))
    cal = p2.calibrate(config, spec)
    cache.write_text(json.dumps(p2._safe(cal)), encoding="utf-8")
    return spec, cal


def _realise(config: Any, spec: p2.NetSpec, cal: dict[str, Any], cls: str, seed: int, family: str) -> tuple[dict, dict, dict]:
    """The redraw loop of v.prepare_split with the round-6 budget MAX_ATTEMPTS (same seed ladder)."""

    sc = v.sample_scenario(spec, cls, seed, family)
    unrealised = gated = 0
    for attempt in range(MAX_ATTEMPTS):
        try:
            sim = v.simulate(config, spec, sc, cal["sigma"])
        except ValueError:
            unrealised += 1
            sc = {**v.sample_scenario(spec, cls, seed + 100_000 * (attempt + 1), family), "seed_origin": seed}
            continue
        if not p2.evaluate_window(spec, sim, sc["window_end_h"], cal)["violated"]:
            gated += 1
            sc = {**v.sample_scenario(spec, cls, seed + 100_000 * (attempt + 1), family), "seed_origin": seed}
            continue
        return sc, sim, {"not_realisable": unrealised, "no_violated_check": gated}
    raise RuntimeError(f"could not realise {spec.name} {cls} seed {seed}")


def prepare_split(config: Any, split: str) -> dict[str, Any]:
    """v.prepare_split for a round-6 split, with the network's own spec and calibration."""

    key = v.split_key(split, FMT)
    target = v.OUTDIR / "splits" / f"{key}.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; split is frozen")
    if split in v.SEALED and not (v.OUTDIR / v.FREEZE_FILE[split]).exists():
        raise RuntimeError(f"sealed splits are generated only after {v.FREEZE_FILE[split]} exists")
    net, per_class, base, family = v.SPLITS[split]
    spec, cal = spec_cal(config, net)
    bcal = v.balance_calibration(config, spec)
    rows = []
    reqdir = v.OUTDIR / "requests" / key
    reqdir.mkdir(parents=True, exist_ok=True)
    for ci, cls in enumerate(v.CLASSES):
        for i in range(per_class):
            seed = base + 1000 * ci + i
            sc, sim, redraws = _realise(config, spec, cal, cls, seed, family)
            we = sc["window_end_h"]
            state = v.build_state(spec, sim, we, cal, bcal, None)
            leaks = v.scan_state(state)
            if leaks:
                raise v.HydroJEVSchemaError(f"state leak tokens {leaks} in {split} seed {seed}")
            x, ev = v.features(spec, sim, we, cal, bcal, None)
            rid = f"{split}_{len(rows):04d}"
            (reqdir / f"{rid}.state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            rows.append({"id": rid, "split": split, "label_internal": cls, "scenario_internal": p2._safe({**sc, **sim["resolved"]}),
                         "redraws": redraws, "r1": p2.r1_classify(ev), "violated": state["violated_checks"], "x": [float(u) for u in x]})
            print(split, rid, cls, sc["subtype"], ev["violated"], flush=True)
    out = {"split": split, "state_format": FMT, "network": net, "family": family, "created_utc": v._now(), "rows": rows,
           "monitors": len(spec.pressure_nodes)}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(p2._safe(out), ensure_ascii=False), encoding="utf-8")
    return out


def describe(config: Any, network: str) -> dict[str, Any]:
    spec = net_spec(config, network)
    return {"tanks": spec.tanks, "pumps": spec.pumps, "actuated": spec.actuated, "control_configuration": p2.control_configuration(spec),
            "pressure_nodes": spec.pressure_nodes, "sources": list(spec.sources), "stations": v.stations(config, spec),
            "fed_tanks": v.fed_tanks(config, spec)}


def probe(config: Any, n_per_class: int = 8) -> dict[str, Any]:
    """Realisability probe on non-sealed seeds: which subtypes each network realises and how often the gates redraw.
    Writes no state and computes no method output (no R1, R2, Jev or LLM)."""

    x_len_ctown = len(v.load_split("train_s2")["rows"][0]["x"])
    out: dict[str, Any] = {"x_len_ctown": x_len_ctown}
    for net in NETWORKS:
        spec, cal = spec_cal(config, net)
        bcal = v.balance_calibration(config, spec)
        out[net] = {"spec": describe(config, net), "calibration_thresholds": cal["thresholds"], "balance": bcal}
        for fam_i, family in enumerate(("v1", "novel4")):
            res = {}
            for ci, cls in enumerate(v.CLASSES):
                subs, red, leaks, xl = [], {"not_realisable": 0, "no_violated_check": 0}, [], set()
                for i in range(n_per_class):
                    seed = PROBE_BASE + 10_000 * fam_i + 1000 * ci + i
                    sc, sim, r = _realise(config, spec, cal, cls, seed, family)
                    subs.append(sc["subtype"])
                    for k in red:
                        red[k] += r[k]
                    state = v.build_state(spec, sim, sc["window_end_h"], cal, bcal, None)
                    leaks += v.scan_state(state)
                    xl.add(len(v.features(spec, sim, sc["window_end_h"], cal, bcal, None)[0]))
                res[cls] = {"subtypes": {s: subs.count(s) for s in sorted(set(subs))}, "redraws": red, "leaks": sorted(set(leaks)), "x_len": sorted(xl)}
                print(net, family, cls, res[cls], flush=True)
            out[net][family] = res
    return out


def run_jev(split_key: str) -> None:
    p1.PILOT_CALL_CAP = r4.UNLIMITED_CAP
    v.run(split_key, REVISION)


def code_sha256() -> str:
    return hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def _main(argv: Sequence[str] | None = None) -> int:
    from hydrojev.config import load_config

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("probe", "describe", "prepare", "train", "jev", "llm"))
    ap.add_argument("--concurrency", type=int, default=20)
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.cmd == "describe":
        print(json.dumps({n: describe(cfg, n) for n in NETWORKS}, indent=1, default=str))
    elif a.cmd == "probe":
        res = probe(cfg)
        (v.OUTDIR / "r6_probe.json").write_text(json.dumps(p2._safe(res), indent=1, default=str), encoding="utf-8")
    elif a.cmd == "train":
        for s in R6_TRAIN:
            prepare_split(cfg, s)
    elif a.cmd == "prepare":
        for s in R6_SPLITS:
            prepare_split(cfg, s)
    elif a.cmd == "jev":
        for key in KEYS.values():
            run_jev(key)
    else:
        jc.run_llms(list(KEYS.values()), a.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

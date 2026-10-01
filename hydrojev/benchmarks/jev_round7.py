"""Round 7 of the Jev pilot P2 v2: round 6 on EPANET Net1 again, with the event generator's pattern-timestep error fixed.

Diagnosis (after the round-6 one-shot evaluation, on burned round-6 data and development seeds only): the WNTR library
Net1.inp has a 2-h pattern timestep (C-Town and Net3: 1 h). The frozen generator indexes pattern multipliers by hour
(p2._noisy_model: mult[t0:] *= f for demand rise/fall; the burst demand pattern [0]*t0 + [1]*...; v._shift_patterns),
so on Net1 every pattern-driven event started at 2*t0 h. With onsets of 24-48 h and 6-h windows ending 2-6 h after
onset, no round-6 Net1 burst, demand rise/fall or demand-pattern shift was active inside its window; the accepted
windows were chance check trips (which is why bursts needed ~23 redraws each). Control-driven events (cyber attacks,
operator switches, pipe closures, pump gain errors) and sensor faults were unaffected. The round-6 Net1 result is
therefore reported as a generator error, and its sealed sets are not re-evaluated.

Round 7 changes exactly one thing: the network 'Net1h' is WNTR Net1 with every pattern resampled to 1 h (each 2-h
multiplier repeated twice, pattern_timestep = 3600). Event-free hydraulics are identical to Net1 (checked by
equivalence(), recorded in the probe). Everything else is round 6 unchanged: the frozen generator, gates and redraw
budget (MAX_ATTEMPTS 400), the monitor rule (k = 4 + pump end), the calibration code (run afresh on Net1h; caches
calibration_net1h.json etc., the round-6 Net1 caches stay untouched), Jev and reviewer priors from C-Town, R1, R2 fitted
on C-Town train_s2, the cascades and the evaluation code of round 6. Seed ladders (13_0x3_500) are disjoint from every
earlier ladder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks import jev_cascade as jc
from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_round6 as r6  # registers round 6 (Net1 loader, CAL_SEED, splits)

FMT = jc.FMT
R7_FREEZE = "design_freeze_r7.json"
NETWORK = "Net1h"
CAL_SEED = {NETWORK: 960_000}  # the round-6 Net1 calibration seeds: same noise draws, hourly patterns
R7_SPLITS = {
    "eval_id_net1h_r7": (NETWORK, 50, 13_033_500, "v1"),
    "eval_novel4_net1h_r7": (NETWORK, 50, 13_043_500, "novel4"),
}
R7_TRAIN = {"train_net1h_r7": (NETWORK, 50, 13_063_500, "v1")}  # native R2 reference only; never sent to Jev or an LLM
SETS = {"N1h-V1": "eval_id_net1h_r7", "N1h-N4": "eval_novel4_net1h_r7"}
KEYS = {name: v.split_key(split, FMT) for name, split in SETS.items()}
TRAIN_OF = {NETWORK: "train_net1h_r7"}
PROBE_BASE = 13_073_500

_r6_load = p2._load_model  # r6._load_model after r6.register()


def hourly(wn: Any) -> Any:
    t = wn.options.time
    if t.pattern_timestep != 3600:
        r = int(t.pattern_timestep // 3600)
        assert r * 3600 == t.pattern_timestep
        for pn in wn.pattern_name_list:
            pat = wn.get_pattern(pn)
            pat.multipliers = list(np.repeat(np.asarray(pat.multipliers, dtype=float), r))
        t.pattern_timestep = 3600
    return wn


def _load_model(config: Any, network: str) -> Any:
    if network == NETWORK:
        return hourly(_r6_load(config, "Net1"))
    return _r6_load(config, network)


def register() -> None:
    p2._load_model = _load_model
    p2.CAL_SEED.update({k: s for k, s in CAL_SEED.items() if k not in p2.CAL_SEED})
    v.SPLITS.update(R7_SPLITS)
    v.SPLITS.update(R7_TRAIN)
    v.SEALED = tuple(dict.fromkeys(v.SEALED + tuple(R7_SPLITS)))
    v.FREEZE_FILE.update({s: R7_FREEZE for s in R7_SPLITS})
    r6.KEYS.update(KEYS)  # lets the frozen r6 evaluate() find the round-7 sets and native train split
    r6.TRAIN_OF.update(TRAIN_OF)


register()


def equivalence(config: Any) -> dict[str, float]:
    """Event-free hydraulics of Net1h vs Net1 (both from the WNTR library file)."""

    a = p2._run(_r6_load(config, "Net1"))
    b = p2._run(_load_model(config, NETWORK))
    return {"max_abs_dpressure_m": float((a.node["pressure"] - b.node["pressure"]).abs().max().max()),
            "max_abs_dhead_m": float((a.node["head"] - b.node["head"]).abs().max().max()),
            "max_abs_dflow": float((a.link["flowrate"] - b.link["flowrate"]).abs().max().max())}


def probe(config: Any, n_per_class: int = 8) -> dict[str, Any]:
    """Round-6 probe on Net1h (gates only; no method output), plus the equivalence check."""

    saved_nets, saved_base = r6.NETWORKS, r6.PROBE_BASE
    try:
        r6.NETWORKS, r6.PROBE_BASE = (NETWORK,), PROBE_BASE
        out = r6.probe(config, n_per_class)
    finally:
        r6.NETWORKS, r6.PROBE_BASE = saved_nets, saved_base
    out["equivalence_event_free_vs_net1"] = equivalence(config)
    return out


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
        print(json.dumps({NETWORK: r6.describe(cfg, NETWORK), "equivalence": equivalence(cfg)}, indent=1, default=str))
    elif a.cmd == "probe":
        target = v.OUTDIR / "r7_probe.json"
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(p2._safe(probe(cfg)), indent=1, default=str), encoding="utf-8")
    elif a.cmd == "train":
        for s in R7_TRAIN:
            r6.prepare_split(cfg, s)
    elif a.cmd == "prepare":
        for s in R7_SPLITS:
            r6.prepare_split(cfg, s)
    elif a.cmd == "jev":
        for key in KEYS.values():
            r6.run_jev(key)
    else:
        jc.run_llms(list(KEYS.values()), a.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

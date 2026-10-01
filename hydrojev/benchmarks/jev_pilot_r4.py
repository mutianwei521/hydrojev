"""Round 4 of the Jev pilot P2 v2: decomposed evidence questions (q5) on state format 3.

Round 3 showed that Jev reads most concealed attacks as asset faults (it does not weigh the format-3 physics checks),
while the code-side physics override (R1b / override_jev) also fires on worn pumps and faulty pressure sensors. Round 4
keeps the round-3 choice question and adds, in the same request, four independent yes/no (noul) questions that each
ask one step of the reasoning the choice question describes; a fixed, training-free tree combines them. Design is done
on burned data only (see PILOT_PROTOCOL_v2.md, amendment v2.5). ``jev_pilot_p2v2`` and ``jev_pilot_p1`` are not
modified (their hashes are frozen by earlier rounds); the call cap is lifted at run time because the author authorised
unlimited calls on 2026-09-26 ("调用上限可以无限").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v

REVISION = 5
UNLIMITED_CAP = 10**9
CLASSES = v.CLASSES
SUBQ = {
    "contradiction": (
        "Is at least one reported actuator status or reported reading contradicted by other, independent measurements of "
        "the network (meters of other assets, pressures, a pump's own curve, a tank's own inflow meter, mass balance), so "
        "that what is reported cannot be what the network is physically doing? A disagreement with the twin alone is no "
        "such contradiction: the twin is a model with nominal consumption and nominal assets, not a measurement."
    ),
    "single_instrument": (
        "Is all of the abnormal evidence explained by one measuring instrument misreporting, that is, one channel and only "
        "the checks computed from that channel disagree, while every channel physically related to it, the actuators and "
        "the hydraulics are normal?"
    ),
    "honest_operation": (
        "Is the network, as measured, operating honestly but differently from the nominal model: consumption higher, "
        "lower, earlier or later acting on the whole network coherently, or an actuator switched by an operator, with every "
        "reported actuator state confirmed by the measurements?"
    ),
    "asset_fault": (
        "Is one physical asset (a pipe or fitting leaking or broken, a pump delivering less than its nominal curve, a valve "
        "stuck or partly closed) behaving differently from the model at one location, while the independent meters agree "
        "with one another and with the reported actuator states?"
    ),
}


def build_questions() -> dict[str, dict[str, Any]]:
    q = v.build_questions(3)
    q.update({k: {"type": "noul", "instructions": t} for k, t in SUBQ.items()})
    return q


def questions_sha256() -> str:
    return hashlib.sha256(json.dumps(build_questions(), sort_keys=True).encode()).hexdigest()


def request_path(split: str, rid: str) -> Path:
    path = v.OUTDIR / "requests" / split / f"{rid}.r{REVISION}.request.json"
    if not path.exists():
        state = json.loads((v.OUTDIR / "requests" / split / f"{rid}.state.json").read_text(encoding="utf-8"))
        v._assert_no_ground_truth(state)
        req = {"model": "jev-latest", "state": v._json_safe(dict(state), path="state"), "questions": build_questions()}
        body = p1.canonical_bytes(req)
        problems = p1.scan_request(body)
        if problems:
            raise v.HydroJEVSchemaError(f"request scan {problems} for {rid}")
        path.write_bytes(body)
    return path


def parse(body: Mapping[str, Any]) -> dict[str, Any]:
    d = p2.parse_response(body)
    nouls = {}
    for k in SUBQ:
        a = body["answers"].get(k)
        if not isinstance(a, Mapping) or a.get("type") != "noul":
            raise v.HydroJEVSchemaError(f"{k} answer must have type 'noul'")
        nouls[k] = float(np.clip(float(a["noul"]), 0.0, 1.0))
    return {"model": d.model, "usage": d.usage, "event_type": d.event_type, "event_probabilities": d.probabilities, "noul": nouls}


def stage_name(split: str, tag: str = "") -> str:
    return f"p2v2_{split}_q{REVISION}{tag}"


def run(split: str, ids: Sequence[str] | None = None) -> None:
    p1.PILOT_CALL_CAP = UNLIMITED_CAP
    st = stage_name(split)
    for r in v.load_split(split)["rows"]:
        if ids is not None and r["id"] not in ids:
            continue
        e = p1.call_live(v.LEDGER_DIR, request_path(split, r["id"]), stage=st, tag=f"{st}_{r['id']}", summarize=parse)
        print(r["id"], e.get("status"), e.get("event_type"), flush=True)


def load(split: str) -> dict[str, dict[str, Any]]:
    st = stage_name(split)
    out = {}
    for f in (v.LEDGER_DIR / "responses" / st).glob("*.response.json"):
        out[f.name.split(".")[0].replace(f"{st}_", "")] = parse(json.loads(f.read_text(encoding="utf-8-sig"))["response"])
    return out


def choice_matrix(ans: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> np.ndarray:
    return np.asarray([[max(float(ans[i]["event_probabilities"].get(c, 0.0)), 0.0) for c in CLASSES] for i in ids])


def tree_matrix(ans: Mapping[str, Mapping[str, Any]], ids: Sequence[str]) -> np.ndarray:
    """Fixed tree over the four nouls: contradiction and not one instrument -> cyber; one instrument -> sensor;
    otherwise honest operation versus asset fault in proportion to their nouls."""

    out = []
    for i in ids:
        n = ans[i]["noul"]
        c, s, w, a = n["contradiction"], n["single_instrument"], n["honest_operation"], n["asset_fault"]
        rest = (1 - s) * (1 - c)
        wa = max(w + a, 1e-3)
        out.append([c * (1 - s), rest * a / wa, rest * w / wa, s])  # CLASSES order: cyber, physical, normal, sensor
    m = np.maximum(np.asarray(out), 1e-3)
    return m / m.sum(1, keepdims=True)


def calibrate(m: np.ndarray, prior: Sequence[float]) -> np.ndarray:
    """Divide by a label-free prior (mean answer over unlabeled windows) and renormalise."""

    q = np.maximum(m, 0.005) / np.maximum(np.asarray(prior, dtype=float), 0.01)
    return q / q.sum(1, keepdims=True)


def adjudicate(ans: Mapping[str, Mapping[str, Any]], ids: Sequence[str], states: Sequence[Mapping[str, Any]],
               choice_prior: Sequence[float], tree_prior: Sequence[float]) -> list[str]:
    """Round-4 decision rule: geometric mean of the calibrated choice and the calibrated tree; a violated physics check
    turns the decision to cyber_attack only when Jev also answers that the report is contradicted (noul > 0.5)."""

    fused = np.exp(0.5 * np.log(calibrate(choice_matrix(ans, ids), choice_prior)) + 0.5 * np.log(calibrate(tree_matrix(ans, ids), tree_prior)))
    out = []
    for i, s, k in zip(ids, states, fused.argmax(1)):
        out.append("cyber_attack" if v.physics_override(s) and ans[i]["noul"]["contradiction"] > 0.5 else CLASSES[k])
    return out


# --------------------------------------------------------------------------- #
# Round-4 sealed splits and the subtype family "novel4" (written before any of its windows existed; never used for
# design; parameters are physical defaults). Registered in jev_pilot_p2v2 at run time so that its frozen file and its
# prepare_split logic (realisability redraws, triage gate, leak scan) stay unchanged.
# --------------------------------------------------------------------------- #

R4_SPLITS = {"eval_id_r4": ("CTOWN", 50, 5_090_000, "v1"), "eval_novel4_r4": ("CTOWN", 50, 5_095_000, "novel4")}
R4_FREEZE = "design_freeze_r4.json"
NOVEL4 = {  # disjoint from v1, novel and novel3; each class pairs a subtype with a hard "mirror" of another class
    "cyber_attack": ("level_spoof_starve", "masked_duty_swap"),
    "physical_fault": ("growing_leak", "pump_trip"),
    "normal_transient": ("gradual_demand_ramp", "erratic_demand"),
    "sensor_fault": ("noise_burst", "lagged_channel"),
}
_NOVEL4_SUBTYPES = {s for x in NOVEL4.values() for s in x}
_orig_sample, _orig_simulate = v.sample_scenario, v.simulate


def sample_novel4(spec: p2.NetSpec, cls: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t0 = int(rng.integers(p2.ONSET_RANGE_H[0], p2.ONSET_RANGE_H[1] + 1))
    we = t0 + int(rng.integers(p2.LAG_RANGE_H[0], p2.LAG_RANGE_H[1] + 1))
    sc: dict[str, Any] = {"seed": seed, "network": spec.name, "cls": cls, "onset_h": t0, "window_end_h": we}
    sub = sc["subtype"] = str(rng.choice(list(NOVEL4[cls])))
    if sub == "level_spoof_starve":  # the level a pump's start controller reads is held high: the pump never starts
        startable = [r for r in spec.level_rules if "open_below" in r and r["link"] in spec.pumps]
        r = startable[int(rng.integers(len(startable)))]
        sc.update(pump=r["link"], tank=r["tank"])
    elif sub == "growing_leak":  # a leak that grows linearly to its final size over a few hours
        sc["leak_fraction"] = float(rng.uniform(0.08, 0.15))
        sc["ramp_h"] = int(rng.integers(3, 7))
    elif sub == "gradual_demand_ramp":  # network-wide consumption drifting up or down over a few hours
        sc["factor"] = float(rng.choice([rng.uniform(1.20, 1.40), rng.uniform(0.65, 0.80)]))
        sc["ramp_h"] = int(rng.integers(3, 7))
    elif sub == "erratic_demand":  # network-wide consumption that jumps from hour to hour
        sc["low_high"] = [0.75, 1.30]
    elif sub in ("noise_burst", "lagged_channel"):
        sc["kind"] = str(rng.choice(["L", "P", "F"]))
        sc["magnitude_sigma"] = float(rng.uniform(4.0, 8.0))
        sc["lag_h"] = int(rng.integers(2, 5))
    sc["pick"] = float(rng.random())
    return sc


def simulate_novel4(config: Any, spec: p2.NetSpec, sc: dict[str, Any], sigma: Mapping[str, float] | None) -> dict[str, Any]:
    sub = sc["subtype"]
    rng = np.random.default_rng(sc["seed"] + 1)
    t0, we = sc["onset_h"], sc["window_end_h"]
    state_seed = rng.integers(1 << 31)
    wn = p2._noisy_model(config, spec, np.random.default_rng(state_seed), we)
    base_res = p2._run(wn)
    base = p2._true_channels(base_res, wn, spec)
    resolved: dict[str, Any] = {}
    win = p2._win(we)

    def pick(options: Sequence[Any]) -> Any:
        return sorted(options)[min(int(sc["pick"] * len(options)), len(options) - 1)]

    def running(p: str) -> bool:
        return base[f"S_{p}"][t0 - 1] > 0.5 and base[f"F_{p}"][t0 - 1] > 1.0

    def scale_patterns(factor_of_hour) -> None:
        for pname in list(wn.pattern_name_list):
            pat = wn.get_pattern(pname)
            m = np.asarray(pat.multipliers, dtype=float)
            if len(m) < 2 or pname.startswith(("atk", "evt", "op")):
                continue
            for h in range(t0, len(m)):
                m[h] *= factor_of_hour(h)
            pat.multipliers = list(m)

    true = base
    if sub not in ("noise_burst", "lagged_channel"):
        wn = p2._noisy_model(config, spec, np.random.default_rng(state_seed), we)
        if sub == "level_spoof_starve":
            p2._override(wn, sc["pump"], t0, None, 0, "atk")
            resolved.update(pump=sc["pump"], tank=sc["tank"])
        elif sub == "masked_duty_swap":
            feed = v.fed_tanks(config, spec)
            pairs = [(a, b) for a in spec.pumps for b in spec.pumps
                     if a != b and feed.get(a) is not None and feed.get(a) == feed.get(b) and running(a) and base[f"S_{b}"][t0 - 1] < 0.5]
            if not pairs:
                raise ValueError("no station with a running and a standby pump")
            a, b = pick(pairs)
            p2._override(wn, a, t0, None, 0, "atk_a")
            p2._override(wn, b, t0, None, 1, "atk_b")
            resolved.update(stopped=a, started=b)
        elif sub == "growing_leak":
            pres = base_res.node["pressure"].median()
            pool = sorted(j for j in wn.junction_name_list if pres[j] > 20.0 and j not in spec.pressure_nodes)
            j = pick(pool)
            total = float(base_res.node["demand"][wn.junction_name_list].sum(axis=1).mean())
            ramp = [0.0] * t0 + [min((h + 1) / sc["ramp_h"], 1.0) for h in range(we - t0 + 3)]
            wn.add_pattern("evt_ramp", ramp)
            wn.get_node(j).add_demand(sc["leak_fraction"] * total, "evt_ramp")
            resolved.update(junction=j, leak_lps=sc["leak_fraction"] * total * 1000.0)
        elif sub == "pump_trip":
            on = [p for p in spec.pumps if running(p)]
            if not on:
                raise ValueError("no running pump at onset")
            p = pick(on)
            p2._override(wn, p, t0, None, 0, "evt")
            resolved.update(pump=p)
        elif sub == "gradual_demand_ramp":
            f, rh = sc["factor"], sc["ramp_h"]
            scale_patterns(lambda h: 1.0 + (f - 1.0) * min((h - t0 + 1) / rh, 1.0))
            resolved.update(factor=f, ramp_h=rh)
        elif sub == "erratic_demand":
            lo, hi = sc["low_high"]
            jumps = np.random.default_rng(sc["seed"] + 3).uniform(lo, hi, size=400)
            scale_patterns(lambda h: float(jumps[h - t0]))
        res = p2._run(wn)
        true = p2._true_channels(res, wn, spec)
        if sub == "level_spoof_starve":
            start = next(r["open_below"] for r in spec.level_rules if r["link"] == sc["pump"] and r["tank"] == sc["tank"])
            if not np.any(true[f"L_{sc['tank']}"][win] < start - p2.LEVEL_TOL_M):
                raise ValueError("the starved tank has not yet passed the start setpoint inside the window")
        if sub == "masked_duty_swap" and float(np.max(true[f"F_{resolved['started']}"][win])) < 1.0:
            raise ValueError("the started pump delivers no flow")
    reported = p2._measure(true, np.random.default_rng(state_seed + 7))
    if sub == "level_spoof_starve":
        k = f"L_{sc['tank']}"
        reported[k][t0:] = reported[k][t0 - 1]  # the historian shows the held level the controller was fed
    elif sub == "masked_duty_swap":
        a, b = resolved["stopped"], resolved["started"]
        fa, fb = reported[f"F_{a}"].copy(), reported[f"F_{b}"].copy()
        reported[f"S_{a}"][t0:], reported[f"S_{b}"][t0:] = 1.0, 0.0
        reported[f"F_{a}"][t0:], reported[f"F_{b}"][t0:] = fb[t0:], fa[t0:]
    elif sub in ("noise_burst", "lagged_channel"):
        kind = sc["kind"]
        options = [f"F_{p}" for p in spec.pumps if running(p)] if kind == "F" else [k for k in reported if p2._kind(k) == kind]
        if not options:
            raise ValueError("no eligible channel")
        chan = pick(options)
        scale = float((sigma or {}).get(chan, p2.SIGMA_FLOOR[kind]))
        x = reported[chan]
        if sub == "noise_burst":
            x[t0:] = x[t0:] + sc["magnitude_sigma"] * scale * np.random.default_rng(sc["seed"] + 2).standard_normal(len(x) - t0)
        else:
            src = x.copy()
            x[t0:] = src[t0 - sc["lag_h"] : len(x) - sc["lag_h"]]
        reported[chan] = np.round(np.clip(x, 0.0, None) if kind == "L" else x, 2)
        if not np.any(np.abs(reported[chan][win] - true[chan][win]) >= 3.0 * scale):
            raise ValueError("faulty reading indistinguishable from the true value")
        resolved.update(channel=chan)
    twin = p2.twin_channels(config, spec, reported, we)
    return {"reported": reported, "twin": twin, "resolved": resolved}


def _sample(spec: p2.NetSpec, cls: str, seed: int, family: str) -> dict[str, Any]:
    return sample_novel4(spec, cls, seed) if family == "novel4" else _orig_sample(spec, cls, seed, family)


def _simulate(config: Any, spec: p2.NetSpec, sc: dict[str, Any], sigma: Mapping[str, float] | None = None) -> dict[str, Any]:
    return simulate_novel4(config, spec, sc, sigma) if sc.get("subtype") in _NOVEL4_SUBTYPES else _orig_simulate(config, spec, sc, sigma)


def register() -> None:
    """Make jev_pilot_p2v2.prepare_split aware of the round-4 splits (sealed behind design_freeze_r4.json) and of novel4."""

    v.SPLITS.update(R4_SPLITS)
    v.SEALED = tuple(dict.fromkeys(v.SEALED + tuple(R4_SPLITS)))
    v.FREEZE_FILE.update({s: R4_FREEZE for s in R4_SPLITS})
    v.sample_scenario, v.simulate = _sample, _simulate


register()


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "prepare"])
    ap.add_argument("--split", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        from hydrojev.config import load_config

        v.prepare_split(load_config(), a.split, v.STATE_FORMAT_PHYSICS)
    else:
        run(a.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

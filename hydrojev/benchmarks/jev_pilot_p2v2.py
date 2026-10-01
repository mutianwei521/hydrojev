"""Jev pilot P2 v2: de-biased cause discrimination with novel event subtypes (docs/jev/PILOT_PROTOCOL_v2.md).

Builds on the frozen v1 generator (``jev_pilot_p2``, unchanged) and fixes what the v1 root-cause
analysis found (artifacts/jev_pilot_rootcause/analysis.json):

* presentation: a neutral system description (no threat-model sentence), ``detail`` only for
  violated checks, and signed summaries (direction of the demand, pressure, tank and pump residuals);
* criteria: class-level definitions without subtype examples, and an instruction to give every
  class non-zero probability, so label-free prior correction can work;
* comparators: R2 gets the same signed summaries as extra features (R1 stays the frozen v1 tree).

Splits (fresh seeds, disjoint from v1):
  train   C-Town, v1 subtypes, 50/class   R2 training; unlabeled pool for Jev prior calibration
  dev     C-Town, v1 subtypes, 10/class   design iteration of the Jev question (labels visible)
  eval_id C-Town, v1 subtypes, 10/class   sealed, in-distribution
  eval_novel C-Town, novel subtypes, 10/class  sealed, subtypes never generated before the freeze

Novel subtypes (one realisation of each class definition that neither R2 nor the question design saw):
  cyber_attack      pump_on_spoofed_off (stopped pump forced on, historian keeps reporting it stopped)
                    overpump_level_replay (interlock defeated, level replayed from 24 h earlier)
  physical_fault    pipe_closure (a main pipe fails closed, e.g. a collapsed or stuck line valve)
  normal_transient  demand_pattern_shift (consumption timing shifted 1-3 h, e.g. a holiday)
  sensor_fault      gain_error (one channel reads x (1 + g))

Round 3 (design_freeze_r3.json): state format 3 adds twin-free physics cross-checks, and a third subtype family novel3
(NOVEL3) gives a sealed set whose subtypes no part of the format-3 design has seen.

Usage:
  python -m hydrojev.benchmarks.jev_pilot_p2v2 prepare --splits train dev
  python -m hydrojev.benchmarks.jev_pilot_p2v2 run --split dev --revision 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.config import HydroJEVConfig, load_config
from hydrojev.methods.hydrojev_detector import HydroJEVSchemaError, _assert_no_ground_truth, _json_safe

SCHEMA_VERSION = "hydrojev.cause_state/2"
OUTDIR = Path("artifacts/jev_pilot_p2v2")
LEDGER_DIR = p1.DEFAULT_OUTDIR
CLASSES = p2.CLASSES
EVENT_TYPES = p2.EVENT_TYPES
SPLITS = {  # name: (network, per class, seed base, subtype family)
    "train": ("CTOWN", 50, 1_100_000, "v1"),
    "dev": ("CTOWN", 10, 1_300_000, "v1"),
    "eval_id": ("CTOWN", 10, 1_500_000, "v1"),
    "eval_novel": ("CTOWN", 10, 1_700_000, "novel"),
    # Round 2 (added after the round-1 sealed evaluation; design otherwise unchanged). Bases are offset by 50_000 and
    # 60_000 from the 100_000 grid so that neither the draws nor the redraw ladder (+100_000 per attempt) can reproduce a
    # scenario seed of any other split.
    "eval_id_r2": ("CTOWN", 50, 5_050_000, "v1"),
    "eval_novel_r2": ("CTOWN", 50, 5_060_000, "novel"),
    # Round 3 (state format 3 was designed on the burned round-2 sets, so round 3 needs fresh sealed data; "novel3" is a
    # subtype family written before any of its windows was generated and never used for design). Offsets 70_000/80_000.
    "eval_id_r3": ("CTOWN", 50, 5_070_000, "v1"),
    "eval_novel3_r3": ("CTOWN", 50, 5_080_000, "novel3"),
}
SEALED = ("eval_id", "eval_novel", "eval_id_r2", "eval_novel_r2", "eval_id_r3", "eval_novel3_r3")
FREEZE_FILE = {"eval_id": "design_freeze.json", "eval_novel": "design_freeze.json",
               "eval_id_r2": "design_freeze_r2.json", "eval_novel_r2": "design_freeze_r2.json",
               "eval_id_r3": "design_freeze_r3.json", "eval_novel3_r3": "design_freeze_r3.json"}
NOVEL = {
    "cyber_attack": ("pump_on_spoofed_off", "overpump_level_replay"),
    "physical_fault": ("pipe_closure",),
    "normal_transient": ("demand_pattern_shift",),
    "sensor_fault": ("gain_error",),
}
NOVEL3 = {  # round 3: disjoint from the v1 and novel families
    "cyber_attack": ("setpoint_tamper_unmasked", "pump_stop_replay24"),
    "physical_fault": ("pump_wear", "dual_leak"),
    "normal_transient": ("duty_pump_rotation", "temporary_demand_peak"),
    "sensor_fault": ("dropout_zero", "intermittent_spikes"),
}

SYSTEM_DESCRIPTION = (
    "Water distribution network with tanks, pumps and valves, hourly SCADA historian. Channel names: "
    "L_<tank> tank level (m), S_<link> actuator status (1 = running/open), F_<link> pump flow (L/s), "
    "P_<node> junction pressure (m), Fin_<tank> tank net inflow meter (L/s, positive = filling), "
    "Fsrc_<source> source outflow meter (L/s). Pumps and valves are switched by local controllers from "
    "their own tank-level instruments or by time schedules (see control_configuration); the local "
    "controllers do not read the historian. The digital twin is a hydraulic model of this network with "
    "nominal demand and nominal assets, re-synchronised to the reported tank levels 11 h before the window "
    "end and driven hour by hour by the reported actuator states, so twin residuals show what the reported "
    "operation cannot explain. z-scores and check thresholds are calibrated on event-free operation of this "
    "network with ordinary demand variability. The window may contain an event that began before or during it."
)

EVENT_CRITERIA_V2 = {
    "cyber_attack": (
        "Concealed manipulation of control or telemetry: the hydraulics really changed (several physically "
        "related channels deviate from the twin), but at least one reported reading or actuator status is "
        "contradicted by those independent measurements, or an actuator runs against its control logic given "
        "the physically estimated level. The reported telemetry does not tell one consistent physical story, "
        "and a single faulty instrument cannot explain the pattern."
    ),
    "physical_fault": (
        "A real fault of a network asset (pipe, fitting, pump or valve) that the twin does not model: measurements "
        "deviate from the twin but stay mutually consistent with one another, actuators follow their control "
        "logic, and the effect is concentrated around one location or asset rather than affecting the whole "
        "network in the same way."
    ),
    "normal_transient": (
        "Ordinary operation: a change in the amount or timing of consumption that acts on the whole network "
        "coherently, or a legitimate actuator switch whose hydraulic effects agree with all measurements. "
        "Telemetry is mutually consistent and the reported actuator states explain the hydraulics."
    ),
    "sensor_fault": (
        "A single measuring instrument misreports: one channel deviates from the twin and from the channels "
        "physically related to it, while those related channels, the actuators and the hydraulics show "
        "nothing unusual."
    ),
    "insufficient_evidence": (
        "Use only when the evidence is missing or genuinely contradictory so that none of the other four "
        "explanations can be preferred."
    ),
}

INSTRUCTIONS = {
    1: (
        "Using only the supplied evidence for this six-hour SCADA window, decide which cause best explains what "
        "the twin residuals and consistency checks show. Treat the four causes as equally likely before looking "
        "at the evidence. First ask whether the reported telemetry tells one mutually consistent physical story "
        "(see signed_summary.corroboration and the violated checks); then ask whether the effect is confined to "
        "one channel, concentrated at one location, or network-wide. Give every cause a non-zero probability "
        "that reflects how well the evidence fits its definition."
    ),
}


# Revision 2 (fixed on dev only, after revision 1): the revision-1 cyber definition also matched honest operation, because
# the twin's estimated levels drift whenever consumption departs from nominal and an operator switch breaks the
# configured logic by design. Revision 2 ties cyber to contradictions between reported values and independent
# measurements, and states what the twin cannot know.
EVENT_CRITERIA_R2 = {
    "cyber_attack": (
        "Concealed manipulation of control or telemetry: a reported actuator status or reading is contradicted by "
        "independent, physically related measurements (meters of other assets, pressures, flows, mass balance), so "
        "the reported operation does not match what the network is physically doing. Concealment may make several "
        "reported channels agree with one another while the physics measured elsewhere disagrees; a single faulty "
        "instrument cannot explain the pattern."
    ),
    "physical_fault": (
        "A real fault of a network asset (pipe, fitting, pump or valve) that the twin does not model: measurements "
        "deviate from the twin, yet independent meters agree with one another and with the reported actuator states; "
        "the effect is concentrated around one location or asset rather than acting on the whole network in the same "
        "way."
    ),
    "normal_transient": (
        "Ordinary, honestly reported operation: consumption higher, lower or earlier/later than nominal acting on the "
        "whole network coherently, or an actuator switched by an operator. The twin assumes nominal consumption and "
        "only knows the configured control logic, so under this cause its residuals, its estimated tank levels and "
        "the control-logic checks can all disagree with operation, while the independent measurements agree with one "
        "another and confirm every reported actuator state."
    ),
    "sensor_fault": EVENT_CRITERIA_V2["sensor_fault"],
    "insufficient_evidence": EVENT_CRITERIA_V2["insufficient_evidence"],
}
INSTRUCTIONS[2] = (
    "Using only the supplied evidence for this six-hour SCADA window, decide which cause best explains it. Treat the "
    "four causes as equally likely before looking at the evidence. Reason in this order: (1) Are the independent "
    "measurements (source and tank meters, pressures, pump flows, reported levels) mutually consistent, and do they "
    "confirm every reported actuator state? A contradiction between a reported value and independent physics points "
    "to manipulation or to a faulty instrument; decide between them by whether one channel alone is wrong. (2) If "
    "everything reported is consistent, the twin disagreement comes from the world: decide whether it acts on the "
    "whole network coherently (ordinary operation) or is concentrated at one location or asset (asset fault). "
    "Remember that the twin uses nominal consumption and the configured logic only. Give every cause a non-zero "
    "probability that reflects how well the evidence fits its definition."
)
# Revision 3 (dev only): revision 2 still read honest consumption changes as manipulation, treating disagreement with
# the twin as a contradiction. The instruction now separates model-versus-measurement from measurement-versus-measurement.
INSTRUCTIONS[3] = (
    "Using only the supplied evidence for this six-hour SCADA window, decide which cause best explains it. Treat the "
    "four causes as equally likely before looking at the evidence. The twin is a model with nominal consumption and "
    "nominal assets, not a measurement: a measurement that disagrees with the twin shows that the world differs from "
    "the model, not that the measurement is false. Only a disagreement between measurements, or between a reported "
    "actuator state and the measurements it should produce, is a contradiction. Reason in this order: (1) Is there "
    "such a contradiction (for example a violated check that compares two reported quantities with each other)? If "
    "yes, decide between concealed manipulation and a faulty instrument by whether one channel alone is wrong while "
    "every channel related to it is normal. (2) If there is none, decide how the world differs from the model: "
    "consumption changed network-wide (pressures, tank levels, source outflow and pump flows all shifted in the "
    "direction that more or less consumption implies), an operator switched an actuator, or an asset fault acts at "
    "one location. Give every cause a non-zero probability that reflects how well the evidence fits its definition."
)
# Revision 4 (round 3, for state format 3; designed on dev_s3 and on the burned round-2 sets, see PILOT_PROTOCOL_v2.md):
# revision 3 read a running pump with little flow as a contradiction and a level hidden from its controller as a faulty
# level instrument. Revision 4 states which asset laws decide those two cases.
INSTRUCTIONS[4] = INSTRUCTIONS[3][: INSTRUCTIONS[3].index(" Give every cause")] + (
    " Asset laws decide whether reported values contradict each other: a running pump may deliver little or no flow "
    "when the path downstream of it is blocked or throttled, and that is no contradiction while its measured head gain "
    "lies on its curve (pump_operating_point_vs_curve consistent); a pump reported running whose measured head and flow "
    "lie off its curve, or a station reported stopped whose discharge head exceeds the head of the tank it feeds, is a "
    "contradiction of the reported pump status itself. A faulty historian instrument corrupts one channel and only the "
    "checks computed from it, and it cannot change what the local controllers do; if a pump kept running although its "
    "tank's meter-integrated level passed the pump's stop setpoint (control_logic_vs_metered_level), the controller "
    "itself acted on a false level. Give every cause a non-zero probability that reflects how well the evidence fits "
    "its definition."
)
CRITERIA = {1: EVENT_CRITERIA_V2, 2: EVENT_CRITERIA_R2, 3: EVENT_CRITERIA_R2, 4: EVENT_CRITERIA_R2}


def build_questions(revision: int = 1) -> dict[str, dict[str, Any]]:
    return {"event_type": {"type": "choice", "instructions": INSTRUCTIONS[revision], "criteria": dict(CRITERIA[revision])}}


def questions_sha256(revision: int = 1) -> str:
    return hashlib.sha256(json.dumps(build_questions(revision), sort_keys=True).encode()).hexdigest()


def build_request(state: Mapping[str, Any], *, revision: int = 1, model: str = "jev-latest") -> dict[str, Any]:
    _assert_no_ground_truth(state)
    request = {"model": model, "state": _json_safe(dict(state), path="state"), "questions": build_questions(revision)}
    json.dumps(request, ensure_ascii=False, allow_nan=False)
    return request


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def sample_scenario(spec: p2.NetSpec, cls: str, seed: int, family: str) -> dict[str, Any]:
    if family == "v1":
        return p2.sample_scenario(spec, cls, seed, None)
    if family == "novel3":
        return _sample_novel3(spec, cls, seed)
    rng = np.random.default_rng(seed)
    t0 = int(rng.integers(p2.ONSET_RANGE_H[0], p2.ONSET_RANGE_H[1] + 1))
    we = t0 + int(rng.integers(p2.LAG_RANGE_H[0], p2.LAG_RANGE_H[1] + 1))
    sc: dict[str, Any] = {"seed": seed, "network": spec.name, "cls": cls, "onset_h": t0, "window_end_h": we}
    sc["subtype"] = str(rng.choice(list(NOVEL[cls])))
    if sc["subtype"] == "overpump_level_replay":
        closable = [r for r in spec.level_rules if "close_above" in r and r["link"] in spec.pumps]
        r = closable[int(rng.integers(len(closable)))]
        sc.update(pump=r["link"], tank=r["tank"])
    elif sc["subtype"] == "demand_pattern_shift":
        sc["shift_h"] = int(rng.choice([-3, -2, -1, 1, 2, 3]))
    elif sc["subtype"] == "gain_error":
        sc["kind"] = str(rng.choice(["L", "P", "F"]))
        sc["magnitude_sigma"] = float(rng.uniform(4.0, 10.0)) * float(rng.choice([-1.0, 1.0]))
    sc["pick"] = float(rng.random())
    return sc


def _shift_patterns(wn: Any, t0: int, shift: int) -> None:
    for pname in list(wn.pattern_name_list):
        pat = wn.get_pattern(pname)
        m = np.asarray(pat.multipliers, dtype=float)
        if len(m) < 2:
            continue
        out = m.copy()
        for h in range(t0, len(m)):
            out[h] = m[min(max(h - shift, 0), len(m) - 1)]
        pat.multipliers = list(out)


def simulate(config: HydroJEVConfig, spec: p2.NetSpec, sc: dict[str, Any], sigma: Mapping[str, float] | None = None) -> dict[str, Any]:
    sub = sc.get("subtype")
    if sub in {s for v in NOVEL3.values() for s in v}:
        return _simulate_novel3(config, spec, sc, sigma)
    if sub not in {s for v in NOVEL.values() for s in v}:
        return p2.simulate(config, spec, sc, sigma)
    rng = np.random.default_rng(sc["seed"] + 1)
    t0, we = sc["onset_h"], sc["window_end_h"]
    state_seed = rng.integers(1 << 31)
    wn = p2._noisy_model(config, spec, np.random.default_rng(state_seed), we)
    base_res = p2._run(wn)
    base = p2._true_channels(base_res, wn, spec)
    resolved: dict[str, Any] = {}

    def pick(options: Sequence[str]) -> str:
        return sorted(options)[min(int(sc["pick"] * len(options)), len(options) - 1)]

    true = base
    if sub != "gain_error":
        wn = p2._noisy_model(config, spec, np.random.default_rng(state_seed), we)
        if sub == "pump_on_spoofed_off":
            stopped = [p for p in spec.pumps if base[f"S_{p}"][t0 - 1] < 0.5]
            if not stopped:
                raise ValueError("no stopped pump at onset")
            p = pick(stopped)
            p2._override(wn, p, t0, None, 1, "atk")
            resolved.update(pump=p)
        elif sub == "overpump_level_replay":
            p2._override(wn, sc["pump"], t0, None, 1, "atk")
            resolved.update(pump=sc["pump"], tank=sc["tank"])
        elif sub == "pipe_closure":
            flows = base_res.link["flowrate"][wn.pipe_name_list].abs().median().sort_values(ascending=False)
            adjacent = {n for ends in spec.pump_ends.values() for n in ends} | set(spec.tanks) | set(spec.sources)
            pool = [l for l in flows.index[:40] if wn.get_link(l).start_node_name not in adjacent and wn.get_link(l).end_node_name not in adjacent]
            l = pick(pool)
            p2._override(wn, l, t0, None, 0, "evt")
            resolved.update(pipe=l)
        elif sub == "demand_pattern_shift":
            _shift_patterns(wn, t0, sc["shift_h"])
            resolved.update(shift_h=sc["shift_h"])
        res = p2._run(wn)
        true = p2._true_channels(res, wn, spec)
        if sub == "overpump_level_replay":
            close = next(r["close_above"] for r in spec.level_rules if r["link"] == sc["pump"] and r["tank"] == sc["tank"])
            if not np.any((true[f"L_{sc['tank']}"][p2._win(we)] > close + p2.LEVEL_TOL_M) & (true[f"S_{sc['pump']}"][p2._win(we)] > 0.5)):
                raise ValueError("interlock not yet defeated inside the window")
        if sub == "pump_on_spoofed_off" and float(np.max(true[f"F_{resolved['pump']}"][p2._win(we)])) < 1.0:
            raise ValueError("forced pump delivers no flow")
    reported = p2._measure(true, np.random.default_rng(state_seed + 7))
    if sub == "pump_on_spoofed_off":
        p = resolved["pump"]
        reported[f"S_{p}"][t0:] = 0.0
        reported[f"F_{p}"][t0:] = reported[f"F_{p}"][t0 - 1]
    elif sub == "overpump_level_replay":
        k = f"L_{sc['tank']}"
        src = reported[k].copy()
        reported[k][t0:] = np.asarray([src[max(h - 24, 0)] for h in range(t0, len(src))])
    elif sub == "gain_error":
        kind = sc["kind"]
        if kind == "F":
            options = [f"F_{p}" for p in spec.pumps if base[f"S_{p}"][t0 - 1] > 0.5 and base[f"F_{p}"][t0 - 1] > 1.0]
        else:
            options = [k for k in reported if p2._kind(k) == kind]
        if not options:
            raise ValueError("no eligible channel")
        chan = pick(options)
        scale = float((sigma or {}).get(chan, p2.SIGMA_FLOOR[kind]))
        ref = max(abs(float(reported[chan][t0 - 1])), 1e-3)
        g = sc["magnitude_sigma"] * scale / ref
        x = reported[chan]
        x[t0:] = x[t0:] * (1.0 + g)
        reported[chan] = np.round(np.clip(x, 0.0, None) if kind == "L" else x, 2)
        if abs(reported[chan][we] - true[chan][we]) < 3.0 * scale:
            raise ValueError("faulty reading indistinguishable from the true value")
        resolved.update(channel=chan, gain=g)
    twin = p2.twin_channels(config, spec, reported, we)
    return {"reported": reported, "twin": twin, "resolved": resolved}


# --------------------------------------------------------------------------- #
# Round-3 subtype family "novel3" (written before any of its windows existed; parameters are physical defaults)
# --------------------------------------------------------------------------- #


def _sample_novel3(spec: p2.NetSpec, cls: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t0 = int(rng.integers(p2.ONSET_RANGE_H[0], p2.ONSET_RANGE_H[1] + 1))
    we = t0 + int(rng.integers(p2.LAG_RANGE_H[0], p2.LAG_RANGE_H[1] + 1))
    sc: dict[str, Any] = {"seed": seed, "network": spec.name, "cls": cls, "onset_h": t0, "window_end_h": we}
    sub = sc["subtype"] = str(rng.choice(list(NOVEL3[cls])))
    if sub == "setpoint_tamper_unmasked":  # controller logic altered to keep the pump on; the historian stays honest
        closable = [r for r in spec.level_rules if "close_above" in r and r["link"] in spec.pumps]
        r = closable[int(rng.integers(len(closable)))]
        sc.update(pump=r["link"], tank=r["tank"])
    elif sub == "pump_wear":  # impeller wear / partial blockage: lower effective speed from the onset
        sc["speed"] = float(rng.uniform(0.70, 0.88))
    elif sub == "dual_leak":  # two simultaneous moderate leaks
        sc["leak_fractions"] = [float(rng.uniform(0.03, 0.07)), float(rng.uniform(0.03, 0.07))]
    elif sub == "duty_pump_rotation":  # operator swaps the running and the standby pump of a station, in manual mode
        sc["duration_h"] = int(rng.integers(2, 7))
    elif sub == "temporary_demand_peak":  # network-wide demand peak (event, heat) that lasts a few hours
        sc["factor"] = float(rng.uniform(1.25, 1.50))
        sc["duration_h"] = int(rng.integers(2, 6))
    elif sub in ("dropout_zero", "intermittent_spikes"):
        sc["kind"] = str(rng.choice(["L", "P", "F"]))
        sc["magnitude_sigma"] = float(rng.uniform(6.0, 12.0)) * float(rng.choice([-1.0, 1.0]))
    sc["pick"] = float(rng.random())
    sc["pick2"] = float(rng.random())
    return sc


def _simulate_novel3(config: HydroJEVConfig, spec: p2.NetSpec, sc: dict[str, Any], sigma: Mapping[str, float] | None) -> dict[str, Any]:
    sub = sc["subtype"]
    rng = np.random.default_rng(sc["seed"] + 1)
    t0, we = sc["onset_h"], sc["window_end_h"]
    state_seed = rng.integers(1 << 31)
    wn = p2._noisy_model(config, spec, np.random.default_rng(state_seed), we)
    base_res = p2._run(wn)
    base = p2._true_channels(base_res, wn, spec)
    resolved: dict[str, Any] = {}

    def pick(options: Sequence[Any], key: str = "pick") -> Any:
        return sorted(options)[min(int(sc[key] * len(options)), len(options) - 1)]

    def running(p: str) -> bool:
        return base[f"S_{p}"][t0 - 1] > 0.5 and base[f"F_{p}"][t0 - 1] > 1.0

    true = base
    if sub not in ("dropout_zero", "intermittent_spikes"):
        wn = p2._noisy_model(config, spec, np.random.default_rng(state_seed), we)
        if sub == "setpoint_tamper_unmasked":
            p2._override(wn, sc["pump"], t0, None, 1, "atk")
            resolved.update(pump=sc["pump"], tank=sc["tank"])
        elif sub == "pump_stop_replay24":
            on = [p for p in spec.pumps if running(p)]
            if not on:
                raise ValueError("no running pump at onset")
            p = pick(on)
            p2._override(wn, p, t0, None, 0, "atk")
            resolved.update(pump=p)
        elif sub == "pump_wear":
            on = [p for p in spec.pumps if running(p)]
            if not on:
                raise ValueError("no running pump at onset")
            p = pick(on)
            wn.add_pattern("evt_speed", [1.0] * t0 + [sc["speed"]] * (we - t0 + 3))
            wn.get_link(p).speed_pattern_name = "evt_speed"
            resolved.update(pump=p, speed=sc["speed"])
        elif sub == "dual_leak":
            pres = base_res.node["pressure"].median()
            pool = sorted(j for j in wn.junction_name_list if pres[j] > 20.0 and j not in spec.pressure_nodes)
            j1 = pick(pool)
            j2 = pick([j for j in pool if j != j1], "pick2")
            total = float(base_res.node["demand"][wn.junction_name_list].sum(axis=1).mean())
            wn.add_pattern("evt_step", [0.0] * t0 + [1.0] * (we - t0 + 3))
            for j, f in zip((j1, j2), sc["leak_fractions"]):
                wn.get_node(j).add_demand(f * total, "evt_step")
            resolved.update(junctions=[j1, j2], leak_lps=[f * total * 1000.0 for f in sc["leak_fractions"]])
        elif sub == "duty_pump_rotation":
            feed = fed_tanks(config, spec)
            pairs = [(a, b) for a in spec.pumps for b in spec.pumps
                     if a != b and feed.get(a) is not None and feed.get(a) == feed.get(b) and running(a) and base[f"S_{b}"][t0 - 1] < 0.5]
            if not pairs:
                raise ValueError("no station with a running and a standby pump")
            a, b = pick(pairs)
            p2._override(wn, a, t0, t0 + sc["duration_h"], 0, "op_a")
            p2._override(wn, b, t0, t0 + sc["duration_h"], 1, "op_b")
            resolved.update(stopped=a, started=b)
        elif sub == "temporary_demand_peak":
            for pname in list(wn.pattern_name_list):
                pat = wn.get_pattern(pname)
                m = np.asarray(pat.multipliers, dtype=float)
                if len(m) < 2:
                    continue
                m[t0 : t0 + sc["duration_h"]] *= sc["factor"]
                pat.multipliers = list(m)
            resolved.update(factor=sc["factor"], duration_h=sc["duration_h"])
        res = p2._run(wn)
        true = p2._true_channels(res, wn, spec)
        if sub == "setpoint_tamper_unmasked":
            close = next(r["close_above"] for r in spec.level_rules if r["link"] == sc["pump"] and r["tank"] == sc["tank"])
            if not np.any((true[f"L_{sc['tank']}"][p2._win(we)] > close + p2.LEVEL_TOL_M) & (true[f"S_{sc['pump']}"][p2._win(we)] > 0.5)):
                raise ValueError("interlock not yet defeated inside the window")
        if sub == "pump_wear" and float(np.max(np.abs(true[f"F_{resolved['pump']}"][p2._win(we)] - base[f"F_{resolved['pump']}"][p2._win(we)]))) < 1.0:
            raise ValueError("worn pump indistinguishable from the healthy one")
    reported = p2._measure(true, np.random.default_rng(state_seed + 7))
    if sub == "pump_stop_replay24":
        p = resolved["pump"]
        for k in (f"S_{p}", f"F_{p}"):
            x = reported[k]
            for h in range(t0, len(x)):
                x[h] = x[h - 24]
        if not np.any((reported[f"S_{p}"][p2._win(we)] > 0.5) & (true[f"S_{p}"][p2._win(we)] < 0.5)):
            raise ValueError("replayed status coincides with the true status")
    elif sub in ("dropout_zero", "intermittent_spikes"):
        kind = sc["kind"]
        if kind == "F":
            options = [f"F_{p}" for p in spec.pumps if running(p)]
        else:
            options = [k for k in reported if p2._kind(k) == kind]
        if not options:
            raise ValueError("no eligible channel")
        chan = pick(options)
        scale = float((sigma or {}).get(chan, p2.SIGMA_FLOOR[kind]))
        x = reported[chan]
        if sub == "dropout_zero":
            x[t0:] = 0.0
        else:
            hits = np.random.default_rng(sc["seed"] + 2).random(len(x) - t0) < 0.5
            x[t0:] = x[t0:] + sc["magnitude_sigma"] * scale * hits
        reported[chan] = np.round(np.clip(x, 0.0, None) if kind == "L" else x, 2)
        if not np.any(np.abs(reported[chan][p2._win(we)] - true[chan][p2._win(we)]) >= 3.0 * scale):
            raise ValueError("faulty reading indistinguishable from the true value")
        resolved.update(channel=chan)
    twin = p2.twin_channels(config, spec, reported, we)
    return {"reported": reported, "twin": twin, "resolved": resolved}


# --------------------------------------------------------------------------- #
# State format 2: source-meter flow balance (a generic independent-meter cross-check)
# --------------------------------------------------------------------------- #

STATE_FORMAT = 2
N_BALANCE_CAL = 30
BALANCE_QUANTILE = 0.975


def stations(config: HydroJEVConfig, spec: p2.NetSpec) -> dict[str, list[str]]:
    """Pumps fed by each source without an intermediate tank or pump (the source's pump station)."""

    wn = p2._load_model(config, spec.name)
    out: dict[str, list[str]] = {}
    for src in spec.sources:
        seen, stack, tank_hit = {src}, [src], False
        while stack:
            n = stack.pop()
            for lname in wn.get_links_for_node(n):
                if lname in wn.pump_name_list:
                    continue
                link = wn.get_link(lname)
                for m in (link.start_node_name, link.end_node_name):
                    if m in seen:
                        continue
                    seen.add(m)
                    if m in wn.tank_name_list or m in wn.reservoir_name_list:
                        tank_hit = True
                        continue
                    stack.append(m)
        pumps = [pu for pu in spec.pumps if spec.pump_ends[pu][0] in seen]
        if pumps and not tank_hit:
            out[src] = pumps
    return out


def balance_series(ch: Mapping[str, np.ndarray], station: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    return {src: np.asarray(ch[f"Fsrc_{src}"], float) - np.sum([np.asarray(ch[f"F_{pu}"], float) for pu in pumps], axis=0)
            for src, pumps in station.items()}


def balance_calibration(config: HydroJEVConfig, spec: p2.NetSpec) -> dict[str, Any]:
    """Event-free spread of (source meter - sum of reported station pump flows) over 6-h windows."""

    cache = OUTDIR / f"balance_calibration_{spec.name.lower()}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    station = stations(config, spec)
    peaks: dict[str, list[float]] = {k: [] for k in station}
    hourly: dict[str, list[float]] = {k: [] for k in station}
    for i in range(N_BALANCE_CAL):
        rng = np.random.default_rng(2_900_000 + i)
        wn = p2._noisy_model(config, spec, rng, 72)
        reported = p2._measure(p2._true_channels(p2._run(wn), wn, spec), rng)
        for src, b in balance_series(reported, station).items():
            hourly[src] += list(b[12:72])
            for we in range(24, 72, 6):
                peaks[src].append(float(np.max(np.abs(b[p2._win(we)] - np.median(b[12:72])))))
    out = {"stations": station, "median": {k: float(np.median(hourly[k])) for k in station},
           "threshold_abs_ls": {k: max(float(np.quantile(peaks[k], BALANCE_QUANTILE)), 3.0) for k in station}}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def balance_check(sim: Mapping[str, Any], we: int, bcal: Mapping[str, Any]) -> dict[str, Any]:
    reported = sim["reported"]
    worst, per = 0.0, {}
    for src, b in balance_series(reported, bcal["stations"]).items():
        dev = b[p2._win(we)] - bcal["median"][src]
        worst = max(worst, float(np.max(np.abs(dev))) / bcal["threshold_abs_ls"][src])
        per[src] = {"station_pumps": list(bcal["stations"][src]),
                    "source_meter_now": round(float(reported[f"Fsrc_{src}"][we]), 1),
                    "sum_reported_station_pump_flows_now": round(float(np.sum([reported[f"F_{pu}"][we] for pu in bcal["stations"][src]])), 1),
                    "imbalance_by_hour_ls": [round(float(v), 1) for v in dev]}
    out = {"status": "violated" if worst > 1.0 else "consistent", "statistic": round(worst, 2), "threshold": 1.0,
           "event_free_violation_rate": round(1.0 - BALANCE_QUANTILE, 3),
           "meaning": "source outflow meter versus the sum of the reported flows of the pumps it feeds directly "
                      "(largest |imbalance| in the window divided by its event-free 97.5th percentile)"}
    if worst > 1.0:
        out["detail"] = per
    return out


# --------------------------------------------------------------------------- #
# State format 3: twin-free physics cross-checks of reported actuator states (round 3; designed on the burned round-2
# sets and train_s2, see docs/jev/PILOT_PROTOCOL_v2.md). Each compares reported values with each other through a law
# of the asset (pump curve, check valve, tank volume), never with the nominal-demand twin.
# --------------------------------------------------------------------------- #

STATE_FORMAT_PHYSICS = 3
N_PHYSICS_CAL = 40
STAT_CLIP = 99.0
PHYSICS_CHECKS = ("pump_operating_point_vs_curve", "stopped_station_discharge_vs_tank_head", "control_logic_vs_metered_level")


def fed_tanks(config: HydroJEVConfig, spec: p2.NetSpec) -> dict[str, str]:
    """Tank each pump delivers to: the nearest tank from its discharge node through non-pump links."""

    wn = p2._load_model(config, spec.name)
    out = {}
    for pu in spec.pumps:
        start = spec.pump_ends[pu][1]
        seen, frontier = {start}, [start]
        while frontier and pu not in out:
            nxt = []
            for n in frontier:
                for lname in wn.get_links_for_node(n):
                    if lname in wn.pump_name_list:
                        continue
                    link = wn.get_link(lname)
                    for m in (link.start_node_name, link.end_node_name):
                        if m in seen:
                            continue
                        seen.add(m)
                        if m in wn.tank_name_list:
                            out.setdefault(pu, m)
                        elif m not in wn.reservoir_name_list:
                            nxt.append(m)
            frontier = sorted(nxt)
    return out


def _curve_head(coef: Sequence[float], flow_ls: np.ndarray) -> np.ndarray:
    a, b, c = coef
    return a - b * (np.maximum(np.asarray(flow_ls, float), 0.0) / 1000.0) ** c


def _metered_levels(spec: p2.NetSpec, rep: Mapping[str, np.ndarray], tank: str, we: int) -> np.ndarray:
    """Tank level over the window integrated from its own inflow meter, starting at the reported level at the sync hour."""

    h0 = we - p2.SYNC_LAG_H
    fin = np.asarray(rep[f"Fin_{tank}"], float)
    steps = 0.5 * (fin[h0 + 1 : we + 1] + fin[h0:we]) * 3.6 / spec.tank_area[tank]
    return (float(rep[f"L_{tank}"][h0]) + np.concatenate([[0.0], np.cumsum(steps)]))[-p2.WINDOW_H :]


def physics_statistics(spec: p2.NetSpec, rep: Mapping[str, np.ndarray], we: int, pcal: Mapping[str, Any]) -> dict[str, Any]:
    w = p2._win(we)
    out: dict[str, Any] = {}
    # 1. running pumps: measured head gain versus the nominal curve at the reported flow
    best, det = 0.0, None
    for pu in spec.pumps:
        on = rep[f"S_{pu}"][w] > 0.5
        if not on.any():
            continue
        gain = p2._gain(spec, rep, pu)[w]
        curve = _curve_head(pcal["curve"][pu], rep[f"F_{pu}"][w])
        r = (gain - curve - pcal["curve_median"][pu]) / pcal["curve_scale"][pu]
        r = np.clip(np.where(on, r, 0.0), -STAT_CLIP, STAT_CLIP)  # WNTR returns unbounded heads at disconnected nodes
        i = int(np.argmax(np.abs(r)))
        if abs(r[i]) > best:
            best = float(abs(r[i]))
            det = {"pump": pu, "hour_index": i, "reported_flow_ls": p2._r(float(rep[f"F_{pu}"][w][i]), 1),
                   "measured_head_gain_m": p2._r(float(np.clip(gain[i], -999, 999)), 1), "curve_head_at_reported_flow_m": p2._r(float(curve[i]), 1),
                   "shutoff_head_m": p2._r(float(pcal["curve"][pu][0]), 1)}
    out["pump_operating_point_vs_curve"] = (best, det)
    # 2. stations whose pumps are all reported stopped: discharge head versus the head of the tank they feed
    best, det = -1e9, None
    for tank, pumps in pcal["stations"].items():
        stopped = np.all([rep[f"S_{pu}"][w] < 0.5 for pu in pumps], axis=0)
        if not stopped.any():
            continue
        th = rep[f"L_{tank}"][w] + pcal["tank_elevation"][tank]
        for pu in pumps:
            d = spec.pump_ends[pu][1]
            lift = np.where(stopped, rep[f"P_{d}"][w] + spec.elevation[d] - th, -1e9)
            i = int(np.argmax(lift))
            if lift[i] > best:
                best = float(lift[i])
                det = {"station_pumps": list(pumps), "fed_tank": tank, "hour_index": i, "discharge_node": d,
                       "discharge_head_m": p2._r(float(rep[f"P_{d}"][w][i] + spec.elevation[d]), 1), "fed_tank_head_m": p2._r(float(th[i]), 1),
                       "reported_station_flow_ls": p2._r(float(sum(rep[f"F_{p}"][w][i] for p in pumps)), 1)}
    out["stopped_station_discharge_vs_tank_head"] = (float(np.clip(best, 0.0, STAT_CLIP)), det)
    # 3. pump running above its stop setpoint by the meter-integrated level while the reported level stays below it
    best, det, worst_any = 0.0, None, 0.0
    for r in spec.level_rules:
        if "close_above" not in r or r["link"] not in spec.pumps:
            continue
        on = rep[f"S_{r['link']}"][w] > 0.5
        lm = _metered_levels(spec, rep, r["tank"], we)
        lr = rep[f"L_{r['tank']}"][w]
        ex = np.where(on, lm - r["close_above"], 0.0)
        worst_any = max(worst_any, float(ex.max()))
        hide = np.minimum(ex, lm - lr)  # above the setpoint by the meter AND the reported level lower by as much
        i = int(np.argmax(hide))
        if hide[i] > best:
            best = float(hide[i])
            det = {"pump": r["link"], "tank": r["tank"], "stop_setpoint_m": r["close_above"], "hour_index": i,
                   "meter_integrated_level_m": p2._r(float(lm[i])), "reported_level_m": p2._r(float(lr[i]))}
    out["control_logic_vs_metered_level"] = (min(best, STAT_CLIP), det)
    out["_metered_excess_any"] = worst_any
    return out


def physics_calibration(config: HydroJEVConfig, spec: p2.NetSpec) -> dict[str, Any]:
    cache = OUTDIR / f"physics_calibration_{spec.name.lower()}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    wn = p2._load_model(config, spec.name)
    feed = fed_tanks(config, spec)
    stations_by_tank: dict[str, list[str]] = {}
    for pu, t in sorted(feed.items()):
        stations_by_tank.setdefault(t, []).append(pu)
    curve = {pu: list(wn.get_link(pu).get_head_curve_coefficients()) for pu in spec.pumps}
    curve_name = {pu: str(wn.get_link(pu).pump_curve_name) for pu in spec.pumps}
    runs = []
    hourly: dict[str, list[float]] = {pu: [] for pu in spec.pumps}
    for i in range(N_PHYSICS_CAL):
        rng = np.random.default_rng(3_100_000 + i)
        m = p2._noisy_model(config, spec, rng, 72)
        rep = p2._measure(p2._true_channels(p2._run(m), m, spec), rng)
        runs.append(rep)
        for pu in spec.pumps:
            on = rep[f"S_{pu}"][12:72] > 0.5
            hourly[pu] += list((p2._gain(spec, rep, pu)[12:72] - _curve_head(curve[pu], rep[f"F_{pu}"][12:72]))[on])
    med, scale = {}, {}
    for pu in spec.pumps:  # pumps that rarely run in event-free operation borrow the scale of pumps with the same curve
        pool = [x for q in spec.pumps if curve_name[q] == curve_name[pu] for x in hourly[q]] if len(hourly[pu]) < 50 else hourly[pu]
        pool = pool or [x for q in spec.pumps for x in hourly[q]]
        med[pu] = float(np.median(pool))
        scale[pu] = max(float(np.quantile(np.abs(np.asarray(pool) - med[pu]), 0.99)), 0.1)
    pcal = {"stations": stations_by_tank, "curve": curve, "curve_median": med, "curve_scale": scale,
            "tank_elevation": {t: float(wn.get_node(t).elevation) for t in spec.tanks}}
    stats: dict[str, list[float]] = {c: [] for c in PHYSICS_CHECKS}
    for rep in runs:
        for we in range(24, 72, 6):
            s = physics_statistics(spec, rep, we, pcal)
            for c in PHYSICS_CHECKS:
                stats[c].append(s[c][0])
    pcal["thresholds"] = {
        "pump_operating_point_vs_curve": max(float(np.quantile(stats["pump_operating_point_vs_curve"], BALANCE_QUANTILE)), 1.0),
        # physical bound: without a running pump the discharge cannot be above the fed tank's head; 0.5 m for sensor noise
        "stopped_station_discharge_vs_tank_head": max(float(np.max(stats["stopped_station_discharge_vs_tank_head"])), 0.0) + 0.5,
        "control_logic_vs_metered_level": max(float(np.quantile(stats["control_logic_vs_metered_level"], BALANCE_QUANTILE)), p2.LEVEL_TOL_M),
    }
    pcal["event_free_violation_rate"] = {c: float(np.mean(np.asarray(stats[c]) > pcal["thresholds"][c])) for c in PHYSICS_CHECKS}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pcal), encoding="utf-8")
    return pcal


PHYSICS_MEANING = {
    "pump_operating_point_vs_curve": "pumps reported running: head gain measured by the pump's own suction and discharge pressure sensors "
                                     "versus the head its nominal pump curve delivers at its reported flow (largest |difference| over "
                                     "running pump-hours, in units of its event-free 99th percentile); a running pump's measured head "
                                     "and flow must lie on its curve",
    "stopped_station_discharge_vs_tank_head": "pump stations whose pumps are all reported stopped: discharge head (pressure + elevation) "
                                              "minus the head of the tank the station feeds (reported level + elevation), largest value "
                                              "in m; with every pump of the station stopped the discharge head cannot exceed that tank's head",
    "control_logic_vs_metered_level": "pumps reported running: level of the tank that controls the pump integrated from the tank's "
                                      "own inflow meter since the sync hour, compared with the pump's stop setpoint and with the "
                                      "reported level; statistic = largest amount (m) by which the meter-integrated level is both above "
                                      "the stop setpoint and above the reported level; a local controller reading an honest level "
                                      "instrument would have stopped the pump",
}


def physics_checks(spec: p2.NetSpec, sim: Mapping[str, Any], we: int, pcal: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    s = physics_statistics(spec, sim["reported"], we, pcal)
    out = {}
    for c in PHYSICS_CHECKS:
        stat, det = s[c]
        thr = pcal["thresholds"][c]
        entry = {"status": "violated" if stat > thr else "consistent", "statistic": p2._r(stat), "threshold": p2._r(thr),
                 "event_free_violation_rate": p2._r(pcal["event_free_violation_rate"][c], 3), "meaning": PHYSICS_MEANING[c]}
        if stat > thr and det is not None:
            entry["detail"] = det
        out[c] = entry
    return out


R1B_DEFICIT_M = 10.0  # train_s3: running-pump head deficits are >= 12.8 m under attack, <= 7.8 m under sensor faults


def r1b_classify(r1: str, state: Mapping[str, Any]) -> str:
    """Round-3 rule baseline: the frozen R1 tree preceded by the three format-3 physics checks (disclosed, not frozen R1)."""

    return "cyber_attack" if physics_override(state) else r1


def physics_override(state: Mapping[str, Any]) -> bool:
    """The format-3 evidence that R1b (and the override+Jev hybrid) reads as a contradicted report."""

    c = state["consistency_checks"]
    if any(c[k]["status"] == "violated" for k in ("stopped_station_discharge_vs_tank_head", "control_logic_vs_metered_level")):
        return True
    pc = c["pump_operating_point_vs_curve"]
    return pc["status"] == "violated" and pc["detail"]["curve_head_at_reported_flow_m"] - pc["detail"]["measured_head_gain_m"] > R1B_DEFICIT_M


def override_jev(pj: np.ndarray, prior: np.ndarray | None, states: Sequence[Mapping[str, Any]]) -> list[str]:
    """Training-free hybrid: physics override to cyber_attack, otherwise the calibrated Jev decision."""

    jev = _decide(pj, prior)
    return ["cyber_attack" if physics_override(s) else d for s, d in zip(states, jev)]


# --------------------------------------------------------------------------- #
# De-biased state and features
# --------------------------------------------------------------------------- #


def _direction(z: float, tol: float = 1.0) -> str:
    return "above twin" if z > tol else ("below twin" if z < -tol else "near twin")


def signed_summary(spec: p2.NetSpec, sim: Mapping[str, Any], we: int, cal: Mapping[str, Any], ev: Mapping[str, Any]) -> dict[str, Any]:
    st = ev["stats"]
    zz = st["z"]
    kz = cal["channel_z_threshold"]
    dz = float(st["detail"]["system_demand_vs_twin"]["mean_z_last3h"])
    tanks = {k: p2._r(float(v[-1]), 1) for k, v in zz.items() if p2._kind(k) == "L"}
    pumps = {k: p2._r(float(v[-1]), 1) for k, v in zz.items() if p2._kind(k) == "F"}
    sp = ev["pressure_spread"]
    anomalous = ev["anomalous_channels"]
    kinds = sorted({p2._kind(k) for k in anomalous})
    return {
        "system_demand": {"mean_z_last3h": p2._r(dz), "direction": _direction(dz, 2.0),
                          "relative_change_last3h": p2._r(float(st["detail"]["system_demand_vs_twin"]["relative_change_last3h"]), 3)},
        "pressures": {"mean_z_all_sensors_last3h": sp["mean_z_all_sensors_last3h"], "direction": _direction(float(sp["mean_z_all_sensors_last3h"]), 1.0),
                      "pattern": sp["pattern"], "fraction_anomalous": sp["fraction_anomalous"]},
        "tank_levels_z_now": dict(sorted(tanks.items())),
        "pump_flows_z_now": {k: v for k, v in sorted(pumps.items()) if abs(v) >= 1.0},
        "corroboration": {
            "n_anomalous_channels": len(anomalous), "channel_kinds_anomalous": kinds,
            "single_anomalous_channel": len(anomalous) == 1,
            "n_violated_checks": len(ev["violated"]),
            "note": "anomalous = |z| above the event-free 99th percentile for its channel kind",
        },
    }


def build_state(spec: p2.NetSpec, sim: Mapping[str, Any], we: int, cal: Mapping[str, Any], bcal: Mapping[str, Any] | None = None,
                pcal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = p2.build_state(spec, sim, we, cal)
    ev = p2.evaluate_window(spec, sim, we, cal)
    for c, entry in state["consistency_checks"].items():
        if entry["status"] == "consistent":
            entry.pop("detail", None)
            if c == "pressure_vs_twin":
                entry.pop("spread", None)
    if bcal is not None and bcal["stations"]:
        bc = balance_check(sim, we, bcal)
        state["consistency_checks"]["source_meter_vs_station_pump_flows"] = bc
        if bc["status"] == "violated":
            state["violated_checks"] = list(state["violated_checks"]) + ["source_meter_vs_station_pump_flows"]
    if pcal is not None:
        for c, entry in physics_checks(spec, sim, we, pcal).items():
            state["consistency_checks"][c] = entry
            if entry["status"] == "violated":
                state["violated_checks"] = list(state["violated_checks"]) + [c]
    state["schema_version"] = SCHEMA_VERSION
    state["system"] = SYSTEM_DESCRIPTION
    state["signed_summary"] = _round(signed_summary(spec, sim, we, cal, ev))
    # v1 check descriptions say "the twin replaying ..."; reword so the leak scan can also forbid "replay"
    return json.loads(json.dumps(state, ensure_ascii=False).replace("the twin replaying", "the twin driven by"))


def _round(v: Any) -> Any:
    return p2._round_tree(p2._safe(v))


def features(spec: p2.NetSpec, sim: Mapping[str, Any], we: int, cal: Mapping[str, Any], bcal: Mapping[str, Any] | None = None,
             pcal: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """v1 features plus the signed summaries and the station balance check shown to Jev (parity)."""

    x, ev = p2.features(spec, sim, we, cal)
    zz = ev["stats"]["z"]
    tz = [float(v[-1]) for k, v in zz.items() if p2._kind(k) == "L"]
    fz = [float(v[-1]) for k, v in zz.items() if p2._kind(k) == "F"]
    pz = [float(np.mean(v[-3:])) for k, v in zz.items() if p2._kind(k) == "P"]
    clip = lambda a: float(np.clip(a, -20, 20)) / 5.0
    extra = [clip(max(tz)), clip(min(tz)), clip(max(fz)), clip(min(fz)), clip(max(pz)), clip(min(pz)),
             float(len(ev["anomalous_channels"]) == 1), float(len({p2._kind(k) for k in ev["anomalous_channels"]}))]
    if bcal is not None and bcal["stations"]:
        bc = balance_check(sim, we, bcal)
        extra += [math.log1p(min(float(bc["statistic"]), 50.0)), float(bc["status"] == "violated")]
    if pcal is not None:
        for c, entry in physics_checks(spec, sim, we, pcal).items():
            extra += [math.log1p(min(float(entry["statistic"]) / float(entry["threshold"]), 50.0)), float(entry["status"] == "violated")]
    return np.concatenate([x, np.asarray(extra)]), ev


LEAK_TOKENS = p2._STATE_LEAK_TOKENS + ("replay", "closure", "gain_error", "pattern_shift", "spoofed")


def scan_state(state: Mapping[str, Any]) -> list[str]:
    text = json.dumps(state, ensure_ascii=False).lower()
    return [t for t in LEAK_TOKENS if t in text]


# --------------------------------------------------------------------------- #
# Prepare
# --------------------------------------------------------------------------- #


def _spec_cal(config: HydroJEVConfig) -> tuple[p2.NetSpec, dict[str, Any]]:
    cache = OUTDIR / "calibration_ctown.json"
    spec = p2.net_spec(config, "CTOWN")
    if cache.exists():
        return spec, json.loads(cache.read_text(encoding="utf-8"))
    cal = p2.calibrate(config, spec)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(p2._safe(cal)), encoding="utf-8")
    return spec, cal


def split_key(split: str, fmt: int) -> str:
    return split if fmt == 1 else f"{split}_s{fmt}"


def prepare_split(config: HydroJEVConfig, split: str, fmt: int = STATE_FORMAT) -> dict[str, Any]:
    key = split_key(split, fmt)
    target = OUTDIR / "splits" / f"{key}.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; split is frozen")
    if split in SEALED and not (OUTDIR / FREEZE_FILE[split]).exists():
        raise RuntimeError(f"sealed splits are generated only after {FREEZE_FILE[split]} exists")
    spec, cal = _spec_cal(config)
    bcal = balance_calibration(config, spec) if fmt >= 2 else None
    pcal = physics_calibration(config, spec) if fmt >= STATE_FORMAT_PHYSICS else None
    net, per_class, base, family = SPLITS[split]
    rows = []
    reqdir = OUTDIR / "requests" / key
    reqdir.mkdir(parents=True, exist_ok=True)
    for ci, cls in enumerate(CLASSES):
        for i in range(per_class):
            seed = base + 1000 * ci + i
            sc = sample_scenario(spec, cls, seed, family)
            unrealised = gated = 0
            for attempt in range(40):
                try:
                    sim = simulate(config, spec, sc, cal["sigma"])
                except ValueError:
                    unrealised += 1
                    sc = {**sample_scenario(spec, cls, seed + 100_000 * (attempt + 1), family), "seed_origin": seed}
                    continue
                if not p2.evaluate_window(spec, sim, sc["window_end_h"], cal)["violated"]:
                    gated += 1
                    sc = {**sample_scenario(spec, cls, seed + 100_000 * (attempt + 1), family), "seed_origin": seed}
                    continue
                break
            else:
                raise RuntimeError(f"could not realise {split} seed {seed}")
            we = sc["window_end_h"]
            state = build_state(spec, sim, we, cal, bcal, pcal)
            leaks = scan_state(state)
            if leaks:
                raise HydroJEVSchemaError(f"state leak tokens {leaks} in {split} seed {seed}")
            x, ev = features(spec, sim, we, cal, bcal, pcal)
            rid = f"{split}_{len(rows):04d}"
            (reqdir / f"{rid}.state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            rows.append({"id": rid, "split": split, "label_internal": cls, "scenario_internal": p2._safe({**sc, **sim["resolved"]}),
                         "redraws": {"not_realisable": unrealised, "no_violated_check": gated}, "r1": p2.r1_classify(ev),
                         "violated": state["violated_checks"], "x": [float(v) for v in x]})
            print(split, rid, cls, sc["subtype"], ev["violated"], flush=True)
    out = {"split": split, "state_format": fmt, "network": net, "family": family, "created_utc": _now(), "rows": rows}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(p2._safe(out), ensure_ascii=False), encoding="utf-8")
    return out


def _now() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def train_key(split: str) -> str:
    base, _, fmt = split.rpartition("_s")
    return f"train_s{fmt}" if base and fmt.isdigit() else "train"


def load_split(split: str) -> dict[str, Any]:
    return json.loads((OUTDIR / "splits" / f"{split}.json").read_text(encoding="utf-8"))


def request_path(split: str, rid: str, revision: int) -> Path:
    path = OUTDIR / "requests" / split / f"{rid}.r{revision}.request.json"
    if not path.exists():
        state = json.loads((OUTDIR / "requests" / split / f"{rid}.state.json").read_text(encoding="utf-8"))
        body = p1.canonical_bytes(build_request(state, revision=revision))
        problems = p1.scan_request(body)
        if problems:
            raise HydroJEVSchemaError(f"request scan {problems} for {rid}")
        path.write_bytes(body)
    return path


# --------------------------------------------------------------------------- #
# Live calls and parsing
# --------------------------------------------------------------------------- #


def stage_name(split: str, revision: int) -> str:
    return f"p2v2_{split}_q{revision}"


def run(split: str, revision: int, ids: Sequence[str] | None = None) -> None:
    rows = load_split(split)["rows"]
    st = stage_name(split, revision)
    for r in rows:
        if ids is not None and r["id"] not in ids:
            continue
        e = p1.call_live(LEDGER_DIR, request_path(split, r["id"], revision), stage=st, tag=f"{st}_{r['id']}", summarize=p2.summarize_response)
        print(r["id"], e.get("status"), e.get("event_type"), flush=True)


def jev_probs(split: str, revision: int) -> dict[str, dict[str, float]]:
    st = stage_name(split, revision)
    out = {}
    for f in (LEDGER_DIR / "responses" / st).glob("*.response.json"):
        rid = f.name.split(".")[0].replace(f"{st}_", "")
        out[rid] = p2.parse_response(json.loads(f.read_text(encoding="utf-8-sig"))["response"]).probabilities
    return out


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #


def _xy(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([r["x"] for r in rows], dtype=float), np.asarray([r["label_internal"] for r in rows])


def fit_r2(rows: Sequence[Mapping[str, Any]]) -> Any:
    x, y = _xy(rows)
    return p2.fit_r2(x, y)


def prior_from(probs: Mapping[str, Mapping[str, float]], ids: Sequence[str] | None = None) -> np.ndarray:
    """Label-free class prior: mean Jev probability vector over unlabeled windows (floored at 0.01)."""

    keys = list(probs) if ids is None else list(ids)
    m = np.asarray([[max(probs[k][c], 0.0) for c in CLASSES] for k in keys]).mean(0)
    return np.maximum(m, 0.01)


def jev_matrix(probs: Mapping[str, Mapping[str, float]], ids: Sequence[str]) -> np.ndarray:
    return np.asarray([[max(float(probs[i].get(c, 0.0)), 0.0) for c in CLASSES] for i in ids])


def _decide(p: np.ndarray, prior: np.ndarray | None = None) -> list[str]:
    q = np.maximum(p, 0.005) / (1.0 if prior is None else prior)
    return [CLASSES[k] for k in q[:, : len(CLASSES)].argmax(1)]


def fuse(pj: np.ndarray, pr: np.ndarray, prior: np.ndarray | None, w: float = 0.5) -> list[str]:
    """Geometric-mean fusion of calibrated Jev and R2 class probabilities (weight fixed a priori)."""

    qj = np.maximum(pj[:, : len(CLASSES)], 0.005) / (1.0 if prior is None else prior[: len(CLASSES)])
    qj = qj / qj.sum(1, keepdims=True)
    return [CLASSES[k] for k in (w * np.log(qj) + (1 - w) * np.log(np.maximum(pr, 1e-3))).argmax(1)]


def report(split: str, revision: int, prior: np.ndarray | None = None) -> dict[str, Any]:
    rows = load_split(split)["rows"]
    probs = jev_probs(split, revision)
    rows = [r for r in rows if r["id"] in probs]
    ids = [r["id"] for r in rows]
    y = [r["label_internal"] for r in rows]
    pj = jev_matrix(probs, ids)
    x, _ = _xy(rows)
    m2 = fit_r2(load_split(train_key(split))["rows"])
    pr = m2.predict_proba(x)
    order = [list(m2.classes_).index(c) for c in CLASSES]
    pr = pr[:, order]
    loo = []
    for i in range(len(ids)):
        mask = np.ones(len(ids), bool)
        mask[i] = False
        loo += _decide(pj[i : i + 1], np.maximum(pj[mask].mean(0), 0.01))
    preds = {"jev_raw": _decide(pj), "jev_loo_calibrated": loo, "r1": [r["r1"] for r in rows],
             "r2": [CLASSES[k] for k in pr.argmax(1)], "fusion_raw": fuse(pj, pr, None)}
    if prior is not None:
        preds["jev_prior_calibrated"] = _decide(pj, prior)
        preds["fusion_calibrated"] = fuse(pj, pr, prior)
    out = {k: p2.cause_metrics(y, v) for k, v in preds.items()}
    out = {k: {"macro_f1": round(float(v["macro_f1"]), 3), "accuracy": round(float(np.mean(np.asarray(preds[k]) == np.asarray(y))), 3)} for k, v in out.items()}
    return {"n": len(ids), "metrics": out, "jev_mean_probability": dict(zip(CLASSES, np.round(pj.mean(0), 3).tolist())),
            "windows": [{"id": i, "truth": t, "subtype": r["scenario_internal"]["subtype"], "violated": r["violated"],
                         "jev": {c: float(v) for c, v in zip(CLASSES, pj[k])}, **{n: preds[n][k] for n in preds}}
                        for k, (i, t, r) in enumerate(zip(ids, y, rows))]}


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--splits", nargs="+", default=[])
    ap.add_argument("--split")
    ap.add_argument("--revision", type=int, default=1)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--format", type=int, default=STATE_FORMAT)
    a = ap.parse_args(argv)
    if a.command == "prepare":
        cfg = load_config()
        for s in a.splits:
            prepare_split(cfg, s, a.format)
        return 0
    run(a.split, a.revision, a.ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

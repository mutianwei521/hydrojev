"""Pre-registered Jev pilot P2: event-cause discrimination on simulated ground truth.

Protocol: ``docs/jev/PILOT_PROTOCOL_v1.md`` section 3. Pilot P1 (benign false-alarm veto on
BATADAL) did not beat the rule and logistic comparators at its debug checkpoint and was stopped
under the author-approved rule; its remaining budget moves here.

Each scenario is an EPANET run of C-Town (or, zero-shot, Net3) with hourly multiplicative demand
noise and exactly one event of four classes, all starting at an undisclosed onset:

* ``cyber_attack``: (a) a pump is forced on past its high-level close interlock while the tank
  level reported to the historian is frozen (over-pumping with masking); (b) a running pump is
  forced off while the historian keeps reporting it running with its last flow.
* ``physical_fault``: a burst, i.e. an extra constant demand at one random junction.
* ``normal_transient``: a system-wide demand rise or fall, or a legitimate operator pump switch
  lasting a few hours, with honest telemetry.
* ``sensor_fault``: one historian channel (tank level, pressure or pump flow) is stuck, drifts
  or is offset. The local control loops use their own instruments (assumption stated in the
  state), so a historian fault has no hydraulic consequence.

Unmasked attacks are excluded: with honest telemetry a forced pump switch is physically
indistinguishable from a legitimate one.

Evidence comes from a digital twin: the same network with nominal demand, re-synchronised to the
reported tank levels 11 h before the window end and replaying the reported actuator states hour
by hour. Residual scales and check thresholds are calibrated on event-free noise-only runs of the
same network (label-free, so Net3 is still zero-shot). Jev, a hand-written decision tree (R1) and
a multinomial logistic regression trained on 200 C-Town scenarios (R2) all see the same features.
Labels never enter a request; they live only in ``prepared.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.config import HydroJEVConfig, load_config
from hydrojev.methods.hydrojev_detector import (
    HydroJEVSchemaError,
    _assert_no_ground_truth,
    _distribution,
    _json_safe,
    _probability,
)

SCHEMA_VERSION = "hydrojev.cause_state/1"
CLASSES = ("cyber_attack", "physical_fault", "normal_transient", "sensor_fault")
EVENT_TYPES = CLASSES + ("insufficient_evidence",)
WINDOW_H = 6
SYNC_LAG_H = 11  # twin re-synchronised this many hours before the window end (always before onset)
ONSET_RANGE_H = (24, 48)
LAG_RANGE_H = (2, 6)  # window end = onset + lag
N_CALIBRATION = 80
CAL_QUANTILE = 0.975
TOP_RESIDUALS = 8
N_MONITORS = 8
DEFAULT_OUTDIR = Path("artifacts/jev_pilot_p2")
LEDGER_DIR = p1.DEFAULT_OUTDIR  # shared ledger: the 400-call cap covers P1 and P2 together
SPLITS = {  # name: (network, n per class, seed base)
    "train": ("CTOWN", 50, 100_000),
    "eval_ctown": ("CTOWN", 10, 500_000),
    "eval_net3": ("Net3", 6, 700_000),
}
CAL_SEED = {"CTOWN": 900_000, "Net3": 950_000}
DEMAND_NOISE_SD = 0.04
LEVEL_TOL_M = 0.25  # hourly sampling: switches are triggered between samples
SIGMA_FLOOR = {"L": 0.02, "P": 0.2, "F": 0.5, "Fin": 0.5, "Fsrc": 1.0}
UNITS = {"L": "m", "P": "m", "F": "L/s", "Fin": "L/s", "Fsrc": "L/s"}
MEAS_NOISE = {"L": 0.005, "P": 0.05, "F": 0.003, "Fin": 0.003, "Fsrc": 0.003}  # F*: relative
CHECKS = (
    "pump_status_vs_flow",
    "pump_head_gain_vs_twin",
    "tank_mass_balance",
    "tank_level_vs_twin",
    "pressure_vs_twin",
    "system_demand_vs_twin",
    "pump_flow_vs_twin",
    "control_logic_vs_reported_level",
    "control_logic_vs_twin_level",
)
CHECK_DESCRIPTIONS = {
    "pump_status_vs_flow": "reported pump status contradicts the same pump's reported flow (count of pump-hours)",
    "pump_head_gain_vs_twin": "head gain across a pump (from its suction/discharge pressure sensors) versus the twin replaying the reported status (max |z|)",
    "tank_mass_balance": "hourly change of a reported tank level versus the volume from that tank's own inflow meter (max |z|)",
    "tank_level_vs_twin": "reported tank level versus the twin (max |z|)",
    "pressure_vs_twin": "junction pressures versus the twin (max |z|); see spread",
    "system_demand_vs_twin": "consumption estimated from source and tank inflow meters versus the twin's nominal consumption (|mean z| over the last 3 h)",
    "pump_flow_vs_twin": "reported pump flows versus the twin replaying the reported statuses (max |z|)",
    "control_logic_vs_reported_level": "actuator states or switches that the configured control logic cannot explain from the REPORTED tank levels or schedule (count)",
    "control_logic_vs_twin_level": "actuator states or switches that the control logic cannot explain from the TWIN-estimated tank levels (count)",
}

SYSTEM_DESCRIPTION = (
    "Water distribution network with tanks, pumps and valves, hourly SCADA historian. Channel names: "
    "L_<tank> tank level (m), S_<link> actuator status (1 = running/open), F_<link> pump flow (L/s), "
    "P_<node> junction pressure (m), Fin_<tank> tank net inflow meter (L/s, positive = filling), "
    "Fsrc_<source> source outflow meter (L/s). Pumps and valves are switched by local controllers from "
    "their own tank-level instruments or by time schedules (see control_configuration); the local "
    "controllers do not read the historian, so an error in a historian channel alone does not change "
    "pump behaviour. An attacker could alter historian readings and controller commands. The digital twin "
    "is a hydraulic model of this network with nominal demand, re-synchronised to the reported tank "
    "levels 11 h before the window end and replaying the reported actuator states hour by hour, so twin "
    "residuals show what the reported operation cannot explain. z-scores and check thresholds are "
    "calibrated on event-free operation of this network with ordinary demand variability. The window may "
    "contain an event that began before or during it."
)

EVENT_CRITERIA = {
    "cyber_attack": (
        "Control or telemetry manipulation: an actuator runs contrary to what its control logic requires "
        "given the physically estimated (twin) tank level, or a reported reading such as a tank level or a "
        "pump status/flow is contradicted by several independent, physically related measurements (e.g. a "
        "reported level that stays flat while its inflow meter and the twin show the tank filling; a pump "
        "reported running while its head gain and downstream tank show it stopped)."
    ),
    "physical_fault": (
        "A real hydraulic fault such as a pipe burst or leak: extra, unexplained consumption (demand above "
        "the twin) with a pressure drop concentrated in part of the network and tank levels below the twin, "
        "while all telemetry stays mutually consistent."
    ),
    "normal_transient": (
        "Ordinary operation: demand variation (possibly large and system-wide, moving demand and pressures "
        "everywhere in the same direction) or a legitimate operator switch of a pump or valve whose "
        "hydraulic effects agree with all measurements; telemetry mutually consistent."
    ),
    "sensor_fault": (
        "One historian instrument is stuck, drifting or offset: a single channel deviates from the twin and "
        "from physically related channels, while actuators follow their control logic and the other "
        "measurements are consistent."
    ),
    "insufficient_evidence": (
        "Use only when the evidence is missing or genuinely contradictory so that none of the other four "
        "explanations can be preferred."
    ),
}


def build_questions() -> dict[str, dict[str, Any]]:
    return {
        "event_type": {
            "type": "choice",
            "instructions": (
                "Using only the supplied evidence for this six-hour SCADA window, choose the most likely "
                "cause of what the twin residuals and consistency checks show."
            ),
            "criteria": dict(EVENT_CRITERIA),
        }
    }


def build_request(state: Mapping[str, Any], *, model: str = "jev-latest") -> dict[str, Any]:
    _assert_no_ground_truth(state)
    request = {"model": model, "state": _json_safe(dict(state), path="state"), "questions": build_questions()}
    json.dumps(request, ensure_ascii=False, allow_nan=False)
    return request


@dataclass(frozen=True)
class CauseDecision:
    event_type: str
    probabilities: dict[str, float]
    confidence: float
    model: str
    usage: dict[str, int]


def parse_response(body: Mapping[str, Any]) -> CauseDecision:
    if not isinstance(body, Mapping) or not isinstance(body.get("answers"), Mapping):
        raise HydroJEVSchemaError("response must contain an answers object")
    event = body["answers"].get("event_type")
    if not isinstance(event, Mapping) or event.get("type") != "choice":
        raise HydroJEVSchemaError("event_type answer must have type 'choice'")
    probs = _distribution({str(k): v for k, v in dict(event.get("probabilities") or {}).items()},
                          expected_keys=set(EVENT_TYPES), label="event_type.probabilities")
    choice = event.get("choice")
    if choice not in EVENT_TYPES:
        raise HydroJEVSchemaError("event_type.choice is outside the taxonomy")
    model = body.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise HydroJEVSchemaError("response model must be a non-empty string")
    usage_raw = body.get("usage", {}) if isinstance(body.get("usage", {}), Mapping) else {}
    return CauseDecision(
        event_type=str(choice), probabilities=probs,
        confidence=_probability(event.get("confidence"), label="event_type.confidence"),
        model=model, usage={k: int(usage_raw.get(k, 0) or 0) for k in ("input_tokens", "output_tokens")},
    )


def summarize_response(body: Mapping[str, Any]) -> dict[str, Any]:
    d = parse_response(body)
    return {"model": d.model, "usage": d.usage, "event_type": d.event_type, "event_probabilities": d.probabilities}


# --------------------------------------------------------------------------- #
# Network description (derived from the model, never from scenario labels)
# --------------------------------------------------------------------------- #


@dataclass
class NetSpec:
    name: str
    tanks: list[str]
    tank_area: dict[str, float]
    tank_elev: dict[str, float]
    pumps: list[str]
    actuated: list[str]
    level_rules: list[dict[str, Any]]  # {link, tank, open_below | close_above | ...}
    schedules: dict[str, list[tuple[float, int]]]  # link -> [(hour, status)]
    pump_ends: dict[str, tuple[str, str]]
    pressure_nodes: list[str]
    sources: dict[str, list[tuple[str, int]]]  # reservoir -> [(link, sign)]
    tank_links: dict[str, list[tuple[str, int]]]
    reservoir_head: dict[str, float]
    elevation: dict[str, float]
    node_meta: dict[str, dict[str, Any]]


def _load_model(config: HydroJEVConfig, network: str) -> Any:
    from hydrojev.datasets.wdn_loader import load_network

    return load_network(network, config=config).model


def _incident(wn: Any, node: str) -> list[tuple[str, int]]:
    out = []
    for name in wn.get_links_for_node(node):
        link = wn.get_link(name)
        out.append((name, +1 if link.end_node_name == node else -1))
    return out


def _control_parts(control: Any) -> tuple[Any, Any, Any]:
    action = control.actions()[0]
    return control.condition, action._target_obj, action._value


def net_spec(config: HydroJEVConfig, network: str) -> NetSpec:
    import networkx as nx
    from wntr.network.controls import SimTimeCondition, ValueCondition

    wn = _load_model(config, network)
    level_rules: dict[tuple[str, str], dict[str, Any]] = {}
    schedules: dict[str, list[tuple[float, int]]] = {}
    actuated = list(wn.pump_name_list)
    for _, control in wn.controls():
        cond, link, value = _control_parts(control)
        if link.name not in actuated:
            actuated.append(link.name)
        if isinstance(cond, SimTimeCondition):
            schedules.setdefault(link.name, []).append((float(cond._threshold) / 3600.0, int(value)))
        elif isinstance(cond, ValueCondition) and cond._source_attr == "level":
            r = level_rules.setdefault((link.name, cond._source_obj.name), {"link": link.name, "tank": cond._source_obj.name})
            key = ("close" if int(value) == 0 else "open") + ("_above" if ">" in str(cond._relation) or "gt" in str(cond._relation) else "_below")
            r[key] = float(cond._threshold)
    for s in schedules.values():
        s.sort()
    pump_ends = {p: (wn.get_link(p).start_node_name, wn.get_link(p).end_node_name) for p in wn.pump_name_list}
    g = nx.Graph(wn.to_graph().to_undirected())
    candidates = []
    for nodes in pump_ends.values():
        candidates += [n for n in nodes if n in wn.junction_name_list and n not in candidates]
    # event-free pressures pick monitoring junctions spread across the network (farthest-point sampling)
    import wntr

    tmp = tempfile.mkdtemp(prefix="p2spec_")
    try:
        wn.options.time.duration = 24 * 3600
        wn.options.time.report_timestep = 3600
        res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(tmp) / "spec"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pmed = res.node["pressure"].median()
    pool = sorted(j for j in wn.junction_name_list if pmed[j] > 15.0 and j not in candidates)
    hops = dict(nx.all_pairs_shortest_path_length(g)) if len(g) < 600 else None
    monitors: list[str] = []
    anchor = list(candidates) + list(wn.tank_name_list)
    while len(monitors) < N_MONITORS and pool:
        chosen = anchor + monitors
        best = max(pool, key=lambda j: (min(hops[j][c] for c in chosen), j))
        monitors.append(best)
        pool.remove(best)
    pressure_nodes = candidates + monitors
    node_meta = {}
    for n in pressure_nodes:
        tank = min(wn.tank_name_list, key=lambda t: (hops[n][t], t))
        role = [f"{'suction' if pump_ends[p][0] == n else 'discharge'} of pump {p}" for p in pump_ends if n in pump_ends[p]]
        node_meta[n] = {"role": ", ".join(role) if role else "monitoring junction", "nearest_tank": tank, "hops_to_nearest_tank": int(hops[n][tank])}
    sources = {r: _incident(wn, r) for r in wn.reservoir_name_list}
    return NetSpec(
        name=network,
        tanks=list(wn.tank_name_list),
        tank_area={t: math.pi * float(wn.get_node(t).diameter) ** 2 / 4.0 for t in wn.tank_name_list},
        tank_elev={t: float(wn.get_node(t).elevation) for t in wn.tank_name_list},
        pumps=list(wn.pump_name_list), actuated=actuated, level_rules=list(level_rules.values()),
        schedules=schedules, pump_ends=pump_ends, pressure_nodes=pressure_nodes, sources=sources,
        tank_links={t: _incident(wn, t) for t in wn.tank_name_list},
        reservoir_head={r: float(wn.get_node(r).base_head) for r in wn.reservoir_name_list},
        elevation={n: float(wn.get_node(n).elevation) for n in wn.junction_name_list},
        node_meta=node_meta,
    )


def control_configuration(spec: NetSpec) -> list[str]:
    out = []
    for r in spec.level_rules:
        kind = "pump" if r["link"] in spec.pumps else "link"
        parts = []
        if "open_below" in r:
            parts.append(f"{'starts' if kind == 'pump' else 'opens'} when L_{r['tank']} < {r['open_below']:.2f} m")
        if "close_above" in r:
            parts.append(f"{'stops' if kind == 'pump' else 'closes'} when L_{r['tank']} > {r['close_above']:.2f} m")
        if "close_below" in r:
            parts.append(f"closes when L_{r['tank']} < {r['close_below']:.2f} m")
        if "open_above" in r:
            parts.append(f"opens when L_{r['tank']} > {r['open_above']:.2f} m")
        out.append(f"{kind} {r['link']}: " + "; ".join(parts))
    for link, sched in spec.schedules.items():
        on = sorted({h % 24 for h, s in sched if s == 1})
        off = sorted({h % 24 for h, s in sched if s == 0})
        out.append(f"pump {link}: time schedule, on at {', '.join(f'{h:02.0f}:00' for h in on)}, off at {', '.join(f'{h:02.0f}:00' for h in off)} daily")
    free = [a for a in spec.actuated if not any(r["link"] == a for r in spec.level_rules) and a not in spec.schedules]
    if free:
        out.append("no automatic control (operator only): " + ", ".join(free))
    return out


# --------------------------------------------------------------------------- #
# Scenario model building and simulation
# --------------------------------------------------------------------------- #


def _override(wn: Any, link_name: str, t0_h: int, t1_h: int | None, status: int, tag: str) -> None:
    """Take ``link_name`` out of automatic control over [t0, t1) and force ``status`` at t0."""

    from wntr.network.base import LinkStatus
    from wntr.network.controls import AndCondition, Control, ControlAction, Rule, SimTimeCondition, ValueCondition

    for name, control in list(wn.controls()):
        cond, link, value = _control_parts(control)
        if link.name != link_name:
            continue
        wn.remove_control(name)
        if isinstance(cond, SimTimeCondition):
            h = float(cond._threshold) / 3600.0
            if h < t0_h or (t1_h is not None and h >= t1_h):
                wn.add_control(name.replace(" ", "_"), Control(cond, ControlAction(link, "status", value), name=name.replace(" ", "_")))
            continue
        base = ValueCondition(cond._source_obj, cond._source_attr, cond._relation, cond._threshold)
        stem = name.replace(" ", "_")
        wn.add_control(f"{stem}_pre", Rule(AndCondition(base, SimTimeCondition(wn, "<", t0_h * 3600)), [ControlAction(link, "status", value)], name=f"{stem}_pre"))
        if t1_h is not None:
            base2 = ValueCondition(cond._source_obj, cond._source_attr, cond._relation, cond._threshold)
            wn.add_control(f"{stem}_post", Rule(AndCondition(base2, SimTimeCondition(wn, ">=", t1_h * 3600)), [ControlAction(link, "status", value)], name=f"{stem}_post"))
    link = wn.get_link(link_name)
    wn.add_control(f"{tag}_force", Control(SimTimeCondition(wn, "=", t0_h * 3600), ControlAction(link, "status", LinkStatus.Open if status else LinkStatus.Closed), name=f"{tag}_force"))


def _run(wn: Any) -> Any:
    import wntr

    tmp = tempfile.mkdtemp(prefix="p2run_")
    try:
        return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(tmp) / "run"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _noisy_model(config: HydroJEVConfig, spec: NetSpec, rng: np.random.Generator, hours: int, *, demand_factor: tuple[int, float] | None = None) -> Any:
    wn = _load_model(config, spec.name)
    n = hours + 2
    scale = float(rng.uniform(0.95, 1.05))
    for pname in list(wn.pattern_name_list):
        pat = wn.get_pattern(pname)
        base = np.asarray(pat.multipliers, dtype=float)
        if len(base) < 2:
            continue
        mult = np.array([base[h % len(base)] for h in range(n)]) * scale * (1.0 + rng.normal(0.0, DEMAND_NOISE_SD, n))
        if demand_factor is not None:
            t0, f = demand_factor
            mult[t0:] *= f
        pat.multipliers = list(np.clip(mult, 0.0, None))
    for t in spec.tanks:
        wn.get_node(t).overflow = True
    wn.options.time.duration = int(hours * 3600)
    wn.options.time.report_timestep = 3600
    return wn


def _true_channels(res: Any, wn: Any, spec: NetSpec) -> dict[str, np.ndarray]:
    ch: dict[str, np.ndarray] = {}
    head = res.node["head"]
    flow = res.link["flowrate"]
    status = res.link["status"]
    for t in spec.tanks:
        ch[f"L_{t}"] = head[t].values - spec.tank_elev[t]
        ch[f"Fin_{t}"] = sum(sign * flow[l].values for l, sign in spec.tank_links[t]) * 1000.0
    for a in spec.actuated:
        ch[f"S_{a}"] = (np.asarray(status[a].values, dtype=float) > 0.5).astype(float)
    for p in spec.pumps:
        ch[f"F_{p}"] = flow[p].values * 1000.0
    for n in spec.pressure_nodes:
        ch[f"P_{n}"] = res.node["pressure"][n].values
    for r, links in spec.sources.items():
        ch[f"Fsrc_{r}"] = -sum(sign * flow[l].values for l, sign in links) * 1000.0
    return {k: np.asarray(v, dtype=float) for k, v in ch.items()}


def _kind(channel: str) -> str:
    return channel.split("_", 1)[0]


def _measure(ch: Mapping[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for k, v in ch.items():
        kind = _kind(k)
        if kind == "S":
            out[k] = v.copy()
            continue
        sd = MEAS_NOISE[kind] * (np.abs(v) + 1.0) if kind.startswith("F") else MEAS_NOISE[kind]
        out[k] = np.round(v + rng.normal(0.0, 1.0, v.shape) * sd, 2)
    return out


def twin_channels(config: HydroJEVConfig, spec: NetSpec, reported: Mapping[str, np.ndarray], window_end: int) -> dict[str, np.ndarray]:
    """Nominal-demand twin re-synchronised at ``window_end - SYNC_LAG_H`` and replaying reported states."""

    from wntr.network.base import LinkStatus
    from wntr.network.controls import Control, ControlAction, SimTimeCondition

    sync = window_end - SYNC_LAG_H
    wn = _load_model(config, spec.name)
    for name, _ in list(wn.controls()):
        wn.remove_control(name)
    for t in spec.tanks:
        node = wn.get_node(t)
        node.init_level = float(np.clip(reported[f"L_{t}"][sync], node.min_level + 1e-3, node.max_level - 1e-3))
        node.overflow = True
    for a in spec.actuated:
        s = reported[f"S_{a}"]
        link = wn.get_link(a)
        link.initial_status = LinkStatus.Open if s[sync] > 0.5 else LinkStatus.Closed
        for k in range(1, SYNC_LAG_H + 1):
            if s[sync + k] != s[sync + k - 1]:
                st = LinkStatus.Open if s[sync + k] > 0.5 else LinkStatus.Closed
                wn.add_control(f"replay_{a}_{k}", Control(SimTimeCondition(wn, "=", k * 3600), ControlAction(link, "status", st), name=f"replay_{a}_{k}"))
    wn.options.time.pattern_start = int(sync * 3600)
    wn.options.time.duration = int(SYNC_LAG_H * 3600)
    wn.options.time.report_timestep = 3600
    res = _run(wn)
    twin = _true_channels(res, wn, spec)
    pad = np.full(sync, np.nan)
    return {k: np.concatenate([pad, v]) for k, v in twin.items()}


def sample_scenario(spec: NetSpec, cls: str | None, seed: int, sigma: Mapping[str, float] | None) -> dict[str, Any]:
    """Draw scenario parameters (class ``None`` = event-free calibration run)."""

    rng = np.random.default_rng(seed)
    t0 = int(rng.integers(ONSET_RANGE_H[0], ONSET_RANGE_H[1] + 1))
    we = t0 + int(rng.integers(LAG_RANGE_H[0], LAG_RANGE_H[1] + 1))
    sc: dict[str, Any] = {"seed": seed, "network": spec.name, "cls": cls, "onset_h": t0, "window_end_h": we}
    if cls is None:
        return sc
    closable = [r for r in spec.level_rules if "close_above" in r and r["link"] in spec.pumps]
    if cls == "cyber_attack":
        sc["subtype"] = str(rng.choice(["overpump_masked_level", "pump_off_spoofed_running"]))
        if sc["subtype"] == "overpump_masked_level":
            r = closable[int(rng.integers(len(closable)))]
            sc.update(pump=r["link"], tank=r["tank"])
    elif cls == "physical_fault":
        sc["subtype"] = "burst"
        sc["burst_fraction"] = float(rng.uniform(0.06, 0.15))
    elif cls == "normal_transient":
        sc["subtype"] = str(rng.choice(["demand_rise", "demand_fall", "operator_switch"]))
        if sc["subtype"] == "demand_rise":
            sc["factor"] = float(rng.uniform(1.15, 1.35))
        elif sc["subtype"] == "demand_fall":
            sc["factor"] = float(rng.uniform(0.65, 0.85))
        else:
            sc["duration_h"] = int(rng.integers(2, 6))
    elif cls == "sensor_fault":
        sc["subtype"] = str(rng.choice(["stuck", "drift", "offset"]))
        sc["kind"] = str(rng.choice(["L", "P", "F"]))
        sc["magnitude_sigma"] = float(rng.uniform(4.0, 10.0)) * float(rng.choice([-1.0, 1.0]))
    sc["pick"] = float(rng.random())  # deterministic choice among candidates found in the event-free run
    return sc


def simulate(config: HydroJEVConfig, spec: NetSpec, sc: dict[str, Any], sigma: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Return reported (historian) channels, twin channels and the resolved event parameters."""

    import networkx as nx

    rng = np.random.default_rng(sc["seed"] + 1)
    t0, we = sc["onset_h"], sc["window_end_h"]
    state_seed = rng.integers(1 << 31)
    wn = _noisy_model(config, spec, np.random.default_rng(state_seed), we)
    base_res = _run(wn)
    base = _true_channels(base_res, wn, spec)
    cls, sub = sc.get("cls"), sc.get("subtype")
    resolved: dict[str, Any] = {}

    def pick(options: Sequence[str]) -> str:
        return sorted(options)[min(int(sc["pick"] * len(options)), len(options) - 1)]

    true = base
    if cls is not None and not (cls == "sensor_fault"):
        factor = (t0, sc["factor"]) if sub in ("demand_rise", "demand_fall") else None
        wn = _noisy_model(config, spec, np.random.default_rng(state_seed), we, demand_factor=factor)
        if sub == "overpump_masked_level":
            _override(wn, sc["pump"], t0, None, 1, "atk")
            if spec.name == "Net3":  # the tank is also relieved by a pressure-zone bypass under level control
                for r in spec.level_rules:
                    if r["tank"] == sc["tank"] and r["link"] not in spec.pumps:
                        _override(wn, r["link"], t0, None, 0, f"atk_{r['link']}")
            resolved.update(pump=sc["pump"], tank=sc["tank"])
        elif sub == "pump_off_spoofed_running":
            running = [p for p in spec.pumps if base[f"S_{p}"][t0 - 1] > 0.5 and base[f"F_{p}"][t0 - 1] > 1.0]
            if not running:
                raise ValueError("no running pump at onset")
            p = pick(running)
            _override(wn, p, t0, None, 0, "atk")
            resolved.update(pump=p)
        elif sub == "operator_switch":
            p = pick([p for p in spec.pumps if p in [r["link"] for r in spec.level_rules] or p in spec.schedules])
            new = 0 if base[f"S_{p}"][t0 - 1] > 0.5 else 1
            _override(wn, p, t0, t0 + sc["duration_h"], new, "op")
            resolved.update(pump=p, forced_status=new)
        elif sub == "burst":
            g = nx.Graph(wn.to_graph().to_undirected())
            del g
            pres = base_res.node["pressure"].median()
            pool = [j for j in wn.junction_name_list if pres[j] > 20.0 and j not in spec.pressure_nodes]
            j = pick(pool)
            total = float(base_res.node["demand"][wn.junction_name_list].sum(axis=1).mean())
            q = sc["burst_fraction"] * total
            wn.add_pattern("evt_step", [0.0] * t0 + [1.0] * (we - t0 + 3))
            wn.get_node(j).add_demand(q, "evt_step")
            resolved.update(junction=j, burst_lps=q * 1000.0)
        res = _run(wn)
        true = _true_channels(res, wn, spec)
        if sub == "overpump_masked_level":
            # the class is defined by a defeated interlock: the pump must actually run past its close setpoint
            close = next(r["close_above"] for r in spec.level_rules if r["link"] == sc["pump"] and r["tank"] == sc["tank"])
            if not np.any((true[f"L_{sc['tank']}"][_win(we)] > close + LEVEL_TOL_M) & (true[f"S_{sc['pump']}"][_win(we)] > 0.5)):
                raise ValueError("interlock not yet defeated inside the window")
    reported = _measure(true, np.random.default_rng(state_seed + 7))
    if sub == "overpump_masked_level":
        reported[f"L_{sc['tank']}"][t0:] = reported[f"L_{sc['tank']}"][t0 - 1]
    elif sub == "pump_off_spoofed_running":
        p = resolved["pump"]
        reported[f"S_{p}"][t0:] = reported[f"S_{p}"][t0 - 1]
        reported[f"F_{p}"][t0:] = reported[f"F_{p}"][t0 - 1]
    elif cls == "sensor_fault":
        kind = sc["kind"]
        if kind == "F":
            options = [f"F_{p}" for p in spec.pumps if base[f"S_{p}"][t0 - 1] > 0.5 and base[f"F_{p}"][t0 - 1] > 1.0]
        else:
            options = [k for k in reported if _kind(k) == kind]
        if not options:
            raise ValueError("no eligible channel")
        chan = pick(options)
        mag = sc["magnitude_sigma"] * float((sigma or {}).get(chan, SIGMA_FLOOR[kind]))
        x = reported[chan]
        if sub == "stuck":
            x[t0:] = x[t0 - 1]
        elif sub == "drift":
            x[t0:] = x[t0:] + mag * (np.arange(1, len(x) - t0 + 1) / 4.0)
        else:
            x[t0:] = x[t0:] + mag
        if kind == "L":
            x[:] = np.clip(x, 0.0, None)
        reported[chan] = np.round(x, 2)
        scale = float((sigma or {}).get(chan, SIGMA_FLOOR[kind]))
        if abs(reported[chan][we] - true[chan][we]) < 3.0 * scale:
            raise ValueError("faulty reading indistinguishable from the true value")  # e.g. stuck on a flat signal
        resolved.update(channel=chan, offset_units=mag)
    twin = twin_channels(config, spec, reported, we)
    return {"reported": reported, "twin": twin, "resolved": resolved}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _win(we: int) -> slice:
    return slice(we - WINDOW_H + 1, we + 1)


def _gain(spec: NetSpec, ch: Mapping[str, np.ndarray], pump: str) -> np.ndarray | None:
    def head(node: str) -> np.ndarray | None:
        if node in spec.reservoir_head:
            return np.full_like(ch[f"S_{pump}"], spec.reservoir_head[node])
        if f"P_{node}" in ch and node in spec.elevation:
            return ch[f"P_{node}"] + spec.elevation[node]
        return None

    s, e = spec.pump_ends[pump]
    hs, he = head(s), head(e)
    return None if hs is None or he is None else he - hs


def raw_statistics(spec: NetSpec, rep: Mapping[str, np.ndarray], twin: Mapping[str, np.ndarray], we: int, cal: Mapping[str, Any] | None) -> dict[str, Any]:
    """Window statistics. ``cal`` supplies residual scales; ``None`` = unit scale (used to calibrate)."""

    w = _win(we)
    sig = (cal or {}).get("sigma", {})
    on_flow = (cal or {}).get("on_flow", {})

    def z(name: str, obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return (obs - pred) / float(sig.get(name, SIGMA_FLOOR.get(_kind(name), 1.0)))

    out: dict[str, Any] = {"z": {}, "detail": {}}
    for k in rep:
        if _kind(k) != "S":
            out["z"][k] = z(k, rep[k][w], twin[k][w])
    # pump status vs flow
    viol = []
    for p in spec.pumps:
        ref = float(on_flow.get(p, 5.0))
        s, f = rep[f"S_{p}"][w], rep[f"F_{p}"][w]
        bad = ((s > 0.5) & (f < 0.1 * ref)) | ((s < 0.5) & (f > 0.1 * ref))
        if bad.any():
            viol.append({"pump": p, "hours": int(bad.sum())})
    out["pump_status_vs_flow"] = float(sum(v["hours"] for v in viol))
    out["detail"]["pump_status_vs_flow"] = viol
    # head gain
    best, det = 0.0, None
    for p in spec.pumps:
        go, gt = _gain(spec, rep, p), _gain(spec, twin, p)
        if go is None:
            continue
        zz = z(f"G_{p}", go[w], gt[w])
        i = int(np.nanargmax(np.abs(zz)))
        if abs(zz[i]) > best:
            best, det = float(abs(zz[i])), {"pump": p, "z": float(zz[i]), "reported_status_now": int(rep[f"S_{p}"][we])}
    out["pump_head_gain_vs_twin"] = best
    out["detail"]["pump_head_gain_vs_twin"] = det
    # tank mass balance on reported channels only
    best, det = 0.0, None
    for t in spec.tanks:
        lv, fin = rep[f"L_{t}"], rep[f"Fin_{t}"]
        idx = np.arange(we - WINDOW_H + 1, we + 1)
        r = (lv[idx] - lv[idx - 1]) - 0.5 * (fin[idx] + fin[idx - 1]) * 3.6 / spec.tank_area[t]
        zz = r / float(sig.get(f"MB_{t}", SIGMA_FLOOR["L"]))
        i = int(np.argmax(np.abs(zz)))
        if abs(zz[i]) > best:
            best, det = float(abs(zz[i])), {"tank": t, "z": float(zz[i]), "reported_level_change_m": float(lv[idx][i] - lv[idx - 1][i]), "meter_implied_change_m": float(0.5 * (fin[idx][i] + fin[idx - 1][i]) * 3.6 / spec.tank_area[t])}
    out["tank_mass_balance"] = best
    out["detail"]["tank_mass_balance"] = det

    def zmax(kind: str) -> tuple[float, str | None]:
        items = [(float(np.max(np.abs(v))), k) for k, v in out["z"].items() if _kind(k) == kind]
        return max(items) if items else (0.0, None)

    out["tank_level_vs_twin"], lt = zmax("L")
    out["detail"]["tank_level_vs_twin"] = {"channel": lt, "z_now": float(out["z"][lt][-1])} if lt else None
    out["pressure_vs_twin"], pt = zmax("P")
    recent = {k: float(np.mean(v[-3:])) for k, v in out["z"].items() if _kind(k) == "P"}
    out["pump_flow_vs_twin"], ft = zmax("F")
    out["detail"]["pump_flow_vs_twin"] = {"channel": ft, "z_now": float(out["z"][ft][-1])} if ft else None
    # system demand
    def demand(ch: Mapping[str, np.ndarray]) -> np.ndarray:
        return sum(ch[f"Fsrc_{r}"] for r in spec.sources) - sum(ch[f"Fin_{t}"] for t in spec.tanks)

    dz = z("D", demand(rep)[w], demand(twin)[w])
    out["system_demand_vs_twin"] = float(abs(np.mean(dz[-3:])))
    dt = demand(twin)[w]
    out["detail"]["system_demand_vs_twin"] = {"mean_z_last3h": float(np.mean(dz[-3:])), "relative_change_last3h": float(np.mean(demand(rep)[w][-3:] - dt[-3:]) / max(1e-6, float(np.mean(np.abs(dt[-3:])))))}
    out["_pressure_recent"] = recent
    # control logic
    for which, levels in (("control_logic_vs_reported_level", rep), ("control_logic_vs_twin_level", twin)):
        events = []
        for r in spec.level_rules:
            s, lv = rep[f"S_{r['link']}"], levels[f"L_{r['tank']}"]
            for h in range(we - WINDOW_H + 1, we + 1):
                on, prev = s[h] > 0.5, s[h - 1] > 0.5
                l_now, l_prev = lv[h], lv[h - 1]
                if not np.isfinite(l_now) or not np.isfinite(l_prev):
                    continue
                hi, lo = r.get("close_above", r.get("open_above")), r.get("open_below", r.get("close_below"))
                pump_like = "close_above" in r or "open_below" in r
                run_state = on if pump_like else not on
                if hi is not None and run_state and l_now > hi + LEVEL_TOL_M:
                    events.append({"link": r["link"], "hour_offset": h - we, "issue": f"{'running' if pump_like else 'closed'} while L_{r['tank']} is above its {hi:.2f} m setpoint"})
                if lo is not None and not run_state and l_now < lo - LEVEL_TOL_M:
                    events.append({"link": r["link"], "hour_offset": h - we, "issue": f"{'stopped' if pump_like else 'open'} while L_{r['tank']} is below its {lo:.2f} m setpoint"})
                if on != prev:
                    started = run_state and not (prev if pump_like else not prev)
                    if started and lo is not None and min(l_now, l_prev) > lo + LEVEL_TOL_M:
                        events.append({"link": r["link"], "hour_offset": h - we, "issue": f"switched to {'running' if pump_like else 'closed'} although L_{r['tank']} stayed above {lo:.2f} m"})
                    if not started and hi is not None and max(l_now, l_prev) < hi - LEVEL_TOL_M:
                        events.append({"link": r["link"], "hour_offset": h - we, "issue": f"switched to {'stopped' if pump_like else 'open'} although L_{r['tank']} stayed below {hi:.2f} m"})
        if which == "control_logic_vs_reported_level":
            for link, sched in spec.schedules.items():
                s = rep[f"S_{link}"]
                for h in range(we - WINDOW_H + 1, we + 1):
                    past = [st for hh, st in sched if hh % 24 <= h % 24]
                    expected = (past or [st for _, st in sched])[-1] if sched else None
                    if expected is not None and int(s[h] > 0.5) != expected:
                        events.append({"link": link, "hour_offset": h - we, "issue": f"state {int(s[h] > 0.5)} contradicts its time schedule"})
            ruled = {r["link"] for r in spec.level_rules} | set(spec.schedules)
            for a in spec.actuated:
                if a in ruled:
                    continue
                s = rep[f"S_{a}"]
                for h in range(we - WINDOW_H + 1, we + 1):
                    if s[h] != s[h - 1]:
                        events.append({"link": a, "hour_offset": h - we, "issue": "switched although it has no automatic control"})
        out[which] = float(len(events))
        out["detail"][which] = events[:6]
    return out


def calibrate(config: HydroJEVConfig, spec: NetSpec, n: int = N_CALIBRATION) -> dict[str, Any]:
    """Residual scales, on-flows and check thresholds from event-free noise-only runs."""

    runs = []
    for i in range(n):
        sc = sample_scenario(spec, None, CAL_SEED[spec.name] + i, None)
        sim = simulate(config, spec, sc)
        runs.append((sc, sim))
    resid: dict[str, list[float]] = {}
    on_flow: dict[str, list[float]] = {}
    for sc, sim in runs:
        rep, tw, we = sim["reported"], sim["twin"], sc["window_end_h"]
        w = _win(we)
        for k in rep:
            if _kind(k) != "S":
                resid.setdefault(k, []).extend((rep[k][w] - tw[k][w]).tolist())
        for p in spec.pumps:
            g1, g2 = _gain(spec, rep, p), _gain(spec, tw, p)
            if g1 is not None:
                resid.setdefault(f"G_{p}", []).extend((g1[w] - g2[w]).tolist())
            on = rep[f"S_{p}"] > 0.5
            on_flow.setdefault(p, []).extend(rep[f"F_{p}"][on].tolist())
        for t in spec.tanks:
            lv, fin = rep[f"L_{t}"], rep[f"Fin_{t}"]
            idx = np.arange(we - WINDOW_H + 1, we + 1)
            resid.setdefault(f"MB_{t}", []).extend(((lv[idx] - lv[idx - 1]) - 0.5 * (fin[idx] + fin[idx - 1]) * 3.6 / spec.tank_area[t]).tolist())
        d_rep = sum(rep[f"Fsrc_{r}"] for r in spec.sources) - sum(rep[f"Fin_{t}"] for t in spec.tanks)
        d_tw = sum(tw[f"Fsrc_{r}"] for r in spec.sources) - sum(tw[f"Fin_{t}"] for t in spec.tanks)
        resid.setdefault("D", []).extend((d_rep[w] - d_tw[w]).tolist())
    sigma = {}
    for k, v in resid.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        mad = 1.4826 * float(np.median(np.abs(a - np.median(a)))) if a.size else 0.0
        kind = _kind(k)
        floor = SIGMA_FLOOR.get(kind, SIGMA_FLOOR["L"] if kind == "MB" else (SIGMA_FLOOR["P"] if kind == "G" else SIGMA_FLOOR["Fsrc"]))
        sigma[k] = max(mad, float(np.std(a)) * 0.5 if a.size else 0.0, floor)
    cal = {"sigma": sigma, "on_flow": {p: float(np.median(v)) if v else 5.0 for p, v in on_flow.items()}}
    stats = [raw_statistics(spec, sim["reported"], sim["twin"], sc["window_end_h"], cal) for sc, sim in runs]
    thresholds, rates = {}, {}
    for c in CHECKS:
        vals = np.asarray([s[c] for s in stats], dtype=float)
        thr = float(np.quantile(vals, CAL_QUANTILE))
        if c in ("pump_status_vs_flow", "control_logic_vs_reported_level", "control_logic_vs_twin_level"):
            thr = max(thr, 0.0)
        else:
            thr = max(thr, 3.0)
        thresholds[c] = thr
        rates[c] = float(np.mean(vals > thr))
    kinds = {}
    for kind in ("L", "P", "F", "Fin", "Fsrc"):
        vals = [float(np.max(np.abs(v))) for s in stats for k, v in s["z"].items() if _kind(k) == kind]
        kinds[kind] = max(3.0, float(np.quantile(vals, 0.99))) if vals else 3.0
    cal.update(thresholds=thresholds, violation_rate=rates, channel_z_threshold=kinds, n_runs=n)
    return cal


# --------------------------------------------------------------------------- #
# Evidence state, features, comparators
# --------------------------------------------------------------------------- #


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _round_tree(value: Any, nd: int = 2) -> Any:
    if isinstance(value, dict):
        return {k: _round_tree(v, nd) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_tree(v, nd) for v in value]
    return round(value, nd) if isinstance(value, float) else value


def _r(x: float, nd: int = 2) -> float:
    return float(round(float(x), nd))


def evaluate_window(spec: NetSpec, sim: Mapping[str, Any], we: int, cal: Mapping[str, Any]) -> dict[str, Any]:
    st = raw_statistics(spec, sim["reported"], sim["twin"], we, cal)
    thr, kz = cal["thresholds"], cal["channel_z_threshold"]
    violated = [c for c in CHECKS if st[c] > thr[c]]
    anomalous = sorted(k for k, v in st["z"].items() if float(np.max(np.abs(v))) > kz[_kind(k)])
    recent = st["_pressure_recent"]
    p_anom = {k: v for k, v in recent.items() if abs(v) > kz["P"]}
    frac = len(p_anom) / max(1, len(recent))
    spread = {
        "n_pressure_sensors": len(recent), "n_anomalous_last3h": len(p_anom), "fraction_anomalous": _r(frac),
        "mean_z_all_sensors_last3h": _r(float(np.mean(list(recent.values()))) if recent else 0.0),
        "pattern": "none" if not p_anom else ("widespread" if frac >= 0.5 else "localized"),
        "anomalous_sensors": [{"channel": k, "mean_z_last3h": _r(v), **spec.node_meta[k[2:]]} for k, v in sorted(p_anom.items(), key=lambda kv: -abs(kv[1]))[:4]],
    }
    return {"stats": st, "violated": violated, "anomalous_channels": anomalous, "pressure_spread": spread}


def build_state(spec: NetSpec, sim: Mapping[str, Any], we: int, cal: Mapping[str, Any]) -> dict[str, Any]:
    ev = evaluate_window(spec, sim, we, cal)
    st, thr = ev["stats"], cal["thresholds"]
    rep, twin = sim["reported"], sim["twin"]
    w = _win(we)
    checks = {}
    for c in CHECKS:
        entry = {"status": "violated" if c in ev["violated"] else "consistent", "statistic": _r(st[c]), "threshold": _r(thr[c]),
                 "event_free_violation_rate": _r(cal["violation_rate"][c], 3), "meaning": CHECK_DESCRIPTIONS[c]}
        d = st["detail"].get(c)
        if c == "pressure_vs_twin":
            entry["spread"] = ev["pressure_spread"]
        elif d:
            entry["detail"] = _round_tree(_json_safe(d, path=c))
        checks[c] = entry
    ranked = sorted(st["z"].items(), key=lambda kv: -float(np.max(np.abs(kv[1]))))[:TOP_RESIDUALS]
    residuals = []
    for k, zz in ranked:
        item = {"channel": k, "unit": UNITS[_kind(k)], "reported": [_r(x) for x in rep[k][w]], "twin": [_r(x) for x in twin[k][w]],
                "z": [_r(x, 1) for x in zz], "anomalous": k in ev["anomalous_channels"]}
        if _kind(k) == "P":
            item.update(spec.node_meta[k[2:]])
        residuals.append(item)
    counts = {kind: sum(1 for k in ev["anomalous_channels"] if _kind(k) == kind) for kind in ("L", "P", "F", "Fin", "Fsrc")}
    state = {
        "schema_version": SCHEMA_VERSION,
        "system": SYSTEM_DESCRIPTION,
        "network": {"tanks": len(spec.tanks), "pumps": len(spec.pumps), "pressure_sensors": len(spec.pressure_nodes), "sources": len(spec.sources)},
        "window": {"length_h": WINDOW_H, "hour_of_day_at_end": int(we % 24), "sampling": "hourly, index 0 = oldest, 5 = latest"},
        "control_configuration": control_configuration(spec),
        "reported_actuator_states": {a: "".join(str(int(x)) for x in rep[f"S_{a}"][w]) for a in spec.actuated},
        "consistency_checks": checks,
        "violated_checks": list(ev["violated"]),
        "anomalous_channel_counts": counts,
        "largest_twin_residuals": residuals,
        "provenance": {"labels_removed": True, "twin_sync_hours_before_window_end": SYNC_LAG_H, "calibration_runs": int(cal["n_runs"])},
    }
    return state


def features(spec: NetSpec, sim: Mapping[str, Any], we: int, cal: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    ev = evaluate_window(spec, sim, we, cal)
    st, thr = ev["stats"], cal["thresholds"]
    x = []
    for c in CHECKS:
        x.append(math.log1p(min(st[c] / max(thr[c], 1.0), 50.0)))
        x.append(float(c in ev["violated"]))
    counts = [sum(1 for k in ev["anomalous_channels"] if _kind(k) == kind) for kind in ("L", "P", "F", "Fin", "Fsrc")]
    sp = ev["pressure_spread"]
    x += [float(len(ev["anomalous_channels"]))] + [float(c) for c in counts]
    x += [sp["fraction_anomalous"], float(np.clip(sp["mean_z_all_sensors_last3h"], -20, 20)) / 5.0,
          float(np.clip(st["detail"]["system_demand_vs_twin"]["mean_z_last3h"], -20, 20)) / 5.0]
    return np.asarray(x, dtype=float), ev


def r1_classify(ev: Mapping[str, Any]) -> str:
    """Hand-written decision tree over the same evidence (frozen before any Jev P2 call)."""

    v = set(ev["violated"])
    n_anom = len(ev["anomalous_channels"])
    spread = ev["pressure_spread"]["pattern"]
    if "system_demand_vs_twin" in v and spread == "widespread":
        return "normal_transient"
    if "pump_head_gain_vs_twin" in v and "tank_level_vs_twin" in v:
        return "cyber_attack"
    if "tank_mass_balance" in v:
        return "cyber_attack" if "control_logic_vs_twin_level" in v else "sensor_fault"
    if "pump_status_vs_flow" in v or (n_anom == 1 and "system_demand_vs_twin" not in v):
        return "sensor_fault"
    if "system_demand_vs_twin" in v or spread == "localized":
        return "physical_fault"
    return "normal_transient"


def fit_r2(x: np.ndarray, y: Sequence[str]) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)).fit(x, list(y))


def cause_metrics(y_true: Sequence[str], y_pred: Sequence[str], probs: Sequence[Mapping[str, float]] | None = None) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix, f1_score

    labels = list(CLASSES)
    out: dict[str, Any] = {
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])),
        "abstention_rate": float(np.mean([p == "insufficient_evidence" for p in y_pred])),
        "per_class_recall": {c: float(np.mean([p == c for t, p in zip(y_true, y_pred) if t == c])) if any(t == c for t in y_true) else None for c in labels},
        "confusion_rows_true_cols_pred": {"labels": list(EVENT_TYPES), "matrix": confusion_matrix(y_true, y_pred, labels=list(EVENT_TYPES)).tolist()},
    }
    if probs is not None:
        out["brier"] = float(np.mean([sum((float(pr.get(c, 0.0)) - float(t == c)) ** 2 for c in EVENT_TYPES) for t, pr in zip(y_true, probs)]))
    return out


# --------------------------------------------------------------------------- #
# Prepare (no network), dry run, live stages
# --------------------------------------------------------------------------- #

_STATE_LEAK_TOKENS = ("cyber", "attack", "burst", "leak", "spoof", "stuck", "drift", "offset_units", "surge", "scenario", "label_", "subtype",
                      "ctown", "c-town", "net3", "batadal", "evt_step", "overpump", "masked", "operator_switch", "sensor_fault", "physical_fault", "normal_transient")


def scan_state(state: Mapping[str, Any]) -> list[str]:
    # the fixed system description (identical in every request) names the threat model; scan the rest
    text = json.dumps({k: v for k, v in state.items() if k != "system"}, ensure_ascii=False).lower()
    return [t for t in _STATE_LEAK_TOKENS if t in text]


def _scenarios(spec: NetSpec, split: str) -> list[dict[str, Any]]:
    network, per_class, base = SPLITS[split]
    out = []
    for ci, cls in enumerate(CLASSES):
        for i in range(per_class):
            out.append(sample_scenario(spec, cls, base + 1000 * ci + i, None))
    return out


def prepare(config: HydroJEVConfig, outdir: Path) -> dict[str, Any]:
    target = outdir / "prepared.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; P2 preparation is frozen")
    outdir.mkdir(parents=True, exist_ok=True)
    specs = {net: net_spec(config, net) for net in ("CTOWN", "Net3")}
    cals = {net: calibrate(config, specs[net]) for net in specs}
    rows: list[dict[str, Any]] = []
    xs: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    for split, (net, _, _) in SPLITS.items():
        spec, cal = specs[net], cals[net]
        for sc in _scenarios(spec, split):
            seed = sc["seed"]
            gated = unrealised = 0
            for attempt in range(40):
                try:
                    sim = simulate(config, spec, sc, cal["sigma"])
                except ValueError:  # event not realisable at the drawn onset (e.g. no running pump): redraw
                    unrealised += 1
                    sc = {**sample_scenario(spec, sc["cls"], seed + 100_000 * (attempt + 1), None), "seed_origin": seed}
                    continue
                if not evaluate_window(spec, sim, sc["window_end_h"], cal)["violated"]:
                    # triage gate, identical for every method: only windows with >= 1 violated check are triaged
                    gated += 1
                    sc = {**sample_scenario(spec, sc["cls"], seed + 100_000 * (attempt + 1), None), "seed_origin": seed}
                    continue
                break
            else:
                raise RuntimeError(f"could not realise scenario seed {seed}")
            we = sc["window_end_h"]
            state = build_state(spec, sim, we, cal)
            leaks = scan_state(state)
            if leaks:
                raise HydroJEVSchemaError(f"state leak tokens {leaks} in {split} seed {seed}")
            x, ev = features(spec, sim, we, cal)
            rid = f"{split}_{len([r for r in rows if r['split'] == split]):04d}"
            req = build_request(state)
            reqdir = outdir / "requests" / split
            reqdir.mkdir(parents=True, exist_ok=True)
            body = p1.canonical_bytes(req)
            problems = p1.scan_request(body)
            if problems:
                raise HydroJEVSchemaError(f"request scan {problems} for {rid}")
            (reqdir / f"{rid}.request.json").write_bytes(body)
            xs[split].append(x)
            rows.append({"id": rid, "split": split, "network": net, "label_internal": sc["cls"], "scenario_internal": {**sc, **sim["resolved"]},
                         "redraws": {"not_realisable": unrealised, "no_violated_check": gated}, "r1": r1_classify(ev), "violated": ev["violated"], "n_anomalous": len(ev["anomalous_channels"]),
                         "request_sha256": hashlib.sha256(body).hexdigest(), "request_bytes": len(body)})
    y_train = [r["label_internal"] for r in rows if r["split"] == "train"]
    model = fit_r2(np.vstack(xs["train"]), y_train)
    for split in SPLITS:
        pr = model.predict_proba(np.vstack(xs[split]))
        split_rows = [r for r in rows if r["split"] == split]
        for r, p in zip(split_rows, pr):
            probs = {c: 0.0 for c in EVENT_TYPES}
            probs.update({c: float(v) for c, v in zip(model.classes_, p)})
            r["r2_probs"] = probs
            r["r2"] = max(probs, key=probs.get)
    comparators = {}
    for split in SPLITS:
        sr = [r for r in rows if r["split"] == split]
        yt = [r["label_internal"] for r in sr]
        comparators[split] = {
            "R1": cause_metrics(yt, [r["r1"] for r in sr], [{c: float(c == r["r1"]) for c in EVENT_TYPES} for r in sr]),
            "R2": cause_metrics(yt, [r["r2"] for r in sr], [r["r2_probs"] for r in sr]),
        }
    prepared = {
        "protocol": p1.PROTOCOL, "schema_version": SCHEMA_VERSION,
        "questions_sha256": hashlib.sha256(json.dumps(build_questions(), sort_keys=True).encode()).hexdigest(),
        "calibration": {net: {k: v for k, v in cal.items() if k != "sigma"} for net, cal in cals.items()},
        "pressure_sensors": {net: s.pressure_nodes for net, s in specs.items()},
        "splits": {s: {"network": v[0], "per_class": v[1]} for s, v in SPLITS.items()},
        "comparators": comparators,
        "note": "R2 is trained on the train split; its train-split metrics are in-sample. Labels are stored only here.",
        "rows": rows,
    }
    target.write_text(json.dumps(_safe(prepared), indent=1, ensure_ascii=False), encoding="utf-8")
    return prepared


def mock_response(row: Mapping[str, Any]) -> dict[str, Any]:
    probs = {c: 0.05 for c in EVENT_TYPES}
    probs[row["r1"]] = 0.8
    return {"model": "mock-p2", "answers": {"event_type": {"type": "choice", "choice": row["r1"], "confidence": 0.8, "probabilities": probs}},
            "usage": {"input_tokens": 0, "output_tokens": 0}}


def select_debug(prepared: Mapping[str, Any], per_class: int = 3) -> list[str]:
    rng = np.random.default_rng(p1.DEFAULT_SAMPLE_SEED)
    out = []
    for cls in CLASSES:
        ids = sorted(r["id"] for r in prepared["rows"] if r["split"] == "train" and r["label_internal"] == cls)
        out += sorted(rng.choice(ids, size=per_class, replace=False).tolist())
    return out


def _decisions_for(outdir: Path, stage: str, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for rid in ids:
        f = LEDGER_DIR / "responses" / stage / f"{stage}_{rid}.response.json"
        if f.exists():
            wrapper = json.loads(f.read_text(encoding="utf-8-sig"))
            d = parse_response(wrapper["response"])
            out[rid] = {"event_type": d.event_type, "probabilities": d.probabilities, "model": d.model, "usage": d.usage}
    return out


def summarize(prepared: Mapping[str, Any], ids: Sequence[str], decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = {r["id"]: r for r in prepared["rows"]}
    ids = [i for i in ids if i in decisions]
    yt = [rows[i]["label_internal"] for i in ids]
    per = [{"id": i, "truth": rows[i]["label_internal"], "subtype": rows[i]["scenario_internal"].get("subtype"), "jev": decisions[i]["event_type"],
            "jev_p_top": _r(max(decisions[i]["probabilities"].values()), 3), "r1": rows[i]["r1"], "r2": rows[i]["r2"]} for i in ids]
    return {
        "n": len(ids),
        "jev": cause_metrics(yt, [decisions[i]["event_type"] for i in ids], [decisions[i]["probabilities"] for i in ids]),
        "r1": cause_metrics(yt, [rows[i]["r1"] for i in ids], [{c: float(c == rows[i]["r1"]) for c in EVENT_TYPES} for i in ids]),
        "r2": cause_metrics(yt, [rows[i]["r2"] for i in ids], [rows[i]["r2_probs"] for i in ids]),
        "models": sorted({decisions[i]["model"] for i in ids}),
        "mean_input_tokens": float(np.mean([decisions[i]["usage"].get("input_tokens", 0) for i in ids])) if ids else 0.0,
        "per_window": per,
    }


def run_stage(outdir: Path, stage: str, ids: Sequence[str], *, live: bool) -> dict[str, Any]:
    prepared = json.loads((outdir / "prepared.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in prepared["rows"]}
    if live:
        for rid in ids:
            req = outdir / "requests" / rows[rid]["split"] / f"{rid}.request.json"
            e = p1.call_live(LEDGER_DIR, req, stage=stage, tag=f"{stage}_{rid}", summarize=summarize_response)
            print(rid, e.get("status"), e.get("event_type"), "reused" if e.get("reused") else "")
        decisions = _decisions_for(outdir, stage, ids)
    else:
        decisions = {}
        for rid in ids:
            d = parse_response(mock_response(rows[rid]))
            decisions[rid] = {"event_type": d.event_type, "probabilities": d.probabilities, "model": d.model, "usage": d.usage}
    summary = summarize(prepared, ids, decisions)
    name = f"{stage}_summary{'' if live else '_mock'}.json"
    (outdir / name).write_text(json.dumps(_safe(summary), indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["prepare", "debug", "eval_ctown", "eval_net3", "repeat"])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "prepare":
        prepared = prepare(load_config(), args.outdir)
        print(json.dumps(_safe(prepared["comparators"]), indent=1)[:4000])
        return 0
    prepared = json.loads((args.outdir / "prepared.json").read_text(encoding="utf-8"))
    if args.command == "debug":
        ids = select_debug(prepared)
        stage = "p2_debug_r1"
    elif args.command == "repeat":
        rng = np.random.default_rng(p1.DEFAULT_SAMPLE_SEED + 2)
        ids = sorted(rng.choice(sorted(r["id"] for r in prepared["rows"] if r["split"] == "eval_ctown"), size=12, replace=False).tolist())
        stage = "p2_repeat"
    else:
        ids = sorted(r["id"] for r in prepared["rows"] if r["split"] == args.command)
        stage = f"p2_{args.command}"
    summary = run_stage(args.outdir, stage, ids, live=args.live)
    print(json.dumps({k: summary[k] for k in ("n", "jev", "r1", "r2")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

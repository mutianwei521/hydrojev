"""Real-hydraulics consequence-mitigation: the reflex interlock on EPANET/CTOWN.

The evasion and transfer benchmarks establish that *detection* can be evaded by a
stealthy adversary (~15% of attacks slip past even the orthogonal ensemble). The
paradigm's answer is that a stealthy cyber attack still has to move *water*, and
water obeys physics that a deterministic local interlock -- the **reflex** -- reads
directly, with no cloud in the loop and nothing for the attacker to spoof. This
module proves that claim on the real **CTOWN** benchmark network with the EPANET
hydraulic solver (via ``wntr``), so the energy and overflow numbers are simulated,
not assumed.

**Threat model (faithful to the BATADAL control-channel attacks).** CTOWN governs
each pump by tank-level interlocks, e.g. ``IF T3 LEVEL ABOVE 5.3 THEN PU4 CLOSED``.
A stealthy attacker who has compromised the SCADA level channel (or the PLC logic)
defeats the *high-level close* interlock, so the pump keeps running past the safe
band and the tank overflows -- wasting pumping energy and spilling treated water.
This is modelled by removing that close control from the network.

**The reflex.** An independent local interlock watches the *true* tank level (a
hardwired float/pressure switch at the pump station, on a channel the cyber
attacker never touched) and closes the pump the moment the level crosses a safety
band just above normal operation. Modelled as a high-priority ``wntr`` control on
the physical level. It exists only in the *defended* run; it sits above the normal
operating band, so it is silent unless an attack drives the tank abnormally high.

Three counterfactual runs share one initial condition and horizon:

* **baseline** -- stock CTOWN controls, no attack;
* **attacked** -- the SCADA close interlock defeated (undefended);
* **defended** -- the same attack, plus the local reflex interlock.

Consequence metrics, all read from the EPANET solution:

* total electrical **pumping energy** (kWh) via the tested
  :func:`hydrojev.state_projection.residual_engine.integrate_pump_energy`;
* **overflow volume** (m^3) -- net inflow through the tank's incident links while
  the level is pinned at its maximum (the water that spills);
* **peak level** and **steps spent overflowing**.

If ``wntr`` or the CTOWN asset is unavailable the runner returns
``status="not_run"`` rather than fabricating numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig
from hydrojev.datasets.wdn_loader import DatasetUnavailableError
from hydrojev.simulation.closed_loop_eval import EnergyTrace, energy_mitigation
from hydrojev.state_projection.residual_engine import integrate_pump_energy


# --------------------------------------------------------------------------- #
# Attack specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OverpumpAttack:
    """A stealthy over-pumping attack: defeat one pump's high-level close interlock.

    ``close_control`` is the control that would normally stop ``pump`` when ``tank``
    reaches its safe upper level; removing it makes the pump over-run. On CTOWN this
    single interlock is enough to overflow the tank. On a network whose pressure-zone
    logic *relieves* an over-full tank through a controlled bypass link (e.g. Net3's
    Pipe 330), a faithful stealthy attacker must also defeat that relief for the tank
    to actually overflow -- captured by ``extra_defeated_controls`` (further controls
    removed) and ``force_closed_links`` (bypass links pinned shut). Both default to
    empty, so the CTOWN attacks are unchanged.
    """

    pump: str
    tank: str
    close_control: str
    label: str = ""
    extra_defeated_controls: tuple[str, ...] = ()
    force_closed_links: tuple[str, ...] = ()

    def display(self) -> str:
        return self.label or f"over-pump {self.tank} via {self.pump}"


# The single level-interlock attack surface on Net3 (EPANET's classic example
# network, structurally very different from CTOWN): PUMP 335 fills Tank 1, whose
# high-level close is `control 16`. Tank 1 also has a pressure-zone relief bypass
# (Pipe 330, opened when the tank is high by `control 18`); the faithful stealthy
# attack defeats both, so the over-pumped water cannot drain and the tank overflows.
NET3_ATTACKS: tuple[OverpumpAttack, ...] = (
    OverpumpAttack(
        "335", "1", "control 16", "over-pump Tank 1 (PU335)",
        extra_defeated_controls=("control 18",),
        force_closed_links=("330",),
    ),
)


# The pumps whose high-level close interlock actually binds in CTOWN (they toggle
# off in normal operation), i.e. the ones an over-pump attack can exploit. Found
# by simulation: PU4->T3, PU7->T4, PU8->T5, PU10->T7.
DEFAULT_ATTACKS: tuple[OverpumpAttack, ...] = (
    OverpumpAttack("PU4", "T3", "control 8", "over-pump T3 (PU4)"),
    OverpumpAttack("PU7", "T4", "control 14", "over-pump T4 (PU7)"),
    OverpumpAttack("PU8", "T5", "control 16", "over-pump T5 (PU8)"),
    OverpumpAttack("PU10", "T7", "control 18", "over-pump T7 (PU10)"),
)


# --------------------------------------------------------------------------- #
# Pure, unit-testable consequence helpers
# --------------------------------------------------------------------------- #


def pump_energy_kwh(
    times_s: np.ndarray,
    flow_m3_s: np.ndarray,
    head_gain_m: np.ndarray,
    *,
    efficiency: float = 0.75,
) -> float:
    """Electrical energy for one pump over the horizon (delegates to the tested integrator).

    Flow magnitude and non-negative head gain are used so a momentarily reversed
    or idle pump contributes no negative energy.
    """

    q = np.abs(np.asarray(flow_m3_s, dtype=float))
    h = np.maximum(np.asarray(head_gain_m, dtype=float), 0.0)
    return integrate_pump_energy(np.asarray(times_s, dtype=float), q, h, efficiency=efficiency)


def overflow_volume_m3(
    net_inflow_m3_s: np.ndarray,
    level_m: np.ndarray,
    max_level_m: float,
    times_s: np.ndarray,
    *,
    level_atol: float = 1e-3,
) -> tuple[float, int]:
    """Spilled volume and number of spilling steps.

    Overflow happens only while the tank is pinned at its maximum level; by
    continuity the level is then constant, so the net inflow at those instants is
    exactly what spills. Volume is integrated trapezoidally on the report grid,
    counting only the positive net-inflow at max-level samples.
    """

    net = np.asarray(net_inflow_m3_s, dtype=float)
    level = np.asarray(level_m, dtype=float)
    t = np.asarray(times_s, dtype=float)
    if not (net.shape == level.shape == t.shape) or t.size < 2:
        raise ValueError("net_inflow, level, and times must share a shape of length >= 2")
    at_max = level >= (float(max_level_m) - float(level_atol))
    spill_rate = np.where(at_max, np.clip(net, 0.0, None), 0.0)
    # Rectangular rule with per-sample forward widths (last width repeated). On the
    # uniform CTOWN report grid this is exactly sum_{at max} max(0, net) * dt, the
    # recipe cross-validated against the tank's reported demand-at-max.
    diffs = np.diff(t)
    widths = np.concatenate([diffs, diffs[-1:]])
    volume = float(np.sum(spill_rate * widths))
    steps_spilling = int(np.sum(spill_rate > 0.0))
    return volume, steps_spilling


def extra_fraction(value: float, baseline: float) -> float:
    """Fractional change of ``value`` relative to a positive ``baseline``."""

    if not np.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return (float(value) - float(baseline)) / float(baseline)


def mitigated_fraction(extra_attacked: float, extra_defended: float) -> float:
    """Fraction of the attack's energy inflation removed by the defence.

    1.0 = defence fully cancels the inflation; >1 = defence undercuts baseline.
    """

    if not np.isfinite(extra_attacked) or extra_attacked <= 0:
        return float("nan")
    return (extra_attacked - extra_defended) / extra_attacked


def averted_fraction(attacked_volume: float, defended_volume: float) -> float:
    """Fraction of the attack's overflow volume the defence prevents."""

    if not np.isfinite(attacked_volume) or attacked_volume <= 0:
        return float("nan")
    return max(0.0, (attacked_volume - defended_volume) / attacked_volume)


def aggregate_attacks(attacks: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Mean consequence metrics across a suite of per-attack result dicts."""

    def _mean(key: str) -> float:
        vals = [float(a[key]) for a in attacks if np.isfinite(a.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_attacks": len(attacks),
        "mean_extra_energy_fraction_attacked": _mean("extra_energy_fraction_attacked"),
        "mean_extra_energy_fraction_defended": _mean("extra_energy_fraction_defended"),
        "mean_energy_mitigated_fraction": _mean("energy_mitigated_fraction"),
        "mean_overflow_averted_fraction": _mean("overflow_averted_fraction"),
        "mean_reflex_operational_cost_fraction": _mean("reflex_operational_cost_fraction"),
        "any_reflex_nuisance_trip": bool(any(a.get("reflex_nuisance_trip") for a in attacks)),
        "total_overflow_attacked_m3": float(
            np.sum([a["attacked"]["overflow_volume_m3"] for a in attacks])
        ),
        "total_overflow_defended_m3": float(
            np.sum([a["defended"]["overflow_volume_m3"] for a in attacks])
        ),
    }


# --------------------------------------------------------------------------- #
# EPANET extraction (thin wrappers around wntr result objects)
# --------------------------------------------------------------------------- #


def _incident_links(wn: Any, tank: str) -> list[tuple[str, int]]:
    """Links touching ``tank`` with a sign: +1 if the tank is the link's end node."""

    incident: list[tuple[str, int]] = []
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        if link.end_node_name == tank:
            incident.append((link_name, +1))
        if link.start_node_name == tank:
            incident.append((link_name, -1))
    return incident


def _net_tank_inflow(res: Any, wn: Any, tank: str) -> np.ndarray:
    """Net volumetric inflow to ``tank`` (sum of incident link flows into it)."""

    total = None
    for link_name, sign in _incident_links(wn, tank):
        flow = np.asarray(res.link["flowrate"][link_name].values, dtype=float)
        total = sign * flow if total is None else total + sign * flow
    if total is None:
        raise ValueError(f"tank {tank} has no incident links")
    return total


def _tank_level(res: Any, wn: Any, tank: str) -> np.ndarray:
    return np.asarray(res.node["head"][tank].values, dtype=float) - float(wn.get_node(tank).elevation)


def _pump_head_gain(res: Any, wn: Any, pump: str) -> np.ndarray:
    link = wn.get_link(pump)
    end = np.asarray(res.node["head"][link.end_node_name].values, dtype=float)
    start = np.asarray(res.node["head"][link.start_node_name].values, dtype=float)
    return end - start


def _total_pump_energy(res: Any, wn: Any, times: np.ndarray, *, efficiency: float) -> float:
    total = 0.0
    for pump in wn.pump_name_list:
        flow = res.link["flowrate"][pump].values
        head = _pump_head_gain(res, wn, pump)
        total += pump_energy_kwh(times, flow, head, efficiency=efficiency)
    return total


# --------------------------------------------------------------------------- #
# Scenario builders and the runner
# --------------------------------------------------------------------------- #


def _build_model(config: HydroJEVConfig, attack: OverpumpAttack, horizon_h: int, mode: str,
                 reflex_safety_fraction: float, network: str = "CTOWN") -> Any:
    """Build a water-network model for one of the scenario modes.

    Modes: ``baseline`` (stock), ``attacked`` (SCADA close interlock defeated),
    ``defended`` (attacked + local reflex), ``baseline_reflex`` (stock controls
    intact + local reflex, no attack -- the operational-neutrality control).
    ``network`` selects the loaded asset (``CTOWN`` by default; ``Net3`` for the
    cross-network generalization).
    """

    from wntr.network.controls import Control, ControlAction, ValueCondition

    from hydrojev.datasets.wdn_loader import load_network

    bundle = load_network(network, config=config)
    wn = bundle.model
    wn.options.time.duration = int(horizon_h * 3600)
    tank = wn.get_node(attack.tank)
    tank.overflow = True  # let the physical tank actually spill

    if mode in ("attacked", "defended"):
        # Attacker defeats the SCADA high-level close interlock (and, where the tank
        # is relieved by a pressure-zone bypass, that relief too -- see OverpumpAttack).
        wn.remove_control(attack.close_control)
        for extra in attack.extra_defeated_controls:
            wn.remove_control(extra)
        if attack.force_closed_links:
            from wntr.network.base import LinkStatus

            for link_name in attack.force_closed_links:
                wn.get_link(link_name).initial_status = LinkStatus.Closed
    if mode in ("defended", "baseline_reflex"):
        # Independent local reflex on the true level, just above the normal band.
        safe_level = float(reflex_safety_fraction) * float(tank.max_level)
        action = ControlAction(wn.get_link(attack.pump), "status", 0)  # 0 = Closed
        condition = ValueCondition(tank, "level", ">=", safe_level)
        wn.add_control(f"reflex_{attack.pump}", Control(condition, action, name=f"reflex_{attack.pump}"))
    return wn


def _run_scenario(config: HydroJEVConfig, attack: OverpumpAttack, horizon_h: int, mode: str,
                  reflex_safety_fraction: float, efficiency: float,
                  network: str = "CTOWN") -> tuple[dict[str, Any], EnergyTrace]:
    import wntr

    wn = _build_model(config, attack, horizon_h, mode, reflex_safety_fraction, network)
    res = wntr.sim.EpanetSimulator(wn).run_sim()

    times = np.asarray(res.node["head"].index.values, dtype=float)
    level = _tank_level(res, wn, attack.tank)
    net_in = _net_tank_inflow(res, wn, attack.tank)
    max_level = float(wn.get_node(attack.tank).max_level)
    volume, steps = overflow_volume_m3(net_in, level, max_level, times)

    metrics = {
        "total_energy_kwh": _total_pump_energy(res, wn, times, efficiency=efficiency),
        "target_pump_energy_kwh": pump_energy_kwh(
            times, res.link["flowrate"][attack.pump].values, _pump_head_gain(res, wn, attack.pump),
            efficiency=efficiency,
        ),
        "tank_peak_level_m": float(level.max()),
        "tank_max_level_m": max_level,
        "tank_peak_fraction": float(level.max() / max_level) if max_level > 0 else float("nan"),
        "overflow_volume_m3": volume,
        "steps_overflowing": steps,
    }
    # EnergyTrace for the target pump (aligned horizon), to reuse energy_mitigation.
    trace = EnergyTrace(
        times_s=times,
        flow_m3_s=np.abs(np.asarray(res.link["flowrate"][attack.pump].values, dtype=float)),
        head_m=np.maximum(_pump_head_gain(res, wn, attack.pump), 0.0),
    )
    return metrics, trace


def run_hydraulic_mitigation(
    config: HydroJEVConfig,
    *,
    mode: Any = None,
    attacks: Sequence[OverpumpAttack] = DEFAULT_ATTACKS,
    horizon_h: int = 72,
    reflex_safety_fraction: float = 0.98,
    efficiency: float = 0.75,
    network: str = "CTOWN",
) -> dict[str, Any]:
    """Simulate baseline/attacked/defended for each attack on real EPANET hydraulics.

    ``network`` defaults to CTOWN (the §7 result); pass ``Net3`` with ``NET3_ATTACKS``
    for the cross-network generalization.
    """

    try:
        import wntr  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        return {"status": "not_run", "reason": f"wntr unavailable: {exc}"}

    attack_results: list[dict[str, Any]] = []
    try:
        for attack in attacks:
            base_m, base_tr = _run_scenario(config, attack, horizon_h, "baseline", reflex_safety_fraction, efficiency, network)
            att_m, att_tr = _run_scenario(config, attack, horizon_h, "attacked", reflex_safety_fraction, efficiency, network)
            def_m, def_tr = _run_scenario(config, attack, horizon_h, "defended", reflex_safety_fraction, efficiency, network)
            # Operational-neutrality control: reflex installed, NO attack. If the
            # reflex never nuisance-trips in normal operation this equals baseline.
            noatk_m, _ = _run_scenario(config, attack, horizon_h, "baseline_reflex", reflex_safety_fraction, efficiency, network)
            reflex_op_cost = extra_fraction(noatk_m["target_pump_energy_kwh"], base_m["target_pump_energy_kwh"])
            # A nuisance trip would cut energy below baseline; flag any material deviation.
            reflex_nuisance = bool(abs(reflex_op_cost) > 0.01) or noatk_m["overflow_volume_m3"] > 0.0

            # Headline energy fractions are on the TARGET pump the attacker over-ran and
            # the reflex controls; total-plant energy is reported separately as context
            # (unrelated pumps dilute the single-station attack, so it under-states severity).
            extra_att = extra_fraction(att_m["target_pump_energy_kwh"], base_m["target_pump_energy_kwh"])
            extra_def = extra_fraction(def_m["target_pump_energy_kwh"], base_m["target_pump_energy_kwh"])
            extra_att_plant = extra_fraction(att_m["total_energy_kwh"], base_m["total_energy_kwh"])
            extra_def_plant = extra_fraction(def_m["total_energy_kwh"], base_m["total_energy_kwh"])
            # Reuse the tested aligned-horizon comparator on the target pump.
            try:
                target_energy = energy_mitigation(base_tr, att_tr, def_tr, efficiency=efficiency)
            except Exception:
                target_energy = {}
            attack_results.append(
                {
                    "pump": attack.pump,
                    "tank": attack.tank,
                    "defeated_control": attack.close_control,
                    "label": attack.display(),
                    "baseline": base_m,
                    "attacked": att_m,
                    "defended": def_m,
                    "baseline_reflex": noatk_m,
                    "reflex_operational_cost_fraction": reflex_op_cost,
                    "reflex_nuisance_trip": reflex_nuisance,
                    "extra_energy_fraction_attacked": extra_att,
                    "extra_energy_fraction_defended": extra_def,
                    "extra_energy_fraction_attacked_plant": extra_att_plant,
                    "extra_energy_fraction_defended_plant": extra_def_plant,
                    "energy_mitigated_fraction": mitigated_fraction(extra_att, extra_def),
                    "energy_mitigated_fraction_plant": mitigated_fraction(extra_att_plant, extra_def_plant),
                    "overflow_averted_fraction": averted_fraction(
                        att_m["overflow_volume_m3"], def_m["overflow_volume_m3"]
                    ),
                    "target_pump_energy_mitigation": target_energy,
                }
            )
    except (DatasetUnavailableError, FileNotFoundError, KeyError, ValueError) as exc:
        return {"status": "not_run", "reason": str(exc)}

    result = {
        "status": "ok",
        "network": network,
        "protocol": (
            f"EPANET/wntr on {network}; a stealthy attacker defeats a pump's SCADA high-level "
            "close interlock (over-pumping to overflow), and any pressure-zone relief bypass that "
            "would otherwise drain the tank; the defended run adds an independent "
            "local reflex interlock on the true tank level at "
            f"{reflex_safety_fraction:.0%} of max. Baseline/attacked/defended share the initial "
            f"condition and a {horizon_h} h horizon; energy via wire-to-water integration, "
            "overflow via net inflow while the tank is pinned at max."
        ),
        "horizon_h": horizon_h,
        "timestep_s": int(config_hydraulic_timestep(config, network)),
        "efficiency": efficiency,
        "reflex_safety_fraction": reflex_safety_fraction,
        "attacks": attack_results,
    }
    result.update(aggregate_attacks(attack_results))
    return result


def config_hydraulic_timestep(config: HydroJEVConfig, network: str = "CTOWN") -> int:
    """Best-effort report of the network's hydraulic timestep (for provenance)."""

    try:
        from hydrojev.datasets.wdn_loader import load_network

        return int(load_network(network, config=config).hydraulic_timestep_s)
    except Exception:  # pragma: no cover
        return 900


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_mitigation_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    lines = [
        f"Reflex consequence-mitigation on real {result.get('network', 'CTOWN')} hydraulics ({result['horizon_h']} h horizon).",
        "Undefended = SCADA close-interlock defeated; Defended = + local reflex interlock.",
        "",
        f"{'attack':<20}{'energy +%(att)':>15}{'energy +%(def)':>15}{'mitig%':>9}"
        f"{'overflow att(m3)':>18}{'overflow def':>13}",
        "-" * 90,
    ]
    for a in result["attacks"]:
        lines.append(
            f"{a['label']:<20}"
            f"{a['extra_energy_fraction_attacked']*100:>14.1f}%"
            f"{a['extra_energy_fraction_defended']*100:>14.1f}%"
            f"{a['energy_mitigated_fraction']*100:>8.0f}%"
            f"{a['attacked']['overflow_volume_m3']:>18.0f}"
            f"{a['defended']['overflow_volume_m3']:>13.0f}"
        )
    lines.append("-" * 90)
    lines.append(
        f"MEAN energy inflation attacked {result['mean_extra_energy_fraction_attacked']*100:.1f}% "
        f"-> defended {result['mean_extra_energy_fraction_defended']*100:.1f}% "
        f"(mitigated {result['mean_energy_mitigated_fraction']*100:.0f}%); "
        f"overflow {result['total_overflow_attacked_m3']:.0f} m3 -> "
        f"{result['total_overflow_defended_m3']:.0f} m3 "
        f"(averted {result['mean_overflow_averted_fraction']*100:.0f}%)."
    )
    op_cost = result.get("mean_reflex_operational_cost_fraction", float("nan"))
    nuisance = result.get("any_reflex_nuisance_trip", False)
    if np.isfinite(op_cost):
        lines.append(
            f"OPERATIONAL NEUTRALITY: reflex installed under NORMAL operation costs "
            f"{op_cost*100:+.2f}% energy; nuisance trip: {'YES' if nuisance else 'NONE'}."
        )
    return "\n".join(lines)


def plot_mitigation(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [a["tank"] for a in result["attacks"]]
    x = np.arange(len(labels))
    w = 0.38

    fig, (axe, axo) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    att_e = [a["extra_energy_fraction_attacked"] * 100 for a in result["attacks"]]
    def_e = [a["extra_energy_fraction_defended"] * 100 for a in result["attacks"]]
    axe.bar(x - w / 2, att_e, w, label="undefended", color="#d62728")
    axe.bar(x + w / 2, def_e, w, label="reflex-defended", color="#2ca02c")
    axe.axhline(0.0, color="black", lw=0.8)
    axe.set_xticks(x)
    axe.set_xticklabels(labels)
    axe.set_ylabel("extra pumping energy vs baseline (%)")
    axe.set_title("Reflex interlock removes the attack's energy inflation", fontsize=10)
    axe.legend(fontsize=8)
    axe.grid(axis="y", alpha=0.3)

    att_o = [a["attacked"]["overflow_volume_m3"] for a in result["attacks"]]
    def_o = [a["defended"]["overflow_volume_m3"] for a in result["attacks"]]
    axo.bar(x - w / 2, att_o, w, label="undefended", color="#d62728")
    axo.bar(x + w / 2, def_o, w, label="reflex-defended", color="#2ca02c")
    axo.set_xticks(x)
    axo.set_xticklabels(labels)
    axo.set_ylabel("overflow volume (m$^3$)")
    axo.set_title("Reflex interlock prevents tank overflow", fontsize=10)
    axo.legend(fontsize=8)
    axo.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cognitive Reflex Paradigm on CTOWN: a stealthy control-channel attack that evades "
        "detection\nstill cannot evade physics -- the local interlock neutralises the hydraulic consequence",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from hydrojev.config import load_config

    parser = argparse.ArgumentParser(description="Reflex consequence-mitigation on CTOWN")
    parser.add_argument("--horizon-h", type=int, default=72)
    parser.add_argument("--reflex-fraction", type=float, default=0.98)
    parser.add_argument("--outdir", default="artifacts/hydraulic_mitigation")
    args = parser.parse_args(argv)

    config = load_config()
    result = run_hydraulic_mitigation(
        config, horizon_h=args.horizon_h, reflex_safety_fraction=args.reflex_fraction
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hydraulic_mitigation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_mitigation_table(result)
        (outdir / "mitigation_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_mitigation(result, str(outdir / "mitigation.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'hydraulic_mitigation.json'}, mitigation_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

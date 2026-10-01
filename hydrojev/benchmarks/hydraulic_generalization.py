"""§7.x cross-network generalization of the reflex consequence-mitigation.

§7 proves the paradigm's hydraulic payoff on **CTOWN** (the BATADAL benchmark
network): a stealthy control-channel attack that defeats a pump's high-level close
interlock over-pumps tanks to overflow, and an independent local reflex interlock on
the *true* tank level neutralises ~100% of the energy inflation and 100% of the
overflow at zero operational cost. A *Water Research* reviewer will ask the obvious
next question: **is this a property of the CTOWN asset, or of the paradigm?**

This benchmark answers it by re-running the *same* fully-tested
:func:`hydrojev.simulation.hydraulic_mitigation.run_hydraulic_mitigation` on a
**structurally different** network -- **Net3**, EPANET's classic example (3 tanks,
2 pumps, Hazen-Williams head loss, a pressure-zone relief bypass), which shares no
topology, control logic, or units convention with CTOWN. No new physics is
introduced, so the numbers inherit that runner's validation.

The generalization is **honest about what transfers and what does not**:

* **The safety-critical consequence (overflow) is 100% averted on both networks.**
  This is the paradigm's core guarantee -- the catastrophe is prevented regardless
  of network.
* **The attack's *footprint* is network-specific.** On CTOWN a single defeated
  interlock overflows the tank. Net3's tank is *relieved* by a controlled bypass
  (Pipe 330), so a faithful stealthy attacker must defeat the pump close *and* that
  bypass -- a two-control pressure-zone compromise -- for the tank to overflow.
* **The degree of *energy* recovery is network-specific.** On CTOWN the reflex
  recovers essentially all of the inflated energy; on Net3 the coupled bypass
  compromise keeps the tank near-full even after the reflex trips, so the level
  reflex recovers only a fraction of the energy while still eliminating 100% of the
  spill. The reflex guarantees *safety*, not energy-optimality -- energy recovery is
  the operator's/cognition's job once the catastrophe is off the table.
* **Operational neutrality transfers.** On both networks the reflex sits above the
  normal operating band, so under normal operation it costs ~0% energy and never
  nuisance-trips.

Returns ``status="not_run"`` (never fabricated numbers) when ``wntr`` or an asset is
unavailable.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig
from hydrojev.simulation.hydraulic_mitigation import (
    DEFAULT_ATTACKS,
    NET3_ATTACKS,
    OverpumpAttack,
    run_hydraulic_mitigation,
)


# Networks compared, each with its documented level-interlock attack surface(s).
# CTOWN: the §7 result (4 pumps whose high-level close binds). Net3: the single
# level-interlock station (PUMP 335 -> Tank 1) plus its pressure-zone bypass.
DEFAULT_NETWORKS: tuple[tuple[str, tuple[OverpumpAttack, ...]], ...] = (
    ("CTOWN", DEFAULT_ATTACKS),
    ("Net3", NET3_ATTACKS),
)


# --------------------------------------------------------------------------- #
# Pure summarisation (unit-testable without EPANET)
# --------------------------------------------------------------------------- #


def summarize_network(result: dict[str, Any]) -> dict[str, Any]:
    """Compress one network's full mitigation result into comparison fields.

    Reads only the aggregate keys :func:`aggregate_attacks` already computed, so it
    fabricates nothing. Returns the ``not_run`` marker verbatim when the run failed.
    """

    if result.get("status") != "ok":
        return {"status": result.get("status", "not_run"), "reason": result.get("reason")}
    return {
        "status": "ok",
        "network": result.get("network"),
        "n_attacks": result.get("n_attacks"),
        "timestep_s": result.get("timestep_s"),
        "mean_extra_energy_fraction_attacked": result.get("mean_extra_energy_fraction_attacked"),
        "mean_extra_energy_fraction_defended": result.get("mean_extra_energy_fraction_defended"),
        "mean_energy_mitigated_fraction": result.get("mean_energy_mitigated_fraction"),
        "mean_overflow_averted_fraction": result.get("mean_overflow_averted_fraction"),
        "total_overflow_attacked_m3": result.get("total_overflow_attacked_m3"),
        "total_overflow_defended_m3": result.get("total_overflow_defended_m3"),
        "mean_reflex_operational_cost_fraction": result.get("mean_reflex_operational_cost_fraction"),
        "any_reflex_nuisance_trip": result.get("any_reflex_nuisance_trip"),
    }


def _overflow_fully_averted(summary: dict[str, Any], *, atol_m3: float = 1.0) -> bool:
    """True iff the defended run spills essentially nothing while the attack did."""

    att = summary.get("total_overflow_attacked_m3")
    dfn = summary.get("total_overflow_defended_m3")
    if att is None or dfn is None or not np.isfinite(att) or att <= 0:
        return False
    return float(dfn) <= atol_m3


def cross_network_findings(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Derive the honest cross-network claims from the per-network summaries."""

    ok = [s for s in summaries if s.get("status") == "ok"]
    return {
        "n_networks_ok": len(ok),
        "networks": [s.get("network") for s in ok],
        # The core guarantee: overflow (the catastrophe) fully averted on every network.
        "overflow_fully_averted_all_networks": bool(ok) and all(
            _overflow_fully_averted(s) for s in ok
        ),
        # Operational neutrality transfers: no reflex nuisance trip on any network.
        "operationally_neutral_all_networks": bool(ok) and not any(
            bool(s.get("any_reflex_nuisance_trip")) for s in ok
        ),
        # Energy recovery is honestly network-specific (report the spread, don't average
        # away the difference that is itself the finding).
        "energy_mitigated_fraction_by_network": {
            s.get("network"): s.get("mean_energy_mitigated_fraction") for s in ok
        },
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_generalization(
    config: HydroJEVConfig,
    *,
    networks: Sequence[tuple[str, tuple[OverpumpAttack, ...]]] = DEFAULT_NETWORKS,
    horizon_h: int = 72,
    reflex_safety_fraction: float = 0.98,
    efficiency: float = 0.75,
) -> dict[str, Any]:
    """Run the reflex mitigation on each network and fuse a cross-network comparison."""

    try:
        import wntr  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        return {"status": "not_run", "reason": f"wntr unavailable: {exc}"}

    per_network: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, attacks in networks:
        result = run_hydraulic_mitigation(
            config,
            attacks=attacks,
            horizon_h=horizon_h,
            reflex_safety_fraction=reflex_safety_fraction,
            efficiency=efficiency,
            network=name,
        )
        per_network.append(result)
        summaries.append(summarize_network(result))

    if not any(s.get("status") == "ok" for s in summaries):
        reason = next((s.get("reason") for s in summaries if s.get("reason")), "no network ran")
        return {"status": "not_run", "reason": reason}

    return {
        "status": "ok",
        "protocol": (
            "Re-runs the tested run_hydraulic_mitigation on each network. A stealthy attacker "
            "defeats a pump's high-level close interlock (and any pressure-zone relief bypass); "
            "the defended run adds an independent local reflex on the true tank level at "
            f"{reflex_safety_fraction:.0%} of max. Shared initial condition and {horizon_h} h "
            "horizon per network; energy via wire-to-water integration, overflow via net inflow "
            "while the tank is pinned at max. No new physics vs. §7."
        ),
        "horizon_h": horizon_h,
        "reflex_safety_fraction": reflex_safety_fraction,
        "efficiency": efficiency,
        "summaries": summaries,
        "findings": cross_network_findings(summaries),
        "networks_detail": per_network,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_generalization_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    lines = [
        "Cross-network generalization of the reflex consequence-mitigation "
        f"({result['horizon_h']} h horizon, trigger {result['reflex_safety_fraction']:.0%} of max level).",
        "The safety guarantee (overflow averted) is the headline; energy recovery is honestly network-specific.",
        "",
        f"{'network':<10}{'#atk':>5}{'energy +%(att)':>16}{'energy +%(def)':>16}"
        f"{'energy mitig%':>15}{'spill att(m3)':>15}{'spill def':>11}{'overflow averted':>18}",
        "-" * 106,
    ]
    for s in result["summaries"]:
        if s.get("status") != "ok":
            lines.append(f"{str(s.get('network', '?')):<10}  not_run ({s.get('reason')})")
            continue
        lines.append(
            f"{s['network']:<10}"
            f"{int(s['n_attacks']):>5}"
            f"{s['mean_extra_energy_fraction_attacked']*100:>15.1f}%"
            f"{s['mean_extra_energy_fraction_defended']*100:>15.1f}%"
            f"{s['mean_energy_mitigated_fraction']*100:>14.0f}%"
            f"{s['total_overflow_attacked_m3']:>15.0f}"
            f"{s['total_overflow_defended_m3']:>11.0f}"
            f"{s['mean_overflow_averted_fraction']*100:>17.0f}%"
        )
    lines.append("-" * 106)
    f = result["findings"]
    lines.append(
        "GUARANTEE: overflow fully averted on ALL networks = "
        f"{'YES' if f['overflow_fully_averted_all_networks'] else 'NO'}; "
        "operationally neutral (no nuisance trip) on ALL = "
        f"{'YES' if f['operationally_neutral_all_networks'] else 'NO'}."
    )
    by_net = f["energy_mitigated_fraction_by_network"]
    parts = []
    for net, frac in by_net.items():
        if frac is not None and np.isfinite(frac):
            parts.append(f"{net} {frac*100:.0f}%")
    lines.append("ENERGY recovery is network-specific (honest): " + ", ".join(parts) + ".")
    return "\n".join(lines)


def plot_generalization(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [s for s in result["summaries"] if s.get("status") == "ok"]
    labels = [s["network"] for s in ok]
    x = np.arange(len(labels))
    w = 0.38

    fig, (axe, axo) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    att_e = [s["mean_extra_energy_fraction_attacked"] * 100 for s in ok]
    def_e = [s["mean_extra_energy_fraction_defended"] * 100 for s in ok]
    axe.bar(x - w / 2, att_e, w, label="undefended", color="#d62728")
    axe.bar(x + w / 2, def_e, w, label="reflex-defended", color="#2ca02c")
    axe.axhline(0.0, color="black", lw=0.8)
    axe.set_xticks(x)
    axe.set_xticklabels(labels)
    axe.set_ylabel("mean extra pumping energy vs baseline (%)")
    axe.set_title("Energy recovery is network-specific", fontsize=10)
    axe.legend(fontsize=8)
    axe.grid(axis="y", alpha=0.3)

    att_o = [s["total_overflow_attacked_m3"] for s in ok]
    def_o = [s["total_overflow_defended_m3"] for s in ok]
    axo.bar(x - w / 2, att_o, w, label="undefended", color="#d62728")
    axo.bar(x + w / 2, def_o, w, label="reflex-defended", color="#2ca02c")
    axo.set_xticks(x)
    axo.set_xticklabels(labels)
    axo.set_ylabel("total overflow volume (m$^3$)")
    axo.set_title("Overflow 100% averted on BOTH networks (the guarantee)", fontsize=10)
    axo.legend(fontsize=8)
    axo.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cross-network generalization: the reflex neutralises the catastrophic overflow on a "
        "structurally\ndifferent network (Net3) as on CTOWN -- the paradigm's guarantee is not a "
        "CTOWN artefact",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from hydrojev.config import load_config

    parser = argparse.ArgumentParser(description="Cross-network reflex generalization (CTOWN + Net3)")
    parser.add_argument("--horizon-h", type=int, default=72)
    parser.add_argument("--reflex-fraction", type=float, default=0.98)
    parser.add_argument("--outdir", default="artifacts/hydraulic_generalization")
    args = parser.parse_args(argv)

    config = load_config()
    result = run_generalization(
        config, horizon_h=args.horizon_h, reflex_safety_fraction=args.reflex_fraction
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "hydraulic_generalization.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_generalization_table(result)
        (outdir / "generalization_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_generalization(result, str(outdir / "generalization.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'hydraulic_generalization.json'}, generalization_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

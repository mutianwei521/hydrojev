"""Head-to-head: constant-hold vs *adaptive per-timestep* temporal evasion (§6.2).

The §6.2 constant-hold attacker (:mod:`hydrojev.benchmarks.temporal_evasion`) holds
each compromised sensor at a single in-range value across the window — an explicit
*conservative lower bound* on attacker power. §8 flags the obvious reviewer follow-up:
a fully adaptive per-timestep attacker (strictly stronger) on the same sensor budget.
This module runs both on **identical** dataset04 windows (shared
:func:`~hydrojev.benchmarks.temporal_evasion.build_temporal_fixture`) and reports
whether the families' non-co-evadability survives the stronger attacker.

The adaptive attacker is warm-started from the constant-hold optimum, so per target it
is provably at least as strong; any window it evades that the constant hold could not
is genuine extra attacker power. The headline robustness question is unchanged and
sharpened: **is the temporal GEpFM drift family ever silenced?** If it survives even
the adaptive attacker, the "silencing one family leaves the decorrelated others firing"
claim holds against a much larger attack class, not just the constant-hold lower bound.

Leakage-safe (fit on dataset03); returns ``status="not_run"`` (never fabricated
numbers) when BATADAL/torch are unavailable.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks.temporal_evasion import (
    FAMILIES,
    build_temporal_fixture,
    temporal_conceal,
    temporal_conceal_adaptive,
)


# --------------------------------------------------------------------------- #
# Aggregation (pure)
# --------------------------------------------------------------------------- #


def _aggregate(evaded: int, fired: dict[str, int], still: dict[str, int], n: int) -> dict[str, Any]:
    """Joint-evasion + per-family robustness summary shared by both attackers."""

    from hydrojev.benchmarks.evasion_benchmark import wilson_interval

    lo, hi = wilson_interval(evaded, n)
    residual_alarm = {
        fam: (still[fam] / fired[fam] if fired[fam] else float("nan")) for fam in FAMILIES
    }
    return {
        "joint_evasion_rate": evaded / n,
        "joint_evasion_wilson95": [lo, hi],
        "n_jointly_evaded": evaded,
        "per_family_still_firing_after_joint_attack": dict(still),
        "per_family_residual_alarm_rate": residual_alarm,
    }


# --------------------------------------------------------------------------- #
# Real-data head-to-head runner
# --------------------------------------------------------------------------- #


def run_adaptive_evasion_comparison(
    config: Any,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 60,
    max_attack_len: int = 24,
    max_targets: int = 20,
    max_changed: int = 4,
    candidate_values: int = 11,
    constant_budget: int = 3000,
    adaptive_budget: int = 8000,
) -> dict[str, Any]:
    """Evade the named families on dataset04 with BOTH attackers on identical targets."""

    fixture = build_temporal_fixture(
        config, mode=mode, seed=seed, max_epochs=max_epochs,
        max_attack_len=max_attack_len, max_targets=max_targets,
    )
    if isinstance(fixture, dict):  # not_run
        return fixture

    seq_len = fixture.seq_len
    conceal_kw = dict(
        reference=fixture.reference, thresholds=fixture.thresholds, scales=fixture.scales,
        seq_len=seq_len, box_low=fixture.box_low, box_high=fixture.box_high,
        modifiable_mask=fixture.modifiable_mask, max_changed=max_changed,
        candidate_values=candidate_values,
    )

    fired = {fam: 0 for fam in FAMILIES}
    still_c = {fam: 0 for fam in FAMILIES}
    still_a = {fam: 0 for fam in FAMILIES}
    evaded_c = 0
    evaded_a = 0
    extra_run_starts: list[int] = []
    q_const: list[int] = []
    q_adapt_refine: list[int] = []
    targets_out: list[dict[str, Any]] = []

    for target in fixture.targets:
        const = temporal_conceal(
            fixture.predict_fn, target.context, target.attack, budget=constant_budget, **conceal_kw
        )
        adapt = temporal_conceal_adaptive(
            fixture.predict_fn, target.context, target.attack, budget=adaptive_budget,
            warm_start=const.adversarial_block, **conceal_kw
        )
        # Both judge originally-alarming on the same true attack -> identical fired set.
        for fam in const.originally_alarming:
            fired[fam] += 1
            if fam in const.still_alarming:
                still_c[fam] += 1
            if fam in adapt.still_alarming:
                still_a[fam] += 1
        if const.evaded_all:
            evaded_c += 1
        if adapt.evaded_all:
            evaded_a += 1
        if adapt.evaded_all and not const.evaded_all:
            extra_run_starts.append(int(target.run_start))
        q_const.append(const.query_count)
        q_adapt_refine.append(adapt.query_count - const.query_count)  # refine-only (warm_start given)

        targets_out.append({
            "run_start": int(target.run_start),
            "attack_len": target.attack_len,
            "originally_alarming": list(const.originally_alarming),
            "constant": {
                "still_alarming": list(const.still_alarming),
                "evaded_all": const.evaded_all,
                "n_changed_sensors": const.n_changed,
                "scaled_l2": const.scaled_l2,
                "queries": const.query_count,
                "termination": const.termination_reason,
            },
            "adaptive": {
                "still_alarming": list(adapt.still_alarming),
                "evaded_all": adapt.evaded_all,
                "n_changed_sensors": adapt.n_changed,
                "scaled_l2": adapt.scaled_l2,
                "queries": adapt.query_count,
                "termination": adapt.termination_reason,
            },
        })

    n = len(fixture.targets)
    gepfm = "GEpFM drift"
    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "protocol": (
            "head-to-head temporal evasion on identical dataset04 windows: the §6.2 "
            "constant-hold attacker (<=%d sensors held at one in-range value over <=%d "
            "steps) vs the adaptive per-timestep attacker (same sensor budget, a distinct "
            "in-range value at every step), warm-started from the constant-hold optimum so "
            "it is provably at least as strong; families/surrogate/box/thresholds fit on "
            "dataset03" % (max_changed, max_attack_len)
        ),
        "seed": fixture.seed,
        "sequence_length": seq_len,
        "context_len": fixture.context_len,
        "max_attack_len": max_attack_len,
        "max_changed": max_changed,
        "constant_budget": constant_budget,
        "adaptive_budget": adaptive_budget,
        "n_targets": n,
        "families": list(FAMILIES),
        "per_family_fired": fired,
        "constant": _aggregate(evaded_c, fired, still_c, n),
        "adaptive": _aggregate(evaded_a, fired, still_a, n),
        "adaptive_extra_evasions": {"count": len(extra_run_starts), "run_starts": extra_run_starts},
        "gepfm": {
            "fired": fired[gepfm],
            "still_constant": still_c[gepfm],
            "still_adaptive": still_a[gepfm],
            "silenced_constant": fired[gepfm] - still_c[gepfm],
            "silenced_adaptive": fired[gepfm] - still_a[gepfm],
        },
        "mean_queries": {
            "constant": float(np.mean(q_const)) if q_const else 0.0,
            "adaptive_refine": float(np.mean(q_adapt_refine)) if q_adapt_refine else 0.0,
            "adaptive_total": float(np.mean([c + r for c, r in zip(q_const, q_adapt_refine)]))
            if q_const else 0.0,
        },
        "targets": targets_out,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_adaptive_comparison_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    c, a = result["constant"], result["adaptive"]
    clo, chi = c["joint_evasion_wilson95"]
    alo, ahi = a["joint_evasion_wilson95"]
    g = result["gepfm"]
    lines = [
        f"Constant-hold vs adaptive per-timestep temporal evasion on {result['evaluate_dataset']} "
        f"({result['n_targets']} flagged windows; fit on {result['train_dataset']}).",
        f"Both attackers own <= {result['max_changed']} sensors over <= {result['max_attack_len']} "
        f"steps; adaptive gets per-timestep freedom + a larger budget "
        f"({result['adaptive_budget']} vs {result['constant_budget']}) and is warm-started from the "
        f"constant-hold optimum (provably >= as strong).",
        "",
        f"Joint evasion (all firing families silenced at once):",
        f"  constant-hold : {c['joint_evasion_rate']:.3f} [{clo:.3f}, {chi:.3f}]  "
        f"({c['n_jointly_evaded']}/{result['n_targets']})",
        f"  adaptive      : {a['joint_evasion_rate']:.3f} [{alo:.3f}, {ahi:.3f}]  "
        f"({a['n_jointly_evaded']}/{result['n_targets']})",
        f"  adaptive evaded {result['adaptive_extra_evasions']['count']} window(s) the constant "
        f"hold could not: {result['adaptive_extra_evasions']['run_starts']}",
        "",
        f"Per-family still firing after the JOINT attack (higher = more robust / orthogonal):",
        "",
        f"{'family':<18}{'fired':>8}{'still (const)':>15}{'still (adaptive)':>18}",
        "-" * 59,
    ]
    for fam in result["families"]:
        fired = result["per_family_fired"][fam]
        sc = c["per_family_still_firing_after_joint_attack"][fam]
        sa = a["per_family_still_firing_after_joint_attack"][fam]
        lines.append(f"{fam:<18}{fired:>8}{sc:>15}{sa:>18}")
    lines += [
        "",
        f"GEpFM drift (the temporal family): fired on {g['fired']} windows; silenced by the "
        f"constant hold on {g['silenced_constant']}, by the adaptive attacker on "
        f"{g['silenced_adaptive']}.",
        "",
        "Reading: if GEpFM is never silenced even by the adaptive attacker, the temporal drift",
        "family's non-co-evadability holds against per-timestep freedom, not just the constant-",
        "hold lower bound -- the §6.2 orthogonality result is strengthened, not merely caveated.",
    ]
    return "\n".join(lines)


def plot_adaptive_comparison(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = result["families"]
    fired = [result["per_family_fired"][f] for f in fams]
    still_c = [result["constant"]["per_family_still_firing_after_joint_attack"][f] for f in fams]
    still_a = [result["adaptive"]["per_family_still_firing_after_joint_attack"][f] for f in fams]
    x = np.arange(len(fams))

    fig, (axr, axb) = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"width_ratios": [0.85, 1]})

    c, a = result["constant"], result["adaptive"]
    clo, chi = c["joint_evasion_wilson95"]
    alo, ahi = a["joint_evasion_wilson95"]
    axr.bar([0], [c["joint_evasion_rate"]], width=0.4, color="#4393c3", label="constant hold",
            yerr=[[c["joint_evasion_rate"] - clo], [chi - c["joint_evasion_rate"]]], capsize=7)
    axr.bar([0.55], [a["joint_evasion_rate"]], width=0.4, color="#b2182b", label="adaptive",
            yerr=[[a["joint_evasion_rate"] - alo], [ahi - a["joint_evasion_rate"]]], capsize=7)
    axr.set_xlim(-0.5, 1.05)
    axr.set_ylim(0.0, 1.0)
    axr.set_xticks([0, 0.55])
    axr.set_xticklabels(["constant\nhold", "adaptive\nper-step"], fontsize=9)
    axr.set_ylabel("joint evasion rate")
    axr.set_title("Joint evasion: constant vs adaptive", fontsize=10)
    axr.grid(axis="y", alpha=0.3)

    axb.bar(x - 0.26, fired, width=0.25, label="fired on true attack", color="#999999")
    axb.bar(x, still_c, width=0.25, label="still fires (constant)", color="#4393c3")
    axb.bar(x + 0.26, still_a, width=0.25, label="still fires (adaptive)", color="#b2182b")
    axb.set_xticks(x)
    axb.set_xticklabels(fams, fontsize=9)
    axb.set_ylabel("# attack windows")
    axb.set_title("Per-family survival under the joint attack", fontsize=10)
    axb.legend(fontsize=8)
    axb.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    parser = argparse.ArgumentParser(
        description="Head-to-head constant-hold vs adaptive per-timestep temporal evasion"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--constant-budget", type=int, default=3000)
    parser.add_argument("--adaptive-budget", type=int, default=8000)
    parser.add_argument("--outdir", default="artifacts/adaptive_evasion")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    epochs = 12 if args.quick else args.max_epochs
    targets = 6 if args.quick else args.max_targets
    result = run_adaptive_evasion_comparison(
        config, mode=mode, seed=args.seed, max_epochs=epochs, max_targets=targets,
        constant_budget=args.constant_budget, adaptive_budget=args.adaptive_budget,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "adaptive_evasion.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_adaptive_comparison_table(result)
        (outdir / "adaptive_evasion_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_adaptive_comparison(result, str(outdir / "adaptive_evasion.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'adaptive_evasion.json'}, adaptive_evasion_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

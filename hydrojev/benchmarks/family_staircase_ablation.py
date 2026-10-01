"""§6.2 defense-in-depth ablation — does each *named* orthogonal family tighten the net?

§5's staircase (each orthogonal family roughly halves stealthy evasion,
0.41 → 0.25 → 0.12 pooled) is measured on the **instantaneous proxy** detectors,
because the per-sample evasion oracle cannot touch the temporal families. This module
supplies the faithful counterpart on the architecture's OWN named families: it deploys
a growing, nested detector stack — ``{CPDZ}`` → ``{CPDZ, GEpFM}`` →
``{CPDZ, GEpFM, RAT}`` — and runs the §6.2 constant-hold trajectory attacker against
each stack on **identical** dataset04 windows (shared
:func:`~hydrojev.benchmarks.temporal_evasion.build_temporal_fixture`), measuring how
joint evasion falls as decorrelated temporal families are added. CPDZ (instantaneous)
fires on every flagged window, so the denominator is constant across rungs and the
staircase is clean.

**Monotone by construction, and honestly so.** An attacker facing a *smaller* deployed
stack is strictly freer (fewer detectors to silence at once), so it may reuse any spoof
that already silenced a *larger* stack — evasion of a superset stack implies evasion of
every subset with the *same* block. We therefore cascade the evaded flags from the
largest rung down, which removes greedy-path artifacts without ever crediting the
attacker with more than it can actually achieve. The drop from ``{CPDZ}`` to the full
stack is the faithful, named-family analogue of §5: it is exactly the *extra* evasion
resistance the decorrelated temporal families buy.

Leakage-safe (families/surrogate/box/thresholds fit on dataset03); returns
``status="not_run"`` (never fabricated numbers) when BATADAL/torch are unavailable.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks.temporal_evasion import (
    FAMILIES,
    _families_alarming,
    build_temporal_fixture,
    temporal_conceal,
)


# --------------------------------------------------------------------------- #
# Aggregation (pure)
# --------------------------------------------------------------------------- #


def _rung_summary(
    deployed: Sequence[str],
    silent_flags: Sequence[bool],
    fired_flags: Sequence[bool],
) -> dict[str, Any]:
    """Evasion rate for one deployed stack over ALL flagged windows (fixed denominator).

    Denominator is every genuine attack window in the fixture, so it is constant across
    rungs and the staircase is directly comparable to §5's pooled evasion rate. A window
    on which the deployed subset never fired counts as evaded (the deployed defense missed
    a real attack) — exactly the coverage a larger stack is meant to buy. ``fired_flags``
    is carried only as a diagnostic (how many windows this stack alarmed on originally).
    """

    from hydrojev.benchmarks.evasion_benchmark import wilson_interval

    n = len(silent_flags)
    n_evaded = int(sum(bool(s) for s in silent_flags))
    n_fired = int(sum(bool(f) for f in fired_flags))
    rate = (n_evaded / n) if n else float("nan")
    lo, hi = wilson_interval(n_evaded, n) if n else (float("nan"), float("nan"))
    return {
        "deployed": list(deployed),
        "n_deployed": len(deployed),
        "n_windows": n,
        "n_fired_originally": n_fired,
        "n_evaded": n_evaded,
        "joint_evasion_rate": rate,
        "joint_evasion_wilson95": [lo, hi],
    }


# --------------------------------------------------------------------------- #
# Real-data runner
# --------------------------------------------------------------------------- #


def run_family_staircase_ablation(
    config: Any,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 60,
    max_attack_len: int = 24,
    max_targets: int = 20,
    max_changed: int = 4,
    candidate_values: int = 11,
    budget: int = 3000,
) -> dict[str, Any]:
    """Evade a growing nested stack of named families on identical dataset04 windows."""

    fixture = build_temporal_fixture(
        config, mode=mode, seed=seed, max_epochs=max_epochs,
        max_attack_len=max_attack_len, max_targets=max_targets,
    )
    if isinstance(fixture, dict):  # not_run
        return fixture

    conceal_kw = dict(
        reference=fixture.reference, thresholds=fixture.thresholds, scales=fixture.scales,
        seq_len=fixture.seq_len, box_low=fixture.box_low, box_high=fixture.box_high,
        modifiable_mask=fixture.modifiable_mask, max_changed=max_changed,
        candidate_values=candidate_values, budget=budget,
    )
    thresholds = fixture.thresholds
    targets = fixture.targets
    n = len(targets)

    # Honest n=5 ceiling (§6.2 scope (i)): how many dataset04 scenarios exist, how many
    # the named-family stack actually detects (only those are evasion targets), and why
    # the rest drop. Carried straight from the shared fixture's single-source-of-truth
    # selection report so the paper's ceiling numbers are reproducible, never fabricated.
    report = list(fixture.selection_report)
    dropped = [
        {"run_start": e["run_start"], "drop_reason": e["drop_reason"],
         "family_scores": e["family_scores"]}
        for e in report if not e["selected"]
    ]
    selection_ceiling = {
        "n_scenarios": len(report),
        "n_selected": sum(1 for e in report if e["selected"]),
        "n_dropped": len(dropped),
        "dropped": dropped,
        "note": (
            "n is a hard property of the real benchmark, not a tunable: only scenarios "
            "the named-family stack detects within a causal window are evasion targets."
        ),
    }

    # Nested cumulative deployment: instantaneous CPDZ first, then the temporal families.
    rungs: list[tuple[str, ...]] = [tuple(FAMILIES[:k]) for k in range(1, len(FAMILIES) + 1)]
    fired = [[False] * n for _ in rungs]
    silent = [[False] * n for _ in rungs]  # deployed stack raises no alarm after the attack
    detail: list[list[dict[str, Any]]] = [[{} for _ in range(n)] for _ in rungs]

    for ri, deployed in enumerate(rungs):
        for ti, target in enumerate(targets):
            res = temporal_conceal(
                fixture.predict_fn, target.context, target.attack,
                target_families=deployed, **conceal_kw,
            )
            # "silent" = NO deployed family alarms at the end (whether it never fired, was
            # actively silenced, or would have been newly triggered by the spoof). Read
            # from adversarial_scores over the deployed set, NOT res.evaded_all — the latter
            # is False on never-fired windows and ignores families the attack newly trips.
            final_alarm = _families_alarming(res.adversarial_scores, thresholds)
            deployed_still = tuple(f for f in final_alarm if f in deployed)
            bystanders = [f for f in final_alarm if f not in deployed]
            fired[ri][ti] = len(res.originally_alarming) > 0
            silent[ri][ti] = len(deployed_still) == 0
            detail[ri][ti] = {
                "run_start": int(target.run_start),
                "deployed": list(deployed),
                "originally_alarming": list(res.originally_alarming),
                "deployed_still_firing": list(deployed_still),
                "silent": bool(len(deployed_still) == 0),
                "actively_evaded_all_fired": bool(res.evaded_all),
                "bystanders_still_firing": bystanders,
                "n_changed_sensors": res.n_changed,
                "scaled_l2": res.scaled_l2,
                "queries": res.query_count,
                "termination": res.termination_reason,
            }

    # Monotone cascade (largest rung -> smallest). SOUND, not heuristic: if the block the
    # attacker found drives EVERY family of a superset stack below threshold, that same
    # block drives every family of any subset below threshold too. So evasion of a larger
    # deployed stack logically implies evasion of every smaller one, and the staircase is
    # non-increasing in |deployed| by construction while only ever crediting the attacker
    # with spoofs it actually achieved.
    for ri in range(len(rungs) - 2, -1, -1):
        for ti in range(n):
            if silent[ri + 1][ti]:
                silent[ri][ti] = True

    staircase = [_rung_summary(rungs[ri], silent[ri], fired[ri]) for ri in range(len(rungs))]

    # Per-single-family evadability (each family deployed ALONE), for the "orthogonal
    # agreement collapses evasion" contrast: single detectors are more evadable — or miss
    # more windows outright — than the stack. Same fixed denominator (all flagged windows).
    single: dict[str, Any] = {}
    single_detail: dict[str, list[dict[str, Any]]] = {}
    for fam in FAMILIES:
        f_fired = [False] * n
        f_silent = [False] * n
        rows: list[dict[str, Any]] = []
        for ti, target in enumerate(targets):
            res = temporal_conceal(
                fixture.predict_fn, target.context, target.attack,
                target_families=(fam,), **conceal_kw,
            )
            final_alarm = _families_alarming(res.adversarial_scores, thresholds)
            f_fired[ti] = len(res.originally_alarming) > 0
            f_silent[ti] = fam not in final_alarm
            rows.append({
                "run_start": int(target.run_start),
                "fired": f_fired[ti],
                "silent": bool(fam not in final_alarm),
                "n_changed_sensors": res.n_changed,
                "termination": res.termination_reason,
            })
        single[fam] = _rung_summary((fam,), f_silent, f_fired)
        single_detail[fam] = rows

    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "protocol": (
            "faithful named-family defense-in-depth ablation: the §6.2 constant-hold "
            "trajectory attacker (<=%d in-range sensors held over <=%d steps) is run "
            "against a growing nested stack of named families deployed as one ensemble; "
            "joint evasion = every deployed family that fired on the true attack driven "
            "below threshold. Monotone cascade (a smaller stack is strictly easier) "
            "removes greedy-path artifacts. Families/surrogate/box/thresholds fit on "
            "dataset03" % (max_changed, max_attack_len)
        ),
        "seed": fixture.seed,
        "sequence_length": fixture.seq_len,
        "context_len": fixture.context_len,
        "max_attack_len": max_attack_len,
        "max_changed": max_changed,
        "budget": budget,
        "n_targets": n,
        "families": list(FAMILIES),
        "deployment_order": [list(r) for r in rungs],
        "selection_ceiling": selection_ceiling,
        "window_selection": report,
        "staircase": staircase,
        "single_family": single,
        "targets": [detail[ri] for ri in range(len(rungs))],
        "single_family_targets": single_detail,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_staircase_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    n_windows = result["n_targets"]
    ceiling = result.get("selection_ceiling", {})
    lines = [
        f"Faithful named-family defense-in-depth ablation on {result['evaluate_dataset']} "
        f"({n_windows} flagged windows; fit on {result['train_dataset']}).",
        f"Attacker: §6.2 constant-hold trajectory spoof (<= {result['max_changed']} sensors, "
        f"<= {result['max_attack_len']} steps). Deployed stack grows CPDZ -> +GEpFM -> +RAT.",
        f"Evasion = deployed stack silent after attack, over ALL {n_windows} windows "
        f"(fixed denominator).",]
    if ceiling:
        dropped_ids = ", ".join(str(d["run_start"]) for d in ceiling.get("dropped", []))
        lines.append(
            f"Window ceiling: {result['evaluate_dataset']} has {ceiling['n_scenarios']} "
            f"attack scenarios; the stack flags {ceiling['n_selected']} within a "
            f"<= {result['max_attack_len']}-step causal window"
            + (f" ({ceiling['n_dropped']} already-missed, not evasion targets: {dropped_ids})"
               if ceiling.get("dropped") else "")
            + ". n is a data ceiling, not a cap."
        )
    lines += [
        "",
        f"{'deployed stack':<34}{'fired':>7}{'evaded':>8}{'evasion rate [95% Wilson]':>30}",
        "-" * 79,
    ]
    for rung in result["staircase"]:
        label = " + ".join(f.split()[0] for f in rung["deployed"])  # CPDZ + GEpFM + RAT
        lo, hi = rung["joint_evasion_wilson95"]
        rate = rung["joint_evasion_rate"]
        rate_s = "n/a" if not np.isfinite(rate) else f"{rate:.3f} [{lo:.3f}, {hi:.3f}]"
        lines.append(
            f"{label:<34}{rung['n_fired_originally']:>7}{rung['n_evaded']:>8}{rate_s:>30}"
        )
    lines += [
        "",
        "Single-family evasion (each family deployed ALONE; higher = misses/evaded more):",
        "",
        f"{'family':<18}{'fired':>7}{'evaded':>8}{'evasion rate':>16}",
        "-" * 49,
    ]
    for fam in result["families"]:
        s = result["single_family"][fam]
        rate = s["joint_evasion_rate"]
        rate_s = "n/a" if not np.isfinite(rate) else f"{rate:.3f}"
        lines.append(f"{fam:<18}{s['n_fired_originally']:>7}{s['n_evaded']:>8}{rate_s:>16}")
    first = result["staircase"][0]["joint_evasion_rate"]
    last = result["staircase"][-1]["joint_evasion_rate"]
    lines += [
        "",
        "Reading: the drop from the CPDZ-only stack to the full stack is the extra evasion",
        "resistance the decorrelated TEMPORAL families buy — the faithful, named-family",
        f"counterpart of the §5 proxy staircase. Here: {first:.3f} -> {last:.3f}.",
    ]
    return "\n".join(lines)


def plot_staircase(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    staircase = result["staircase"]
    labels = [" + ".join(f.split()[0] for f in r["deployed"]) for r in staircase]
    rates = [r["joint_evasion_rate"] for r in staircase]
    los = [r["joint_evasion_wilson95"][0] for r in staircase]
    his = [r["joint_evasion_wilson95"][1] for r in staircase]
    x = np.arange(len(staircase))
    yerr = [[max(0.0, rt - lo) for rt, lo in zip(rates, los)],
            [max(0.0, hi - rt) for rt, hi in zip(rates, his)]]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.step(x, rates, where="mid", color="#2166ac", lw=2, zorder=2)
    ax.errorbar(x, rates, yerr=yerr, fmt="o", color="#b2182b", capsize=6, zorder=3)
    for xi, rt in zip(x, rates):
        ax.annotate(f"{rt:.2f}", (xi, rt), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("joint evasion rate")
    ax.set_xlabel("deployed named-family stack (nested)")
    ax.set_title("Faithful defense-in-depth staircase (named families, development set)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

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
        description="Faithful named-family defense-in-depth ablation (§6.2 staircase)"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--outdir", default="artifacts/family_staircase")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    epochs = 12 if args.quick else args.max_epochs
    targets = 8 if args.quick else args.max_targets
    result = run_family_staircase_ablation(
        config, mode=mode, seed=args.seed, max_epochs=epochs, max_targets=targets, budget=args.budget
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "family_staircase.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_staircase_table(result)
        (outdir / "family_staircase_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_staircase(result, str(outdir / "family_staircase.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'family_staircase.json'}, family_staircase_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

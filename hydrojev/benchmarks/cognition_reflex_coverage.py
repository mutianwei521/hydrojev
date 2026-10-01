"""Cognition vs reflex: orthogonal *coverage domains* on real BATADAL data.

The paper makes two independent claims that must be shown to work *together*:

* the **cognition** layer (the named orthogonal residual families of §6) detects
  stealthy cyber-physical attacks, and
* the deterministic, air-gapped **reflex** layer (§7) reads the *true physical
  state* and neutralises the hydraulic consequence of any attack that evades
  detection.

A reviewer will immediately ask: *do these two layers actually cover different
things, or is one redundant?* This module answers that on real ``dataset04`` by
measuring, per labelled attack scenario, whether

1. the cognition layer alarms (any named family exceeds its ``dataset03``-fit
   threshold at any point in the scenario), and
2. the attack drives any tank **above its entire ``dataset03`` healthy operating
   envelope** — the leakage-safe, data-domain analogue of the §7 reflex band (a
   float/pressure interlock that is silent unless the tank goes abnormally high).

The finding this makes measurable is the architecture's *division of labour*:

* Most real BATADAL attacks are stealthy — they never push physics past the normal
  envelope, so a true-level reflex correctly stays silent and **only the cognition
  layer catches them** (``cognition_only``). The reflex is not a detector.
* A minority drive a tank abnormally high — a genuine hydraulic-catastrophe risk
  and exactly what §7's reflex neutralises. There the reflex is the guaranteed
  floor, and where cognition *also* fires we report the concrete timing (cognition
  lead over the physical crossing).

Neither layer alone covers every scenario; together they do. This is the honest,
*measured* replacement for the hand-wavy "the reflex also keys on residuals" —
the reflex deliberately does **not** key on evadable residuals (that would
reintroduce evadability into the guarantee layer); the residual families are the
cognition layer, and the two occupy orthogonal coverage domains.

Everything is leakage-safe: the surrogate, family references, thresholds, and the
tank envelope are all estimated on attack-free ``dataset03`` and applied causally
to ``dataset04``. The runner returns ``status="not_run"`` (never fabricated
numbers) when BATADAL or a tank-level column is unavailable.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig
from hydrojev.benchmarks.detection_baselines import contiguous_runs
from hydrojev.benchmarks.named_family_benchmark import build_named_family_scores


# --------------------------------------------------------------------------- #
# Pure, unit-testable helpers (no torch, no EPANET)
# --------------------------------------------------------------------------- #


def healthy_level_envelope(clean_levels: np.ndarray) -> np.ndarray:
    """Per-tank healthy operating ceiling = column max over attack-free rows.

    ``clean_levels`` is ``[samples, tanks]`` from ``dataset03`` (attack-free), so
    the returned ceiling is the highest level each tank reaches in normal
    operation — a leakage-safe abnormality band.
    """

    arr = np.asarray(clean_levels, dtype=float)
    if arr.ndim != 2:
        raise ValueError("clean_levels must be 2D [samples, tanks]")
    if arr.shape[0] < 1:
        raise ValueError("clean_levels needs at least one healthy sample")
    if not np.all(np.max(arr, axis=0) > 0.0):
        raise ValueError("every tank ceiling must be positive")
    return np.max(arr, axis=0)


def physical_overlevel_series(
    eval_levels: np.ndarray, ceiling: np.ndarray, *, margin: float = 0.0
) -> np.ndarray:
    """Per-timestep boolean: any tank strictly exceeds ``ceiling * (1 + margin)``.

    This is the (true-state) trigger an air-gapped level reflex would act on — it
    fires only when a tank is driven above its healthy operating envelope.
    """

    arr = np.asarray(eval_levels, dtype=float)
    ceil = np.asarray(ceiling, dtype=float)
    if arr.ndim != 2:
        raise ValueError("eval_levels must be 2D [samples, tanks]")
    if ceil.shape != (arr.shape[1],):
        raise ValueError("ceiling shape must be (n_tanks,)")
    if margin < 0.0:
        raise ValueError("margin must be >= 0")
    band = ceil * (1.0 + margin)
    return np.any(arr > band, axis=1)


def overlevel_ratio_series(eval_levels: np.ndarray, ceiling: np.ndarray) -> np.ndarray:
    """Per-timestep max-over-tanks of ``level / ceiling`` (how far over the envelope)."""

    arr = np.asarray(eval_levels, dtype=float)
    ceil = np.asarray(ceiling, dtype=float)
    if arr.ndim != 2:
        raise ValueError("eval_levels must be 2D [samples, tanks]")
    if ceil.shape != (arr.shape[1],):
        raise ValueError("ceiling shape must be (n_tanks,)")
    if not np.all(ceil > 0.0):
        raise ValueError("every tank ceiling must be positive")
    return np.max(arr / ceil, axis=1)


def cognition_alarm_series(
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    families: Sequence[str],
) -> np.ndarray:
    """Per-timestep OR over named families of ``finite(score) and score >= threshold``.

    Non-finite scores (e.g. a family's warmup window) count as *not* alarming.
    """

    families = tuple(families)
    if not families:
        raise ValueError("families must be non-empty")
    fired: np.ndarray | None = None
    length: int | None = None
    for fam in families:
        sc = np.asarray(scores[fam], dtype=float)
        if sc.ndim != 1:
            raise ValueError(f"score series for {fam!r} must be 1D")
        if length is None:
            length = sc.shape[0]
        elif sc.shape[0] != length:
            raise ValueError("all family score series must share length")
        thr = float(thresholds[fam])
        f = np.isfinite(sc) & (sc >= thr)
        fired = f if fired is None else (fired | f)
    assert fired is not None
    return fired


def first_true_in_runs(
    flag: np.ndarray, runs: Sequence[tuple[int, int]]
) -> list[int | None]:
    """First absolute index inside each ``(a, b)`` run where ``flag`` is True, else None."""

    flag = np.asarray(flag, dtype=bool)
    out: list[int | None] = []
    for a, b in runs:
        window = np.where(flag[a:b])[0]
        out.append(int(a + window[0]) if window.size else None)
    return out


def coverage_crosstab(
    cognition_flag: np.ndarray,
    physical_flag: np.ndarray,
    runs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Per-scenario cognition/physical catch and the aggregate coverage cross-tab.

    ``cognition_lead_steps`` = physical-crossing index − cognition-alarm index
    (positive ⇒ cognition alarms *before* the tank breaches the envelope). It is
    ``None`` unless both fire in the scenario.
    """

    cog_first = first_true_in_runs(cognition_flag, runs)
    phys_first = first_true_in_runs(physical_flag, runs)
    scenarios: list[dict[str, Any]] = []
    for (a, b), cf, pf in zip(runs, cog_first, phys_first):
        lead = (pf - cf) if (cf is not None and pf is not None) else None
        scenarios.append(
            {
                "start": int(a),
                "end": int(b),
                "length": int(b - a),
                "cognition_alarms": cf is not None,
                "cognition_first_step": (cf - a) if cf is not None else None,
                "physical_abnormal": pf is not None,
                "physical_first_step": (pf - a) if pf is not None else None,
                "cognition_lead_steps": lead,
            }
        )
    n = len(scenarios)
    cog = np.array([s["cognition_alarms"] for s in scenarios], dtype=bool)
    phys = np.array([s["physical_abnormal"] for s in scenarios], dtype=bool)
    aggregate = {
        "n_scenarios": n,
        "cognition_recall": float(cog.mean()) if n else float("nan"),
        "physical_abnormal_count": int(phys.sum()),
        "cognition_only": int(np.sum(cog & ~phys)),
        "physical_only": int(np.sum(phys & ~cog)),
        "both": int(np.sum(cog & phys)),
        "neither": int(np.sum(~cog & ~phys)),
        "union_recall": float(np.mean(cog | phys)) if n else float("nan"),
    }
    return {"scenarios": scenarios, "aggregate": aggregate}


# --------------------------------------------------------------------------- #
# Real-data runner
# --------------------------------------------------------------------------- #


def run_cognition_reflex_coverage(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 60,
    level_margin: float = 0.0,
) -> dict[str, Any]:
    """Measure cognition vs reflex coverage over labelled ``dataset04`` scenarios."""

    built = build_named_family_scores(config, mode=mode, seed=seed, max_epochs=max_epochs)
    if isinstance(built, dict):  # not_run
        return built

    from hydrojev.datasets.wdn_loader import (
        DatasetUnavailableError,
        classify_batadal_columns,
        load_batadal,
    )

    try:
        clean = load_batadal("dataset03", config=config)
        labelled = load_batadal("dataset04", config=config)
    except DatasetUnavailableError as exc:  # pragma: no cover - guarded above already
        return {"status": "not_run", "reason": str(exc)}

    tanks = classify_batadal_columns(clean.feature_columns)["tank_level"]
    tanks = tuple(t for t in tanks if t in labelled.frame.columns)
    if not tanks:
        return {"status": "not_run", "reason": "no tank_level columns available in BATADAL"}

    clean_levels = clean.frame[list(tanks)].to_numpy(dtype=float)
    ceiling = healthy_level_envelope(clean_levels)
    eval_levels_full = labelled.frame[list(tanks)].to_numpy(dtype=float)

    # Align raw tank levels to the scored/labelled window of build_named_family_scores.
    n = int(built.labels.shape[0])
    eval_levels = eval_levels_full[built.start_index : built.start_index + n]
    if eval_levels.shape[0] != n:  # defensive: keep everything the same length
        n = min(n, eval_levels.shape[0])
        eval_levels = eval_levels[:n]

    labels = built.labels[:n]
    runs = contiguous_runs(labels, 1)

    cognition = cognition_alarm_series(
        {f: built.scores[f][:n] for f in built.families}, built.thresholds, built.families
    )
    physical = physical_overlevel_series(eval_levels, ceiling, margin=level_margin)
    ratio = overlevel_ratio_series(eval_levels, ceiling)

    crosstab = coverage_crosstab(cognition, physical, runs)

    # Enrich each scenario with the peak over-level margin, which named families
    # fired, and absolute dataset04 indices for traceability.
    for scenario, (a, b) in zip(crosstab["scenarios"], runs):
        scenario["abs_start"] = int(built.start_index + a)
        scenario["abs_end"] = int(built.start_index + b)
        scenario["peak_overlevel_ratio"] = (
            float(np.max(ratio[a:b])) if b > a else float("nan")
        )
        firing: list[str] = []
        for fam in built.families:
            seg = np.asarray(built.scores[fam][:n][a:b], dtype=float)
            finite = seg[np.isfinite(seg)]
            if finite.size and float(finite.max()) >= float(built.thresholds[fam]):
                firing.append(fam)
        scenario["families_alarming"] = firing

    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "protocol": (
            "cognition = OR of the named residual families (CPDZ/GEpFM/RAT on a GRU one-step-ahead "
            "surrogate) exceeding their dataset03-fit 99.5% thresholds; reflex band = each tank's "
            "dataset03 healthy operating max (level_margin applied); a scenario is 'physically "
            "abnormal' iff some tank is driven above that band (what an air-gapped true-level reflex "
            "would act on). Surrogate/references/thresholds/envelope all fit on dataset03; scored "
            "causally on labelled dataset04. Cross-tabulates the two layers' per-scenario coverage."
        ),
        "seed": int(config.experiments.seed if seed is None else seed),
        "sequence_length": int(config.surrogate.sequence_length),
        "start_index": int(built.start_index),
        "n_scored_samples": int(n),
        "level_margin": float(level_margin),
        "n_tanks": len(tanks),
        "tank_columns": list(tanks),
        "tank_ceiling": {t: float(c) for t, c in zip(tanks, ceiling)},
        "families": list(built.families),
        "scenarios": crosstab["scenarios"],
        "aggregate": crosstab["aggregate"],
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_coverage_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    agg = result["aggregate"]
    lines = [
        f"Cognition vs reflex coverage on {result['evaluate_dataset']} "
        f"({agg['n_scenarios']} attack scenarios; families + tank envelope fit on "
        f"{result['train_dataset']}).",
        f"Reflex band = per-tank dataset03 healthy max (level_margin={result['level_margin']:.0%}); "
        f"a scenario is 'physical' iff a tank is driven above its band.",
        "",
        f"{'scenario (abs)':<18}{'len':>5}{'cognition@':>12}{'physical@':>12}"
        f"{'peak x':>8}{'lead':>7}  families",
        "-" * 82,
    ]
    for s in result["scenarios"]:
        cog = f"+{s['cognition_first_step']}" if s["cognition_alarms"] else "-"
        phys = f"+{s['physical_first_step']}" if s["physical_abnormal"] else "-"
        lead = "" if s["cognition_lead_steps"] is None else f"{s['cognition_lead_steps']:+d}"
        fams = ",".join(f.split()[0] for f in s["families_alarming"]) or "-"
        lines.append(
            f"[{s['abs_start']}:{s['abs_end']}]".ljust(18)
            + f"{s['length']:>5}{cog:>12}{phys:>12}"
            + f"{s['peak_overlevel_ratio']:>8.2f}{lead:>7}  {fams}"
        )
    lines += [
        "",
        f"cognition recall            : {agg['cognition_recall']:.3f}  "
        f"({int(round(agg['cognition_recall'] * agg['n_scenarios']))}/{agg['n_scenarios']} scenarios)",
        f"physically-abnormal (reflex): {agg['physical_abnormal_count']}/{agg['n_scenarios']} "
        f"scenarios drive a tank above its healthy envelope",
        f"cognition-only              : {agg['cognition_only']}  (stealthy: only cognition catches)",
        f"both layers                 : {agg['both']}  (physical event cognition also flags)",
        f"physical-only               : {agg['physical_only']}  (physical event cognition misses)",
        f"neither                     : {agg['neither']}",
        f"union recall (both layers)  : {agg['union_recall']:.3f}",
    ]
    return "\n".join(lines)


def plot_coverage(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = result["aggregate"]
    scenarios = result["scenarios"]

    fig, (axc, axr) = plt.subplots(1, 2, figsize=(12.4, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

    # Left: coverage decomposition across scenarios.
    cats = ["cognition\nonly", "both\nlayers", "physical\nonly", "neither"]
    vals = [agg["cognition_only"], agg["both"], agg["physical_only"], agg["neither"]]
    colors = ["#2166ac", "#6a51a3", "#b2182b", "#bdbdbd"]
    axc.bar(cats, vals, color=colors)
    axc.set_ylabel("attack scenarios")
    axc.set_title(
        "Cognition and reflex cover orthogonal attack populations\n"
        f"(dataset04, n={agg['n_scenarios']}; union recall {agg['union_recall']:.2f})",
        fontsize=9,
    )
    for i, v in enumerate(vals):
        axc.text(i, v + 0.03, str(v), ha="center", va="bottom", fontsize=9)
    axc.set_ylim(0, max(vals) + 1 if vals else 1)
    axc.grid(axis="y", alpha=0.3)

    # Right: per-scenario peak over-level ratio vs the reflex band (=1.0).
    x = np.arange(len(scenarios))
    ratios = [s["peak_overlevel_ratio"] for s in scenarios]
    bar_colors = ["#b2182b" if s["physical_abnormal"] else "#92c5de" for s in scenarios]
    axr.bar(x, ratios, color=bar_colors)
    axr.axhline(1.0 + result["level_margin"], color="black", ls="--", lw=1.2,
                label="reflex band (healthy max)")
    axr.set_xticks(x)
    axr.set_xticklabels([f"{s['abs_start']}" for s in scenarios], rotation=25, ha="right", fontsize=8)
    axr.set_ylabel("peak tank level / healthy max")
    axr.set_xlabel("attack scenario (dataset04 onset index)")
    axr.set_title(
        "Most attacks never breach the physical envelope\n"
        "(red = drives a tank abnormally high → reflex acts)",
        fontsize=9,
    )
    axr.legend(fontsize=8)
    axr.grid(axis="y", alpha=0.3)

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
        description="Cognition vs reflex coverage on real BATADAL dataset04"
    )
    parser.add_argument("--quick", action="store_true", help="bounded fast smoke run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--level-margin", type=float, default=0.0,
                        help="fractional margin above the healthy envelope for the reflex band")
    parser.add_argument("--outdir", default="artifacts/cognition_reflex_coverage")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    epochs = 12 if args.quick else args.max_epochs
    result = run_cognition_reflex_coverage(
        config, mode=mode, seed=args.seed, max_epochs=epochs, level_margin=args.level_margin
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cognition_reflex_coverage.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_coverage_table(result)
        (outdir / "coverage_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_coverage(result, str(outdir / "cognition_reflex_coverage.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'cognition_reflex_coverage.json'}, coverage_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

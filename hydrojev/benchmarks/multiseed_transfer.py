"""Multi-seed aggregation of the transfer / ensemble-evasion benchmark.

The single-seed transfer benchmark (:mod:`transfer_benchmark`) establishes the
qualitative story; a *Water Research*-grade claim needs its two headline numbers
-- the autoencoder-alone evasion rate and the ensemble joint-evasion rate -- to be
stable across the stochastic parts of the pipeline (autoencoder initialisation and
training, Isolation-Forest sampling). This module re-runs the whole transfer
benchmark under several seeds and reports, per quantity:

* the per-seed rates (transparency),
* their mean and standard deviation across seeds, and
* a *pooled* estimate: sum successes and trials over all seeds and take one Wilson
  95% interval on the pooled counts (each attacked alarm is an independent trial).

Pooling and the per-seed spread answer different questions -- the pooled interval
is the tightest honest statement of the rate, the spread shows seed sensitivity --
so both are reported. The runner degrades to ``status="not_run"`` if BATADAL is
unavailable for the first seed.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig
from hydrojev.benchmarks.evasion_benchmark import wilson_interval


def _pool(per_seed_evaded: Sequence[int], per_seed_targets: Sequence[int]) -> dict[str, Any]:
    """Mean/std of per-seed rates plus a pooled Wilson interval over all trials."""

    rates = [
        (e / t if t else float("nan"))
        for e, t in zip(per_seed_evaded, per_seed_targets)
    ]
    finite = [r for r in rates if np.isfinite(r)]
    pooled_evaded = int(np.sum(per_seed_evaded))
    pooled_targets = int(np.sum(per_seed_targets))
    pooled_rate = pooled_evaded / pooled_targets if pooled_targets else float("nan")
    lo, hi = wilson_interval(pooled_evaded, pooled_targets)
    return {
        "per_seed_rates": [float(r) for r in rates],
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "std": float(np.std(finite)) if finite else float("nan"),
        "n_seeds": len(finite),
        "pooled_evaded": pooled_evaded,
        "pooled_targets": pooled_targets,
        "pooled_rate": pooled_rate,
        "pooled_ci95_low": lo,
        "pooled_ci95_high": hi,
    }


def aggregate_seed_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine several per-seed :func:`run_transfer_benchmark` dicts.

    Only seeds whose result has ``status == "ok"`` contribute. Ensembles are
    matched by label; the AE-alone rate is pooled from its per-seed rate x target
    count. Transfer-matrix off-diagonals are averaged across seeds.
    """

    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {"status": "not_run", "reason": "no successful seed runs"}

    seeds = [int(r["seed"]) for r in ok]

    # AE-alone: recover integer evaded counts from rate x targets.
    ae_evaded, ae_targets = [], []
    for r in ok:
        t = int(r["ae_detected_targets"])
        ae_evaded.append(int(round(float(r["ae_single_evasion_rate"]) * t)))
        ae_targets.append(t)
    ae_single = _pool(ae_evaded, ae_targets)

    # Ensembles: use the stored integer counts directly.
    labels = list(ok[0]["ensembles"].keys())
    ensembles: dict[str, Any] = {}
    for label in labels:
        ev = [int(r["ensembles"][label]["n_evaded"]) for r in ok if label in r["ensembles"]]
        tg = [int(r["ensembles"][label]["n_targets"]) for r in ok if label in r["ensembles"]]
        pooled = _pool(ev, tg)
        pooled["members"] = ok[0]["ensembles"][label].get("members", [])
        # Mean orthogonal-catch rate (naive transfer) across seeds, for context.
        naive = [
            float(r["ensembles"][label]["naive_ensemble_evasion_of_ae_samples"])
            for r in ok
            if label in r["ensembles"]
            and np.isfinite(r["ensembles"][label]["naive_ensemble_evasion_of_ae_samples"])
        ]
        pooled["mean_naive_ensemble_evasion"] = float(np.mean(naive)) if naive else float("nan")
        ensembles[label] = pooled

    # Transfer matrix off-diagonal means (row evaded -> col still fires).
    order = ok[0]["detector_order"]
    transfer_mean: dict[str, dict[str, float]] = {}
    for src in order:
        transfer_mean[src] = {}
        for dst in order:
            vals = [
                float(r["transfer_matrix"][src]["still_alerts"][dst])
                for r in ok
                if np.isfinite(r["transfer_matrix"][src]["still_alerts"][dst])
            ]
            transfer_mean[src][dst] = float(np.mean(vals)) if vals else float("nan")

    return {
        "status": "ok",
        "seeds": seeds,
        "n_seeds": len(ok),
        "train_dataset": ok[0].get("train_dataset"),
        "evaluate_dataset": ok[0].get("evaluate_dataset"),
        "max_changed_sensors": ok[0].get("max_changed_sensors"),
        "query_budget": ok[0].get("query_budget"),
        "detector_order": order,
        "ae_single": ae_single,
        "ensembles": ensembles,
        "transfer_off_diagonal_mean": transfer_mean,
    }


def run_multiseed_transfer(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seeds: Sequence[int],
    max_targets: int = 60,
    budget: int = 4000,
) -> dict[str, Any]:
    """Run the transfer benchmark once per seed and aggregate the outcomes."""

    from hydrojev.benchmarks.transfer_benchmark import run_transfer_benchmark

    results = []
    for seed in seeds:
        results.append(
            run_transfer_benchmark(
                config, mode=mode, seed=int(seed), max_targets=max_targets, budget=budget
            )
        )
    agg = aggregate_seed_results(results)
    if agg.get("status") == "ok":
        agg["per_seed"] = [
            {
                "seed": r.get("seed"),
                "ae_single_evasion_rate": r.get("ae_single_evasion_rate"),
                "ensembles": {k: v.get("joint_evasion_rate") for k, v in r.get("ensembles", {}).items()},
            }
            for r in results
            if r.get("status") == "ok"
        ]
    return agg


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_multiseed_table(agg: dict[str, Any]) -> str:
    if agg.get("status") != "ok":
        return "not_run"
    lines = [
        f"Multi-seed evasion robustness over seeds {agg['seeds']} "
        f"(<= {agg['max_changed_sensors']} in-range spoofs, budget {agg['query_budget']}).",
        "Pooled rate = successes/trials over all seeds; +-std is the seed spread.",
        "",
        f"{'quantity':<28}{'pooled rate [95% CI]':>28}{'mean+-std':>16}{'trials':>9}",
        "-" * 81,
    ]

    def line(name: str, p: dict[str, Any]) -> str:
        rate = f"{p['pooled_rate']:.3f} [{p['pooled_ci95_low']:.2f},{p['pooled_ci95_high']:.2f}]"
        ms = f"{p['mean']:.3f}+-{p['std']:.3f}"
        return f"{name:<28}{rate:>28}{ms:>16}{p['pooled_targets']:>9}"

    lines.append(line("AE alone", agg["ae_single"]))
    for label, p in agg["ensembles"].items():
        lines.append(line(label, p))
    return "\n".join(lines)


def plot_multiseed(agg: dict[str, Any], path: str) -> str | None:
    if agg.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["AE alone"] + list(agg["ensembles"].keys())
    pooled = [agg["ae_single"]] + [agg["ensembles"][k] for k in agg["ensembles"]]
    rates = [p["pooled_rate"] for p in pooled]
    los = [max(0.0, rates[i] - p["pooled_ci95_low"]) for i, p in enumerate(pooled)]
    his = [max(0.0, p["pooled_ci95_high"] - rates[i]) for i, p in enumerate(pooled)]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colours = ["#9ecae1"] + ["#e6550d"] * (len(names) - 1)
    ax.bar(x, rates, yerr=[los, his], capsize=4, color=colours, zorder=2)
    # Per-seed points to show the spread honestly.
    for i, p in enumerate(pooled):
        ys = [r for r in p["per_seed_rates"] if np.isfinite(r)]
        ax.scatter([x[i]] * len(ys), ys, color="black", s=18, zorder=3, alpha=0.7)
    for i, r in enumerate(rates):
        ax.text(x[i], r + his[i] + 0.008, f"{r:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("stealthy evasion success (lower = more robust)")
    ax.set_ylim(0.0, max(0.4, max(rates) * 1.4))
    ax.set_title(
        f"Evasion robustness pooled over {agg['n_seeds']} seeds (bars = pooled rate + Wilson 95% CI,\n"
        "dots = per-seed rates). Orthogonal agreement stays more robust than the AE alone.",
        fontsize=9,
    )
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

    parser = argparse.ArgumentParser(description="Multi-seed transfer / ensemble evasion")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--max-targets", type=int, default=60)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--outdir", default="artifacts/multiseed_transfer")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    agg = run_multiseed_transfer(
        config, mode=mode, seeds=args.seeds, max_targets=args.max_targets, budget=args.budget
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "multiseed.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
    if agg.get("status") == "ok":
        table = format_multiseed_table(agg)
        (outdir / "multiseed_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_multiseed(agg, str(outdir / "multiseed.png"))
        print(table)
        print(f"\nwrote: {outdir/'multiseed.json'}, multiseed_table.txt, {figure}")
    else:
        print(json.dumps(agg, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

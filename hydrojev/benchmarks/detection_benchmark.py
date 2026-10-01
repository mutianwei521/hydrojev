"""Head-to-head detection benchmark on real BATADAL (dataset03 -> dataset04).

This runner places HydroJEV's reconstruction detector against three standard
unsupervised baselines (PCA reconstruction residual, Mahalanobis distance,
Isolation Forest) under one rigorously fair protocol:

* **Same training data** — every detector is fit on the attack-free BATADAL
  ``dataset03`` partition and its scaler/subspace/covariance/forest see no labels
  and no evaluation data.
* **Same threshold policy** — each alarm threshold is the same quantile (default
  the 99.5th percentile) of that detector's own *training* score distribution.
* **Same labelled benchmark** — all are evaluated on ``dataset04`` with both
  threshold-independent discrimination (ROC-AUC, PR-AUC) and the operating-point
  metrics a control room sees (scenario/point recall, benign false-positive rate).
* **Honest uncertainty** — the two stochastic detectors (the AE and the Isolation
  Forest) are refit across multiple seeds and reported as mean with a 95%
  Student-t confidence interval; the deterministic baselines report a point value.

If the BATADAL assets are absent the runner returns ``status="not_run"`` with the
exact reason — it never fabricates a benchmark number.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import numpy as np
from scipy import stats

from hydrojev.config import HydroJEVConfig
from hydrojev.datasets.wdn_loader import DatasetUnavailableError, load_batadal
from hydrojev.benchmarks.detection_baselines import (
    evaluate_scores,
    fit_isolation_forest_detector,
    fit_mahalanobis_detector,
    fit_pca_residual_detector,
)

# Metrics aggregated across seeds. ROC-AUC / PR-AUC are threshold-independent;
# the remainder are read at each detector's own calibrated operating point.
_METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "scenario_recall",
    "point_recall",
    "point_false_positive_rate",
    "benign_scenario_fp_rate",
)

DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean and 95% Student-t confidence interval over finite values."""

    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "ci95_low": nan, "ci95_high": nan, "n": 0}
    mean = float(arr.mean())
    if n == 1:
        # A single (deterministic) observation has no sampling spread.
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1}
    std = float(arr.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return {"mean": mean, "std": std, "ci95_low": mean - half, "ci95_high": mean + half, "n": n}


def _aggregate(runs: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate a detector's per-seed metric dicts into per-metric summaries."""

    return {key: _summarize([run[key] for run in runs]) for key in _METRIC_KEYS}


def _seed_values(runs: list[dict[str, float]]) -> dict[str, list[float]]:
    return {key: [float(run[key]) for run in runs] for key in _METRIC_KEYS}


def run_detection_benchmark(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Run the fair BATADAL detection benchmark; return a structured comparison.

    ``mode`` is a :class:`~hydrojev.run_experiments.RunMode` (used only to bound
    the AE's training rows/epochs so quick smoke runs stay fast). ``seeds`` sets
    the number of independent refits for the stochastic detectors.
    """

    # Lazy import: build_reconstruction_detector pulls in torch; keep this module
    # importable (and its pure helpers testable) without paying that cost.
    from hydrojev.run_experiments import build_reconstruction_detector

    try:
        clean = load_batadal("dataset03", config=config)
        labelled = load_batadal("dataset04", config=config)
    except DatasetUnavailableError as exc:
        return {"status": "not_run", "reason": str(exc)}

    if labelled.label_column is None:
        return {"status": "not_run", "reason": f"{labelled.name} has no attack-label column"}

    columns = list(clean.feature_columns)
    missing = [c for c in columns if c not in labelled.frame.columns]
    if missing:
        return {"status": "not_run", "reason": f"feature columns absent in {labelled.name}: {missing[:3]}"}

    quantile = float(config.concealment.threshold_quantile)
    rows = mode.batadal_train_rows
    # Attack-free training partition (dataset03) shared by every baseline. The AE
    # additionally uses dataset03's validation split for early stopping; both are
    # attack-free dataset03 rows, so no dataset04 information leaks to any detector.
    train_matrix = clean.train_frame[columns].to_numpy(dtype=float)
    if rows is not None:
        train_matrix = train_matrix[:rows]
    eval_features = labelled.frame[columns].to_numpy(dtype=float)
    eval_labels = labelled.frame[labelled.label_column].to_numpy().astype(int)

    def _score(detector) -> dict[str, float]:
        return evaluate_scores(detector.score_samples(eval_features), detector.threshold, eval_labels)

    detectors: dict[str, Any] = {}

    # ----- HydroJEV reconstruction detector (stochastic: torch init/shuffle) --- #
    ae_runs: list[dict[str, float]] = []
    ae_provenance = ""
    for seed in seeds:
        cfg_seed = dataclasses.replace(
            config, experiments=dataclasses.replace(config.experiments, seed=int(seed))
        )
        artifacts = build_reconstruction_detector(cfg_seed, mode=mode)
        ae_runs.append(_score(artifacts.detector))
        ae_provenance = (
            f"reconstruction autoencoder (n-2n-4n-2n-n); trained on {artifacts.train_dataset} "
            f"train split, threshold=quantile(train scores, {quantile})"
        )
    detectors["HydroJEV AE"] = {
        "family": "reconstruction_ae",
        "is_hydrojev": True,
        "stochastic": True,
        "n_runs": len(ae_runs),
        "provenance": ae_provenance,
        "metrics": _aggregate(ae_runs),
        "seed_values": _seed_values(ae_runs),
    }

    # ----- Deterministic baselines (single fit; no sampling variability) ------- #
    for name, fitter in (
        ("PCA residual", fit_pca_residual_detector),
        ("Mahalanobis", fit_mahalanobis_detector),
    ):
        det = fitter(train_matrix, threshold_quantile=quantile)
        runs = [_score(det)]
        detectors[name] = {
            "family": det.family,
            "is_hydrojev": False,
            "stochastic": False,
            "n_runs": 1,
            "provenance": det.provenance,
            "metrics": _aggregate(runs),
            "seed_values": _seed_values(runs),
        }

    # ----- Isolation Forest (stochastic: random feature/split sampling) -------- #
    if_runs: list[dict[str, float]] = []
    if_provenance = ""
    for seed in seeds:
        det = fit_isolation_forest_detector(train_matrix, threshold_quantile=quantile, seed=int(seed))
        if_runs.append(_score(det))
        if_provenance = det.provenance
    detectors["Isolation Forest"] = {
        "family": "isolation_forest",
        "is_hydrojev": False,
        "stochastic": True,
        "n_runs": len(if_runs),
        "provenance": if_provenance,
        "metrics": _aggregate(if_runs),
        "seed_values": _seed_values(if_runs),
    }

    # Scenario counts are a property of the labels, identical for every detector.
    reference = ae_runs[0]
    return {
        "status": "ok",
        "train_dataset": clean.name,
        "evaluate_dataset": labelled.name,
        "protocol": (
            "fit on attack-free dataset03 partition; threshold = quantile(train scores, "
            f"{quantile}); evaluate labelled dataset04; no leakage"
        ),
        "threshold_quantile": quantile,
        "seeds": [int(s) for s in seeds],
        "n_features": len(columns),
        "n_train_rows": int(train_matrix.shape[0]),
        "n_eval_samples": int(len(eval_labels)),
        "attack_scenarios": int(reference["attack_scenarios"]),
        "benign_scenarios": int(reference["benign_scenarios"]),
        "detectors": detectors,
    }


# --------------------------------------------------------------------------- #
# Presentation: comparison table + figure
# --------------------------------------------------------------------------- #

# Metrics shown in the comparison table/figure and whether higher is better.
_REPORT_METRICS = (
    ("roc_auc", "ROC-AUC", True),
    ("pr_auc", "PR-AUC", True),
    ("scenario_recall", "Scenario recall", True),
    ("point_recall", "Point recall", True),
    ("point_false_positive_rate", "Point FPR", False),
)


def benchmark_to_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a benchmark result into one row per (detector, metric)."""

    if result.get("status") != "ok":
        return []
    rows: list[dict[str, Any]] = []
    for name, entry in result["detectors"].items():
        for key, label, higher_better in _REPORT_METRICS:
            summary = entry["metrics"][key]
            rows.append(
                {
                    "detector": name,
                    "is_hydrojev": entry["is_hydrojev"],
                    "family": entry["family"],
                    "metric": label,
                    "metric_key": key,
                    "higher_is_better": higher_better,
                    "mean": summary["mean"],
                    "std": summary["std"],
                    "ci95_low": summary["ci95_low"],
                    "ci95_high": summary["ci95_high"],
                    "n_runs": summary["n"],
                }
            )
    return rows


def benchmark_to_dataframe(result: dict[str, Any]):
    """Return a tidy pandas DataFrame of the comparison (empty if not run)."""

    import pandas as pd

    return pd.DataFrame(benchmark_to_rows(result))


def plot_benchmark(result: dict[str, Any], path: str) -> str | None:
    """Grouped-bar figure comparing detectors across metrics with 95% CIs.

    Returns the written path, or ``None`` if there is nothing to plot.
    """

    if result.get("status") != "ok":
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detector_names = list(result["detectors"].keys())
    metrics = [(label, key, hb) for key, label, hb in _REPORT_METRICS]
    n_metrics = len(metrics)
    n_det = len(detector_names)
    width = 0.8 / n_det

    fig, ax = plt.subplots(figsize=(1.7 * n_metrics + 2.0, 4.2))
    palette = plt.get_cmap("tab10")
    x = np.arange(n_metrics)

    for di, name in enumerate(detector_names):
        entry = result["detectors"][name]
        means, lows, highs = [], [], []
        for _label, key, _hb in metrics:
            s = entry["metrics"][key]
            mean = s["mean"]
            means.append(mean)
            # These metrics are bounded in [0, 1]; clip the drawn whisker to the
            # ceiling so a wide small-sample t-interval does not render above 1.0.
            lows.append(max(0.0, mean - s["ci95_low"]))
            highs.append(min(1.0 - mean, s["ci95_high"] - mean))
        offset = (di - (n_det - 1) / 2) * width
        is_hj = entry["is_hydrojev"]
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=[lows, highs] if entry["stochastic"] else None,
            capsize=3,
            label=name + (" (ours)" if is_hj else ""),
            color=palette(di),
            edgecolor="black" if is_hj else "none",
            linewidth=1.4 if is_hj else 0.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _k, _hb in metrics], rotation=15, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"{result['evaluate_dataset'].replace('BATADAL:dataset04', 'BATADAL development-set')} detection: HydroJEV vs. standard baselines\n"
        f"(fit on the attack-free training set; identical {result['threshold_quantile']:.3f}-quantile threshold; "
        f"error bars 95% CI over {len(result['seeds'])} seeds)",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Reproducible entry point: python -m hydrojev.benchmarks.detection_benchmark
# --------------------------------------------------------------------------- #


def _format_table(result: dict[str, Any]) -> str:
    """A compact fixed-width comparison table for the console/report."""

    header = f"{'detector':<18}" + "".join(f"{label:>16}" for _k, label, _hb in _REPORT_METRICS)
    lines = [header, "-" * len(header)]
    for name, entry in result["detectors"].items():
        tag = " *" if entry["is_hydrojev"] else "  "
        cells = []
        for key, _label, _hb in _REPORT_METRICS:
            s = entry["metrics"][key]
            if entry["stochastic"] and s["n"] > 1:
                cells.append(f"{s['mean']:.3f}+/-{(s['ci95_high'] - s['mean']):.3f}")
            else:
                cells.append(f"{s['mean']:.3f}")
        lines.append(f"{name + tag:<18}" + "".join(f"{c:>16}" for c in cells))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    parser = argparse.ArgumentParser(description="Fair BATADAL detection benchmark: HydroJEV vs baselines")
    parser.add_argument("--quick", action="store_true", help="bounded fast smoke run")
    parser.add_argument("--seeds", type=int, default=5, help="number of refit seeds for stochastic detectors")
    parser.add_argument("--outdir", default="artifacts/detection_benchmark", help="output directory")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    base_seed = int(config.experiments.seed)
    seeds = tuple(range(base_seed, base_seed + max(1, args.seeds)))

    result = run_detection_benchmark(config, mode=mode, seeds=seeds)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "benchmark.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if result.get("status") == "ok":
        benchmark_to_dataframe(result).to_csv(outdir / "benchmark.csv", index=False)
        figure = plot_benchmark(result, str(outdir / "benchmark.png"))
        table = _format_table(result)
        (outdir / "benchmark_table.txt").write_text(table + "\n", encoding="utf-8")
        print(
            f"BATADAL {result['train_dataset']} -> {result['evaluate_dataset']} | "
            f"{result['attack_scenarios']} attack / {result['benign_scenarios']} benign scenarios | "
            f"{result['n_eval_samples']} samples | seeds={list(seeds)}"
        )
        print("(* = HydroJEV; ± = 95% CI half-width over seeds)\n")
        print(table)
        print(f"\nwrote: {outdir/'benchmark.json'}, benchmark.csv, benchmark_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fair adversarial-evasion robustness benchmark: how hard is each detector to fool?

This is the experiment that speaks to the paper's *stealthy cyber-physical attack*
contribution. Clean recall (see :mod:`detection_benchmark`) shows several detectors
already flag BATADAL attacks; the sharper question is whether a bounded, stealthy
adversary can slip a *detected* attack back under the alarm threshold.

Protocol (identical for every detector):

1. Fit the detector on attack-free ``dataset03`` (same as the clean benchmark).
2. Take the attack samples on ``dataset04`` that this detector *actually detects*
   (score >= its own threshold) — the adversary only needs to hide true alarms.
3. Attack each with the shared, detector-agnostic coordinate-descent evasion
   (:func:`evade_detector`): at most ``max_changed`` compromised sensors, each held
   within the normal-operation range, discrete pump/valve status immutable, a fixed
   query budget.
4. Report the evasion success rate with a Wilson 95% interval over the targets,
   plus the median attacker effort (compromised sensors, in-range L2, queries).

A detector that is *easy* to evade offers little protection against the paper's
threat model even if its clean recall is high — which is exactly the gap the
HydroJEV architecture is designed to close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig
from hydrojev.datasets.wdn_loader import (
    DatasetUnavailableError,
    classify_batadal_columns,
    load_batadal,
)
from hydrojev.benchmarks.detection_baselines import (
    fit_isolation_forest_detector,
    fit_mahalanobis_detector,
    fit_pca_residual_detector,
)
from hydrojev.benchmarks.evasion_attack import EvasionResult, evade_detector


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""

    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _median(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def _select_targets(scores: np.ndarray, labels: np.ndarray, threshold: float, max_targets: int) -> np.ndarray:
    """Detected attack indices (label==1 and score>=threshold), evenly subsampled."""

    detected = np.flatnonzero((labels == 1) & (scores >= threshold))
    if detected.size <= max_targets:
        return detected
    picks = np.unique(np.linspace(0, detected.size - 1, max_targets).round().astype(int))
    return detected[picks]


def _summarize_detector(
    name: str,
    detector: Any,
    *,
    is_hydrojev: bool,
    stochastic: bool,
    provenance: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_low: np.ndarray,
    feature_high: np.ndarray,
    modifiable_mask: np.ndarray,
    max_changed: int,
    candidate_values: int,
    budget: int,
    max_targets: int,
) -> dict[str, Any]:
    scores = np.asarray(detector.score_samples(features), dtype=float)
    threshold = float(detector.threshold)
    targets = _select_targets(scores, labels, threshold, max_targets)

    results: list[EvasionResult] = []
    for idx in targets:
        results.append(
            evade_detector(
                detector,
                features[idx],
                feature_low=feature_low,
                feature_high=feature_high,
                modifiable_mask=modifiable_mask,
                max_changed=max_changed,
                candidate_values=candidate_values,
                budget=budget,
            )
        )

    n = len(results)
    evaded = [r for r in results if r.evaded]
    rate = len(evaded) / n if n else float("nan")
    lo, hi = wilson_interval(len(evaded), n)
    # Fraction by which the alarm score was suppressed, averaged over all targets.
    suppression = [
        (r.original_score - r.adversarial_score) / r.original_score
        for r in results
        if r.original_score > 0
    ]
    return {
        "family": getattr(detector, "family", name),
        "is_hydrojev": is_hydrojev,
        "stochastic": stochastic,
        "provenance": provenance,
        "n_detected_attack_points": int(np.sum((labels == 1) & (scores >= threshold))),
        "n_targets": n,
        "n_evaded": len(evaded),
        "evasion_success_rate": rate,
        "evasion_ci95_low": lo,
        "evasion_ci95_high": hi,
        "median_score_suppression": _median(suppression),
        "median_sensors_changed_evaded": _median([r.n_changed for r in evaded]),
        "median_scaled_l2_evaded": _median([r.scaled_l2 for r in evaded]),
        "median_queries": _median([r.query_count for r in results]),
    }


@dataclass
class EvasionFixture:
    """Everything the fair evasion protocol shares: fitted detectors + stealth box.

    Built once and reused by the per-detector evasion benchmark *and* the
    cross-detector transfer benchmark, so both attack the identical detectors,
    stealth box, and immutability mask (the fairness invariant).
    """

    train_dataset: str
    evaluate_dataset: str
    columns: list[str]
    features: np.ndarray
    labels: np.ndarray
    feature_low: np.ndarray
    feature_high: np.ndarray
    modifiable_mask: np.ndarray
    max_changed: int
    candidate_values: int
    quantile: float
    seed: int
    # (name, detector, is_hydrojev, stochastic, provenance)
    detectors: list[tuple[str, Any, bool, bool, str]]


def build_evasion_fixture(
    config: HydroJEVConfig, *, mode: Any, seed: int | None = None
) -> EvasionFixture:
    """Load BATADAL, fit all detectors, and assemble the shared stealth box.

    Raises :class:`DatasetUnavailableError` (with a human reason) when the real
    BATADAL assets, an attack-label column, or the shared feature columns are
    missing, so callers can degrade to ``status="not_run"``.
    """

    from hydrojev.run_experiments import build_reconstruction_detector

    clean = load_batadal("dataset03", config=config)
    labelled = load_batadal("dataset04", config=config)
    if labelled.label_column is None:
        raise DatasetUnavailableError(f"{labelled.name} has no attack-label column")

    columns = list(clean.feature_columns)
    missing = [c for c in columns if c not in labelled.frame.columns]
    if missing:
        raise DatasetUnavailableError(
            f"feature columns absent in {labelled.name}: {missing[:3]}"
        )

    quantile = float(config.concealment.threshold_quantile)
    max_changed = int(config.concealment.max_modified_features)
    candidate_values = int(config.concealment.candidate_values)
    rows = mode.batadal_train_rows

    train_matrix = clean.train_frame[columns].to_numpy(dtype=float)
    if rows is not None:
        train_matrix = train_matrix[:rows]
    features = labelled.frame[columns].to_numpy(dtype=float)
    labels = labelled.frame[labelled.label_column].to_numpy().astype(int)

    # Stealth box = per-sensor range of *normal* operation; a spoof outside it is
    # trivially caught by a range check, so the adversary is confined to it.
    feature_low = train_matrix.min(axis=0)
    feature_high = train_matrix.max(axis=0)

    # Discrete pump/valve status channels are not free-form spoofable here.
    groups = classify_batadal_columns(columns)
    immutable = set(groups.get("pump_status", ())) | set(groups.get("valve_status", ()))
    modifiable_mask = np.array([c not in immutable for c in columns], dtype=bool)

    seed = int(config.experiments.seed if seed is None else seed)
    import dataclasses

    cfg_seed = dataclasses.replace(
        config, experiments=dataclasses.replace(config.experiments, seed=seed)
    )
    ae = build_reconstruction_detector(cfg_seed, mode=mode).detector
    pca = fit_pca_residual_detector(train_matrix, threshold_quantile=quantile)
    maha = fit_mahalanobis_detector(train_matrix, threshold_quantile=quantile)
    iso = fit_isolation_forest_detector(train_matrix, threshold_quantile=quantile, seed=seed)

    return EvasionFixture(
        train_dataset=clean.name,
        evaluate_dataset=labelled.name,
        columns=columns,
        features=features,
        labels=labels,
        feature_low=feature_low,
        feature_high=feature_high,
        modifiable_mask=modifiable_mask,
        max_changed=max_changed,
        candidate_values=candidate_values,
        quantile=quantile,
        seed=seed,
        detectors=[
            ("HydroJEV AE", ae, True, True, "reconstruction autoencoder"),
            ("PCA residual", pca, False, False, pca.provenance),
            ("Mahalanobis", maha, False, False, maha.provenance),
            ("Isolation Forest", iso, False, True, iso.provenance),
        ],
    )


def run_evasion_benchmark(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    max_targets: int = 80,
    budget: int = 4000,
) -> dict[str, Any]:
    """Run the fair evasion-robustness benchmark; return a structured comparison."""

    try:
        fx = build_evasion_fixture(config, mode=mode, seed=seed)
    except DatasetUnavailableError as exc:
        return {"status": "not_run", "reason": str(exc)}

    common = dict(
        features=fx.features, labels=fx.labels, feature_low=fx.feature_low,
        feature_high=fx.feature_high, modifiable_mask=fx.modifiable_mask,
        max_changed=fx.max_changed, candidate_values=fx.candidate_values,
        budget=budget, max_targets=max_targets,
    )
    detectors = {
        name: _summarize_detector(name, det, is_hydrojev=hj, stochastic=st, provenance=prov, **common)
        for name, det, hj, st, prov in fx.detectors
    }

    max_changed = fx.max_changed
    quantile = fx.quantile
    modifiable_mask = fx.modifiable_mask
    columns = fx.columns
    seed = fx.seed
    clean_name, labelled_name = fx.train_dataset, fx.evaluate_dataset
    return {
        "status": "ok",
        "train_dataset": clean_name,
        "evaluate_dataset": labelled_name,
        "protocol": (
            f"fit on attack-free dataset03; attack each detector's detected dataset04 alarms "
            f"with shared coordinate-descent evasion (<= {max_changed} sensors, in normal range, "
            f"budget {budget} queries); Wilson 95% CI over targets"
        ),
        "threshold_quantile": quantile,
        "max_changed_sensors": max_changed,
        "query_budget": budget,
        "seed": seed,
        "n_modifiable_sensors": int(modifiable_mask.sum()),
        "n_features": len(columns),
        "detectors": detectors,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _format_table(result: dict[str, Any]) -> str:
    header = f"{'detector':<18}{'evasion rate':>26}{'targets':>10}{'med sensors':>13}{'med L2':>9}{'med supp':>10}"
    lines = [header, "-" * len(header)]
    for name, e in result["detectors"].items():
        tag = " *" if e["is_hydrojev"] else "  "
        rate = f"{e['evasion_success_rate']:.3f} [{e['evasion_ci95_low']:.2f},{e['evasion_ci95_high']:.2f}]"
        lines.append(
            f"{name + tag:<18}{rate:>26}{e['n_targets']:>10}"
            f"{e['median_sensors_changed_evaded']:>13.1f}{e['median_scaled_l2_evaded']:>9.3f}"
            f"{e['median_score_suppression']:>10.3f}"
        )
    return "\n".join(lines)


def plot_evasion(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["detectors"].keys())
    rates = [result["detectors"][n]["evasion_success_rate"] for n in names]
    # Wilson bounds are centred on the score-adjusted centre, not the raw
    # proportion, so an arm can be a hair negative at extreme rates; clamp to 0.
    los = [max(0.0, rates[i] - result["detectors"][n]["evasion_ci95_low"]) for i, n in enumerate(names)]
    his = [max(0.0, result["detectors"][n]["evasion_ci95_high"] - rates[i]) for i, n in enumerate(names)]
    palette = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(
        range(len(names)), rates, yerr=[los, his], capsize=4,
        color=[palette(i) for i in range(len(names))],
        edgecolor=["black" if result["detectors"][n]["is_hydrojev"] else "none" for n in names],
        linewidth=[1.6 if result["detectors"][n]["is_hydrojev"] else 0.0 for n in names],
    )
    for i, n in enumerate(names):
        e = result["detectors"][n]
        ax.text(i, min(1.0, rates[i] + his[i]) + 0.02, f"{rates[i]:.0%}\n(n={e['n_targets']})",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n + (" (ours)" if result["detectors"][n]["is_hydrojev"] else "") for n in names],
                       rotation=15, ha="right")
    ax.set_ylabel("stealthy-evasion success rate (lower = more robust)")
    ax.set_ylim(0.0, 1.15)
    ax.set_title(
        f"Adversarial evasion on {result['evaluate_dataset'].replace('BATADAL:dataset04', 'the BATADAL development set')}: fraction of detected attacks\n"
        f"hidden with <= {result['max_changed_sensors']} in-range sensor spoofs "
        f"(shared attack; Wilson 95% CI)",
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

    parser = argparse.ArgumentParser(description="Fair adversarial-evasion robustness benchmark")
    parser.add_argument("--quick", action="store_true", help="bounded fast smoke run")
    parser.add_argument("--max-targets", type=int, default=80)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--outdir", default="artifacts/evasion_benchmark")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    result = run_evasion_benchmark(config, mode=mode, max_targets=args.max_targets, budget=args.budget)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "evasion.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if result.get("status") == "ok":
        table = _format_table(result)
        (outdir / "evasion_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_evasion(result, str(outdir / "evasion.png"))
        print(result["protocol"])
        print("(* = HydroJEV; rate lower = more robust)\n")
        print(table)
        print(f"\nwrote: {outdir/'evasion.json'}, evasion_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

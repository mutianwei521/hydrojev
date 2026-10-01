"""Cross-detector transferability and ensemble-evasion: HydroJEV's defence-in-depth.

The single-detector evasion benchmark (:mod:`evasion_benchmark`) shows the learned
autoencoder is the hardest *individual* detector to fool, but even it loses a
quarter of its alarms. HydroJEV never trusts one detector: its decision requires
corroboration across an *orthogonal* residual family (a learned reconstruction
manifold, a covariance/Mahalanobis view, a subspace/PCA residual). The
architectural claim this module tests is that these views fail *independently*, so
an attack tuned to slip past one is unlikely to slip past all at once.

Two experiments, both under the identical stealth box, immutability mask, and
query budget as the single-detector benchmark (fairness invariant, shared via
:func:`evasion_benchmark.build_evasion_fixture`):

1. **Transfer matrix** ``T[s][d]`` — of the samples a stealthy attacker crafts to
   *successfully* evade detector ``s`` (attacking ``s`` alone), the fraction on
   which detector ``d`` *still* raises an alarm. A high off-diagonal means evasion
   does not transfer: the orthogonal detector catches what the attacked one misses.

2. **Ensemble joint-evasion** — on the attack points the primary autoencoder
   detects, how often can a *single* attacker drive the AE *and* the orthogonal
   detectors below threshold simultaneously (:func:`evade_detectors_jointly`)?
   This is the honest worst case (the attacker knows the whole ensemble). The
   hypothesis: the joint-evasion rate is far below even the AE-alone rate, so
   requiring orthogonal agreement is markedly more robust than any single detector.

If BATADAL (or the fitted assets) are unavailable the runner returns
``status="not_run"`` rather than fabricating numbers.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig
from hydrojev.datasets.wdn_loader import DatasetUnavailableError
from hydrojev.benchmarks.evasion_attack import evade_detector, evade_detectors_jointly
from hydrojev.benchmarks.evasion_benchmark import (
    _select_targets,
    build_evasion_fixture,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# Pure, unit-testable aggregation helpers
# --------------------------------------------------------------------------- #


def score_scale(detector: Any, features: np.ndarray) -> float:
    """Robust positive scale for a detector's score (its eval-set std, or 1.0).

    Used to make per-detector margins comparable in the joint attack so no single
    high-variance detector dominates the shared objective.
    """

    s = np.asarray(detector.score_samples(features), dtype=float)
    sd = float(np.std(s))
    return sd if sd > 0 and np.isfinite(sd) else 1.0


def still_alerting_fraction(detector: Any, samples: np.ndarray) -> float:
    """Fraction of ``samples`` on which ``detector`` scores at or above threshold."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return float("nan")
    scores = np.asarray(detector.score_samples(samples), dtype=float)
    return float(np.mean(scores >= float(detector.threshold)))


def ensemble_all_silent_fraction(detectors: Sequence[Any], samples: np.ndarray) -> float:
    """Fraction of ``samples`` on which *every* detector is below its threshold.

    This is the naive transfer of an attack to an OR-ensemble (which alarms if any
    member fires): the attack only defeats the ensemble where all members are
    simultaneously silent.
    """

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        return float("nan")
    silent = np.ones(samples.shape[0], dtype=bool)
    for det in detectors:
        scores = np.asarray(det.score_samples(samples), dtype=float)
        silent &= scores < float(det.threshold)
    return float(np.mean(silent))


def evaluate_joint_evasion(
    members: Sequence[Any],
    target_features: np.ndarray,
    *,
    scales: Sequence[float],
    names: Sequence[str],
    feature_low: np.ndarray,
    feature_high: np.ndarray,
    modifiable_mask: np.ndarray,
    max_changed: int,
    candidate_values: int,
    budget: int,
) -> dict[str, Any]:
    """Run the joint (ensemble) attack on each target row; summarise success.

    ``evaded_all`` counts only when *every* member is driven below threshold under
    the shared constraints. Returns the rate with a Wilson 95% interval and the
    median attacker effort among successful joint evasions.
    """

    target_features = np.asarray(target_features, dtype=float)
    evaded = 0
    n = int(target_features.shape[0]) if target_features.ndim == 2 else 0
    changed_when_evaded: list[int] = []
    l2_when_evaded: list[float] = []
    for i in range(n):
        res = evade_detectors_jointly(
            members,
            target_features[i],
            scales=scales,
            names=names,
            feature_low=feature_low,
            feature_high=feature_high,
            modifiable_mask=modifiable_mask,
            max_changed=max_changed,
            candidate_values=candidate_values,
            budget=budget,
        )
        if res.evaded_all:
            evaded += 1
            changed_when_evaded.append(res.n_changed)
            l2_when_evaded.append(res.scaled_l2)
    rate = evaded / n if n else float("nan")
    lo, hi = wilson_interval(evaded, n)
    return {
        "n_targets": n,
        "n_evaded": evaded,
        "joint_evasion_rate": rate,
        "evasion_ci95_low": lo,
        "evasion_ci95_high": hi,
        "median_sensors_changed_evaded": (
            float(np.median(changed_when_evaded)) if changed_when_evaded else float("nan")
        ),
        "median_scaled_l2_evaded": (
            float(np.median(l2_when_evaded)) if l2_when_evaded else float("nan")
        ),
    }


# --------------------------------------------------------------------------- #
# Real-data runner
# --------------------------------------------------------------------------- #

# Ensembles analysed, most-orthogonal-pair first. Every ensemble includes the
# autoencoder (HydroJEV's primary detector); the others add orthogonal views.
_ENSEMBLE_SPECS: list[tuple[str, tuple[str, ...]]] = [
    ("AE + Mahalanobis", ("HydroJEV AE", "Mahalanobis")),
    ("AE + Mahalanobis + PCA", ("HydroJEV AE", "Mahalanobis", "PCA residual")),
    ("all four detectors", ("HydroJEV AE", "Mahalanobis", "PCA residual", "Isolation Forest")),
]


def run_transfer_benchmark(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    max_targets: int = 60,
    budget: int = 4000,
) -> dict[str, Any]:
    """Cross-detector transfer matrix + ensemble joint-evasion on real BATADAL."""

    try:
        fx = build_evasion_fixture(config, mode=mode, seed=seed)
    except DatasetUnavailableError as exc:
        return {"status": "not_run", "reason": str(exc)}

    det_by_name = {name: det for name, det, *_ in fx.detectors}
    order = [name for name, *_ in fx.detectors]
    scales = {name: score_scale(det, fx.features) for name, det in det_by_name.items()}
    box = dict(
        feature_low=fx.feature_low,
        feature_high=fx.feature_high,
        modifiable_mask=fx.modifiable_mask,
        max_changed=fx.max_changed,
        candidate_values=fx.candidate_values,
        budget=budget,
    )

    # ---- (1) transfer matrix: attack each detector alone on its own alarms ---- #
    transfer: dict[str, dict[str, Any]] = {}
    for src in order:
        det = det_by_name[src]
        scores = np.asarray(det.score_samples(fx.features), dtype=float)
        targets = _select_targets(scores, fx.labels, float(det.threshold), max_targets)
        adv: list[np.ndarray] = []
        for idx in targets:
            r = evade_detector(det, fx.features[idx], **box)
            if r.evaded:
                adv.append(np.asarray(r.adversarial_sample, dtype=float))
        adv_matrix = np.vstack(adv) if adv else np.empty((0, len(fx.columns)))
        row = {
            "n_detected": int(np.sum((fx.labels == 1) & (scores >= float(det.threshold)))),
            "n_targets": int(targets.size),
            "n_evaded_single": int(adv_matrix.shape[0]),
            # of THIS detector's successful adversarial samples, how often each
            # detector still fires (diagonal is ~0 by construction).
            "still_alerts": {
                dst: still_alerting_fraction(det_by_name[dst], adv_matrix) for dst in order
            },
        }
        transfer[src] = row

    # ---- (2) ensemble joint-evasion on the AE's detected attack points ------- #
    ae = det_by_name["HydroJEV AE"]
    ae_scores = np.asarray(ae.score_samples(fx.features), dtype=float)
    ae_targets = _select_targets(ae_scores, fx.labels, float(ae.threshold), max_targets)
    ae_target_features = fx.features[ae_targets]

    # AE-alone reference on the same target set.
    ae_adv: list[np.ndarray] = []
    for idx in ae_targets:
        r = evade_detector(ae, fx.features[idx], **box)
        if r.evaded:
            ae_adv.append(np.asarray(r.adversarial_sample, dtype=float))
    ae_adv_matrix = np.vstack(ae_adv) if ae_adv else np.empty((0, len(fx.columns)))
    ae_single_rate = ae_adv_matrix.shape[0] / ae_targets.size if ae_targets.size else float("nan")
    ae_lo, ae_hi = wilson_interval(ae_adv_matrix.shape[0], int(ae_targets.size))

    ensembles: dict[str, Any] = {}
    for label, member_names in _ENSEMBLE_SPECS:
        members = [det_by_name[m] for m in member_names]
        member_scales = [scales[m] for m in member_names]
        # Naive transfer: of AE's successful evasions, how often is the whole
        # ensemble already silent (no active attack on the other members)?
        orth = [det_by_name[m] for m in member_names if m != "HydroJEV AE"]
        naive_ensemble_evasion = ensemble_all_silent_fraction(members, ae_adv_matrix)
        summary = evaluate_joint_evasion(
            members,
            ae_target_features,
            scales=member_scales,
            names=list(member_names),
            feature_low=fx.feature_low,
            feature_high=fx.feature_high,
            modifiable_mask=fx.modifiable_mask,
            max_changed=fx.max_changed,
            candidate_values=fx.candidate_values,
            budget=budget,
        )
        summary["members"] = list(member_names)
        summary["naive_ensemble_evasion_of_ae_samples"] = naive_ensemble_evasion
        summary["orthogonal_catch_rate_of_ae_samples"] = (
            (1.0 - naive_ensemble_evasion) if np.isfinite(naive_ensemble_evasion) else float("nan")
        )
        ensembles[label] = summary

    return {
        "status": "ok",
        "train_dataset": fx.train_dataset,
        "evaluate_dataset": fx.evaluate_dataset,
        "protocol": (
            "fit all detectors on attack-free dataset03; (1) transfer matrix = attack each "
            "detector alone on its own detected dataset04 alarms and measure how often every "
            "other detector still fires on those adversarial samples; (2) ensemble joint-evasion "
            "= on the autoencoder's detected alarms, drive all ensemble members below threshold "
            f"at once (<= {fx.max_changed} in-range sensor spoofs, budget {budget}); Wilson 95% CI"
        ),
        "seed": fx.seed,
        "threshold_quantile": fx.quantile,
        "max_changed_sensors": fx.max_changed,
        "query_budget": budget,
        "n_features": len(fx.columns),
        "n_modifiable_sensors": int(fx.modifiable_mask.sum()),
        "detector_order": order,
        "transfer_matrix": transfer,
        "ae_single_evasion_rate": ae_single_rate,
        "ae_single_ci95_low": ae_lo,
        "ae_single_ci95_high": ae_hi,
        "ae_detected_targets": int(ae_targets.size),
        "ensembles": ensembles,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_transfer_table(result: dict[str, Any]) -> str:
    order = result["detector_order"]
    corner = "attacked \\ still fires"
    head = f"{corner:<24}" + "".join(f"{n[:12]:>14}" for n in order)
    lines = [
        "Transfer matrix: of samples that EVADE the row detector, fraction on which the",
        "column detector STILL fires (higher off-diagonal = evasion does not transfer).",
        "",
        head,
        "-" * len(head),
    ]
    for src in order:
        row = result["transfer_matrix"][src]
        cells = []
        for dst in order:
            v = row["still_alerts"][dst]
            cells.append("   n/a" if not np.isfinite(v) else f"{v:>14.2f}")
        lines.append(f"{src[:24]:<24}" + "".join(cells) + f"   (n={row['n_evaded_single']})")
    lines.append("")
    lines.append(
        f"AE-alone evasion rate on its {result['ae_detected_targets']} detected alarms: "
        f"{result['ae_single_evasion_rate']:.3f} "
        f"[{result['ae_single_ci95_low']:.2f},{result['ae_single_ci95_high']:.2f}]"
    )
    lines.append("")
    lines.append(f"{'ensemble':<26}{'joint-evasion [95% CI]':>28}{'naive':>9}{'targets':>9}")
    lines.append("-" * 72)
    for label, e in result["ensembles"].items():
        rate = f"{e['joint_evasion_rate']:.3f} [{e['evasion_ci95_low']:.2f},{e['evasion_ci95_high']:.2f}]"
        naive = e["naive_ensemble_evasion_of_ae_samples"]
        naive_s = "n/a" if not np.isfinite(naive) else f"{naive:.2f}"
        lines.append(f"{label:<26}{rate:>28}{naive_s:>9}{e['n_targets']:>9}")
    return "\n".join(lines)


def plot_transfer(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = result["detector_order"]
    mat = np.array(
        [[result["transfer_matrix"][s]["still_alerts"][d] for d in order] for s in order],
        dtype=float,
    )

    fig, (axm, axb) = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.05, 1]})

    im = axm.imshow(np.nan_to_num(mat, nan=0.0), cmap="viridis", vmin=0.0, vmax=1.0)
    axm.set_xticks(range(len(order)))
    axm.set_xticklabels([n.replace(" ", "\n") for n in order], fontsize=8)
    axm.set_yticks(range(len(order)))
    axm.set_yticklabels(order, fontsize=8)
    axm.set_xlabel("detector that still fires")
    axm.set_ylabel("detector the attack evaded")
    for i in range(len(order)):
        for j in range(len(order)):
            v = mat[i, j]
            if np.isfinite(v):
                axm.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v < 0.6 else "black", fontsize=8)
    axm.set_title("Evasion does not transfer:\nfraction of adversarial samples the other detector still catches", fontsize=9)
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)

    labels = list(result["ensembles"].keys())
    joint = [result["ensembles"][l]["joint_evasion_rate"] for l in labels]
    los = [max(0.0, joint[i] - result["ensembles"][l]["evasion_ci95_low"]) for i, l in enumerate(labels)]
    his = [max(0.0, result["ensembles"][l]["evasion_ci95_high"] - joint[i]) for i, l in enumerate(labels)]
    x = np.arange(len(labels) + 1)
    heights = [result["ae_single_evasion_rate"]] + joint
    lo_all = [max(0.0, result["ae_single_evasion_rate"] - result["ae_single_ci95_low"])] + los
    hi_all = [max(0.0, result["ae_single_ci95_high"] - result["ae_single_evasion_rate"])] + his
    colours = ["#9ecae1"] + ["#e6550d"] * len(labels)
    axb.bar(x, heights, yerr=[lo_all, hi_all], capsize=4, color=colours)
    axb.set_xticks(x)
    axb.set_xticklabels(["AE alone"] + labels, rotation=18, ha="right", fontsize=8)
    axb.set_ylabel("stealthy evasion success (lower = more robust)")
    axb.set_ylim(0.0, max(0.4, max(heights) * 1.3))
    axb.set_title("Requiring orthogonal agreement collapses the\nevasion rate below the best single detector", fontsize=9)
    for i, h in enumerate(heights):
        axb.text(x[i], h + hi_all[i] + 0.005, f"{h:.0%}", ha="center", va="bottom", fontsize=8)
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

    parser = argparse.ArgumentParser(description="Cross-detector transfer + ensemble evasion")
    parser.add_argument("--quick", action="store_true", help="bounded fast smoke run")
    parser.add_argument("--max-targets", type=int, default=60)
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--outdir", default="artifacts/transfer_benchmark")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    result = run_transfer_benchmark(config, mode=mode, max_targets=args.max_targets, budget=args.budget)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "transfer.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if result.get("status") == "ok":
        table = format_transfer_table(result)
        (outdir / "transfer_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_transfer(result, str(outdir / "transfer.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'transfer.json'}, transfer_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

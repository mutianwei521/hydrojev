"""§6.1(a) detection quality — confidence intervals on ROC-AUC / PR-AUC.

§6.1(a) reports point ROC-AUC / PR-AUC for the three named families and the three
instantaneous references, and claims the named families "match or exceed" the
references. On `dataset04` those metrics rest on only *seven* contiguous attack
scenarios, so a reviewer will (rightly) ask for uncertainty on them and on the
comparison. This module supplies it with the *same* moving-block bootstrap used for
the §6.1(b) orthogonality matrix — resampling contiguous blocks of the aligned
`dataset04` stream (preserving temporal autocorrelation) and recomputing each
detector's ROC-AUC / PR-AUC on the resampled (scores, labels) pair.

Two honest products:

* a 95% percentile interval on every detector's ROC-AUC and PR-AUC (necessarily wide
  — seven attack scenarios is not many, and the interval says so rather than hiding
  it); and
* a **paired** bootstrap interval on ROC-AUC(named family) − ROC-AUC(strongest
  reference). The claim we can defend is "on par": if that interval contains zero the
  family is statistically indistinguishable from the best reference, which is exactly
  the "match or exceed / credible, not vestigial" wording — not an overclaim of
  superiority.

Leakage-safe (scores come from the §6.1 builder: families/surrogate/references/
thresholds fit on dataset03, scored causally on dataset04). Returns
``status="not_run"`` (never fabricated numbers) when BATADAL/torch are unavailable.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks.orthogonality_significance import moving_block_indices


def _auc_pair(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """(ROC-AUC, PR-AUC) via the same sklearn estimators the detection benchmark uses;
    NaN when a resample is single-class or has a non-finite score."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    n_pos = int(np.sum(labels == 1))
    if not (0 < n_pos < labels.shape[0]) or not np.isfinite(scores).all():
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def bootstrap_detection_cis(
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    block_len: int,
    n_boot: int,
    seed: int,
    contrast_reference: str,
) -> dict[str, Any]:
    """Moving-block bootstrap CIs for each detector's ROC-AUC / PR-AUC, plus paired
    ROC-AUC differences of every other detector against ``contrast_reference``.

    A single resampled index is applied to the labels and to *every* detector's scores
    each replicate, so the paired differences are computed on identical samples (correct
    interval on the difference). Replicates that turn degenerate (single-class draw) are
    dropped from every series' percentile.
    """

    labels = np.asarray(labels).astype(int)
    names = list(scores.keys())
    if contrast_reference not in scores:
        raise ValueError(f"contrast_reference {contrast_reference!r} not among {names}")
    cols = {name: np.asarray(scores[name], dtype=float) for name in names}
    n = labels.shape[0]

    point = {name: _auc_pair(labels, cols[name]) for name in names}

    rng = np.random.default_rng(seed)
    roc_reps = {name: np.full(n_boot, np.nan) for name in names}
    pr_reps = {name: np.full(n_boot, np.nan) for name in names}
    diff_reps = {name: np.full(n_boot, np.nan) for name in names if name != contrast_reference}
    for b in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        lab = labels[idx]
        roc_b: dict[str, float] = {}
        for name in names:
            roc, pr = _auc_pair(lab, cols[name][idx])
            roc_reps[name][b] = roc
            pr_reps[name][b] = pr
            roc_b[name] = roc
        ref = roc_b[contrast_reference]
        for name in diff_reps:
            diff_reps[name][b] = roc_b[name] - ref

    def ci(a: np.ndarray) -> tuple[list[float], int]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return [float("nan"), float("nan")], 0
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], int(a.size)

    detectors: dict[str, Any] = {}
    for name in names:
        roc_ci, n_valid = ci(roc_reps[name])
        pr_ci, _ = ci(pr_reps[name])
        detectors[name] = {
            "roc_auc": {"point": point[name][0], "ci95": roc_ci},
            "pr_auc": {"point": point[name][1], "ci95": pr_ci},
            "n_boot_valid": n_valid,
        }

    contrast: dict[str, Any] = {}
    for name, reps in diff_reps.items():
        d_ci, _ = ci(reps)
        finite = reps[np.isfinite(reps)]
        frac_ge0 = float(np.mean(finite >= 0.0)) if finite.size else float("nan")
        contrast[name] = {
            "roc_auc_diff_point": point[name][0] - point[contrast_reference][0],
            "ci95": d_ci,
            "frac_ge_zero": frac_ge0,
            # "on par" when the paired 95% interval contains zero.
            "indistinguishable": bool(d_ci[0] <= 0.0 <= d_ci[1]) if np.isfinite(d_ci[0]) else False,
        }

    return {
        "block_len": int(block_len),
        "contrast_reference": contrast_reference,
        "detectors": detectors,
        "roc_auc_vs_reference": contrast,
    }


def run_detection_significance(
    config: Any,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 80,
    n_boot: int = 2000,
    block_lengths: Sequence[int] = (25, 50, 100),
    primary_block_len: int = 50,
    contrast_reference: str = "Mahalanobis",
) -> dict[str, Any]:
    """Bootstrap CIs for the §6.1(a) detection metrics on the shared dataset04 series."""

    try:
        import torch  # noqa: F401
    except Exception as exc:
        return {"status": "not_run", "reason": f"torch unavailable: {exc}"}

    from hydrojev.benchmarks.named_family_benchmark import build_named_family_scores

    built = build_named_family_scores(config, mode=mode, seed=seed, max_epochs=max_epochs)
    if isinstance(built, dict):  # not_run
        return built
    if contrast_reference not in built.scores:
        return {"status": "not_run", "reason": f"reference {contrast_reference} absent"}

    labels = np.asarray(built.labels).astype(int)
    n = labels.shape[0]
    if n < max(block_lengths) + 5 or not (0 < int(labels.sum()) < n):
        return {"status": "not_run", "reason": "insufficient labelled samples for the bootstrap"}

    base_seed = int(config.experiments.seed if seed is None else seed)
    primary = bootstrap_detection_cis(
        dict(built.scores), labels, block_len=primary_block_len,
        n_boot=n_boot, seed=base_seed + 202, contrast_reference=contrast_reference,
    )
    by_block = [
        bootstrap_detection_cis(
            dict(built.scores), labels, block_len=L,
            n_boot=n_boot, seed=base_seed + 2000 + L, contrast_reference=contrast_reference,
        )
        for L in block_lengths
    ]

    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "protocol": (
            "moving-block bootstrap (preserves temporal autocorrelation) of ROC-AUC/PR-AUC "
            "on the shared dataset04 scores; paired ROC-AUC differences vs the strongest "
            "reference test whether each named family is on par (interval contains 0) rather "
            "than assuming superiority"
        ),
        "seed": base_seed,
        "n_scored_samples": int(n),
        "n_attack_scenarios": int(_count_runs(labels)),
        "n_boot": int(n_boot),
        "primary_block_len": int(primary_block_len),
        "families": list(built.families),
        "references": list(built.references),
        "contrast_reference": contrast_reference,
        "primary": primary,
        "by_block_len": by_block,
    }


def _count_runs(labels: np.ndarray) -> int:
    from hydrojev.benchmarks.detection_baselines import contiguous_runs

    return len(contiguous_runs(np.asarray(labels).astype(int), 1))


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _auc_cell(point: float, ci: Sequence[float]) -> str:
    lo, hi = ci
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def format_detection_significance_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    primary = result["primary"]
    fam = set(result["families"])
    lines = [
        f"Detection-quality CIs on {result['evaluate_dataset']} "
        f"({result['n_scored_samples']} samples, {result['n_attack_scenarios']} attack "
        f"scenarios; fit on {result['train_dataset']}).",
        f"Moving-block bootstrap, {result['n_boot']} replicates, "
        f"primary block length {result['primary_block_len']}. * = architecture's family.",
        "",
        f"{'detector':<18}{'ROC-AUC [95% CI]':>26}{'PR-AUC [95% CI]':>26}",
        "-" * 70,
    ]
    for name, d in primary["detectors"].items():
        tag = " *" if name in fam else ""
        roc = _auc_cell(d["roc_auc"]["point"], d["roc_auc"]["ci95"])
        pr = _auc_cell(d["pr_auc"]["point"], d["pr_auc"]["ci95"])
        lines.append(f"{name + tag:<18}{roc:>26}{pr:>26}")
    ref = result["contrast_reference"]
    lines += [
        "",
        f"Paired ROC-AUC difference vs the strongest reference ({ref}); an interval",
        "containing 0 means the family is statistically ON PAR (not vestigial), not superior:",
        "",
        f"{'detector':<18}{'dROC-AUC [95% CI]':>28}{'on par?':>10}",
        "-" * 56,
    ]
    for name, c in primary["roc_auc_vs_reference"].items():
        tag = " *" if name in fam else ""
        cell = _auc_cell(c["roc_auc_diff_point"], c["ci95"])
        verdict = "yes" if c["indistinguishable"] else "no"
        lines.append(f"{name + tag:<18}{cell:>28}{verdict:>10}")
    return "\n".join(lines)


def plot_detection_significance(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = result["primary"]["detectors"]
    names = list(detectors.keys())
    fam = set(result["families"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, key, color, label in [
        (-0.18, "roc_auc", "#1b7837", "ROC-AUC"),
        (0.18, "pr_auc", "#762a83", "PR-AUC"),
    ]:
        pts = [detectors[n][key]["point"] for n in names]
        los = [detectors[n][key]["point"] - detectors[n][key]["ci95"][0] for n in names]
        his = [detectors[n][key]["ci95"][1] - detectors[n][key]["point"] for n in names]
        ax.errorbar(x + offset, pts, yerr=[los, his], fmt="o", color=color, capsize=4, label=label)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8, label="chance (ROC)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") + ("\n*" if n in fam else "") for n in names],
                       fontsize=7)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "§6.1(a) detection quality with moving-block bootstrap 95% CIs\n"
        "(wide intervals reflect only seven attack scenarios — reported honestly)",
        fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=8)
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

    parser = argparse.ArgumentParser(description="§6.1(a) detection-quality bootstrap CIs")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--reference", default="Mahalanobis")
    parser.add_argument("--outdir", default="artifacts/detection_significance")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    epochs = 12 if args.quick else args.max_epochs
    n_boot = 300 if args.quick else args.n_boot
    result = run_detection_significance(
        config, mode=mode, seed=args.seed, max_epochs=epochs, n_boot=n_boot,
        contrast_reference=args.reference,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "detection_significance.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_detection_significance_table(result)
        (outdir / "detection_significance_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_detection_significance(result, str(outdir / "detection_significance.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'detection_significance.json'}, table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

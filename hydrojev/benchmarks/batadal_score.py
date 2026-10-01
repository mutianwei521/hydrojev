"""External anchor: HydroJEV under the *official* BATADAL scoring protocol.

Every other experiment scores HydroJEV against detectors we implement ourselves
(Mahalanobis, PCA-residual, AE). A *Water Research* reviewer will ask the obvious
next question — how does it compare to the established public benchmark? The BATADAL
"Battle of the Attack Detection Algorithms" (Taormina et al., 2018, *J. Water Resour.
Plann. Manage.* 144(8)) is exactly that: seven teams scored on a held-out CTOWN test
set under a fixed metric. We compute HydroJEV's score under the identical metric on
the identical held-out set, so the number is directly comparable to the published
leaderboard — not a proxy.

**Official metric (Taormina et al., 2018).** With ``n`` attacks in the test set:

* ``S_TTD`` (time-to-detection): for each attack, the elapsed time to the first alarm
  inside the attack window, normalised by that attack's duration (never detected → 1);
  ``S_TTD = 1 - mean(normalised TTD)`` ∈ [0, 1], higher = faster.
* ``S_CLF`` (classification): balanced accuracy over every test timestamp,
  ``S_CLF = (TPR + TNR) / 2``.
* ``S = gamma * S_TTD + (1 - gamma) * S_CLF``, with the competition's ``gamma = 0.5``.

**Leakage-safe, competition-faithful protocol.** The surrogate is trained only on
attack-free ``dataset03``; each detector's alarm threshold is tuned on the labelled
``dataset04`` (the development set the competitors were given); the score is then
computed on the held-out ``test_dataset`` — the very set the teams were ranked on.
Nothing is fit on the test set. Returns ``status="not_run"`` (never fabricated
numbers) when torch/BATADAL are unavailable.

We report HydroJEV's measured ``S`` and compare it *qualitatively* to the published
range; we never invent per-team decimals (honesty invariant).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks.named_family_benchmark import (
    build_named_family_scores,
    contiguous_runs,
)
from hydrojev.config import HydroJEVConfig

# Documented, citable context — NOT fabricated numbers of our own. The seven ranked
# BATADAL entries' test-set S values are reported by Taormina et al. (2018); the
# leaderboard spans roughly S≈0.5 to the top entry's S≈0.97 (Housh & Ohar). We cite
# the range/source, never a competitor number as if it were ours.
BATADAL_REFERENCE = {
    "citation": (
        "Taormina et al. (2018), 'Battle of the Attack Detection Algorithms: "
        "Disclosing Cyber Attacks on Water Distribution Networks', "
        "J. Water Resour. Plann. Manage. 144(8):04018048"
    ),
    "n_ranked_entries": 7,
    "published_S_range_approx": [0.5, 0.97],
    "note": (
        "published test-set S of the seven ranked entries as reported by the source; "
        "cited qualitatively for context, not reproduced as HydroJEV results"
    ),
}


# --------------------------------------------------------------------------- #
# Pure scoring primitives (unit-testable, no torch / no EPANET)
# --------------------------------------------------------------------------- #


def time_to_detection_score(
    labels: np.ndarray, alarms: np.ndarray
) -> tuple[float, list[dict[str, Any]]]:
    """Official BATADAL ``S_TTD`` from binary labels + binary alarms on the same grid.

    Each contiguous attack contributes ``TTD / duration`` (samples), where ``TTD`` is
    the offset of the first alarm inside the window, or the full duration if the attack
    is never flagged. ``S_TTD = 1 - mean`` over attacks.
    """

    labels = np.asarray(labels).astype(int)
    alarms = np.asarray(alarms).astype(bool)
    runs = contiguous_runs(labels, 1)
    if not runs:
        return float("nan"), []
    per_attack: list[dict[str, Any]] = []
    norm_ttds: list[float] = []
    for start, end in runs:
        duration = end - start
        window = alarms[start:end]
        hit = np.flatnonzero(window)
        ttd = int(hit[0]) if hit.size else int(duration)
        norm = ttd / duration if duration > 0 else 1.0
        norm_ttds.append(norm)
        per_attack.append(
            {"start": int(start), "duration": int(duration), "ttd_samples": ttd,
             "detected": bool(hit.size), "normalized_ttd": float(norm)}
        )
    return float(1.0 - np.mean(norm_ttds)), per_attack


def classification_score(labels: np.ndarray, alarms: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Official BATADAL ``S_CLF`` = balanced accuracy ``(TPR + TNR) / 2``."""

    labels = np.asarray(labels).astype(bool)
    alarms = np.asarray(alarms).astype(bool)
    tp = int(np.sum(alarms & labels))
    fn = int(np.sum(~alarms & labels))
    tn = int(np.sum(~alarms & ~labels))
    fp = int(np.sum(alarms & ~labels))
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    s_clf = float(np.nanmean([tpr, tnr]))
    return s_clf, {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
                   "TPR": float(tpr), "TNR": float(tnr)}


def batadal_score(
    labels: np.ndarray, alarms: np.ndarray, *, gamma: float = 0.5
) -> dict[str, Any]:
    """Combine the two BATADAL sub-scores into ``S = gamma*S_TTD + (1-gamma)*S_CLF``."""

    s_ttd, ttd_detail = time_to_detection_score(labels, alarms)
    s_clf, clf_detail = classification_score(labels, alarms)
    s = float(gamma * s_ttd + (1.0 - gamma) * s_clf)
    return {"S": s, "S_TTD": float(s_ttd), "S_CLF": float(s_clf),
            "gamma": float(gamma), "classification": clf_detail, "per_attack": ttd_detail}


def tune_threshold(
    scores: np.ndarray, labels: np.ndarray, *, gamma: float = 0.5, n_grid: int = 201
) -> tuple[float, float]:
    """Pick the alarm threshold that maximises BATADAL ``S`` on a *development* set.

    Candidates are quantiles of the finite scores (scale-free), plus a never-alarm
    endpoint. Non-finite scores never alarm. Returns ``(best_threshold, best_S)``.
    """

    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return float("inf"), float("nan")
    qs = np.linspace(0.0, 1.0, n_grid)
    candidates = np.unique(np.quantile(finite, qs))
    # add an endpoint strictly above the max so "never alarm" is reachable
    candidates = np.append(candidates, np.nextafter(finite.max(), np.inf))
    best_thr, best_s = float(candidates[-1]), -np.inf
    for thr in candidates:
        alarms = np.isfinite(scores) & (scores >= thr)
        s = batadal_score(labels, alarms, gamma=gamma)["S"]
        if np.isfinite(s) and s > best_s:
            best_s, best_thr = s, float(thr)
    return best_thr, float(best_s)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def _score_detector(
    name: str,
    dev_scores: np.ndarray,
    dev_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    *,
    gamma: float,
) -> dict[str, Any]:
    thr, dev_s = tune_threshold(dev_scores, dev_labels, gamma=gamma)
    test = batadal_score(test_labels, np.isfinite(test_scores) & (test_scores >= thr), gamma=gamma)
    return {"dev_threshold": thr, "dev_S": dev_s, "test": test}


def run_batadal_official_score(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 80,
    gamma: float = 0.5,
) -> dict[str, Any]:
    """Score HydroJEV on the held-out BATADAL test set under the official metric."""

    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        return {"status": "not_run", "reason": f"torch unavailable: {exc}"}

    seed = int(config.experiments.seed if seed is None else seed)
    dev = build_named_family_scores(config, mode=mode, seed=seed, max_epochs=max_epochs,
                                    evaluate_variant="dataset04")
    if isinstance(dev, dict):
        return dev
    test = build_named_family_scores(config, mode=mode, seed=seed, max_epochs=max_epochs,
                                     evaluate_variant="test_dataset")
    if isinstance(test, dict):
        return test

    common = [n for n in test.scores if n in dev.scores]
    detectors: dict[str, Any] = {}
    for name in common:
        detectors[name] = _score_detector(
            name, dev.scores[name], dev.labels, test.scores[name], test.labels, gamma=gamma
        )

    # HydroJEV cognition stack operating as defense-in-depth: alarm if ANY named family
    # fires, each at its own dev-tuned threshold (the union is how the layer runs).
    families = list(test.families)
    fam_thr = {n: detectors[n]["dev_threshold"] for n in families if n in detectors}

    def union_alarms(built: Any, names: Sequence[str]) -> np.ndarray:
        acc = np.zeros(built.labels.shape[0], dtype=bool)
        for n in names:
            s = np.asarray(built.scores[n], dtype=float)
            acc |= np.isfinite(s) & (s >= fam_thr[n])
        return acc

    fam_union_test = batadal_score(test.labels, union_alarms(test, list(fam_thr)), gamma=gamma)
    fam_union_dev = batadal_score(dev.labels, union_alarms(dev, list(fam_thr)), gamma=gamma)

    # rank the individual detectors by test S for readability
    ranking = sorted(common, key=lambda n: detectors[n]["test"]["S"], reverse=True)

    return {
        "status": "ok",
        "protocol": (
            "official BATADAL scoring (Taormina et al. 2018): surrogate trained on "
            "attack-free dataset03, per-detector alarm threshold tuned on labelled "
            "dataset04, S computed on the held-out competition test_dataset; "
            "S = 0.5*S_TTD + 0.5*S_CLF"
        ),
        "seed": seed,
        "gamma": float(gamma),
        "evaluate_dataset": "test_dataset (held-out BATADAL competition set)",
        "tune_dataset": "dataset04 (labelled development set)",
        "n_test_attacks": len(contiguous_runs(test.labels, 1)),
        "n_test_samples": int(test.labels.shape[0]),
        "families": families,
        "references": list(test.references),
        "detectors": detectors,
        "ranking_by_test_S": ranking,
        "hydrojev_family_union": {"test": fam_union_test, "dev": fam_union_dev,
                                  "members": list(fam_thr)},
        "published_reference": BATADAL_REFERENCE,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_batadal_score_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    lines = [
        f"HydroJEV under the official BATADAL metric (Taormina et al. 2018), "
        f"held-out test set ({result['n_test_attacks']} attacks, "
        f"{result['n_test_samples']} samples; seed {result['seed']}).",
        "S = 0.5*S_TTD + 0.5*S_CLF; threshold tuned on labelled dataset04, scored on test_dataset.",
        "",
        f"{'detector':<16}{'S':>8}{'S_TTD':>9}{'S_CLF':>9}{'TPR':>7}{'TNR':>7}{'dev S':>8}",
        "-" * 64,
    ]
    for name in result["ranking_by_test_S"]:
        d = result["detectors"][name]
        t = d["test"]
        c = t["classification"]
        tag = " (family)" if name in result["families"] else ""
        lines.append(
            f"{name:<16}{t['S']:>8.3f}{t['S_TTD']:>9.3f}{t['S_CLF']:>9.3f}"
            f"{c['TPR']:>7.2f}{c['TNR']:>7.2f}{d['dev_S']:>8.3f}{tag}"
        )
    u = result["hydrojev_family_union"]["test"]
    lines += [
        "-" * 64,
        f"{'family union':<16}{u['S']:>8.3f}{u['S_TTD']:>9.3f}{u['S_CLF']:>9.3f}"
        f"{u['classification']['TPR']:>7.2f}{u['classification']['TNR']:>7.2f}"
        f"   (OR of {len(result['hydrojev_family_union']['members'])} named families)",
        "",
        f"Context — {result['published_reference']['citation']}:",
        f"  the {result['published_reference']['n_ranked_entries']} ranked entries' "
        f"published test-set S span approx "
        f"{result['published_reference']['published_S_range_approx'][0]:.2f}"
        f"-{result['published_reference']['published_S_range_approx'][1]:.2f} "
        f"(cited for context; not HydroJEV numbers).",
    ]
    return "\n".join(lines)


def plot_batadal_score(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["ranking_by_test_S"])
    s_ttd = [result["detectors"][n]["test"]["S_TTD"] for n in names]
    s_clf = [result["detectors"][n]["test"]["S_CLF"] for n in names]
    s_tot = [result["detectors"][n]["test"]["S"] for n in names]
    fam = set(result["families"])
    labels = [n + (" *" if n in fam else "") for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 0.2, s_ttd, width=0.38, color="#8c9eff", label="S_TTD (timeliness)")
    ax.bar(x + 0.2, s_clf, width=0.38, color="#69b34c", label="S_CLF (balanced acc.)")
    ax.plot(x, s_tot, "ko-", label="S (combined)")

    lo, hi = result["published_reference"]["published_S_range_approx"]
    ax.axhspan(lo, hi, color="#d0d0d0", alpha=0.35, zorder=0)
    ax.axhline(hi, color="#888888", ls="--", lw=1.0)
    ax.text(len(names) - 0.5, hi + 0.01, "BATADAL published range (Taormina 2018)",
            ha="right", va="bottom", fontsize=7, color="#555555")

    u = result["hydrojev_family_union"]["test"]
    ax.axhline(u["S"], color="#d62728", ls=":", lw=1.4,
               label=f"HydroJEV family union S={u['S']:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("BATADAL score")
    ax.set_ylim(0, 1.05)
    ax.set_title("HydroJEV on the held-out BATADAL test set, official metric "
                 "(* = named orthogonal family)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, axis="y")
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

    parser = argparse.ArgumentParser(description="HydroJEV under the official BATADAL metric")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir", default="artifacts/batadal_score")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    result = run_batadal_official_score(
        config, mode=mode, seed=args.seed,
        max_epochs=12 if args.quick else args.max_epochs,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "batadal_score.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_batadal_score_table(result)
        (outdir / "batadal_score_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_batadal_score(result, str(outdir / "batadal_score.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'batadal_score.json'}, table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

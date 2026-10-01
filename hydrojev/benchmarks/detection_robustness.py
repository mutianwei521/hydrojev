"""Detection-layer robustness under measurement degradation (§4.2).

§4.1 established that HydroJEV's whole detector stack is competition-grade on the
*official* BATADAL metric under *ideal* measurements. A *Water Research* reviewer
will immediately ask the symmetric question to §7's reflex-robustness sweep: what
happens to the cognitive detectors when the sensors themselves degrade — additive
noise, or partial observability as sensors drop offline? This module answers it
honestly by re-scoring the identical, deployment-frozen detector on a *degraded*
held-out test stream and tracing how the official ``S`` decays.

**Faithful, deployed-detector protocol.** The GRU surrogate, standardizer, healthy
references and per-detector alarm thresholds are trained/calibrated once on clean
data (dataset03 + labelled dataset04) and then held FIXED — a real deployed detector
does not get to re-tune when its inputs start degrading, and never sees the
degradation coming. Degradation is injected into the RAW observed measurements *at
the source* (via ``score_named_families(perturb_eval=...)``), so every detector —
the surrogate-residual families and the instantaneous references alike — reads the
same corrupted feed, exactly as it would in the field.

**Two degradation mechanisms, both scaled by frozen dataset03 statistics (no leakage):**

* *Additive sensor noise* — per channel, ``N(0, (sigma_frac * clean_std_c)^2)``, where
  ``clean_std_c`` is that sensor's healthy operating std from dataset03 (the
  standardizer's own per-channel scale). ``sigma_frac`` is swept.
* *Sensor dropout / partial observability* — a fraction of sensors go offline and are
  imputed to their clean-operation mean (dataset03), the natural missing-channel
  imputation (zero in standardized space). Channels are dropped in nested subsets so
  higher fractions strictly include lower ones; averaged over several random subsets.

For each degradation level we recompute the official BATADAL ``S`` (Taormina et al.,
2018) for every detector and for the named-family union, tuning thresholds ONCE on
clean dataset04. Noise/dropout draws are averaged over ``repeats`` seeds (mean ± std).
Returns ``status="not_run"`` (never fabricated numbers) when torch/BATADAL are absent.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks.batadal_score import BATADAL_REFERENCE, batadal_score, tune_threshold
from hydrojev.benchmarks.named_family_benchmark import (
    PerturbFn,
    contiguous_runs,
    score_named_families,
    train_named_families,
)
from hydrojev.config import HydroJEVConfig


# --------------------------------------------------------------------------- #
# Pure, seeded perturbation factories (unit-testable, no torch / no EPANET)
# --------------------------------------------------------------------------- #


def gaussian_sensor_noise(
    sigma_frac: float, channel_scale: np.ndarray, *, seed: int
) -> PerturbFn:
    """Additive per-channel Gaussian noise at ``sigma_frac * channel_scale`` std.

    ``channel_scale`` is the per-sensor clean-operation std (dataset03). The closure is
    deterministic in ``seed``: invoking it twice yields the identical noise draw.
    """

    scale = np.asarray(channel_scale, dtype=float)

    def perturb(raw: np.ndarray, columns: list[str]) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if scale.shape[0] != raw.shape[1]:
            raise ValueError("channel_scale does not match the measurement width")
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(raw.shape) * (sigma_frac * scale)[None, :]
        return raw + noise

    return perturb


def sensor_dropout(dropped_idx: Sequence[int], channel_fill: np.ndarray) -> PerturbFn:
    """Take a set of sensor indices offline, imputing each to its clean-operation mean.

    ``channel_fill`` is the per-sensor clean mean (dataset03). Deterministic; the entire
    dropped column is replaced by its mean for the whole stream (sensor offline).
    """

    fill = np.asarray(channel_fill, dtype=float)
    dropped = np.asarray(dropped_idx, dtype=int)

    def perturb(raw: np.ndarray, columns: list[str]) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if fill.shape[0] != raw.shape[1]:
            raise ValueError("channel_fill does not match the measurement width")
        out = raw.copy()
        if dropped.size:
            out[:, dropped] = fill[dropped][None, :]
        return out

    return perturb


def nested_dropout_indices(n_features: int, frac: float, *, seed: int) -> np.ndarray:
    """First ``round(frac*n_features)`` channels of a seeded permutation (nested subsets)."""

    perm = np.random.default_rng(seed).permutation(int(n_features))
    k = int(round(float(frac) * n_features))
    return np.sort(perm[:k])


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _alarms(scores: np.ndarray, thr: float) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    return np.isfinite(s) & (s >= thr)


def _aggregate(reps: list[dict[str, dict[str, Any]]], names: Sequence[str]) -> dict[str, dict[str, float]]:
    """Mean/std across repeats of the per-detector BATADAL sub-scores."""

    agg: dict[str, dict[str, float]] = {}
    for name in names:
        S = np.array([r[name]["S"] for r in reps], dtype=float)
        sttd = np.array([r[name]["S_TTD"] for r in reps], dtype=float)
        sclf = np.array([r[name]["S_CLF"] for r in reps], dtype=float)
        tpr = np.array([r[name]["classification"]["TPR"] for r in reps], dtype=float)
        tnr = np.array([r[name]["classification"]["TNR"] for r in reps], dtype=float)
        agg[name] = {
            "S_mean": float(np.nanmean(S)), "S_std": float(np.nanstd(S)),
            "S_TTD_mean": float(np.nanmean(sttd)), "S_CLF_mean": float(np.nanmean(sclf)),
            "TPR_mean": float(np.nanmean(tpr)), "TNR_mean": float(np.nanmean(tnr)),
        }
    return agg


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_detection_robustness(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 80,
    gamma: float = 0.5,
    noise_levels: Sequence[float] | None = None,
    dropout_levels: Sequence[float] | None = None,
    repeats: int = 3,
) -> dict[str, Any]:
    """Trace official BATADAL ``S`` vs sensor noise and vs sensor dropout, detector fixed."""

    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        return {"status": "not_run", "reason": f"torch unavailable: {exc}"}

    seed = int(config.experiments.seed if seed is None else seed)
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5] if noise_levels is None else list(noise_levels)
    dropout_levels = [0.0, 0.1, 0.2, 0.35, 0.5] if dropout_levels is None else list(dropout_levels)
    repeats = max(1, int(repeats))

    # ----- train once, freeze everything (deployed detector) ----------------- #
    trained = train_named_families(config, mode=mode, seed=seed, max_epochs=max_epochs)
    if isinstance(trained, dict):  # not_run
        return trained

    dev = score_named_families(trained, config, evaluate_variant="dataset04")
    if isinstance(dev, dict):
        return dev
    test0 = score_named_families(trained, config, evaluate_variant="test_dataset")
    if isinstance(test0, dict):
        return test0

    names = [n for n in test0.scores if n in dev.scores]
    # thresholds tuned ONCE on clean dataset04, frozen across every degradation level
    dev_thr = {n: tune_threshold(dev.scores[n], dev.labels, gamma=gamma)[0] for n in names}
    families = [f for f in trained.families if f in dev_thr]

    channel_scale = np.asarray(trained.standardizer.scale, dtype=float)
    channel_fill = np.asarray(trained.standardizer.mean, dtype=float)
    n_features = len(trained.columns)

    def score_condition(perturb: PerturbFn | None) -> dict[str, dict[str, Any]]:
        t = score_named_families(trained, config, evaluate_variant="test_dataset", perturb_eval=perturb)
        if isinstance(t, dict):  # should not happen once test0 succeeded
            raise RuntimeError(t.get("reason", "scoring failed under perturbation"))
        out: dict[str, dict[str, Any]] = {}
        for n in names:
            out[n] = batadal_score(t.labels, _alarms(t.scores[n], dev_thr[n]), gamma=gamma)
        acc = np.zeros(t.labels.shape[0], dtype=bool)
        for f in families:
            acc |= _alarms(t.scores[f], dev_thr[f])
        out["__union__"] = batadal_score(t.labels, acc, gamma=gamma)
        return out

    curve_names = list(names) + ["__union__"]

    # ----- baseline (level 0) == §4.1 held-out numbers (built-in sanity anchor) #
    base = score_condition(None)
    baseline = {n: float(base[n]["S"]) for n in curve_names}

    # ----- noise sweep ------------------------------------------------------- #
    noise_curves: dict[str, list[dict[str, Any]]] = {n: [] for n in curve_names}
    for li, level in enumerate(noise_levels):
        if level <= 0.0:
            reps = [base]
        else:
            reps = [
                score_condition(
                    gaussian_sensor_noise(level, channel_scale, seed=seed * 1000 + li * 17 + r)
                )
                for r in range(repeats)
            ]
        agg = _aggregate(reps, curve_names)
        for n in curve_names:
            noise_curves[n].append({"level": float(level), **agg[n]})

    # ----- dropout sweep ----------------------------------------------------- #
    dropout_curves: dict[str, list[dict[str, Any]]] = {n: [] for n in curve_names}
    dropped_names_by_level: dict[str, list[str]] = {}
    for level in dropout_levels:
        if level <= 0.0:
            reps = [base]
            dropped_names_by_level[f"{level:.2f}"] = []
        else:
            reps = []
            example_dropped: list[str] = []
            for r in range(repeats):
                idx = nested_dropout_indices(n_features, level, seed=seed * 1000 + 7 + r)
                if r == 0:
                    example_dropped = [trained.columns[i] for i in idx]
                reps.append(score_condition(sensor_dropout(idx, channel_fill)))
            dropped_names_by_level[f"{level:.2f}"] = example_dropped
        agg = _aggregate(reps, curve_names)
        for n in curve_names:
            dropout_curves[n].append({"level": float(level), **agg[n]})

    return {
        "status": "ok",
        "protocol": (
            "detection-layer robustness under measurement degradation: GRU surrogate + "
            "references + thresholds frozen on dataset03/dataset04 (deployed detector), "
            "raw held-out test measurements degraded at the source; official BATADAL "
            "S = 0.5*S_TTD + 0.5*S_CLF recomputed per level (mean +/- std over repeats)"
        ),
        "seed": seed,
        "gamma": float(gamma),
        "repeats": repeats,
        "n_test_attacks": len(contiguous_runs(test0.labels, 1)),
        "n_test_samples": int(test0.labels.shape[0]),
        "n_features": n_features,
        "detectors": list(names),
        "families": families,
        "references": list(trained.references),
        "thresholds_tuned_on": "dataset04 (clean); frozen across all degradation levels",
        "baseline_S": baseline,
        "noise": {
            "unit": "sigma_frac = noise std as a fraction of each sensor's clean-operation std (dataset03)",
            "levels": [float(x) for x in noise_levels],
            "curves": noise_curves,
        },
        "dropout": {
            "unit": "fraction of sensors offline, imputed to clean-operation mean (dataset03); nested subsets",
            "levels": [float(x) for x in dropout_levels],
            "curves": dropout_curves,
            "example_dropped_sensors": dropped_names_by_level,
        },
        "published_reference": BATADAL_REFERENCE,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _degradation_table(title: str, block: dict[str, Any], order: Sequence[str]) -> list[str]:
    levels = block["levels"]
    lines = [title, block["unit"], ""]
    header = f"{'detector':<16}" + "".join(f"{lv:>8.2f}" for lv in levels)
    lines.append(header)
    lines.append("-" * len(header))
    for name in order:
        curve = block["curves"][name]
        label = "family union" if name == "__union__" else name
        row = f"{label:<16}" + "".join(f"{pt['S_mean']:>8.3f}" for pt in curve)
        lines.append(row)
    return lines


def format_detection_robustness_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    # order rows by clean-baseline S (best first), union last
    dets = sorted(result["detectors"], key=lambda n: result["baseline_S"][n], reverse=True)
    order = dets + ["__union__"]
    fam = set(result["families"])
    lines = [
        f"HydroJEV detection-layer robustness under measurement degradation "
        f"({result['n_test_attacks']} attacks, {result['n_test_samples']} test samples; "
        f"seed {result['seed']}, {result['repeats']} repeats).",
        "Official BATADAL S on the held-out test set; detector + thresholds frozen "
        "(tuned on clean dataset04). Cells = mean S. (family) = named orthogonal family.",
        "",
    ]
    lines += _degradation_table(
        "[A] additive per-sensor Gaussian noise", result["noise"], order
    )
    lines.append("")
    lines += _degradation_table(
        "[B] sensor dropout / partial observability", result["dropout"], order
    )
    lines += [
        "",
        "families: " + ", ".join(n for n in result["detectors"] if n in fam),
        f"Context — {result['published_reference']['citation']}: 7 ranked entries span "
        f"S approx {result['published_reference']['published_S_range_approx'][0]:.2f}"
        f"-{result['published_reference']['published_S_range_approx'][1]:.2f} "
        f"(cited for context; not HydroJEV numbers).",
    ]
    return "\n".join(lines)


def plot_detection_robustness(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fam = set(result["families"])
    dets = sorted(result["detectors"], key=lambda n: result["baseline_S"][n], reverse=True)
    palette = plt.get_cmap("tab10")
    colors = {n: palette(i % 10) for i, n in enumerate(dets)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
    panels = [
        (axes[0], result["noise"], "additive sensor noise  (sigma / clean-op std)", "[A] Gaussian sensor noise"),
        (axes[1], result["dropout"], "fraction of sensors offline", "[B] sensor dropout"),
    ]
    lo, hi = result["published_reference"]["published_S_range_approx"]
    for ax, block, xlabel, title in panels:
        levels = np.asarray(block["levels"], dtype=float)
        ax.axhspan(lo, hi, color="#d0d0d0", alpha=0.30, zorder=0)
        for n in dets:
            curve = block["curves"][n]
            y = np.array([pt["S_mean"] for pt in curve])
            e = np.array([pt["S_std"] for pt in curve])
            style = "-" if n in fam else "--"
            lab = n + (" *" if n in fam else "")
            ax.plot(levels, y, style, color=colors[n], marker="o", ms=3.5, lw=1.5, label=lab)
            ax.fill_between(levels, y - e, y + e, color=colors[n], alpha=0.12, lw=0)
        u = block["curves"]["__union__"]
        yu = np.array([pt["S_mean"] for pt in u])
        ax.plot(levels, yu, "-", color="#d62728", marker="s", ms=4, lw=2.0, label="family union")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(levels.min(), levels.max())
    axes[0].set_ylabel("official BATADAL S (held-out test)")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(fontsize=6.5, loc="lower left", ncol=2)
    fig.suptitle(
        "HydroJEV detection-layer graceful degradation under measurement noise & dropout "
        "(detector frozen; * = named orthogonal family)", fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    parser = argparse.ArgumentParser(description="HydroJEV detection-layer robustness (BATADAL S)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir", default="artifacts/detection_robustness")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    kwargs: dict[str, Any] = {}
    if args.quick:
        kwargs.update(noise_levels=[0.0, 0.1, 0.3], dropout_levels=[0.0, 0.2], repeats=2)
    result = run_detection_robustness(
        config, mode=mode, seed=args.seed,
        max_epochs=12 if args.quick else args.max_epochs,
        repeats=args.repeats if not args.quick else 2,
        **kwargs,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "detection_robustness.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_detection_robustness_table(result)
        (outdir / "detection_robustness_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_detection_robustness(result, str(outdir / "detection_robustness.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'detection_robustness.json'}, table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

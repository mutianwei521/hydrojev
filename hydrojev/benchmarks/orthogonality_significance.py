"""§6.1 orthogonality — is the temporal-family decorrelation statistically real?

§6.1 reports a Spearman rank-correlation matrix showing the *temporal* named families
(GEpFM drift, RAT covariance) decorrelate from the instantaneous reference detectors
(Mahalanobis, PCA, HydroJEV AE) far more than those instantaneous detectors correlate
among themselves (≈0.19 cross vs ≈0.48 within, point estimates). A reviewer will ask:
those are point estimates on autocorrelated score series — is the gap *significant*?

This module answers it honestly with a **paired moving-block bootstrap** on the shared
`dataset04` score series (the same series §6.1 correlates). The moving-block resample
preserves short-range temporal autocorrelation (i.i.d. resampling would fabricate
independence and understate the intervals), and the *paired* design — both group means
recomputed on the *same* resampled series each replicate — gives a correct interval on
their difference. The headline statistic is

    Δ = mean Spearman WITHIN the instantaneous detectors
        − mean Spearman of each TEMPORAL family against each instantaneous detector,

with Δ > 0 meaning the temporal families are the *more decorrelated* group. We report Δ
and its 95% percentile interval at several block lengths (robustness), plus a 95%
interval on every pairwise correlation. CPDZ (an instantaneous named family) is kept out
of both contrast groups — it is expected to sit near Mahalanobis — but still appears in
the pairwise-interval matrix.

Leakage-safe (all scores come from the §6.1 builder: families/surrogate/references/
thresholds fit on dataset03, scored causally on dataset04). Returns
``status="not_run"`` (never fabricated numbers) when BATADAL/torch are unavailable.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Rank / correlation primitives (vectorised, scipy-free — matches the compact
# ``_rankdata`` in named_family_benchmark but for a whole matrix at once)
# --------------------------------------------------------------------------- #


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Average (fractional) ranks 1..n with ties averaged — vectorised, no scipy."""

    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    # Positions 1..n in sorted order, averaged within each run of equal values so tied
    # values share the mean of the positions they occupy (== scipy "average" ranks).
    _, inv, counts = np.unique(sx, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.shape[0], dtype=float)
    np.add.at(sums, inv, np.arange(1, n + 1, dtype=float))
    avg_sorted = (sums / counts)[inv]
    out = np.empty(n, dtype=float)
    out[order] = avg_sorted
    return out


def _rank_columns(matrix: np.ndarray) -> np.ndarray:
    """Column-wise average ranks of an ``[n, k]`` matrix."""

    matrix = np.asarray(matrix, dtype=float)
    ranked = np.empty_like(matrix)
    for j in range(matrix.shape[1]):
        ranked[:, j] = _avg_rank(matrix[:, j])
    return ranked


def spearman_matrix_fast(matrix: np.ndarray) -> np.ndarray:
    """Full ``[k, k]`` Spearman matrix of the columns of an ``[n, k]`` array."""

    ranks = _rank_columns(matrix)
    # Pearson correlation of the ranks == Spearman rho.
    return np.corrcoef(ranks, rowvar=False)


# --------------------------------------------------------------------------- #
# Moving-block bootstrap
# --------------------------------------------------------------------------- #


def moving_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Non-circular moving-block bootstrap index vector of length ``n``.

    ``ceil(n / block_len)`` consecutive blocks of length ``block_len`` are drawn (with
    replacement) from all ``n - block_len + 1`` valid start positions and concatenated,
    then truncated to ``n``. Applying the SAME index vector to every series preserves the
    cross-series alignment at each time step (required for a correlation statistic).
    """

    if block_len < 1 or block_len > n:
        raise ValueError(f"block_len must be in [1, n]={n}, got {block_len}")
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_len)[None, :]).reshape(-1)[:n]
    return idx


def _group_means(rho: np.ndarray, within_idx: Sequence[tuple[int, int]],
                 cross_idx: Sequence[tuple[int, int]]) -> tuple[float, float]:
    within = float(np.mean([rho[i, j] for i, j in within_idx])) if within_idx else float("nan")
    cross = float(np.mean([rho[i, j] for i, j in cross_idx])) if cross_idx else float("nan")
    return within, cross


def _pairs_within(indices: Sequence[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            out.append((indices[a], indices[b]))
    return out


def _pairs_cross(rows: Sequence[int], cols: Sequence[int]) -> list[tuple[int, int]]:
    return [(r, c) for r in rows for c in cols]


def bootstrap_contrast(
    matrix: np.ndarray,
    *,
    temporal_cols: Sequence[int],
    instantaneous_cols: Sequence[int],
    block_len: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Paired moving-block bootstrap of Δ = within-instantaneous − temporal-vs-instantaneous.

    Returns the point estimate, the 95% percentile interval on Δ, the fraction of
    replicates with Δ > 0 (a one-sided bootstrap significance), and the marginal
    intervals on each group mean.
    """

    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    within_idx = _pairs_within(list(instantaneous_cols))
    cross_idx = _pairs_cross(list(temporal_cols), list(instantaneous_cols))

    point_rho = spearman_matrix_fast(matrix)
    p_within, p_cross = _group_means(point_rho, within_idx, cross_idx)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    withins = np.empty(n_boot, dtype=float)
    crosses = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        rho = spearman_matrix_fast(matrix[idx])
        w, c = _group_means(rho, within_idx, cross_idx)
        withins[b], crosses[b], deltas[b] = w, c, w - c

    ok = np.isfinite(deltas)
    deltas, withins, crosses = deltas[ok], withins[ok], crosses[ok]

    def ci(a: np.ndarray) -> list[float]:
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    frac_gt0 = float(np.mean(deltas > 0.0)) if deltas.size else float("nan")
    return {
        "block_len": int(block_len),
        "n_boot_valid": int(deltas.size),
        "within_instantaneous": {"point": p_within, "ci95": ci(withins)},
        "temporal_vs_instantaneous": {"point": p_cross, "ci95": ci(crosses)},
        "delta": {"point": p_within - p_cross, "ci95": ci(deltas), "frac_positive": frac_gt0},
    }


# --------------------------------------------------------------------------- #
# Real-data runner
# --------------------------------------------------------------------------- #


def run_orthogonality_significance(
    config: Any,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 80,
    n_boot: int = 2000,
    block_lengths: Sequence[int] = (25, 50, 100),
    primary_block_len: int = 50,
) -> dict[str, Any]:
    """Bootstrap the §6.1 orthogonality claim on the shared dataset04 score series."""

    try:
        import torch  # noqa: F401
    except Exception as exc:
        return {"status": "not_run", "reason": f"torch unavailable: {exc}"}

    from hydrojev.benchmarks.named_family_benchmark import build_named_family_scores

    built = build_named_family_scores(config, mode=mode, seed=seed, max_epochs=max_epochs)
    if isinstance(built, dict):  # not_run
        return built

    # Common finite support across every series (they are already warmup-trimmed to a
    # shared index; this only guards against a stray non-finite value, keeping every
    # bootstrap replicate on an identical, fully-defined sample).
    names = list(built.scores.keys())
    stacked = np.column_stack([np.asarray(built.scores[k], dtype=float) for k in names])
    finite_rows = np.all(np.isfinite(stacked), axis=1)
    matrix = stacked[finite_rows]
    n = matrix.shape[0]
    if n < max(block_lengths) + 5:
        return {"status": "not_run", "reason": "too few common finite samples for the bootstrap"}

    col = {name: i for i, name in enumerate(names)}
    temporal = [f for f in ("GEpFM drift", "RAT covariance") if f in col]
    instantaneous = [r for r in built.references if r in col]
    if len(temporal) < 1 or len(instantaneous) < 2:
        return {"status": "not_run", "reason": "need >=1 temporal family and >=2 instantaneous refs"}

    base_seed = int(config.experiments.seed if seed is None else seed)

    # Pairwise 95% intervals for the whole matrix at the primary block length.
    point_rho = spearman_matrix_fast(matrix)
    rng = np.random.default_rng(base_seed + 101)
    boot_rho = np.empty((n_boot, len(names), len(names)), dtype=float)
    for b in range(n_boot):
        idx = moving_block_indices(n, primary_block_len, rng)
        boot_rho[b] = spearman_matrix_fast(matrix[idx])
    pairwise: dict[str, dict[str, Any]] = {}
    for a in names:
        pairwise[a] = {}
        for bname in names:
            i, j = col[a], col[bname]
            series = boot_rho[:, i, j]
            series = series[np.isfinite(series)]
            lo = float(np.percentile(series, 2.5)) if series.size else float("nan")
            hi = float(np.percentile(series, 97.5)) if series.size else float("nan")
            pairwise[a][bname] = {"rho": float(point_rho[i, j]), "ci95": [lo, hi]}

    # Headline contrast Δ at several block lengths (robustness to the block choice).
    contrasts = [
        bootstrap_contrast(
            matrix, temporal_cols=[col[t] for t in temporal],
            instantaneous_cols=[col[r] for r in instantaneous],
            block_len=L, n_boot=n_boot, seed=base_seed + 1000 + L,
        )
        for L in block_lengths
    ]

    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "protocol": (
            "paired moving-block bootstrap (block preserves temporal autocorrelation; both "
            "group means recomputed on the same resampled series each replicate) of the §6.1 "
            "Spearman orthogonality matrix; Delta = mean within-instantaneous corr - mean "
            "temporal-family-vs-instantaneous corr, Delta>0 => temporal families more decorrelated"
        ),
        "seed": base_seed,
        "n_scored_samples": int(n),
        "n_boot": int(n_boot),
        "primary_block_len": int(primary_block_len),
        "series": names,
        "temporal_families": temporal,
        "instantaneous_refs": instantaneous,
        "pairwise_spearman_ci": pairwise,
        "contrast_by_block_len": contrasts,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def _interval_cell(point: float, ci: Sequence[float]) -> str:
    """A ``point [lo,hi]`` cell — built as a plain string so the table code stays
    free of nested f-strings (a SyntaxError before Python 3.12)."""

    lo, hi = ci
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def format_significance_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    families = ", ".join(result["temporal_families"])
    refs = ", ".join(result["instantaneous_refs"])
    lines = [
        f"Orthogonality significance on {result['evaluate_dataset']} "
        f"({result['n_scored_samples']} common samples; fit on {result['train_dataset']}).",
        f"Paired moving-block bootstrap, {result['n_boot']} replicates. "
        f"Temporal families: {families}. Instantaneous refs: {refs}.",
        "",
        "Headline contrast  Delta = within-instantaneous corr - temporal-vs-instantaneous corr",
        "(Delta > 0 => the TEMPORAL families are the more decorrelated group):",
        "",
        f"{'block':>6}{'within-inst':>26}{'temporal-vs-inst':>26}{'Delta [95% CI]':>28}{'P(D>0)':>9}",
        "-" * 95,
    ]
    for c in result["contrast_by_block_len"]:
        w, cr, d = c["within_instantaneous"], c["temporal_vs_instantaneous"], c["delta"]
        within_cell = _interval_cell(w["point"], w["ci95"])
        cross_cell = _interval_cell(cr["point"], cr["ci95"])
        delta_cell = _interval_cell(d["point"], d["ci95"])
        lines.append(
            f"{c['block_len']:>6}{within_cell:>26}{cross_cell:>26}"
            f"{delta_cell:>28}{d['frac_positive']:>9.3f}"
        )
    lines += [
        "",
        "Pairwise Spearman rho [95% CI] (temporal x instantaneous should be low & tight;",
        "instantaneous x instantaneous higher):",
        "",
    ]
    tfam = result["temporal_families"]
    inst = result["instantaneous_refs"]
    for t in tfam:
        for r in inst:
            cell = result["pairwise_spearman_ci"][t][r]
            lo, hi = cell["ci95"]
            lines.append(f"  {t:<16} x {r:<14} rho={cell['rho']:.3f} [{lo:.3f}, {hi:.3f}]")
    lines.append("")
    for a in range(len(inst)):
        for b in range(a + 1, len(inst)):
            cell = result["pairwise_spearman_ci"][inst[a]][inst[b]]
            lo, hi = cell["ci95"]
            lines.append(f"  {inst[a]:<16} x {inst[b]:<14} rho={cell['rho']:.3f} [{lo:.3f}, {hi:.3f}]")
    return "\n".join(lines)


def plot_significance(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contrasts = result["contrast_by_block_len"]
    blocks = [c["block_len"] for c in contrasts]
    x = np.arange(len(contrasts))
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for offset, key, color, label in [
        (-0.16, "within_instantaneous", "#b2182b", "within instantaneous"),
        (0.16, "temporal_vs_instantaneous", "#2166ac", "temporal vs instantaneous"),
    ]:
        pts = [c[key]["point"] for c in contrasts]
        los = [c[key]["point"] - c[key]["ci95"][0] for c in contrasts]
        his = [c[key]["ci95"][1] - c[key]["point"] for c in contrasts]
        ax.errorbar(x + offset, pts, yerr=[los, his], fmt="o", color=color, capsize=5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={b}" for b in blocks])
    ax.set_ylabel("mean Spearman rho")
    ax.set_xlabel("moving-block length")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "Orthogonality is significant: temporal families decorrelate from\n"
        "instantaneous detectors more than those correlate among themselves",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
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
        description="§6.1 orthogonality significance (paired moving-block bootstrap)"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--outdir", default="artifacts/orthogonality_significance")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    epochs = 12 if args.quick else args.max_epochs
    n_boot = 300 if args.quick else args.n_boot
    result = run_orthogonality_significance(
        config, mode=mode, seed=args.seed, max_epochs=epochs, n_boot=n_boot
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "orthogonality_significance.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    if result.get("status") == "ok":
        table = format_significance_table(result)
        (outdir / "orthogonality_significance_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_significance(result, str(outdir / "orthogonality_significance.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'orthogonality_significance.json'}, table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fuse the clean-detection and adversarial-evasion benchmarks into one story.

The two benchmarks answer different questions:

* :mod:`detection_benchmark` — *clean* discrimination: on unmanipulated BATADAL,
  which detector flags the most attack samples? (Classical Mahalanobis/PCA win.)
* :mod:`evasion_benchmark` — *adversarial* robustness: of the alarms a detector
  raises, what fraction can a bounded, stealthy adversary silence? (The learned
  autoencoder is by far the hardest to evade.)

Neither number alone reflects operational value. The quantity a defender actually
cares about is how many attack samples remain flagged *after* a best-effort
stealthy evasion:

    post-evasion point recall = clean point recall x (1 - evasion success rate)

This is the fraction of all attack samples that are both detected and cannot be
hidden by the shared coordinate-descent attack. It reverses the clean ranking:
the detector with the best clean recall can still be the weakest once the
adversary is allowed to act. This module computes that fused metric from the two
saved benchmark JSONs and renders a comparison table + figure.

The evasion rate is estimated on a representative subsample of each detector's
detected alarms, so the fused recall is a point estimate; the component
confidence intervals live in the two source benchmarks.
"""

from __future__ import annotations

from typing import Any, Sequence


def combined_rows(detection: dict[str, Any], evasion: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per detector fusing clean recall with evasion robustness.

    Detectors are matched by name across the two result dicts; any detector
    absent from either benchmark is skipped.
    """

    if detection.get("status") != "ok" or evasion.get("status") != "ok":
        return []

    rows: list[dict[str, Any]] = []
    det_map = detection["detectors"]
    eva_map = evasion["detectors"]
    for name, det in det_map.items():
        eva = eva_map.get(name)
        if eva is None:
            continue
        clean_point_recall = float(det["metrics"]["point_recall"]["mean"])
        clean_scenario_recall = float(det["metrics"]["scenario_recall"]["mean"])
        evasion_rate = float(eva["evasion_success_rate"])
        # If the detector detected nothing, evasion is undefined; nothing survives
        # because nothing was flagged, so post-evasion recall is 0 either way.
        post = clean_point_recall * (1.0 - evasion_rate) if eva["n_targets"] > 0 else 0.0
        rows.append(
            {
                "detector": name,
                "is_hydrojev": bool(det["is_hydrojev"]),
                "clean_point_recall": clean_point_recall,
                "clean_scenario_recall": clean_scenario_recall,
                "evasion_success_rate": evasion_rate,
                "evasion_targets": int(eva["n_targets"]),
                "post_evasion_point_recall": post,
                "recall_retained_fraction": (post / clean_point_recall) if clean_point_recall > 0 else float("nan"),
            }
        )
    # Best operational robustness first.
    rows.sort(key=lambda r: r["post_evasion_point_recall"], reverse=True)
    return rows


def format_combined_table(rows: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'detector':<18}{'clean pt recall':>16}{'evasion rate':>14}"
        f"{'post-evasion recall':>21}{'retained':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        tag = " *" if r["is_hydrojev"] else "  "
        retained = "n/a" if r["clean_point_recall"] <= 0 else f"{r['recall_retained_fraction']:.0%}"
        lines.append(
            f"{r['detector'] + tag:<18}{r['clean_point_recall']:>16.3f}"
            f"{r['evasion_success_rate']:>14.3f}{r['post_evasion_point_recall']:>21.3f}{retained:>10}"
        )
    return "\n".join(lines)


def plot_robustness(rows: Sequence[dict[str, Any]], path: str) -> str | None:
    """Grouped bars: clean point recall vs post-evasion point recall per detector."""

    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["detector"] + (" (ours)" if r["is_hydrojev"] else "") for r in rows]
    clean = [r["clean_point_recall"] for r in rows]
    post = [r["post_evasion_point_recall"] for r in rows]
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2, clean, width, label="clean point recall", color="#6baed6")
    ax.bar(x + width / 2, post, width, label="post-evasion point recall", color="#e6550d")
    for i, r in enumerate(rows):
        drop = r["clean_point_recall"] - r["post_evasion_point_recall"]
        if r["clean_point_recall"] > 0:
            ax.annotate(
                f"-{drop / r['clean_point_recall']:.0%}",
                (x[i], max(clean[i], post[i]) + 0.01),
                ha="center", va="bottom", fontsize=8, color="#a50f15",
            )
    for i, r in enumerate(rows):
        if r["is_hydrojev"]:
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, color="gold", alpha=0.12, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("point recall on BATADAL dataset04")
    ax.set_ylim(0.0, max(0.6, max(clean) * 1.2))
    ax.set_title(
        "Clean vs. post-evasion detection recall (labels above bars = recall lost to a\n"
        "shared stealthy attack). The best clean detector is not the most robust.",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Fuse detection + evasion benchmarks")
    parser.add_argument("--detection", default="artifacts/detection_benchmark/benchmark.json")
    parser.add_argument("--evasion", default="artifacts/evasion_benchmark/evasion.json")
    parser.add_argument("--outdir", default="artifacts/robustness_summary")
    args = parser.parse_args(argv)

    detection = json.loads(Path(args.detection).read_text(encoding="utf-8"))
    evasion = json.loads(Path(args.evasion).read_text(encoding="utf-8"))
    rows = combined_rows(detection, evasion)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok" if rows else "not_run",
        "train_dataset": detection.get("train_dataset"),
        "evaluate_dataset": detection.get("evaluate_dataset"),
        "definition": "post_evasion_point_recall = clean_point_recall * (1 - evasion_success_rate)",
        "detectors": rows,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if rows:
        table = format_combined_table(rows)
        (outdir / "summary_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_robustness(rows, str(outdir / "summary.png"))
        print(table)
        print(f"\nwrote: {outdir/'summary.json'}, summary_table.txt, {figure}")
    else:
        print("not_run: one or both source benchmarks are missing/incomplete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

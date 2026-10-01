"""Figure 1 — the Cognitive Reflex Paradigm architecture schematic.

This is a *conceptual* figure (no measured data), faithful to the method in §3:

* **Cognition path** (dashed, blue): sensors -> SCADA/telemetry channel -> detector
  layer (GRU surrogate + named families CPDZ/GEpFM/RAT and references Maha/PCA/AE)
  -> cloud arbiter. It is network-dependent, *advisory*, and -- as §5/§6 show --
  *evadable*. It rides the same SCADA channel a cyber attacker can spoof.
* **Reflex path** (solid, green): a hardwired tap on the *true* tank level feeds a
  local, deterministic interlock that closes the pump above a safety band, with no
  cloud, no learned model, and no network in the loop. It reads ground truth on an
  air-gapped channel -- nothing for the attacker to spoof.

The single visual message: the safety guarantee is the *solid* path, which bypasses
the (dashed, attackable) cyber layer entirely.

No fabricated numbers: the two latency annotations (reflex 23.6 us, cognition
~1.1 ms/step) are the measured §4.3 medians, and are labelled as such.
"""

from __future__ import annotations

from typing import Any, Sequence


# Style constants shared by the schematic (kept module-level so a test can assert
# the cognition/reflex visual encoding is distinct and not accidentally unified).
COGNITION_STYLE: dict[str, Any] = {
    "edgecolor": "#1f4e79",
    "facecolor": "#eaf1f8",
    "linestyle": "--",
    "linewidth": 1.6,
}
REFLEX_STYLE: dict[str, Any] = {
    "edgecolor": "#1b6b2e",
    "facecolor": "#e6f3ea",
    "linestyle": "-",
    "linewidth": 2.6,
}
ATTACK_STYLE: dict[str, Any] = {
    "edgecolor": "#b3261e",
    "facecolor": "#fdeceb",
    "linestyle": "--",
    "linewidth": 1.8,
}
PLANT_STYLE: dict[str, Any] = {
    "edgecolor": "#444444",
    "facecolor": "#f2f2f2",
    "linestyle": "-",
    "linewidth": 1.6,
}


def _box(ax, cx: float, cy: float, w: float, h: float, text: str, style: dict[str, Any],
         *, fontsize: float = 8.5, weight: str = "normal", textcolor: str = "black") -> None:
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        mutation_aspect=1.0,
        **style,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            weight=weight, color=textcolor, zorder=5)


def _arrow(ax, xy_from: tuple[float, float], xy_to: tuple[float, float], *,
           color: str, style: str = "-", lw: float = 1.6, label: str | None = None,
           label_dx: float = 0.0, label_dy: float = 0.0, label_color: str | None = None) -> None:
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style,
                        shrinkA=2, shrinkB=2),
        zorder=4,
    )
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2 + label_dx
        my = (xy_from[1] + xy_to[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.0,
                color=label_color or color, style="italic", zorder=6)


def plot_paradigm(path: str) -> str:
    """Render the paradigm schematic to ``path`` and return it."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 8.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # --- physical plant (ground truth) ------------------------------------- #
    _box(ax, 5.0, 1.0, 8.4, 1.0,
         "PHYSICAL PLANT — water distribution network\n"
         "tanks · pumps · valves · sensors  (TRUE physical state)",
         PLANT_STYLE, fontsize=9.0, weight="bold")

    # --- attacker + SCADA channel (the spoofable cyber layer) --------------- #
    _box(ax, 3.75, 3.2, 3.3, 1.05,
         "SCADA / PLC telemetry + control\n⚡ CYBER ATTACK SURFACE (spoofable)",
         ATTACK_STYLE, fontsize=8.3, weight="bold", textcolor="#7a1a15")
    _box(ax, 0.95, 3.2, 1.4, 0.85,
         "adversary\n(FDI / command)", ATTACK_STYLE, fontsize=7.5, textcolor="#7a1a15")
    _arrow(ax, (1.65, 3.2), (2.10, 3.2), color="#b3261e", lw=1.8,
           label="inject", label_dy=0.28)  # attacker -> SCADA (rightward)

    # --- cognition path (dashed, advisory, network-dependent) --------------- #
    _box(ax, 3.2, 5.25, 4.4, 1.25,
         "COGNITION · detector layer\n"
         "GRU physics surrogate → CPDZ · GEpFM · RAT\n"
         "references: Mahalanobis · PCA · AE",
         COGNITION_STYLE, fontsize=8.2, textcolor="#123a5c")
    _box(ax, 3.2, 7.15, 4.4, 1.15,
         "COGNITION · cloud arbiter  (ADVISORY)\n"
         "behind hard data-freshness / interlock gates\n"
         "network-dependent · evadable · may be stale/offline",
         COGNITION_STYLE, fontsize=8.2, textcolor="#123a5c")

    # --- reflex path (solid, deterministic, air-gapped) --------------------- #
    _box(ax, 7.9, 5.6, 3.5, 1.9,
         "REFLEX · local interlock\n\n"
         "if TRUE level > safety band\n→ CLOSE PUMP (full stop)\n\n"
         "deterministic · air-gapped\nno model · no network",
         REFLEX_STYLE, fontsize=8.6, weight="bold", textcolor="#0f4a20")

    # --- actuation (reflex has hard priority) ------------------------------- #
    _box(ax, 5.0, 9.25, 5.4, 0.9,
         "ACTUATION — pump / valve  (REFLEX has hard priority over cognition)",
         PLANT_STYLE, fontsize=8.8, weight="bold")

    # --- edges -------------------------------------------------------------- #
    # cognition (dashed blue): plant -> SCADA -> detector -> arbiter -> actuation
    _arrow(ax, (3.75, 1.5), (3.75, 2.675), color="#1f4e79", style="--", lw=1.6,
           label="spoofable\ntelemetry", label_dx=-0.95)
    _arrow(ax, (3.6, 3.725), (3.3, 4.63), color="#1f4e79", style="--", lw=1.6)
    _arrow(ax, (3.2, 5.875), (3.2, 6.575), color="#1f4e79", style="--", lw=1.6)
    _arrow(ax, (3.9, 7.725), (4.6, 9.0), color="#1f4e79", style="--", lw=1.6,
           label="advisory\nrecommend", label_dx=0.78, label_dy=-0.1)

    # reflex (solid green): plant -> reflex (air-gapped tap) -> actuation
    _arrow(ax, (7.9, 1.5), (7.9, 4.65), color="#1b6b2e", style="-", lw=2.8,
           label="air-gapped\ntrue-level tap\n(unspoofable)", label_dx=1.15)
    _arrow(ax, (7.9, 6.55), (6.0, 9.0), color="#1b6b2e", style="-", lw=2.8,
           label="hard\ncut-off", label_dx=0.55, label_dy=-0.15)

    # --- title + reading key ------------------------------------------------ #
    fig.suptitle(
        "The Cognitive Reflex Paradigm: detection is advisory; the guarantee is a "
        "deterministic, air-gapped physical reflex",
        fontsize=11.5, weight="bold", y=0.985,
    )
    ax.text(
        5.0, 0.18,
        "Dashed blue = cognition path: rides the spoofable SCADA channel, advisory & "
        "evadable, ~1.1 ms per step (measured median).\n"
        "Solid green = safety path: reads ground truth on an air-gapped channel, "
        "23.6 µs per decision (measured median), nothing to spoof.",
        ha="center", va="center", fontsize=7.8, color="#333333",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Render Figure 1: paradigm schematic")
    parser.add_argument("--outdir", default="artifacts/paradigm_figure")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = plot_paradigm(str(outdir / "paradigm_architecture.png"))
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

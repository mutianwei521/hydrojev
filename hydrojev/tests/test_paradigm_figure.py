"""Tests for Figure 1 (the paradigm architecture schematic)."""

from __future__ import annotations

from hydrojev.benchmarks.paradigm_figure import (
    COGNITION_STYLE,
    REFLEX_STYLE,
    ATTACK_STYLE,
    plot_paradigm,
    main,
)


def test_cognition_and_reflex_are_visually_distinct():
    # The whole point of the figure is that the two paths read differently: the
    # reflex is a solid, heavier line (the guarantee) and cognition is dashed.
    assert REFLEX_STYLE["linestyle"] == "-"
    assert COGNITION_STYLE["linestyle"] == "--"
    assert REFLEX_STYLE["linewidth"] > COGNITION_STYLE["linewidth"]
    # the attack surface is its own (red) dashed encoding, distinct from cognition
    assert ATTACK_STYLE["edgecolor"] != COGNITION_STYLE["edgecolor"]


def test_plot_paradigm_writes_file(tmp_path):
    out = plot_paradigm(str(tmp_path / "fig1.png"))
    assert out is not None
    p = tmp_path / "fig1.png"
    assert p.exists()
    # a real rendered figure is non-trivial in size
    assert p.stat().st_size > 5000


def test_main_writes_to_outdir(tmp_path):
    rc = main(["--outdir", str(tmp_path / "out")])
    assert rc == 0
    assert (tmp_path / "out" / "paradigm_architecture.png").exists()

"""Tests for the detection+evasion fusion (post-evasion recall) helpers."""

from __future__ import annotations

from hydrojev.benchmarks.robustness_summary import (
    combined_rows,
    format_combined_table,
    plot_robustness,
)


def _detection() -> dict:
    def det(pr, sr, hj=False):
        return {
            "is_hydrojev": hj,
            "metrics": {
                "point_recall": {"mean": pr},
                "scenario_recall": {"mean": sr},
            },
        }

    return {
        "status": "ok",
        "train_dataset": "BATADAL:dataset03",
        "evaluate_dataset": "BATADAL:dataset04",
        "detectors": {
            "HydroJEV AE": det(0.30, 0.94, hj=True),
            "Mahalanobis": det(0.47, 1.00),
            "PCA residual": det(0.34, 1.00),
        },
    }


def _evasion() -> dict:
    def eva(rate, targets=80):
        return {"evasion_success_rate": rate, "n_targets": targets}

    return {
        "status": "ok",
        "detectors": {
            "HydroJEV AE": eva(0.25),
            "Mahalanobis": eva(0.75),
            "PCA residual": eva(0.99),
        },
    }


def test_combined_rows_computes_post_evasion_recall() -> None:
    rows = combined_rows(_detection(), _evasion())
    by = {r["detector"]: r for r in rows}
    # AE: 0.30 * (1 - 0.25) = 0.225
    assert abs(by["HydroJEV AE"]["post_evasion_point_recall"] - 0.225) < 1e-9
    # Mahalanobis: 0.47 * 0.25 = 0.1175
    assert abs(by["Mahalanobis"]["post_evasion_point_recall"] - 0.1175) < 1e-9
    # PCA: 0.34 * 0.01 = 0.0034
    assert abs(by["PCA residual"]["post_evasion_point_recall"] - 0.0034) < 1e-9


def test_combined_rows_sorted_by_robustness_and_ours_wins() -> None:
    rows = combined_rows(_detection(), _evasion())
    # Despite the WORST clean recall, the AE has the best post-evasion recall.
    assert rows[0]["detector"] == "HydroJEV AE"
    assert rows[0]["is_hydrojev"] is True
    posts = [r["post_evasion_point_recall"] for r in rows]
    assert posts == sorted(posts, reverse=True)


def test_combined_rows_retained_fraction() -> None:
    rows = combined_rows(_detection(), _evasion())
    by = {r["detector"]: r for r in rows}
    assert abs(by["HydroJEV AE"]["recall_retained_fraction"] - 0.75) < 1e-9
    assert abs(by["Mahalanobis"]["recall_retained_fraction"] - 0.25) < 1e-9


def test_combined_rows_skips_unmatched_and_handles_not_run() -> None:
    det = _detection()
    det["detectors"]["Isolation Forest"] = {
        "is_hydrojev": False,
        "metrics": {"point_recall": {"mean": 0.02}, "scenario_recall": {"mean": 0.69}},
    }
    # Isolation Forest absent from evasion -> skipped, no crash.
    rows = combined_rows(det, _evasion())
    assert "Isolation Forest" not in {r["detector"] for r in rows}
    assert combined_rows({"status": "not_run"}, _evasion()) == []


def test_zero_detection_gives_zero_post_recall() -> None:
    det = _detection()
    det["detectors"]["Dead"] = {
        "is_hydrojev": False,
        "metrics": {"point_recall": {"mean": 0.0}, "scenario_recall": {"mean": 0.0}},
    }
    eva = _evasion()
    eva["detectors"]["Dead"] = {"evasion_success_rate": float("nan"), "n_targets": 0}
    rows = combined_rows(det, eva)
    dead = next(r for r in rows if r["detector"] == "Dead")
    assert dead["post_evasion_point_recall"] == 0.0


def test_table_and_plot(tmp_path) -> None:
    rows = combined_rows(_detection(), _evasion())
    text = format_combined_table(rows)
    assert "post-evasion recall" in text
    assert "HydroJEV AE *" in text
    out = tmp_path / "s.png"
    assert plot_robustness(rows, str(out)) == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert plot_robustness([], str(tmp_path / "none.png")) is None

"""Tests for the one-command experiments CLI and its honest report schema.

These tests assert *structure, provenance, and scientific invariants* -- not
quick-mode performance numbers, which are deliberately under-powered for speed.
They prove: WADI is reported ``not_run`` with a reason; every result row carries
the full provenance schema; the closed-loop Jev source is unmistakably ``mock``;
BATADAL recall is a real cross-dataset number in [0, 1] tagged with both
datasets; latency is reported as p50/p95/p99 with the water-hammer caveat; and
the synthetic scenarios reach the arbiter as evidence only (no ground-truth
label), with stealthy attacks kept on the cognitive air-gap path.
"""

from __future__ import annotations

import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("wntr")

from hydrojev.arbiter.decision_primitives import JevSource, ThreatCause
from hydrojev.config import load_config
from hydrojev.run_experiments import (
    _ROW_FIELDS,
    RunMode,
    batadal_detector_recall,
    build_manifest,
    build_reconstruction_detector,
    build_parser,
    inspect_assets,
    main,
    measure_latency,
    run_all,
    run_attack_concealment,
    run_attack_fs_fdi,
    run_cluster,
    run_evaluate,
    simulate_closed_loop,
    simulate_energy_mitigation,
    write_result_tables,
    _zone_state,
)
from hydrojev.simulation.closed_loop_eval import ReflexPolicy, distill_evidence
from hydrojev.simulation.mock_jev_server import decide_locally

# A deliberately tiny mode so the whole suite runs in seconds.
TEST_MODE = RunMode(
    name="test",
    batadal_train_rows=200,
    ae_epochs=2,
    scenarios_per_category=3,
    latency_iterations=8,
    concealment_budget=20,
    cluster_net3=False,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def _batadal_available(config) -> bool:
    assets = inspect_assets(config, hash_files=False)
    return (
        assets["batadal"]["dataset03"]["available"]
        and assets["batadal"]["dataset04"]["available"]
    )


@pytest.fixture(scope="module")
def detector(config):
    if not _batadal_available(config):
        pytest.skip("BATADAL dataset03/dataset04 not present in refCode")
    return build_reconstruction_detector(config, mode=TEST_MODE)


# --------------------------------------------------------------------------- #
# Run modes
# --------------------------------------------------------------------------- #


def test_run_mode_resolve_bounds() -> None:
    dry = RunMode.resolve(dry=True, quick=False)
    quick = RunMode.resolve(dry=False, quick=True)
    full = RunMode.resolve(dry=False, quick=False)
    assert dry.dry is True
    assert quick.dry is False and quick.batadal_train_rows == 400 and quick.cluster_net3 is False
    assert full.dry is False and full.batadal_train_rows is None and full.cluster_net3 is True
    # dry takes precedence over quick.
    assert RunMode.resolve(dry=True, quick=True).dry is True


# --------------------------------------------------------------------------- #
# Asset inspection and manifest
# --------------------------------------------------------------------------- #


def test_inspect_reports_networks_and_wadi_not_run(config) -> None:
    assets = inspect_assets(config, hash_files=False)
    assert "Net1" in assets["networks"]
    assert "dataset03" in assets["batadal"] and "dataset04" in assets["batadal"]
    # WADI is a gated benchmark: absent, with an explicit reason (never fabricated).
    assert assets["wadi"]["available"] is False
    assert "wadi" in assets["wadi"]["reason"].lower() or "no csv" in assets["wadi"]["reason"].lower()


def test_manifest_has_reproducibility_fields(config) -> None:
    manifest = build_manifest(config, mode=TEST_MODE, hash_files=False)
    for key in ("created_utc", "mode", "seed", "python", "packages", "torch_device", "solver", "jev", "assets"):
        assert key in manifest
    assert manifest["seed"] == config.experiments.seed
    assert manifest["packages"]["numpy"] != ""
    assert manifest["jev"]["source_offline_default"] == JevSource.MOCK.value


def test_manifest_hashes_present_assets(config) -> None:
    if not _batadal_available(config):
        pytest.skip("BATADAL not present")
    manifest = build_manifest(config, mode=TEST_MODE, hash_files=True)
    d03 = manifest["assets"]["batadal"]["dataset03"]
    assert d03["available"] is True
    assert len(d03["sha256"]) == 64  # full sha256 hex digest


# --------------------------------------------------------------------------- #
# CPDZ topology / clustering
# --------------------------------------------------------------------------- #


def test_cluster_net1_partitions(config) -> None:
    result = run_cluster(config, "Net1", partition=True)
    assert result["network"] == "Net1"
    assert result["pipes"] >= 1
    assert isinstance(result["partition"], dict)
    assert result["partition"]["clusters"] >= 1


def test_cluster_topology_only(config) -> None:
    result = run_cluster(config, "Net1", partition=False)
    assert result["partition"] == "topology_only"


# --------------------------------------------------------------------------- #
# BATADAL real-data detection (cross-dataset)
# --------------------------------------------------------------------------- #


def test_batadal_detector_recall_is_real_and_bounded(config, detector) -> None:
    result = batadal_detector_recall(detector, config)
    assert result["status"] == "ok"
    assert result["train_dataset"] != result["evaluate_dataset"]  # cross-dataset, no leakage
    assert result["attack_scenarios"] >= 1
    recall = result["scenario_recall"]
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= result["point_recall"] <= 1.0
    assert 0.0 <= result["point_false_positive_rate"] <= 1.0


def test_concealment_attack_on_real_data(config, detector) -> None:
    result = run_attack_concealment(config, mode=TEST_MODE, artifacts=detector)
    assert result["dataset"].startswith("BATADAL")
    assert result["n_changed"] <= config.concealment.max_modified_features  # k <= 4
    assert result["query_count"] >= 1
    assert isinstance(result["converged"], bool)
    # Discrete status channels are held immutable, so fewer than all features move.
    assert result["modifiable_features"] < len(detector.feature_columns)


# --------------------------------------------------------------------------- #
# Synthetic closed-loop scenarios: evidence-only, correct causal routing
# --------------------------------------------------------------------------- #


def test_zone_states_are_evidence_only_and_classify_correctly() -> None:
    import random

    rng = random.Random(0)
    expected = {
        "adversarial": ThreatCause.ADVERSARIAL_AE_EVASION,
        "fs_fdi": ThreatCause.HYDRAULIC_FS_FDI,
        "burst": ThreatCause.PHYSICAL_BURST,
        "benign": ThreatCause.NORMAL_TRANSIENT,
    }
    forbidden = {"ground_truth", "true_cause", "label", "attack_label", "category"}
    for kind, cause in expected.items():
        state = _zone_state(kind, rng)
        # No ground-truth label reaches the arbiter.
        assert forbidden.isdisjoint(state.keys())
        assert forbidden.isdisjoint(state["hydraulic_consistency"].keys())
        decision = decide_locally(state, source=JevSource.MOCK)
        assert decision.threat_cause.as_threat_cause() is cause


def test_stealthy_attacks_take_cognitive_airgap_not_reflex() -> None:
    import random

    rng = random.Random(1)
    policy = ReflexPolicy.from_safety_config(load_config().safety)
    # Stealthy cyber attacks keep mass balance within tolerance -> NOT a reflex burst.
    for kind in ("adversarial", "fs_fdi"):
        ev = distill_evidence(_zone_state(kind, rng), policy=policy)
        assert ev.severe_hydraulic_violation is False
        assert ev.local_physics_alert is True  # cpdz residual corroborates
    # A physical burst is a severe hydraulic violation -> reflex path.
    burst_ev = distill_evidence(_zone_state("burst", rng), policy=policy)
    assert burst_ev.severe_hydraulic_violation is True


def test_simulate_closed_loop_recall_and_mock_source(config) -> None:
    summary = simulate_closed_loop(config, mode=TEST_MODE)
    assert summary["n_runs"] == 4 * TEST_MODE.scenarios_per_category
    recall = summary["detection"]["recall"]
    assert 0.0 <= recall <= 1.0
    # Every decision in the offline loop is mock-sourced -- never silently "live".
    assert set(summary["jev_source_counts"]) == {JevSource.MOCK.value}
    assert summary["mean_attack_extra_energy_fraction"] > 0.0


def test_energy_mitigation_tied_to_attack_intensity(config) -> None:
    result = simulate_energy_mitigation(config, attacked_extra_fraction=0.4)
    assert result["extra_energy_fraction_attacked"] == pytest.approx(0.4, abs=1e-6)
    # Defended residual equals the configured tolerated increase.
    assert result["residual_extra_fraction"] <= config.experiments.maximum_energy_increase_fraction + 1e-6
    assert result["mitigated_fraction"] == pytest.approx(0.4 - config.experiments.maximum_energy_increase_fraction, abs=1e-3)


# --------------------------------------------------------------------------- #
# Latency accounting
# --------------------------------------------------------------------------- #


def test_measure_latency_reports_percentiles_and_wave_caveat(config) -> None:
    latency = measure_latency(config, mode=TEST_MODE)
    for pct in ("p50", "p95", "p99"):
        assert pct in latency["total_ms"]
    assert set(latency["per_stage_ms"]) == {"surrogate_ms", "residual_ms", "jev_decision_ms"}
    assert latency["jev_source"] == JevSource.MOCK.value
    assert latency["device"] in {"cpu", "cuda"}
    # Net1 has finite pipes -> a minimum wave-propagation time is reported.
    assert latency["minimum_wave_propagation_ms"] > 0.0


# --------------------------------------------------------------------------- #
# FS-FDI bounded smoke (solver provenance)
# --------------------------------------------------------------------------- #


def test_fs_fdi_smoke_reports_solver_and_status(config) -> None:
    result = run_attack_fs_fdi(config)
    assert result["status"] in {"optimal", "feasible", "infeasible", "solver_unavailable", "error"}
    assert "provenance" in result
    if result["status"] in {"optimal", "feasible"}:
        # The sparsity cap that must be respected is the *instance's* budget, not
        # the library default: a hydraulically consistent attack on a fully
        # instrumented serial zone must co-falsify the whole head cascade, so the
        # instance deliberately raises the budget to the measurement count.
        assert result["n_changed"] <= result["sensor_budget"]


def test_fs_fdi_smoke_finds_verified_stealthy_attack(config) -> None:
    """The paper's 'stealthy cyber-physical attack' claim must be non-vacuous."""
    result = run_attack_fs_fdi(config)
    if result["status"] not in {"optimal", "feasible"}:
        pytest.skip(f"solver returned {result['status']!r}; no attack to verify")

    # A genuine, non-zero, hydraulically consistent, CUSUM-subthreshold attack.
    assert result["impact_flow_increase_m3_s"] > 0.0
    assert result["verification_accepted"] is True
    assert result["verification"]["stealth_ok"] is True
    assert result["verification"]["mass_balance_ok"] is True
    assert result["verification"]["energy_pwl_ok"] is True

    # The headline attack is bounded by physical plausibility, not detectability.
    assert result["binding_constraint"] == "physical_deviation_bound"

    # Sparsity frontier: a feasibility threshold exists, and impact is monotone
    # non-decreasing in the attacker's sensor budget (more sensors never hurt).
    frontier = result["sparsity_frontier"]
    assert result["min_stealthy_sensor_budget"] is not None
    fractions = [row["impact_flow_increase_fraction"] for row in frontier]
    assert fractions == sorted(fractions)
    for row in frontier:
        # Every point on the frontier is itself a verified, stealthy solution.
        assert row["verification_accepted"] is True
        assert row["stealth_ok"] is True


# --------------------------------------------------------------------------- #
# Full evaluation rows: schema, provenance, honest separation
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def evaluation(config):
    if not _batadal_available(config):
        pytest.skip("BATADAL not present")
    return run_evaluate(config, mode=TEST_MODE)


def test_every_row_has_full_provenance_schema(evaluation) -> None:
    rows = evaluation["rows"]
    assert rows
    for row in rows:
        assert set(row.keys()) >= set(_ROW_FIELDS)
        assert row["status"] in {"pass", "fail", "not_run", "measured"}
        assert row["dataset"]  # never blank
        assert row["mode"] == TEST_MODE.name


def test_rows_separate_benchmark_from_simulation(evaluation) -> None:
    rows = {row["metric"]: row for row in evaluation["rows"]}
    # Real benchmark recall is tagged with a BATADAL dataset and cross-dataset provenance.
    bat = rows["reconstruction_detector_scenario_recall"]
    assert bat["dataset"].startswith("BATADAL")
    assert 0.0 <= float(bat["observed"]) <= 1.0
    # Closed-loop recall is explicitly simulation, never mislabeled as a benchmark.
    sim = rows["closed_loop_recall"]
    assert sim["dataset"] == "simulation"
    assert "mock" in sim["provenance"].lower()


def test_wadi_row_is_not_run_with_reason(evaluation) -> None:
    wadi = next(r for r in evaluation["rows"] if r["metric"] == "wadi_benchmark")
    assert wadi["status"] == "not_run"
    assert wadi["observed"] is None
    assert "wadi" in wadi["detail"].lower()


def test_latency_rows_report_all_percentiles(evaluation) -> None:
    metrics = {row["metric"] for row in evaluation["rows"]}
    assert {"total_loop_latency_p50", "total_loop_latency_p95", "total_loop_latency_p99"} <= metrics
    p95 = next(r for r in evaluation["rows"] if r["metric"] == "total_loop_latency_p95")
    assert "water-hammer relay" in p95["detail"].lower()


def test_energy_row_uses_fraction_unit_and_target(evaluation) -> None:
    energy = next(r for r in evaluation["rows"] if r["metric"] == "energy_increase_defended")
    assert energy["unit"] == "fraction"
    assert energy["target"] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# Output tables
# --------------------------------------------------------------------------- #


def test_write_result_tables_csv_and_json(tmp_path) -> None:
    rows = [
        {
            "metric": "demo", "dataset": "simulation", "mode": "test", "target": 0.95,
            "observed": 0.9, "unit": "fraction", "status": "fail", "samples": 3,
            "provenance": "unit-test", "detail": "",
        }
    ]
    outputs = write_result_tables(rows, tmp_path / "out")
    csv_path = Path(outputs["csv"])
    json_path = Path(outputs["json"])
    assert csv_path.is_file() and json_path.is_file()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(_ROW_FIELDS)
        loaded = list(reader)
    assert loaded[0]["metric"] == "demo"
    with json_path.open(encoding="utf-8") as handle:
        assert json.load(handle)[0]["dataset"] == "simulation"


# --------------------------------------------------------------------------- #
# The `all` orchestration
# --------------------------------------------------------------------------- #


def test_run_all_dry_skips_heavy_compute(config, tmp_path) -> None:
    report = run_all(config, mode=RunMode.resolve(dry=True, quick=False), output_dir=tmp_path / "dry")
    assert report["dry_run"] is True
    assert "manifest" in report and "planned" in report
    assert "cluster" not in report and "train_ae" not in report
    assert Path(report["outputs"]["csv"]).is_file()  # header-only table still written


def test_run_all_test_mode_builds_manifest_and_tables(config, tmp_path) -> None:
    if not _batadal_available(config):
        pytest.skip("BATADAL not present")
    report = run_all(config, mode=TEST_MODE, output_dir=tmp_path / "all")
    assert "manifest" in report
    assert report["cluster"]["Net1"]["partition"]["clusters"] >= 1
    # Net3 is topology-only in the reduced mode (bounded runtime, honestly labelled).
    assert report["cluster"]["Net3"]["partition"] == "topology_only"
    assert report["train_ae"]["train_dataset"].startswith("BATADAL")
    assert report["rows"]
    assert Path(report["outputs"]["csv"]).is_file()
    assert Path(report["outputs"]["json"]).is_file()


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def test_parser_accepts_all_commands() -> None:
    parser = build_parser()
    for command in ("inspect", "cluster", "train-ae", "attack-concealment", "attack-fs-fdi", "evaluate", "all"):
        args = parser.parse_args([command])
        assert args.command == command


def test_main_inspect_prints_json_and_returns_zero() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["inspect"])
    assert code == 0
    payload = json.loads(buffer.getvalue())
    assert "assets" in payload and "wadi" in payload["assets"]


def test_main_all_dry_runs_end_to_end(tmp_path) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["all", "--dry", "--output-dir", str(tmp_path / "cli")])
    assert code == 0
    payload = json.loads(buffer.getvalue())
    assert payload["dry_run"] is True
    assert Path(payload["outputs"]["csv"]).is_file()

"""One-command HydroJEV experiments with honest, provenance-tagged reporting.

This CLI ties the whole pipeline together and, crucially, keeps the paper's
empirical claims *separated by what was actually reproduced*:

* **Real benchmark data (BATADAL).** The reconstruction detector is trained and
  threshold-calibrated on the attack-free ``dataset03`` and evaluated on the
  labelled ``dataset04``. The recall it reports is a genuine cross-dataset
  anomaly-detection result on a named benchmark, tagged ``dataset='BATADAL:...'``.
* **Simulation (synthetic orthogonal-residual scenarios).** The closed-loop
  cognitive-reflex recall, isolation delay, and energy mitigation are measured on
  synthetic defence-zone states that carry HydroJEV's orthogonal residual
  evidence, arbitrated by the evidence-only mock. They are tagged
  ``dataset='simulation'`` and are **never** presented as BATADAL/Net3 benchmark
  numbers.
* **WADI.** Absent from ``refCode`` and therefore reported ``not_run`` with the
  exact missing-data reason -- never fabricated.

Every result row carries ``target``, ``observed``, ``unit``, ``status``,
``dataset``, ``mode``, ``samples`` and ``provenance`` so a reader can see what
each number is and is not. Latency is reported as p50/p95/p99 and compared to the
150 ms research target *and* the minimum pipe wave-propagation time, with the
standing caveat that the cloud loop is advisory, not the water-hammer relay.

The command runs fully offline: the Jev source is the deterministic mock unless a
process-only ``TYPESAFE_API_KEY`` and network are present (never stored here).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field, replace as dataclasses_replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig, load_config
from hydrojev.arbiter.decision_primitives import JevSource, ThreatCause
from hydrojev.datasets.wdn_loader import (
    DatasetUnavailableError,
    classify_batadal_columns,
    load_batadal,
    load_network,
)
from hydrojev.simulation.closed_loop_eval import (
    EnergyTrace,
    ReflexController,
    ReflexPolicy,
    ScenarioRun,
    distill_evidence,
    energy_mitigation,
    recall_summary,
    summarize_latency,
)
from hydrojev.simulation.mock_jev_server import decide_locally

# --------------------------------------------------------------------------- #
# Run modes
# --------------------------------------------------------------------------- #

_ATTACK_KINDS = ("adversarial", "fs_fdi", "burst")
_ALL_KINDS = _ATTACK_KINDS + ("benign",)
_CATEGORY_OF_KIND = {
    "adversarial": ThreatCause.ADVERSARIAL_AE_EVASION.value,
    "fs_fdi": ThreatCause.HYDRAULIC_FS_FDI.value,
    "burst": ThreatCause.PHYSICAL_BURST.value,
    "benign": ThreatCause.NORMAL_TRANSIENT.value,
}


@dataclass(frozen=True)
class RunMode:
    """Bounds that make a run either a fast smoke or a fuller experiment.

    ``dry`` short-circuits all heavy compute (training, solving, scenario
    sweeps): only asset inspection and the manifest are produced, so the wiring
    can be exercised in CI without a GPU or minutes of runtime.
    """

    name: str
    dry: bool = False
    batadal_train_rows: int | None = None
    ae_epochs: int = 40
    scenarios_per_category: int = 25
    latency_iterations: int = 200
    concealment_budget: int = 200
    cluster_net3: bool = True

    @classmethod
    def resolve(cls, *, dry: bool, quick: bool) -> "RunMode":
        if dry:
            return cls(name="dry", dry=True)
        if quick:
            return cls(
                name="quick",
                batadal_train_rows=400,
                ae_epochs=4,
                scenarios_per_category=4,
                latency_iterations=30,
                concealment_budget=40,
                cluster_net3=False,
            )
        return cls(name="full")


# --------------------------------------------------------------------------- #
# Result rows
# --------------------------------------------------------------------------- #

_ROW_FIELDS = (
    "metric",
    "dataset",
    "mode",
    "target",
    "observed",
    "unit",
    "status",
    "samples",
    "provenance",
    "detail",
)


@dataclass(frozen=True)
class ResultRow:
    """One reported measurement, honest about what it is and is not."""

    metric: str
    dataset: str
    mode: str
    target: float | None
    observed: float | None
    unit: str
    status: str  # "pass" | "fail" | "not_run" | "measured"
    samples: int | None = None
    provenance: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "dataset": self.dataset,
            "mode": self.mode,
            "target": self.target,
            "observed": self.observed,
            "unit": self.unit,
            "status": self.status,
            "samples": self.samples,
            "provenance": self.provenance,
            "detail": self.detail,
        }


def _status_against_target(
    observed: float | None, target: float | None, *, direction: str
) -> str:
    """``direction='min'``: observed must be >= target. ``'max'``: <= target."""

    if observed is None or target is None or not np.isfinite(observed):
        return "measured"
    if direction == "min":
        return "pass" if observed >= target else "fail"
    if direction == "max":
        return "pass" if observed <= target else "fail"
    raise ValueError("direction must be 'min' or 'max'")


# --------------------------------------------------------------------------- #
# Asset inspection and manifest
# --------------------------------------------------------------------------- #


def _sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_record(path: Path, *, hash_file: bool) -> dict[str, Any]:
    exists = path.is_file()
    record: dict[str, Any] = {"path": str(path), "available": exists}
    if exists:
        record["size_bytes"] = path.stat().st_size
        if hash_file:
            record["sha256"] = _sha256(path)
    return record


def inspect_assets(config: HydroJEVConfig, *, hash_files: bool = True) -> dict[str, Any]:
    """Report presence/size/hash of every local asset the experiments touch."""

    ref = config.paths.ref_code
    networks = {
        "Net1": ref / "FDI-WDNs" / "Net1.inp",
        "CTOWN": ref / "BATADAL" / "CTOWN.INP",
    }
    batadal = {
        "dataset03": ref / "BATADAL" / "BATADAL_dataset03.csv",
        "dataset04": ref / "BATADAL" / "BATADAL_dataset04.csv",
        "test_dataset": ref / "BATADAL" / "BATADAL_test_dataset.csv",
    }

    report: dict[str, Any] = {
        "networks": {name: _asset_record(p, hash_file=hash_files) for name, p in networks.items()},
        "batadal": {name: _asset_record(p, hash_file=hash_files) for name, p in batadal.items()},
    }

    # Net3 ships inside WNTR, not refCode; record it as library-provided.
    try:
        net3 = load_network("Net3", config=config)
        report["networks"]["Net3"] = {
            "path": str(net3.source_path),
            "available": True,
            "provider": "wntr-library",
            "junctions": len(net3.junctions),
            "links": len(net3.pipes) + len(net3.pumps) + len(net3.valves),
        }
    except Exception as exc:  # noqa: BLE001 - inspection must never crash the run
        report["networks"]["Net3"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    # WADI is a gated benchmark; record exactly why it cannot run.
    wadi_root = ref / "WADI"
    wadi_csvs = sorted(wadi_root.rglob("*.csv")) if wadi_root.is_dir() else []
    report["wadi"] = {
        "available": bool(wadi_csvs),
        "path": str(wadi_root),
        "reason": (
            "present" if wadi_csvs else f"no CSV files under {wadi_root} (gated benchmark not supplied)"
        ),
    }
    return report


def _package_versions() -> dict[str, str]:
    names = (
        "numpy", "pandas", "scipy", "torch", "wntr", "networkx",
        "httpx", "fastapi", "pulp", "gurobipy", "scikit-learn", "pyyaml",
    )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _torch_device_note() -> dict[str, str]:
    try:
        from hydrojev.physics.surrogate_models import select_torch_device

        selection = select_torch_device()
        return {"device": selection.device.type, "reason": selection.reason}
    except Exception as exc:  # noqa: BLE001
        return {"device": "unknown", "reason": f"{type(exc).__name__}: {exc}"}


def _solver_note(config: HydroJEVConfig) -> dict[str, Any]:
    try:
        from hydrojev.attacks.fs_fdi_milp import select_solver

        solver, name, attempts = select_solver(config.fs_fdi.solver_order)
        return {"selected": name, "available": solver is not None, "attempts": list(attempts)}
    except Exception as exc:  # noqa: BLE001
        return {"selected": None, "available": False, "reason": f"{type(exc).__name__}: {exc}"}


def build_manifest(config: HydroJEVConfig, *, mode: RunMode, hash_files: bool = True) -> dict[str, Any]:
    """Assemble a reproducibility manifest: hashes, versions, seeds, providers."""

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode.name,
        "seed": config.experiments.seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "torch_device": _torch_device_note(),
        "solver": _solver_note(config),
        "jev": {
            "endpoint": config.jev.endpoint,
            "model": config.jev.model,
            "source_offline_default": JevSource.MOCK.value,
        },
        "assets": inspect_assets(config, hash_files=hash_files),
    }


# --------------------------------------------------------------------------- #
# CPDZ topology / clustering
# --------------------------------------------------------------------------- #


def run_cluster(config: HydroJEVConfig, network: str, *, partition: bool = True) -> dict[str, Any]:
    """Load a network and (optionally) partition it into CPDZs."""

    from hydrojev.physics.cpdz_clustering import partition_network
    from hydrojev.physics.distance_metrics import build_physical_distances

    bundle = load_network(network, config=config)
    topology = {
        "network": bundle.name,
        "source_path": str(bundle.source_path),
        "junctions": len(bundle.junctions),
        "reservoirs": len(bundle.reservoirs),
        "tanks": len(bundle.tanks),
        "pipes": len(bundle.pipes),
        "pumps": len(bundle.pumps),
        "valves": len(bundle.valves),
        "protected_nodes": len(bundle.protected_nodes),
        "headloss_model": bundle.headloss_model,
        "hydraulic_timestep_s": bundle.hydraulic_timestep_s,
    }
    if not partition:
        topology["partition"] = "topology_only"
        return topology

    distances = build_physical_distances(
        bundle,
        alpha=config.clustering.alpha,
        beta=config.clustering.beta,
        acoustic_speed_m_s=config.clustering.acoustic_speed_m_s,
        default_flow_m3_s=config.clustering.default_flow_m3_s,
        darcy_friction_factor=config.clustering.darcy_friction_factor,
    )
    result = partition_network(bundle, distances, threshold=config.clustering.threshold)
    topology["partition"] = {
        "threshold": config.clustering.threshold,
        "clusters": len(result.clusters),
        "boundary_links": len(result.boundary_links),
        "merges": len(result.merge_history),
        "protected_nodes": len(result.protected_nodes),
    }
    return topology


def _min_wave_propagation_time_s(config: HydroJEVConfig, network: str = "Net1") -> float | None:
    """Minimum pipe L/a over a real network, for the water-hammer latency caveat."""

    try:
        bundle = load_network(network, config=config)
    except Exception:  # noqa: BLE001
        return None
    speed = config.clustering.acoustic_speed_m_s
    times = []
    for pipe_name in bundle.pipes:
        length = float(getattr(bundle.model.get_link(pipe_name), "length", 0.0) or 0.0)
        if length > 0.0:
            times.append(length / speed)
    return min(times) if times else None


# --------------------------------------------------------------------------- #
# BATADAL reconstruction detector (real benchmark data)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectorArtifacts:
    detector: Any
    feature_columns: tuple[str, ...]
    train_dataset: str
    best_epoch: int
    threshold: float


def _feature_matrix(frame, columns: Sequence[str], *, rows: int | None = None) -> np.ndarray:
    subset = frame if rows is None else frame.iloc[:rows]
    return subset[list(columns)].to_numpy(dtype=float)


def build_reconstruction_detector(config: HydroJEVConfig, *, mode: RunMode) -> DetectorArtifacts:
    """Train the reconstruction detector on attack-free BATADAL ``dataset03``."""

    from hydrojev.attacks.concealment_ae import train_reconstruction_detector

    clean = load_batadal("dataset03", config=config)
    columns = clean.feature_columns
    train = _feature_matrix(clean.train_frame, columns, rows=mode.batadal_train_rows)
    validation = _feature_matrix(clean.validation_frame, columns, rows=mode.batadal_train_rows)
    result = train_reconstruction_detector(
        train,
        validation,
        threshold_quantile=config.concealment.threshold_quantile,
        max_epochs=mode.ae_epochs,
        seed=config.experiments.seed,
    )
    return DetectorArtifacts(
        detector=result.detector,
        feature_columns=tuple(columns),
        train_dataset=clean.name,
        best_epoch=result.best_epoch,
        threshold=result.detector.threshold,
    )


def _contiguous_runs(flags: np.ndarray, target: int) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of maximal runs equal to ``target``."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(flags):
        if int(value) == target and start is None:
            start = index
        elif int(value) != target and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def batadal_detector_recall(
    artifacts: DetectorArtifacts, config: HydroJEVConfig, *, evaluate_variant: str = "dataset04"
) -> dict[str, Any]:
    """Cross-dataset detection recall on labelled BATADAL attack scenarios.

    A scenario is a maximal contiguous run of the attack flag; it is *detected*
    if the reconstruction score crosses the train-calibrated threshold on at
    least one of its samples. Point-wise recall and the benign false-positive
    rate are reported alongside, all on real benchmark data.
    """

    labelled = load_batadal(evaluate_variant, config=config)
    if labelled.label_column is None:
        return {"status": "not_run", "reason": f"{labelled.name} has no attack-label column"}
    columns = list(artifacts.feature_columns)
    missing = [c for c in columns if c not in labelled.frame.columns]
    if missing:
        return {"status": "not_run", "reason": f"feature columns absent in {labelled.name}: {missing[:3]}"}

    features = labelled.frame[columns].to_numpy(dtype=float)
    labels = labelled.frame[labelled.label_column].to_numpy().astype(int)
    scores = artifacts.detector.score_samples(features)
    alerting = scores >= artifacts.detector.threshold

    attack_runs = _contiguous_runs(labels, 1)
    benign_runs = _contiguous_runs(labels, 0)
    detected = sum(1 for (a, b) in attack_runs if bool(alerting[a:b].any()))
    benign_tripped = sum(1 for (a, b) in benign_runs if bool(alerting[a:b].any()))

    point_tp = int(np.sum(alerting & (labels == 1)))
    point_fp = int(np.sum(alerting & (labels == 0)))
    n_attack_points = int(np.sum(labels == 1))
    n_benign_points = int(np.sum(labels == 0))

    scenario_recall = detected / len(attack_runs) if attack_runs else float("nan")
    return {
        "status": "ok",
        "evaluate_dataset": labelled.name,
        "train_dataset": artifacts.train_dataset,
        "attack_scenarios": len(attack_runs),
        "benign_scenarios": len(benign_runs),
        "scenarios_detected": detected,
        "scenario_recall": scenario_recall,
        "benign_scenarios_tripped": benign_tripped,
        "benign_scenario_fp_rate": benign_tripped / len(benign_runs) if benign_runs else float("nan"),
        "point_recall": point_tp / n_attack_points if n_attack_points else float("nan"),
        "point_false_positive_rate": point_fp / n_benign_points if n_benign_points else float("nan"),
        "threshold": artifacts.threshold,
        "n_samples": int(len(labels)),
    }


def run_train_ae(config: HydroJEVConfig, *, mode: RunMode) -> dict[str, Any]:
    """Train the detector and report a JSON summary (no model object)."""

    artifacts = build_reconstruction_detector(config, mode=mode)
    return {
        "train_dataset": artifacts.train_dataset,
        "features": len(artifacts.feature_columns),
        "best_epoch": artifacts.best_epoch,
        "threshold": artifacts.threshold,
        "train_rows_cap": mode.batadal_train_rows,
        "epochs_cap": mode.ae_epochs,
    }


# --------------------------------------------------------------------------- #
# Constrained concealment attack (real benchmark data)
# --------------------------------------------------------------------------- #


def run_attack_concealment(
    config: HydroJEVConfig, *, mode: RunMode, artifacts: DetectorArtifacts | None = None
) -> dict[str, Any]:
    """Run white-box coordinate-descent concealment on the most anomalous row."""

    from hydrojev.attacks.concealment_ae import coordinate_descent_concealment

    if artifacts is None:
        artifacts = build_reconstruction_detector(config, mode=mode)

    labelled = load_batadal("dataset04", config=config)
    columns = list(artifacts.feature_columns)
    features = labelled.frame[columns].to_numpy(dtype=float)
    scores = artifacts.detector.score_samples(features)
    sample = features[int(np.argmax(scores))]

    # Continuous sensors are freely modifiable; discrete pump/valve *status*
    # channels are held immutable (their spoofing is postprocessed, not searched).
    groups = classify_batadal_columns(columns)
    status_columns = set(groups.get("pump_status", ())) | set(groups.get("valve_status", ()))
    modifiable = np.array([col not in status_columns for col in columns], dtype=bool)

    result = coordinate_descent_concealment(
        artifacts.detector,
        sample,
        modifiable_mask=modifiable,
        max_modified_features=config.concealment.max_modified_features,
        candidate_values=config.concealment.candidate_values,
        budget=mode.concealment_budget,
        patience=config.concealment.patience,
    )
    return {
        "dataset": labelled.name,
        "original_score": result.original_score,
        "adversarial_score": result.adversarial_score,
        "threshold": result.threshold,
        "changed_indices": list(result.changed_indices),
        "n_changed": len(result.changed_indices),
        "query_count": result.query_count,
        "converged": result.converged,
        "termination_reason": result.termination_reason,
        "modifiable_features": int(modifiable.sum()),
    }


# --------------------------------------------------------------------------- #
# Localized FS-FDI MILP (bounded Net1 smoke)
# --------------------------------------------------------------------------- #


def _net1_fs_fdi_problem(config: HydroJEVConfig):
    """Build a well-posed, hydraulically consistent localized FS-FDI zone.

    The instance is a serial defence-zone chain ``R -> J1 -> J2 -> J3`` whose
    pipes carry the *real geometry* (length/diameter/roughness) of Net1's first
    pipes. The base operating point is made hydraulically self-consistent so the
    MILP has a feasible starting state to perturb:

    * Base flows follow nodal conservation for a uniform per-junction demand: the
      terminal pipe carries one demand, the next two, the source pipe all three.
      (A naive constant per-pipe flow makes mass balance unsatisfiable at the
      chain's leaf, which is what forces a spurious ``infeasible``.)
    * Base heads are computed from the *same* piecewise-linear Hazen-Williams law
      the MILP enforces, so ``head_up - head_down == headloss(base_flow)`` holds
      exactly at the base point and the perceived-state energy constraints are
      feasible within the head-deviation envelope.

    This is a localized zone using Net1 pipe physics, not a full Net1 hydraulic
    solve; it is labelled as such in the provenance.
    """

    from hydrojev.attacks.fs_fdi_milp import FSFDIProblem, PipeSpec, hazen_williams_pwl

    bundle = load_network("Net1", config=config)
    model = bundle.model
    n_pipes = 3
    demand_per_junction = 0.01  # m^3/s drawn at each junction

    junctions = tuple(f"J{i}" for i in range(1, n_pipes + 1))
    source = "R"
    nodes = (source,) + junctions  # R -> J1 -> J2 -> J3

    pipes: list[PipeSpec] = []
    for i, pipe_name in enumerate(bundle.pipes[:n_pipes]):
        link = model.get_link(pipe_name)
        # Conservation on a serial chain: pipe i (from node i) carries the total
        # demand of every junction at or below it.
        base_flow = demand_per_junction * (n_pipes - i)
        pipes.append(
            PipeSpec(
                name=pipe_name,
                start=nodes[i],
                end=nodes[i + 1],
                length_m=float(getattr(link, "length", 100.0) or 100.0),
                diameter_m=float(getattr(link, "diameter", 0.3) or 0.3),
                roughness_c=float(getattr(link, "roughness", 130.0) or 130.0),
                base_flow_m3_s=base_flow,
                max_flow_m3_s=base_flow * 3.0,
            )
        )

    # Base heads on the PWL energy curve the MILP uses, propagated from the source.
    source_head = 100.0
    base_head: dict[str, float] = {source: source_head}
    running = source_head
    for pipe in pipes:
        pwl = hazen_williams_pwl(
            pipe.length_m, pipe.diameter_m, pipe.roughness_c,
            pipe.max_flow_m3_s, config.fs_fdi.piecewise_segments,
        )
        running -= pwl.evaluate(pipe.base_flow_m3_s)  # head drops along flow
        base_head[pipe.end] = running

    # Every quantity is a compromised sensor (flow per pipe + head per junction +
    # demand per junction = 3*n measurements): the most *conservative* threat
    # model, because the attacker must hold ALL of them mutually consistent. On a
    # serial chain this matters: raising the head-pipe flow forces the perceived
    # demand up (mass balance) and drags the ENTIRE downstream head cascade with
    # it (energy equalities), so the minimal hydraulically-consistent attack is
    # not 1 sensor but a coordinated set. The budget must admit that cascade or
    # the only feasible "attack" is the vacuous zero-perturbation optimum; the
    # actual stealth/impact trade-off is reported as a sparsity frontier by the
    # caller rather than hidden behind an arbitrary cap here.
    n_measurements = 3 * len(pipes)
    return FSFDIProblem(
        junctions=junctions,
        pipes=tuple(pipes),
        boundary_heads_m={source: source_head},
        base_head_m=base_head,
        base_demand_m3_s={j: demand_per_junction for j in junctions},
        measured_head_junctions=junctions,
        impact_pipes=(pipes[0].name,),  # inflate perceived source-pipe flow
        max_changed_measurements=n_measurements,
        n_segments=config.fs_fdi.piecewise_segments,
    )


def run_attack_fs_fdi(config: HydroJEVConfig) -> dict[str, Any]:
    """Solve the bounded Net1 FS-FDI MILP and report solver provenance.

    Reports two things the paper needs kept distinct:

    * the *headline* attack at the full compromised-sensor budget — the strongest
      hydraulically-consistent, CUSUM-subthreshold false-data injection the MILP
      can find, and which constraint (physical deviation bound vs sparsity cap)
      actually limits it; and
    * the *sparsity frontier* — the maximum stealthy impact achievable at each
      attacker sensor budget ``k``. This quantifies the attacker's stealth/impact
      trade-off and exposes the smallest budget that admits any attack at all
      (below it the only consistent "attack" on a fully-instrumented serial zone
      is the vacuous zero perturbation).
    """

    from hydrojev.attacks.fs_fdi_milp import solve_fs_fdi

    problem = _net1_fs_fdi_problem(config)
    impact_pipe = problem.impact_pipes[0]
    base_impact_flow = next(p.base_flow_m3_s for p in problem.pipes if p.name == impact_pipe)
    max_stealth_fraction = problem.max_flow_fraction  # physical deviation ceiling

    # The impact is a physical flow deviation (~1e-2 m^3/s), so the tie-breaking
    # sparsity penalty must be several orders below that scale or it cancels the
    # objective and yields a spurious zero-attack optimum. It only breaks ties
    # toward sparser attacks; it must never make a genuine attack unprofitable.
    def _solve_at_budget(max_changed: int):
        bounded = dataclasses_replace(problem, max_changed_measurements=max_changed)
        return solve_fs_fdi(
            bounded,
            solver_order=config.fs_fdi.solver_order,
            impact_direction={impact_pipe: 1.0},
            sparsity_penalty=1e-6,
            time_limit_s=config.fs_fdi.solver_time_limit_s,
        )

    def _increase(result) -> float | None:
        attacked = result.attacked_flows_m3_s.get(impact_pipe)
        return attacked - base_impact_flow if attacked is not None else None

    # Headline attack: full compromised-sensor budget (the instance sets it to the
    # measurement count), so the limiting factor is physical plausibility, not the
    # cap.
    result = _solve_at_budget(problem.max_changed_measurements)
    flow_increase = _increase(result)

    # Sparsity frontier: strongest stealthy impact per attacker sensor budget.
    frontier: list[dict[str, Any]] = []
    min_stealthy_budget: int | None = None
    for budget in range(1, problem.max_changed_measurements + 1):
        r = _solve_at_budget(budget)
        inc = _increase(r)
        accepted = bool(r.verification.get("accepted", False))
        nonzero = inc is not None and abs(inc) > 1e-9
        if nonzero and accepted and min_stealthy_budget is None:
            min_stealthy_budget = budget
        frontier.append(
            {
                "sensor_budget": budget,
                "impact_flow_increase_m3_s": _round(inc) if inc is not None else None,
                "impact_flow_increase_fraction": (
                    _round(inc / base_impact_flow) if inc is not None and base_impact_flow else None
                ),
                "n_changed": len(r.changed_measurements),
                "verification_accepted": accepted,
                "stealth_ok": bool(r.verification.get("stealth_ok", False)),
            }
        )

    # Which constraint binds the headline attack?
    if flow_increase is not None and base_impact_flow:
        at_physical_ceiling = abs(flow_increase / base_impact_flow - max_stealth_fraction) <= 1e-3
        binding_constraint = (
            "physical_deviation_bound" if at_physical_ceiling
            else "sparsity_cap" if len(result.changed_measurements) >= problem.max_changed_measurements
            else "milp_optimum"
        )
    else:
        binding_constraint = "no_attack"

    return {
        "status": result.status.value,
        "solver": result.solver,
        "objective": result.objective,
        "impact_pipe": impact_pipe,
        "impact_flow_increase_m3_s": _round(flow_increase) if flow_increase is not None else None,
        "impact_flow_increase_fraction": (
            _round(flow_increase / base_impact_flow)
            if flow_increase is not None and base_impact_flow else None
        ),
        "binding_constraint": binding_constraint,
        "max_stealth_fraction": max_stealth_fraction,
        "changed_measurements": list(result.changed_measurements),
        "n_changed": len(result.changed_measurements),
        "sensor_budget": problem.max_changed_measurements,
        "min_stealthy_sensor_budget": min_stealthy_budget,
        "sparsity_frontier": frontier,
        "verification_accepted": bool(result.verification.get("accepted", False)),
        "verification": {k: v for k, v in result.verification.items()},
        "instance": (
            "localized serial defence-zone chain (R->J1->J2->J3) built from Net1 "
            "pipe geometry with a hydraulically consistent base operating point; "
            "fully instrumented (flow+head+demand), so a consistent attack must "
            "co-falsify the downstream head cascade"
        ),
        "provenance": dict(result.provenance),
    }


# --------------------------------------------------------------------------- #
# Closed-loop simulation (synthetic orthogonal-residual scenarios)
# --------------------------------------------------------------------------- #


def _zone_state(kind: str, rng: random.Random) -> dict[str, Any]:
    """A JSON-safe defence-zone state carrying evidence only -- no label.

    The intensities are drawn from honest ranges so the evidence-only mock
    arbiter produces the intended cause with an air-gap probability above the
    safety gate; the closed-loop controller (not this function) decides isolation.
    Stealthy cyber scenarios keep mass balance within tolerance and energy below
    the reflex's severe threshold, so they exercise the *cognitive* air gap.
    """

    tol = 0.5
    if kind == "adversarial":
        detectors = {
            "cpdz_residual_normalized": "alerting",
            "rat_covariance_distortion": "alerting",
            "gepfm_baseline_drift": "alerting",
            "reconstruction_autoencoder": "nominal",
        }
        within, residual, energy = True, rng.uniform(0.02, 0.10), rng.uniform(0.25, 0.45)
    elif kind == "fs_fdi":
        detectors = {
            "cpdz_residual_normalized": "alerting",
            "state_estimator_residual": "nominal",
            "gepfm_baseline_drift": "alerting",
            "rat_covariance_distortion": "alerting",
        }
        within, residual, energy = True, rng.uniform(0.02, 0.10), rng.uniform(0.30, 0.45)
    elif kind == "burst":
        detectors = {
            "cpdz_residual_normalized": "alerting",
            "reconstruction_autoencoder": "alerting",
        }
        within, residual, energy = False, rng.uniform(4.0, 8.0) * tol, rng.uniform(0.05, 0.15)
    elif kind == "benign":
        detectors = {
            "cpdz_residual_normalized": "nominal",
            "reconstruction_autoencoder": "nominal",
        }
        within, residual, energy = True, rng.uniform(0.0, 0.10), 0.0
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown scenario kind {kind!r}")

    return {
        "freshness": {"is_fresh": True, "state_age_s": 0.5},
        "detectors": {name: {"state": state} for name, state in detectors.items()},
        "hydraulic_consistency": {
            "within_tolerance": within,
            "mass_balance_residual_m3_s": residual,
            "mass_balance_tolerance_m3_s": tol,
            "extra_energy_fraction": energy,
        },
    }


def _scenario_run(
    kind: str, index: int, *, policy: ReflexPolicy, rng: random.Random, steps: int = 4
) -> tuple[ScenarioRun, float]:
    """Arbitrate an evidence-only scenario and run the reflex controller.

    Returns the scored run plus the mean extra-energy fraction observed, so the
    energy counterfactual can be tied to the simulated attack intensity.
    """

    controller = ReflexController(policy)
    state = _zone_state(kind, rng)
    energy = float(state["hydraulic_consistency"]["extra_energy_fraction"])
    pairs = []
    for _ in range(steps):
        decision = decide_locally(state, source=JevSource.MOCK)
        evidence = distill_evidence(state, policy=policy)
        pairs.append((decision, evidence))
    records = tuple(controller.run(pairs))
    run = ScenarioRun(
        scenario_id=f"{kind}-{index}",
        category=_CATEGORY_OF_KIND[kind],
        is_attack=kind != "benign",
        onset_step=0,
        step_interval_s=900.0,
        records=records,
    )
    return run, energy


def simulate_closed_loop(config: HydroJEVConfig, *, mode: RunMode) -> dict[str, Any]:
    """Run the synthetic cognitive-reflex scenarios and score them honestly."""

    policy = ReflexPolicy.from_safety_config(config.safety)
    rng = random.Random(config.experiments.seed)
    runs: list[ScenarioRun] = []
    attack_energies: list[float] = []
    for kind in _ALL_KINDS:
        for index in range(mode.scenarios_per_category):
            run, energy = _scenario_run(kind, index, policy=policy, rng=rng)
            runs.append(run)
            if kind in ("adversarial", "fs_fdi"):
                attack_energies.append(energy)

    detection = recall_summary(runs)
    source_counts: dict[str, int] = {}
    for run in runs:
        for record in run.records:
            source_counts[record.jev_source] = source_counts.get(record.jev_source, 0) + 1
    mean_attack_energy = float(np.mean(attack_energies)) if attack_energies else 0.0
    return {
        "detection": detection,
        "jev_source_counts": source_counts,
        "n_runs": len(runs),
        "mean_attack_extra_energy_fraction": mean_attack_energy,
    }


def simulate_energy_mitigation(
    config: HydroJEVConfig, *, attacked_extra_fraction: float
) -> dict[str, Any]:
    """Counterfactual pump energy on one aligned horizon (illustrative).

    The attacked over-pumping fraction is tied to the simulated stealthy-attack
    intensity; the defended residual is the configured tolerated increase, so the
    mitigation is not an arbitrary number.
    """

    times = np.linspace(0.0, 3600.0, 13)
    flow = np.full_like(times, 0.05)
    base_head = 30.0
    residual = config.experiments.maximum_energy_increase_fraction
    baseline = EnergyTrace(times, flow, np.full_like(times, base_head))
    attacked = EnergyTrace(times, flow, np.full_like(times, base_head * (1.0 + attacked_extra_fraction)))
    defended = EnergyTrace(times, flow, np.full_like(times, base_head * (1.0 + residual)))
    return energy_mitigation(baseline, attacked, defended, efficiency=0.75)


# --------------------------------------------------------------------------- #
# Latency accounting
# --------------------------------------------------------------------------- #


def measure_latency(config: HydroJEVConfig, *, mode: RunMode) -> dict[str, Any]:
    """Measure surrogate + residual + mock-Jev per-step latency, then summarize.

    The surrogate architecture is the compact GRU of the paper; its inference
    latency is weight-independent, so an untrained instance of the correct shape
    is timed. The Jev stage is the deterministic mock (offline); live latency is
    a separate, network-bound measurement and is not claimed here.
    """

    from hydrojev.physics.surrogate_models import GRUSurrogate, select_torch_device
    from hydrojev.state_projection.residual_engine import normalized_cpdz_residual, robust_feature_scale

    n_features = 43  # BATADAL sensor width; the surrogate mirrors the observed vector
    seq_len = config.surrogate.sequence_length
    selection = select_torch_device()

    import torch

    surrogate = GRUSurrogate(
        input_size=n_features,
        output_size=n_features,
        hidden_size=config.surrogate.hidden_size,
        num_layers=config.surrogate.num_layers,
        dense_size=config.surrogate.dense_size,
    ).to(selection.device)
    surrogate.eval()
    window = torch.zeros((1, seq_len, n_features), dtype=torch.float32, device=selection.device)

    rng = np.random.default_rng(config.experiments.seed)
    baseline = rng.normal(size=(32, n_features))
    scale = robust_feature_scale(baseline)
    observed = rng.normal(size=n_features)
    predicted = rng.normal(size=n_features)

    iterations = mode.latency_iterations
    warmup = min(5, iterations)
    with torch.inference_mode():
        for _ in range(warmup):
            surrogate(window)

    per_step: list[dict[str, float]] = []
    state = _zone_state("adversarial", random.Random(config.experiments.seed))
    with torch.inference_mode():
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            surrogate(window)
            if selection.device.type == "cuda":
                torch.cuda.synchronize(selection.device)
            t1 = time.perf_counter_ns()
            normalized_cpdz_residual(observed, predicted, scale)
            t2 = time.perf_counter_ns()
            decide_locally(state, source=JevSource.MOCK)
            t3 = time.perf_counter_ns()
            per_step.append(
                {
                    "surrogate_ms": (t1 - t0) / 1e6,
                    "residual_ms": (t2 - t1) / 1e6,
                    "jev_decision_ms": (t3 - t2) / 1e6,
                }
            )

    min_wave = _min_wave_propagation_time_s(config, "Net1")
    summary = summarize_latency(
        per_step,
        target_ms=config.experiments.maximum_total_latency_ms,
        minimum_wave_propagation_time_s=min_wave,
    )
    summary["device"] = selection.device.type
    summary["jev_source"] = JevSource.MOCK.value
    return summary


# --------------------------------------------------------------------------- #
# Row assembly and evaluation
# --------------------------------------------------------------------------- #


def _evaluation_rows(
    config: HydroJEVConfig,
    *,
    mode: RunMode,
    simulation: Mapping[str, Any],
    batadal: Mapping[str, Any],
    energy: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> list[ResultRow]:
    rows: list[ResultRow] = []
    recall_target = config.experiments.recall_target
    energy_target = config.experiments.maximum_energy_increase_fraction
    latency_target = config.experiments.maximum_total_latency_ms

    detection = simulation["detection"]
    sim_provenance = "mock arbiter; synthetic orthogonal-residual scenarios (evidence-only)"
    rows.append(
        ResultRow(
            metric="closed_loop_recall",
            dataset="simulation",
            mode=mode.name,
            target=recall_target,
            observed=_finite(detection.get("recall")),
            unit="fraction",
            status=_status_against_target(_finite(detection.get("recall")), recall_target, direction="min"),
            samples=detection.get("attack_scenarios"),
            provenance=sim_provenance,
            detail=f"wilson95={detection.get('recall_wilson_95')}",
        )
    )
    rows.append(
        ResultRow(
            metric="closed_loop_false_positive_rate",
            dataset="simulation",
            mode=mode.name,
            target=0.0,
            observed=_finite(detection.get("false_positive_rate")),
            unit="fraction",
            status=_status_against_target(
                _finite(detection.get("false_positive_rate")), 0.0, direction="max"
            ),
            samples=detection.get("benign_scenarios"),
            provenance=sim_provenance,
        )
    )
    isolation = detection.get("attack_to_isolation_s", {})
    for pct in ("p50", "p95"):
        rows.append(
            ResultRow(
                metric=f"attack_to_isolation_{pct}",
                dataset="simulation",
                mode=mode.name,
                target=None,
                observed=_finite(isolation.get(pct)),
                unit="s",
                status="measured",
                samples=isolation.get("count"),
                provenance=sim_provenance,
            )
        )

    # Real BATADAL benchmark recall.
    if batadal.get("status") == "ok":
        rows.append(
            ResultRow(
                metric="reconstruction_detector_scenario_recall",
                dataset=f"{batadal['train_dataset']}->{batadal['evaluate_dataset']}",
                mode=mode.name,
                target=recall_target,
                observed=_finite(batadal.get("scenario_recall")),
                unit="fraction",
                status=_status_against_target(
                    _finite(batadal.get("scenario_recall")), recall_target, direction="min"
                ),
                samples=batadal.get("attack_scenarios"),
                provenance="reconstruction_autoencoder; cross-dataset BATADAL evaluation (offline)",
                detail=(
                    f"point_recall={_round(batadal.get('point_recall'))} "
                    f"point_fpr={_round(batadal.get('point_false_positive_rate'))}"
                ),
            )
        )
    else:
        rows.append(
            ResultRow(
                metric="reconstruction_detector_scenario_recall",
                dataset="BATADAL",
                mode=mode.name,
                target=recall_target,
                observed=None,
                unit="fraction",
                status="not_run",
                provenance="reconstruction_autoencoder",
                detail=str(batadal.get("reason", "unavailable")),
            )
        )

    # Energy mitigation (simulation counterfactual).
    rows.append(
        ResultRow(
            metric="energy_increase_defended",
            dataset="simulation",
            mode=mode.name,
            target=energy_target,
            observed=_finite(energy.get("residual_extra_fraction")),
            unit="fraction",
            status=_status_against_target(
                _finite(energy.get("residual_extra_fraction")), energy_target, direction="max"
            ),
            samples=None,
            provenance="pump-energy counterfactual on one aligned horizon (illustrative)",
            detail=(
                f"attacked={_round(energy.get('extra_energy_fraction_attacked'))} "
                f"mitigated={_round(energy.get('mitigated_fraction'))}"
            ),
        )
    )

    # Latency rows.
    total = latency.get("total_ms", {})
    for pct, direction, target in (("p50", None, None), ("p95", "max", latency_target), ("p99", None, None)):
        rows.append(
            ResultRow(
                metric=f"total_loop_latency_{pct}",
                dataset="simulation",
                mode=mode.name,
                target=target,
                observed=_finite(total.get(pct)),
                unit="ms",
                status=(
                    _status_against_target(_finite(total.get(pct)), target, direction=direction)
                    if direction
                    else "measured"
                ),
                samples=latency.get("count"),
                provenance=f"surrogate+residual+mock-Jev; device={latency.get('device')}",
                detail=(
                    "cloud loop is advisory, NOT the water-hammer relay; "
                    f"min_wave_ms={_round(latency.get('minimum_wave_propagation_ms'))}"
                ),
            )
        )
    for stage in ("surrogate_ms", "jev_decision_ms"):
        stage_stats = latency.get("per_stage_ms", {}).get(stage, {})
        rows.append(
            ResultRow(
                metric=f"{stage}_p95",
                dataset="simulation",
                mode=mode.name,
                target=None,
                observed=_finite(stage_stats.get("p95")),
                unit="ms",
                status="measured",
                samples=stage_stats.get("count"),
                provenance=f"{'compact GRU surrogate (weight-independent latency)' if stage.startswith('surrogate') else 'mock Jev (offline)'}",
            )
        )

    # WADI: gated benchmark, explicitly not run.
    rows.append(
        ResultRow(
            metric="wadi_benchmark",
            dataset="WADI",
            mode=mode.name,
            target=recall_target,
            observed=None,
            unit="fraction",
            status="not_run",
            provenance="none",
            detail="WADI CSVs absent from refCode/WADI (gated benchmark not supplied)",
        )
    )
    return rows


def run_evaluate(config: HydroJEVConfig, *, mode: RunMode) -> dict[str, Any]:
    """Full offline evaluation: simulation + real BATADAL + energy + latency."""

    simulation = simulate_closed_loop(config, mode=mode)
    artifacts = build_reconstruction_detector(config, mode=mode)
    batadal = batadal_detector_recall(artifacts, config)
    energy = simulate_energy_mitigation(
        config, attacked_extra_fraction=simulation["mean_attack_extra_energy_fraction"]
    )
    latency = measure_latency(config, mode=mode)
    rows = _evaluation_rows(
        config, mode=mode, simulation=simulation, batadal=batadal, energy=energy, latency=latency
    )
    return {
        "simulation": simulation,
        "batadal": batadal,
        "energy": energy,
        "latency": latency,
        "rows": [row.to_dict() for row in rows],
    }


# --------------------------------------------------------------------------- #
# Output tables
# --------------------------------------------------------------------------- #


def write_result_tables(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, str]:
    """Write the result rows to both CSV and JSON with a fixed schema."""

    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    json_path = out_dir / "results.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in _ROW_FIELDS})
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, indent=2, default=str)
    return {"csv": str(csv_path), "json": str(json_path)}


def run_all(config: HydroJEVConfig, *, mode: RunMode, output_dir: Path | None = None) -> dict[str, Any]:
    """Inspect assets, build a manifest, run every experiment, and write tables."""

    out_dir = output_dir or (config.paths.artifacts / f"run_{mode.name}")
    manifest = build_manifest(config, mode=mode)
    report: dict[str, Any] = {"manifest": manifest, "mode": mode.name}

    if mode.dry:
        report["dry_run"] = True
        report["planned"] = [
            "cluster(Net1, Net3)", "train-ae(dataset03)", "attack-concealment(dataset04)",
            "attack-fs-fdi(Net1)", "evaluate(simulation + BATADAL + energy + latency)",
        ]
        report["outputs"] = write_result_tables([], out_dir)
        return report

    report["cluster"] = {
        "Net1": run_cluster(config, "Net1", partition=True),
        "Net3": run_cluster(config, "Net3", partition=mode.cluster_net3),
    }
    artifacts = build_reconstruction_detector(config, mode=mode)
    report["train_ae"] = {
        "train_dataset": artifacts.train_dataset,
        "features": len(artifacts.feature_columns),
        "best_epoch": artifacts.best_epoch,
        "threshold": artifacts.threshold,
    }
    report["attack_concealment"] = run_attack_concealment(config, mode=mode, artifacts=artifacts)
    try:
        report["attack_fs_fdi"] = run_attack_fs_fdi(config)
    except Exception as exc:  # noqa: BLE001 - solver availability varies by machine
        report["attack_fs_fdi"] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    evaluation = run_evaluate(config, mode=mode)
    report["evaluate"] = {k: v for k, v in evaluation.items() if k != "rows"}
    report["rows"] = evaluation["rows"]
    report["outputs"] = write_result_tables(evaluation["rows"], out_dir)
    return report


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _round(value: Any, digits: int = 4) -> Any:
    number = _finite(value)
    return round(number, digits) if number is not None else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_experiments",
        description="Reproducible, provenance-tagged HydroJEV experiments (offline by default).",
    )
    parser.add_argument(
        "command",
        choices=("inspect", "cluster", "train-ae", "attack-concealment", "attack-fs-fdi", "evaluate", "all"),
    )
    parser.add_argument("--config", default=None, help="optional YAML config override path")
    parser.add_argument("--network", default="Net1", help="network for the cluster command")
    parser.add_argument("--output-dir", default=None, help="directory for result tables (all command)")
    parser.add_argument("--quick", action="store_true", help="bounded fast run")
    parser.add_argument("--dry", action="store_true", help="wiring smoke: no heavy compute")
    return parser


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    mode = RunMode.resolve(dry=args.dry, quick=args.quick)
    command = args.command

    if command == "inspect":
        return {"mode": mode.name, "assets": inspect_assets(config)}
    if command == "cluster":
        return {"mode": mode.name, "cluster": run_cluster(config, args.network, partition=not mode.dry)}
    if mode.dry and command in ("train-ae", "attack-concealment", "attack-fs-fdi", "evaluate"):
        return {"mode": mode.name, "dry_run": True, "command": command, "note": "skipped heavy compute"}
    if command == "train-ae":
        return {"mode": mode.name, "train_ae": run_train_ae(config, mode=mode)}
    if command == "attack-concealment":
        return {"mode": mode.name, "attack_concealment": run_attack_concealment(config, mode=mode)}
    if command == "attack-fs-fdi":
        return {"mode": mode.name, "attack_fs_fdi": run_attack_fs_fdi(config)}
    if command == "evaluate":
        evaluation = run_evaluate(config, mode=mode)
        out_dir = Path(args.output_dir) if args.output_dir else (config.paths.artifacts / f"eval_{mode.name}")
        evaluation["outputs"] = write_result_tables(evaluation["rows"], out_dir)
        return {"mode": mode.name, "evaluate": {k: v for k, v in evaluation.items() if k != "rows"},
                "rows": evaluation["rows"], "outputs": evaluation["outputs"]}
    if command == "all":
        out_dir = Path(args.output_dir) if args.output_dir else None
        return run_all(config, mode=mode, output_dir=out_dir)
    raise ValueError(f"unknown command {command!r}")  # pragma: no cover


@contextlib.contextmanager
def _stdout_noise_to_stderr():
    """Route OS-level stdout (fd 1) to stderr for the duration.

    Some third-party backends write directly to file descriptor 1 -- notably
    gurobipy's restricted-license banner -- which would otherwise corrupt the
    machine-readable JSON this CLI emits on stdout. We redirect that low-level
    noise to stderr during the compute phase and keep stdout pure for the report.
    The redirection is restored unconditionally; if fd duplication is unavailable
    (e.g. stdout is not a real file descriptor under a capture harness), we simply
    run without redirecting so behaviour is never worse than before.
    """

    try:
        sys.stdout.flush()
        saved_fd = os.dup(1)
    except (OSError, ValueError):  # no real fd 1 to duplicate (e.g. captured stdout)
        yield
        return
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Keep stdout pure JSON: solver/library banners are routed to stderr while the
    # report is computed, then the JSON is written to the restored stdout.
    with _stdout_noise_to_stderr():
        result = _dispatch(args)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

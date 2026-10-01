"""Leakage-safe BATADAL benchmark for the standalone Jev-driven HydroJEV detector.

Detector scores are fitted on attack-free BATADAL training data.  Jev states are
assembled without ATT_FLAG, attack IDs or expected causes.  Labels are consumed
only after inference to assemble episode metrics.  The live path invokes the
repository PowerShell wrapper and never falls back to a mock response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks.detection_baselines import contiguous_runs, evaluate_scores, fit_isolation_forest_detector
from hydrojev.benchmarks.named_family_benchmark import NamedFamilyScores, TrainedNamedFamilies, score_named_families, train_named_families
from hydrojev.config import HydroJEVConfig, load_config
from hydrojev.datasets.wdn_loader import load_batadal
from hydrojev.methods.hydrojev_detector import (
    DETECTION_QUESTION_IDS,
    DetectionEpisode,
    HydroJEVDetector,
    build_detection_request,
    build_detection_state,
    deterministic_no_jev,
    episode_metrics,
    parse_detection_response,
)
from hydrojev.run_experiments import RunMode

DETECTOR_ORDER = (
    "Mahalanobis", "PCA residual", "Isolation Forest", "HydroJEV AE",
    "CPDZ residual", "GEpFM drift", "RAT covariance",
)
TEMPORAL_DETECTORS = {"GEpFM drift", "RAT covariance"}
STATE_WINDOW_SAMPLES = 24


@dataclass(frozen=True)
class FittedEvidence:
    trained_named: TrainedNamedFamilies
    isolation_forest: Any
    thresholds: dict[str, float]
    columns: tuple[str, ...]
    train_rows: int
    threshold_quantile: float
    training_source: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
    return value


def fit_evidence_model(config: HydroJEVConfig, *, seed: int = 42, gru_epochs: int = 12, ae_epochs: int = 12) -> FittedEvidence | dict[str, Any]:
    mode = RunMode(name="hydrojev_live_benchmark", batadal_train_rows=None, ae_epochs=int(ae_epochs), scenarios_per_category=1, latency_iterations=1, concealment_budget=1, cluster_net3=False)
    trained = train_named_families(config, mode=mode, seed=int(seed), max_epochs=int(gru_epochs))
    if isinstance(trained, dict): return trained
    clean = load_batadal("dataset03", config=config)
    columns = tuple(clean.feature_columns)
    train_values = clean.train_frame[list(columns)].to_numpy(dtype=float)
    isolation = fit_isolation_forest_detector(train_values, threshold_quantile=float(config.concealment.threshold_quantile), seed=int(seed))
    thresholds = dict(trained.thresholds)
    thresholds["Isolation Forest"] = float(isolation.threshold)
    return FittedEvidence(trained, isolation, {k: float(v) for k, v in thresholds.items()}, columns, int(len(train_values)), float(config.concealment.threshold_quantile), clean.name)


def _score_variant(fitted: FittedEvidence, config: HydroJEVConfig, variant: str) -> dict[str, Any]:
    bundle = load_batadal(variant, config=config)
    named = score_named_families(fitted.trained_named, config, evaluate_variant=variant)
    if isinstance(named, dict): return named
    raw = bundle.frame[list(fitted.columns)].to_numpy(dtype=float)
    start = int(named.start_index)
    scores = {str(k): np.asarray(v, dtype=float) for k, v in named.scores.items()}
    scores["Isolation Forest"] = np.asarray(fitted.isolation_forest.score_samples(raw[start:]), dtype=float)
    labels = bundle.frame[bundle.label_column].to_numpy(dtype=int)[start:]
    timestamps = bundle.frame[bundle.timestamp_column].iloc[start:].astype(str).to_numpy()
    if len(labels) != len(next(iter(scores.values()))): raise ValueError("score alignment failed")
    return {"variant": variant, "dataset": bundle.name, "scores": scores, "labels": labels, "timestamps": timestamps, "start_index": start, "sample_interval_s": float(bundle.sample_interval_s), "n_samples": int(len(labels)), "attack_scenarios": len(contiguous_runs(labels, 1)), "benign_scenarios": len(contiguous_runs(labels, 0))}


def baseline_rows(scored: Mapping[str, Any], fitted: FittedEvidence, *, detector_names: Iterable[str] = DETECTOR_ORDER) -> list[dict[str, Any]]:
    labels = np.asarray(scored["labels"], dtype=int)
    rows: list[dict[str, Any]] = []
    for name in detector_names:
        if name not in scored["scores"] or name not in fitted.thresholds: continue
        metrics = evaluate_scores(np.asarray(scored["scores"][name], dtype=float), float(fitted.thresholds[name]), labels)
        rows.append({"dataset": scored["variant"], "detector": name, "threshold": float(fitted.thresholds[name]), "provenance": "fit on attack-free dataset03; no labels used in fitting", **_json_safe(metrics)})
    return rows


def _margin(values: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0 or not np.isfinite(threshold): return np.full(np.asarray(values).shape, np.nan)
    return np.asarray(values, dtype=float) / threshold - 1.0


def _candidate(scored: Mapping[str, Any], fitted: FittedEvidence, start: int, end: int) -> bool:
    for name, threshold in fitted.thresholds.items():
        if name in scored["scores"] and np.any(np.isfinite(scored["scores"][name][start:end]) & (scored["scores"][name][start:end] >= threshold)): return True
    return False


def _episode_state(scored: Mapping[str, Any], fitted: FittedEvidence, start: int, end: int, *, candidate: bool, included: set[str] | None = None) -> dict[str, Any]:
    included = included or set(fitted.thresholds)
    detectors: dict[str, Any] = {}
    windows: dict[str, Any] = {}
    margins: list[float] = []
    for name in DETECTOR_ORDER:
        if name not in included or name not in scored["scores"]: continue
        values = np.asarray(scored["scores"][name][start:end], dtype=float)
        finite = values[np.isfinite(values)]
        threshold = float(fitted.thresholds[name])
        if finite.size == 0:
            detectors[name] = {"state": "unavailable", "value": "unavailable", "threshold": threshold}
            continue
        normalized = _margin(finite, threshold)
        margins.extend(normalized.tolist())
        fraction = float(np.mean(finite >= threshold))
        detectors[name] = {"state": "alerting" if bool(np.any(finite >= threshold)) else "nominal", "value": float(np.max(finite)), "threshold": threshold, "normalized_margin": float(np.max(normalized)), "summary": {"maximum": float(np.max(finite)), "p95": float(np.quantile(finite, .95)), "mean": float(np.mean(finite)), "alert_fraction": fraction}}
        windows[name] = {"p95_score": float(np.quantile(finite, .95)), "alert_fraction": fraction}
    max_margin = max(margins) if margins else float("nan")
    return build_detection_state(timestamp=str(scored["timestamps"][max(start, end - 1)]), freshness={"is_fresh": True, "state_age_s": 0.0}, detectors=detectors, temporal_summary=windows, hydraulic_consistency={"state": "unavailable", "within_tolerance": "unavailable", "reason": "BATADAL has SCADA observations but no demand or flow-balance truth"}, routing={"candidate": bool(candidate), "candidate_rule": "at least one detector crosses its clean threshold", "max_normalized_margin": float(max_margin) if np.isfinite(max_margin) else "unavailable", "alerting_detector_count": sum(v.get("state") == "alerting" for v in detectors.values())}, provenance={"observations": "BATADAL SCADA telemetry", "training_source": fitted.training_source, "label_fields_removed": True, "scenario_identifier_removed": True, "hydraulic_truth": "unavailable"})


def _canonical_request(request: Mapping[str, Any]) -> bytes:
    return json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _call_live(state: Mapping[str, Any], *, output_dir: Path, dataset: str, episode_index: int, mode: str, model: str = "jev-latest", reuse_existing: bool = False) -> tuple[Any | None, dict[str, Any]]:
    request = build_detection_request(state, model=model)
    raw_dir = output_dir / "raw" / dataset / mode
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_index:03d}"
    request_path, response_path = raw_dir / f"{stem}.request.json", raw_dir / f"{stem}.response.json"
    request_bytes = _canonical_request(request)
    request_path.write_bytes(request_bytes)
    record: dict[str, Any] = {"dataset": dataset, "episode_index": int(episode_index), "request_sha256": hashlib.sha256(request_bytes).hexdigest(), "request_path": str(request_path), "response_path": str(response_path), "status": "failed"}
    script = Path(__file__).resolve().parents[2] / "tools" / "jev" / "invoke-jev.ps1"
    # pwsh ships the Microsoft.PowerShell.Security module in this workstation's
    # bundled runtime; Windows PowerShell 5 may fail to autoload it from a
    # non-interactive Python child process.
    command = ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-RequestPath", str(request_path), "-OutputPath", str(response_path)]
    if reuse_existing and response_path.is_file():
        record.update({"replayed": True, "returncode": 0, "stdout": "", "stderr": "", "wrapper_elapsed_seconds": 0.0})
    else:
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        record["wrapper_elapsed_seconds"] = time.perf_counter() - started
        record["returncode"] = int(completed.returncode)
        record["stdout"] = completed.stdout[-1000:]
        record["stderr"] = completed.stderr[-2000:]
        if completed.returncode != 0 or not response_path.is_file():
            record["failure"] = "jev_wrapper_failed"
            return None, record
    try:
        wrapper = json.loads(response_path.read_text(encoding="utf-8-sig"))
        decision = parse_detection_response(wrapper["response"], source="live", timing_ms={"elapsed_seconds": float(wrapper.get("elapsed_seconds", record["wrapper_elapsed_seconds"]))})
        record.update({"status": "ok", "model": decision.model, "elapsed_seconds": float(wrapper.get("elapsed_seconds", record["wrapper_elapsed_seconds"])), "attempts": int(wrapper.get("attempts", 1)), "usage": _json_safe(decision.usage)})
        return decision, record
    except Exception as exc:  # noqa: BLE001
        record["failure"] = f"response_schema_error:{type(exc).__name__}"
        return None, record


def _episode_variant(scored: Mapping[str, Any], fitted: FittedEvidence, *, output_dir: Path, mode: str, live: bool, detector: HydroJEVDetector, included: set[str] | None = None, reuse_existing: bool = False) -> dict[str, Any]:
    labels = np.asarray(scored["labels"], dtype=int)
    all_runs = [(a, b, 1) for a, b in contiguous_runs(labels, 1)] + [(a, b, 0) for a, b in contiguous_runs(labels, 0)]
    all_runs.sort(key=lambda item: item[0])
    episodes: list[DetectionEpisode] = []
    rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for index, (start, end, label) in enumerate(all_runs):
        # The Jev state is a causal onset window rather than a label-sized
        # summary.  This prevents a benign episode from being contaminated by
        # unrelated operational changes many hours later.
        state_end = min(end, start + STATE_WINDOW_SAMPLES)
        candidate = _candidate(scored, fitted, start, state_end)
        call_candidate = candidate if mode == "candidate_gate" else True
        state = _episode_state(scored, fitted, start, state_end, candidate=call_candidate, included=included)
        record: dict[str, Any] = {"dataset": scored["variant"], "episode_index": index, "start_index": int(start), "end_index": int(end), "state_end_index": int(state_end), "duration_samples": int(end - start), "state_window_samples": int(state_end - start), "label_internal": int(label), "candidate_internal": bool(candidate), "mode": mode}
        if mode == "no_jev":
            result = deterministic_no_jev(state); record.update({"source": "no_jev", "status": "ok"})
        elif not call_candidate:
            result = detector.predict_response(state, None, source="no_call"); record.update({"source": "no_call", "status": "not_called"})
        elif not live:
            result = detector.failure_result(state, reason="live_not_run"); record.update({"source": "live", "status": "not_run"})
        else:
            decision, live_record = _call_live(state, output_dir=output_dir, dataset=scored["variant"], episode_index=index, mode=mode, reuse_existing=reuse_existing)
            record.update(live_record)
            result = detector.failure_result(state, reason=record.get("failure", "live_failure")) if decision is None else detector.predict_decision(state, decision)
        episodes.append(DetectionEpisode(f"{scored['variant']}_{index:03d}", bool(label), (result,), onset_index=0, step_interval_s=float(scored["sample_interval_s"])))
        rows.append({"dataset": scored["variant"], "episode_index": index, "label_internal": int(label), "candidate_internal": bool(candidate), "state": getattr(result.state, "value", str(result.state)), "source": str(result.source), "event_type": result.event_type, "alarm_probability": result.alarm_probability, "abstain_reason": result.abstain_reason})
        calls.append(record)
    return {"dataset": scored["variant"], "mode": mode, "live_requested": bool(live), "episode_metrics": _json_safe(episode_metrics(episodes)), "episode_rows": _json_safe(rows), "call_records": _json_safe(calls), "n_episodes": len(episodes)}


def run_benchmark(config: HydroJEVConfig, *, output_dir: Path, live: bool, seed: int = 42, gru_epochs: int = 12, ae_epochs: int = 12, replay_raw: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted = fit_evidence_model(config, seed=seed, gru_epochs=gru_epochs, ae_epochs=ae_epochs)
    if isinstance(fitted, dict): return fitted
    scored = {variant: _score_variant(fitted, config, variant) for variant in ("dataset04", "test_dataset")}
    detector = HydroJEVDetector(alarm_threshold=.5, normal_threshold=.2)
    baselines = [row for item in scored.values() for row in baseline_rows(item, fitted)]
    events: list[dict[str, Any]] = []
    for item in scored.values():
        events.append(_episode_variant(item, fitted, output_dir=output_dir, mode="live_full", live=live, detector=detector, reuse_existing=replay_raw))
        events.append(_episode_variant(item, fitted, output_dir=output_dir, mode="candidate_gate", live=live and item["variant"] == "dataset04", detector=detector, reuse_existing=replay_raw))
        events.append(_episode_variant(item, fitted, output_dir=output_dir, mode="no_jev", live=False, detector=detector))
        events.append(_episode_variant(item, fitted, output_dir=output_dir, mode="no_temporal", live=False, detector=detector, included=set(fitted.thresholds) - TEMPORAL_DETECTORS, reuse_existing=False))
    result = {"status": "ok", "protocol": {"training_dataset": "dataset03 (attack-free)", "development_dataset": "dataset04 (labels used after inference for episode scoring)", "held_out_dataset": "test_dataset (labels used after inference for final scoring)", "label_leakage_guard": "ATT_FLAG, attack IDs and expected causes are absent from every Jev state", "hydraulic_truth": "unavailable in BATADAL; no synthetic mass-balance values inserted", "live_source": "TypeSafe Jev through tools/jev/invoke-jev.ps1" if live else "not_run", "model_alias": "jev-latest", "question_ids": list(DETECTION_QUESTION_IDS)}, "fit": {"seed": int(seed), "train_rows": fitted.train_rows, "columns": list(fitted.columns), "threshold_quantile": fitted.threshold_quantile, "thresholds": fitted.thresholds, "source": fitted.training_source, "gru_epochs": int(gru_epochs), "ae_epochs": int(ae_epochs)}, "baseline_rows": baselines, "event_results": events, "raw_directory": str(output_dir / "raw")}
    (output_dir / "results.json").write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="artifacts/hydrojev_live_benchmark")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gru-epochs", type=int, default=12)
    parser.add_argument("--ae-epochs", type=int, default=12)
    parser.add_argument("--replay", action="store_true", help="reuse previously saved live raw responses; never calls the API")
    args = parser.parse_args(argv)
    result = run_benchmark(load_config(), output_dir=Path(args.outdir), live=bool(args.live), seed=args.seed, gru_epochs=args.gru_epochs, ae_epochs=args.ae_epochs, replay_raw=bool(args.replay))
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_main())

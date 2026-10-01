"""Causal, label-free grid benchmark for the standalone HydroJEV detector.

This module never builds a request from an attack-sized label segment. It fits
the front end on the 7,008-row attack-free portion of BATADAL dataset03 and
evaluates a fixed six-hour causal grid beginning after the 71-hour warm-up
needed by the surrogate and covariance streams. Labels are read only after a
response is produced, then forward-held to hourly samples for metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks.batadal_score import batadal_score
from hydrojev.benchmarks.detection_baselines import contiguous_runs, evaluate_scores, fit_isolation_forest_detector
from hydrojev.benchmarks.named_family_benchmark import score_named_families, train_named_families
from hydrojev.config import HydroJEVConfig, load_config
from hydrojev.datasets.wdn_loader import load_batadal
from hydrojev.methods.hydrojev_detector import HydroJEVDetector, build_detection_request, build_detection_state, deterministic_no_jev, parse_detection_response
from hydrojev.run_experiments import RunMode

DETECTOR_ORDER = ("Mahalanobis", "PCA residual", "Isolation Forest", "HydroJEV AE", "CPDZ residual", "GEpFM drift", "RAT covariance")
TEMPORAL = {"GEpFM drift", "RAT covariance"}
INSTANTANEOUS = {"Mahalanobis", "PCA residual", "Isolation Forest", "HydroJEV AE"}
WARMUP = 71
STRIDE = 6
MIN_CANDIDATE_ALERTS = 3


def _safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, Path): return str(v)
    if isinstance(v, dict): return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_safe(x) for x in v]
    return v


@dataclass(frozen=True)
class FrontEnd:
    trained: Any
    isolation: Any
    thresholds: dict[str, float]
    columns: tuple[str, ...]
    train_rows: int
    threshold_quantile: float


def fit_front_end(config: HydroJEVConfig, *, seed: int = 42, gru_epochs: int = 12, ae_epochs: int = 12) -> FrontEnd | dict[str, Any]:
    clean = load_batadal("dataset03", config=config)
    mode = RunMode(name="hydrojev_detection_v2", batadal_train_rows=None, ae_epochs=ae_epochs, scenarios_per_category=1, latency_iterations=1, concealment_budget=1, cluster_net3=False)
    trained = train_named_families(config, mode=mode, seed=seed, max_epochs=gru_epochs)
    if isinstance(trained, dict): return trained
    columns = tuple(clean.feature_columns)
    train_values = clean.train_frame[list(columns)].to_numpy(dtype=float)
    isolation = fit_isolation_forest_detector(train_values, threshold_quantile=float(config.concealment.threshold_quantile), seed=seed)
    thresholds = dict(trained.thresholds); thresholds["Isolation Forest"] = float(isolation.threshold)
    return FrontEnd(trained, isolation, {k: float(v) for k, v in thresholds.items()}, columns, len(train_values), float(config.concealment.threshold_quantile))


def score_stream(front: FrontEnd, config: HydroJEVConfig, variant: str) -> dict[str, Any]:
    bundle = load_batadal(variant, config=config)
    named = score_named_families(front.trained, config, evaluate_variant=variant)
    if isinstance(named, dict): return named
    raw = bundle.frame[list(front.columns)].to_numpy(dtype=float)
    scores = {str(k): np.asarray(v, dtype=float) for k, v in named.scores.items()}
    scores["Isolation Forest"] = front.isolation.score_samples(raw[int(named.start_index):])
    labels = bundle.frame[bundle.label_column].to_numpy(dtype=int)[int(named.start_index):]
    return {"variant": variant, "bundle": bundle, "scores": scores, "labels": labels, "start": int(named.start_index), "interval_s": float(bundle.sample_interval_s), "n_raw": len(bundle.frame)}


def _state_for_grid(stream: Mapping[str, Any], front: FrontEnd, raw_index: int, *, included: set[str] | None = None, candidate: bool = True) -> dict[str, Any]:
    included = included or set(front.thresholds)
    start = int(stream["start"]); j = raw_index - start
    left = max(0, j - STRIDE + 1); right = j + 1
    detectors: dict[str, Any] = {}; temporal: dict[str, Any] = {}
    alert_count = 0
    for name in DETECTOR_ORDER:
        if name not in included or name not in stream["scores"]: continue
        values = np.asarray(stream["scores"][name][left:right], dtype=float)
        finite = values[np.isfinite(values)]
        thr = front.thresholds[name]
        if finite.size == 0:
            detectors[name] = {"state": "unavailable", "value": "unavailable", "threshold": thr}; continue
        margin = finite / thr - 1.0
        is_alert = bool(np.any(finite >= thr)); alert_count += int(is_alert)
        detectors[name] = {"state": "alerting" if is_alert else "nominal", "value": float(finite[-1]), "threshold": thr, "normalized_margin": float(np.max(margin)), "summary": {"window_max": float(np.max(finite)), "window_p95": float(np.quantile(finite, .95)), "alert_fraction": float(np.mean(finite >= thr))}}
        temporal[name] = {"window_max": float(np.max(finite)), "alert_fraction": float(np.mean(finite >= thr))}
    return build_detection_state(timestamp="current_causal_window", freshness={"is_fresh": True, "state_age_s": 0.0}, detectors=detectors, temporal_summary=temporal, hydraulic_consistency={"state": "unavailable", "within_tolerance": "unavailable", "reason": "BATADAL does not provide demand or complete mass-balance truth"}, routing={"candidate": bool(candidate), "candidate_rule": f"at least {MIN_CANDIDATE_ALERTS} independent detector streams alert in the causal six-hour window", "alerting_detector_count": int(alert_count), "raw_index_external": "omitted from state"}, provenance={"observation_source": "BATADAL SCADA", "labels_removed": True, "scenario_ids_removed": True, "fixed_causal_window_samples": STRIDE})


def _call_live(state: Mapping[str, Any], *, outdir: Path, variant: str, grid_index: int, reuse_existing: bool = False) -> tuple[Any | None, dict[str, Any]]:
    rawdir = outdir / "raw" / variant / "full_live"; rawdir.mkdir(parents=True, exist_ok=True)
    request = build_detection_request(state, model="jev-latest")
    body = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    req = rawdir / f"grid_{grid_index:05d}.request.json"; resp = rawdir / f"grid_{grid_index:05d}.response.json"
    req.write_bytes(body)
    record: dict[str, Any] = {"variant": variant, "grid_index": grid_index, "request_sha256": hashlib.sha256(body).hexdigest(), "request_path": str(req), "response_path": str(resp)}
    script = Path(__file__).resolve().parents[2] / "tools" / "jev" / "invoke-jev.ps1"
    cmd = ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-RequestPath", str(req), "-OutputPath", str(resp)]
    if reuse_existing and resp.is_file():
        record.update({"replayed": True, "returncode": 0, "wrapper_elapsed_seconds": 0.0, "stdout": "", "stderr": ""})
    else:
        t0 = time.perf_counter(); p = subprocess.run(cmd, capture_output=True, text=True, check=False); wrapper_s = time.perf_counter() - t0
        record.update({"returncode": int(p.returncode), "wrapper_elapsed_seconds": wrapper_s, "stdout": p.stdout[-500:], "stderr": p.stderr[-1000:]})
        if p.returncode != 0 or not resp.is_file(): record["status"] = "failed"; record["failure"] = "wrapper_failed"; return None, record
    try:
        wrapper = json.loads(resp.read_text(encoding="utf-8-sig")); elapsed_default = float(record.get("wrapper_elapsed_seconds", 0.0)); decision = parse_detection_response(wrapper["response"], source="live", timing_ms={"elapsed_seconds": float(wrapper.get("elapsed_seconds", elapsed_default))}, attempts=int(wrapper.get("attempts", 1)))
        record.update({"status": "ok", "model": decision.model, "elapsed_seconds": float(wrapper.get("elapsed_seconds", elapsed_default)), "attempts": int(wrapper.get("attempts", 1)), "usage": decision.usage})
        return decision, record
    except Exception as exc: record["status"] = "failed"; record["failure"] = f"schema:{type(exc).__name__}"; return None, record


def _runs_metrics(labels: np.ndarray, alarms: np.ndarray, *, interval_s: float, start: int) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int); alarms = np.asarray(alarms, dtype=bool)
    attacks = contiguous_runs(labels, 1); benign = contiguous_runs(labels, 0)
    detected = 0; delays: list[float] = []; per_attack: list[dict[str, Any]] = []
    for a, b in attacks:
        hits = np.flatnonzero(alarms[a:b]); d = int(hits[0]) if hits.size else None
        detected += int(d is not None); delays.append(float(d * interval_s / 3600.0) if d is not None else float("nan")); per_attack.append({"start": int(a + start), "end": int(b + start), "ttd_hours": delays[-1] if d is not None else None, "detected": d is not None})
    benign_hit = sum(bool(np.any(alarms[a:b])) for a, b in benign)
    score = batadal_score(labels, alarms)
    return {"attack_scenarios": len(attacks), "benign_scenarios": len(benign), "scenario_recall": detected / len(attacks) if attacks else None, "benign_scenario_fp_rate": benign_hit / len(benign) if benign else None, "benign_hour_fpr": float(np.sum(alarms & (labels == 0)) / max(1, np.sum(labels == 0))), "mean_ttd_hours": float(np.nanmean(delays)) if any(np.isfinite(delays)) else None, "per_attack": per_attack, "batadal_score": score}


def run_variant(stream: Mapping[str, Any], front: FrontEnd, *, outdir: Path, live: bool, ablation: str = "full", candidate_only: bool = True, reuse_existing: bool = False) -> dict[str, Any]:
    included = set(front.thresholds)
    if ablation == "no_temporal": included -= TEMPORAL
    if ablation == "no_instantaneous": included -= INSTANTANEOUS
    detector = HydroJEVDetector(alarm_threshold=.5, normal_threshold=.2)
    start = int(stream["start"]); n = len(stream["labels"]); grid = np.arange(max(WARMUP, start), n + start, STRIDE, dtype=int)
    outputs: list[Any] = []; calls: list[dict[str, Any]] = []; probabilities: list[float | None] = []; event_types: list[str | None] = []; severities: list[float | None] = []; candidate_flags: list[bool] = []
    for raw_i in grid:
        scores_idx = raw_i - start; left = max(0, scores_idx - STRIDE + 1); right = scores_idx + 1; count = 0
        for name, thr in front.thresholds.items():
            if name not in included: continue
            vals = np.asarray(stream["scores"].get(name, [])[left:right], dtype=float); count += int(np.any(np.isfinite(vals) & (vals >= thr))) if vals.size else 0
        candidate = count >= MIN_CANDIDATE_ALERTS; candidate_flags.append(candidate)
        state = _state_for_grid(stream, front, raw_i, included=included, candidate=(candidate if candidate_only else True))
        if ablation == "no_jev" or (not live and ablation in {"no_temporal", "no_instantaneous"}):
            result = deterministic_no_jev(state)
            calls.append({"variant": stream["variant"], "raw_index": int(raw_i), "candidate": candidate, "status": "ok", "source": "no_jev", "ablation": ablation})
        elif not candidate_only or candidate:
            if live:
                decision, rec = _call_live(state, outdir=outdir, variant=stream["variant"], grid_index=int(raw_i), reuse_existing=reuse_existing); calls.append({**rec, "raw_index": int(raw_i), "candidate": candidate, "alerting_count": count}); result = detector.failure_result(state, reason=rec.get("failure", "live_failure")) if decision is None else detector.predict_decision(state, decision)
            else:
                result = detector.failure_result(state, reason="live_not_run"); calls.append({"variant": stream["variant"], "raw_index": int(raw_i), "candidate": candidate, "status": "not_run"})
        else:
            result = detector.predict_response(state, None, source="no_call"); calls.append({"variant": stream["variant"], "raw_index": int(raw_i), "candidate": False, "status": "not_called"})
        outputs.append(result); probabilities.append(result.alarm_probability); event_types.append(result.event_type); severities.append(result.severity_score)
    grid_labels = np.asarray([stream["labels"][i - start] for i in grid], dtype=int); grid_alarm = np.asarray([o.state.value == "alarm" for o in outputs], dtype=bool)
    held = np.zeros(n, dtype=bool); held_probs = np.full(n, np.nan, dtype=float)
    for k, raw_i in enumerate(grid):
        nxt = int(grid[k + 1]) if k + 1 < len(grid) else n + start; a, b = max(0, int(raw_i - start)), min(n, int(nxt - start)); held[a:b] = grid_alarm[k];
        if probabilities[k] is not None: held_probs[a:b] = float(probabilities[k])
    m = _runs_metrics(np.asarray(stream["labels"], dtype=int), held, interval_s=float(stream["interval_s"]), start=start)
    valid_prob = np.isfinite(held_probs); abstain = sum(o.state.value == "abstain" for o in outputs); status_counts: dict[str, int] = {}
    for o in outputs: status_counts[o.source] = status_counts.get(o.source, 0) + 1
    lat = [float(c["elapsed_seconds"]) for c in calls if c.get("status") == "ok" and c.get("elapsed_seconds") is not None]
    called = sum(c.get("status") == "ok" and c.get("candidate", False) for c in calls)
    brier = float(np.mean((np.asarray([p for p in probabilities if p is not None]) - np.asarray([y for p, y in zip(probabilities, grid_labels) if p is not None])) ** 2)) if any(p is not None for p in probabilities) else None
    m.update({"abstention_rate_grid": abstain / len(outputs) if outputs else None, "candidate_coverage_grid": float(np.mean(candidate_flags)) if candidate_flags else None, "candidate_attack_grid_recall": float(np.mean([c for c, y in zip(candidate_flags, grid_labels) if y == 1])) if np.any(grid_labels == 1) else None, "grid_points": len(grid), "called_points": called, "failed_points": sum(c.get("status") == "failed" for c in calls), "source_counts": status_counts, "latency_median_s": float(np.median(lat)) if lat else None, "latency_p95_s": float(np.percentile(lat, 95)) if lat else None, "tokens_input": int(sum(c.get("usage", {}).get("input_tokens", 0) for c in calls if isinstance(c.get("usage"), dict))), "tokens_output": int(sum(c.get("usage", {}).get("output_tokens", 0) for c in calls if isinstance(c.get("usage"), dict))), "grid_alarm_rate": float(np.mean(grid_alarm)) if len(grid_alarm) else None, "probability_coverage": float(np.mean(valid_prob)) if len(valid_prob) else None, "brier_score_available": brier, "event_type_counts": {str(k): event_types.count(k) for k in sorted(set(event_types), key=lambda x: str(x))}, "severity_mean_available": float(np.mean([s for s in severities if s is not None])) if any(s is not None for s in severities) else None})
    return {"variant": stream["variant"], "ablation": ablation, "candidate_only": candidate_only, "metrics": _safe(m), "grid": {"raw_indices": grid.tolist(), "labels_internal": grid_labels.tolist(), "alarm": grid_alarm.tolist(), "probabilities": [None if p is None else float(p) for p in probabilities], "event_types": event_types, "severity_scores": severities}, "calls": _safe(calls)}


def run_benchmark(config: HydroJEVConfig, *, outdir: Path, live: bool, seed: int = 42, gru_epochs: int = 12, ae_epochs: int = 12, replay: bool = False) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True); front = fit_front_end(config, seed=seed, gru_epochs=gru_epochs, ae_epochs=ae_epochs)
    if isinstance(front, dict): return front
    streams = {v: score_stream(front, config, v) for v in ("dataset04", "test_dataset")}; baselines: list[dict[str, Any]] = []
    for v, s in streams.items():
        for name, score in s["scores"].items():
            if name in front.thresholds: baselines.append({"dataset": v, "detector": name, **_safe(evaluate_scores(score, front.thresholds[name], s["labels"]))})
    variants: list[dict[str, Any]] = []
    for v, s in streams.items():
        variants.append(run_variant(s, front, outdir=outdir, live=live, ablation="full", candidate_only=True, reuse_existing=replay)); variants.append(run_variant(s, front, outdir=outdir, live=False, ablation="no_jev", candidate_only=True)); variants.append(run_variant(s, front, outdir=outdir, live=False, ablation="no_temporal", candidate_only=True)); variants.append(run_variant(s, front, outdir=outdir, live=False, ablation="no_instantaneous", candidate_only=True)); variants.append(run_variant(s, front, outdir=outdir, live=False, ablation="full_all_grid", candidate_only=False))
    result = {"status": "ok", "protocol": {"training": "dataset03 attack-free 7,008 rows", "warmup_samples": WARMUP, "grid_stride_hours": STRIDE, "candidate_rule": f"at least {MIN_CANDIDATE_ALERTS} independent detectors in a causal six-hour window", "state_label_leakage": "ATT_FLAG, row IDs, attack IDs and expected causes excluded", "test_status": "retrospective fixed-protocol evaluation; this split was previously inspected during project audit", "live_source": "TypeSafe Jev via invoke-jev.ps1" if live else "not_run"}, "fit": {"seed": seed, "threshold_quantile": front.threshold_quantile, "thresholds": front.thresholds, "train_rows": front.train_rows, "columns": list(front.columns), "gru_epochs": gru_epochs, "ae_epochs": ae_epochs}, "baseline_rows": baselines, "variants": variants}
    (outdir / "results.json").write_text(json.dumps(_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"); return result


def _main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--outdir", default="artifacts/hydrojev_detection_v2"); p.add_argument("--live", action="store_true"); p.add_argument("--replay", action="store_true"); p.add_argument("--seed", type=int, default=42); p.add_argument("--gru-epochs", type=int, default=12); p.add_argument("--ae-epochs", type=int, default=12); a = p.parse_args(argv)
    result = run_benchmark(load_config(), outdir=Path(a.outdir), live=a.live, seed=a.seed, gru_epochs=a.gru_epochs, ae_epochs=a.ae_epochs, replay=a.replay); print(json.dumps(_safe(result), indent=2, ensure_ascii=False)); return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__": raise SystemExit(_main())

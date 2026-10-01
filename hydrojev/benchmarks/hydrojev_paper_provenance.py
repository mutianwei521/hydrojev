"""Build a compact, paper-facing summary from the formal HydroJEV benchmark.

The source of truth is artifacts/hydrojev_detection_v2/results.json and the raw
request/response files referenced by that result. This script deliberately does
not contain experiment numbers: metrics, counts, hashes, model versions,
latencies and token totals are derived from the artifacts on every run.

The output keeps the four estimand variants needed by the manuscript under
explicit dataset/ablation keys and omits the full_all_grid placeholder, which
was recorded as not_run rather than as a measured method.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "artifacts" / "hydrojev_detection_v2" / "results.json"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "hydrojev_detection_v2" / "paper_summary.json"

KEEP_ABLATIONS = ("full", "no_jev", "no_temporal", "no_instantaneous")
EXCLUDED_ABLATIONS = ("full_all_grid",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_recorded_path(project_root: Path, recorded: Any) -> Path | None:
    if not recorded:
        return None
    path = Path(str(recorded))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _numeric(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            output.append(number)
    return output


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _response_details(project_root: Path, call: Mapping[str, Any]) -> dict[str, Any]:
    """Read and verify one recorded raw response without exposing request state."""

    request_path = _resolve_recorded_path(project_root, call.get("request_path"))
    response_path = _resolve_recorded_path(project_root, call.get("response_path"))
    request_exists = bool(request_path and request_path.is_file())
    response_exists = bool(response_path and response_path.is_file())

    request_hash = _sha256(request_path) if request_exists and request_path else None
    response_hash = _sha256(response_path) if response_exists and response_path else None
    recorded_request_hash = call.get("request_sha256")
    request_hash_matches = bool(recorded_request_hash and request_hash and str(recorded_request_hash) == request_hash)

    wrapper: dict[str, Any] = {}
    response: dict[str, Any] = {}
    schema_valid = False
    if response_exists and response_path:
        try:
            wrapper = _json_read(response_path)
            response = wrapper.get("response", {})
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            schema_valid = (
                isinstance(wrapper, dict)
                and isinstance(response, dict)
                and isinstance(response.get("model"), str)
                and isinstance(response.get("answers"), dict)
                and isinstance(usage, dict)
            )
            if schema_valid:
                from hydrojev.methods.hydrojev_detector import parse_detection_response
                parse_detection_response(response, source="live", attempts=wrapper.get("attempts"))
                if set(response["answers"]) != {"alarm", "event_type", "severity"}:
                    raise ValueError("unexpected question keys")
                if not {"input_tokens", "output_tokens"}.issubset(usage):
                    raise ValueError("token fields missing")
                if not isinstance(wrapper.get("elapsed_seconds"), (int, float)) or not math.isfinite(wrapper["elapsed_seconds"]) or wrapper["elapsed_seconds"] < 0:
                    raise ValueError("invalid latency")
                if not isinstance(wrapper.get("timestamp_utc"), str):
                    raise ValueError("timestamp missing")
        except (OSError, ValueError, TypeError):
            schema_valid = False
            wrapper = {}
            response = {}

    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    model = response.get("model") if isinstance(response, dict) else None
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    elapsed = wrapper.get("elapsed_seconds") if isinstance(wrapper, dict) else None
    attempts = wrapper.get("attempts") if isinstance(wrapper, dict) else None
    timestamp = wrapper.get("timestamp_utc") if isinstance(wrapper, dict) else None

    return {
        "dataset": call.get("dataset"),
        "ablation": call.get("ablation", "full"),
        "raw_index": call.get("raw_index"),
        "candidate": call.get("candidate"),
        "status": call.get("status"),
        "source": "live" if response_exists else None,
        "replayed": bool(call.get("replayed", False)),
        "model": model,
        "elapsed_seconds": elapsed,
        "attempts": attempts,
        "timestamp_utc": timestamp,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_path": _relative(request_path, project_root) if request_path else None,
        "response_path": _relative(response_path, project_root) if response_path else None,
        "request_sha256": request_hash,
        "response_sha256": response_hash,
        "recorded_response_sha256": call.get("response_sha256"),
        "response_hash_matches": bool(call.get("response_sha256") and response_hash and call["response_sha256"] == response_hash),
        "response_hash_status": "missing_recorded_hash" if not call.get("response_sha256") else ("match" if response_hash == call.get("response_sha256") else "mismatch_or_missing_file"),
        "recorded_request_sha256": recorded_request_hash,
        "request_hash_matches": bool(request_hash_matches),
        "request_file_present": request_exists,
        "response_file_present": response_exists,
        "response_schema_valid": bool(schema_valid),
    }


def _summarize_calls(
    project_root: Path,
    calls: list[Mapping[str, Any]],
    *,
    dataset: str | None = None,
    ablation: str | None = None,
    include_live_records: bool = False,
) -> dict[str, Any]:
    status_counts = Counter(str(call.get("status")) for call in calls)
    def effective_source(call: Mapping[str, Any]) -> str:
        source = call.get("source")
        if source:
            return str(source)
        if call.get("response_path"):
            return "live"
        if call.get("status") == "not_called":
            return "not_called"
        if call.get("status") == "not_run":
            return "not_run"
        return "unlabelled"

    source_counts = Counter(effective_source(call) for call in calls)
    model_counts = Counter(str(call.get("model")) for call in calls if call.get("model") is not None)
    return {
        "record_count": len(calls),
        "status_counts": dict(sorted(status_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "model_counts": dict(sorted(model_counts.items())),
        "live_record_count": sum(1 for call in calls if call.get("response_path")),
        "records": [
            _response_details(
                project_root,
                {**call, "dataset": dataset or call.get("dataset"), "ablation": ablation or call.get("ablation")},
            )
            for call in calls
        ] if include_live_records else None,
    }


def _grid_summary(grid: Mapping[str, Any]) -> dict[str, Any]:
    raw_indices = list(grid.get("raw_indices", []))
    labels = list(grid.get("labels_internal", []))
    alarms = list(grid.get("alarm", []))
    probabilities = list(grid.get("probabilities", []))
    event_types = list(grid.get("event_types", []))
    severity_scores = list(grid.get("severity_scores", []))
    return {
        "grid_points": len(raw_indices),
        "raw_index_first": raw_indices[0] if raw_indices else None,
        "raw_index_last": raw_indices[-1] if raw_indices else None,
        "positive_label_points": sum(1 for value in labels if value == 1),
        "alarm_points": sum(1 for value in alarms if bool(value)),
        "probability_available_points": sum(1 for value in probabilities if value is not None),
        "probability_missing_points": sum(1 for value in probabilities if value is None),
        "event_type_counts": dict(sorted(Counter(str(value) for value in event_types if value is not None).items())),
        "severity_available_points": sum(1 for value in severity_scores if value is not None),
    }


def _balanced_accuracy(metrics: Mapping[str, Any]) -> float | None:
    """Derive balanced accuracy from the retained BATADAL classification counts."""

    score = metrics.get("batadal_score", {})
    classification = score.get("classification", {}) if isinstance(score, Mapping) else {}
    tpr = classification.get("TPR")
    tnr = classification.get("TNR")
    if tpr is None or tnr is None:
        return None
    return (float(tpr) + float(tnr)) / 2.0


def _semantic_not_run(reason: str) -> dict[str, str]:
    return {"status": "not_run", "reason": reason}


def _baseline_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add stable names and derived fields without changing measured values."""

    detector = str(row.get("detector"))
    method_id = {
        "Mahalanobis": "mahalanobis",
        "PCA residual": "pca_residual",
        "Isolation Forest": "isolation_forest",
        "HydroJEV AE": "reconstruction_ae",
        "CPDZ residual": "cpdz",
        "GEpFM drift": "gepfm",
        "RAT covariance": "rat",
    }.get(detector, detector.lower().replace(" ", "_"))
    out = dict(row)
    out["method_id"] = method_id
    out["detector_name"] = "Reconstruction AE" if detector == "HydroJEV AE" else detector
    tpr = row.get("point_recall")
    fpr = row.get("point_false_positive_rate")
    out["balanced_accuracy"] = ((float(tpr) + 1.0 - float(fpr)) / 2.0
                                 if tpr is not None and fpr is not None else None)
    out["event_ttd_hours"] = {
        "status": "not_run",
        "reason": "baseline runner did not retain forward-held event time-to-detection",
    }
    return out


def _variant_summary(project_root: Path, variant: Mapping[str, Any]) -> dict[str, Any]:
    calls = list(variant.get("calls", []))
    live_records = [call for call in calls if call.get("response_path")]
    dataset = str(variant.get("variant"))
    ablation = str(variant.get("ablation"))
    metrics = dict(variant.get("metrics", {}))
    metrics["balanced_accuracy"] = _balanced_accuracy(metrics)
    metrics["semantic"] = {
        "cause_macro_f1": _semantic_not_run("BATADAL has no independent four-class cause labels"),
        "severity_mae": _semantic_not_run("BATADAL has no independent operational severity labels"),
        "calibration": _semantic_not_run("BATADAL has no independent semantic labels"),
    }
    if ablation in {"no_jev", "no_temporal", "no_instantaneous"}:
        metrics["semantic"] = {
            "cause_macro_f1": _semantic_not_run("no typed semantic output in deterministic no-Jev control"),
            "severity_mae": _semantic_not_run("no typed severity output in deterministic no-Jev control"),
            "calibration": _semantic_not_run("no typed semantic output in deterministic no-Jev control"),
        }
    return {
        "dataset": dataset,
        "ablation": ablation,
        "candidate_only": variant.get("candidate_only"),
        "metrics": metrics,
        "grid": _grid_summary(variant.get("grid", {})),
        "calls": _summarize_calls(
            project_root,
            calls,
            dataset=dataset,
            ablation=ablation,
            include_live_records=bool(live_records),
        ),
    }


def _live_provenance(project_root: Path, variants: list[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    raw_calls: list[dict[str, Any]] = []
    for variant in variants:
        for call in variant.get("calls", []):
            # A response path is the artifact-level definition of a raw live call.
            # No method result or label is used here.
            if call.get("response_path"):
                raw_calls.append(_response_details(
                    project_root,
                    {**call, "dataset": variant.get("variant"), "ablation": variant.get("ablation")},
                ))

    elapsed = _numeric(call.get("elapsed_seconds") for call in raw_calls)
    input_tokens = _numeric(call.get("input_tokens") for call in raw_calls)
    output_tokens = _numeric(call.get("output_tokens") for call in raw_calls)
    response_models = Counter(str(call["model"]) for call in raw_calls if call.get("model") is not None)
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted({str(call.get("dataset")) for call in raw_calls}):
        subset = [call for call in raw_calls if str(call.get("dataset")) == dataset]
        subset_elapsed = _numeric(call.get("elapsed_seconds") for call in subset)
        subset_input = _numeric(call.get("input_tokens") for call in subset)
        subset_output = _numeric(call.get("output_tokens") for call in subset)
        by_dataset[dataset] = {
            "raw_call_count": len(subset),
            "models": dict(sorted(Counter(str(call["model"]) for call in subset if call.get("model") is not None).items())),
            "latency_seconds": {
                "n": len(subset_elapsed),
                "median": statistics.median(subset_elapsed) if subset_elapsed else None,
                "p95": _percentile(subset_elapsed, 95.0),
            },
            "tokens": {
                "input": int(sum(subset_input)) if subset_input else 0,
                "output": int(sum(subset_output)) if subset_output else 0,
            },
        }

    request_state_forbidden = 0
    for call in raw_calls:
        request_path = _resolve_recorded_path(project_root, call.get("request_path"))
        if not request_path or not request_path.is_file():
            continue
        try:
            request = _json_read(request_path)
            state = request.get("state", {}) if isinstance(request, Mapping) else {}
            forbidden = {"ATT_FLAG", "attack_id", "attack_ids", "scenario_id", "scenario_ids",
                         "expected_cause", "ground_truth", "severity_label", "true_label", "label"}
            stack = [state]
            while stack:
                value = stack.pop()
                if isinstance(value, Mapping):
                    request_state_forbidden += sum(1 for key in value if str(key) in forbidden)
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
        except (OSError, ValueError, TypeError):
            request_state_forbidden += 1

    request_sha256_match_count = sum(1 for call in raw_calls if call.get("request_hash_matches"))
    request_file_count = sum(1 for call in raw_calls if call.get("request_file_present"))
    response_file_count = sum(1 for call in raw_calls if call.get("response_file_present"))
    schema_valid_count = sum(1 for call in raw_calls if call.get("response_schema_valid"))
    replayed_count = sum(1 for call in raw_calls if call.get("replayed"))

    return {
        "source": protocol.get("live_source"),
        "model_alias": "jev-latest",
        "raw_call_count": len(raw_calls),
        "successful_response_count": sum(1 for call in raw_calls if call.get("response_schema_valid")),
        "response_schema_failure_count": sum(1 for call in raw_calls if not call.get("response_schema_valid")),
        "actual_models": dict(sorted(response_models.items())),
        "latency_seconds": {
            "n": len(elapsed),
            "median": statistics.median(elapsed) if elapsed else None,
            "p95": _percentile(elapsed, 95.0),
            "min": min(elapsed) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
        "tokens": {
            "rows_with_input_tokens": len(input_tokens),
            "rows_with_output_tokens": len(output_tokens),
            "input": int(sum(input_tokens)) if input_tokens else 0,
            "output": int(sum(output_tokens)) if output_tokens else 0,
        },
        "hashes": {
            "algorithm": "SHA-256",
            "request_count": sum(1 for call in raw_calls if call.get("request_sha256")),
            "response_count": sum(1 for call in raw_calls if call.get("response_sha256")),
            "unique_request_count": len({call["request_sha256"] for call in raw_calls if call.get("request_sha256")}),
            "unique_response_count": len({call["response_sha256"] for call in raw_calls if call.get("response_sha256")}),
        },
        "integrity": {
            "all_request_files_present": all(call.get("request_file_present") for call in raw_calls),
            "all_response_files_present": all(call.get("response_file_present") for call in raw_calls),
            "all_request_hashes_match": all(call.get("request_hash_matches") for call in raw_calls),
            "all_response_schemas_valid": all(call.get("response_schema_valid") for call in raw_calls),
            "request_file_count": request_file_count,
            "response_file_count": response_file_count,
            "request_sha256_match_count": request_sha256_match_count,
            "response_schema_valid_count": schema_valid_count,
            "forbidden_label_key_count": request_state_forbidden,
            "replayed_response_count": replayed_count,
            "repeat_consistency": {"status": "not_run", "reason": "no repeated state hashes retained by the runner"},
            "state_hash_algorithm": {"status": "not_run", "reason": "state hashes are not retained by the runner"},
        },
        "by_dataset": by_dataset,
        "raw_calls": raw_calls,
        "retry_count": sum(max(0, int(call.get("attempts") or 1) - 1) for call in raw_calls),
    }


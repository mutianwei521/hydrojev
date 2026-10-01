"""Merge immutable measured sources and compute matched paper estimands; no API."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hydrojev.benchmarks import hydrojev_paper_provenance as legacy
from hydrojev.benchmarks.hydrojev_paper_metrics import (gate_metrics, hold_grid, hourly_metrics, paired_comparison,
                                       probability_metrics, rank_metrics)
from hydrojev.methods.hydrojev_detector import HydroJEVDetector, parse_detection_response

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/hydrojev_detection_v2/results.json"
FRESH = ROOT / "artifacts/hydrojev_detection_v2_fresh_full/results.json"
ABLATIONS = ROOT / "artifacts/hydrojev_detection_v2_live_ablations/results.json"
OUT = BASE.parent / "paper_summary.json"
CACHE = BASE.parent / "summary_derivations/matched_evidence_scores.npz"
MANIFEST = BASE.parent / "summary_derivations/matched_evidence_manifest.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value


def _evidence(key: str, names: list[str]) -> dict[str, Any]:
    removed = []
    if "no_temporal" in key: removed = ["GEpFM drift", "RAT covariance"]
    elif "no_instantaneous" in key: removed = ["Mahalanobis", "PCA residual", "Isolation Forest", "HydroJEV AE"]
    return {"evidence_removed": removed, "evidence_retained": [n for n in names if n not in removed],
            "definition_note": "no_instantaneous removes four direct-observation reference streams; CPDZ remains. "
                               "no_temporal removes GEpFM/RAT; remaining streams retain six-hour window summaries. "
                               "Detector streams are correlated, not statistically independent."}


def _states(v: dict[str, Any]) -> np.ndarray:
    grid = v["grid"]
    states = np.where(np.asarray(grid["alarm"], bool), "alarm", "normal").astype(object)
    for i, call in enumerate(v["calls"]):
        if call.get("status") in {"failed", "not_run"}: states[i] = "abstain"
        elif grid["probabilities"][i] is not None and not grid["alarm"][i] and float(grid["probabilities"][i]) > .2:
            states[i] = "abstain"
    return states


def _validate_raw_outputs(v: dict[str, Any], *, require_stored_response_hash: bool) -> dict:
    """Pin every used grid judgment to its typed raw response and request bytes."""
    successful = failed = 0
    detector = HydroJEVDetector(alarm_threshold=.5, normal_threshold=.2)
    for i, call in enumerate(v["calls"]):
        if not call.get("request_path"): continue
        req = Path(call["request_path"])
        if not req.is_absolute(): req = ROOT / req
        if not req.is_file() or not call.get("request_sha256") or sha(req) != call["request_sha256"]:
            raise ValueError("raw request hash missing or mismatched")
        request = read(req)
        if not {"model", "state", "questions"}.issubset(request): raise ValueError("request schema incomplete")
        if set(request["questions"]) != {"alarm", "event_type", "severity"}: raise ValueError("request question contract changed")
        if call.get("status") == "failed":
            if v["grid"]["probabilities"][i] is not None or v["grid"]["alarm"][i]:
                raise ValueError("failed service call imputed into a prediction")
            failed += 1
            continue
        resp = Path(call["response_path"])
        if not resp.is_absolute(): resp = ROOT / resp
        if not resp.is_file(): raise ValueError("successful call has no raw response")
        if require_stored_response_hash and not call.get("response_sha256"):
            raise ValueError("primary successful call lacks a recorded response hash")
        if call.get("response_sha256") and sha(resp) != call["response_sha256"]: raise ValueError("response hash mismatch")
        wrapper = read(resp)
        decision = parse_detection_response(wrapper["response"], source="live", attempts=wrapper["attempts"])
        output = detector.predict_decision(request["state"], decision)
        if output.alarm_probability != v["grid"]["probabilities"][i]: raise ValueError("raw probability differs from scored probability")
        if output.event_type != v["grid"]["event_types"][i]: raise ValueError("raw event differs from scored event")
        if output.severity_score != v["grid"]["severity_scores"][i]: raise ValueError("raw severity differs from scored severity")
        if (output.state.value == "alarm") != v["grid"]["alarm"][i]: raise ValueError("raw judgment does not reproduce the alarm")
        if call.get("model") != decision.model: raise ValueError("recorded and raw actual model disagree")
        successful += 1
    return {"status": "verified", "successful_raw_outputs_matched": successful, "failed_calls_retained": failed}


def _variant(v: dict[str, Any], key: str, path: Path, cache: Any, warmup: int, names: list[str]) -> dict[str, Any]:
    ds = v["variant"]
    labels = cache[ds + "__raw_labels"]
    indices = np.asarray(v["grid"]["raw_indices"], int)
    if not np.array_equal(labels[indices], v["grid"]["labels_internal"]): raise ValueError("grid labels differ from source")
    if len(v["calls"]) != len(indices): raise ValueError("call grid mismatch")
    call_indices = [c["raw_index"] for c in v["calls"]]
    if not np.array_equal(call_indices, indices): raise ValueError("calls not aligned")
    alarm = np.asarray(v["grid"]["alarm"], bool)
    candidate = np.asarray([c["candidate"] for c in v["calls"]], bool)
    called = np.asarray([bool(c.get("response_path")) for c in v["calls"]], bool)
    raw_validation = _validate_raw_outputs(v, require_stored_response_hash=(path != BASE))
    typed = np.asarray([x is not None for x in v["grid"]["event_types"]], bool)
    known = np.asarray([x is not None and x != "insufficient_evidence" for x in v["grid"]["event_types"]], bool)
    m = {**v["metrics"], **hourly_metrics(labels, hold_grid(indices, alarm, len(labels)), warmup)}
    # Verify recomputation against recorded measurements without replacing sources.
    for field in ["scenario_recall", "benign_hour_fpr", "mean_ttd_hours"]:
        old = v["metrics"].get(field)
        if old is not None and not np.isclose(old, m[field], atol=1e-12): raise ValueError(f"metric mismatch {ds}/{key}/{field}")
    if not np.isclose(m["batadal_score"]["S"], v["metrics"]["batadal_score"]["S"], atol=1e-12): raise ValueError("BATADAL mismatch")
    states = _states(v)
    m["abstention_rate_grid"] = float(np.mean(states == "abstain"))
    m["called_points"] = int(sum(c.get("status") == "ok" and bool(c.get("response_path")) for c in v["calls"]))
    m["attempted_points"] = int(called.sum())
    m["failed_points"] = int(sum(c.get("status") == "failed" for c in v["calls"]))
    failures = np.asarray([c.get("status") == "failed" for c in v["calls"]], bool)
    m["missing_service_hours"] = int(hold_grid(indices, failures, len(labels)).sum())
    m["service_failure_count"] = m["failed_points"]
    m["service_failure_hours"] = m["missing_service_hours"]
    m["invocation_failure_count"] = m["failed_points"]
    m["failure_interpretation"] = "Invocation failed or lacked a usable recorded response; the remote HTTP outcome/cause is undetermined. Legacy service_failure fields denote unavailable judgments, not a confirmed remote service outage."
    m["source_counts"] = dict(Counter("live_failure" if c.get("status") == "failed" else
                                     ("live" if c.get("response_path") else ("no_call" if not c["candidate"] else "no_jev"))
                                     for c in v["calls"]))
    m["semantic"] = {"cause_macro_f1": legacy._semantic_not_run("No independent cause labels"),
                     "severity_mae": legacy._semantic_not_run("No independent severity labels"),
                     "calibration": legacy._semantic_not_run("No independent semantic labels; alarm probability is scored separately")}
    semantic = {"non_unknown_count": int(known.sum()), "typed_response_count": int(typed.sum()),
                "non_unknown_rate_called": float(known.sum() / typed.sum()) if typed.any() else None,
                "non_unknown_rate_all_grid": float(known.sum() / len(indices)),
                "unknown_count": int((typed & ~known).sum()), "not_called_count": int((~called).sum()),
                "missing_response_count": int((called & ~typed).sum()),
                "denominator": "valid typed response count for called rate; complete grid count for all-grid rate",
                "meaning": "Coverage only, not semantic accuracy; insufficient_evidence is never a ground-truth class."}
    source_kind = "live" if called.any() else "deterministic_no_jev"
    result = {"dataset": ds, "ablation": key, "source": source_kind,
              "execution_source": "fresh_http" if called.any() and not any(c.get("replayed") for c in v["calls"]) else source_kind,
              "source_artifact": str(path.relative_to(ROOT)), "candidate_only": v["candidate_only"],
              "raw_output_validation": raw_validation,
              "metrics": m, "probability_metrics": probability_metrics(labels[indices], v["grid"]["probabilities"], called),
              "semantic_coverage": semantic,
              "gate_metrics": gate_metrics(labels, indices, candidate, alarm, warmup),
              "grid": legacy._grid_summary(v["grid"]),
              "calls": legacy._summarize_calls(ROOT, v["calls"], dataset=ds, ablation=key, include_live_records=False),
              **_evidence(key, names)}
    return result


def _matched_baselines(source: dict, cache: Any, manifest: dict, warmup: int, stride: int) -> tuple[dict, list]:
    baselines, rows = {}, []
    for ds in manifest["datasets"]:
        labels = cache[ds + "__raw_labels"]
        start = int(cache[ds + "__start"][0])
        grid = np.arange(warmup, len(labels), stride)
        baselines[ds] = {}
        for detector, mid in manifest["method_ids"].items():
            scores = cache[ds + "__" + mid]
            threshold = source["fit"]["thresholds"][detector]
            maxima = np.asarray([np.max(scores[max(0, j - start - stride + 1): j - start + 1]) for j in grid])
            alarm = maxima >= threshold
            m = hourly_metrics(labels, hold_grid(grid, alarm, len(labels)), warmup)
            rank = rank_metrics(labels[grid], maxima)
            row = {"dataset": ds, "method_id": mid, "detector": detector,
                   "detector_name": "Reconstruction AE" if detector == "HydroJEV AE" else detector,
                   "threshold": threshold, "protocol": "causal six-hour maximum, grid from raw index 71, forward hold until next grid point",
                   "metric_scope": "matched_six_hour_grid_forward_hold", **m,
                   "roc_auc": rank["roc_auc"], "pr_auc": rank["average_precision"],
                   "ranking": {**rank, "denominator": "grid labels and trailing-window maximum score; not hourly ranking"},
                   "grid_points": len(grid), "grid_alarm_points": int(alarm.sum()),
                   "event_ttd_hours": m["mean_ttd_hours"], "score_source": str(CACHE.relative_to(ROOT))}
            baselines[ds][mid] = row; rows.append(row)
    return baselines, rows


def _provenance(variants: list[dict], protocol: dict) -> dict:
    out = legacy._live_provenance(ROOT, variants, protocol)
    raw = out["raw_calls"]
    successes = [c for c in raw if c["response_schema_valid"]]
    out["failure_count"] = len(raw) - len(successes)
    out["invocation_failure_count"] = out["failure_count"]
    out["failure_classification"] = "observed_invocation_failure; remote HTTP outcome/cause undetermined" if out["failure_count"] else "none_observed"
    out["missing_response_count"] = sum(not c["response_file_present"] for c in raw)
    out["response_schema_failure_count"] = sum(c["response_file_present"] and not c["response_schema_valid"] for c in raw)
    out["failed_request_count"] = out["failure_count"]
    out["attempted_call_count"] = len(raw)
    out["fresh_invocation_count"] = sum(not c["replayed"] for c in raw)
    out["replayed_reference_count"] = sum(c["replayed"] for c in raw)
    out["known_successful_http_attempts"] = sum(c["attempts"] for c in successes)
    out["unknown_http_attempt_count_rows"] = sum(c["attempts"] is None for c in raw)
    out["total_http_attempts"] = None if out["unknown_http_attempt_count_rows"] else out["known_successful_http_attempts"]
    out["known_retry_count"] = out["retry_count"]
    out["retry_count_interpretation"] = "Known retries from retained response wrappers; failed invocation attempts can be unknown."
    out["total_retry_count"] = None if out["unknown_http_attempt_count_rows"] else out["known_retry_count"]
    for record in raw:
        record["invocation_status"] = "invoked_now" if not record["replayed"] else "replayed_reference"
        record["execution_source"] = "fresh_http" if not record["replayed"] else "replayed_reference"
        req = ROOT / record["request_path"]
        state = read(req)["state"] if req.is_file() else None
        record["state_sha256"] = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() if state is not None else None
        # Call-level stored hashes remain distinct from hashes newly computed by this audit.
        original = next(c for v in variants if v["variant"] == record["dataset"] and v["ablation"] == record["ablation"]
                        for c in v["calls"] if c["raw_index"] == record["raw_index"])
        record["recorded_state_sha256"] = original.get("state_sha256")
        record["failure_kind"] = "observed_invocation_failure" if original.get("status") == "failed" else None
        if record["failure_kind"]:
            record["execution_source"] = "fresh_invocation_http_outcome_unknown"
        record["recorded_failure"] = original.get("failure")
        record["returncode"] = original.get("returncode")
        record["state_hash_matches"] = bool(original.get("state_sha256") and record["state_sha256"] == original["state_sha256"])
        record["state_hash_status"] = "missing_recorded_hash" if not original.get("state_sha256") else ("match" if record["state_hash_matches"] else "mismatch")
    out["integrity"].update({
        "all_present_response_schemas_valid": all(c["response_schema_valid"] for c in raw if c["response_file_present"]),
        "successful_response_hash_match_count": sum(c["response_hash_matches"] for c in successes),
        "all_successful_response_hashes_match": bool(successes) and all(c["response_hash_matches"] for c in successes),
        "missing_recorded_response_hash_count": sum(c["recorded_response_sha256"] is None for c in raw),
        "state_hash_match_count": sum(c["state_hash_matches"] for c in raw),
        "state_hash_algorithm": "SHA-256 of UTF-8 canonical sorted compact JSON state",
        "all_recorded_state_hashes_match": bool(raw) and all(c["state_hash_matches"] for c in raw),
    })
    out["by_method"] = {key: {"raw_call_count": sum(c["ablation"] == key for c in raw),
                             "successful_response_count": sum(c["ablation"] == key for c in successes),
                             "failure_count": sum(c["ablation"] == key and not c["response_schema_valid"] for c in raw)}
                        for key in sorted({c["ablation"] for c in raw})}
    for dataset, entry in out["by_dataset"].items():
        entry["successful_response_count"] = sum(c["dataset"] == dataset for c in successes)
        entry["failure_count"] = sum(c["dataset"] == dataset and not c["response_schema_valid"] for c in raw)
    return out


def build_summary() -> dict:
    source, fresh, ablations, manifest = read(BASE), read(FRESH), read(ABLATIONS), read(MANIFEST)
    if sha(BASE) != manifest["source_result_sha256"] or sha(CACHE) != manifest["score_file_sha256"]: raise ValueError("baseline cache provenance mismatch")
    for item in manifest["datasets"].values():
        if sha(Path(item["source_path"])) != item["source_sha256"]: raise ValueError("dataset bytes changed")
    for name, current in [("fresh", fresh), ("ablations", ablations)]:
        if current["fit"]["thresholds"] != source["fit"]["thresholds"]: raise ValueError(f"{name} thresholds differ")
    cache = np.load(CACHE, allow_pickle=False)
    warmup, stride = source["protocol"]["warmup_samples"], source["protocol"]["grid_stride_hours"]
    names = list(source["fit"]["thresholds"])
    selected = []
    for v in source["variants"]:
        if v["ablation"] in {"no_jev", "no_temporal", "no_instantaneous"}: selected.append((v, v["ablation"], BASE))
    selected += [(v, "full", FRESH) for v in fresh["variants"]]
    selected += [(v, "live_" + v["ablation"], ABLATIONS) for v in ablations["variants"]]
    variants, measured = {}, {}
    primary_live = []
    for v, key, path in selected:
        ds = v["variant"]
        variants.setdefault(ds, {})[key] = _variant(v, key, path, cache, warmup, names)
        measured[(ds, key)] = v
        if key == "full" or key.startswith("live_"): primary_live.append({**v, "ablation": key})
    baselines, rows = _matched_baselines(source, cache, manifest, warmup, stride)
    hourly_rows = [legacy._baseline_row(r) for r in source["baseline_rows"]]
    paired = {}
    for ds in variants:
        paired[ds] = {}
        for a, b in [("full", "no_jev"), ("live_no_temporal", "no_temporal"),
                     ("live_no_instantaneous", "no_instantaneous"), ("full", "live_no_temporal"), ("full", "live_no_instantaneous")]:
            av, bv = measured[(ds, a)], measured[(ds, b)]
            ai = np.asarray(av["grid"]["raw_indices"], int)
            if not np.array_equal(ai, bv["grid"]["raw_indices"]): raise ValueError("paired grids differ")
            pair = paired_comparison(cache[ds + "__raw_labels"], ai, np.asarray(av["grid"]["alarm"]), np.asarray(bv["grid"]["alarm"]),
                                      _states(av), _states(bv), av["grid"]["probabilities"], warmup)
            pair.update({"live_variant": a, "control_variant": b, "direction": a + "_minus_" + b})
            if b.startswith("live_"):
                pair["probability_vs_hard_prediction"]["note"] = (
                    "The reference is the control method's hard alarm, not its returned probability. "
                    "Candidate sets can differ; this diagnostic is restricted to valid numerator-method calls."
                )
            paired[ds][a + "_minus_" + b] = pair
    refs = [v for v in source["variants"] if v["ablation"] == "full"]
    ref_prov = _provenance(refs, source["protocol"])
    provenance = _provenance(primary_live, fresh["protocol"])
    prior = {(c["dataset"], c["raw_index"]): c for c in ref_prov["raw_calls"]}
    repeat_pairs = []
    unmatched_pairs = []
    for c in provenance["raw_calls"]:
        if c["ablation"] != "full" or not c["response_schema_valid"]: continue
        old = prior[(c["dataset"], c["raw_index"])]
        if not old["response_schema_valid"] or c["state_sha256"] != old["state_sha256"]:
            unmatched_pairs.append({"dataset": c["dataset"], "raw_index": c["raw_index"],
                                    "reason": "reference response invalid or evidence-state hash differs"})
            continue
        a, b = read(ROOT / c["response_path"])["response"]["answers"], read(ROOT / old["response_path"])["response"]["answers"]
        new_variant = measured[(c["dataset"], "full")]
        old_variant = next(v for v in refs if v["variant"] == c["dataset"])
        new_pos = new_variant["grid"]["raw_indices"].index(c["raw_index"])
        old_pos = old_variant["grid"]["raw_indices"].index(c["raw_index"])
        repeat_pairs.append({"dataset": c["dataset"], "raw_index": c["raw_index"], "state_hash_matches": c["state_sha256"] == old["state_sha256"],
            "binary_equal": new_variant["grid"]["alarm"][new_pos] == old_variant["grid"]["alarm"][old_pos],
            "fusion_state_equal": _states(new_variant)[new_pos] == _states(old_variant)[old_pos],
            "event_type_equal": a["event_type"]["choice"] == b["event_type"]["choice"],
            "probability_absolute_difference": abs(a["alarm"]["noul"] - b["alarm"]["noul"]),
            "severity_absolute_difference": abs(a["severity"]["score"] - b["severity"]["score"])})
    repeat = {"status": "paired_repeated_measurements", "state_matched_pair_count": sum(c["state_hash_matches"] for c in repeat_pairs),
              "successful_pair_count": len(repeat_pairs), "additional_independent_sample_count": 0,
              "excluded_unmatched_pair_count": len(unmatched_pairs), "excluded_unmatched_pairs": unmatched_pairs,
              "binary_output_agreement": float(np.mean([c["binary_equal"] for c in repeat_pairs])),
              "fusion_state_agreement": float(np.mean([c["fusion_state_equal"] for c in repeat_pairs])),
              "event_type_agreement": float(np.mean([c["event_type_equal"] for c in repeat_pairs])),
              "alarm_probability_mean_absolute_difference": float(np.mean([c["probability_absolute_difference"] for c in repeat_pairs])),
              "severity_mean_absolute_difference": float(np.mean([c["severity_absolute_difference"] for c in repeat_pairs])),
              "note": "Repeated calls are reproducibility checks, not new independent events or network replications.", "pairs": repeat_pairs}
    provenance["integrity"]["repeat_consistency"] = repeat
    protocol = {**source["protocol"], "candidate_rule": "at least 3 of the retained multiple detector streams alert in a causal six-hour window",
        "fusion_alarm_threshold": .5, "fusion_normal_threshold": .2, "thresholds": source["fit"]["thresholds"],
        "state_schema_version": "hydrojev.detection_state/1", "method_scope": "candidate-gated Jev anomaly-alert method",
        "statistics": {"status": "post_hoc_descriptive", "event_unit": "attack episode; seven per split, one water network",
            "bootstrap_resamples": 1000, "paired_temporal_block_hours": 48, "random_seed": 42,
            "note": "Wilson and paired intervals are descriptive, chosen after results; shared temporal dependence limits inference. No p-values or superiority/equivalence claims."}}
    summary = {"schema_version": "hydrojev.paper_summary/3", "status": "observed_invocation_failure" if provenance["failure_count"] else "ok",
        "source_artifact": str(BASE.relative_to(ROOT)),
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha(p)} for p in [BASE, FRESH, ABLATIONS, CACHE, MANIFEST]],
        "protocol": protocol, "fit": {**source["fit"], "status": "ok", "reproduction_verified": True},
        "variants": variants, "baselines": baselines, "baseline_rows": rows, "paired": paired,
        "hourly_reference": {"role": "original hourly scoring; not directly ranked against six-hour HydroJEV", "baseline_rows": hourly_rows,
                             "baselines": {ds: {r["method_id"]: r for r in hourly_rows if r["dataset"] == ds} for ds in variants}},
        "live_provenance": provenance, "repeat_consistency": repeat,
        "reproducibility": {"reference_full": {v["variant"]: _variant(v, "full", BASE, cache, warmup, names) for v in refs},
                            "reference_provenance": ref_prov, "reference_raw_call_count": ref_prov["raw_call_count"]},
        "excluded_variants": [{"dataset": v["variant"], "ablation": v["ablation"], "status": "not_run"} for v in source["variants"] if v["ablation"] == "full_all_grid"],
        "dataset": {"source_file_mapping": {
            "training": {"reader_name": "attack-free training set", "source_identifier": "dataset03", "hours": manifest["training_source"]["rows"]},
            "development": {"reader_name": "labelled development set", "source_identifier": "dataset04", "hours": manifest["datasets"]["dataset04"]["raw_rows"]},
            "test": {"reader_name": "retrospective test set", "source_identifier": "test_dataset", "hours": manifest["datasets"]["test_dataset"]["raw_rows"]}},
            "evaluation_grid": {"warmup_samples": warmup, "stride_hours": stride,
                                "development_grid_points": variants["dataset04"]["full"]["grid"]["grid_points"],
                                "test_grid_points": variants["test_dataset"]["full"]["grid"]["grid_points"]}},
        "semantic_evaluation": {"status": "not_run", "label_source": legacy._semantic_not_run("Only binary attack labels are available"),
                                "denominator": legacy._semantic_not_run("No independently labelled cause/severity subset")}}
    OUT.write_text(json.dumps(safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return safe(summary)


def main() -> int:
    result = build_summary()
    live = result["live_provenance"]
    print(json.dumps({"status": result["status"], "output": str(OUT),
                      "raw_calls": live["raw_call_count"],
                      "valid_responses": live["successful_response_count"],
                      "invocation_failures": live["failure_count"],
                      "api_calls_by_summary": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run standalone HydroJEV inference from a saved, label-free request.

Examples::

    python -m hydrojev.methods.detect --request state.request.json \
        --mode replay --response saved.response.json --output result.json
    python -m hydrojev.methods.detect --request state.request.json \
        --mode no-jev --output local.result.json
    python -m hydrojev.methods.detect --request state.request.json \
        --mode live --output live.result.json

The CLI freezes the benchmark operating thresholds at alarm=0.5 and
normal=0.2.  This intentionally differs from the reusable detector class's
conservative constructor default, alarm=0.8.  It never changes fusion logic.
The live mode uses the project's authorized invoke-jev.ps1 wrapper; replay
and no-jev are entirely offline and cannot be labelled as live inference.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from hydrojev.methods.hydrojev_detector import (
    HydroJEVDetector,
    build_detection_questions,
    build_detection_request,
    deterministic_no_jev,
    parse_detection_response,
)

BENCHMARK_ALARM_THRESHOLD = 0.5
BENCHMARK_NORMAL_THRESHOLD = 0.2
INFERENCE_SCHEMA_VERSION = "hydrojev.inference_result/1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_saved_response(path: Path, *, source: str):
    wrapper = _read_json(path)
    body = wrapper.get("response", wrapper)
    timing = {}
    if wrapper.get("elapsed_seconds") is not None:
        timing["elapsed_seconds"] = float(wrapper["elapsed_seconds"])
    decision = parse_detection_response(
        body, source=source, timing_ms=timing,
        attempts=int(wrapper.get("attempts", 1)),
    )
    return decision, wrapper


def run_inference(
    *, request_path: Path, mode: str, output_path: Path,
    response_path: Path | None = None,
) -> dict[str, Any]:
    """Run one explicit mode and return a JSON-safe result with provenance."""

    request_path = request_path.resolve()
    output_path = output_path.resolve()
    if response_path is not None:
        response_path = response_path.resolve()
    if output_path in {request_path, response_path}:
        raise ValueError("Output must not overwrite the input request or saved response")
    if mode == "replay" and response_path is None:
        raise ValueError("--mode replay requires --response")
    if mode != "replay" and response_path is not None:
        raise ValueError("--response is only accepted with --mode replay")
    if mode not in {"replay", "no-jev", "live"}:
        raise ValueError("mode must be replay, no-jev or live")

    supplied = _read_json(request_path)
    state = supplied.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("--request must contain a state object")
    # Rebuild every effective payload from the current question contract.
    # This also rejects label/attack/row metadata and all nonfinite values.
    request = build_detection_request(state, model=str(supplied.get("model", "jev-latest")))
    questions_match = supplied.get("questions") == build_detection_questions()
    if mode == "replay" and not questions_match:
        raise ValueError("Replay requires the current question contract; a saved response cannot validate different questions")

    detector = HydroJEVDetector(
        alarm_threshold=BENCHMARK_ALARM_THRESHOLD,
        normal_threshold=BENCHMARK_NORMAL_THRESHOLD,
    )
    provenance: dict[str, Any] = {
        "mode": mode, "execution_source": "no_jev" if mode == "no-jev" else mode,
        "replayed": mode == "replay", "live_api_invoked": False,
        "input_request_path": str(request_path),
        "input_request_sha256": _sha_bytes(request_path.read_bytes()),
        "effective_request_sha256": _sha_bytes(_json_bytes(request)),
        "state_sha256": _sha_bytes(_json_bytes(request["state"])),
        "question_contract_sha256": _sha_bytes(_json_bytes(request["questions"])),
        "current_question_contract_used": True,
        "input_question_contract_matches": questions_match,
        "created_utc": _utc_now(),
    }
    status = "ok"
    if mode == "no-jev":
        result = deterministic_no_jev(request["state"])
    elif mode == "replay":
        assert response_path is not None
        decision, wrapper = _parse_saved_response(response_path, source="replay")
        result = detector.predict_decision(request["state"], decision)
        provenance.update({
            "response_path": str(response_path),
            "response_sha256": _sha_bytes(response_path.read_bytes()),
            "saved_response_timestamp_utc": wrapper.get("timestamp_utc"),
            "saved_response_model": decision.model,
            "note": "Inference reused a saved response; no HTTP request was made by this CLI run.",
        })
    elif not bool(request["state"].get("routing", {}).get("candidate", True)):
        result = detector.predict_response(request["state"], None, source="no_call")
        provenance["execution_source"] = "no_call"
    else:
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "tools" / "jev" / "invoke-jev.ps1"
        if not script.is_file():
            raise FileNotFoundError("The configured invoke-jev.ps1 wrapper is absent")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submitted_path = output_path.with_name(output_path.stem + ".submitted.request.json")
        live_response_path = output_path.with_name(output_path.stem + ".live.response.json")
        if live_response_path.exists():
            raise ValueError("Live response artifact already exists; choose a new output path to preserve provenance")
        if submitted_path.resolve() == request_path:
            raise ValueError("Submitted payload path would overwrite the input request")
        submitted_path.write_bytes(_json_bytes(request))
        provenance.update({
            "execution_source": "fresh_http", "live_api_invoked": True,
            "submitted_request_path": str(submitted_path),
            "response_path": str(live_response_path),
            "invocation_started_utc": _utc_now(),
        })
        command = ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                   "-RequestPath", str(submitted_path), "-OutputPath", str(live_response_path)]
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", check=False)
        provenance.update({
            "returncode": int(process.returncode),
            "invocation_finished_utc": _utc_now(),
            "wrapper_elapsed_seconds": time.perf_counter() - started,
        })
        if process.returncode != 0 or not live_response_path.is_file():
            status = "failed"
            provenance["failure"] = "live_wrapper_failed_or_response_absent"
            result = detector.failure_result(request["state"], reason=provenance["failure"])
        else:
            provenance["response_sha256"] = _sha_bytes(live_response_path.read_bytes())
            try:
                decision, wrapper = _parse_saved_response(live_response_path, source="live")
                result = detector.predict_decision(request["state"], decision)
                provenance.update({"timestamp_utc": wrapper.get("timestamp_utc"),
                                   "actual_model": decision.model, "attempts": decision.attempts})
            except (ValueError, KeyError, TypeError) as error:
                status = "failed"
                provenance["failure"] = f"response_schema:{type(error).__name__}"
                result = detector.failure_result(request["state"], reason=provenance["failure"])

    return {
        "schema_version": INFERENCE_SCHEMA_VERSION, "status": status,
        "method": {
            "name": "HydroJEV", "alarm_threshold": BENCHMARK_ALARM_THRESHOLD,
            "normal_threshold": BENCHMARK_NORMAL_THRESHOLD,
            "threshold_policy": "fixed benchmark operating point",
            "library_constructor_default_alarm_threshold": HydroJEVDetector().alarm_threshold,
            "no_jev_min_alerting_detectors": 2,
        },
        "result": result.to_dict(), "provenance": provenance,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--request", type=Path, required=True, help="Saved JSON request containing a label-free state")
    parser.add_argument("--mode", choices=("replay", "no-jev", "live"), default="replay",
                        help="Default: replay (offline). live performs an authorized HTTP API call.")
    parser.add_argument("--response", type=Path, help="Saved raw response or wrapper; required only for replay")
    parser.add_argument("--output", type=Path, required=True, help="Inference result JSON")
    args = parser.parse_args(argv)
    try:
        payload = run_inference(request_path=args.request, mode=args.mode,
                                response_path=args.response, output_path=args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"HydroJEV inference failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()), "status": payload["status"],
                      "decision": payload["result"]["state"], "source": payload["result"]["source"],
                      "live_api_invoked": payload["provenance"]["live_api_invoked"]}, ensure_ascii=False))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

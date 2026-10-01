"""Retrain the deterministic front end and cache evidence for matched evaluation.

This program invokes no inference service. Cached source scores are validated
against every retained threshold and the original hourly baseline metrics.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# This import order avoids competing Windows OpenMP initialisation in the
# workstation runtime; no duplicate-runtime override is used.
import numpy as np
import sklearn
import torch

from hydrojev.benchmarks.detection_baselines import evaluate_scores
from hydrojev.benchmarks.hydrojev_detection_benchmark import fit_front_end, score_stream
from hydrojev.config import load_config
from hydrojev.datasets.wdn_loader import load_batadal

OUT = ROOT / "artifacts/hydrojev_detection_v2/summary_derivations"
SOURCE = ROOT / "artifacts/hydrojev_detection_v2/results.json"
SCORE_PATH = OUT / "matched_evidence_scores.npz"
MANIFEST_PATH = OUT / "matched_evidence_manifest.json"
METHOD_IDS = {"Mahalanobis": "mahalanobis", "PCA residual": "pca_residual",
              "Isolation Forest": "isolation_forest", "HydroJEV AE": "reconstruction_ae",
              "CPDZ residual": "cpdz", "GEpFM drift": "gepfm", "RAT covariance": "rat"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    fit = source["fit"]
    cfg = load_config()
    front = fit_front_end(cfg, seed=fit["seed"], gru_epochs=fit["gru_epochs"], ae_epochs=fit["ae_epochs"])
    if isinstance(front, dict):
        raise RuntimeError(f"front-end training failed: {front}")
    threshold_differences = {k: float(front.thresholds[k] - value) for k, value in fit["thresholds"].items()}
    if not all(np.isclose(front.thresholds[k], value, rtol=1e-10, atol=1e-12) for k, value in fit["thresholds"].items()):
        raise ValueError(f"retrained thresholds do not reproduce source: {threshold_differences}")
    if front.train_rows != fit["train_rows"] or list(front.columns) != fit["columns"]:
        raise ValueError("training rows or feature columns changed")
    arrays = {}
    dataset_manifest = {}
    original = {(r["dataset"], r["detector"]): r for r in source["baseline_rows"]}
    for dataset in sorted({v["variant"] for v in source["variants"]}):
        stream = score_stream(front, cfg, dataset)
        bundle = stream["bundle"]
        labels = bundle.frame[bundle.label_column].to_numpy(dtype=int)
        arrays[dataset + "__raw_labels"] = labels
        arrays[dataset + "__start"] = np.asarray([stream["start"]])
        arrays[dataset + "__interval_s"] = np.asarray([stream["interval_s"]])
        differences = {}
        for detector, values in stream["scores"].items():
            mid = METHOD_IDS[detector]
            arrays[dataset + "__" + mid] = np.asarray(values, dtype=float)
            measured = evaluate_scores(values, front.thresholds[detector], stream["labels"])
            expected = original[(dataset, detector)]
            differences[mid] = {}
            for field, value in measured.items():
                if field not in expected:
                    raise ValueError(f"source missing {dataset}/{detector}/{field}")
                delta = float(value - expected[field])
                differences[mid][field] = delta
                if not np.isclose(value, expected[field], rtol=1e-10, atol=1e-12, equal_nan=True):
                    raise ValueError(f"hourly baseline failed reproduction: {dataset}/{detector}/{field}: {value} != {expected[field]}")
        dataset_manifest[dataset] = {"source_path": str(bundle.source_path), "source_sha256": sha(bundle.source_path),
            "raw_rows": len(labels), "start": stream["start"], "sample_interval_s": stream["interval_s"],
            "hourly_reproduction_differences": differences}
    normal = load_batadal("dataset03", config=cfg)
    paths = [ROOT / "hydrojev/benchmarks/hydrojev_detection_benchmark.py",
             ROOT / "hydrojev/benchmarks/named_family_benchmark.py",
             ROOT / "hydrojev/benchmarks/detection_baselines.py",
             ROOT / "hydrojev/datasets/wdn_loader.py"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCORE_PATH, **arrays)
    manifest = {"status": "ok", "source_result": str(SOURCE), "source_result_sha256": sha(SOURCE),
        "score_file": str(SCORE_PATH), "score_file_sha256": sha(SCORE_PATH),
        "runtime": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
        "fit": fit, "threshold_differences": threshold_differences, "method_ids": METHOD_IDS,
        "training_source": {"path": str(normal.source_path), "sha256": sha(normal.source_path), "rows": len(normal.frame)},
        "code_sha256": {str(path): sha(path) for path in paths}, "datasets": dataset_manifest,
        "api_calls": 0, "note": "Deterministic front-end reconstruction only; all threshold and original hourly metrics reproduced."}
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(SCORE_PATH), "threshold_differences": threshold_differences,
                      "datasets": {k: {"raw_rows": v["raw_rows"], "start": v["start"]} for k, v in dataset_manifest.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

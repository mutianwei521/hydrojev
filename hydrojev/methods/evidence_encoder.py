"""Leakage-safe, causal evidence encoding for the standalone HydroJEV method.

The encoder is the bridge between the numerical detector stack and a Jev
request.  It is deliberately independent from the Jev client: fitting and
transforming this module never performs network I/O and never sees an attack
label.  A fitted encoder consumes a clean training matrix once, produces
complete per-row score streams, and exposes six-hour causal evidence states on
a fixed grid.

The public API is intentionally small::

    encoder = HydroJEVEvidenceEncoder.fit(clean_x, columns, config=config)
    encoded = encoder.transform(raw_x, ablation="full")

``encoded.numeric_features`` is the same evidence used by the state builder,
so a logistic baseline can be fit without silently switching to labels or to a
different window summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks.detection_baselines import fit_isolation_forest_detector
from hydrojev.benchmarks.named_family_benchmark import (
    cpdz_series,
    gepfm_series,
    one_step_residual_stream,
    rat_series,
)
from hydrojev.config import HydroJEVConfig, load_config
from hydrojev.datasets.wdn_loader import classify_batadal_columns


SCHEMA_VERSION = "hydrojev.evidence_state/1"
ENCODER_VERSION = "hydrojev.evidence_encoder/1"
UNAVAILABLE = "unavailable"
DEFAULT_WINDOW_STEPS = 6
DEFAULT_STRIDE_STEPS = 6
DEFAULT_GRU_EPOCHS = 60
DEFAULT_AE_EPOCHS = 40
DEFAULT_SEED = 42

ABLATIONS = frozenset({
    "full",
    "no_temporal",
    "no_instantaneous",
    "no_sensor_context",
})

# Public score names are stable and do not expose implementation-specific
# display names such as "HydroJEV AE".  They map to the names emitted by
# ``train_named_families`` below.
_TEMPORAL = ("cpdz_residual", "gepfm_drift", "rat_covariance")
_INSTANTANEOUS = (
    "mahalanobis",
    "pca_residual",
    "autoencoder",
    "isolation_forest",
)
_SCORE_ORDER = _TEMPORAL + _INSTANTANEOUS
_TRAINED_NAMES = {
    "cpdz_residual": "CPDZ residual",
    "gepfm_drift": "GEpFM drift",
    "rat_covariance": "RAT covariance",
    "mahalanobis": "Mahalanobis",
    "pca_residual": "PCA residual",
    "autoencoder": "HydroJEV AE",
    "isolation_forest": "Isolation Forest",
}


def _as_finite_matrix(values: Any, name: str) -> np.ndarray:
    """Validate and copy a two-dimensional numeric stream."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 2D array") from exc
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(array, dtype=np.float64)


def _hash_matrix(values: np.ndarray, feature_names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    digest.update("\0".join(str(name) for name in feature_names).encode("utf-8"))
    return digest.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    """Convert metadata/state values without turning missing values into zero."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else UNAVAILABLE
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _empirical_percentile(reference: np.ndarray, value: float) -> float | None:
    """Return a deterministic percentile in [0, 100] from a clean reference."""

    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not math.isfinite(float(value)):
        return None
    # Mid-rank ties keep a score exactly at a repeated healthy value from being
    # reported as either the bottom or top tail.
    below = float(np.sum(ref < value))
    equal = float(np.sum(ref == value))
    return float(100.0 * (below + 0.5 * equal) / ref.size)


def _tail_fraction(reference: np.ndarray, value: float) -> float | None:
    """Return P(clean score >= current score), the empirical upper-tail mass."""

    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0 or not math.isfinite(float(value)):
        return None
    return float(np.mean(ref >= value))


def _safe_quantile(values: np.ndarray, quantile: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, quantile))


def _threshold_from_bundle(bundle: Any, canonical: str) -> float | None:
    thresholds = getattr(bundle, "thresholds", {}) or {}
    name = _TRAINED_NAMES[canonical]
    if name in thresholds:
        return _finite_or_none(thresholds[name])
    # Accept case/spacing variations in small test doubles and old persisted
    # bundles, while keeping the public canonical names stable.
    normalized = {str(key).strip().lower().replace("_", " " ): value for key, value in thresholds.items()}
    wanted = name.lower().replace("_", " " )
    for key, value in normalized.items():
        if key == wanted:
            return _finite_or_none(value)
    return None


def _detector_from_bundle(bundle: Any, canonical: str) -> Any | None:
    attr = {
        "mahalanobis": "maha",
        "pca_residual": "pca",
        "autoencoder": "ae",
    }.get(canonical)
    return getattr(bundle, attr, None) if attr else None


def _healthy_family(bundle: Any, canonical: str) -> np.ndarray:
    ref = getattr(bundle, "ref", None)
    attr = {
        "cpdz_residual": "clean_cpdz",
        "gepfm_drift": "clean_gepfm",
        "rat_covariance": "clean_rat",
    }.get(canonical)
    values = getattr(ref, attr, np.array([], dtype=float)) if ref is not None else np.array([], dtype=float)
    return np.asarray(values, dtype=float).copy()


def _standardized_stream(bundle: Any, raw: np.ndarray) -> np.ndarray:
    standardizer = getattr(bundle, "standardizer", None)
    if standardizer is None or not hasattr(standardizer, "transform"):
        # A fitted bundle is expected to carry the training standardizer.  The
        # fallback keeps lightweight test doubles useful and does not invent
        # hydraulic quantities: it is simply the identity transform.
        return raw.copy()
    result = np.asarray(standardizer.transform(raw), dtype=float)
    if result.shape != raw.shape or not np.isfinite(result).all():
        raise ValueError("fitted standardizer returned an invalid stream")
    return result


def _make_training_mode(ae_epochs: int) -> Any:
    """Return the minimal RunMode-like object needed by the trainer."""

    return SimpleNamespace(
        name="hydrojev_evidence_encoder",
        dry=False,
        batadal_train_rows=None,
        ae_epochs=int(ae_epochs),
        scenarios_per_category=1,
        latency_iterations=1,
        concealment_budget=1,
        cluster_net3=False,
    )


@dataclass(frozen=True)
class EncodedStream:
    """Scores and causal Jev evidence states for one raw stream."""

    scores: dict[str, np.ndarray]
    state_indices: np.ndarray
    states: tuple[dict[str, Any], ...]
    numeric_features: np.ndarray
    numeric_feature_names: tuple[str, ...]
    numeric_feature_mask: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def numeric_feature_matrix(self) -> np.ndarray:
        """Compatibility alias for downstream logistic baselines."""

        return self.numeric_features


@dataclass
class HydroJEVEvidenceEncoder:
    """Fit-on-clean, transform-only numerical evidence encoder."""

    feature_names: tuple[str, ...]
    trained: Any
    isolation_forest: Any
    thresholds: dict[str, float | None]
    healthy_scores: dict[str, np.ndarray]
    sensor_groups: dict[str, tuple[str, ...]]
    config: HydroJEVConfig | None = None
    seed: int = DEFAULT_SEED
    gru_epochs: int = DEFAULT_GRU_EPOCHS
    ae_epochs: int = DEFAULT_AE_EPOCHS
    training_rows: int = 0
    input_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        clean_features: np.ndarray,
        feature_names: Sequence[str],
        *,
        config: HydroJEVConfig | None = None,
        trained: Any | None = None,
        trainer: Callable[..., Any] | None = None,
        trainer_metadata: Mapping[str, Any] | None = None,
        seed: int = DEFAULT_SEED,
        gru_epochs: int = DEFAULT_GRU_EPOCHS,
        ae_epochs: int = DEFAULT_AE_EPOCHS,
        threshold_quantile: float | None = None,
        n_estimators: int = 200,
    ) -> "HydroJEVEvidenceEncoder":
        """Fit all detector evidence on clean observations only.

        ``train_named_families`` is called exactly once when ``trained`` is not
        supplied.  Passing a pre-fitted ``trained`` bundle is useful for tests
        and for callers that already own the GRU/AE training artifact.
        ``clean_features`` is never allowed to contain labels; this API accepts
        only a numeric matrix and feature names.
        """

        clean = _as_finite_matrix(clean_features, "clean_features")
        names = tuple(str(name) for name in feature_names)
        if len(names) != clean.shape[1]:
            raise ValueError("feature_names length must match clean_features width")
        if not names or any(not name.strip() for name in names):
            raise ValueError("feature_names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        forbidden = {
            "att_flag",
            "label",
            "ground_truth",
            "true_label",
            "timestamp",
            "datetime",
            "dataset",
            "row_id",
        }
        forbidden_names = [
            name for name in names if name.strip().lower() in forbidden
        ]
        if forbidden_names:
            raise ValueError(
                "clean_features must not include labels or row metadata: "
                + ", ".join(forbidden_names)
            )
        if int(gru_epochs) < 1 or int(ae_epochs) < 1:
            raise ValueError("gru_epochs and ae_epochs must be positive")
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be positive")

        cfg = config
        if trained is None:
            if trainer is None:
                if cfg is None:
                    cfg = load_config()
                from hydrojev.benchmarks.named_family_benchmark import train_named_families

                trainer = train_named_families
            if cfg is None:
                # Injected trainers in unit tests often do not inspect config.
                # Keep the argument explicit and avoid reading benchmark assets.
                cfg = config
            mode = _make_training_mode(int(ae_epochs))
            trained = trainer(
                cfg, mode=mode, seed=int(seed), max_epochs=int(gru_epochs)
            )
            if isinstance(trained, Mapping) and trained.get("status") != "ok":
                raise RuntimeError(
                    "named-family training did not produce a fitted bundle: "
                    + str(trained.get("reason", trained.get("status")))
                )
        if trained is None or isinstance(trained, Mapping):
            raise ValueError("trained must be a fitted named-family bundle")

        q = threshold_quantile
        if q is None:
            q = getattr(getattr(cfg, "concealment", None), "threshold_quantile", 0.995)
        q = float(q)
        if not 0.0 < q < 1.0:
            raise ValueError("threshold_quantile must lie in (0, 1)")

        # Fit the independent instantaneous forest once on the same clean rows.
        isolation = fit_isolation_forest_detector(
            clean, threshold_quantile=q, seed=int(seed), n_estimators=int(n_estimators)
        )

        # Healthy references for the temporal families are supplied by the
        # named-family trainer.  Recompute only when a small test double omitted
        # them; the recomputation still uses clean data and no labels.
        healthy: dict[str, np.ndarray] = {}
        for canonical in ("cpdz_residual", "gepfm_drift", "rat_covariance"):
            healthy[canonical] = _healthy_family(trained, canonical)
        # The instantaneous references and the isolation forest must use the
        # same clean distribution for percentile/tail summaries as the stream
        # scores use at transform time.
        for canonical in ("mahalanobis", "pca_residual", "autoencoder"):
            detector = _detector_from_bundle(trained, canonical)
            if detector is not None and hasattr(detector, "score_samples"):
                healthy[canonical] = np.asarray(
                    detector.score_samples(clean), dtype=float
                ).reshape(-1)
            else:
                healthy[canonical] = np.array([], dtype=float)
        healthy["isolation_forest"] = np.asarray(
            isolation.score_samples(clean), dtype=float
        ).reshape(-1)

        seq_len = int(getattr(trained, "seq_len", getattr(getattr(cfg, "surrogate", None), "sequence_length", 12)))
        rat_window = int(getattr(trained, "rat_window", max(60, clean.shape[1] + 1)))
        if any(values.size == 0 for values in healthy.values()):
            try:
                clean_std = _standardized_stream(trained, clean)
                residuals, _ = one_step_residual_stream(
                    trained.predict_fn, clean_std, seq_len
                )
                ref = getattr(trained, "ref", None)
                if healthy["cpdz_residual"].size == 0 and ref is not None:
                    healthy["cpdz_residual"] = cpdz_series(residuals, np.asarray(ref.scale, dtype=float))
                if healthy["gepfm_drift"].size == 0 and ref is not None:
                    cpdz = cpdz_series(residuals, np.asarray(ref.scale, dtype=float))
                    healthy["gepfm_drift"] = gepfm_series(
                        cpdz,
                        reference_mean=float(ref.gepfm_ref_mean),
                        reference_scale=float(ref.gepfm_ref_scale),
                        smoothing=float(ref.gepfm_smoothing),
                    )
                if healthy["rat_covariance"].size == 0 and ref is not None:
                    healthy["rat_covariance"] = rat_series(
                        residuals, np.asarray(ref.rat_ref_cov, dtype=float), window=rat_window
                    )
            except Exception:
                # A detector without its optional temporal reference simply
                # reports unavailable temporal evidence; no synthetic zeros.
                pass

        detector_thresholds: dict[str, float | None] = {
            canonical: _threshold_from_bundle(trained, canonical)
            for canonical in (
                "cpdz_residual",
                "gepfm_drift",
                "rat_covariance",
                "mahalanobis",
                "pca_residual",
                "autoencoder",
            )
        }
        detector_thresholds["isolation_forest"] = float(isolation.threshold)

        for canonical in ("mahalanobis", "pca_residual", "autoencoder"):
            detector = _detector_from_bundle(trained, canonical)
            if detector is None:
                detector_thresholds[canonical] = None

        groups = {
            str(group): tuple(str(column) for column in columns if str(column) in names)
            for group, columns in classify_batadal_columns(names).items()
        }
        groups = {group: columns for group, columns in groups.items() if columns}
        if not groups:
            groups = {"sensor": names}

        metadata = {
            "schema_version": ENCODER_VERSION,
            "seed": int(seed),
            "training_rows": int(clean.shape[0]),
            "feature_count": int(clean.shape[1]),
            "feature_names": list(names),
            "input_sha256": _hash_matrix(clean, names),
            "gru_max_epochs": int(gru_epochs),
            "ae_max_epochs": int(ae_epochs),
            "sequence_length": int(seq_len),
            "rat_window": int(rat_window),
            "threshold_quantile": float(q),
            "fit_source": "clean observations only; labels excluded by API",
        }
        if trainer_metadata:
            metadata["trainer_metadata"] = _json_safe(dict(trainer_metadata))

        return cls(
            feature_names=names,
            trained=trained,
            isolation_forest=isolation,
            thresholds=detector_thresholds,
            healthy_scores=healthy,
            sensor_groups=groups,
            config=cfg,
            seed=int(seed),
            gru_epochs=int(gru_epochs),
            ae_epochs=int(ae_epochs),
            training_rows=int(clean.shape[0]),
            input_sha256=str(metadata["input_sha256"]),
            metadata=metadata,
        )

    def _stream_scores(self, raw: np.ndarray) -> dict[str, np.ndarray]:
        """Build one full-length causal score vector per detector."""

        n = int(raw.shape[0])
        full = {canonical: np.full(n, np.nan, dtype=float) for canonical in _SCORE_ORDER}
        std = _standardized_stream(self.trained, raw)
        seq_len = int(getattr(self.trained, "seq_len", 12))
        residuals, residual_start = one_step_residual_stream(
            self.trained.predict_fn, std, seq_len
        )
        ref = getattr(self.trained, "ref", None)
        if ref is not None and hasattr(ref, "scale"):
            cpdz = cpdz_series(residuals, np.asarray(ref.scale, dtype=float))
            full["cpdz_residual"][residual_start:] = cpdz
            try:
                gepfm = gepfm_series(
                    cpdz,
                    reference_mean=float(ref.gepfm_ref_mean),
                    reference_scale=float(ref.gepfm_ref_scale),
                    smoothing=float(ref.gepfm_smoothing),
                )
                full["gepfm_drift"][residual_start:] = gepfm
            except (AttributeError, ValueError, TypeError):
                pass
            try:
                rat = rat_series(
                    residuals,
                    np.asarray(ref.rat_ref_cov, dtype=float),
                    window=int(getattr(self.trained, "rat_window", 60)),
                )
                full["rat_covariance"][residual_start:] = rat
            except (AttributeError, ValueError, TypeError):
                pass

        for canonical in ("mahalanobis", "pca_residual", "autoencoder"):
            detector = _detector_from_bundle(self.trained, canonical)
            if detector is None or not hasattr(detector, "score_samples"):
                continue
            values = np.asarray(detector.score_samples(raw), dtype=float).reshape(-1)
            if values.shape != (n,):
                raise ValueError(f"{canonical} detector returned {values.shape}, expected {(n,)}")
            full[canonical] = values

        values = np.asarray(self.isolation_forest.score_samples(raw), dtype=float).reshape(-1)
        if values.shape != (n,):
            raise ValueError(f"isolation_forest detector returned {values.shape}, expected {(n,)}")
        full["isolation_forest"] = values
        return full

    def _included_scores(self, ablation: str) -> tuple[str, ...]:
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {sorted(ABLATIONS)}")
        names = list(_SCORE_ORDER)
        if ablation == "no_temporal":
            names = [name for name in names if name not in {"gepfm_drift", "rat_covariance"}]
        elif ablation == "no_instantaneous":
            names = [name for name in names if name not in set(_INSTANTANEOUS)]
        return tuple(
            name
            for name in names
            if name == "cpdz_residual"
            or self.thresholds.get(name) is not None
            or name == "isolation_forest"
        )

    def _sensor_context(
        self, raw: np.ndarray, index: int, *, include: bool
    ) -> dict[str, Any] | None:
        if not include:
            return None
        std = _standardized_stream(self.trained, raw)
        row = np.asarray(std[index], dtype=float)
        context_groups: dict[str, Any] = {}
        positions = {name: idx for idx, name in enumerate(self.feature_names)}
        for group, columns in self.sensor_groups.items():
            values = np.asarray([row[positions[column]] for column in columns], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                context_groups[group] = {"mean_abs_z": UNAVAILABLE, "max_abs_z": UNAVAILABLE}
            else:
                context_groups[group] = {
                    "mean_abs_z": float(np.mean(np.abs(values))),
                    "max_abs_z": float(np.max(np.abs(values))),
                }
        order = np.argsort(-np.abs(row), kind="mergesort")
        top: list[dict[str, Any]] = []
        for position in order[:4]:
            value = _finite_or_none(row[int(position)])
            if value is None:
                continue
            top.append({"channel": self.feature_names[int(position)], "z": float(value)})
        return {"groups": context_groups, "top_deviations": top}

    def _detector_state(
        self,
        canonical: str,
        values: np.ndarray,
        index: int,
        start: int,
    ) -> tuple[dict[str, Any], dict[str, float | None]]:
        threshold = self.thresholds.get(canonical)
        latest = _finite_or_none(values[index])
        window_values = np.asarray(values[start : index + 1], dtype=float)
        finite = window_values[np.isfinite(window_values)]
        if latest is None:
            return (
                {
                    "state": "unavailable",
                    "value": UNAVAILABLE,
                    "threshold": threshold if threshold is not None else UNAVAILABLE,
                    "percentile": UNAVAILABLE,
                    "tail_fraction": UNAVAILABLE,
                    "window_alarm_fraction": UNAVAILABLE,
                },
                {
                    "latest": None,
                    "percentile": None,
                    "tail_fraction": None,
                    "window_alarm_fraction": None,
                },
            )
        reference = self.healthy_scores.get(canonical, np.array([], dtype=float))
        percentile = _empirical_percentile(reference, latest)
        tail = _tail_fraction(reference, latest)
        alarm_fraction = (
            float(np.mean(finite >= threshold))
            if finite.size and threshold is not None and math.isfinite(float(threshold))
            else None
        )
        alerting = bool(
            threshold is not None
            and math.isfinite(float(threshold))
            and latest >= float(threshold)
        )
        state = {
            "state": "alerting" if alerting else "nominal",
            "value": float(latest),
            "threshold": float(threshold) if threshold is not None else UNAVAILABLE,
            "percentile": float(percentile) if percentile is not None else UNAVAILABLE,
            "tail_fraction": float(tail) if tail is not None else UNAVAILABLE,
            "window_alarm_fraction": (
                float(alarm_fraction) if alarm_fraction is not None else UNAVAILABLE
            ),
        }
        numeric = {
            "latest": float(latest),
            "percentile": percentile,
            "tail_fraction": tail,
            "window_alarm_fraction": alarm_fraction,
        }
        return state, numeric

    def _build_state(
        self,
        raw: np.ndarray,
        scores: Mapping[str, np.ndarray],
        index: int,
        *,
        ablation: str,
    ) -> tuple[dict[str, Any], dict[str, float | None], dict[str, Any] | None]:
        start = max(0, int(index) - DEFAULT_WINDOW_STEPS + 1)
        included = self._included_scores(ablation)
        detectors: dict[str, Any] = {}
        numeric: dict[str, float | None] = {}
        for canonical in included:
            state, numeric_values = self._detector_state(
                canonical, scores[canonical], int(index), start
            )
            detectors[canonical] = state
            for suffix, value in numeric_values.items():
                numeric[f"{canonical}__{suffix}"] = value

        temporal_summary: dict[str, Any] = {}
        for canonical in ("gepfm_drift", "rat_covariance"):
            if canonical not in included:
                continue
            value = detectors[canonical]
            temporal_summary[canonical] = {
                "latest": value["value"],
                "window_alarm_fraction": value["window_alarm_fraction"],
            }

        context = self._sensor_context(
            raw, int(index), include=ablation != "no_sensor_context"
        )
        if context is not None:
            for group, values in context["groups"].items():
                for suffix in ("mean_abs_z", "max_abs_z"):
                    numeric[f"sensor__{group}__{suffix}"] = _finite_or_none(values[suffix])
            for rank, entry in enumerate(context["top_deviations"]):
                numeric[f"sensor__top_{rank}__abs_z"] = abs(float(entry["z"]))

        # Candidate routing is causal and window-local: a detector crossing at
        # any of the most recent six observations keeps the state eligible for
        # Jev, even if the newest observation has already fallen below threshold.
        candidate = any(
            value.get("window_alarm_fraction") not in (None, UNAVAILABLE)
            and float(value["window_alarm_fraction"]) > 0.0
            for value in detectors.values()
        )
        numeric["routing__candidate"] = float(bool(candidate))
        state: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "window": {"length_steps": DEFAULT_WINDOW_STEPS, "stride_steps": DEFAULT_STRIDE_STEPS},
            "freshness": {"is_fresh": True, "state_age_s": UNAVAILABLE},
            "detectors": detectors,
            "temporal_summary": temporal_summary,
            "hydraulic_consistency": {
                "state": UNAVAILABLE,
                "within_tolerance": UNAVAILABLE,
                "mass_balance_residual_m3_s": UNAVAILABLE,
                "demand": UNAVAILABLE,
                "reason": "BATADAL SCADA stream has no trusted demand or mass-balance channel",
            },
            "routing": {
                "candidate": bool(candidate),
                "candidate_rule": (
                    "at least one included detector crosses its clean threshold "
                    "within the recent six-step causal window"
                ),
            },
        }
        if context is not None:
            state["sensor_context"] = context
        return _json_safe(state), numeric, context

    def _numeric_matrix(
        self,
        raw: np.ndarray,
        scores: Mapping[str, np.ndarray],
        indices: np.ndarray,
        *,
        ablation: str,
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
        rows: list[dict[str, float | None]] = []
        for index in indices:
            _, numeric, _ = self._build_state(raw, scores, int(index), ablation=ablation)
            rows.append(numeric)
        names: list[str] = []
        for row in rows:
            for name in row:
                if name not in names:
                    names.append(name)
        if not names:
            return (
                np.empty((len(rows), 0), dtype=float),
                tuple(),
                np.empty((len(rows), 0), dtype=bool),
            )
        matrix = np.zeros((len(rows), len(names)), dtype=float)
        mask = np.zeros(matrix.shape, dtype=bool)
        for row_idx, row in enumerate(rows):
            for column_idx, name in enumerate(names):
                value = _finite_or_none(row.get(name))
                if value is not None:
                    matrix[row_idx, column_idx] = value
                    mask[row_idx, column_idx] = True
        return matrix, tuple(names), mask

    def transform(
        self,
        raw_features: np.ndarray,
        *,
        ablation: str = "full",
        stride: int = DEFAULT_STRIDE_STEPS,
    ) -> EncodedStream:
        """Score a raw stream and encode only six-step causal windows."""

        columns = getattr(raw_features, "columns", None)
        if columns is not None:
            forbidden = {
                "att_flag",
                "label",
                "ground_truth",
                "true_label",
                "timestamp",
                "datetime",
                "dataset",
                "row_id",
            }
            leaked = [
                str(column)
                for column in columns
                if str(column).strip().lower() in forbidden
            ]
            if leaked:
                raise ValueError(
                    "raw_features must not include labels or row metadata: "
                    + ", ".join(leaked)
                )
        raw = _as_finite_matrix(raw_features, "raw_features")
        if raw.shape[1] != len(self.feature_names):
            raise ValueError("raw_features width does not match fitted feature_names")
        if int(stride) < 1:
            raise ValueError("stride must be a positive integer")
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; expected one of {sorted(ABLATIONS)}")

        scores = self._stream_scores(raw)
        seq_len = int(getattr(self.trained, "seq_len", 12))
        rat_window = int(getattr(self.trained, "rat_window", 60))
        warmup = int(seq_len + rat_window - 1)
        # The first deployable state is exactly 71 for BATADAL (12 + 60 - 1).
        # Keep the formula so a custom fitted bundle remains internally causal.
        indices = np.arange(warmup, len(raw), int(stride), dtype=int)
        states: list[dict[str, Any]] = []
        for index in indices:
            state, _, _ = self._build_state(raw, scores, int(index), ablation=ablation)
            states.append(state)
        numeric, names, mask = self._numeric_matrix(
            raw, scores, indices, ablation=ablation
        )
        metadata = dict(self.metadata)
        metadata.update(
            {
                "transform_schema": SCHEMA_VERSION,
                "ablation": ablation,
                "stride_steps": int(stride),
                "window_steps": DEFAULT_WINDOW_STEPS,
                "warmup_start_index": int(warmup),
                "raw_rows": int(len(raw)),
                "raw_input_sha256": _hash_matrix(raw, self.feature_names),
                "state_count": int(len(indices)),
                "included_scores": list(self._included_scores(ablation)),
            }
        )
        included_scores = self._included_scores(ablation)
        return EncodedStream(
            scores={
                name: np.asarray(scores[name], dtype=float)
                for name in included_scores
            },
            state_indices=indices.copy(),
            states=tuple(states),
            numeric_features=numeric,
            numeric_feature_names=names,
            numeric_feature_mask=mask,
            metadata=_json_safe(metadata),
        )

    def save(self, path: str | Path) -> None:
        """Persist the fitted encoder using cloudpickle."""

        try:
            import cloudpickle
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("cloudpickle is required to save a fitted encoder") from exc
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            cloudpickle.dump(self, handle, protocol=cloudpickle.DEFAULT_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "HydroJEVEvidenceEncoder":
        """Load a cloudpickle-persisted fitted encoder."""

        try:
            import cloudpickle
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("cloudpickle is required to load a fitted encoder") from exc
        with Path(path).open("rb") as handle:
            value = cloudpickle.load(handle)
        if not isinstance(value, cls):
            raise TypeError("persisted object is not a HydroJEVEvidenceEncoder")
        return value


__all__ = [
    "ABLATIONS",
    "EncodedStream",
    "HydroJEVEvidenceEncoder",
    "SCHEMA_VERSION",
]

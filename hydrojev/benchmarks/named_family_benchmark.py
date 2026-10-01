"""Faithful benchmark of HydroJEV's *named* orthogonal residual families.

The evasion/transfer benchmarks (:mod:`evasion_benchmark`, :mod:`transfer_benchmark`,
:mod:`multiseed_transfer`) demonstrate the orthogonality *principle* with fair,
instantaneous detectors (a learned autoencoder, a Mahalanobis/covariance view, a
PCA/subspace view). Those linear views stand in as *proxies* for the architecture's
own named families. This module closes that faithfulness gap for the **detection**
claim: it wires the architecture's actual three residual families and measures
whether they are (a) credible detectors and (b) genuinely orthogonal.

The three families all sit on one shared **surrogate residual stream**. A compact
GRU (:mod:`hydrojev.physics.surrogate_models`, the paper's surrogate) is trained on
attack-free ``dataset03`` to predict the current standardized sensor vector from the
preceding ``sequence_length`` timesteps (one-step-ahead autoregression). The residual
``r_t = x_t - \\hat x_t`` then feeds:

* **Normalized CPDZ residual** (primary, instantaneous on the residual): RMS of the
  per-channel residual standardized by a robust healthy scale
  (:func:`~hydrojev.state_projection.residual_engine.normalized_cpdz_residual`).
* **GEpFM baseline drift** (auxiliary, slow-drift): standardized drift of the EWMA
  of the CPDZ level from its healthy mean
  (:func:`~hydrojev.state_projection.residual_engine.gepfm_baseline_drift`).
* **RAT covariance distortion** (auxiliary, structural): normalized Frobenius
  distance of the residual covariance over a trailing window from a healthy
  reference covariance
  (:func:`~hydrojev.state_projection.residual_engine.rat_covariance_distortion`).

Because CPDZ predicts *from the past* and GEpFM/RAT integrate *over time*, these are
**temporal** detectors — orthogonal by construction to the instantaneous AE/Maha/PCA
views, and (equally by construction) NOT expressible through the per-sample
coordinate-descent evasion attack, which perturbs a single instantaneous vector.
Their contribution is therefore measured here at the *detection* level: fair
cross-dataset detection metrics (fit on dataset03, threshold on the training score
distribution, evaluate labelled dataset04 — the identical policy as the clean
benchmark) plus an explicit **orthogonality** analysis (score rank-correlation and
complementary-scenario catch) against the instantaneous families.

Everything is leakage-safe (standardizer, surrogate, robust scale, references, and
all thresholds are estimated on dataset03 only) and the runner returns
``status="not_run"`` rather than fabricating numbers when BATADAL is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hydrojev.config import HydroJEVConfig
from hydrojev.benchmarks.detection_baselines import (
    calibrate_threshold,
    contiguous_runs,
    evaluate_scores,
)
from hydrojev.state_projection.residual_engine import (
    ewma_level,
    gepfm_baseline_drift,
    normalized_cpdz_residual,
    rat_covariance_distortion,
    robust_feature_scale,
)

PredictFn = Callable[[np.ndarray], np.ndarray]
# perturb(raw_eval[n_samples, n_features], columns) -> degraded raw_eval (same shape).
PerturbFn = Callable[[np.ndarray, list[str]], np.ndarray]


# --------------------------------------------------------------------------- #
# Pure, unit-testable stream helpers (no torch, no EPANET)
# --------------------------------------------------------------------------- #


def one_step_residual_stream(
    predict_fn: PredictFn, standardized_stream: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, int]:
    """Residuals ``x_t - predict_fn(window ending at t-1)`` over a time-ordered stream.

    ``predict_fn`` maps a batch of windows ``[n, sequence_length, features]`` to a
    batch of predicted vectors ``[n, features]`` for the timestep *after* each
    window. Returns ``(residuals[n, features], start_index)`` where ``start_index``
    is the absolute stream index of the first residual (``= sequence_length``); rows
    ``0 .. sequence_length-1`` are warmup context and have no residual.
    """

    stream = np.asarray(standardized_stream, dtype=float)
    if stream.ndim != 2:
        raise ValueError("standardized_stream must be 2D [samples, features]")
    n, features = stream.shape
    if not 1 <= sequence_length < n:
        raise ValueError("sequence_length must satisfy 1 <= L < n_samples")
    count = n - sequence_length
    windows = np.stack(
        [stream[k : k + sequence_length] for k in range(count)], axis=0
    )
    predicted = np.asarray(predict_fn(windows), dtype=float)
    if predicted.shape != (count, features):
        raise ValueError(
            f"predict_fn returned {predicted.shape}, expected {(count, features)}"
        )
    observed = stream[sequence_length:]
    return observed - predicted, int(sequence_length)


def cpdz_series(residuals: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-timestep normalized CPDZ residual (RMS standardized residual)."""

    res = np.asarray(residuals, dtype=float)
    if res.ndim != 2:
        raise ValueError("residuals must be 2D [samples, features]")
    return np.array(
        [normalized_cpdz_residual(row, np.zeros_like(row), scale) for row in res],
        dtype=float,
    )


def gepfm_series(
    level_series: np.ndarray,
    *,
    reference_mean: float,
    reference_scale: float,
    smoothing: float = 0.2,
) -> np.ndarray:
    """Causal per-timestep GEpFM drift of an online EWMA of ``level_series``.

    Element ``t`` equals the standardized drift of the EWMA over the prefix
    ``level_series[: t + 1]`` — i.e. exactly what
    :func:`gepfm_baseline_drift` reports on that prefix, computed online.
    """

    values = np.asarray(level_series, dtype=float)
    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("level_series must be a non-empty 1D array")
    if reference_scale <= 0:
        raise ValueError("reference_scale must be positive")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must lie in (0, 1]")
    out = np.empty(values.shape[0], dtype=float)
    level = float(values[0])
    out[0] = abs(level - reference_mean) / reference_scale
    for t in range(1, values.shape[0]):
        level = smoothing * float(values[t]) + (1.0 - smoothing) * level
        out[t] = abs(level - reference_mean) / reference_scale
    return out


def rat_series(
    residuals: np.ndarray,
    reference_covariance: np.ndarray,
    *,
    window: int,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Causal per-timestep RAT covariance distortion over a trailing window.

    Element ``t`` is the normalized Frobenius distortion of the residual
    covariance over ``residuals[t - window + 1 : t + 1]`` from the reference; the
    first ``window - 1`` elements (and any window the engine deems too short) are
    ``nan``.
    """

    res = np.asarray(residuals, dtype=float)
    if res.ndim != 2:
        raise ValueError("residuals must be 2D [samples, features]")
    n = res.shape[0]
    if not 2 <= window <= n:
        raise ValueError("window must satisfy 2 <= window <= n_samples")
    out = np.full(n, np.nan, dtype=float)
    for t in range(window - 1, n):
        reading = rat_covariance_distortion(
            res[t - window + 1 : t + 1],
            reference_covariance,
            ridge=ridge,
            minimum_samples=window,
        )
        if reading.value is not None:
            out[t] = float(reading.value)
    return out


@dataclass(frozen=True)
class FamilyReference:
    """Healthy (dataset03) statistics that define all three families' scores.

    Everything here is estimated on attack-free data only, so scoring any other
    stream against it is leakage-safe.
    """

    scale: np.ndarray  # robust per-feature residual scale (CPDZ standardizer)
    gepfm_ref_mean: float
    gepfm_ref_scale: float
    gepfm_smoothing: float
    rat_ref_cov: np.ndarray
    rat_window: int
    clean_cpdz: np.ndarray
    clean_gepfm: np.ndarray
    clean_rat: np.ndarray


def healthy_family_reference(
    predict_fn: PredictFn,
    clean_standardized: np.ndarray,
    *,
    seq_len: int,
    rat_window: int,
    gepfm_smoothing: float = 0.2,
) -> FamilyReference:
    """Fit the three families' healthy references from an attack-free stream."""

    clean_res, _ = one_step_residual_stream(predict_fn, clean_standardized, seq_len)
    scale = robust_feature_scale(clean_res)
    clean_cpdz = cpdz_series(clean_res, scale)
    ref_mean = float(np.mean(clean_cpdz))
    ref_scale = float(np.std(clean_cpdz)) or 1.0
    ref_cov = np.cov(clean_res, rowvar=False)
    clean_gepfm = gepfm_series(
        clean_cpdz, reference_mean=ref_mean, reference_scale=ref_scale, smoothing=gepfm_smoothing
    )
    clean_rat = rat_series(clean_res, ref_cov, window=rat_window)
    return FamilyReference(
        scale=scale,
        gepfm_ref_mean=ref_mean,
        gepfm_ref_scale=ref_scale,
        gepfm_smoothing=gepfm_smoothing,
        rat_ref_cov=ref_cov,
        rat_window=int(rat_window),
        clean_cpdz=clean_cpdz,
        clean_gepfm=clean_gepfm,
        clean_rat=clean_rat,
    )


def spearman_matrix(score_columns: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Pairwise Spearman rank correlation among named score series.

    Rows with a NaN in *either* member of a pair are dropped for that pair, so a
    family with a longer warmup still contributes on its valid range.
    """

    names = list(score_columns)
    arrays = {k: np.asarray(v, dtype=float) for k, v in score_columns.items()}
    out: dict[str, dict[str, float]] = {}
    for a in names:
        out[a] = {}
        for b in names:
            xa, xb = arrays[a], arrays[b]
            ok = np.isfinite(xa) & np.isfinite(xb)
            if int(ok.sum()) < 3:
                out[a][b] = float("nan")
                continue
            out[a][b] = _spearman(xa[ok], xb[ok])
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via Pearson on average ranks (ties handled)."""

    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (1..n), ties averaged — a compact stand-in for scipy.rankdata."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=float)
    # Average tied groups.
    sorted_v = values[order]
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def scenario_alarms(scores: np.ndarray, threshold: float, labels: np.ndarray) -> list[bool]:
    """Per-attack-scenario detection flags (a run counts if any sample alarms)."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    alerting = np.where(np.isfinite(scores), scores, -np.inf) >= threshold
    return [bool(alerting[a:b].any()) for (a, b) in contiguous_runs(labels, 1)]


def complementary_catch(
    a_alarms: Sequence[bool], b_alarms: Sequence[bool]
) -> dict[str, Any]:
    """How the two detectors' per-scenario catches complement each other.

    ``a_only`` = scenarios ``a`` catches that ``b`` misses (the orthogonal value of
    ``a`` over ``b``); ``union_recall`` = scenarios either catches.
    """

    a = np.asarray(a_alarms, dtype=bool)
    b = np.asarray(b_alarms, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("alarm vectors must share length")
    n = int(a.shape[0])
    if n == 0:
        return {"n_scenarios": 0, "a_only": 0, "b_only": 0, "both": 0,
                "union_recall": float("nan"), "a_recall": float("nan"), "b_recall": float("nan")}
    return {
        "n_scenarios": n,
        "a_only": int(np.sum(a & ~b)),
        "b_only": int(np.sum(b & ~a)),
        "both": int(np.sum(a & b)),
        "a_recall": float(np.mean(a)),
        "b_recall": float(np.mean(b)),
        "union_recall": float(np.mean(a | b)),
    }


# --------------------------------------------------------------------------- #
# Fitted family container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NamedFamilyScores:
    """Aligned, warmup-trimmed per-timestep scores + thresholds for every family.

    ``scores`` and ``labels`` are all indexed identically (absolute stream indices
    ``start_index ..``), so any pair is directly comparable and every detection
    metric is computed on the same rows.
    """

    start_index: int
    labels: np.ndarray
    scores: dict[str, np.ndarray]
    thresholds: dict[str, float]
    families: tuple[str, ...]  # temporal named families
    references: tuple[str, ...]  # instantaneous reference detectors


@dataclass(frozen=True)
class TrainedNamedFamilies:
    """Everything fit on attack-free ``dataset03`` and then held FIXED at deployment.

    Splitting the fitted artifacts out of :func:`build_named_family_scores` lets a
    caller train the surrogate + references + thresholds *once* and then score many
    different (possibly perturbed) evaluation streams against the identical detector
    — the deployed-detector story a robustness study needs. Holds live torch model
    references, so it is in-process only.
    """

    columns: list[str]
    seq_len: int
    rat_window: int
    gepfm_smoothing: float
    standardizer: Any
    predict_fn: PredictFn
    ref: FamilyReference
    thresholds: dict[str, float]
    maha: Any
    pca: Any
    ae: Any  # optional (may be None)
    families: tuple[str, ...]
    references: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Real-data runner
# --------------------------------------------------------------------------- #


def _torch_predict_fn(model: Any, device: Any) -> PredictFn:
    from hydrojev.physics.surrogate_models import predict_surrogate

    def predict(windows: np.ndarray) -> np.ndarray:
        return predict_surrogate(model, windows, device)

    return predict


def train_named_families(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    rat_window: int | None = None,
    gepfm_smoothing: float = 0.2,
    max_epochs: int = 60,
) -> TrainedNamedFamilies | dict[str, Any]:
    """Fit the GRU surrogate + healthy references + thresholds on attack-free ``dataset03``.

    Returns a :class:`TrainedNamedFamilies` bundle (all deployment-frozen artifacts) or
    a ``{"status": "not_run"}`` dict when BATADAL/torch is unavailable. Split out of
    :func:`build_named_family_scores` so a caller can train ONCE and then score many
    (possibly degraded) evaluation streams against the identical fixed detector.
    """

    from hydrojev.datasets.wdn_loader import DatasetUnavailableError, load_batadal

    try:
        clean = load_batadal("dataset03", config=config)
    except DatasetUnavailableError as exc:
        return {"status": "not_run", "reason": str(exc)}

    columns = list(clean.feature_columns)

    import torch  # noqa: F401  (device probing happens inside surrogate_models)

    from hydrojev.physics.surrogate_models import (
        Standardizer,
        build_sequence_dataset,
        select_torch_device,
        train_surrogate,
    )

    seed = int(config.experiments.seed if seed is None else seed)
    seq_len = int(config.surrogate.sequence_length)
    n_features = len(columns)
    rat_window = int(rat_window if rat_window is not None else max(60, n_features + 1))

    # ----- standardize on dataset03 (leakage-safe) --------------------------- #
    clean_train = clean.train_frame[columns].to_numpy(dtype=float)
    rows = mode.batadal_train_rows
    if rows is not None:
        clean_train = clean_train[:rows]
    if clean_train.shape[0] <= seq_len + rat_window + 8:
        return {"status": "not_run", "reason": "attack-free stream too short for the surrogate window"}
    standardizer = Standardizer.fit(clean_train)
    clean_std = standardizer.transform(clean_train)

    # ----- train one-step-ahead surrogate on dataset03 ----------------------- #
    # windows[k] = clean_std[k : k+L] predict clean_std[k+L]  (one step ahead).
    inputs, targets = clean_std[:-1], clean_std[1:]
    windows, aligned = build_sequence_dataset(inputs, targets, seq_len)
    split = int(len(windows) * 0.85)
    if split < 1 or split >= len(windows):
        return {"status": "not_run", "reason": "attack-free stream too short to split train/val"}
    device = select_torch_device()
    result = train_surrogate(
        windows[:split], aligned[:split], windows[split:], aligned[split:],
        hidden_size=config.surrogate.hidden_size,
        num_layers=config.surrogate.num_layers,
        dense_size=config.surrogate.dense_size,
        max_epochs=max_epochs, seed=seed, device=device,
    )
    predict_fn = _torch_predict_fn(result.model, result.device)

    # ----- healthy references + thresholds (dataset03 only) ------------------ #
    ref = healthy_family_reference(
        predict_fn, clean_std, seq_len=seq_len, rat_window=rat_window, gepfm_smoothing=gepfm_smoothing
    )
    q = float(config.concealment.threshold_quantile)
    thresholds: dict[str, float] = {
        "CPDZ residual": calibrate_threshold(ref.clean_cpdz, q),
        "GEpFM drift": calibrate_threshold(ref.clean_gepfm, q),
        "RAT covariance": calibrate_threshold(ref.clean_rat[np.isfinite(ref.clean_rat)], q),
    }

    # ----- instantaneous reference detectors (fit on dataset03 raw) ---------- #
    from hydrojev.benchmarks.detection_baselines import (
        fit_mahalanobis_detector,
        fit_pca_residual_detector,
    )

    maha = fit_mahalanobis_detector(clean_train, threshold_quantile=q)
    pca = fit_pca_residual_detector(clean_train, threshold_quantile=q)
    thresholds["Mahalanobis"] = float(maha.threshold)
    thresholds["PCA residual"] = float(pca.threshold)
    references = ["Mahalanobis", "PCA residual"]
    ae = None
    try:
        import dataclasses

        from hydrojev.run_experiments import build_reconstruction_detector

        cfg_seed = dataclasses.replace(
            config, experiments=dataclasses.replace(config.experiments, seed=seed)
        )
        ae = build_reconstruction_detector(cfg_seed, mode=mode).detector
    except Exception:  # torch AE is optional context; families/linear refs are the core.
        ae = None
    if ae is not None:
        thresholds["HydroJEV AE"] = float(ae.threshold)
        references.append("HydroJEV AE")

    return TrainedNamedFamilies(
        columns=columns,
        seq_len=seq_len,
        rat_window=rat_window,
        gepfm_smoothing=gepfm_smoothing,
        standardizer=standardizer,
        predict_fn=predict_fn,
        ref=ref,
        thresholds=thresholds,
        maha=maha,
        pca=pca,
        ae=ae,
        families=("CPDZ residual", "GEpFM drift", "RAT covariance"),
        references=tuple(references),
    )


def score_named_families(
    trained: TrainedNamedFamilies,
    config: HydroJEVConfig,
    *,
    evaluate_variant: str = "dataset04",
    perturb_eval: PerturbFn | None = None,
) -> NamedFamilyScores | dict[str, Any]:
    """Score a FIXED :class:`TrainedNamedFamilies` on ``evaluate_variant``.

    ``perturb_eval(raw_eval, columns) -> raw_eval'`` is applied to the RAW observed
    measurement matrix *at the source*, before standardization and before every
    detector reads it — so a sensor perturbation (noise, dropout) degrades the
    surrogate-residual families and the instantaneous references through the SAME
    feed, exactly as it would a deployed detector whose training is long frozen.
    Nothing is re-fit here. Returns a :class:`NamedFamilyScores` or ``not_run``.
    """

    from hydrojev.datasets.wdn_loader import DatasetUnavailableError, load_batadal

    try:
        labelled = load_batadal(evaluate_variant, config=config)
    except DatasetUnavailableError as exc:
        return {"status": "not_run", "reason": str(exc)}
    if labelled.label_column is None:
        return {"status": "not_run", "reason": f"{labelled.name} has no attack-label column"}

    columns = trained.columns
    missing = [c for c in columns if c not in labelled.frame.columns]
    if missing:
        return {"status": "not_run", "reason": f"feature columns absent in {labelled.name}: {missing[:3]}"}

    raw_eval = labelled.frame[columns].to_numpy(dtype=float)
    if perturb_eval is not None:
        raw_eval = np.asarray(perturb_eval(raw_eval, columns), dtype=float)
        if raw_eval.shape != (labelled.frame.shape[0], len(columns)):
            return {"status": "not_run", "reason": "perturb_eval changed the measurement matrix shape"}
    labels_full = labelled.frame[labelled.label_column].to_numpy().astype(int)

    eval_std = trained.standardizer.transform(raw_eval)
    seq_len, rat_window = trained.seq_len, trained.rat_window
    ref, gepfm_smoothing = trained.ref, trained.gepfm_smoothing

    eval_res, start = one_step_residual_stream(trained.predict_fn, eval_std, seq_len)
    eval_cpdz = cpdz_series(eval_res, ref.scale)
    eval_gepfm = gepfm_series(eval_cpdz, reference_mean=ref.gepfm_ref_mean,
                              reference_scale=ref.gepfm_ref_scale, smoothing=gepfm_smoothing)
    eval_rat = rat_series(eval_res, ref.rat_ref_cov, window=rat_window)

    # Trim everything to the common valid range: skip the RAT warmup so every
    # family and reference is scored on an identical, fully-defined index set.
    trim = rat_window - 1  # RAT is the longest warmup within the residual stream
    idx = slice(trim, None)
    labels = labels_full[start:][idx]

    def ref_scores(det: Any) -> np.ndarray:
        return np.asarray(det.score_samples(raw_eval), dtype=float)[start:][idx]

    scores: dict[str, np.ndarray] = {
        "CPDZ residual": eval_cpdz[idx],
        "GEpFM drift": eval_gepfm[idx],
        "RAT covariance": eval_rat[idx],
        "Mahalanobis": ref_scores(trained.maha),
        "PCA residual": ref_scores(trained.pca),
    }
    if trained.ae is not None:
        scores["HydroJEV AE"] = ref_scores(trained.ae)

    return NamedFamilyScores(
        start_index=int(start + trim),
        labels=labels,
        scores=scores,
        thresholds=dict(trained.thresholds),
        families=trained.families,
        references=trained.references,
    )


def build_named_family_scores(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    rat_window: int | None = None,
    gepfm_smoothing: float = 0.2,
    max_epochs: int = 60,
    evaluate_variant: str = "dataset04",
    perturb_eval: PerturbFn | None = None,
) -> NamedFamilyScores | dict[str, Any]:
    """Train the surrogate on dataset03, score all families on ``evaluate_variant``.

    ``evaluate_variant`` defaults to ``"dataset04"`` (the labelled attack set used
    throughout §6); pass ``"test_dataset"`` to score the held-out BATADAL competition
    test set with the *identical* dataset03-trained surrogate, standardizer, healthy
    references and thresholds — leakage-safe by construction (nothing is fit on the
    evaluated set). ``perturb_eval`` optionally degrades the raw evaluation
    measurements at the source (see :func:`score_named_families`). Returns a
    :class:`NamedFamilyScores` on success, or a ``{"status": "not_run"}`` dict when
    BATADAL (or a required column) is unavailable.

    Thin wrapper over :func:`train_named_families` + :func:`score_named_families`;
    behavior is identical to the pre-split implementation.
    """

    trained = train_named_families(
        config, mode=mode, seed=seed, rat_window=rat_window,
        gepfm_smoothing=gepfm_smoothing, max_epochs=max_epochs,
    )
    if isinstance(trained, dict):  # not_run
        return trained
    return score_named_families(
        trained, config, evaluate_variant=evaluate_variant, perturb_eval=perturb_eval
    )


def run_named_family_benchmark(
    config: HydroJEVConfig,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 60,
) -> dict[str, Any]:
    """Detection metrics + measured orthogonality for the named residual families."""

    built = build_named_family_scores(config, mode=mode, seed=seed, max_epochs=max_epochs)
    if isinstance(built, dict):  # not_run
        return built

    labels = built.labels
    detection: dict[str, Any] = {}
    for name, sc in built.scores.items():
        thr = built.thresholds[name]
        finite = np.where(np.isfinite(sc), sc, np.nanmin(sc[np.isfinite(sc)]) - 1.0 if np.isfinite(sc).any() else 0.0)
        metrics = evaluate_scores(finite, thr, labels)
        metrics["is_named_family"] = name in built.families
        metrics["threshold"] = float(thr)
        detection[name] = metrics

    # Orthogonality: Spearman correlation of the score series (lower cross = more
    # orthogonal), and complementary-scenario catch of each named family over the
    # clean-benchmark winner (Mahalanobis).
    corr = spearman_matrix(built.scores)
    winner = "Mahalanobis"
    winner_alarms = scenario_alarms(built.scores[winner], built.thresholds[winner], labels)
    complementarity: dict[str, Any] = {}
    for fam in built.families:
        fam_alarms = scenario_alarms(built.scores[fam], built.thresholds[fam], labels)
        complementarity[fam] = complementary_catch(fam_alarms, winner_alarms)

    # OR of the three named families vs Mahalanobis: does the temporal stack cover
    # scenarios the strongest instantaneous detector misses?
    fam_alarm_mat = np.array(
        [scenario_alarms(built.scores[f], built.thresholds[f], labels) for f in built.families],
        dtype=bool,
    )
    named_union = fam_alarm_mat.any(axis=0) if fam_alarm_mat.size else np.zeros(0, dtype=bool)
    union_vs_winner = complementary_catch(list(named_union), winner_alarms)

    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "protocol": (
            "compact GRU one-step-ahead surrogate trained on attack-free dataset03; three "
            "named residual families (CPDZ instantaneous, GEpFM EWMA-drift, RAT covariance) on "
            "the shared residual stream; robust scale + references + all thresholds (99.5% "
            "quantile of the healthy score) estimated on dataset03; scored causally on labelled "
            "dataset04 with its true history; instantaneous Mahalanobis/PCA(/AE) as references; "
            "all series trimmed to one common fully-defined index set for a fair comparison"
        ),
        "seed": int(config.experiments.seed if seed is None else seed),
        "sequence_length": int(config.surrogate.sequence_length),
        "start_index": built.start_index,
        "n_scored_samples": int(labels.shape[0]),
        "n_attack_scenarios": int(len(contiguous_runs(labels, 1))),
        "families": list(built.families),
        "references": list(built.references),
        "detection": detection,
        "score_correlation_spearman": corr,
        "complementary_catch_over_mahalanobis": complementarity,
        "named_union_vs_mahalanobis": union_vs_winner,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_named_family_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    lines = [
        f"Named orthogonal residual families on {result['evaluate_dataset']} "
        f"({result['n_scored_samples']} samples, {result['n_attack_scenarios']} attack scenarios; "
        f"fit on {result['train_dataset']}).",
        "Detection (* = architecture's named family; others are instantaneous references):",
        "",
        f"{'detector':<18}{'ROC-AUC':>9}{'PR-AUC':>9}{'scen.rec':>10}{'pt.rec':>9}{'pt.FPR':>9}",
        "-" * 64,
    ]
    for name, m in result["detection"].items():
        tag = " *" if m.get("is_named_family") else "  "
        lines.append(
            f"{name + tag:<18}{m['roc_auc']:>9.3f}{m['pr_auc']:>9.3f}"
            f"{m['scenario_recall']:>10.3f}{m['point_recall']:>9.3f}"
            f"{m['point_false_positive_rate']:>9.3f}"
        )

    lines += ["", "Score rank-correlation (Spearman; lower between families and references = more orthogonal):", ""]
    names = list(result["score_correlation_spearman"].keys())
    lines.append(" " * 16 + "".join(f"{n[:10]:>12}" for n in names))
    for a in names:
        row = result["score_correlation_spearman"][a]
        cells = "".join("   n/a" if not np.isfinite(row[b]) else f"{row[b]:>12.2f}" for b in names)
        lines.append(f"{a[:16]:<16}{cells}")

    lines += ["", "Complementary scenario catch of each named family vs Mahalanobis (clean winner):", ""]
    lines.append(f"{'family':<18}{'family-only':>12}{'maha-only':>11}{'both':>7}{'union rec':>11}")
    lines.append("-" * 59)
    for fam, c in result["complementary_catch_over_mahalanobis"].items():
        lines.append(
            f"{fam:<18}{c['a_only']:>12}{c['b_only']:>11}{c['both']:>7}{c['union_recall']:>11.3f}"
        )
    u = result["named_union_vs_mahalanobis"]
    lines.append(
        f"{'UNION(3 families)':<18}{u['a_only']:>12}{u['b_only']:>11}{u['both']:>7}{u['union_recall']:>11.3f}"
    )
    return "\n".join(lines)


def plot_named_family(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    det = result["detection"]
    names = list(det.keys())
    corr_names = list(result["score_correlation_spearman"].keys())
    mat = np.array(
        [[result["score_correlation_spearman"][a][b] for b in corr_names] for a in corr_names],
        dtype=float,
    )

    fig, (axb, axm) = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})

    x = np.arange(len(names))
    roc = [det[n]["roc_auc"] for n in names]
    prc = [det[n]["pr_auc"] for n in names]
    fam = [det[n].get("is_named_family", False) for n in names]
    axb.bar(x - 0.2, roc, width=0.38, label="ROC-AUC",
            color=["#2166ac" if f else "#92c5de" for f in fam])
    axb.bar(x + 0.2, prc, width=0.38, label="PR-AUC",
            color=["#b2182b" if f else "#f4a582" for f in fam])
    axb.set_xticks(x)
    axb.set_xticklabels([n + (" *" if fam[i] else "") for i, n in enumerate(names)],
                        rotation=18, ha="right", fontsize=8)
    axb.set_ylim(0.0, 1.0)
    axb.set_ylabel("area under curve")
    axb.set_title("Named families are credible detectors\n(* = architecture's named residual family)", fontsize=9)
    axb.legend(fontsize=8)
    axb.grid(axis="y", alpha=0.3)

    im = axm.imshow(np.nan_to_num(mat, nan=0.0), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axm.set_xticks(range(len(corr_names)))
    axm.set_xticklabels([n.replace(" ", "\n") for n in corr_names], fontsize=7)
    axm.set_yticks(range(len(corr_names)))
    axm.set_yticklabels(corr_names, fontsize=7)
    for i in range(len(corr_names)):
        for j in range(len(corr_names)):
            v = mat[i, j]
            if np.isfinite(v):
                axm.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if abs(v) > 0.6 else "black", fontsize=7)
    axm.set_title("Score rank-correlation:\ntemporal families decorrelate from instantaneous views", fontsize=9)
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from hydrojev.config import load_config
    from hydrojev.run_experiments import RunMode

    parser = argparse.ArgumentParser(description="Named orthogonal residual families: detection + orthogonality")
    parser.add_argument("--quick", action="store_true", help="bounded fast smoke run")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--outdir", default="artifacts/named_family_benchmark")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    epochs = 12 if args.quick else args.max_epochs
    result = run_named_family_benchmark(config, mode=mode, seed=args.seed, max_epochs=epochs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "named_family.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if result.get("status") == "ok":
        table = format_named_family_table(result)
        (outdir / "named_family_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_named_family(result, str(outdir / "named_family.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'named_family.json'}, named_family_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

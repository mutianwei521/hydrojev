"""Pure metrics for the standalone Jev-driven HydroJEV benchmark.

The live runner emits a probability on a causal six-hour grid.  This module
keeps that grid product separate from the hourly operational product obtained
by carrying each grid value forward to the next grid point.  It never calls an
API, loads a data set, or uses attack onsets supplied by a caller.

Missing Jev probabilities stay missing.  They are counted as service failures
and treated as no alarm only for lower-bound operational counts.  They are
never imputed or back-filled.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


DEFAULT_GRID_STEP_HOURS = 6.0
DEFAULT_WARMUP_INDEX = 71


def _as_labels(values: Sequence[Any], *, name: str = "labels") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        return array.astype(np.int8)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain binary values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return numeric.astype(np.int8)


def _as_probabilities(values: Sequence[Any], *, name: str = "probabilities") -> np.ndarray:
    array = np.asarray(list(values), dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    output = np.full(array.shape, np.nan, dtype=float)
    for index, value in enumerate(array.tolist()):
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be in [0, 1] or null") from exc
        if not np.isfinite(number):
            continue
        if number < 0.0 or number > 1.0:
            raise ValueError(f"{name}[{index}] must be in [0, 1] or null")
        output[index] = number
    return output


def _finite_mask(probabilities: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(probabilities, dtype=float))


def contiguous_runs(values: Sequence[Any], value: int = 1) -> list[tuple[int, int]]:
    """Return half-open runs [start, end) equal to value."""

    labels = _as_labels(values, name="values")
    target = int(value)
    if target not in (0, 1):
        raise ValueError("value must be 0 or 1")
    padded = np.concatenate(([0], (labels == target).astype(np.int8), [0]))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return (float("nan"), float("nan"))
    if successes < 0 or successes > trials:
        raise ValueError("successes must lie in [0, trials]")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    margin = z / denominator * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _auc_ap(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float | None, float | None]:
    mask = _finite_mask(probabilities)
    if not np.any(mask):
        return None, None
    y = labels[mask]
    p = probabilities[mask]
    if np.unique(y).size < 2:
        return None, None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        return float(roc_auc_score(y, p)), float(average_precision_score(y, p))
    except Exception:
        positives = p[y == 1]
        negatives = p[y == 0]
        comparisons = (positives[:, None] > negatives[None, :]).sum()
        ties = (positives[:, None] == negatives[None, :]).sum()
        roc = float((comparisons + 0.5 * ties) / (len(positives) * len(negatives)))
        order = np.argsort(-p, kind="mergesort")
        ys = y[order]
        cumulative = np.cumsum(ys)
        ranks = np.arange(1, len(ys) + 1)
        ap = float(np.sum((cumulative / ranks) * ys) / max(1, int(ys.sum())))
        return roc, ap


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    mask = _finite_mask(probabilities)
    if not np.any(mask):
        return None
    return float(np.mean((probabilities[mask] - labels[mask]) ** 2))


def _ece(labels: np.ndarray, probabilities: np.ndarray, *, bins: int) -> tuple[float | None, list[dict[str, Any]]]:
    if bins < 1:
        raise ValueError("bins must be positive")
    mask = _finite_mask(probabilities)
    if not np.any(mask):
        return None, []
    y = labels[mask]
    p = probabilities[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    detail: list[dict[str, Any]] = []
    total = float(len(p))
    ece = 0.0
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        inside = (p >= lower) & ((p < upper) if index < bins - 1 else (p <= upper))
        count = int(np.sum(inside))
        if count == 0:
            detail.append({"bin": index, "lower": lower, "upper": upper, "count": 0,
                           "mean_probability": None, "empirical_frequency": None,
                           "absolute_gap": None})
            continue
        mean_probability = float(np.mean(p[inside]))
        empirical_frequency = float(np.mean(y[inside]))
        gap = abs(mean_probability - empirical_frequency)
        ece += count / total * gap
        detail.append({"bin": index, "lower": lower, "upper": upper, "count": count,
                       "mean_probability": mean_probability,
                       "empirical_frequency": empirical_frequency,
                       "absolute_gap": float(gap)})
    return float(ece), detail


def _binary_confusion(labels: np.ndarray, alarms: np.ndarray) -> dict[str, Any]:
    y = labels.astype(bool)
    a = alarms.astype(bool)
    tp = int(np.sum(y & a))
    fn = int(np.sum(y & ~a))
    tn = int(np.sum(~y & ~a))
    fp = int(np.sum(~y & a))
    tpr = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    tnr = tn / (tn + fp) if tn + fp else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = (2.0 * precision * tpr / (precision + tpr)) if precision is not None and tpr and precision + tpr else None
    accuracy = (tp + tn) / len(y) if len(y) else None
    return {"TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "TPR": tpr, "recall": tpr, "FPR": fpr, "TNR": tnr,
            "precision": precision, "F1": f1, "accuracy": accuracy,
            "n": int(len(y))}


def _ternary_summary(labels: np.ndarray, probabilities: np.ndarray, *, low: float, high: float) -> dict[str, Any]:
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("ternary thresholds must satisfy 0 <= low < high <= 1")
    finite = _finite_mask(probabilities)
    alarm = finite & (probabilities >= high)
    normal = finite & (probabilities <= low)
    abstain = finite & ~(alarm | normal)
    missing = ~finite
    decided = alarm | normal
    empty_y = np.array([], dtype=np.int8)
    empty_a = np.array([], dtype=bool)
    decided_confusion = _binary_confusion(labels[decided], alarm[decided]) if np.any(decided) else _binary_confusion(empty_y, empty_a)
    n = len(labels)
    return {"lower_threshold": float(low), "upper_threshold": float(high),
            "n": int(n), "n_alarm": int(np.sum(alarm)), "n_normal": int(np.sum(normal)),
            "n_abstain_gray": int(np.sum(abstain)), "n_missing": int(np.sum(missing)),
            "coverage": float(np.sum(decided) / n) if n else None,
            "abstention_rate": float(np.sum(abstain) / n) if n else None,
            "service_failure_rate": float(np.sum(missing) / n) if n else None,
            "decided_confusion": decided_confusion}


def binary_grid_metrics(labels: Sequence[Any], probabilities: Sequence[Any], *, threshold: float = 0.5, ece_bins: int = 10) -> dict[str, Any]:
    """Score binary Jev probabilities on the causal grid.

    ROC-AUC and AP are top-level values only for a complete grid.  When service
    failures occur, they are null and the available-subset values are reported
    with explicit coverage.  Brier/ECE use finite probabilities.  Missing values
    are lower-bound no alarms for threshold counts.
    """

    y = _as_labels(labels)
    p = _as_probabilities(probabilities)
    if len(y) != len(p):
        raise ValueError("labels and probabilities must have equal length")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    finite = _finite_mask(p)
    n = len(y)
    available = int(np.sum(finite))
    missing = n - available
    available_roc, available_ap = _auc_ap(y, p)
    brier = _brier(y, p)
    ece, ece_detail = _ece(y, p, bins=ece_bins)
    alarms = finite & (p >= threshold)
    empty_y = np.array([], dtype=np.int8)
    empty_a = np.array([], dtype=bool)
    available_confusion = _binary_confusion(y[finite], alarms[finite]) if available else _binary_confusion(empty_y, empty_a)
    lower_confusion = _binary_confusion(y, alarms)
    ternary = _ternary_summary(y, p, low=0.2, high=0.8)
    available_subset = {"n": available, "coverage": float(available / n) if n else None,
                        "missing": int(missing), "positive": int(np.sum(y[finite] == 1)),
                        "negative": int(np.sum(y[finite] == 0)), "roc_auc": available_roc,
                        "average_precision": available_ap, "brier_score": brier, "ece": ece,
                        "threshold_confusion": available_confusion}
    return {"metric_scope": "causal_grid", "n_grid_rows": n, "n_available": available,
            "n_missing": int(missing), "service_failure_count": int(missing),
            "coverage": float(available / n) if n else None, "complete_grid": bool(missing == 0),
            "roc_auc": available_roc if missing == 0 else None,
            "average_precision": available_ap if missing == 0 else None,
            "brier_score": brier, "ece": ece, "ece_bins": ece_detail,
            "threshold": float(threshold), "threshold_confusion": lower_confusion,
            "available_threshold_confusion": available_confusion,
            "lower_bound_alarm_count": int(np.sum(alarms)), "ternary": ternary,
            "available_subset": available_subset}


def forward_hold_grid_predictions(grid_indices: Sequence[int], grid_probabilities: Sequence[Any], *, raw_row_count: int, warmup_start: int = DEFAULT_WARMUP_INDEX) -> dict[str, Any]:
    """Forward-hold values to raw rows without back-filling or imputation."""

    raw_row_count = int(raw_row_count)
    warmup_start = int(warmup_start)
    if raw_row_count < 0 or warmup_start < 0 or warmup_start > raw_row_count:
        raise ValueError("raw_row_count and warmup_start are out of range")
    indices = np.asarray(list(grid_indices), dtype=int)
    probs = _as_probabilities(grid_probabilities)
    if indices.ndim != 1 or len(indices) != len(probs):
        raise ValueError("grid_indices and grid_probabilities must have equal length")
    if len(indices) and np.any(np.diff(indices) <= 0):
        raise ValueError("grid_indices must be strictly increasing")
    if len(indices) and (indices[0] < 0 or indices[-1] >= raw_row_count):
        raise ValueError("grid index lies outside raw row range")
    hourly = np.full(raw_row_count, np.nan, dtype=float)
    source_grid = np.full(raw_row_count, -1, dtype=int)
    failure = np.zeros(raw_row_count, dtype=bool)
    for position, start in enumerate(indices.tolist()):
        stop = int(indices[position + 1]) if position + 1 < len(indices) else raw_row_count
        if stop <= start:
            raise ValueError("grid intervals must be non-empty")
        source_grid[start:stop] = position
        if np.isfinite(probs[position]):
            hourly[start:stop] = probs[position]
        else:
            failure[start:stop] = True
    evaluated = np.arange(raw_row_count, dtype=int) >= warmup_start
    alarms = np.isfinite(hourly) & (hourly >= 0.5)
    alarms[~evaluated] = False
    return {"hourly_probabilities": hourly, "hourly_alarms": alarms,
            "source_grid_position": source_grid, "service_failure": failure,
            "evaluation_mask": evaluated, "raw_row_count": raw_row_count,
            "warmup_start": warmup_start, "excluded_prefix_rows": int(np.sum(~evaluated)),
            "grid_row_count": int(len(indices)), "grid_failure_count": int(np.sum(~np.isfinite(probs))),
            "hourly_missing_count": int(np.sum(evaluated & ~np.isfinite(hourly)))}


def reconstruct_online_hourly(grid_indices: Sequence[int], grid_probabilities: Sequence[Any], *, raw_row_count: int, warmup_start: int = DEFAULT_WARMUP_INDEX, threshold: float = 0.5) -> dict[str, Any]:
    """Forward-hold grid values and apply a binary threshold."""

    result = forward_hold_grid_predictions(grid_indices, grid_probabilities, raw_row_count=raw_row_count, warmup_start=warmup_start)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    probabilities = np.asarray(result["hourly_probabilities"], dtype=float)
    alarms = np.isfinite(probabilities) & (probabilities >= threshold)
    alarms[~np.asarray(result["evaluation_mask"], dtype=bool)] = False
    result["threshold"] = float(threshold)
    result["hourly_alarms"] = alarms
    result["lower_bound_alarm_count"] = int(np.sum(alarms))
    return result


def official_batadal_score(labels: Sequence[Any], alarms: Sequence[Any], *, gamma: float = 0.5) -> dict[str, Any]:
    """Compute the official BATADAL score from an aligned binary alarm grid."""

    y = _as_labels(labels)
    a = np.asarray(alarms, dtype=bool)
    if a.ndim != 1 or len(y) != len(a):
        raise ValueError("labels and alarms must be one-dimensional and equally long")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    runs = contiguous_runs(y, 1)
    per_attack: list[dict[str, Any]] = []
    normalized: list[float] = []
    for start, end in runs:
        duration = end - start
        hits = np.flatnonzero(a[start:end])
        ttd = int(hits[0]) if len(hits) else duration
        ratio = float(ttd / duration) if duration else 1.0
        normalized.append(ratio)
        per_attack.append({"start_index": int(start), "end_index": int(end), "duration_samples": int(duration),
                           "ttd_samples": int(ttd), "detected": bool(len(hits)), "normalized_ttd": ratio})
    s_ttd = float(1.0 - np.mean(normalized)) if normalized else None
    confusion = _binary_confusion(y, a)
    tpr, tnr = confusion["TPR"], confusion["TNR"]
    s_clf = ((tpr + tnr) / 2.0) if tpr is not None and tnr is not None else None
    score = (gamma * s_ttd + (1.0 - gamma) * s_clf) if s_ttd is not None and s_clf is not None else None
    return {"S": score, "S_TTD": s_ttd, "S_CLF": s_clf, "gamma": float(gamma),
            "classification": confusion, "per_attack": per_attack, "n_rows": int(len(y)),
            "n_attack_scenarios": int(len(runs))}


def _masked_runs(labels: np.ndarray, mask: np.ndarray, value: int) -> list[tuple[int, int]]:
    active = (labels == int(value)) & mask
    padded = np.concatenate(([False], active, [False]))
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def episode_metrics(labels: Sequence[Any], alarms: Sequence[Any], *, evaluation_mask: Sequence[Any] | None = None, probabilities: Sequence[Any] | None = None, sample_interval_hours: float = 1.0, false_alarm_horizon_days: float = 30.0) -> dict[str, Any]:
    """Derive attack runs from labels after inference and score hourly alarms."""

    y = _as_labels(labels)
    a = np.asarray(alarms, dtype=bool)
    if a.ndim != 1 or len(y) != len(a):
        raise ValueError("labels and alarms must be one-dimensional and equally long")
    if sample_interval_hours <= 0.0 or false_alarm_horizon_days <= 0.0:
        raise ValueError("time intervals and horizon must be positive")
    mask = np.ones(len(y), dtype=bool) if evaluation_mask is None else np.asarray(evaluation_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(y):
        raise ValueError("evaluation_mask must match labels")
    attack_runs = _masked_runs(y, mask, 1)
    benign = mask & (y == 0)
    attack = mask & (y == 1)
    per_attack: list[dict[str, Any]] = []
    detected_count = 0
    detected_ttd: list[float] = []
    for start, end in attack_runs:
        hits = np.flatnonzero(a[start:end])
        duration = end - start
        detected = bool(len(hits))
        ttd = int(hits[0]) if detected else int(duration)
        if detected:
            detected_count += 1
            detected_ttd.append(float(ttd * sample_interval_hours))
        per_attack.append({"start_index": int(start), "end_index": int(end),
                           "duration_samples": int(duration), "duration_hours": float(duration * sample_interval_hours),
                           "detected": detected, "ttd_samples": ttd,
                           "ttd_hours": float(ttd * sample_interval_hours) if detected else None,
                           "normalized_ttd": float(ttd / duration) if duration else 1.0})
    attack_count = len(attack_runs)
    benign_hours = float(np.sum(benign) * sample_interval_hours)
    false_alarm_hours = int(np.sum(benign & a))
    false_alarm_starts = 0
    for index in np.flatnonzero(benign & a):
        previous = int(index) - 1
        if previous < 0 or not (mask[previous] and a[previous]):
            false_alarm_starts += 1
    exposure_days = benign_hours / 24.0
    starts_per_horizon = (false_alarm_starts / exposure_days * false_alarm_horizon_days) if exposure_days else None
    missing_hours = None
    if probabilities is not None:
        p = _as_probabilities(probabilities)
        if len(p) != len(y):
            raise ValueError("probabilities must match labels")
        missing_hours = int(np.sum(mask & ~_finite_mask(p)))
    recall = detected_count / attack_count if attack_count else None
    ci = wilson_interval(detected_count, attack_count) if attack_count else (float("nan"), float("nan"))
    batadal = official_batadal_score(y[mask], a[mask]) if np.any(mask) else {"S": None, "S_TTD": None, "S_CLF": None, "gamma": 0.5, "classification": {}, "per_attack": [], "n_rows": 0, "n_attack_scenarios": 0}
    return {"metric_scope": "hourly_forward_hold", "n_rows": int(len(y)), "evaluated_rows": int(np.sum(mask)),
            "excluded_rows": int(np.sum(~mask)), "attack_hours": float(np.sum(attack) * sample_interval_hours),
            "benign_hours": benign_hours, "alarm_hours": int(np.sum(mask & a)),
            "attack_scenarios": int(attack_count), "detected_attack_scenarios": int(detected_count),
            "episode_recall": recall, "scenario_recall": recall,
            "episode_recall_wilson95": [float(ci[0]), float(ci[1])],
            "benign_hour_false_positive_rate": false_alarm_hours / int(np.sum(benign)) if np.sum(benign) else None,
            "false_positive_hours": false_alarm_hours, "false_alarm_event_starts": int(false_alarm_starts),
            "false_alarm_event_starts_per_30d": starts_per_horizon, "per_attack": per_attack,
            "mean_detected_ttd_hours": float(np.mean(detected_ttd)) if detected_ttd else None,
            "median_detected_ttd_hours": float(np.median(detected_ttd)) if detected_ttd else None,
            "missing_service_hours": missing_hours, "service_failure_count": missing_hours,
            "official_batadal": batadal,
            "missing_alarm_convention": "missing service probabilities are lower-bound no-alarm"}


def score_grid_and_online(raw_labels: Sequence[Any], grid_indices: Sequence[int], grid_probabilities: Sequence[Any], *, warmup_start: int = DEFAULT_WARMUP_INDEX, threshold: float = 0.5, ece_bins: int = 10) -> dict[str, Any]:
    """Return grid metrics and forward-held hourly metrics in one structure."""

    labels = _as_labels(raw_labels, name="raw_labels")
    indices = np.asarray(list(grid_indices), dtype=int)
    probs = _as_probabilities(grid_probabilities)
    if len(indices) != len(probs):
        raise ValueError("grid_indices and grid_probabilities must have equal length")
    if len(indices) and (np.any(indices < 0) or np.any(indices >= len(labels))):
        raise ValueError("grid index lies outside raw_labels")
    grid_mask = indices >= int(warmup_start)
    grid_labels = labels[indices[grid_mask]] if len(indices) else np.array([], dtype=np.int8)
    grid_probs = probs[grid_mask]
    grid_metrics = binary_grid_metrics(grid_labels, grid_probs, threshold=threshold, ece_bins=ece_bins)
    hourly = reconstruct_online_hourly(indices, probs, raw_row_count=len(labels), warmup_start=warmup_start, threshold=threshold)
    hourly_metrics = episode_metrics(labels, hourly["hourly_alarms"], evaluation_mask=hourly["evaluation_mask"], probabilities=hourly["hourly_probabilities"])
    return {"grid": grid_metrics, "hourly": hourly_metrics,
            "alignment": {"raw_row_count": int(len(labels)), "grid_row_count": int(np.sum(grid_mask)),
                          "grid_indices": indices.tolist(), "warmup_start": int(warmup_start),
                          "excluded_grid_prefix": int(np.sum(~grid_mask)), "forward_hold": True,
                          "backfill": False, "service_failure_count": int(np.sum(~_finite_mask(grid_probs)))},
            "hourly_arrays": hourly}


def _summary_metric(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    roc, ap = _auc_ap(labels, probabilities)
    return {"roc_auc": roc, "average_precision": ap, "brier_score": _brier(labels, probabilities)}


def _stream_metric_with_coverage(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Apply the complete-grid AUC rule to one probability stream."""

    finite = _finite_mask(probabilities)
    available = _summary_metric(labels[finite], probabilities[finite]) if np.any(finite) else {
        "roc_auc": None, "average_precision": None, "brier_score": None
    }
    complete = bool(np.all(finite))
    return {
        "complete_grid": complete,
        "n": int(len(labels)),
        "n_available": int(np.sum(finite)),
        "coverage": float(np.mean(finite)) if len(labels) else None,
        "roc_auc": available["roc_auc"] if complete else None,
        "average_precision": available["average_precision"] if complete else None,
        "brier_score": available["brier_score"],
        "available_subset": available,
    }


def moving_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Non-circular moving-block bootstrap indices of length n."""

    n = int(n)
    block_len = int(block_len)
    if n < 1 or block_len < 1 or block_len > n:
        raise ValueError(f"block_len must be in [1, n]={n}")
    starts = rng.integers(0, n - block_len + 1, size=int(math.ceil(n / block_len)))
    return (starts[:, None] + np.arange(block_len)[None, :]).reshape(-1)[:n]


def paired_moving_block_bootstrap(labels: Sequence[Any], full_probabilities: Sequence[Any], no_jev_probabilities: Sequence[Any], *, block_len: int = 8, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Paired moving-block CIs for full HydroJEV versus no-Jev probabilities."""

    y = _as_labels(labels)
    full = _as_probabilities(full_probabilities, name="full_probabilities")
    nojev = _as_probabilities(no_jev_probabilities, name="no_jev_probabilities")
    if len(y) != len(full) or len(y) != len(nojev):
        raise ValueError("labels and probability streams must have equal length")
    n = len(y)
    if int(n_boot) < 1:
        raise ValueError("n_boot must be positive")
    common = _finite_mask(full) & _finite_mask(nojev)
    status = "complete_grid" if bool(np.all(common)) else "available_subset"
    point_full = _stream_metric_with_coverage(y, full)
    point_nojev = _stream_metric_with_coverage(y, nojev)
    point_common_full = _summary_metric(y[common], full[common]) if np.any(common) else {"roc_auc": None, "average_precision": None, "brier_score": None}
    point_common_nojev = _summary_metric(y[common], nojev[common]) if np.any(common) else {"roc_auc": None, "average_precision": None, "brier_score": None}
    draws = {name: [] for name in ("roc_auc", "average_precision", "brier_score")}
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_boot)):
        index = moving_block_indices(n, int(block_len), rng)
        paired = common[index]
        if not np.any(paired):
            continue
        ys = y[index][paired]
        fs = full[index][paired]
        ns = nojev[index][paired]
        fm = _summary_metric(ys, fs)
        nm = _summary_metric(ys, ns)
        for name in draws:
            if fm[name] is not None and nm[name] is not None:
                draws[name].append(float(fm[name] - nm[name]))

    paired_results: dict[str, Any] = {}
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        difference = None
        if point_common_full[name] is not None and point_common_nojev[name] is not None:
            difference = float(point_common_full[name] - point_common_nojev[name])
        if array.size:
            ci = [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))]
        else:
            ci = [None, None]
        paired_results[name] = {"difference": difference, "ci95": ci, "n_valid": int(array.size)}
    return {"metric_scope": "paired_grid_moving_block_bootstrap", "status": status,
            "n_grid_rows": int(n), "n_common_available": int(np.sum(common)),
            "common_coverage": float(np.mean(common)) if n else None,
            "service_failure_full": int(np.sum(~_finite_mask(full))),
            "service_failure_no_jev": int(np.sum(~_finite_mask(nojev))),
            "block_len": int(block_len), "block_hours": float(block_len * DEFAULT_GRID_STEP_HOURS),
            "n_boot": int(n_boot), "seed": int(seed), "full": point_full, "no_jev": point_nojev,
            "paired_complete_case": {"full": point_common_full, "no_jev": point_common_nojev},
            "paired": paired_results,
            "note": "AUC/AP/Brier paired differences use complete cases; missing service rows are never imputed."}


score_grid = binary_grid_metrics
reconstruct_forward_hold = forward_hold_grid_predictions
score_episodes = episode_metrics
official_score = official_batadal_score
paired_bootstrap = paired_moving_block_bootstrap


__all__ = ["DEFAULT_GRID_STEP_HOURS", "DEFAULT_WARMUP_INDEX", "binary_grid_metrics", "contiguous_runs", "episode_metrics", "forward_hold_grid_predictions", "moving_block_indices", "official_batadal_score", "paired_moving_block_bootstrap", "reconstruct_forward_hold", "reconstruct_online_hourly", "score_episodes", "score_grid", "score_grid_and_online", "official_score", "paired_bootstrap", "wilson_interval"]

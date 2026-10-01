"""Detector-agnostic, constrained evasion attack for a fair robustness comparison.

The point of the detection benchmark is that several detectors reach high *clean*
recall. The paper's threat model, however, is a *stealthy* adversary who nudges a
few compromised sensors to slip a real attack past the detector while staying
within the range of normal operation. The scientific question is therefore not
"who detects best on clean data" but "who is hardest to evade".

To compare fairly, every detector is attacked by the *same* algorithm through the
*same* interface it already exposes — ``score_samples(X) -> higher-is-worse`` and a
``threshold``. The attack is a greedy, gradient-free coordinate descent that, at
each step, picks the single (sensor, value) change that most reduces the alarm
score, subject to:

* a sparsity cap (at most ``max_changed`` compromised sensors),
* an immutability mask (discrete pump/valve status channels are not spoofed here),
* a stay-in-range box (each spoofed value must lie within the per-sensor min/max
  observed in *normal* training data — this is what keeps the manipulation
  physically plausible and statistically stealthy), and
* a finite query budget.

Because it depends on nothing but the score function, it applies identically to
the reconstruction autoencoder and to every classical baseline, so evasion
success rates are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class Scorer(Protocol):
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EvasionResult:
    original_score: float
    adversarial_score: float
    threshold: float
    evaded: bool
    changed_indices: tuple[int, ...]
    n_changed: int
    query_count: int
    scaled_l2: float  # perturbation magnitude in units of per-feature training span
    termination_reason: str
    adversarial_sample: tuple[float, ...] = ()  # full spoofed feature vector


def evade_detector(
    scorer: Scorer,
    sample: np.ndarray,
    *,
    feature_low: np.ndarray,
    feature_high: np.ndarray,
    modifiable_mask: np.ndarray,
    max_changed: int = 4,
    candidate_values: int = 21,
    budget: int = 4000,
    tolerance: float = 1e-12,
) -> EvasionResult:
    """Greedily minimise ``scorer.score_samples`` for one sample under constraints.

    Returns the best adversarial sample found; ``evaded`` is True iff its score
    fell below the detector's threshold within the sparsity/range/query budget.
    """

    original = np.asarray(sample, dtype=float)
    low = np.asarray(feature_low, dtype=float)
    high = np.asarray(feature_high, dtype=float)
    mask = np.asarray(modifiable_mask, dtype=bool)
    if not (original.ndim == 1 and original.shape == low.shape == high.shape == mask.shape):
        raise ValueError("sample, bounds, and mask must share one-dimensional shape")
    if not np.isfinite(original).all():
        raise ValueError("sample contains NaN or infinity")
    if not 1 <= max_changed <= int(mask.sum() or 1):
        raise ValueError("max_changed must be between 1 and the number of modifiable features")
    if candidate_values < 2 or budget < 1:
        raise ValueError("candidate_values and budget must be positive")

    span = np.where((high - low) <= np.finfo(float).eps, 1.0, high - low)
    threshold = float(scorer.threshold)
    current = original.copy()
    original_score = float(scorer.score_samples(current[None, :])[0])
    current_score = original_score
    queries = 0
    reason = "no_improvement"

    while True:
        if current_score < threshold:
            reason = "threshold_reached"
            break
        if queries >= budget:
            reason = "budget_exhausted"
            break

        changed_now = ~np.isclose(current, original, rtol=1e-9, atol=1e-12)
        eligible = mask.copy()
        if int(changed_now.sum()) >= max_changed:
            # No new sensors may be compromised; only refine already-changed ones.
            eligible &= changed_now
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size == 0:
            reason = "no_modifiable_coordinate"
            break

        best_gain = tolerance
        best_coord: int | None = None
        best_value = 0.0
        best_score = current_score
        for coord in eligible_idx:
            if queries >= budget:
                break
            candidates = np.linspace(low[coord], high[coord], candidate_values)
            trials = np.repeat(current[None, :], candidates.size, axis=0)
            trials[:, coord] = candidates
            scores = np.asarray(scorer.score_samples(trials), dtype=float)
            queries += candidates.size
            j = int(np.argmin(scores))
            gain = current_score - float(scores[j])
            if gain > best_gain:
                best_gain = gain
                best_coord = int(coord)
                best_value = float(candidates[j])
                best_score = float(scores[j])

        if best_coord is None:
            reason = "no_improvement"
            break
        current[best_coord] = best_value
        current_score = best_score

    changed_indices = tuple(
        int(i) for i in np.flatnonzero(~np.isclose(current, original, rtol=1e-9, atol=1e-12))
    )
    scaled_l2 = float(np.sqrt(np.sum(((current - original) / span) ** 2)))
    return EvasionResult(
        original_score=original_score,
        adversarial_score=current_score,
        threshold=threshold,
        evaded=current_score < threshold,
        changed_indices=changed_indices,
        n_changed=len(changed_indices),
        query_count=queries,
        scaled_l2=scaled_l2,
        termination_reason=reason,
        adversarial_sample=tuple(float(v) for v in current),
    )


@dataclass(frozen=True)
class JointEvasionResult:
    """Outcome of attacking several detectors at once (an ensemble defence)."""

    detector_names: tuple[str, ...]
    original_margins: tuple[float, ...]  # (score - threshold) / scale, per detector
    adversarial_margins: tuple[float, ...]
    evaded_all: bool  # True iff every detector's score fell below its threshold
    n_detectors_evaded: int
    changed_indices: tuple[int, ...]
    n_changed: int
    query_count: int
    scaled_l2: float
    termination_reason: str
    adversarial_sample: tuple[float, ...] = ()  # full spoofed feature vector


def evade_detectors_jointly(
    scorers: Sequence[Scorer],
    sample: np.ndarray,
    *,
    scales: Sequence[float],
    names: Sequence[str] | None = None,
    feature_low: np.ndarray,
    feature_high: np.ndarray,
    modifiable_mask: np.ndarray,
    max_changed: int = 4,
    candidate_values: int = 21,
    budget: int = 4000,
    tolerance: float = 1e-12,
) -> JointEvasionResult:
    """Greedily evade an *ensemble*: drive every detector's score below threshold.

    The objective minimised at each step is the *total normalised threshold
    exceedance*, ``sum_k max(0, (score_k - threshold_k) / scale_k)`` -- it is zero
    iff every detector is already below its threshold, and any reduction of any
    still-alerting detector lowers it, so the attacker keeps making progress until
    the whole ensemble is satisfied (unlike a max-of-margins objective, which
    stalls whenever two detectors are tied at the top). ``scales`` (one positive
    number per detector, e.g. the detector's score std on the evaluation set)
    makes the per-detector margins comparable. ``evaded_all`` is True only if the
    final sample is below *every* threshold, under the shared sparsity, in-range,
    and query-budget constraints.
    """

    original = np.asarray(sample, dtype=float)
    low = np.asarray(feature_low, dtype=float)
    high = np.asarray(feature_high, dtype=float)
    mask = np.asarray(modifiable_mask, dtype=bool)
    scale_arr = np.asarray(scales, dtype=float)
    if not (original.ndim == 1 and original.shape == low.shape == high.shape == mask.shape):
        raise ValueError("sample, bounds, and mask must share one-dimensional shape")
    if len(scorers) < 1 or scale_arr.shape != (len(scorers),):
        raise ValueError("scales must provide one positive value per scorer")
    if np.any(scale_arr <= 0) or not np.isfinite(scale_arr).all():
        raise ValueError("scales must be positive and finite")
    if not np.isfinite(original).all():
        raise ValueError("sample contains NaN or infinity")
    if not 1 <= max_changed <= int(mask.sum() or 1):
        raise ValueError("max_changed must be between 1 and the number of modifiable features")
    if candidate_values < 2 or budget < 1:
        raise ValueError("candidate_values and budget must be positive")

    thresholds = np.array([float(s.threshold) for s in scorers], dtype=float)
    names_t = tuple(names) if names is not None else tuple(f"detector_{i}" for i in range(len(scorers)))
    span = np.where((high - low) <= np.finfo(float).eps, 1.0, high - low)

    def exceedance(batch: np.ndarray) -> np.ndarray:
        # (K, rows) normalised margins -> summed positive exceedance per row.
        margins = np.vstack(
            [(np.asarray(s.score_samples(batch), dtype=float) - thresholds[k]) / scale_arr[k]
             for k, s in enumerate(scorers)]
        )
        return np.maximum(0.0, margins).sum(axis=0)

    def margins_at(vec: np.ndarray) -> np.ndarray:
        row = vec[None, :]
        return np.array(
            [(float(scorers[k].score_samples(row)[0]) - thresholds[k]) / scale_arr[k]
             for k in range(len(scorers))]
        )

    current = original.copy()
    original_margins = margins_at(current)
    current_exceed = float(np.maximum(0.0, original_margins).sum())
    queries = 0
    reason = "no_improvement"

    while True:
        if bool(np.all(margins_at(current) < 0.0)):
            reason = "threshold_reached"
            break
        if queries >= budget:
            reason = "budget_exhausted"
            break

        changed_now = ~np.isclose(current, original, rtol=1e-9, atol=1e-12)
        eligible = mask.copy()
        if int(changed_now.sum()) >= max_changed:
            eligible &= changed_now
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size == 0:
            reason = "no_modifiable_coordinate"
            break

        best_gain = tolerance
        best_coord: int | None = None
        best_value = 0.0
        best_exceed = current_exceed
        for coord in eligible_idx:
            if queries >= budget:
                break
            candidates = np.linspace(low[coord], high[coord], candidate_values)
            trials = np.repeat(current[None, :], candidates.size, axis=0)
            trials[:, coord] = candidates
            exceed = exceedance(trials)
            queries += candidates.size
            j = int(np.argmin(exceed))
            gain = current_exceed - float(exceed[j])
            if gain > best_gain:
                best_gain = gain
                best_coord = int(coord)
                best_value = float(candidates[j])
                best_exceed = float(exceed[j])
        if best_coord is None:
            reason = "no_improvement"
            break
        current[best_coord] = best_value
        current_exceed = best_exceed

    adversarial_margins = margins_at(current)
    changed_indices = tuple(
        int(i) for i in np.flatnonzero(~np.isclose(current, original, rtol=1e-9, atol=1e-12))
    )
    scaled_l2 = float(np.sqrt(np.sum(((current - original) / span) ** 2)))
    return JointEvasionResult(
        detector_names=names_t,
        original_margins=tuple(float(m) for m in original_margins),
        adversarial_margins=tuple(float(m) for m in adversarial_margins),
        evaded_all=bool(np.all(adversarial_margins < 0.0)),
        n_detectors_evaded=int(np.sum(adversarial_margins < 0.0)),
        changed_indices=changed_indices,
        n_changed=len(changed_indices),
        query_count=queries,
        scaled_l2=scaled_l2,
        termination_reason=reason,
        adversarial_sample=tuple(float(v) for v in current),
    )

"""Unit tests for the detector-agnostic evasion attack on toy scorers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from hydrojev.benchmarks.evasion_attack import evade_detector, evade_detectors_jointly


@dataclass
class LinearScorer:
    """score = x @ weights (higher = more anomalous)."""

    weights: np.ndarray
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) @ self.weights


@dataclass
class QuadraticScorer:
    """score = (x[:, 0] - center) ** 2 — its minimiser can sit outside the box."""

    center: float
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        x0 = np.asarray(values, dtype=float)[:, 0]
        return (x0 - self.center) ** 2


def _box(n: int):
    return np.zeros(n), np.ones(n), np.ones(n, dtype=bool)


def test_evades_when_budget_permits() -> None:
    scorer = LinearScorer(weights=np.array([1.0, 1.0, 1.0]), threshold=1.5)
    low, high, mask = _box(3)
    res = evade_detector(
        scorer, np.ones(3), feature_low=low, feature_high=high,
        modifiable_mask=mask, max_changed=3, candidate_values=11,
    )
    assert res.original_score == pytest.approx(3.0)
    assert res.evaded is True
    assert res.adversarial_score < scorer.threshold
    assert res.termination_reason == "threshold_reached"
    assert 1 <= res.n_changed <= 3


def test_sparsity_cap_can_prevent_evasion() -> None:
    # Three equal contributors; one change (max 3->0) leaves score 2 > 1.5.
    scorer = LinearScorer(weights=np.array([1.0, 1.0, 1.0]), threshold=1.5)
    low, high, mask = _box(3)
    res = evade_detector(
        scorer, np.ones(3), feature_low=low, feature_high=high,
        modifiable_mask=mask, max_changed=1, candidate_values=11,
    )
    assert res.n_changed == 1
    assert res.evaded is False
    assert res.adversarial_score == pytest.approx(2.0)


def test_immutable_channels_block_evasion() -> None:
    # Only the immutable coordinate drives the score.
    scorer = LinearScorer(weights=np.array([0.0, 0.0, 1.0]), threshold=0.5)
    low, high = np.zeros(3), np.ones(3)
    mask = np.array([True, True, False])
    sample = np.array([0.0, 0.0, 1.0])
    res = evade_detector(
        scorer, sample, feature_low=low, feature_high=high,
        modifiable_mask=mask, max_changed=2,
    )
    assert res.evaded is False
    assert res.adversarial_score == pytest.approx(1.0)  # unchanged
    assert 2 not in res.changed_indices


def test_range_box_limits_reduction() -> None:
    # Minimiser at x0=5 is outside [0, 1]; best in-range score is 16 > 1.
    scorer = QuadraticScorer(center=5.0, threshold=1.0)
    low, high, mask = _box(2)
    res = evade_detector(
        scorer, np.zeros(2), feature_low=low, feature_high=high,
        modifiable_mask=mask, max_changed=2, candidate_values=21,
    )
    assert res.evaded is False
    assert res.adversarial_score == pytest.approx(16.0)  # x0 driven to the boundary 1
    assert res.scaled_l2 == pytest.approx(1.0)


def test_deterministic() -> None:
    scorer = LinearScorer(weights=np.array([1.0, 2.0, 0.5]), threshold=1.0)
    low, high, mask = _box(3)
    kw = dict(feature_low=low, feature_high=high, modifiable_mask=mask, max_changed=2)
    a = evade_detector(scorer, np.ones(3), **kw)
    b = evade_detector(scorer, np.ones(3), **kw)
    assert (a.changed_indices, a.adversarial_score, a.query_count) == (
        b.changed_indices, b.adversarial_score, b.query_count
    )


def test_query_budget_is_respected() -> None:
    scorer = LinearScorer(weights=np.array([1.0, 1.0, 1.0]), threshold=-1.0)  # impossible
    low, high, mask = _box(3)
    res = evade_detector(
        scorer, np.ones(3), feature_low=low, feature_high=high,
        modifiable_mask=mask, max_changed=3, candidate_values=10, budget=25,
    )
    assert res.evaded is False
    assert res.query_count <= 25 + 10  # stops within one coordinate sweep of the cap
    assert res.termination_reason in {"budget_exhausted", "no_improvement"}


def test_rejects_bad_shapes() -> None:
    scorer = LinearScorer(weights=np.array([1.0, 1.0]), threshold=1.0)
    with pytest.raises(ValueError):
        evade_detector(
            scorer, np.ones(2), feature_low=np.zeros(3), feature_high=np.ones(3),
            modifiable_mask=np.ones(2, dtype=bool),
        )


# --------------------------------------------------------------------------- #
# Joint (ensemble) evasion — the defence-in-depth attacker
# --------------------------------------------------------------------------- #


@dataclass
class CoordScorer:
    """score = x[coord] (higher = more anomalous); threshold on that one axis."""

    coord: int
    threshold: float

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)[:, self.coord]


def test_joint_evades_independent_detectors_with_enough_budget() -> None:
    # A watches x0, B watches x1; both start above threshold; two changes suffice.
    a = CoordScorer(coord=0, threshold=0.5)
    b = CoordScorer(coord=1, threshold=0.5)
    low, high, mask = _box(2)
    res = evade_detectors_jointly(
        [a, b], np.ones(2), scales=[1.0, 1.0], names=["A", "B"],
        feature_low=low, feature_high=high, modifiable_mask=mask,
        max_changed=2, candidate_values=11,
    )
    assert res.evaded_all is True
    assert res.n_detectors_evaded == 2
    assert res.n_changed == 2
    assert res.termination_reason == "threshold_reached"
    assert all(m < 0 for m in res.adversarial_margins)


def test_joint_sparsity_cap_blocks_full_ensemble_evasion() -> None:
    # Two orthogonal detectors but only one sensor may be spoofed -> at most one falls.
    a = CoordScorer(coord=0, threshold=0.5)
    b = CoordScorer(coord=1, threshold=0.5)
    low, high, mask = _box(2)
    res = evade_detectors_jointly(
        [a, b], np.ones(2), scales=[1.0, 1.0], names=["A", "B"],
        feature_low=low, feature_high=high, modifiable_mask=mask,
        max_changed=1, candidate_values=11,
    )
    assert res.evaded_all is False
    assert res.n_detectors_evaded == 1
    assert res.n_changed == 1


def test_joint_conflicting_detectors_cannot_be_evaded_together() -> None:
    # A wants x0 low (<0.5); B wants x0 high (>0.5). No single value satisfies both.
    a = CoordScorer(coord=0, threshold=0.5)

    @dataclass
    class InverseScorer:
        threshold: float

        def score_samples(self, values: np.ndarray) -> np.ndarray:
            return 1.0 - np.asarray(values, dtype=float)[:, 0]

    b = InverseScorer(threshold=0.5)  # score = 1 - x0; needs x0 > 0.5
    low, high, mask = _box(2)
    res = evade_detectors_jointly(
        [a, b], np.array([0.8, 0.0]), scales=[1.0, 1.0], names=["A", "B"],
        feature_low=low, feature_high=high, modifiable_mask=mask,
        max_changed=2, candidate_values=21,
    )
    assert res.evaded_all is False  # irreducible conflict
    # Whichever it lands on, at most one detector is below threshold.
    assert res.n_detectors_evaded <= 1


def test_joint_scales_normalise_margins() -> None:
    # B's raw scores are 1000x A's; without scaling B would dominate the objective.
    a = CoordScorer(coord=0, threshold=0.5)

    @dataclass
    class BigScorer:
        threshold: float

        def score_samples(self, values: np.ndarray) -> np.ndarray:
            return 1000.0 * np.asarray(values, dtype=float)[:, 1]

    b = BigScorer(threshold=500.0)  # crosses at x1 = 0.5, like A at x0 = 0.5
    low, high, mask = _box(2)
    res = evade_detectors_jointly(
        [a, b], np.ones(2), scales=[0.5, 500.0], names=["A", "B"],
        feature_low=low, feature_high=high, modifiable_mask=mask,
        max_changed=2, candidate_values=11,
    )
    # With comparable normalised margins both detectors get attacked and evaded.
    assert res.evaded_all is True
    assert res.n_changed == 2


def test_joint_deterministic() -> None:
    a = CoordScorer(coord=0, threshold=0.5)
    b = CoordScorer(coord=1, threshold=0.5)
    low, high, mask = _box(3)
    kw = dict(scales=[1.0, 1.0], feature_low=low, feature_high=high,
              modifiable_mask=mask, max_changed=2)
    r1 = evade_detectors_jointly([a, b], np.ones(3), **kw)
    r2 = evade_detectors_jointly([a, b], np.ones(3), **kw)
    assert (r1.changed_indices, r1.adversarial_margins, r1.query_count) == (
        r2.changed_indices, r2.adversarial_margins, r2.query_count
    )


def test_joint_rejects_bad_scales() -> None:
    a = CoordScorer(coord=0, threshold=0.5)
    low, high, mask = _box(2)
    with pytest.raises(ValueError):
        evade_detectors_jointly(
            [a], np.ones(2), scales=[0.0],  # non-positive scale
            feature_low=low, feature_high=high, modifiable_mask=mask,
        )
    with pytest.raises(ValueError):
        evade_detectors_jointly(
            [a], np.ones(2), scales=[1.0, 1.0],  # wrong length
            feature_low=low, feature_high=high, modifiable_mask=mask,
        )

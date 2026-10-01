"""Standard unsupervised anomaly-detection baselines and a fair scoring harness.

The point of this module is a *fair* comparison. Every baseline obeys the same
contract as HydroJEV's own reconstruction detector:

* it is fitted on the attack-free training partition **only** (its scaler,
  components, covariance, or forest see no labels and no evaluation data);
* its alarm threshold is calibrated as the ``threshold_quantile`` (default the
  99.5th percentile) of its *training* score distribution — identical to the
  reconstruction detector's policy; and
* it exposes ``score_samples(X) -> higher-is-more-anomalous`` and a ``threshold``.

Evaluation reports both threshold-independent discrimination (ROC-AUC, PR-AUC —
which do not depend on any calibration choice) and the operating-point metrics a
control room actually sees (scenario recall, point recall, benign false-positive
rate). This lets the paper claim discrimination quality without hiding behind a
single hand-picked threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ScoreFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FittedDetector:
    """A fitted detector exposing the shared scoring contract."""

    name: str
    family: str
    score_fn: ScoreFn
    threshold: float
    provenance: str

    def score_samples(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self.score_fn(values), dtype=float)


def calibrate_threshold(train_scores: np.ndarray, quantile: float) -> float:
    """Alarm threshold = the ``quantile`` of the *training* score distribution."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    return float(np.quantile(np.asarray(train_scores, dtype=float), quantile))


def contiguous_runs(flags: np.ndarray, target: int) -> list[tuple[int, int]]:
    """Return ``[start, end)`` index pairs of maximal runs equal to ``target``."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.asarray(flags).astype(int)):
        if int(value) == target and start is None:
            start = index
        elif int(value) != target and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def _validate_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


# --------------------------------------------------------------------------- #
# Baseline fitters (each fit on the attack-free training partition only)
# --------------------------------------------------------------------------- #


def fit_pca_residual_detector(
    train: np.ndarray, *, threshold_quantile: float = 0.995, variance_retained: float = 0.95
) -> FittedDetector:
    """PCA reconstruction-residual detector (a classic linear-subspace baseline).

    Standardize on train, keep the leading components that retain
    ``variance_retained`` of variance, and score a sample by the mean squared
    residual of its projection back into the full space.
    """

    values = _validate_2d(train, "train")
    if not 0.0 < variance_retained < 1.0:
        raise ValueError("variance_retained must be in (0, 1)")
    scaler = StandardScaler().fit(values)
    standardized = scaler.transform(values)
    # n_components as a variance fraction lets sklearn pick the rank; guard the
    # degenerate single-sample/constant case by clamping to >=1 component.
    max_components = min(standardized.shape)
    pca = PCA(n_components=variance_retained if max_components > 1 else 1, svd_solver="full")
    pca.fit(standardized)

    def score(x: np.ndarray) -> np.ndarray:
        z = scaler.transform(_validate_2d(x, "input"))
        reconstructed = pca.inverse_transform(pca.transform(z))
        return np.mean(np.square(z - reconstructed), axis=1)

    threshold = calibrate_threshold(score(values), threshold_quantile)
    return FittedDetector(
        name="PCA residual",
        family="pca_residual",
        score_fn=score,
        threshold=threshold,
        provenance=(
            f"PCA subspace residual; standardized; components retain "
            f"{variance_retained:.0%} variance ({pca.n_components_} of {values.shape[1]})"
        ),
    )


def fit_mahalanobis_detector(
    train: np.ndarray, *, threshold_quantile: float = 0.995
) -> FittedDetector:
    """Mahalanobis-distance detector with Ledoit-Wolf shrinkage covariance.

    Shrinkage keeps the covariance invertible on the correlated / near-constant
    SCADA channels where a raw empirical covariance would be singular.
    """

    values = _validate_2d(train, "train")
    estimator = LedoitWolf().fit(values)

    def score(x: np.ndarray) -> np.ndarray:
        return estimator.mahalanobis(_validate_2d(x, "input"))

    threshold = calibrate_threshold(score(values), threshold_quantile)
    return FittedDetector(
        name="Mahalanobis",
        family="mahalanobis",
        score_fn=score,
        threshold=threshold,
        provenance=f"squared Mahalanobis distance; Ledoit-Wolf shrinkage={estimator.shrinkage_:.3g}",
    )


def fit_isolation_forest_detector(
    train: np.ndarray,
    *,
    threshold_quantile: float = 0.995,
    seed: int = 42,
    n_estimators: int = 200,
) -> FittedDetector:
    """Isolation Forest detector (a standard modern nonparametric baseline)."""

    values = _validate_2d(train, "train")
    scaler = StandardScaler().fit(values)
    forest = IsolationForest(
        n_estimators=n_estimators, random_state=seed, contamination="auto"
    ).fit(scaler.transform(values))

    def score(x: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal, so negate for anomaly score.
        return -forest.decision_function(scaler.transform(_validate_2d(x, "input")))

    threshold = calibrate_threshold(score(values), threshold_quantile)
    return FittedDetector(
        name="Isolation Forest",
        family="isolation_forest",
        score_fn=score,
        threshold=threshold,
        provenance=f"IsolationForest(n_estimators={n_estimators}, seed={seed}); standardized",
    )


# --------------------------------------------------------------------------- #
# Fair scoring on a labelled benchmark
# --------------------------------------------------------------------------- #


def evaluate_scores(scores: np.ndarray, threshold: float, labels: np.ndarray) -> dict[str, float]:
    """Score a detector's per-sample scores against binary attack labels.

    Reports threshold-independent discrimination (ROC-AUC, PR-AUC) and the
    operating-point metrics at the calibrated ``threshold``: scenario recall (a
    maximal contiguous attack run counts as detected if any of its samples
    alarms), point recall, and the benign false-positive rate at both the point
    and the scenario granularity.
    """

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same shape")
    alerting = scores >= threshold

    attack_runs = contiguous_runs(labels, 1)
    benign_runs = contiguous_runs(labels, 0)
    detected = sum(1 for (a, b) in attack_runs if bool(alerting[a:b].any()))
    benign_tripped = sum(1 for (a, b) in benign_runs if bool(alerting[a:b].any()))

    point_tp = int(np.sum(alerting & (labels == 1)))
    point_fp = int(np.sum(alerting & (labels == 0)))
    n_attack = int(np.sum(labels == 1))
    n_benign = int(np.sum(labels == 0))

    # ROC/PR-AUC need both classes present and finite scores.
    both_classes = 0 < n_attack < len(labels)
    finite = np.isfinite(scores).all()
    roc = float(roc_auc_score(labels, scores)) if both_classes and finite else float("nan")
    pr = float(average_precision_score(labels, scores)) if both_classes and finite else float("nan")

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "scenario_recall": detected / len(attack_runs) if attack_runs else float("nan"),
        "point_recall": point_tp / n_attack if n_attack else float("nan"),
        "point_false_positive_rate": point_fp / n_benign if n_benign else float("nan"),
        "benign_scenario_fp_rate": benign_tripped / len(benign_runs) if benign_runs else float("nan"),
        "attack_scenarios": len(attack_runs),
        "benign_scenarios": len(benign_runs),
        "scenarios_detected": detected,
        "n_samples": int(len(labels)),
    }

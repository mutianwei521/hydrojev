"""Pure, post hoc descriptive metrics for the HydroJEV paper summary.

No inference service is invoked. Labels are used here only for evaluation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score

from hydrojev.benchmarks.hydrojev_metrics import (
    binary_grid_metrics, contiguous_runs, episode_metrics, moving_block_indices,
    wilson_interval,
)


def hold_grid(indices: np.ndarray, values: np.ndarray, n_raw: int, *, fill: Any = False) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    values = np.asarray(values)
    if len(indices) != len(values) or np.any(np.diff(indices) <= 0):
        raise ValueError("grid alignment is invalid")
    if len(indices) and (indices[0] < 0 or indices[-1] >= n_raw):
        raise ValueError("grid lies outside raw rows")
    held = np.full(n_raw, fill, dtype=values.dtype)
    for k, start in enumerate(indices):
        stop = indices[k + 1] if k + 1 < len(indices) else n_raw
        held[start:stop] = values[k]
    return held


def hourly_metrics(raw_labels: np.ndarray, held_alarm: np.ndarray, warmup: int) -> dict[str, Any]:
    """Return legacy-compatible names and explicit same-grid hourly denominators."""
    raw_labels = np.asarray(raw_labels, dtype=int)
    held_alarm = np.asarray(held_alarm, dtype=bool)
    mask = np.arange(len(raw_labels)) >= warmup
    m = episode_metrics(raw_labels, held_alarm, evaluation_mask=mask)
    score = m["official_batadal"]
    ben_runs = contiguous_runs(raw_labels[warmup:], 0)
    benign_hit = sum(bool(np.any(held_alarm[warmup + a:warmup + b])) for a, b in ben_runs)
    m.update({
        "scenario_recall": m["episode_recall"],
        "benign_scenarios": len(ben_runs),
        "benign_scenario_fp_rate": benign_hit / len(ben_runs) if ben_runs else None,
        "benign_hour_fpr": m["benign_hour_false_positive_rate"],
        "point_false_positive_rate": m["benign_hour_false_positive_rate"],
        "mean_ttd_hours": m["mean_detected_ttd_hours"],
        "balanced_accuracy": score["S_CLF"],
        "batadal_score": score,
        "point_recall": score["classification"].get("TPR"),
        "n_samples": int(np.sum(mask)),
        "scenarios_detected": m["detected_attack_scenarios"],
        "episode_recall_ci_note": (
            "Post hoc descriptive Wilson 95% interval over attack episodes. "
            "Episodes share one network time series; independence is not established."
        ),
    })
    return m


def probability_metrics(labels: np.ndarray, probabilities: list[Any], called: np.ndarray) -> dict[str, Any]:
    """Assess actual returned alarm probabilities; not-called rows are not zeros."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray([np.nan if value is None else float(value) for value in probabilities])
    called = np.asarray(called, dtype=bool)
    if not (len(y) == len(p) == len(called)):
        raise ValueError("probability metric alignment failed")
    available = np.isfinite(p)
    if np.any(available & ~called):
        raise ValueError("probability exists on an uncalled row")
    if np.any((p[available] < 0) | (p[available] > 1)):
        raise ValueError("invalid alarm probability")
    actual = binary_grid_metrics(y[available], p[available]) if available.any() else None
    complete = bool(available.all()) and bool(len(y))
    result = {
        "status": "available_called_subset" if available.any() else "not_available",
        "denominator": "valid live alarm probabilities on called grid points",
        "n_grid": len(y), "n_called": int(called.sum()), "n_available": int(available.sum()),
        "n_not_called": int((~called).sum()), "n_called_missing": int((called & ~available).sum()),
        "available_positive_labels": int(np.sum(y[available] == 1)),
        "available_negative_labels": int(np.sum(y[available] == 0)),
        "coverage_all_grid": float(available.mean()) if len(y) else None,
        "coverage_called": float(available.sum() / called.sum()) if called.any() else None,
        "roc_auc": actual["roc_auc"] if actual else None,
        "average_precision": actual["average_precision"] if actual else None,
        "brier_score": actual["brier_score"] if actual else None,
        "ece": actual["ece"] if actual else None,
        "ece_bins": actual["ece_bins"] if actual else [],
        "all_grid": {
            "complete_grid": complete,
            "roc_auc": actual["roc_auc"] if actual and complete else None,
            "average_precision": actual["average_precision"] if actual and complete else None,
            "brier_score": actual["brier_score"] if actual and complete else None,
        },
        "note": "Called-subset ranking is selected by the gate; it is not whole-stream detection ranking. "
                "Not-called and failed rows remain missing; no class or probability is imputed.",
    }
    return result


def rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if y.shape != s.shape or not np.isfinite(s).all():
        raise ValueError("ranking arrays must be aligned and finite")
    both = len(np.unique(y)) == 2
    return {"n": len(y), "roc_auc": float(roc_auc_score(y, s)) if both else None,
            "average_precision": float(average_precision_score(y, s)) if both else None}


def gate_metrics(raw_labels: np.ndarray, indices: np.ndarray, candidates: np.ndarray, alarms: np.ndarray,
                 warmup: int) -> dict[str, Any]:
    y = np.asarray(raw_labels, dtype=int)
    gy = y[indices]
    c, a = np.asarray(candidates, dtype=bool), np.asarray(alarms, dtype=bool)
    gh = hold_grid(indices, c, len(y))
    ah = hold_grid(indices, a, len(y))
    gm = hourly_metrics(y, gh, warmup)
    fm = hourly_metrics(y, ah, warmup)
    positive = gy == 1
    gate_misses = [i + 1 for i, row in enumerate(gm["per_attack"]) if not row["detected"]]
    post_gate_misses = [i + 1 for i, (g, f) in enumerate(zip(gm["per_attack"], fm["per_attack"]))
                       if g["detected"] and not f["detected"]]
    recall = float(c[positive].mean()) if positive.any() else None
    return {
        "candidate_grid_points": int(c.sum()), "total_grid_points": len(c),
        "candidate_rate": float(c.mean()) if len(c) else None,
        "attack_grid_points": int(positive.sum()),
        "attack_grid_points_passed": int((c & positive).sum()),
        "attack_grid_points_excluded": int((~c & positive).sum()),
        "attack_grid_recall": recall, "attack_grid_loss": 1.0 - recall if recall is not None else None,
        "episode_count": gm["attack_scenarios"], "episode_recall": gm["scenario_recall"],
        "missed_episodes": gate_misses, "post_gate_missed_episodes": post_gate_misses,
        "episode_recall_upper_bound_given_gate": gm["scenario_recall"],
        "gate_earliest_per_attack": gm["per_attack"],
        "gate_hourly_false_positive_rate": gm["benign_hour_fpr"],
        "note": "Gate recall at grid labels and event coverage after forward hold answer different questions. "
                "Gate time is the earliest this gated method could alarm, independent of the Jev answer.",
    }


def _cp_bounds(successes: int, n: int, *, tail: float = .0125) -> tuple[float, float]:
    # Four bounds jointly use a Bonferroni 5% error budget.
    lo = float(beta.ppf(tail, successes, n - successes + 1)) if successes else 0.0
    hi = float(beta.ppf(1 - tail, successes + 1, n - successes)) if successes < n else 1.0
    return lo, hi


def paired_event_interval(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired event outcomes must align")
    n = len(a)
    plus, minus = int(np.sum(a & ~b)), int(np.sum(~a & b))
    pl, pu = _cp_bounds(plus, n) if n else (None, None)
    ml, mu = _cp_bounds(minus, n) if n else (None, None)
    return {
        "difference": float(np.mean(a.astype(float) - b.astype(float))) if n else None,
        "n_event_pairs": n, "live_only_detected": plus, "control_only_detected": minus,
        "both_detected": int(np.sum(a & b)), "neither_detected": int(np.sum(~a & ~b)),
        "ci95": [pl - mu, pu - ml] if n else [None, None],
        "method": "Bonferroni conservative bounds for paired gain/loss probabilities using exact binomial bounds",
        "status": "post_hoc_descriptive",
        "note": "Uses paired episode discordance, not independent treatment groups. Binomial exchangeability "
                "is only an approximation for seven events from one network; identical outcomes do not imply equivalence.",
    }


def _confusion_values(y: np.ndarray, a: np.ndarray) -> tuple[int, int, int, int]:
    return (int(np.sum((y == 1) & a)), int(np.sum((y == 0) & a)),
            int(np.sum((y == 0) & ~a)), int(np.sum((y == 1) & ~a)))


def _classification_from_counts(v: np.ndarray) -> tuple[float, float]:
    tp, fp, tn, fn = np.asarray(v, dtype=float).sum(axis=0)
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return .5 * (tpr + 1 - fpr), fpr


def _ci(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"ci95": np.percentile(a, [2.5, 97.5]).tolist() if len(a) else [None, None],
            "n_valid": len(a), "degenerate_empirical_distribution": bool(len(a) and np.ptp(a) == 0),
            "status": "post_hoc_descriptive"}


def paired_comparison(raw_labels: np.ndarray, indices: np.ndarray,
                      live_alarm: np.ndarray, control_alarm: np.ndarray,
                      live_states: np.ndarray, control_states: np.ndarray,
                      live_probabilities: list[Any], warmup: int, *, n_boot: int = 1000,
                      seed: int = 42) -> dict[str, Any]:
    """Paired same-row differences with explicit temporal/event uncertainty units."""
    y = np.asarray(raw_labels, dtype=int)
    la, ca = np.asarray(live_alarm, dtype=bool), np.asarray(control_alarm, dtype=bool)
    ls, cs = np.asarray(live_states), np.asarray(control_states)
    lh, ch = hold_grid(indices, la, len(y)), hold_grid(indices, ca, len(y))
    lm, cm = hourly_metrics(y, lh, warmup), hourly_metrics(y, ch, warmup)
    mask = np.arange(len(y)) >= warmup
    changes = ls != cs
    alarm_changes = la != ca
    delta = {name: (lm[name] - cm[name] if lm[name] is not None and cm[name] is not None else None)
             for name in ("scenario_recall", "benign_hour_fpr", "balanced_accuracy", "mean_ttd_hours")}
    delta["batadal_score"] = lm["batadal_score"]["S"] - cm["batadal_score"]["S"]
    delta["S_TTD"] = lm["batadal_score"]["S_TTD"] - cm["batadal_score"]["S_TTD"]
    event_live = np.asarray([row["detected"] for row in lm["per_attack"]])
    event_control = np.asarray([row["detected"] for row in cm["per_attack"]])
    event_recall_ci = paired_event_interval(event_live, event_control)
    common_ttd = [l["ttd_hours"] - c["ttd_hours"] for l, c in zip(lm["per_attack"], cm["per_attack"])
                  if l["detected"] and c["detected"]]
    p = np.asarray([np.nan if q is None else q for q in live_probabilities], dtype=float)
    available = np.isfinite(p)
    gy = y[indices]
    hard_brier = float(np.mean((ca[available].astype(float) - gy[available]) ** 2)) if available.any() else None
    live_brier = float(np.mean((p[available] - gy[available]) ** 2)) if available.any() else None

    # Moving blocks resample contiguous GRID cells, each containing its original
    # forward-held hourly labels. No episode TTD is computed across block seams.
    lcounts, ccounts = [], []
    for k, start in enumerate(indices):
        stop = indices[k + 1] if k + 1 < len(indices) else len(y)
        lcounts.append(_confusion_values(y[start:stop], lh[start:stop]))
        ccounts.append(_confusion_values(y[start:stop], ch[start:stop]))
    lcounts, ccounts = np.asarray(lcounts), np.asarray(ccounts)
    ba_draws, fp_draws, brier_draws = [], [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        idx = moving_block_indices(len(indices), min(8, len(indices)), rng)
        lba, lfp = _classification_from_counts(lcounts[idx])
        cba, cfp = _classification_from_counts(ccounts[idx])
        ba_draws.append(lba - cba); fp_draws.append(lfp - cfp)
        ok = available[idx]
        if ok.any():
            jj = idx[ok]
            brier_draws.append(float(np.mean((p[jj] - gy[jj]) ** 2 - (ca[jj].astype(float) - gy[jj]) ** 2)))

    # Attack-centred non-overlapping clusters retain each real event intact.
    # These seven clusters include adjacent benign exposure; block resampling
    # does not invent attack onsets or concatenate fragments for TTD.
    runs = [(a + warmup, b + warmup) for a, b in contiguous_runs(y[warmup:], 1)]
    boundaries = [warmup] + [(runs[k - 1][1] + runs[k][0]) // 2 for k in range(1, len(runs))] + [len(y)]
    cluster_l, cluster_c = [], []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        cluster_l.append(_confusion_values(y[a:b], lh[a:b])); cluster_c.append(_confusion_values(y[a:b], ch[a:b]))
    cluster_l, cluster_c = np.asarray(cluster_l), np.asarray(cluster_c)
    lnorm = np.asarray([row["normalized_ttd"] for row in lm["per_attack"]])
    cnorm = np.asarray([row["normalized_ttd"] for row in cm["per_attack"]])
    s_draws, ttd_draws = [], []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        ids = rng.integers(0, len(runs), size=len(runs))
        lba, _ = _classification_from_counts(cluster_l[ids]); cba, _ = _classification_from_counts(cluster_c[ids])
        td = float(np.mean(cnorm[ids] - lnorm[ids]))
        ttd_draws.append(td); s_draws.append(.5 * td + .5 * (lba - cba))

    return {
        "direction": "live_minus_control", "delta": delta,
        "decision_changes": {
            "grid_count": int(changes.sum()), "grid_total": len(changes), "grid_rate": float(changes.mean()),
            "alarm_grid_count": int(alarm_changes.sum()), "alarm_grid_rate": float(alarm_changes.mean()),
            "hourly_count": int(np.sum((lh != ch) & mask)), "hourly_total": int(mask.sum()),
            "hourly_rate": float(np.mean((lh != ch)[mask])),
            "alarm_added": int(np.sum(la & ~ca)), "alarm_removed": int(np.sum(~la & ca)),
            "new_abstentions": int(np.sum((ls == "abstain") & (cs != "abstain"))),
            "grid_corrected": int(np.sum((la == gy) & (ca != gy))),
            "grid_harmed": int(np.sum((la != gy) & (ca == gy))),
        },
        "paired_ttd_common_detected": {"n_event_pairs": len(common_ttd),
            "mean_difference_hours": float(np.mean(common_ttd)) if common_ttd else None,
            "differences_hours": common_ttd},
        "probability_vs_hard_prediction": {
            "n_common": int(available.sum()), "denominator": "same valid live-called grid rows",
            "live_brier": live_brier, "control_hard_prediction_brier": hard_brier,
            "difference": live_brier - hard_brier if live_brier is not None else None,
            "difference_moving_block_ci": _ci(brier_draws),
            "note": "The deterministic control outputs a hard alarm, not a calibrated probability; this is a hard-prediction reference.",
        },
        "intervals": {
            "episode_recall": event_recall_ci,
            "balanced_accuracy": _ci(ba_draws), "benign_hour_fpr": _ci(fp_draws),
            "batadal_score": _ci(s_draws), "S_TTD": _ci(ttd_draws),
            "hourly_interval_method": "paired non-circular moving-block bootstrap; eight 6-hour grid cells (48 h)",
            "event_interval_method": "paired bootstrap of seven intact attack-centred clusters including adjacent benign exposure",
            "n_boot": n_boot, "seed": seed, "n_episode_clusters": len(runs),
            "status": "post_hoc_descriptive", "p_value": None,
            "note": "Intervals were chosen after inspecting results and are descriptive, not preregistered. "
                    "One network and seven temporally related events limit generalisation. Degenerate bootstrap "
                    "intervals from identical paired outcomes are not evidence of equivalence. No multiplicity-adjusted superiority claim is made.",
        },
    }

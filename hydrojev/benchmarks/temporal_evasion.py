"""Temporal (trajectory) evasion attack against the *named* residual families.

The per-sample evasion oracle (:mod:`hydrojev.benchmarks.evasion_attack`) perturbs
one instantaneous vector and therefore cannot touch the architecture's temporal
families (CPDZ on a one-step GRU prediction, GEpFM's EWMA drift, RAT's windowed
covariance) — they are defined over history. This module supplies the missing
counterpart: a *trajectory* evasion attack that spoofs a contiguous window of a
compromised sensor's readings and asks whether the three families can be silenced
at once. It is the faithful stress test of the "measured orthogonality" result in
:mod:`hydrojev.benchmarks.named_family_benchmark`: if the families are genuinely
orthogonal, tuning a spoof to hide one should leave the others firing.

**Threat model (realistic, and a conservative lower bound on attacker power).** The
adversary owns at most ``max_changed`` sensors for the duration of an attack window
and holds each compromised sensor at a single **constant, in-range value** (a
stuck/replay spoof) chosen to minimise the families' joint alarm over the window.
Holding a constant value across the window is strictly weaker than per-timestep
freedom, so the evasion rates reported here *upper-bound* the families' robustness;
the natural (stronger) follow-up — a fully adaptive per-timestep spoof on the same
sensor budget — is realised by :func:`temporal_conceal_adaptive` in this module and
benchmarked head-to-head in :mod:`hydrojev.benchmarks.adaptive_evasion_benchmark`.
Scoring couples through the GRU automatically: overriding a sensor at the attack
timesteps changes both its own residuals and every subsequent one-step prediction
that consumes it.

Leakage-safe throughout: the surrogate, robust scale, references, and thresholds
come from attack-free ``dataset03`` (via
:func:`~hydrojev.benchmarks.named_family_benchmark.healthy_family_reference`); the
attack runs on labelled ``dataset04`` windows the families actually flag. The
runner returns ``status="not_run"`` when BATADAL/torch are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from hydrojev.benchmarks.named_family_benchmark import (
    FamilyReference,
    PredictFn,
    cpdz_series,
    gepfm_series,
    one_step_residual_stream,
    rat_series,
)

FAMILIES = ("CPDZ residual", "GEpFM drift", "RAT covariance")


# --------------------------------------------------------------------------- #
# Pure window scoring + greedy trajectory concealment
# --------------------------------------------------------------------------- #


def temporal_family_window_scores(
    predict_fn: PredictFn,
    stream_standardized: np.ndarray,
    *,
    seq_len: int,
    reference: FamilyReference,
    window_start: int,
) -> dict[str, float]:
    """Worst-case (max) per-family score over absolute indices ``>= window_start``.

    ``stream_standardized`` is ``[context + attack]`` in standardized units. A
    detector is considered to fire on the window iff its window-max exceeds its
    threshold, so the max is the correct stealth criterion (no alarm at any step).
    Returns ``nan`` for a family with no finite score on the window.
    """

    residuals, start = one_step_residual_stream(predict_fn, stream_standardized, seq_len)
    cpdz = cpdz_series(residuals, reference.scale)
    gepfm = gepfm_series(
        cpdz,
        reference_mean=reference.gepfm_ref_mean,
        reference_scale=reference.gepfm_ref_scale,
        smoothing=reference.gepfm_smoothing,
    )
    rat = rat_series(residuals, reference.rat_ref_cov, window=reference.rat_window)
    if window_start < start:
        raise ValueError("window_start must be at or after the first residual index")
    tail = slice(window_start - start, None)

    def wmax(series: np.ndarray) -> float:
        vals = series[tail]
        vals = vals[np.isfinite(vals)]
        return float(np.max(vals)) if vals.size else float("nan")

    return {"CPDZ residual": wmax(cpdz), "GEpFM drift": wmax(gepfm), "RAT covariance": wmax(rat)}


def _window_exceedance(
    window_scores: dict[str, float],
    thresholds: dict[str, float],
    scales: dict[str, float],
    families: Sequence[str] = FAMILIES,
) -> float:
    """Total normalised threshold exceedance over ``families`` (0 iff all silent).

    ``families`` defaults to the full stack; a subset restricts the objective to a
    *deployed* set of detectors — the mechanism behind the §6.2 defense-in-depth
    ablation (:mod:`hydrojev.benchmarks.family_staircase_ablation`).
    """

    total = 0.0
    for fam in families:
        value = window_scores[fam]
        if not np.isfinite(value):
            continue
        total += max(0.0, (value - thresholds[fam]) / scales[fam])
    return total


def _families_alarming(
    window_scores: dict[str, float],
    thresholds: dict[str, float],
    families: Sequence[str] = FAMILIES,
) -> tuple[str, ...]:
    return tuple(
        fam for fam in families
        if np.isfinite(window_scores[fam]) and window_scores[fam] >= thresholds[fam]
    )


@dataclass(frozen=True)
class TemporalEvasionResult:
    original_scores: dict[str, float]
    adversarial_scores: dict[str, float]
    thresholds: dict[str, float]
    originally_alarming: tuple[str, ...]
    still_alarming: tuple[str, ...]  # families still firing after the joint attack
    evaded_all: bool  # every family that fired originally is now silent
    n_families_evaded: int
    changed_sensors: tuple[int, ...]
    n_changed: int
    query_count: int
    scaled_l2: float  # spoof magnitude over the window, per-feature-span units
    termination_reason: str
    adversarial_block: np.ndarray | None = None  # the spoofed attack window (standardized)


def temporal_conceal(
    predict_fn: PredictFn,
    context_standardized: np.ndarray,
    attack_standardized: np.ndarray,
    *,
    reference: FamilyReference,
    thresholds: dict[str, float],
    scales: dict[str, float],
    seq_len: int,
    box_low: np.ndarray,
    box_high: np.ndarray,
    modifiable_mask: np.ndarray,
    max_changed: int = 4,
    candidate_values: int = 11,
    budget: int = 3000,
    tolerance: float = 1e-12,
    target_families: Sequence[str] | None = None,
) -> TemporalEvasionResult:
    """Greedily hold <=``max_changed`` sensors constant to silence all families.

    Coordinate = (compromised sensor, held value). At each step the single
    (sensor, value) hold that most reduces the families' joint window exceedance is
    applied, subject to the sparsity cap, the per-sensor in-range box, and the query
    budget. Evasion succeeds iff every family that originally fired is driven below
    its threshold across the whole window.

    ``target_families`` restricts the objective *and* the evasion criterion to a
    deployed subset of detectors (default: the full stack). Silencing a larger
    deployed set is strictly harder, which is what the §6.2 defense-in-depth ablation
    (:mod:`hydrojev.benchmarks.family_staircase_ablation`) measures. ``adversarial_scores``
    is always reported over the full stack so a caller can inspect bystander families.
    """

    context = np.asarray(context_standardized, dtype=float)
    attack = np.asarray(attack_standardized, dtype=float)
    low = np.asarray(box_low, dtype=float)
    high = np.asarray(box_high, dtype=float)
    mask = np.asarray(modifiable_mask, dtype=bool)
    features = attack.shape[1]
    if not (context.ndim == 2 and attack.ndim == 2 and context.shape[1] == features):
        raise ValueError("context and attack must be 2D with matching feature count")
    if not (low.shape == high.shape == mask.shape == (features,)):
        raise ValueError("box bounds and mask must be 1D of length features")
    if not 1 <= max_changed <= int(mask.sum() or 1):
        raise ValueError("max_changed must be between 1 and the number of modifiable sensors")
    if candidate_values < 2 or budget < 1:
        raise ValueError("candidate_values and budget must be positive")
    objective_families = FAMILIES if target_families is None else tuple(target_families)
    if not set(objective_families) <= set(FAMILIES) or not objective_families:
        raise ValueError("target_families must be a non-empty subset of FAMILIES")

    window_start = context.shape[0]
    span = np.where((high - low) <= np.finfo(float).eps, 1.0, high - low)

    def score(block: np.ndarray) -> dict[str, float]:
        stream = np.vstack([context, block])
        return temporal_family_window_scores(
            predict_fn, stream, seq_len=seq_len, reference=reference, window_start=window_start
        )

    current = attack.copy()
    original_scores = score(current)
    originally_alarming = _families_alarming(original_scores, thresholds, objective_families)
    current_ex = _window_exceedance(original_scores, thresholds, scales, objective_families)
    queries = 0
    reason = "no_improvement"

    while True:
        if current_ex <= tolerance:
            reason = "threshold_reached"
            break
        if queries >= budget:
            reason = "budget_exhausted"
            break

        changed_now = np.array(
            [bool(np.any(~np.isclose(current[:, s], attack[:, s]))) for s in range(features)]
        )
        eligible = mask.copy()
        if int(changed_now.sum()) >= max_changed:
            eligible &= changed_now
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size == 0:
            reason = "no_modifiable_sensor"
            break

        best_gain = tolerance
        best_sensor: int | None = None
        best_value = 0.0
        best_ex = current_ex
        for sensor in eligible_idx:
            if queries >= budget:
                break
            for value in np.linspace(low[sensor], high[sensor], candidate_values):
                trial = current.copy()
                trial[:, sensor] = value
                ex = _window_exceedance(score(trial), thresholds, scales, objective_families)
                queries += 1
                gain = current_ex - ex
                if gain > best_gain:
                    best_gain = gain
                    best_sensor = int(sensor)
                    best_value = float(value)
                    best_ex = ex
        if best_sensor is None:
            reason = "no_improvement"
            break
        current[:, best_sensor] = best_value
        current_ex = best_ex

    adversarial_scores = score(current)
    still = _families_alarming(adversarial_scores, thresholds, objective_families)
    changed_sensors = tuple(
        int(s) for s in range(features) if bool(np.any(~np.isclose(current[:, s], attack[:, s])))
    )
    scaled_l2 = float(np.sqrt(np.sum(((current - attack) / span) ** 2)))
    # Evasion is judged only against families that actually fired on the true attack.
    still_of_interest = tuple(f for f in still if f in originally_alarming)
    return TemporalEvasionResult(
        original_scores=original_scores,
        adversarial_scores=adversarial_scores,
        thresholds=dict(thresholds),
        originally_alarming=originally_alarming,
        still_alarming=still_of_interest,
        evaded_all=len(still_of_interest) == 0 and len(originally_alarming) > 0,
        n_families_evaded=len(originally_alarming) - len(still_of_interest),
        changed_sensors=changed_sensors,
        n_changed=len(changed_sensors),
        query_count=queries,
        scaled_l2=scaled_l2,
        termination_reason=reason,
        adversarial_block=current.copy(),
    )


# --------------------------------------------------------------------------- #
# Adaptive (per-timestep) concealer — the strictly stronger attacker
# --------------------------------------------------------------------------- #


def temporal_conceal_adaptive(
    predict_fn: PredictFn,
    context_standardized: np.ndarray,
    attack_standardized: np.ndarray,
    *,
    reference: FamilyReference,
    thresholds: dict[str, float],
    scales: dict[str, float],
    seq_len: int,
    box_low: np.ndarray,
    box_high: np.ndarray,
    modifiable_mask: np.ndarray,
    max_changed: int = 4,
    candidate_values: int = 11,
    budget: int = 8000,
    tolerance: float = 1e-12,
    warm_start: np.ndarray | None = None,
) -> TemporalEvasionResult:
    """Per-timestep coordinate descent — the strict generalisation of :func:`temporal_conceal`.

    The attacker still owns at most ``max_changed`` sensors, but may now write a
    **different in-range value at every timestep** (the constant hold is the special
    case where all timesteps are equal), so this attacker is provably at least as
    strong. It is **warm-started from the constant-hold optimum**: when ``warm_start``
    is ``None`` we first run :func:`temporal_conceal` and refine its block, so the
    result can only match or beat §6.2. (A caller that already has the constant-hold
    block may pass it as ``warm_start`` to skip the re-solve; ``query_count`` then
    counts only the refinement queries.)

    Warm-starting is also what makes the greedy well-posed: the alarm objective is the
    *window-max* per family, on which a single-cell move is flat while several
    timesteps are tied at the maximum. The constant hold collapses those ties in one
    stroke; per-timestep refinement then bites exactly where the GRU makes the per-step
    residuals distinct — i.e. on real trajectories, not on degenerate flat spikes.

    Coordinate = (compromised sensor, timestep, held value). Each iteration applies the
    single (sensor, t, value) move that most reduces the families' joint window
    exceedance, subject to the same per-sensor box, sparsity cap, and query budget.
    """

    context = np.asarray(context_standardized, dtype=float)
    attack = np.asarray(attack_standardized, dtype=float)
    low = np.asarray(box_low, dtype=float)
    high = np.asarray(box_high, dtype=float)
    mask = np.asarray(modifiable_mask, dtype=bool)
    features = attack.shape[1]
    attack_len = attack.shape[0]
    if not (context.ndim == 2 and attack.ndim == 2 and context.shape[1] == features):
        raise ValueError("context and attack must be 2D with matching feature count")
    if not (low.shape == high.shape == mask.shape == (features,)):
        raise ValueError("box bounds and mask must be 1D of length features")
    if not 1 <= max_changed <= int(mask.sum() or 1):
        raise ValueError("max_changed must be between 1 and the number of modifiable sensors")
    if candidate_values < 2 or budget < 1:
        raise ValueError("candidate_values and budget must be positive")

    window_start = context.shape[0]

    def score(block: np.ndarray) -> dict[str, float]:
        stream = np.vstack([context, block])
        return temporal_family_window_scores(
            predict_fn, stream, seq_len=seq_len, reference=reference, window_start=window_start
        )

    seed_queries = 0
    if warm_start is None:
        seed = temporal_conceal(
            predict_fn, context, attack, reference=reference, thresholds=thresholds,
            scales=scales, seq_len=seq_len, box_low=low, box_high=high,
            modifiable_mask=mask, max_changed=max_changed,
            candidate_values=candidate_values, budget=budget, tolerance=tolerance,
        )
        current = np.asarray(seed.adversarial_block, dtype=float).copy()
        seed_queries = seed.query_count
    else:
        current = np.asarray(warm_start, dtype=float).copy()
        if current.shape != attack.shape:
            raise ValueError("warm_start must match the attack block shape")

    original_scores = score(attack)  # originally-alarming is judged on the TRUE attack
    originally_alarming = _families_alarming(original_scores, thresholds)
    original_set = set(originally_alarming)

    def alarming_of_interest(window_scores: dict[str, float]) -> set[str]:
        return set(_families_alarming(window_scores, thresholds)) & original_set

    cur_ws = score(current)
    current_ex = _window_exceedance(cur_ws, thresholds, scales)
    current_alarm = alarming_of_interest(cur_ws)  # silenced families must stay silenced
    queries = seed_queries  # honest total attacker effort (constant hold + refinement)
    budget_cap = seed_queries + budget  # refinement gets its own full budget on top of the seed
    reason = "no_improvement"

    while True:
        if current_ex <= tolerance:
            reason = "threshold_reached"
            break
        if queries >= budget_cap:
            reason = "budget_exhausted"
            break

        changed_now = np.array(
            [bool(np.any(~np.isclose(current[:, s], attack[:, s]))) for s in range(features)]
        )
        eligible = mask.copy()
        if int(changed_now.sum()) >= max_changed:
            eligible &= changed_now
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size == 0:
            reason = "no_modifiable_sensor"
            break

        best_gain = tolerance
        best_move: tuple[int, int, float] | None = None
        best_ex = current_ex
        best_alarm = current_alarm
        stop = False
        for sensor in eligible_idx:
            if stop:
                break
            grid = np.linspace(low[sensor], high[sensor], candidate_values)
            for t in range(attack_len):
                if queries >= budget_cap:
                    stop = True
                    break
                original_val = current[t, sensor]
                for value in grid:
                    trial = current
                    trial[t, sensor] = value
                    ws = score(trial)
                    queries += 1
                    ex = _window_exceedance(ws, thresholds, scales)
                    gain = current_ex - ex
                    # Never re-fire a silenced originally-alarming family: the total
                    # exceedance objective could otherwise trade a re-fire for a bigger
                    # drop elsewhere, breaking the "at least as strong as constant" claim.
                    if gain > best_gain:
                        trial_alarm = alarming_of_interest(ws)
                        if trial_alarm <= current_alarm:
                            best_gain = gain
                            best_move = (int(sensor), int(t), float(value))
                            best_ex = ex
                            best_alarm = trial_alarm
                    if queries >= budget_cap:
                        break
                current[t, sensor] = original_val  # revert the probe
        if best_move is None:
            reason = "no_improvement"
            break
        s, t, v = best_move
        current[t, s] = v
        current_ex = best_ex
        current_alarm = best_alarm

    adversarial_scores = score(current)
    still = _families_alarming(adversarial_scores, thresholds)
    changed_sensors = tuple(
        int(s) for s in range(features) if bool(np.any(~np.isclose(current[:, s], attack[:, s])))
    )
    span = np.where((high - low) <= np.finfo(float).eps, 1.0, high - low)
    scaled_l2 = float(np.sqrt(np.sum(((current - attack) / span) ** 2)))
    still_of_interest = tuple(f for f in still if f in originally_alarming)
    return TemporalEvasionResult(
        original_scores=original_scores,
        adversarial_scores=adversarial_scores,
        thresholds=dict(thresholds),
        originally_alarming=originally_alarming,
        still_alarming=still_of_interest,
        evaded_all=len(still_of_interest) == 0 and len(originally_alarming) > 0,
        n_families_evaded=len(originally_alarming) - len(still_of_interest),
        changed_sensors=changed_sensors,
        n_changed=len(changed_sensors),
        query_count=queries,
        scaled_l2=scaled_l2,
        termination_reason=reason,
        adversarial_block=current.copy(),
    )


# --------------------------------------------------------------------------- #
# Shared fixture: fit families on dataset03, select flagged dataset04 windows
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TemporalTarget:
    run_start: int
    attack_len: int
    context: np.ndarray
    attack: np.ndarray
    originally_alarming: tuple[str, ...]


@dataclass(frozen=True)
class TemporalFixture:
    predict_fn: PredictFn
    reference: FamilyReference
    thresholds: dict[str, float]
    scales: dict[str, float]
    seq_len: int
    context_len: int
    box_low: np.ndarray
    box_high: np.ndarray
    modifiable_mask: np.ndarray
    max_attack_len: int
    seed: int
    targets: tuple[TemporalTarget, ...]
    # One entry per examined dataset04 attack scenario recording why it was kept or
    # dropped (the honest n=5 ceiling: 7 scenarios, gates, family scores, selected flag).
    selection_report: tuple[dict[str, Any], ...] = ()


def build_temporal_fixture(
    config: Any,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 60,
    max_attack_len: int = 24,
    max_targets: int = 20,
) -> "TemporalFixture | dict[str, Any]":
    """Fit the named families on dataset03 and collect the dataset04 windows they flag.

    Single source of truth shared by the constant-hold runner and the adaptive
    head-to-head, so both attackers are evaluated on *identical* targets. Returns a
    ``not_run`` dict (never fabricated numbers) when BATADAL/torch are unavailable.
    """

    from hydrojev.datasets.wdn_loader import DatasetUnavailableError, load_batadal

    try:
        clean = load_batadal("dataset03", config=config)
        labelled = load_batadal("dataset04", config=config)
    except DatasetUnavailableError as exc:
        return {"status": "not_run", "reason": str(exc)}
    if labelled.label_column is None:
        return {"status": "not_run", "reason": f"{labelled.name} has no attack-label column"}

    columns = list(clean.feature_columns)
    if any(c not in labelled.frame.columns for c in columns):
        return {"status": "not_run", "reason": "feature columns absent in dataset04"}

    try:
        import torch  # noqa: F401
    except Exception as exc:  # torch is required for the surrogate
        return {"status": "not_run", "reason": f"torch unavailable: {exc}"}

    from hydrojev.benchmarks.detection_baselines import calibrate_threshold, contiguous_runs
    from hydrojev.benchmarks.named_family_benchmark import (
        _torch_predict_fn,
        healthy_family_reference,
    )
    from hydrojev.datasets.wdn_loader import classify_batadal_columns
    from hydrojev.physics.surrogate_models import (
        Standardizer,
        build_sequence_dataset,
        select_torch_device,
        train_surrogate,
    )

    seed = int(config.experiments.seed if seed is None else seed)
    seq_len = int(config.surrogate.sequence_length)
    n_features = len(columns)
    rat_window = max(60, n_features + 1)
    context_len = seq_len + rat_window

    clean_train = clean.train_frame[columns].to_numpy(dtype=float)
    rows = mode.batadal_train_rows
    if rows is not None:
        clean_train = clean_train[:rows]
    if clean_train.shape[0] <= context_len + 8:
        return {"status": "not_run", "reason": "attack-free stream too short for the surrogate window"}

    standardizer = Standardizer.fit(clean_train)
    clean_std = standardizer.transform(clean_train)
    eval_std = standardizer.transform(labelled.frame[columns].to_numpy(dtype=float))
    labels = labelled.frame[labelled.label_column].to_numpy().astype(int)

    inputs, targets = clean_std[:-1], clean_std[1:]
    windows, aligned = build_sequence_dataset(inputs, targets, seq_len)
    split = int(len(windows) * 0.85)
    if not 1 <= split < len(windows):
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

    reference = healthy_family_reference(
        predict_fn, clean_std, seq_len=seq_len, rat_window=rat_window
    )
    q = float(config.concealment.threshold_quantile)
    thresholds = {
        "CPDZ residual": calibrate_threshold(reference.clean_cpdz, q),
        "GEpFM drift": calibrate_threshold(reference.clean_gepfm, q),
        "RAT covariance": calibrate_threshold(reference.clean_rat[np.isfinite(reference.clean_rat)], q),
    }
    scales = {
        "CPDZ residual": float(np.std(reference.clean_cpdz)) or 1.0,
        "GEpFM drift": float(np.std(reference.clean_gepfm)) or 1.0,
        "RAT covariance": float(np.std(reference.clean_rat[np.isfinite(reference.clean_rat)])) or 1.0,
    }

    # Stealth box + modifiable mask (standardized units), same policy as the
    # instantaneous benchmark: per-sensor training min/max; pump/valve status frozen.
    box_low = clean_std.min(axis=0)
    box_high = clean_std.max(axis=0)
    groups = classify_batadal_columns(columns)
    frozen = set(groups.get("pump_status", ())) | set(groups.get("valve_status", ()))
    modifiable_mask = np.array([col not in frozen for col in columns], dtype=bool)

    # Walk every labelled attack scenario, recording *why* each is kept or dropped. This
    # is the single source of truth for the honest n=5 ceiling reported in §6.2: BATADAL
    # `dataset04` has a fixed set of scenarios, and only those the named-family stack
    # actually detects within a <=max_attack_len causal window are evasion targets (one
    # cannot evade a detector that never fires). No knob enlarges this without a harder
    # benchmark; sub-windowing a scenario would fabricate statistically dependent samples.
    selected: list[TemporalTarget] = []
    selection_report: list[dict[str, Any]] = []
    for run_start, run_end in contiguous_runs(labels, 1):
        if len(selected) >= max_targets:
            break  # selection cap reached (does not bind on dataset04: 7 < 20)
        raw_len = int(run_end - run_start)
        attack_len = int(min(raw_len, max_attack_len))
        entry: dict[str, Any] = {
            "run_start": int(run_start),
            "run_end": int(run_end),
            "raw_len": raw_len,
            "attack_len": attack_len,
            "history_ok": bool(run_start >= context_len),
            "length_ok": bool(attack_len >= 2),
            "family_scores": None,
            "alarming": [],
            "selected": False,
            "drop_reason": None,
        }
        if run_start < context_len:
            entry["drop_reason"] = "insufficient_history"  # no full RAT window before attack
            selection_report.append(entry)
            continue
        if attack_len < 2:
            entry["drop_reason"] = "attack_too_short"
            selection_report.append(entry)
            continue
        context = eval_std[run_start - context_len : run_start]
        attack = eval_std[run_start : run_start + attack_len]
        original = temporal_family_window_scores(
            predict_fn, np.vstack([context, attack]), seq_len=seq_len,
            reference=reference, window_start=context_len,
        )
        alarming = _families_alarming(original, thresholds)
        entry["family_scores"] = {f: float(original[f]) for f in FAMILIES}
        entry["alarming"] = list(alarming)
        if not alarming:
            entry["drop_reason"] = "families_silent"  # already missed -> not an evasion target
            selection_report.append(entry)
            continue
        entry["selected"] = True
        selection_report.append(entry)
        selected.append(
            TemporalTarget(
                run_start=int(run_start), attack_len=attack_len,
                context=context, attack=attack, originally_alarming=alarming,
            )
        )

    if not selected:
        return {"status": "not_run", "reason": "no dataset04 attack window with sufficient history is flagged by the families"}

    return TemporalFixture(
        predict_fn=predict_fn, reference=reference, thresholds=thresholds, scales=scales,
        seq_len=seq_len, context_len=context_len, box_low=box_low, box_high=box_high,
        modifiable_mask=modifiable_mask, max_attack_len=max_attack_len, seed=seed,
        targets=tuple(selected), selection_report=tuple(selection_report),
    )


# --------------------------------------------------------------------------- #
# Real-data runner (constant-hold; §6.2)
# --------------------------------------------------------------------------- #


def run_temporal_evasion_benchmark(
    config: Any,
    *,
    mode: Any,
    seed: int | None = None,
    max_epochs: int = 60,
    max_attack_len: int = 24,
    max_targets: int = 20,
    max_changed: int = 4,
    candidate_values: int = 11,
    budget: int = 3000,
) -> dict[str, Any]:
    """Fit families on dataset03, then temporally evade them on dataset04 windows."""

    from hydrojev.benchmarks.evasion_benchmark import wilson_interval

    fixture = build_temporal_fixture(
        config, mode=mode, seed=seed, max_epochs=max_epochs,
        max_attack_len=max_attack_len, max_targets=max_targets,
    )
    if isinstance(fixture, dict):  # not_run
        return fixture

    seed = fixture.seed
    seq_len = fixture.seq_len
    context_len = fixture.context_len

    targets_out: list[dict[str, Any]] = []
    still_counts = {fam: 0 for fam in FAMILIES}
    fired_counts = {fam: 0 for fam in FAMILIES}
    evaded = 0
    for target in fixture.targets:
        res = temporal_conceal(
            fixture.predict_fn, target.context, target.attack, reference=fixture.reference,
            thresholds=fixture.thresholds, scales=fixture.scales, seq_len=seq_len,
            box_low=fixture.box_low, box_high=fixture.box_high,
            modifiable_mask=fixture.modifiable_mask, max_changed=max_changed,
            candidate_values=candidate_values, budget=budget,
        )
        if res.evaded_all:
            evaded += 1
        for fam in res.originally_alarming:
            fired_counts[fam] += 1
            if fam in res.still_alarming:
                still_counts[fam] += 1
        targets_out.append({
            "run_start": int(target.run_start),
            "attack_len": target.attack_len,
            "originally_alarming": list(res.originally_alarming),
            "still_alarming": list(res.still_alarming),
            "evaded_all": res.evaded_all,
            "n_changed_sensors": res.n_changed,
            "scaled_l2": res.scaled_l2,
            "queries": res.query_count,
            "termination": res.termination_reason,
        })

    n_targets = len(fixture.targets)
    lo, hi = wilson_interval(evaded, n_targets)
    # Orthogonality-under-evasion: when the joint attack silences the union, which
    # families most often remain firing? (High residual-alarm = robust family.)
    residual_alarm = {
        fam: (still_counts[fam] / fired_counts[fam] if fired_counts[fam] else float("nan"))
        for fam in FAMILIES
    }
    return {
        "status": "ok",
        "train_dataset": "dataset03",
        "evaluate_dataset": "dataset04",
        "protocol": (
            "trajectory (temporal) evasion: an attacker holding <=%d in-range sensors constant "
            "over an attack window of <=%d steps greedily minimises the three named families' "
            "joint window exceedance; families/surrogate/box/thresholds all fit on dataset03; "
            "constant-hold spoof is a conservative lower bound on attacker power (per-timestep "
            "freedom is strictly stronger)" % (max_changed, max_attack_len)
        ),
        "seed": seed,
        "sequence_length": seq_len,
        "context_len": context_len,
        "max_attack_len": max_attack_len,
        "max_changed": max_changed,
        "budget": budget,
        "n_targets": n_targets,
        "families": list(FAMILIES),
        "joint_evasion_rate": evaded / n_targets,
        "joint_evasion_wilson95": [lo, hi],
        "n_jointly_evaded": evaded,
        "per_family_fired": fired_counts,
        "per_family_still_firing_after_joint_attack": still_counts,
        "per_family_residual_alarm_rate": residual_alarm,
        "targets": targets_out,
    }


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #


def format_temporal_evasion_table(result: dict[str, Any]) -> str:
    if result.get("status") != "ok":
        return "not_run"
    lo, hi = result["joint_evasion_wilson95"]
    lines = [
        f"Temporal (trajectory) evasion of the named families on {result['evaluate_dataset']} "
        f"({result['n_targets']} flagged attack windows; fit on {result['train_dataset']}).",
        f"Attacker: hold <= {result['max_changed']} in-range sensors constant over <= "
        f"{result['max_attack_len']} steps (conservative lower bound on attacker power).",
        "",
        f"Joint evasion (all firing families silenced): {result['joint_evasion_rate']:.3f} "
        f"[{lo:.3f}, {hi:.3f}]  ({result['n_jointly_evaded']}/{result['n_targets']})",
        "",
        "Per-family robustness under the JOINT attack "
        "(residual-alarm = still fires after the attacker targets the whole stack; higher = more robust):",
        "",
        f"{'family':<18}{'fired on':>10}{'still fires':>13}{'residual-alarm':>16}",
        "-" * 57,
    ]
    for fam in result["families"]:
        fired = result["per_family_fired"][fam]
        still = result["per_family_still_firing_after_joint_attack"][fam]
        rate = result["per_family_residual_alarm_rate"][fam]
        rate_s = "n/a" if not np.isfinite(rate) else f"{rate:.3f}"
        lines.append(f"{fam:<18}{fired:>10}{still:>13}{rate_s:>16}")
    lines += [
        "",
        "Reading: a fully orthogonal stack should be hard to silence jointly (low joint",
        "evasion) even though each family alone may be evadable — silencing one leaves the",
        "decorrelated others firing. This is the evasion-side test of the §6.1 correlation matrix.",
    ]
    return "\n".join(lines)


def plot_temporal_evasion(result: dict[str, Any], path: str) -> str | None:
    if result.get("status") != "ok":
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = result["families"]
    fired = [result["per_family_fired"][f] for f in fams]
    still = [result["per_family_still_firing_after_joint_attack"][f] for f in fams]
    x = np.arange(len(fams))

    fig, (axr, axb) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [0.8, 1]})

    rate = result["joint_evasion_rate"]
    lo, hi = result["joint_evasion_wilson95"]
    axr.bar([0], [rate], width=0.5, color="#b2182b",
            yerr=[[rate - lo], [hi - rate]], capsize=8)
    axr.set_xlim(-0.6, 0.6)
    axr.set_ylim(0.0, 1.0)
    axr.set_xticks([0])
    axr.set_xticklabels(["all 3 families\nsilenced jointly"], fontsize=9)
    axr.set_ylabel("joint evasion rate")
    axr.set_title(f"Joint temporal evasion\n{rate:.2f} [{lo:.2f}, {hi:.2f}] "
                  f"({result['n_jointly_evaded']}/{result['n_targets']})", fontsize=9)
    axr.grid(axis="y", alpha=0.3)

    axb.bar(x - 0.2, fired, width=0.38, label="fired on true attack", color="#4393c3")
    axb.bar(x + 0.2, still, width=0.38, label="still fires after joint attack", color="#d6604d")
    axb.set_xticks(x)
    axb.set_xticklabels(fams, fontsize=9)
    axb.set_ylabel("# attack windows")
    axb.set_title("Per-family robustness under the joint attack\n"
                  "(residual firing = orthogonal survival)", fontsize=9)
    axb.legend(fontsize=8)
    axb.grid(axis="y", alpha=0.3)

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

    parser = argparse.ArgumentParser(description="Temporal (trajectory) evasion of the named residual families")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--max-targets", type=int, default=20)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--outdir", default="artifacts/temporal_evasion")
    args = parser.parse_args(argv)

    config = load_config()
    mode = RunMode.resolve(dry=False, quick=args.quick)
    epochs = 12 if args.quick else args.max_epochs
    targets = 8 if args.quick else args.max_targets
    result = run_temporal_evasion_benchmark(
        config, mode=mode, seed=args.seed, max_epochs=epochs, max_targets=targets, budget=args.budget
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "temporal_evasion.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    if result.get("status") == "ok":
        table = format_temporal_evasion_table(result)
        (outdir / "temporal_evasion_table.txt").write_text(table + "\n", encoding="utf-8")
        figure = plot_temporal_evasion(result, str(outdir / "temporal_evasion.png"))
        print(result["protocol"])
        print()
        print(table)
        print(f"\nwrote: {outdir/'temporal_evasion.json'}, temporal_evasion_table.txt, {figure}")
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline root-cause analysis of the Jev pilot v1 (no network, no paid calls).

Why did Jev lose to the R1 rule and the R2 logistic regression in P1 (benign false-alarm veto)
and in P2 debug round 1 (cause discrimination)? Everything here is recomputed from frozen
artifacts: the request states (the exact evidence Jev saw), the recorded Jev responses and the
prepared rows. The P2 evaluation splits are not touched: every P2 analysis uses the train split
only, so the design of protocol v2 cannot be tuned on evaluation data.

Comparator features are rebuilt from the request states. P2 states carry every quantity of
``jev_pilot_p2.features`` (to two decimals), P1 states all but the exact pump-status statistic,
so each rebuilt R2 is checked against the probabilities recorded at prepare time.

Usage:  python -m hydrojev.benchmarks.jev_pilot_rootcause --out artifacts/jev_pilot_rootcause/analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.benchmarks import jev_pilot_p2 as p2

P1_DIR = p1.DEFAULT_OUTDIR
P2_DIR = p2.DEFAULT_OUTDIR
N_DRAWS = 200
K_PER_CLASS = (1, 2, 4, 8, 16, 32)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _auc(y: Sequence[int], s: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(np.asarray(y), np.asarray(s, dtype=float)))


def _balanced(y: np.ndarray, pred: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    return float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def _q(v: Sequence[float]) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    return {"median": round(float(np.median(v)), 3), "q25": round(float(np.quantile(v, 0.25)), 3), "q75": round(float(np.quantile(v, 0.75)), 3)}


# --------------------------------------------------------------------------- #
# P1: benign false-alarm veto on BATADAL
# --------------------------------------------------------------------------- #


def p1_state_features(state: Mapping[str, Any]) -> np.ndarray:
    """``jev_pilot_p1.grid_features`` rebuilt from the request state (pump-status ratio approximated)."""

    x: list[float] = []
    for name in p1.DETECTOR_ORDER:
        d = state["detectors"].get(name, {})
        if d.get("state") in (None, "unavailable"):
            x += [0.0, 0.0, 0.0]
            continue
        x += [min(float(d["peak_score_over_threshold"]), 10.0), float(d["state"] == "alerting"), float(d["hours_above_threshold_in_window"]) / p1.STRIDE]
    pc = state["physical_consistency"]
    x.append(min(2.0 * len(pc["pump_status_vs_flow"]["violating_pumps"]), 10.0))
    t = pc["tank_level_change_vs_flows"]
    x.append(min(max(t["peak_abs_z_by_tank"].values()) / t["violation_threshold_abs_z"], 10.0))
    j = pc["junction_pressure_vs_operation"]
    x.append(min(max(j["largest_abs_z"].values()) / j["violation_threshold_abs_z"], 10.0))
    x.append(float(pc["tank_level_envelope"]["status"] == "violated"))
    peak = max(float(r["peak_abs_residual_sigma_in_window"]) for r in state["largest_surrogate_residuals"])
    x += [min(peak, 50.0), float(state["operating_context"]["peak_count_atypical_pumps_in_window"]), float(state["detector_pattern"]["alerting_count"])]
    return np.asarray(x, dtype=float)


def _p1_segments(block: Mapping[str, Any]) -> dict[int, int]:
    """Blocked-CV groups: a new segment starts at every hourly label change (each attack episode is one segment)."""

    labels = np.asarray(block["labels_hourly_internal"], dtype=int)
    seg = np.concatenate([[0], np.cumsum(labels[1:] != labels[:-1])])
    start = int(block["start"])
    return {int(r["raw_index"]): int(seg[int(r["raw_index"]) - start]) for r in block["grid"] if r["candidate"]}


def _p1_jev(stage: str) -> dict[int, float]:
    out = {}
    for f in (P1_DIR / "responses" / stage).glob("*.response.json"):
        out[int(f.stem.split("_")[-1].split(".")[0])] = float(_json(f)["response"]["answers"]["alarm"]["noul"])
    return out


def p1_analysis() -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold

    prepared = _json(P1_DIR / "prepared.json")
    block = prepared["variants"]["dataset04"]
    cand = [r for r in block["grid"] if r["candidate"]]
    ids = [int(r["raw_index"]) for r in cand]
    y = np.asarray([int(r["label_internal"]) for r in cand])
    x = np.vstack([p1_state_features(_json(Path(r["request_path"]))["state"]) for r in cand])
    seg = _p1_segments(block)
    groups = np.asarray([seg[i] for i in ids])
    recorded = np.asarray([float(r["r2_probability"]) for r in cand])
    rebuilt_in = p1.fit_r2(x, y).predict_proba(x)[:, 1]

    # out-of-fold R2: blocked by label segment, so no window is scored by a model that saw its episode
    oof = np.zeros(len(y))
    n_splits = min(10, len(set(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(x, y, groups):
        oof[te] = p1.fit_r2(x[tr], y[tr]).predict_proba(x[te])[:, 1]

    debug = _json(P1_DIR / "debug_summary_live_r2.json")["chosen"]
    pos = {i: k for k, i in enumerate(ids)}
    d = np.asarray([pos[i] for i in debug])
    yd = y[d]
    j1, j2 = _p1_jev("p1_debug_r1"), _p1_jev("p1_debug_r2")
    a1 = np.asarray([j1[i] for i in debug])
    a2 = np.asarray([j2[i] for i in debug])
    r1 = np.asarray([bool(cand[k]["r1"]) for k in d])
    detector_auc_debug = {}
    detector_auc_dev = {}
    for k, name in enumerate(p1.DETECTOR_ORDER):
        detector_auc_debug[name] = round(_auc(yd, x[d, 3 * k]), 3)
        detector_auc_dev[name] = round(_auc(y, x[:, 3 * k]), 3)

    # LOO stacking of Jev and out-of-fold R2 on the 30 debug windows
    from sklearn.linear_model import LogisticRegression

    def logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-3, 1 - 1e-3)
        return np.log(p / (1 - p))

    z = np.column_stack([logit(a2), logit(oof[d])])
    stack = np.zeros(len(d))
    for i in range(len(d)):
        m = np.ones(len(d), bool)
        m[i] = False
        stack[i] = LogisticRegression(C=1.0).fit(z[m], yd[m]).predict_proba(z[i : i + 1])[0, 1]

    # label-scarce R2: k labelled attack + k benign windows drawn from other segments than the scored window
    rng = np.random.default_rng(p1.DEFAULT_SAMPLE_SEED)
    fold_of = np.zeros(len(y), int)
    for f, (_, te) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups)):
        fold_of[te] = f
    curve = {}
    for k in K_PER_CLASS:
        aucs, bals = [], []
        for _ in range(N_DRAWS):
            score = np.zeros(len(d))
            for f in set(fold_of[d]):
                pool = np.flatnonzero(fold_of != f)
                att, ben = pool[y[pool] == 1], pool[y[pool] == 0]
                if len(att) < k or len(ben) < k:
                    continue
                tr = np.concatenate([rng.choice(att, k, replace=False), rng.choice(ben, k, replace=False)])
                sel = np.flatnonzero(fold_of[d] == f)
                score[sel] = p1.fit_r2(x[tr], y[tr]).predict_proba(x[d[sel]])[:, 1]
            aucs.append(_auc(yd, score))
            bals.append(_balanced(yd, score >= 0.5))
        curve[str(k)] = {"auc": _q(aucs), "balanced_at_0.5": _q(bals)}

    jev_err = (a2 >= 0.5) != yd.astype(bool)
    single_jp = np.asarray([cand[k]["violated_checks"] == ["junction_pressure_vs_operation"] for k in d])
    paired = {f"{a}_minus_{b}": _paired_auc_ci(yd, sa, sb, rng) for a, b, sa, sb in (
        ("jev_round1", "r2_oof", a1, oof[d]), ("jev_round2", "r2_oof", a2, oof[d]), ("jev_round1", "mahalanobis", a1, x[d, 0]),
        ("jev_round2", "mahalanobis", a2, x[d, 0]))}
    return {
        "n_dev_candidates": int(len(y)), "n_dev_attack": int(y.sum()), "n_segments": int(len(set(groups))),
        "rebuild_check": {"spearman_rebuilt_vs_recorded_r2_in_sample": round(_spearman(rebuilt_in, recorded), 3)},
        "debug_30": {
            "auc": {
                "jev_round1_alarm": round(_auc(yd, a1), 3), "jev_round2_alarm": round(_auc(yd, a2), 3),
                "r2_recorded_in_sample": round(_auc(yd, recorded[d]), 3), "r2_out_of_fold_blocked": round(_auc(yd, oof[d]), 3),
                "stack_jev_r2oof_loo": round(_auc(yd, stack), 3), "r1_binary": round(_auc(yd, r1.astype(float)), 3),
                "alert_count": round(_auc(yd, x[d, -1]), 3),
            },
            "balanced_accuracy_at_0.5": {
                "jev_round1": round(_balanced(yd, a1 >= 0.5), 3), "jev_round2": round(_balanced(yd, a2 >= 0.5), 3),
                "r1": round(_balanced(yd, r1), 3), "r2_recorded_in_sample": round(_balanced(yd, recorded[d] >= 0.5), 3),
                "r2_out_of_fold_blocked": round(_balanced(yd, oof[d] >= 0.5), 3), "stack_loo": round(_balanced(yd, stack >= 0.5), 3),
            },
            "prior_art_detector_peak_ratio_auc": detector_auc_debug,
            "paired_bootstrap_auc_difference_95ci": paired,
            "spearman_jev_round2_vs_r2_recorded": round(_spearman(a2, recorded[d]), 3),
            "spearman_jev_round2_vs_r2_oof": round(_spearman(a2, oof[d]), 3),
            "jev_round2_errors": int(jev_err.sum()),
            "jev_round2_errors_on_single_junction_pressure_windows": int((jev_err & single_jp).sum()),
            "single_junction_pressure_windows": int(single_jp.sum()),
            "attack_share_of_single_junction_pressure_windows_dev_outside_debug": _share_single_jp(cand, set(debug)),
        },
        "dev_156_prior_art_detector_peak_ratio_auc": detector_auc_dev,
        "dev_156_r2_out_of_fold_auc": round(_auc(y, oof), 3),
        "label_scarce_r2_on_debug_30": curve,
        "label_scarce_note": "k attack + k benign dev windows per draw, drawn only from other label segments than the scored window; 200 draws",
    }


def _paired_auc_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> dict[str, float]:
    diffs = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        diffs.append(_auc(y[i], a[i]) - _auc(y[i], b[i]))
    return {"point": round(_auc(y, a) - _auc(y, b), 3), "lo": round(float(np.quantile(diffs, 0.025)), 3), "hi": round(float(np.quantile(diffs, 0.975)), 3)}


def _share_single_jp(cand: Sequence[Mapping[str, Any]], debug: set[int]) -> dict[str, Any]:
    rows = [r for r in cand if r["violated_checks"] == ["junction_pressure_vs_operation"] and int(r["raw_index"]) not in debug]
    return {"n": len(rows), "attack_share": round(float(np.mean([r["label_internal"] for r in rows])), 3) if rows else None}


# --------------------------------------------------------------------------- #
# P2: cause discrimination (train split only)
# --------------------------------------------------------------------------- #


def p2_state_features(state: Mapping[str, Any]) -> np.ndarray:
    """``jev_pilot_p2.features`` rebuilt from the request state (two-decimal rounding)."""

    cc = state["consistency_checks"]
    x: list[float] = []
    for c in p2.CHECKS:
        x.append(math.log1p(min(float(cc[c]["statistic"]) / max(float(cc[c]["threshold"]), 1.0), 50.0)))
        x.append(float(cc[c]["status"] == "violated"))
    counts = [float(state["anomalous_channel_counts"][k]) for k in ("L", "P", "F", "Fin", "Fsrc")]
    sp = cc["pressure_vs_twin"]["spread"]
    x += [sum(counts)] + counts
    x += [float(sp["fraction_anomalous"]), float(np.clip(sp["mean_z_all_sensors_last3h"], -20, 20)) / 5.0,
          float(np.clip(cc["system_demand_vs_twin"]["detail"]["mean_z_last3h"], -20, 20)) / 5.0]
    return np.asarray(x, dtype=float)


def _macro_f1(y: Sequence[str], pred: Sequence[str]) -> float:
    return float(p2.cause_metrics(list(y), list(pred))["macro_f1"])


def _consistent_check_issues(state: Mapping[str, Any]) -> int:
    """Consistent checks whose state entry still lists 'issue' details (a presentation cue, not evidence)."""

    n = 0
    for c, e in state["consistency_checks"].items():
        if e["status"] == "consistent" and c.startswith("control_logic") and e.get("detail"):
            n += len(e["detail"])
    return n


def _p2_jev(stage: str) -> dict[str, dict[str, float]]:
    out = {}
    for f in (P1_DIR / "responses" / stage).glob("*.response.json"):
        body = _json(f)["response"]
        rid = f.name.split(".")[0].replace(f"{stage}_", "")
        out[rid] = p2.parse_response(body).probabilities
    return out


def p2_analysis() -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    prepared = _json(P2_DIR / "prepared.json")
    train = [r for r in prepared["rows"] if r["split"] == "train"]
    states = {r["id"]: _json(P2_DIR / "requests" / "train" / f"{r['id']}.request.json")["state"] for r in train}
    x = np.vstack([p2_state_features(states[r["id"]]) for r in train])
    y = np.asarray([r["label_internal"] for r in train])
    sub = np.asarray([r["scenario_internal"]["subtype"] for r in train])
    recorded = np.asarray([r["r2"] for r in train])
    rebuilt = p2.fit_r2(x, y).predict(x)

    # out-of-fold R2 (5-fold stratified, 20 repeats)
    rng = np.random.default_rng(p1.DEFAULT_SAMPLE_SEED)
    oof_f1, oof_pred_first = [], None
    for rep in range(20):
        pred = np.empty(len(y), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(x, y):
            pred[te] = p2.fit_r2(x[tr], y[tr]).predict(x[te])
        oof_f1.append(_macro_f1(y, pred))
        if oof_pred_first is None:
            oof_pred_first = pred

    debug_ids = p2.select_debug(prepared)
    idx = {r["id"]: k for k, r in enumerate(train)}
    d = np.asarray([idx[i] for i in debug_ids])
    jev = _p2_jev("p2_debug_r1")
    probs = np.asarray([[jev[i][c] for c in p2.CLASSES] for i in debug_ids])
    raw = [p2.CLASSES[k] for k in probs.argmax(1)]
    batch = probs / probs.mean(0, keepdims=True)
    cal = [p2.CLASSES[k] for k in batch.argmax(1)]
    loo = []
    for i in range(len(d)):  # label-free prior from the other 11 windows only
        m = np.ones(len(d), bool)
        m[i] = False
        loo.append(p2.CLASSES[int((probs[i] / probs[m].mean(0)).argmax())])
    top2 = float(np.mean([y[d][k] in [p2.CLASSES[j] for j in np.argsort(-probs[k])[:2]] for k in range(len(d))]))

    # R2 on the same 12 debug windows when those windows are held out of training
    held = np.ones(len(y), bool)
    held[d] = False
    r2_heldout_debug = p2.fit_r2(x[held], y[held]).predict(x[d])

    # label-scarce R2: k labelled windows per class from train (debug windows excluded), scored on the 12 debug windows
    # and on all other held-out train windows
    curve = {}
    pool = np.flatnonzero(held)
    for k in K_PER_CLASS[:-1] + (40,):
        f_dbg, f_rest = [], []
        for _ in range(N_DRAWS):
            tr = np.concatenate([rng.choice(pool[y[pool] == c], k, replace=False) for c in p2.CLASSES])
            m = p2.fit_r2(x[tr], y[tr])
            rest = np.setdiff1d(pool, tr)
            f_dbg.append(_macro_f1(y[d], m.predict(x[d])))
            f_rest.append(_macro_f1(y[rest], m.predict(x[rest])))
        curve[str(k)] = {"debug_12": _q(f_dbg), "other_train_windows": _q(f_rest)}

    # leave-one-subtype-out: how R2 does on an event subtype it never saw
    loso = {}
    for s in sorted(set(sub)):
        m = sub != s
        pr = p2.fit_r2(x[m], y[m]).predict(x[~m])
        loso[s] = {"class": str(y[~m][0]), "n": int((~m).sum()), "r2_recall_unseen": round(float(np.mean(pr == y[~m])), 3),
                   "r2_recall_seen_oof": round(float(np.mean(oof_pred_first[~m] == y[~m])), 3),
                   "r1_recall": round(float(np.mean([train[k]["r1"] == y[k] for k in np.flatnonzero(~m)])), 3)}

    issues = {c: round(float(np.mean([_consistent_check_issues(states[r["id"]]) > 0 for r in train if r["label_internal"] == c])), 3) for c in p2.CLASSES}
    demand_sign = {}
    for c in p2.CLASSES:
        z = [float(states[r["id"]]["consistency_checks"]["system_demand_vs_twin"]["detail"]["mean_z_last3h"]) for r in train if r["label_internal"] == c]
        demand_sign[c] = {"share_negative": round(float(np.mean(np.asarray(z) < 0)), 3), "median_z": round(float(np.median(z)), 2)}
    profiles = {c: {chk: round(float(np.mean([chk in r["violated"] for r in train if r["label_internal"] == c])), 2) for chk in p2.CHECKS} for c in p2.CLASSES}
    per_window = [{"id": i, "truth": str(y[k]), "subtype": str(sub[k]), "jev": dict(zip(p2.CLASSES, map(float, probs[n]))), "jev_raw": raw[n],
                   "jev_batch_calibrated": cal[n], "jev_loo_calibrated": loo[n], "r1": train[k]["r1"], "r2_heldout": str(r2_heldout_debug[n]),
                   "consistent_check_issue_lines": _consistent_check_issues(states[i])} for n, (i, k) in enumerate(zip(debug_ids, d))]
    yd = list(y[d])
    return {
        "rebuild_check": {"r2_rebuilt_equals_recorded_share": round(float(np.mean(rebuilt == recorded)), 3)},
        "train_r2_in_sample_macro_f1": round(_macro_f1(y, recorded), 3),
        "train_r2_out_of_fold_macro_f1": _q(oof_f1),
        "train_r1_macro_f1": round(_macro_f1(y, [r["r1"] for r in train]), 3),
        "debug_12": {
            "macro_f1": {"jev_raw": round(_macro_f1(yd, raw), 3), "jev_batch_calibrated": round(_macro_f1(yd, cal), 3),
                         "jev_loo_calibrated": round(_macro_f1(yd, loo), 3), "r1": round(_macro_f1(yd, [train[k]["r1"] for k in d]), 3),
                         "r2_in_sample": round(_macro_f1(yd, recorded[d]), 3), "r2_debug_held_out": round(_macro_f1(yd, r2_heldout_debug), 3)},
            "correct_of_12": {"jev_raw": int(sum(a == b for a, b in zip(raw, yd))), "jev_batch_calibrated": int(sum(a == b for a, b in zip(cal, yd))),
                              "jev_loo_calibrated": int(sum(a == b for a, b in zip(loo, yd))), "r1": int(sum(train[k]["r1"] == y[k] for k in d)),
                              "r2_debug_held_out": int(sum(a == b for a, b in zip(r2_heldout_debug, yd)))},
            "jev_top2_accuracy": round(top2, 3),
            "jev_mean_probability": dict(zip(p2.CLASSES, np.round(probs.mean(0), 3).tolist())),
            "windows": per_window,
        },
        "label_scarce_r2_macro_f1": curve,
        "leave_one_subtype_out": loso,
        "presentation_cue_share_with_consistent_control_logic_issue_lines": issues,
        "system_demand_residual_by_class": demand_sign,
        "violated_check_profile_by_class": profiles,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; write the analysis to a new file")
    result = {"note": "offline, no paid calls; P2 uses the train split only", "p1": p1_analysis(), "p2": p2_analysis()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()}, indent=1, ensure_ascii=False)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

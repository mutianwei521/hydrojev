"""One-shot sealed evaluation of the Jev pilot P2 v2 (reads design_freeze.json; writes eval_analysis.json once).

Written before any Jev response on the sealed sets existed. Implements exactly the analysis plan frozen in
artifacts/jev_pilot_p2v2/design_freeze.json.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v

REVISION = 3
N_BOOT = 10_000
BOOT_SEED = 20260924
K_GRID = (1, 2, 4, 8, 16, 40)
N_DRAWS = 200
TARGET = v.OUTDIR / "eval_analysis.json"


def macro_f1(y: np.ndarray, p: np.ndarray, k: int = len(v.CLASSES)) -> float:
    """Macro-F1 over the four causes (abstention counted wrong), identical to p2.cause_metrics."""

    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        den = np.sum(p == c) + np.sum(y == c)
        f.append(0.0 if den == 0 else 2.0 * tp / den)
    return float(np.mean(f))


def _codes(labels: Sequence[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(v.CLASSES)}
    return np.asarray([idx.get(c, len(v.CLASSES)) for c in labels])


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    strata = [np.flatnonzero(y == c) for c in np.unique(y)]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in strata])
        d[i] = macro_f1(y[s], a[s]) - macro_f1(y[s], b[s])
    return {"delta": round(macro_f1(y, a) - macro_f1(y, b), 4), "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)],
            "p_one_sided": float(np.mean(d <= 0.0))}


def holm(pvals: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    order = sorted(pvals, key=pvals.get)
    out, still = {}, True
    for r, k in enumerate(order):
        thr = alpha / (len(order) - r)
        still = still and pvals[k] <= thr
        out[k] = {"p": pvals[k], "holm_threshold": thr, "reject_null": bool(still)}
    return out


def _r2_proba(train_rows: Sequence[Mapping[str, Any]], x: np.ndarray) -> np.ndarray:
    m = v.fit_r2(train_rows)
    pr = m.predict_proba(x)
    out = np.full((len(x), len(v.CLASSES)), 1e-3)
    for j, c in enumerate(m.classes_):
        out[:, v.CLASSES.index(c)] = pr[:, j]
    return out


def _set(split: str, prior: np.ndarray, train: Sequence[Mapping[str, Any]], loso: bool) -> dict[str, Any]:
    rows = v.load_split(split)["rows"]
    probs = v.jev_probs(split, REVISION)
    missing = [r["id"] for r in rows if r["id"] not in probs]
    rows = [r for r in rows if r["id"] in probs]
    ids = [r["id"] for r in rows]
    y = [r["label_internal"] for r in rows]
    sub = np.asarray([r["scenario_internal"]["subtype"] for r in rows])
    pj = v.jev_matrix(probs, ids)
    x = np.asarray([r["x"] for r in rows], dtype=float)
    pr_full = _r2_proba(train, x)
    loo = []
    for i in range(len(ids)):
        m = np.ones(len(ids), bool)
        m[i] = False
        loo += v._decide(pj[i : i + 1], np.maximum(pj[m].mean(0), 0.01))
    preds = {"jev_calibrated": v._decide(pj, prior), "jev_raw": v._decide(pj), "jev_loo_transductive": loo,
             "r1": [r["r1"] for r in rows], "r2_full": [v.CLASSES[k] for k in pr_full.argmax(1)],
             "fusion_calibrated": v.fuse(pj, pr_full, prior)}
    if loso:
        tsub = np.asarray([r["scenario_internal"]["subtype"] for r in train])
        r2l = np.empty(len(ids), dtype=object)
        for s in sorted(set(sub)):
            keep = [r for r, ts in zip(train, tsub) if ts != s]
            r2l[sub == s] = [v.CLASSES[k] for k in _r2_proba(keep, x[sub == s]).argmax(1)]
        preds["r2_loso"] = list(r2l)
    metrics = {k: {kk: p2.cause_metrics(y, p)[kk] for kk in ("macro_f1", "accuracy", "per_class_recall", "confusion_rows_true_cols_pred")} for k, p in preds.items()}
    for k, p in preds.items():
        assert abs(macro_f1(_codes(y), _codes(p)) - metrics[k]["macro_f1"]) < 1e-9
    per_sub = {s: {k: round(float(np.mean(np.asarray(p)[sub == s] == np.asarray(y)[sub == s])), 3) for k, p in preds.items()} | {"n": int(np.sum(sub == s))}
               for s in sorted(set(sub))}
    return {"split": split, "n": len(ids), "missing_responses": missing, "metrics": metrics, "per_subtype_recall": per_sub,
            "jev_mean_probability": dict(zip(v.CLASSES, np.round(pj.mean(0), 3).tolist())),
            "_y": y, "_preds": preds, "_x": x, "_pj": pj,
            "windows": [{"id": i, "truth": t, "subtype": s, "jev": dict(zip(v.CLASSES, pj[k].round(3).tolist())), **{n: preds[n][k] for n in preds}}
                        for k, (i, t, s) in enumerate(zip(ids, y, sub))]}


def label_scarce(train: Sequence[Mapping[str, Any]], res: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    y = _codes(res["_y"])
    jev = macro_f1(y, _codes(res["_preds"]["jev_calibrated"]))
    by = {c: [r for r in train if r["label_internal"] == c] for c in v.CLASSES}
    curve = {}
    for k in K_GRID:
        f = []
        for _ in range(N_DRAWS if k < 50 else 1):
            pick = [by[c][i] for c in v.CLASSES for i in rng.choice(len(by[c]), min(k, len(by[c])), replace=False)]
            f.append(macro_f1(y, _codes([v.CLASSES[j] for j in _r2_proba(pick, res["_x"]).argmax(1)])))
        curve[str(k)] = {"mean": round(float(np.mean(f)), 4), "q05": round(float(np.quantile(f, 0.05)), 4), "q95": round(float(np.quantile(f, 0.95)), 4),
                         "share_draws_r2_below_jev": round(float(np.mean(np.asarray(f) < jev)), 3)}
    cross = next((int(k) for k, c in curve.items() if c["mean"] >= jev), None)
    return {"jev_calibrated_macro_f1": round(jev, 4), "curve": curve, "crossover_k_per_class": cross}


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the sealed sets are evaluated once")
    freeze = json.loads((v.OUTDIR / "design_freeze.json").read_text(encoding="utf-8"))
    assert freeze["questions_sha256"] == v.questions_sha256(REVISION)
    prior = np.asarray(freeze["inductive_prior"]["vector"])
    train = v.load_split("train_s2")["rows"]
    rng = np.random.default_rng(BOOT_SEED)
    e1 = _set("eval_id_s2", prior, train, loso=True)
    e2 = _set("eval_novel_s2", prior, train, loso=False)
    tests = {}
    for name, res, comp in (("E2: Jev vs R2_full", e2, "r2_full"), ("E2: Jev vs R1", e2, "r1"),
                            ("E1-LOSO: Jev vs R2_loso", e1, "r2_loso"), ("E1-LOSO: Jev vs R1", e1, "r1")):
        y = _codes(res["_y"])
        tests[name] = paired_bootstrap(y, _codes(res["_preds"]["jev_calibrated"]), _codes(res["_preds"][comp]), rng)
    h = holm({k: t["p_one_sided"] for k, t in tests.items()})
    for k in tests:
        tests[k].update(h[k])
    secondary = {
        "H3_E1_in_distribution_r2_minus_jev": paired_bootstrap(_codes(e1["_y"]), _codes(e1["_preds"]["r2_full"]), _codes(e1["_preds"]["jev_calibrated"]), rng),
        "H4_E2_jev_minus_r1": round(e2["metrics"]["jev_calibrated"]["macro_f1"] - e2["metrics"]["r1"]["macro_f1"], 4),
        "fusion_minus_r2_full": {s: paired_bootstrap(_codes(r["_y"]), _codes(r["_preds"]["fusion_calibrated"]), _codes(r["_preds"]["r2_full"]), rng) for s, r in (("E1", e1), ("E2", e2))},
        "label_scarce": {"E1": label_scarce(train, e1, rng), "E2": label_scarce(train, e2, rng)},
    }
    secondary["H4_E2_jev_minus_r1_pass"] = secondary["H4_E2_jev_minus_r1"] >= 0.10
    out = {"created_utc": v._now(), "design_freeze_sha256": hashlib.sha256((v.OUTDIR / "design_freeze.json").read_bytes()).hexdigest(),
           "analysis_code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "revision": REVISION,
           "primary_H1": tests, "secondary": secondary,
           "E1": {k: val for k, val in e1.items() if not k.startswith("_")}, "E2": {k: val for k, val in e2.items() if not k.startswith("_")}}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"primary_H1": tests, "E1": {k: e1["metrics"][k]["macro_f1"] for k in e1["metrics"]},
                      "E2": {k: e2["metrics"][k]["macro_f1"] for k in e2["metrics"]}, "H4": secondary["H4_E2_jev_minus_r1"],
                      "fusion": {s: t["delta"] for s, t in secondary["fusion_minus_r2_full"].items()},
                      "crossover": {s: t["crossover_k_per_class"] for s, t in secondary["label_scarce"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

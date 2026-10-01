"""One-shot sealed evaluation of round 3 of the Jev pilot P2 v2 (state format 3: twin-free physics cross-checks).

Written before any round-3 sealed window or Jev response existed. Implements the plan frozen in
artifacts/jev_pilot_p2v2/design_freeze_r3.json. Round 3 is the third sealed round; format 3 was designed on the burned
round-2 sets, so it is judged on fresh sets only: E1r3 (v1 subtypes, in-distribution for R2) and E3 (the novel3
family, written before generation and never used for design; unseen by R2, by the question and by format 3).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks.jev_pilot_p2v2_eval import _codes, _r2_proba, _set, holm, label_scarce, paired_bootstrap

BOOT_SEED = 20260926
FREEZE = v.OUTDIR / "design_freeze_r3.json"
TARGET = v.OUTDIR / "eval_analysis_r3.json"
SETS = {"E1r3": ("eval_id_r3", True), "E3": ("eval_novel3_r3", False)}


def _metrics(y: Sequence[str], p: Sequence[str]) -> dict[str, Any]:
    m = p2.cause_metrics(list(y), list(p))
    return {k: m[k] for k in ("macro_f1", "accuracy", "per_class_recall", "confusion_rows_true_cols_pred")}


def evaluate(name: str, freeze: Mapping[str, Any]) -> dict[str, Any]:
    split, loso = SETS[name]
    prior3 = np.asarray(freeze["inductive_prior_s3"]["vector"])
    res = _set(f"{split}_s3", prior3, v.load_split("train_s3")["rows"], loso=loso)
    ids = [w["id"] for w in res["windows"]]
    rows = {r["id"]: r for r in v.load_split(f"{split}_s3")["rows"]}
    states = [json.loads((v.OUTDIR / "requests" / f"{split}_s3" / f"{i}.state.json").read_text(encoding="utf-8")) for i in ids]
    preds = res["_preds"]
    preds["r1b"] = [v.r1b_classify(rows[i]["r1"], s) for i, s in zip(ids, states)]
    preds["override_jev"] = v.override_jev(res["_pj"], prior3, states)
    # format 2 (the round-2 state) on the same scenarios: R2 retrained on its own features, Jev where answered
    rows2 = {r["id"]: r for r in v.load_split(f"{split}_s2")["rows"]}
    assert all(rows2[i]["scenario_internal"] == rows[i]["scenario_internal"] for i in ids)
    x2 = np.asarray([rows2[i]["x"] for i in ids], dtype=float)
    preds["r2_full_s2"] = [v.CLASSES[k] for k in _r2_proba(v.load_split("train_s2")["rows"], x2).argmax(1)]
    probs2 = v.jev_probs(f"{split}_s2", 3)
    both = [k for k, i in enumerate(ids) if i in probs2]
    y = res["_y"]
    out_metrics = {k: _metrics(y, p) for k, p in preds.items()}
    s2 = None
    if both:
        pj2 = v.jev_matrix(probs2, [ids[k] for k in both])
        j2 = v._decide(pj2, np.asarray(freeze["inductive_prior_s2"]["vector"]))
        yb = [y[k] for k in both]
        s2 = {"n": len(both), "jev_s2_calibrated": j2, "jev_s3_calibrated": [preds["jev_calibrated"][k] for k in both], "_y": yb,
              "metrics": {"jev_s2_calibrated": _metrics(yb, j2), "jev_s3_calibrated": _metrics(yb, [preds["jev_calibrated"][k] for k in both])}}
    sub = np.asarray([w["subtype"] for w in res["windows"]])
    per_sub = {s: {k: round(float(np.mean(np.asarray(p)[sub == s] == np.asarray(y)[sub == s])), 3) for k, p in preds.items()} | {"n": int(np.sum(sub == s))}
               for s in sorted(set(sub))}
    hits = {s: {c: round(float(np.mean([st["consistency_checks"][c]["status"] == "violated" for st, ss in zip(states, sub) if ss == s])), 3)
                for c in v.PHYSICS_CHECKS} | {"physics_override": round(float(np.mean([v.physics_override(st) for st, ss in zip(states, sub) if ss == s])), 3)}
            for s in sorted(set(sub))}
    for w, k in zip(res["windows"], range(len(ids))):
        w.update({n: preds[n][k] for n in ("r1b", "override_jev", "r2_full_s2")})
    res.update(metrics=out_metrics, per_subtype_recall=per_sub, physics_check_rate_by_subtype=hits, _s2=s2)
    return res


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the round-3 sealed sets are evaluated once")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["questions_sha256"] == v.questions_sha256(3) and freeze["state_format"] == v.STATE_FORMAT_PHYSICS
    assert freeze["eval_code_sha256"] == hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "eval code changed after the freeze"
    rng = np.random.default_rng(BOOT_SEED)
    e1, e3 = evaluate("E1r3", freeze), evaluate("E3", freeze)

    def test(res, a, b):
        return paired_bootstrap(_codes(res["_y"]), _codes(res["_preds"][a]), _codes(res["_preds"][b]), rng)

    primary = {"E3: Jev vs R1": test(e3, "jev_calibrated", "r1"), "E3: Jev vs R1b": test(e3, "jev_calibrated", "r1b"),
               "E1r3: Jev vs R1": test(e1, "jev_calibrated", "r1"), "E1r3: Jev vs R1b": test(e1, "jev_calibrated", "r1b"),
               "E1r3-LOSO: Jev vs R2_loso": test(e1, "jev_calibrated", "r2_loso")}
    h = holm({k: t["p_one_sided"] for k, t in primary.items()})
    for k in primary:
        primary[k].update(h[k])

    def s2test(res):
        s2 = res["_s2"]
        if not s2:
            return None
        return paired_bootstrap(_codes(s2["_y"]), _codes(s2["jev_s3_calibrated"]), _codes(s2["jev_s2_calibrated"]), rng) | {"n": s2["n"]}

    sets = (("E1r3", e1), ("E3", e3))
    secondary = {
        "format3_gain_jev_s3_minus_jev_s2": {s: s2test(r) for s, r in sets},
        "override_jev_vs_r1b": {s: test(r, "override_jev", "r1b") for s, r in sets},
        "override_jev_vs_jev": {s: test(r, "override_jev", "jev_calibrated") for s, r in sets},
        "jev_vs_r2_full_s3_reported_not_claimed": {s: test(r, "jev_calibrated", "r2_full") for s, r in sets},
        "override_jev_vs_r2_full_s3_reported_not_claimed": {s: test(r, "override_jev", "r2_full") for s, r in sets},
        "fusion_minus_r2_full": {s: test(r, "fusion_calibrated", "r2_full") for s, r in sets},
        "training_free_variants_vs_r1b": {s: {k: test(r, k, "r1b") for k in ("jev_raw", "jev_loo_transductive")} for s, r in sets},
        "label_scarce_jev": {s: label_scarce(v.load_split("train_s3")["rows"], r, rng) for s, r in sets},
    }
    out = {"created_utc": v._now(), "round": 3, "design_freeze_r3_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
           "analysis_code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "primary_H1": primary, "secondary": secondary}
    for s, r in sets:
        out[s] = {k: val for k, val in r.items() if not k.startswith("_")}
        out[s]["format2_comparison"] = None if not r["_s2"] else {k: val for k, val in r["_s2"].items() if not k.startswith("_")}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    m = lambda r, k: round(r["metrics"][k]["macro_f1"], 3)
    print(json.dumps({"primary_H1": primary, **{s: {k: m(r, k) for k in r["metrics"]} for s, r in sets},
                      "format3_gain": {s: (t or {}).get("delta") for s, t in secondary["format3_gain_jev_s3_minus_jev_s2"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

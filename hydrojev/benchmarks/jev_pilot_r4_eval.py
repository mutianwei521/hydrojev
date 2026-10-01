"""One-shot sealed evaluation of round 4 of the Jev pilot P2 v2 (q5: the q3 choice plus four decomposed noul questions).

Written before any round-4 sealed window or Jev response existed. Implements the plan frozen in
artifacts/jev_pilot_p2v2/design_freeze_r4.json. Round 4 is the fourth sealed round; the q5 questions and the adjudicated
decision rule were designed on burned sets only (train_s3, eval_novel_r2_s3) and confirmed on the burned round-3 sets,
so they are judged on fresh sets only: E1r4 (v1 subtypes, in-distribution for R2) and E4 (the novel4 family, written
before generation and never used for design; unseen by R2, by the questions and by the decision rule).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_pilot_p2v2_eval as ev
from hydrojev.benchmarks import jev_pilot_r4 as r4
from hydrojev.benchmarks.jev_pilot_p2v2_eval import _codes, _set, holm, label_scarce, paired_bootstrap

BOOT_SEED = 20260927
FREEZE = v.OUTDIR / r4.R4_FREEZE
TARGET = v.OUTDIR / "eval_analysis_r4.json"
SETS = {"E1r4": ("eval_id_r4", True), "E4": ("eval_novel4_r4", False)}
JEV_SYSTEMS = ("adjudicated", "jev_calibrated", "tree_calibrated", "fuse_calibrated", "override_jev")


def evaluate(name: str, freeze: Mapping[str, Any]) -> dict[str, Any]:
    split, loso = SETS[name]
    key = f"{split}_s3"
    cprior, tprior = (np.asarray(freeze["priors"][k]) for k in ("choice", "tree"))
    ev.REVISION = r4.REVISION  # _set reads the q5 stage; its choice answer is the q3 question
    res = _set(key, cprior, v.load_split("train_s3")["rows"], loso=loso)
    ans = r4.load(key)
    ids = [w["id"] for w in res["windows"]]
    rows = {r["id"]: r for r in v.load_split(key)["rows"]}
    states = [json.loads((v.OUTDIR / "requests" / key / f"{i}.state.json").read_text(encoding="utf-8")) for i in ids]
    preds = res["_preds"]
    chc = r4.calibrate(r4.choice_matrix(ans, ids), cprior)
    trc = r4.calibrate(r4.tree_matrix(ans, ids), tprior)
    preds["tree_calibrated"] = [v.CLASSES[k] for k in trc.argmax(1)]
    preds["fuse_calibrated"] = [v.CLASSES[k] for k in np.exp(0.5 * np.log(chc) + 0.5 * np.log(trc)).argmax(1)]
    preds["adjudicated"] = r4.adjudicate(ans, ids, states, cprior, tprior)
    preds["override_jev"] = v.override_jev(res["_pj"], cprior, states)
    preds["r1b"] = [v.r1b_classify(rows[i]["r1"], s) for i, s in zip(ids, states)]
    y = res["_y"]
    res["metrics"] = {k: {kk: p2.cause_metrics(y, p)[kk] for kk in ("macro_f1", "accuracy", "per_class_recall", "confusion_rows_true_cols_pred")}
                      for k, p in preds.items()}
    sub = np.asarray([w["subtype"] for w in res["windows"]])
    res["per_subtype_recall"] = {s: {k: round(float(np.mean(np.asarray(p)[sub == s] == np.asarray(y)[sub == s])), 3) for k, p in preds.items()}
                                 | {"n": int(np.sum(sub == s))} for s in sorted(set(sub))}
    res["mean_noul_by_true_class"] = {q: {c: round(float(np.mean([ans[i]["noul"][q] for i, t in zip(ids, y) if t == c])), 3) for c in v.CLASSES}
                                      for q in r4.SUBQ}
    res["jev_models"] = sorted({a["model"] for a in ans.values()})
    for k, w in enumerate(res["windows"]):
        w.update({n: preds[n][k] for n in ("adjudicated", "tree_calibrated", "fuse_calibrated", "override_jev", "r1b")}, noul=ans[w["id"]]["noul"])
    return res


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the round-4 sealed sets are evaluated once")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["questions_sha256"] == r4.questions_sha256() and freeze["state_format"] == v.STATE_FORMAT_PHYSICS
    assert freeze["eval_code_sha256"] == hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "eval code changed after the freeze"
    assert freeze["r4_module_sha256"] == hashlib.sha256(open(r4.__file__, "rb").read()).hexdigest(), "round-4 module changed after the freeze"
    rng = np.random.default_rng(BOOT_SEED)
    e1, e4 = evaluate("E1r4", freeze), evaluate("E4", freeze)

    def test(res, a, b):
        return paired_bootstrap(_codes(res["_y"]), _codes(res["_preds"][a]), _codes(res["_preds"][b]), rng)

    primary = {f"{s}: adjudicated vs {b}": test(r, "adjudicated", b) for s, r in (("E4", e4), ("E1r4", e1)) for b in ("r1", "r1b", "override_jev")}
    primary["E1r4-LOSO: adjudicated vs R2_loso"] = test(e1, "adjudicated", "r2_loso")
    h = holm({k: t["p_one_sided"] for k, t in primary.items()})
    for k in primary:
        primary[k].update(h[k])
    sets = (("E1r4", e1), ("E4", e4))
    secondary = {
        "q5_gain_adjudicated_minus_q3_choice_calibrated": {s: test(r, "adjudicated", "jev_calibrated") for s, r in sets},
        "adjudicated_vs_r2_full_reported_not_claimed": {s: test(r, "adjudicated", "r2_full") for s, r in sets},
        "other_training_free_variants_vs_r1b": {s: {k: test(r, k, "r1b") for k in ("jev_calibrated", "tree_calibrated", "fuse_calibrated", "override_jev")}
                                                for s, r in sets},
        "fusion_minus_r2_full": {s: test(r, "fusion_calibrated", "r2_full") for s, r in sets},
        "label_scarce_adjudicated": {s: label_scarce(v.load_split("train_s3")["rows"], {**r, "_preds": {**r["_preds"], "jev_calibrated": r["_preds"]["adjudicated"]}}, rng)
                                     for s, r in sets},
    }
    out = {"created_utc": v._now(), "round": 4, "design_freeze_r4_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
           "analysis_code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "primary_H1": primary, "secondary": secondary}
    for s, r in sets:
        out[s] = {k: val for k, val in r.items() if not k.startswith("_")}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    m = lambda r, k: round(r["metrics"][k]["macro_f1"], 3)
    print(json.dumps({"primary_H1": primary, **{s: {k: m(r, k) for k in r["metrics"]} for s, r in sets}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

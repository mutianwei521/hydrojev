"""One-shot sealed evaluation of round 6 (cross-network transfer of the round-5 screen) of the Jev pilot P2 v2.

Written before any round-6 sealed window, Jev response or LLM response existed. Implements the plan frozen in
artifacts/jev_pilot_p2v2/design_freeze_r6.json. Decision rules, priors and failure handling are those of round 5
(jev_cascade_eval): a failed Jev call is an 'insufficient_evidence' screen verdict (never benign, so the window goes to
the reviewer); a failed reviewer answer is a wrong answer. R2 is fitted on C-Town train_s2 only (zero-shot transfer);
the native R2, fitted on a v1 training split of the target network, is reported and never tested.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from hydrojev.benchmarks import jev_cascade as jc
from hydrojev.benchmarks import jev_cascade_eval as jce
from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_round6 as r6
from hydrojev.benchmarks import llm_baselines as lb
from hydrojev.benchmarks import llm_baselines_eval as le
from hydrojev.benchmarks.jev_pilot_p2v2_eval import _codes, _r2_proba, holm, label_scarce, macro_f1

BOOT_SEED = 20260930
NI_MARGIN = jce.NI_MARGIN
FREEZE = v.OUTDIR / r6.R6_FREEZE
CASCADE_FREEZE = v.OUTDIR / jc.CASCADE_FREEZE
TARGET = v.OUTDIR / "eval_analysis_r6.json"
ROUND5 = v.OUTDIR / "eval_analysis_cascade.json"  # read only, for the three-network summary


def _label(p: np.ndarray) -> np.ndarray:
    return np.asarray([v.CLASSES[k] for k in p.argmax(1)], dtype=object)


def evaluate(name: str, priors: Mapping[str, Any], ctown_train: list[Mapping[str, Any]]) -> dict[str, Any]:
    key = r6.KEYS[name]
    split = v.load_split(key)
    rows = split["rows"]
    net = split["network"]
    ids = [r["id"] for r in rows]
    y = np.asarray([r["label_internal"] for r in rows], dtype=object)
    sub = np.asarray([str(r["scenario_internal"]["subtype"]) for r in rows], dtype=object)
    probs = v.jev_probs(key, jc.REVISION)
    jvalid = np.asarray([i in probs for i in ids])
    pj = np.asarray([[float(probs[i].get(c, 0.0)) if i in probs else 0.25 for c in v.CLASSES] for i in ids])
    jev = jc.decisions(pj, priors["jev"], jvalid)
    r1 = np.asarray([r["r1"] for r in rows], dtype=object)
    x = np.asarray([r["x"] for r in rows], dtype=float)
    native = v.load_split(v.split_key(r6.TRAIN_OF[net], jc.FMT))["rows"]
    preds: dict[str, np.ndarray] = {"jev": jev, "r1": r1, "r2_transfer": _label(_r2_proba(ctown_train, x)),
                                    "r2_native_reference": _label(_r2_proba(native, x))}
    if split["family"] == "v1":  # unseen subtype AND unseen network: C-Town train_s2 without the tested subtype
        tsub = np.asarray([r["scenario_internal"]["subtype"] for r in ctown_train])
        r2l = np.empty(len(ids), dtype=object)
        for s in sorted(set(sub)):
            keep = [r for r, ts in zip(ctown_train, tsub) if ts != s]
            r2l[sub == s] = _label(_r2_proba(keep, x[sub == s]))
        preds["r2_transfer_loso"] = r2l
    preds["rule_cascade"], keep_rule = jc.rule_cascade(jev, r1)
    screen = {"rule_cascade": keep_rule}
    invalid, fallback = {}, {}
    for m in jc.REVIEWERS:
        pm, ok = le.model_matrix(m, key, ids)
        tag = m.split(":")[0]
        preds[tag] = jc.decisions(pm, priors[m], ok)
        preds[f"llm_cascade_{tag}"], screen[f"llm_cascade_{tag}"] = jc.llm_cascade(jev, r1, preds[tag])
        fallback[f"llm_cascade_{tag}_fallback_to_jev"] = jc.llm_cascade(jev, r1, np.where(ok, preds[tag], jev))[0]  # secondary, as round 5
        invalid[tag] = int((~ok).sum())
    metrics = {k: {kk: p2.cause_metrics(list(y), list(p))[kk] for kk in ("macro_f1", "accuracy", "per_class_recall", "confusion_rows_true_cols_pred")}
               for k, p in preds.items()}
    for k, p in preds.items():
        assert abs(macro_f1(_codes(y), _codes(p)) - metrics[k]["macro_f1"]) < 1e-9
    scr = {}
    for k, keep in screen.items():
        n_ok = int(np.sum(jev[keep] == y[keep]))
        scr[k] = {"accepted": int(keep.sum()), "accept_rate": round(float(keep.mean()), 4), "accept_precision": round(n_ok / max(int(keep.sum()), 1), 4),
                  "accept_precision_wilson95": jce.wilson(n_ok, int(keep.sum())), "deferred": int((~keep).sum()),
                  "accepted_by_true_class": {c: int(np.sum(keep & (y == c))) for c in v.CLASSES},
                  "accepted_true_attacks_by_subtype": {s: int(np.sum(keep & (y == "cyber_attack") & (sub == s))) for s in sorted(set(sub[y == "cyber_attack"]))}}
    jl = jce.jev_latency(key)
    api = [a["api_s"] for a in jl.values() if a["api_s"] is not None]
    wall = [a["wall_s"] for a in jl.values() if a["wall_s"] is not None]
    lat = {"jev_api_s": {"n": len(jl), "median": jce.q(api, 0.5), "p90": jce.q(api, 0.9)},
           "jev_wall_s": {"median": jce.q(wall, 0.5), "p90": jce.q(wall, 0.9)}}
    pos = {i: k for k, i in enumerate(ids)}
    for m in jc.REVIEWERS:
        tag = m.split(":")[0]
        ll = {}
        for rid in jc.latency_ids(key):
            f = lb._path(m, key, rid)
            rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            if rec.get("valid"):
                ll[rid] = rec["latency_s"]
        keep = screen[f"llm_cascade_{tag}"]
        casc = [(jl.get(i, {}).get("api_s") or 0.0) + (0.0 if keep[pos[i]] else t) for i, t in ll.items()]
        lat[tag] = {"n_timed_sequential": len(ll), "median_s": jce.q(list(ll.values()), 0.5), "p90_s": jce.q(list(ll.values()), 0.9),
                    "mean_s": round(float(np.mean(list(ll.values()))), 3) if ll else None,
                    "cascade_mean_s": round(float(np.mean(casc)), 3) if casc else None}
    per_sub = {s: {k: round(float(np.mean(p[sub == s] == y[sub == s])), 3) for k, p in preds.items()} | {"n": int(np.sum(sub == s))}
               for s in sorted(set(sub))}
    return {"split": key, "network": net, "family": split["family"], "n": len(ids), "monitors": split.get("monitors"),
            "subtype_counts": {s: int(np.sum(sub == s)) for s in sorted(set(sub))},
            "jev_missing": [i for i, o in zip(ids, jvalid) if not o], "jev_models": jce.jev_models(key), "reviewer_invalid": invalid,
            "fallback_macro_f1": {k: round(macro_f1(_codes(y), _codes(p)), 4) for k, p in fallback.items()}, "metrics": metrics, "screen": scr, "latency": lat, "per_subtype_recall": per_sub,
            "_y": y, "_preds": preds, "_x": x, "_native": native,
            "windows": [{"id": i, "truth": t, "subtype": s, **{n: str(preds[n][k]) for n in preds}} for k, (i, t, s) in enumerate(zip(ids, y, sub))]}


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the round-6 sealed sets are evaluated once")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["questions_sha256"] == v.questions_sha256(jc.REVISION) and freeze["llm_builder_sha256"] == lb.builder_sha256()
    assert freeze["eval_code_sha256"] == hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "eval code changed after the freeze"
    assert freeze["round6_module_sha256"] == r6.code_sha256(), "round-6 module changed after the freeze"
    assert freeze["cascade_module_sha256"] == hashlib.sha256(open(jc.__file__, "rb").read()).hexdigest(), "cascade module changed"
    priors = json.loads(CASCADE_FREEZE.read_text(encoding="utf-8"))["priors"]
    assert priors == freeze["priors"], "priors must be the frozen round-5 C-Town priors"
    ctown_train = v.load_split("train_s2")["rows"]
    rng = np.random.default_rng(BOOT_SEED)
    res = {name: evaluate(name, priors, ctown_train) for name in r6.SETS}

    def t(name, a, b):
        r = res[name]
        return jce.boot(_codes(r["_y"]), _codes(r["_preds"][a]), _codes(r["_preds"][b]), rng)

    primary = {}
    for name in r6.SETS:
        primary[f"{name}: llm_cascade_glm-5.2 non-inferior to glm-5.2 (margin {NI_MARGIN})"] = {**t(name, "llm_cascade_glm-5.2", "glm-5.2"), "kind": "noninferiority"}
        primary[f"{name}: jev > r2_transfer"] = {**t(name, "jev", "r2_transfer"), "kind": "superiority"}
    pv = {k: (p["p_noninferiority"] if p["kind"] == "noninferiority" else p["p_superiority"]) for k, p in primary.items()}
    h = holm(pv)
    for k in primary:
        primary[k].update(h[k])
    names = tuple(r6.SETS)
    secondary = {
        "jev_vs_r1": {s: t(s, "jev", "r1") for s in names},
        "rule_cascade_vs_r1": {s: t(s, "rule_cascade", "r1") for s in names},
        "rule_cascade_vs_jev": {s: t(s, "rule_cascade", "jev") for s in names},
        "llm_cascade_deepseek_vs_deepseek": {s: t(s, "llm_cascade_deepseek-v4-pro", "deepseek-v4-pro") for s in names},
        "llm_cascade_glm_vs_rule_cascade": {s: t(s, "llm_cascade_glm-5.2", "rule_cascade") for s in names},
        "jev_vs_llms": {s: {m: t(s, "jev", m) for m in ("glm-5.2", "deepseek-v4-pro")} for s in names},
        "jev_vs_r2_transfer_loso": {s: t(s, "jev", "r2_transfer_loso") for s in names if "r2_transfer_loso" in res[s]["_preds"]},
        "reference_not_tested_jev_vs_r2_native": {s: t(s, "jev", "r2_native_reference") for s in names},
        "label_scarce_native_jev": {s: label_scarce(r["_native"], {**r, "_y": list(r["_y"]), "_preds": {"jev_calibrated": list(r["_preds"]["jev"])}}, rng)
                                    for s, r in res.items()},
        "label_scarce_native_rule_cascade": {s: label_scarce(r["_native"], {**r, "_y": list(r["_y"]), "_preds": {"jev_calibrated": list(r["_preds"]["rule_cascade"])}}, rng)
                                             for s, r in res.items()},
    }
    three = {}
    if ROUND5.exists():
        r5 = json.loads(ROUND5.read_text(encoding="utf-8-sig"))
        for s5, lab in (("E1c", "CTOWN-V1"), ("E4c", "CTOWN-N4")):
            three[lab] = {k: round(m["macro_f1"], 4) for k, m in r5[s5]["metrics"].items()}
    for s, r in res.items():
        three[s] = {k: round(m["macro_f1"], 4) for k, m in r["metrics"].items()}
    out = {"created_utc": v._now(), "round": 6, "design_freeze_r6_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
           "analysis_code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "primary": primary, "secondary": secondary,
           "macro_f1_three_networks": three}
    for s, r in res.items():
        out[s] = {k: val for k, val in r.items() if not k.startswith("_")}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"primary": primary, "macro_f1": three, **{s: {"screen": r["screen"], "latency": r["latency"], "subtypes": r["subtype_counts"]}
                                                                for s, r in res.items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

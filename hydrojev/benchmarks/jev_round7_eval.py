"""One-shot sealed evaluation of round 7 (round 6 on Net1 with the pattern-timestep error fixed) of the Jev pilot P2 v2.

Written before any round-7 sealed window, Jev response or LLM response existed. Implements the plan frozen in
artifacts/jev_pilot_p2v2/design_freeze_r7.json. The per-set evaluation is jev_round6_eval.evaluate, unchanged (same
decision rules, priors, failure handling, R2 transfer from C-Town train_s2, native R2 reference, screen and latency
statistics); only the set list, the Holm family (four primary tests) and the bootstrap seed are new.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from hydrojev.benchmarks import jev_cascade as jc
from hydrojev.benchmarks import jev_cascade_eval as jce
from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_round6 as r6
from hydrojev.benchmarks import jev_round6_eval as r6e
from hydrojev.benchmarks import jev_round7 as r7
from hydrojev.benchmarks import llm_baselines as lb
from hydrojev.benchmarks.jev_pilot_p2v2_eval import _codes, holm, label_scarce

BOOT_SEED = 20261007
NI_MARGIN = r6e.NI_MARGIN
FREEZE = v.OUTDIR / r7.R7_FREEZE
TARGET = v.OUTDIR / "eval_analysis_r7.json"
ROUND5 = v.OUTDIR / "eval_analysis_cascade.json"  # read only
ROUND6 = v.OUTDIR / "eval_analysis_r6.json"  # read only (Net3 sets, and the round-6 Net1 sets reported as a generator error)


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the round-7 sealed sets are evaluated once")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["questions_sha256"] == v.questions_sha256(jc.REVISION) and freeze["llm_builder_sha256"] == lb.builder_sha256()
    assert freeze["eval_code_sha256"] == sha(__file__), "eval code changed after the freeze"
    assert freeze["round7_module_sha256"] == r7.code_sha256(), "round-7 module changed after the freeze"
    assert freeze["round6_module_sha256"] == r6.code_sha256() and freeze["round6_eval_module_sha256"] == sha(r6e.__file__)
    assert freeze["cascade_module_sha256"] == sha(jc.__file__), "cascade module changed"
    priors = json.loads(r6e.CASCADE_FREEZE.read_text(encoding="utf-8"))["priors"]
    assert priors == freeze["priors"], "priors must be the frozen round-5 C-Town priors"
    ctown_train = v.load_split("train_s2")["rows"]
    rng = np.random.default_rng(BOOT_SEED)
    res = {name: r6e.evaluate(name, priors, ctown_train) for name in r7.SETS}

    def t(name, a, b):
        r = res[name]
        return jce.boot(_codes(r["_y"]), _codes(r["_preds"][a]), _codes(r["_preds"][b]), rng)

    primary = {}
    for name in r7.SETS:
        primary[f"{name}: llm_cascade_glm-5.2 non-inferior to glm-5.2 (margin {NI_MARGIN})"] = {**t(name, "llm_cascade_glm-5.2", "glm-5.2"), "kind": "noninferiority"}
        primary[f"{name}: jev > r2_transfer"] = {**t(name, "jev", "r2_transfer"), "kind": "superiority"}
    pv = {k: (p["p_noninferiority"] if p["kind"] == "noninferiority" else p["p_superiority"]) for k, p in primary.items()}
    h = holm(pv)
    for k in primary:
        primary[k].update(h[k])
    names = tuple(r7.SETS)
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
    if ROUND6.exists():
        r6r = json.loads(ROUND6.read_text(encoding="utf-8-sig"))
        for s in ("N3-V1", "N3-N4", "N1-V1", "N1-N4"):
            three[s + " (r6)"] = {k: round(m["macro_f1"], 4) for k, m in r6r[s]["metrics"].items()}
    for s, r in res.items():
        three[s] = {k: round(m["macro_f1"], 4) for k, m in r["metrics"].items()}
    out = {"created_utc": v._now(), "round": 7, "design_freeze_r7_sha256": sha(FREEZE), "analysis_code_sha256": sha(__file__),
           "primary": primary, "secondary": secondary, "macro_f1_all_networks": three}
    for s, r in res.items():
        out[s] = {k: val for k, val in r.items() if not k.startswith("_")}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"primary": primary, "macro_f1": three, **{s: {"screen": r["screen"], "latency": r["latency"], "subtypes": r["subtype_counts"]}
                                                                for s, r in res.items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

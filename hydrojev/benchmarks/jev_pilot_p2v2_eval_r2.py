"""One-shot sealed evaluation of round 2 of the Jev pilot P2 v2 (larger sealed sets, design unchanged).

Written before any round-2 sealed data or Jev response existed. Implements the plan frozen in
artifacts/jev_pilot_p2v2/design_freeze_r2.json: Jev is training-free, so the primary comparator is the training-free
rule tree R1 on the same evidence; the supervised R2 is compared only where its labels do not cover the event (LOSO),
and against full labels as a reported secondary.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks.jev_pilot_p2v2_eval import _codes, _set, holm, label_scarce, paired_bootstrap

BOOT_SEED = 20260925
TARGET = v.OUTDIR / "eval_analysis_r2.json"


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the round-2 sealed sets are evaluated once")
    freeze = json.loads((v.OUTDIR / "design_freeze_r2.json").read_text(encoding="utf-8"))
    assert freeze["questions_sha256"] == v.questions_sha256(3)
    prior = np.asarray(freeze["inductive_prior"]["vector"])
    train = v.load_split("train_s2")["rows"]
    rng = np.random.default_rng(BOOT_SEED)
    e1 = _set("eval_id_r2_s2", prior, train, loso=True)
    e2 = _set("eval_novel_r2_s2", prior, train, loso=False)

    def test(res, a, b):
        return paired_bootstrap(_codes(res["_y"]), _codes(res["_preds"][a]), _codes(res["_preds"][b]), rng)

    primary = {"E2r2: Jev vs R1": test(e2, "jev_calibrated", "r1"),
               "E1r2: Jev vs R1": test(e1, "jev_calibrated", "r1"),
               "E1r2-LOSO: Jev vs R2_loso": test(e1, "jev_calibrated", "r2_loso")}
    h = holm({k: t["p_one_sided"] for k, t in primary.items()})
    for k in primary:
        primary[k].update(h[k])
    m = lambda res, k: res["metrics"][k]["macro_f1"]
    secondary = {
        "jev_vs_r2_full_reported_not_claimed": {"E1r2": test(e1, "jev_calibrated", "r2_full"), "E2r2": test(e2, "jev_calibrated", "r2_full")},
        "H4_jev_minus_r1": {"E1r2": round(m(e1, "jev_calibrated") - m(e1, "r1"), 4), "E2r2": round(m(e2, "jev_calibrated") - m(e2, "r1"), 4)},
        "training_free_variants_vs_r1": {s: {k: test(r, k, "r1") for k in ("jev_raw", "jev_loo_transductive")} for s, r in (("E1r2", e1), ("E2r2", e2))},
        "fusion_minus_r2_full": {"E1r2": test(e1, "fusion_calibrated", "r2_full"), "E2r2": test(e2, "fusion_calibrated", "r2_full")},
        "label_scarce": {"E1r2": label_scarce(train, e1, rng), "E2r2": label_scarce(train, e2, rng)},
    }
    out = {"created_utc": v._now(),
           "design_freeze_r2_sha256": hashlib.sha256((v.OUTDIR / "design_freeze_r2.json").read_bytes()).hexdigest(),
           "analysis_code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "primary_H1": primary, "secondary": secondary,
           "E1r2": {k: val for k, val in e1.items() if not k.startswith("_")}, "E2r2": {k: val for k, val in e2.items() if not k.startswith("_")}}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"primary_H1": primary, "E1r2": {k: round(m(e1, k), 3) for k in e1["metrics"]},
                      "E2r2": {k: round(m(e2, k), 3) for k in e2["metrics"]}, "H4": secondary["H4_jev_minus_r1"],
                      "vs_r2_full": {s: t["delta"] for s, t in secondary["jev_vs_r2_full_reported_not_claimed"].items()},
                      "crossover": {s: t["crossover_k_per_class"] for s, t in secondary["label_scarce"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

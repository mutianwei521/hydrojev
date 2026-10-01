"""Write artifacts/jev_pilot_p2v2/design_freeze_r7.json (round 7), once, before any round-7 sealed window exists."""

from __future__ import annotations

import hashlib
import json

from hydrojev.benchmarks import jev_cascade as jc
from hydrojev.benchmarks import jev_cascade_eval as jce
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_round6 as r6
from hydrojev.benchmarks import jev_round6_eval as r6e
from hydrojev.benchmarks import jev_round7 as r7
from hydrojev.benchmarks import jev_round7_eval as r7e
from hydrojev.benchmarks import llm_baselines as lb


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    target = v.OUTDIR / r7.R7_FREEZE
    if target.exists():
        raise FileExistsError(f"{target} exists")
    for split in r7.R7_SPLITS:
        if (v.OUTDIR / "splits" / f"{v.split_key(split, r7.FMT)}.json").exists():
            raise RuntimeError(f"sealed split {split} already exists; the freeze must precede it")
    c5 = json.loads(r6e.CASCADE_FREEZE.read_text(encoding="utf-8"))
    probe = json.loads((v.OUTDIR / "r7_probe.json").read_text(encoding="utf-8"))
    names = tuple(r7.SETS)
    out = {
        "created_utc": v._now(),
        "round": 7,
        "protocol_amendment": "PILOT_PROTOCOL_v2.md v2.8 section R",
        "reason": "round-6 Net1 generator error: WNTR Net1.inp has a 2-h pattern timestep and the frozen generator indexes pattern multipliers "
                  "by hour, so every pattern-driven Net1 event (burst, demand rise/fall, demand-pattern shift) started at 2*onset and was never "
                  "active inside its window; the round-6 Net1 sealed sets are reported as invalid for those subtypes and are not re-evaluated",
        "design": "round 6 unchanged on 'Net1h' = WNTR Net1 with every pattern resampled to 1 h (each multiplier repeated, pattern_timestep 3600); "
                  "the only change of round 7 is this loader; generator, gates, MAX_ATTEMPTS, monitor rule, calibration code, priors, R1, R2 "
                  "(C-Town train_s2), cascades and per-set evaluation (jev_round6_eval.evaluate) are those of round 6",
        "equivalence_event_free_vs_net1": probe["equivalence_event_free_vs_net1"],
        "sets": dict(r7.SETS),
        "split_keys": dict(r7.KEYS),
        "seeds": {s: list(cfg) for s, cfg in r7.R7_SPLITS.items()},
        "native_train_reference": {s: {"config": list(cfg), "sha256": sha(v.OUTDIR / "splits" / f"{v.split_key(s, r7.FMT)}.json")} for s, cfg in r7.R7_TRAIN.items()},
        "calibration_sha256": {f.name: sha(f) for f in sorted(v.OUTDIR.glob("*calibration_net1h.json"))},
        "probe_sha256": sha(v.OUTDIR / "r7_probe.json"),
        "reviewers": {"primary": jc.REVIEWERS[0], "secondary": jc.REVIEWERS[1]},
        "priors": c5["priors"],
        "classes": list(v.CLASSES),
        "questions_sha256": v.questions_sha256(jc.REVISION),
        "llm_builder_sha256": lb.builder_sha256(),
        "eval_code_sha256": sha(r7e.__file__),
        "round7_module_sha256": r7.code_sha256(),
        "round6_module_sha256": r6.code_sha256(),
        "round6_eval_module_sha256": sha(r6e.__file__),
        "cascade_module_sha256": sha(jc.__file__),
        "cascade_eval_module_sha256": sha(jce.__file__),
        "hypotheses_primary_holm_alpha_0.05": [h for s in names for h in (f"{s}: llm_cascade_glm-5.2 non-inferior to glm-5.2 (margin {r7e.NI_MARGIN})",
                                                                            f"{s}: jev > r2_transfer")],
        "secondary": "as round 6 (design_freeze_r6.json), on the two round-7 sets",
        "reference_not_tested": "r2_native_reference (R2 fitted on train_net1h_r7)",
        "ni_margin": r7e.NI_MARGIN,
        "boot_seed": r7e.BOOT_SEED,
        "n_boot": jce.N_BOOT,
        "latency_design": "as rounds 5-6: windows with index % 10 == 0 (20 per set) sent one at a time per reviewer before the concurrent run",
        "call_cap": "unlimited (author authorisation of 2026-09-26, extended to round 7 by the author's request of 2026-09-27 to fix Net1)",
        "rounds_disclosure": "rounds 1-5 on C-Town, round 6 on Net3 and Net1 (Net1 affected by the generator error), round 7 on Net1h; all disclosed",
        "design_exposure": "the round-7 design was chosen after the round-6 one-shot evaluation, from a diagnosis on burned round-6 Net1 data and on "
                           "development seeds (200_000_000+, gates and R1 only, no Jev or LLM call): R1 macro-F1 on 64 dev windows 0.420 with the "
                           "frozen Net1 loader vs 0.689 with the hourly loader; a causal-gate variant (0.450 / 0.704) and sigma-scaled event "
                           "magnitudes were tried on dev seeds and rejected in favour of changing only the error. No Jev or LLM output exists on any "
                           "Net1h window; the only Net1h windows before the freeze are the dev probe, r7_probe.json (seeds 13_073_500+, gates only) "
                           "and the native train split (not sealed, never sent to Jev or an LLM)",
    }
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target, sha(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

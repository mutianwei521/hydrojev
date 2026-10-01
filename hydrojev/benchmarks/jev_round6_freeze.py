"""Write artifacts/jev_pilot_p2v2/design_freeze_r6.json (round 6), once, before any round-6 sealed window exists."""

from __future__ import annotations

import hashlib
import json

from hydrojev.benchmarks import jev_cascade as jc
from hydrojev.benchmarks import jev_cascade_eval as jce
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_round6 as r6
from hydrojev.benchmarks import jev_round6_eval as r6e
from hydrojev.benchmarks import llm_baselines as lb


def sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> int:
    target = v.OUTDIR / r6.R6_FREEZE
    if target.exists():
        raise FileExistsError(f"{target} exists")
    for split in r6.R6_SPLITS:
        if (v.OUTDIR / "splits" / f"{v.split_key(split, r6.FMT)}.json").exists():
            raise RuntimeError(f"sealed split {split} already exists; the freeze must precede it")
    c5 = json.loads(r6e.CASCADE_FREEZE.read_text(encoding="utf-8"))
    names = tuple(r6.SETS)
    out = {
        "created_utc": v._now(),
        "round": 6,
        "protocol_amendment": "PILOT_PROTOCOL_v2.md v2.7 section P",
        "design": "round-5 screen and cascades, unchanged, on the other two networks of this project (EPANET Net3, EPANET Net1); every "
                  "fitted item is frozen from C-Town and transferred zero-shot (Jev and reviewer dev_s2 priors, R1, R2 fitted on C-Town train_s2); "
                  "only network properties are recomputed with frozen code (channels, residual/check calibration, station balance calibration)",
        "networks": {"Net3": "refCode EPANET Net3 as in p2._load_model", "Net1": "WNTR library Net1.inp (canonical EPANET Net1 with its tank-level pump control); "
                     "the refCode/FDI-WDNs copy has an empty [CONTROLS] section"},
        "monitor_rule": f"k = min({r6.MAX_MONITORS}, floor(pool/2)), pool = junctions with median pressure > 15 m that are not pump ends; Net3 k = 8, Net1 k = 4 (+ pump end 10)",
        "redraw_budget": f"MAX_ATTEMPTS = {r6.MAX_ATTEMPTS} (frozen loop 40, same +100_000 ladder); needed on Net1 where ~3.5 % of v1 bursts trip a check (r6_probe.json)",
        "sets": dict(r6.SETS),
        "split_keys": dict(r6.KEYS),
        "seeds": {s: list(cfg) for s, cfg in r6.R6_SPLITS.items()},
        "native_train_reference": {s: {"config": list(cfg), "sha256": sha(v.OUTDIR / "splits" / f"{v.split_key(s, r6.FMT)}.json")} for s, cfg in r6.R6_TRAIN.items()},
        "calibration_sha256": {f.name: sha(f) for f in sorted(v.OUTDIR.glob("*calibration_net[13].json"))},
        "probe_sha256": sha(v.OUTDIR / "r6_probe.json"),
        "reviewers": {"primary": jc.REVIEWERS[0], "secondary": jc.REVIEWERS[1]},
        "priors": c5["priors"],
        "classes": list(v.CLASSES),
        "questions_sha256": v.questions_sha256(jc.REVISION),
        "llm_builder_sha256": lb.builder_sha256(),
        "eval_code_sha256": sha(r6e.__file__),
        "round6_module_sha256": r6.code_sha256(),
        "cascade_module_sha256": sha(jc.__file__),
        "cascade_eval_module_sha256": sha(jce.__file__),
        "hypotheses_primary_holm_alpha_0.05": [h for s in names for h in (f"{s}: llm_cascade_glm-5.2 non-inferior to glm-5.2 (margin {r6e.NI_MARGIN})",
                                                                            f"{s}: jev > r2_transfer")],
        "secondary": ["jev vs r1", "rule_cascade vs r1 and vs jev", "llm_cascade_deepseek non-inferiority", "llm_cascade_glm vs rule_cascade",
                      "jev vs each reviewer", "jev vs r2_transfer_loso (V1 sets)", "reviewer failure falls back to jev",
                      "label-scarce curve of R2 fitted on the native train split", "screen accept rate/precision, latency, per-subtype recall, realised subtype mix",
                      "three-network macro-F1 summary with round-5 C-Town (eval_analysis_cascade.json, read only)"],
        "reference_not_tested": "r2_native_reference (R2 fitted on the target network's own v1 train split; needs labelled target-network data)",
        "ni_margin": r6e.NI_MARGIN,
        "boot_seed": r6e.BOOT_SEED,
        "n_boot": jce.N_BOOT,
        "latency_design": "as round 5: windows with index % 10 == 0 (20 per set) sent one at a time per reviewer before the concurrent run",
        "call_cap": "unlimited (author authorisation of 2026-09-26, extended to round 6 by the author's request of 2026-09-27 to run every network)",
        "rounds_disclosure": "rounds 1-5 are all on C-Town and are disclosed in PILOT_PROTOCOL_v2.md; round 6 is the first on Net3 and Net1",
        "design_exposure": "no Jev or LLM output has been computed on any Net3/Net1 window and no output of any method on a sealed round-6 window; the only Net3/Net1 windows seen before the freeze are the "
                           "realisability probe (seeds 11_070_000+, gates only), a gate diagnostic of 30 v1 bursts per network (seeds 12_000_000-12_000_029, "
                           "check statistics only) and the native train splits. On those train splits (not sealed) a smoke test of jev_round6_eval.evaluate "
                           "computed R1/R2 outputs with no Jev or LLM answer present (macro-F1 Net3: r1 0.650, r2_transfer 0.783, r2_native 0.970 in-sample; "
                           "Net1: r1 0.453, r2_transfer 0.352, r2_native 0.930 in-sample); the hypotheses and the evaluation code were written before "
                           "that smoke test and were not changed after it, except converting subtype names to plain str (a JSON-key fix)",
    }
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target, sha(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

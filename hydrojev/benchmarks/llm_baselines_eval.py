"""One-shot analysis: Jev versus general-purpose LLMs (Ollama) as training-free baselines on the round-2 sealed sets.

Written before any LLM answer on dev_s2 or the round-2 sealed sets existed; implements the plan frozen in
artifacts/llm_baselines/design_freeze_llm.json and writes artifacts/llm_baselines/analysis.json once.
Every LLM is calibrated exactly as Jev is: inductive prior = mean of its own four-cause probabilities on the 40
unlabeled dev_s2 windows, decision argmax(p / prior). A window whose response stayed invalid is scored as an abstention
(always wrong). Direction is not assumed, so the Jev-versus-model tests are two-sided.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import llm_baselines as lb
from hydrojev.benchmarks.jev_pilot_p2v2_eval import N_BOOT, _codes, _set, holm, macro_f1

BOOT_SEED = 20260926
TARGET = lb.OUTDIR / "analysis.json"
FREEZE = lb.OUTDIR / "design_freeze_llm.json"
AMENDMENT = lb.OUTDIR / "design_freeze_llm_amendment1.json"
AMENDMENT2 = lb.OUTDIR / "design_freeze_llm_amendment2.json"
AMENDMENT3 = lb.OUTDIR / "design_freeze_llm_amendment3.json"
AMENDMENT4 = lb.OUTDIR / "design_freeze_llm_amendment4.json"
SETS = {"E1r2": "eval_id_r2_s2", "E2r2": "eval_novel_r2_s2"}


def model_matrix(model: str, split: str, ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    probs, _ = lb.load_probs(model, split)
    valid =np.asarray([lb.is_valid(model, split, i) for i in ids])
    return v.jev_matrix(probs, ids), valid


def prior_of(model: str) -> np.ndarray:
    ids = [r["id"] for r in v.load_split("dev_s2")["rows"]]
    pm, ok = model_matrix(model, "dev_s2", ids)
    return np.maximum(pm[ok].mean(0), 0.01) if ok.any() else np.full(len(v.CLASSES), 0.25)


def decide(pm: np.ndarray, valid: np.ndarray, prior: np.ndarray | None) -> list[str]:
    return [d if ok else "insufficient_evidence" for d, ok in zip(v._decide(pm, prior), valid)]


def two_sided(y: np.ndarray, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    strata = [np.flatnonzero(y == c) for c in np.unique(y)]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in strata])
        d[i] = macro_f1(y[s], a[s]) - macro_f1(y[s], b[s])
    return {"delta": round(macro_f1(y, a) - macro_f1(y, b), 4), "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0.0), np.mean(d >= 0.0))))}


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the LLM comparison is analysed once")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["user_message_builder_sha256"] == lb.builder_sha256() and freeze["models"] == list(lb.AVAILABLE)
    amend = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert amend["freeze_sha256"] == hashlib.sha256(FREEZE.read_bytes()).hexdigest()
    amend2 = json.loads(AMENDMENT2.read_text(encoding="utf-8"))
    assert amend2["amendment1_sha256"] == hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
    amend3 = json.loads(AMENDMENT3.read_text(encoding="utf-8"))
    assert amend3["amendment2_sha256"] == hashlib.sha256(AMENDMENT2.read_bytes()).hexdigest()
    amend4 = json.loads(AMENDMENT4.read_text(encoding="utf-8"))
    assert amend4["primary_models"] == list(lb.COMPLETING) and amend4["amendment3_sha256"] == hashlib.sha256(AMENDMENT3.read_bytes()).hexdigest()
    jprior = np.asarray(json.loads((v.OUTDIR / "design_freeze_r2.json").read_text(encoding="utf-8"))["inductive_prior"]["vector"])
    train = v.load_split("train_s2")["rows"]
    rng = np.random.default_rng(BOOT_SEED)
    priors = {m: prior_of(m) for m in lb.AVAILABLE + lb.ADDED}
    sets, tests = {}, {}
    for s, split in SETS.items():
        res = _set(split, jprior, train, loso=False)
        ids = [w["id"] for w in res["windows"]]
        y = res["_y"]
        preds = {"jev_calibrated": res["_preds"]["jev_calibrated"], "jev_raw": res["_preds"]["jev_raw"], "r1": res["_preds"]["r1"],
                 "r2_full_supervised": res["_preds"]["r2_full"]}
        info, partial = {}, {}
        for m in list(lb.RETIRED_MIDRUN) + list(lb.DROPPED):  # descriptive only, on the windows answered before retirement / stop, Jev on the same windows
            pm, ok = model_matrix(m, split, ids)
            if ok.any():
                sel = np.flatnonzero(ok)
                ys = [y[j] for j in sel]
                pc = decide(pm[sel], ok[sel], priors[m])
                partial[m] = {"n_answered": int(ok.sum()), "model_calibrated": p2.cause_metrics(ys, pc)["macro_f1"],
                              "model_raw": p2.cause_metrics(ys, decide(pm[sel], ok[sel], None))["macro_f1"],
                              "jev_calibrated_same_windows": p2.cause_metrics(ys, [preds["jev_calibrated"][j] for j in sel])["macro_f1"],
                              "r1_same_windows": p2.cause_metrics(ys, [preds["r1"][j] for j in sel])["macro_f1"],
                              "per_class_n": {c: ys.count(c) for c in v.CLASSES}}
            else:
                partial[m] = {"n_answered": 0}
        for m in lb.COMPLETING:
            pm, ok = model_matrix(m, split, ids)
            preds[f"{m}|calibrated"] = decide(pm, ok, priors[m])
            preds[f"{m}|raw"] = decide(pm, ok, None)
            info[m] = {"invalid": int((~ok).sum()), "mean_probability": dict(zip(v.CLASSES, np.round(pm[ok].mean(0), 3).tolist())) if ok.any() else None}
            tests[f"{s}: Jev vs {m}"] = two_sided(_codes(y), _codes(preds["jev_calibrated"]), _codes(preds[f"{m}|calibrated"]), rng)
        metrics = {k: {kk: p2.cause_metrics(y, p)[kk] for kk in ("macro_f1", "accuracy", "per_class_recall")} for k, p in preds.items()}
        ranking = sorted(((k, metrics[k]["macro_f1"]) for k in preds), key=lambda t: -t[1])
        sets[s] = {"split": split, "n": len(ids), "metrics": metrics, "ranking_macro_f1": ranking, "models": info,
                   "retired_midrun_partial_descriptive": partial,
                   "llm_vs_r1": {m: two_sided(_codes(y), _codes(preds[f"{m}|calibrated"]), _codes(preds["r1"]), rng) for m in lb.COMPLETING},
                   "windows": [{"id": i, "truth": t, **{k: preds[k][j] for k in preds}} for j, (i, t) in enumerate(zip(ids, y))]}
    h = holm({k: t["p_two_sided"] for k, t in tests.items()})
    for k in tests:
        tests[k].update(h[k])
    out = {"created_utc": v._now(), "design_freeze_llm_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
           "analysis_code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
           "priors": {m: dict(zip(v.CLASSES, np.round(p, 4).tolist())) for m, p in priors.items()},
           "amendment1_sha256": hashlib.sha256(AMENDMENT.read_bytes()).hexdigest(),
           "amendment2_sha256": hashlib.sha256(AMENDMENT2.read_bytes()).hexdigest(),
           "amendment3_sha256": hashlib.sha256(AMENDMENT3.read_bytes()).hexdigest(),
           "amendment4_sha256": hashlib.sha256(AMENDMENT4.read_bytes()).hexdigest(),
           "primary_jev_vs_llm": tests, "sets": sets, "retired_unavailable": lb.RETIRED, "retired_midrun": lb.RETIRED_MIDRUN, "added_amendment2": list(lb.ADDED), "dropped_amendment3_4": lb.DROPPED}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({s: sets[s]["ranking_macro_f1"] for s in sets}, indent=1))
    print(json.dumps({k: {kk: t[kk] for kk in ("delta", "ci95", "p_two_sided", "reject_null")} for k, t in tests.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

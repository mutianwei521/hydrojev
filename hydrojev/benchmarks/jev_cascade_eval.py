"""One-shot sealed evaluation of round 5 (the Jev-screen cascades) of the Jev pilot P2 v2.

Written before any round-5 sealed window, Jev response or LLM response existed. Implements the plan frozen in
artifacts/jev_pilot_p2v2/design_freeze_cascade.json. A failed Jev call is an 'insufficient_evidence' screen verdict
(never benign, so the window goes to the reviewer); a failed reviewer answer is a wrong answer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from hydrojev.benchmarks import jev_cascade as jc
from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.benchmarks import jev_pilot_p2 as p2
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import llm_baselines as lb
from hydrojev.benchmarks import llm_baselines_eval as le
from hydrojev.benchmarks.jev_pilot_p2v2_eval import _codes, _r2_proba, holm, label_scarce, macro_f1

BOOT_SEED = 20260928
N_BOOT = 10_000
NI_MARGIN = 0.03
FREEZE = v.OUTDIR / jc.CASCADE_FREEZE
TARGET = v.OUTDIR / "eval_analysis_cascade.json"


def boot(y: np.ndarray, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Class-stratified paired bootstrap of macro-F1(a) - macro-F1(b)."""

    strata = [np.flatnonzero(y == c) for c in np.unique(y)]
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in strata])
        d[i] = macro_f1(y[s], a[s]) - macro_f1(y[s], b[s])
    return {"delta": round(macro_f1(y, a) - macro_f1(y, b), 4), "ci95": [round(float(np.quantile(d, 0.025)), 4), round(float(np.quantile(d, 0.975)), 4)],
            "p_superiority": float(np.mean(d <= 0.0)), "p_noninferiority": float(np.mean(d <= -NI_MARGIN))}


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p, z = k / n, 1.959964
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 4), round(float(c + h), 4)]


def jev_latency(key: str) -> dict[str, dict[str, float]]:
    st = v.stage_name(key, jc.REVISION)
    wall = {e["tag"].replace(f"{st}_", ""): e.get("wall_seconds") for e in p1_entries() if e.get("stage") == st}
    out = {}
    for f in (v.LEDGER_DIR / "responses" / st).glob("*.response.json"):
        rid = f.name.split(".")[0].replace(f"{st}_", "")
        out[rid] = {"api_s": json.loads(f.read_text(encoding="utf-8-sig")).get("elapsed_seconds"), "wall_s": wall.get(rid)}
    return out


def p1_entries() -> list[dict[str, Any]]:
    return p1.ledger_entries(v.LEDGER_DIR)


def jev_models(key: str) -> list[str]:
    st = v.stage_name(key, jc.REVISION)
    return sorted({p2.parse_response(json.loads(f.read_text(encoding="utf-8-sig"))["response"]).model
                   for f in (v.LEDGER_DIR / "responses" / st).glob("*.response.json")})


def llm_latency(model: str, key: str) -> dict[str, float]:
    out = {}
    for rid in jc.latency_ids(key):
        f = lb._path(model, key, rid)
        rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        if rec.get("valid"):
            out[rid] = rec["latency_s"]
    return out


def q(x, p):
    return round(float(np.quantile(np.asarray(x, dtype=float), p)), 3) if len(x) else None


def evaluate(name: str, freeze: Mapping[str, Any], train: list[Mapping[str, Any]]) -> dict[str, Any]:
    key = jc.KEYS[name]
    rows = v.load_split(key)["rows"]
    ids = [r["id"] for r in rows]
    y = np.asarray([r["label_internal"] for r in rows], dtype=object)
    sub = np.asarray([r["scenario_internal"]["subtype"] for r in rows])
    probs = v.jev_probs(key, jc.REVISION)
    jvalid = np.asarray([i in probs for i in ids])
    pj = np.asarray([[float(probs[i].get(c, 0.0)) if i in probs else 0.25 for c in v.CLASSES] for i in ids])
    jev = jc.decisions(pj, freeze["priors"]["jev"], jvalid)
    r1 = np.asarray([r["r1"] for r in rows], dtype=object)
    x = np.asarray([r["x"] for r in rows], dtype=float)
    pr_full = _r2_proba(train, x)
    preds: dict[str, np.ndarray] = {"jev": jev, "r1": r1, "r2_full": np.asarray([v.CLASSES[k] for k in pr_full.argmax(1)], dtype=object)}
    if name == "E1c":
        tsub = np.asarray([r["scenario_internal"]["subtype"] for r in train])
        r2l = np.empty(len(ids), dtype=object)
        for s in sorted(set(sub)):
            keep = [r for r, ts in zip(train, tsub) if ts != s]
            r2l[sub == s] = [v.CLASSES[k] for k in _r2_proba(keep, x[sub == s]).argmax(1)]
        preds["r2_loso"] = r2l
    preds["rule_cascade"], keep_rule = jc.rule_cascade(jev, r1)
    screen = {"rule_cascade": keep_rule}
    fallback = {}
    for m in jc.REVIEWERS:
        pm, ok = le.model_matrix(m, key, ids)
        tag = m.split(":")[0]
        preds[tag] = jc.decisions(pm, freeze["priors"][m], ok)
        preds[f"llm_cascade_{tag}"], screen[f"llm_cascade_{tag}"] = jc.llm_cascade(jev, r1, preds[tag])
        fb = np.where(ok, preds[tag], jev)  # secondary: a failed reviewer answer falls back to the screen verdict
        fallback[f"llm_cascade_{tag}_fallback_to_jev"] = jc.llm_cascade(jev, r1, fb)[0]
        fallback[f"{tag}_invalid"] = int((~ok).sum())
    metrics = {k: {kk: p2.cause_metrics(list(y), list(p))[kk] for kk in ("macro_f1", "accuracy", "per_class_recall", "confusion_rows_true_cols_pred")}
               for k, p in preds.items()}
    for k, p in preds.items():
        assert abs(macro_f1(_codes(y), _codes(p)) - metrics[k]["macro_f1"]) < 1e-9
    scr = {}
    for k, keep in screen.items():
        n_ok = int(np.sum(jev[keep] == y[keep]))
        scr[k] = {"accepted": int(keep.sum()), "accept_rate": round(float(keep.mean()), 4), "accept_precision": round(n_ok / max(int(keep.sum()), 1), 4),
                  "accept_precision_wilson95": wilson(n_ok, int(keep.sum())), "deferred": int((~keep).sum()),
                  "accepted_by_true_class": {c: int(np.sum(keep & (y == c))) for c in v.CLASSES},
                  "accepted_wrong_true_attacks": int(np.sum(keep & (y == "cyber_attack")))}
    jl = jev_latency(key)
    lat = {"jev_api_s": {"n": len(jl), "median": q([a["api_s"] for a in jl.values() if a["api_s"] is not None], 0.5),
                         "p90": q([a["api_s"] for a in jl.values() if a["api_s"] is not None], 0.9)},
           "jev_wall_s": {"median": q([a["wall_s"] for a in jl.values() if a["wall_s"] is not None], 0.5),
                          "p90": q([a["wall_s"] for a in jl.values() if a["wall_s"] is not None], 0.9)}}
    pos = {i: k for k, i in enumerate(ids)}
    for m in jc.REVIEWERS:
        tag = m.split(":")[0]
        ll = llm_latency(m, key)
        keep = screen[f"llm_cascade_{tag}"]
        casc = [(jl.get(i, {}).get("api_s") or 0.0) + (0.0 if keep[pos[i]] else t) for i, t in ll.items()]
        lat[tag] = {"n_timed_sequential": len(ll), "median_s": q(list(ll.values()), 0.5), "p90_s": q(list(ll.values()), 0.9),
                    "mean_s": round(float(np.mean(list(ll.values()))), 3) if ll else None, "min_s": q(list(ll.values()), 0.0),
                    "cascade_median_s": q(casc, 0.5), "cascade_mean_s": round(float(np.mean(casc)), 3) if casc else None}
    per_sub = {s: {k: round(float(np.mean(p[sub == s] == y[sub == s])), 3) for k, p in {**preds, **fallback}.items() if isinstance(p, np.ndarray)} | {"n": int(np.sum(sub == s))}
               for s in sorted(set(sub))}
    return {"split": key, "n": len(ids), "jev_missing": [i for i, o in zip(ids, jvalid) if not o], "jev_models": jev_models(key),
            "metrics": metrics, "screen": scr, "latency": lat, "per_subtype_recall": per_sub,
            "fallback_macro_f1": {k: round(macro_f1(_codes(y), _codes(p)), 4) for k, p in fallback.items() if isinstance(p, np.ndarray)},
            "reviewer_invalid": {k: val for k, val in fallback.items() if not isinstance(val, np.ndarray)},
            "_y": y, "_preds": preds, "_x": x, "_fallback": fallback,
            "windows": [{"id": i, "truth": t, "subtype": s, **{n: str(preds[n][k]) for n in preds}} for k, (i, t, s) in enumerate(zip(ids, y, sub))]}


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"{TARGET} exists; the round-5 sealed sets are evaluated once")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["questions_sha256"] == v.questions_sha256(jc.REVISION) and freeze["llm_builder_sha256"] == lb.builder_sha256()
    assert freeze["eval_code_sha256"] == hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "eval code changed after the freeze"
    assert freeze["cascade_module_sha256"] == hashlib.sha256(open(jc.__file__, "rb").read()).hexdigest(), "cascade module changed after the freeze"
    train = v.load_split("train_s2")["rows"]
    rng = np.random.default_rng(BOOT_SEED)
    res = {name: evaluate(name, freeze, train) for name in jc.SETS}

    def t(name, a, b):
        r = res[name]
        return boot(_codes(r["_y"]), _codes(r["_preds"][a]), _codes(r["_preds"][b]), rng)

    primary = {}
    for name in ("E4c", "E1c"):
        for b in ("r1", "jev"):
            primary[f"{name}: rule_cascade > {b}"] = {**t(name, "rule_cascade", b), "kind": "superiority"}
        primary[f"{name}: llm_cascade_glm-5.2 non-inferior to glm-5.2 (margin {NI_MARGIN})"] = {**t(name, "llm_cascade_glm-5.2", "glm-5.2"), "kind": "noninferiority"}
    primary["E1c-LOSO: jev > r2_loso"] = {**t("E1c", "jev", "r2_loso"), "kind": "superiority"}
    pv = {k: (p["p_noninferiority"] if p["kind"] == "noninferiority" else p["p_superiority"]) for k, p in primary.items()}
    h = holm(pv)
    for k in primary:
        primary[k].update(h[k])
    sets = tuple(res.items())
    secondary = {
        "llm_cascade_deepseek_vs_deepseek": {s: t(s, "llm_cascade_deepseek-v4-pro", "deepseek-v4-pro") for s, _ in sets},
        "llm_cascade_glm_vs_rule_cascade": {s: t(s, "llm_cascade_glm-5.2", "rule_cascade") for s, _ in sets},
        "rule_cascade_vs_llms": {s: {m: t(s, "rule_cascade", m) for m in ("glm-5.2", "deepseek-v4-pro")} for s, _ in sets},
        "reported_not_claimed_vs_r2_full": {s: {k: t(s, k, "r2_full") for k in ("jev", "rule_cascade", "llm_cascade_glm-5.2")} for s, _ in sets},
        "E1c_rule_cascade_vs_r2_loso": t("E1c", "rule_cascade", "r2_loso"),
        "label_scarce_jev": {s: label_scarce(train, {**r, "_y": list(r["_y"]), "_preds": {"jev_calibrated": list(r["_preds"]["jev"])}}, rng) for s, r in sets},
        "label_scarce_rule_cascade": {s: label_scarce(train, {**r, "_y": list(r["_y"]), "_preds": {"jev_calibrated": list(r["_preds"]["rule_cascade"])}}, rng)
                                      for s, r in sets},
    }
    out = {"created_utc": v._now(), "round": 5, "design_freeze_cascade_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
           "analysis_code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "primary": primary, "secondary": secondary}
    for s, r in sets:
        out[s] = {k: val for k, val in r.items() if not k.startswith("_")}
    TARGET.write_text(json.dumps(p2._safe(out), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"primary": primary, **{s: {"macro_f1": {k: round(m["macro_f1"], 3) for k, m in r["metrics"].items()}, "screen": r["screen"],
                                                 "latency": r["latency"]} for s, r in sets}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

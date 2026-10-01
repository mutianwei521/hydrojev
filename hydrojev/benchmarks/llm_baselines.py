"""Training-free general-purpose LLM baselines for the P2 v2 cause-discrimination task (via a local Ollama server).

Every model receives exactly the state and the frozen revision-3 question that Jev received, and answers with a choice
and a probability distribution over the same five event types. Responses are cached one file per window and never
re-requested once a valid answer exists. Calibration mirrors Jev's: an inductive prior from the model's own answers on
the 40 unlabeled dev_s2 windows, decisions argmax(p / prior).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydrojev.benchmarks import jev_pilot_p2v2 as v

OLLAMA = "http://localhost:11434/api/chat"
OUTDIR = Path("artifacts/llm_baselines")
REVISION = 3
SPLITS = ("dev_s2", "eval_id_r2_s2", "eval_novel_r2_s2")
MODELS = (
    "deepseek-v4-flash:cloud", "deepseek-v4-pro:cloud", "minimax-m2.7:cloud", "qwen3.5:397b-cloud", "glm-5.2:cloud",
    "gemma4:cloud", "gemma4:31b-cloud", "kimi-k2.5:cloud", "glm-4.6:cloud", "minimax-m2:cloud", "qwen3-vl:235b-cloud",
    "qwen3-coder:480b-cloud", "deepseek-v3.1:671b-cloud", "gpt-oss:20b-cloud", "gpt-oss:120b-cloud",
)
RETIRED = {  # returned HTTP 410 "was retired at <date>" by Ollama Cloud on 2026-09-25, before any evaluation call
    "kimi-k2.5:cloud": "2026-07-31", "glm-4.6:cloud": "2026-06-16", "minimax-m2:cloud": "2026-06-16", "qwen3-vl:235b-cloud": "2026-06-16",
    "qwen3-coder:480b-cloud": "2026-07-15", "deepseek-v3.1:671b-cloud": "2026-07-15",
}
AVAILABLE = tuple(m for m in MODELS if m not in RETIRED)  # the 9 pre-registered in design_freeze_llm.json
RETIRED_MIDRUN = {  # HTTP 410 "was retired at 2026-09-25 00:00:00 -0700 PDT" while the run was in progress
    "deepseek-v4-flash:cloud": "2026-09-25 (served snapshot deepseek-v4-flash:0731)", "qwen3.5:397b-cloud": "2026-09-25",
}
ADDED = (  # author request 2026-09-25 (amendment2), before any analysis: successor of the retired flash model, plus glm-5.3
    "deepseek-v4.1-flash:cloud", "glm-5.3:cloud",
)
DROPPED = {  # amendment3, author decision 2026-09-26 before any analysis: collection stopped by the account's monthly usage limit
    "glm-5.3:cloud": "stopped at 40/40 dev, 199/200 E1r2, 54/200 E2r2; reported descriptively on answered windows",
    # amendment4, 2026-09-26: HTTP 402 'this model is not included in your free usage' once the account fell back to the free tier
    "deepseek-v4.1-flash:cloud": "stopped at 40/40 dev, 200/200 E1r2, 118/200 E2r2 (HTTP 402); reported descriptively on answered windows",
}
COMPLETING = tuple(m for m in AVAILABLE if m not in RETIRED_MIDRUN) + tuple(m for m in ADDED if m not in DROPPED)
MAX_ATTEMPTS = 3  # answered-but-unparseable responses per window
MAX_TRANSPORT_ATTEMPTS = 8  # including transport / rate-limit failures
TIMEOUT_S = 600


class AdaptiveLimit:
    """Account-wide cap on concurrent Ollama Cloud requests, found online (AIMD): +1 per 5 successes, halved on HTTP 429."""

    def __init__(self, start: int, ceiling: int) -> None:
        self.limit, self.ceiling, self.active, self.ok = start, ceiling, 0, 0
        self.cv = threading.Condition()

    def __enter__(self) -> "AdaptiveLimit":
        with self.cv:
            while self.active >= self.limit:
                self.cv.wait()
            self.active += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with self.cv:
            self.active -= 1
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                self.limit, self.ok = max(4, self.limit // 2), 0
            elif exc is None:
                self.ok += 1
                if self.ok >= 5 and self.limit < self.ceiling:
                    self.limit, self.ok = self.limit + 1, 0
            self.cv.notify_all()


_inflight = AdaptiveLimit(4, 4)

SYSTEM_PROMPT = (
    "You are an expert analyst of water distribution SCADA data. You receive the evidence for one six-hour window and "
    "one multiple-choice question. Follow the question's instructions and criteria. Answer ONLY with a JSON object "
    '{"choice": <one option>, "probabilities": {<option>: <number>, ...}} giving a probability for every option; the '
    "probabilities must sum to 1."
)


def schema() -> dict[str, Any]:
    props = {c: {"type": "number"} for c in v.EVENT_TYPES}
    return {"type": "object", "properties": {"choice": {"type": "string", "enum": list(v.EVENT_TYPES)},
                                             "probabilities": {"type": "object", "properties": props, "required": list(v.EVENT_TYPES)}},
            "required": ["choice", "probabilities"]}


def user_message(state: Mapping[str, Any]) -> str:
    req = v.build_request(state, revision=REVISION)
    q = req["questions"]["event_type"]
    return json.dumps({"evidence": req["state"], "question": {"options": list(v.EVENT_TYPES), "instructions": q["instructions"],
                                                              "criteria": q["criteria"]}}, ensure_ascii=False)


def parse(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object")
    obj = json.loads(m.group(0))
    raw = obj.get("probabilities") or {}
    p = {c: max(float(raw.get(c, 0.0) or 0.0), 0.0) for c in v.EVENT_TYPES}
    s = sum(p.values())
    if s <= 0:
        raise ValueError("no probability mass")
    p = {c: x / s for c, x in p.items()}
    choice = obj.get("choice")
    if choice not in v.EVENT_TYPES:
        choice = max(p, key=p.get)
    # Jev's interface returns probabilities rounded to 2 decimals; apply the same resolution to every baseline.
    return {"choice": choice, "probabilities": {c: round(x, 2) for c, x in p.items()}}


def _path(model: str, split: str, rid: str) -> Path:
    return OUTDIR / "responses" / model.replace(":", "_").replace("/", "_") / split / f"{rid}.json"


def ask(model: str, split: str, rid: str) -> dict[str, Any]:
    if split in SPLITS and not (OUTDIR / "design_freeze_llm.json").exists():
        raise RuntimeError("dev/sealed splits are queried only after design_freeze_llm.json is written")
    out = _path(model, split, rid)
    if out.exists():
        rec = json.loads(out.read_text(encoding="utf-8"))
        if rec.get("valid"):
            return rec
    state = json.loads((v.OUTDIR / "requests" / split / f"{rid}.state.json").read_text(encoding="utf-8"))
    body = {"model": model, "stream": False, "format": schema(), "options": {"temperature": 0, "seed": 20260925},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message(state)}]}
    errors, answered, rec, admitted_waits = [], 0, None, 0
    for attempt in range(MAX_TRANSPORT_ATTEMPTS):
        t0 = time.time()
        try:
            req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            while True:
                try:
                    with _inflight, urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                        resp = json.loads(r.read())
                    break
                except urllib.error.HTTPError as e:  # 429 = not admitted (account concurrency cap): wait, do not spend an attempt
                    if e.code == 410:  # model retired by the provider: permanent, never retried
                        raise RuntimeError(f"model retired: {e.read()[:200].decode(errors='replace')}") from e
                    if e.code != 429 or admitted_waits >= 2000:
                        raise
                    msg = e.read()[:300].decode(errors="replace")
                    if "usage limit" in msg:  # account quota exhausted, not a concurrency cap: stop, do not wait it out
                        raise RuntimeError("usage limit reached") from e
                    admitted_waits += 1
                    time.sleep(3 + random.random() * 7)
        except Exception as e:  # transport / HTTP (rate limit, overload, timeout): backoff and retry
            errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            print(f"{v._now()} RETRY {model} {split} {rid} attempt {attempt + 1}: {errors[-1][:120]}", flush=True)
            if errors[-1].startswith("RuntimeError: model retired"):
                break
            if errors[-1].startswith("RuntimeError: usage limit"):
                raise SystemExit("Ollama account usage limit reached; stopping without recording this window")
            time.sleep(min(30 * (attempt + 1), 180))
            continue
        content = resp.get("message", {}).get("content", "")
        answered += 1
        try:
            ans = parse(content)
        except Exception as e:  # the model answered but not in the required format
            errors.append(f"parse {type(e).__name__}: {str(e)[:200]} | {content[:200]}")
            if answered >= MAX_ATTEMPTS:
                break
            continue
        rec = {"model": model, "resolved_model": resp.get("model"), "split": split, "id": rid, "valid": True, "attempts": attempt + 1,
               "latency_s": round(time.time() - t0, 2), "concurrency_waits": admitted_waits, "answer": ans, "raw_content": content[:4000], "eval_count": resp.get("eval_count"),
               "errors": errors}
        break
    if rec is None:
        rec = {"model": model, "split": split, "id": rid, "valid": False, "answered": answered, "errors": errors}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


_lock = threading.Lock()


def run_model(model: str, splits: Sequence[str], limit: int | None = None, concurrency: int = 4) -> None:
    for split in splits:
        ids = [r["id"] for r in v.load_split(split)["rows"]][:limit]
        done = bad = 0
        with ThreadPoolExecutor(concurrency) as ex:
            for rec in ex.map(lambda rid: ask(model, split, rid), ids):
                done += 1
                bad += not rec["valid"]
                if done % 20 == 0 or done == len(ids):
                    with _lock:
                        print(f"{v._now()} {model} {split} {done}/{len(ids)} invalid={bad} limit={_inflight.limit} active={_inflight.active}", flush=True)


def is_valid(model: str, split: str, rid: str) -> bool:
    f = _path(model, split, rid)
    return f.exists() and bool(json.loads(f.read_text(encoding="utf-8")).get("valid"))


def builder_sha256() -> str:
    """Hash of everything that shapes what a model sees and how its answer is read."""

    parts = [SYSTEM_PROMPT, json.dumps(schema(), sort_keys=True), inspect.getsource(user_message), inspect.getsource(parse),
             v.questions_sha256(REVISION), hashlib.sha256(v.SYSTEM_DESCRIPTION.encode("utf-8")).hexdigest()]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def load_probs(model: str, split: str) -> tuple[dict[str, dict[str, float]], int]:
    """Valid answers as probability dicts; an invalid window counts as a wrong 'insufficient_evidence' answer."""

    out, invalid = {}, 0
    for r in v.load_split(split)["rows"]:
        f = _path(model, split, r["id"])
        rec = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"valid": False}
        if rec.get("valid"):
            out[r["id"]] = rec["answer"]["probabilities"]
        else:
            invalid += 1
            out[r["id"]] = {c: float(c == "insufficient_evidence") for c in v.EVENT_TYPES}
    return out, invalid


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=list(COMPLETING))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=4, help="parallel requests per model")
    ap.add_argument("--max-inflight", type=int, default=10, help="starting concurrent requests across all models")
    ap.add_argument("--ceiling", type=int, default=90, help="upper bound for the adaptive concurrency")
    a = ap.parse_args(argv)
    global _inflight
    _inflight = AdaptiveLimit(a.max_inflight, a.ceiling)
    with ThreadPoolExecutor(len(a.models)) as ex:
        list(ex.map(lambda m: run_model(m, a.splits, a.limit, a.concurrency), a.models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

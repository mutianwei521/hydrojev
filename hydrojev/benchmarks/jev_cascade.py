"""Round 5 of the Jev pilot P2 v2: Jev as a fast, training-free first-line screen, reviewed by a rule or an LLM.

Jev (q3 question, state format 2, inductive dev_s2 prior calibration) screens every window. Two cascades are tested:

* rule cascade (no training, no LLM): Jev's verdict is kept when it is benign (normal_transient or sensor_fault);
  every other window is decided by the transparent rule R1;
* LLM cascade: Jev's verdict is kept when it is benign AND the rule R1 agrees; every other window (attack or asset
  verdicts, disagreements) is sent to a reviewer LLM, whose answer is calibrated with its own dev_s2 prior.

Both were designed on burned round-2 sets and checked on burned round-3 sets (PILOT_PROTOCOL_v2.md, amendment v2.6);
they are judged once on fresh sealed sets built from the round-4 seed ladders reserved in amendment v2.5 and never
generated before: E1c (v1 subtypes, in-distribution for R2) and E4c (the novel4 family, unseen by R2, by every design
step and by every earlier round). The frozen modules are not modified; the Jev call cap is lifted at run time because
the author authorised unlimited calls on 2026-09-26 ("调用上限可以无限").
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p1 as p1
from hydrojev.benchmarks import jev_pilot_p2v2 as v
from hydrojev.benchmarks import jev_pilot_r4 as r4  # registers the round-4 splits and the novel4 family
from hydrojev.benchmarks import llm_baselines as lb

REVISION = 3  # the Jev question and the LLM prompt are the round-2 q3 ones, unchanged
FMT = 2
CASCADE_FREEZE = "design_freeze_cascade.json"
SETS = {"E1c": "eval_id_r4", "E4c": "eval_novel4_r4"}
KEYS = {name: v.split_key(split, FMT) for name, split in SETS.items()}
BENIGN = ("normal_transient", "sensor_fault")
REVIEWERS = ("glm-5.2:cloud", "deepseek-v4-pro:cloud")  # primary, secondary
LATENCY_EVERY = 10  # windows with index % 10 == 0 (5 per class) are sent strictly one at a time, before any other window, to time them

for _s in SETS.values():
    v.FREEZE_FILE[_s] = CASCADE_FREEZE  # the round-4 q5 freeze was never written; the seeds are used by this round


def calibrated(pm: np.ndarray, prior: Sequence[float]) -> np.ndarray:
    q = np.maximum(pm, 0.005) / np.maximum(np.asarray(prior, dtype=float), 0.01)
    return q / q.sum(1, keepdims=True)


def decisions(pm: np.ndarray, prior: Sequence[float], valid: np.ndarray | None = None) -> np.ndarray:
    d = np.asarray([v.CLASSES[k] for k in calibrated(pm, prior).argmax(1)], dtype=object)
    if valid is not None:
        d[~np.asarray(valid, bool)] = "insufficient_evidence"  # a failed answer is a wrong answer
    return d


def rule_cascade(jev: np.ndarray, r1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(jev, BENIGN)
    return np.where(keep, jev, r1), keep


def llm_cascade(jev: np.ndarray, r1: np.ndarray, reviewer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isin(jev, BENIGN) & (jev == r1)
    return np.where(keep, jev, reviewer), keep


def latency_ids(split_key: str) -> list[str]:
    return [r["id"] for k, r in enumerate(v.load_split(split_key)["rows"]) if k % LATENCY_EVERY == 0]


def prepare() -> None:
    from hydrojev.config import load_config

    cfg = load_config()
    for split in SETS.values():
        v.prepare_split(cfg, split, FMT)


def run_jev(split_key: str) -> None:
    p1.PILOT_CALL_CAP = r4.UNLIMITED_CAP
    v.run(split_key, REVISION)


def run_llms(split_keys: Sequence[str], concurrency: int) -> None:
    """Timed subset first (one request at a time per model, both models in parallel), then the rest concurrently."""

    lb._inflight = lb.AdaptiveLimit(len(REVIEWERS), len(REVIEWERS))
    lock = threading.Lock()

    def timed(model: str) -> None:
        for key in split_keys:
            for rid in latency_ids(key):
                rec = lb.ask(model, key, rid)
                with lock:
                    print(f"{v._now()} TIMED {model} {key} {rid} valid={rec['valid']} latency={rec.get('latency_s')}", flush=True)

    with ThreadPoolExecutor(len(REVIEWERS)) as ex:
        list(ex.map(timed, REVIEWERS))
    lb._inflight = lb.AdaptiveLimit(concurrency, max(concurrency, 60))
    with ThreadPoolExecutor(len(REVIEWERS)) as ex:
        list(ex.map(lambda m: lb.run_model(m, split_keys, None, concurrency), REVIEWERS))


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("prepare", "jev", "llm"))
    ap.add_argument("--concurrency", type=int, default=20)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "jev":
        for key in KEYS.values():
            run_jev(key)
    else:
        run_llms(list(KEYS.values()), a.concurrency)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

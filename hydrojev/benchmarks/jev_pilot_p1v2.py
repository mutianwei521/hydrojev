"""Jev pilot P1 v2 runner (docs/jev/PILOT_PROTOCOL_v2.md).

Dev (dataset04) may be called and analysed freely: every design choice (question revision, threshold,
fusion weights) is fixed on dev. The test split (test_dataset) is called once per frozen design and its
comparator values are only read by ``analyse --split test`` after the Jev test run is complete.

Usage:
  python -m hydrojev.benchmarks.jev_pilot_p1v2 run --split dataset04 --round 2
  python -m hydrojev.benchmarks.jev_pilot_p1v2 analyse --split dataset04 --round 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrojev.benchmarks import jev_pilot_p1 as p1

OUT = p1.DEFAULT_OUTDIR


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _prepared(rnd: int) -> dict[str, Any]:
    return _json(OUT / ("prepared.json" if rnd == 1 else f"prepared_r{rnd}.json"))


def stage_name(split: str, rnd: int) -> str:
    return f"p1v2_{'dev' if split == 'dataset04' else 'test'}_r{rnd}"


def candidates(split: str, rnd: int) -> list[dict[str, Any]]:
    return [r for r in _prepared(rnd)["variants"][split]["grid"] if r["candidate"]]


def jev_scores(split: str, rnd: int) -> dict[int, float]:
    """Alarm probability per candidate; dev reuses the v1 debug responses of the same question revision."""

    out: dict[int, float] = {}
    stages = [stage_name(split, rnd)] + ([f"p1_debug_r{rnd}"] if split == "dataset04" else [])
    for st in stages:
        for f in (OUT / "responses" / st).glob("*.response.json"):
            i = int(f.stem.split("_")[-1].split(".")[0]) if not f.stem.endswith("_a") else None
            if i is None:
                continue
            out.setdefault(i, float(_json(f)["response"]["answers"]["alarm"]["noul"]))
    return out


def run(split: str, rnd: int) -> None:
    have = jev_scores(split, rnd)
    st = stage_name(split, rnd)
    for r in candidates(split, rnd):
        i = int(r["raw_index"])
        if i in have:
            continue
        e = p1.call_live(OUT, Path(r["request_path"]), stage=st, tag=f"{st}_{split}_{i:05d}")
        print(i, e.get("status"), e.get("alarm_probability"), flush=True)


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run",))
    ap.add_argument("--split", required=True, choices=("dataset04", "test_dataset"))
    ap.add_argument("--round", type=int, default=2)
    a = ap.parse_args(argv)
    run(a.split, a.round)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

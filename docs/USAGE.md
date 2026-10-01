# HydroJEV

A reproducible research codebase for the *Cognitive Reflex Paradigm* that
protects smart water distribution networks against **stealthy cyber-physical
attacks** and **hydraulic catastrophes**. It loads the available water-network
and BATADAL assets, implements the cited physical and adversarial mechanisms,
performs bounded TypeSafe Jev arbitration behind deterministic safety gates, and
reports every claim with explicit provenance.

The design principle throughout: the **reflex** (fast, deterministic, local
safety gates and air-gap tripping) handles bursts and neutralizes catastrophes
even on stale data, cloud-independent; the **cognition** (Jev cloud arbitration)
performs causal threat discrimination for stealthy attacks but is *advisory* and
**cannot bypass** the deterministic freshness / evidence / hard-hydraulic
interlocks. The whole loop runs fully offline against a transparent,
evidence-only mock arbiter — no API key, no network required.

---

## 1. Setup

Use the preconfigured conda environment (PyTorch, including any CUDA build, is
already installed there — **do not reinstall it**):

```bash
conda activate hydroJEV
```

Install the remaining pinned dependencies (PyTorch is intentionally *not* pinned
here; it is supplied by the environment):

```bash
pip install -r hydrojev/requirements.txt
```

Run the tests to confirm the environment:

```bash
python -m pytest hydrojev/tests -q
```

**Assets.** Immutable reference data is read in place from `refCode/` through the
configured paths (never copied into the package): `Net1.inp` (local EPANET),
`CTOWN.INP` + `BATADAL_dataset0{3,4}.csv` (BATADAL), and Net3 (supplied by the
WNTR library). WADI is a gated benchmark and is not present; it is reported
`not_run`, never fabricated.

---

## 2. Commands

All experiments run through one entry point:

```bash
python -m hydrojev.run_experiments <command> [--quick | --dry] [--output-dir DIR]
```

| Command | What it does |
| --- | --- |
| `inspect` | Report presence / size / SHA-256 of every local asset, and WADI's `not_run` reason. |
| `cluster --network Net1` | Load a network and partition it into Critical Physical Defence Zones (CPDZs). `--network` accepts `Net1`, `Net3`, `CTOWN`. |
| `train-ae` | Train the reconstruction detector on the **attack-free** BATADAL `dataset03`. |
| `attack-concealment` | White-box coordinate-descent concealment (k ≤ 4, discrete channels immutable) against the trained detector, on real `dataset04` samples. |
| `attack-fs-fdi` | Solve the bounded, localized false-data-injection MILP on Net1; reports the selected solver (Gurobi → HiGHS → PuLP). |
| `evaluate` | Full offline evaluation: synthetic closed-loop recall/energy/latency + real BATADAL detection recall; writes CSV + JSON tables. |
| `all` | `inspect` → build a reproducibility manifest → cluster → train → attacks → `evaluate` → write tables. |

**Modes.**

- **(default, "full")** — full training and scenario sweeps; the numbers you cite.
- `--quick` — bounded smoke: capped training rows/epochs, fewer scenarios, Net3
  reported topology-only. For CI and wiring checks, *not* for reported metrics.
- `--dry` — no heavy compute at all: only asset inspection and the manifest, so
  the pipeline wiring can be exercised without a GPU.

Examples:

```bash
python -m hydrojev.run_experiments inspect
```

```bash
python -m hydrojev.run_experiments all --output-dir artifacts/run_full
```

```bash
python -m hydrojev.run_experiments evaluate --quick --output-dir artifacts/smoke
```

---

## 3. Interpreting the results

`evaluate` and `all` write `results.csv` and `results.json`. **Every row** has
the same schema so a reader can see exactly what each number is *and is not*:

| Column | Meaning |
| --- | --- |
| `metric` | The measured quantity (e.g. `closed_loop_recall`). |
| `dataset` | **The honest provenance of the number.** `simulation` = synthetic orthogonal-residual scenarios through the mock arbiter. `BATADAL:...` = real benchmark data. `WADI` = gated benchmark. |
| `mode` | `full` / `quick` / `test` / `dry`. |
| `target` | The research target, or `null` for a pure diagnostic. |
| `observed` | The measured value, or `null` if `not_run`. |
| `unit` | `fraction`, `s`, or `ms`. |
| `status` | `pass` / `fail` (against a target), `measured` (diagnostic, no pass/fail), or `not_run` (missing/unreproduced — **never** a fabricated number). |
| `samples` | Sample count behind the number. |
| `provenance` | The exact mechanism/source, including the Jev source and compute device. |
| `detail` | Extra context (Wilson CI, point-wise recall/FPR, water-hammer caveat, …). |

### The two empirical claims are kept separate on purpose

- **Real benchmark detection (`dataset='BATADAL:dataset03->BATADAL:dataset04'`).**
  The reconstruction detector is trained and threshold-calibrated on the
  attack-free `dataset03` and evaluated on the labelled `dataset04` — a genuine
  cross-dataset anomaly-detection result on a named benchmark, with no label
  leakage. It reports **scenario recall** (was each labelled attack run detected
  at all) alongside point-wise recall and the benign false-positive rate.
- **Closed-loop cognitive-reflex behaviour (`dataset='simulation'`).** Recall,
  isolation delay, and energy mitigation are measured on synthetic defence-zone
  states that carry HydroJEV's orthogonal residual evidence, arbitrated by the
  **evidence-only** mock (`jev_source='mock'`). These measure the
  *discriminability of the residual features and the safety of the gating
  logic* — they are never presented as BATADAL/Net3 benchmark numbers.

The mock arbiter reasons **only from evidence features** — detector states,
hydraulic consistency, energy accounting, freshness — and a runtime guard
(`_assert_evidence_only`) rejects any state containing a ground-truth label. So
the offline recall is not a circular replay of hidden answers.

### Latency

Total loop latency is reported as **p50 / p95 / p99** (never a single request).
The p95 is compared to the 150 ms research target **and** to the minimum pipe
wave-propagation time `min(L/a)` on Net1. See the safety note below.

---

## 4. Safety limitations (read this before citing latency)

- **The cloud loop is advisory and is NOT the primary water-hammer relay.** A
  fast total-loop latency (well under 150 ms) does **not** constitute a
  water-hammer protection guarantee. Transient over-pressure propagates on the
  acoustic timescale `L/a` (tens of milliseconds on Net1); the *local reflex*,
  not cloud arbitration, is what trips `SAFE_FALLBACK` for bursts — and it does
  so even on stale data. Every latency row carries this caveat.
- **The cloud cannot cut remote-control authority on its own.** Isolation
  requires *all* deterministic gates to pass: data freshness, a minimum
  corroborating evidence count, a local hydraulic corroboration, attack
  persistence (debounce), and a minimum air-gap probability — plus optional
  operator confirmation. A confident cloud verdict on stale or weak evidence
  cannot isolate. (Proven directly in `tests/test_closed_loop_eval.py`.)
- **`quick` numbers are not results.** Only `full`-mode rows should be cited.
- **`not_run` is honest, not a failure.** WADI and any unreproduced benchmark
  stay `not_run` with an exact reason.

---

## 5. Providing the TypeSafe API key (without storing it)

The system defaults to the **offline mock arbiter** and needs no key. To make
one optional live Jev call, provide the key through the shell **only** — it is
read at runtime from the `TYPESAFE_API_KEY` environment variable and is **never**
written to config, code, fixtures, snapshots, or logs:

```bash
export TYPESAFE_API_KEY="…"   # process-only; do not commit or echo into a script
python -m hydrojev.run_experiments evaluate --output-dir artifacts/run_full
unset TYPESAFE_API_KEY
```

Only the endpoint and timeout/retry policy live in
`hydrojev/configs/default_config.yaml`; the key is environment-only, and the
config loader actively rejects any secret-like field. Live and mock decisions
are tagged distinctly (`jev_source`) in every audit record and result row, so
they can never be silently mixed.

---

## 6. Reproducibility manifest

`all` writes a manifest with, for every run: the mode and random seed, the Python
version and platform, resolved package versions, the selected Torch device (with
the reason CUDA was or was not used), the selected MILP solver, the Jev endpoint
and offline default source, and the SHA-256 hash + size of every input asset.
This makes each result table traceable to the exact code, data, and environment
that produced it.

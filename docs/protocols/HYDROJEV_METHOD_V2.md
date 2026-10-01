# HydroJEV detection method v2

This is a runnable detector, not a newly trained Jev model and not evidence that
Jev developed or improved the method. Its scientific value is determined by the
matched experiments. The current evaluation does not establish a detection gain.

## Executed method

The input is a causal window of water-network SCADA evidence. The front end
contains Mahalanobis, PCA residual, Isolation Forest, a reconstruction AE, and
CPDZ/GEpFM/RAT summaries derived from a GRU residual. Objects are fitted on the
attack-free record; development/test labels are never serialized into requests.

At each six-hour grid time, each detector alerts if any of its last six scored
observations reaches its frozen threshold. At least three stream alerts produce
a candidate. Non-candidates return normal without a Jev request. The streams
are correlated views of telemetry, not statistically independent observations.

For candidates, Jev supplies an alarm Noul, event-type Choice and severity Score.
The raw answer is validated before deterministic fusion. Stale or unusable
candidate evidence and contradictory answers produce abstention. Otherwise,
alarm probability >=0.50 produces alarm, <=0.20 produces normal, and intermediate
values produce abstention. An insufficient_evidence Choice expresses uncertain
cause and does not veto a high-probability anomaly alarm. The general detector
class has a configurable default of 0.80; the benchmark and CLI explicitly use
the evaluated 0.50 cutoff. These settings must not be silently interchanged.

No control commands are emitted. Hydraulic consistency is unavailable in the
formal BATADAL state; the implementation does not invent mass-balance data.

## Independently runnable inference

Run from the project root. Replaying a retained response makes no network call:

    python -m hydrojev.methods.detect --request artifacts/hydrojev_detection_v2_fresh_full/raw/dataset04/full_live/grid_00077.request.json --response artifacts/hydrojev_detection_v2_fresh_full/raw/dataset04/full_live/grid_00077.response.json --mode replay --output artifacts/hydrojev_cli_validation/replay_example.json

The same state can be evaluated without Jev by using --mode no-jev. A fresh
inference uses --mode live and a new output location; credentials are loaded by
the protected caller documented in JEV_CALLING_GUIDE.md. Replay and no-Jev
outputs are source-labelled and are not represented as fresh service calls.

## Full SCADA-to-result benchmark

    python -m hydrojev.benchmarks.hydrojev_detection_benchmark --live --outdir artifacts/hydrojev_detection_new_run

This trains the local front end, builds causal SCADA evidence, calls the service,
and records results. The --live option consumes real service requests. A new
run directory prevents accidental replacement of the retained experiment.
The no-live path records absent live inference as not_run; it never substitutes
mock responses. Saved-response evaluation must be identified as replay.

## Result sources for the reconstruction

The original result artifact is retained as a reference; fresh primary full
results are in artifacts/hydrojev_detection_v2_fresh_full, and real live
evidence ablations are in artifacts/hydrojev_detection_v2_live_ablations.
The paper-facing summary identifies source files and byte hashes, separates
the hourly reference from the matched six-hour baselines, and retains failures.

The temporal ablation removes GEpFM/RAT. The four-statistical-detector ablation
removes Mahalanobis/PCA/Isolation Forest/AE and retains CPDZ/GEpFM/RAT. Both
recompute the gate and therefore test the whole pipeline, not just semantic
inference with a fixed candidate set. A sensor-context ablation is not measured
because the formal state has no populated sensor_context field.

## Paper rebuild without new API calls

    python -m hydrojev.benchmarks.hydrojev_paper_summary
    python paper/WR_hydrojev_method_v2/build/make_figures_v2.py
    python paper/WR_hydrojev_method_v2/build/make_topology_v2.py
    python paper/WR_hydrojev_method_v2/build/render_paper.py

Edit source-bound prose in paper/WR_hydrojev_method_v2/build/templates. The
renderer resolves measured values from the summary; missing paths are errors,
and unavailable measurements are never converted to zero. Rendered Markdown
and DOCX are generated in paper/WR_hydrojev_method_v2. Source data, editable
figures and validation reports accompany them.

"""Benchmark harnesses that place HydroJEV's detector against standard baselines.

These modules produce the head-to-head, real-benchmark evidence a top-tier paper
needs: every detector is fit on the *same* attack-free training partition, given
the *same* threshold-calibration policy, and evaluated on the *same* labelled
benchmark with both threshold-independent (ROC-AUC, PR-AUC) and operating-point
(scenario/point recall, false-positive rate) metrics. Nothing here fabricates a
number: if a benchmark's data is absent the caller reports ``not_run``.
"""

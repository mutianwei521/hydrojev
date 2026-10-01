"""Bounded TypeSafe Jev arbitration for HydroJEV.

The arbiter turns a structured defence-zone state into three typed judgments:

* ``threat_cause`` (choice) -- the single most likely cause of the anomaly,
  drawn from the HydroJEV threat taxonomy;
* ``trip_edge_airgap`` (noul) -- the probability that the zone's edge controller
  should be physically air-gapped from SCADA now;
* ``severity_score`` (score) -- an ordered severity level.

Remote Jev is advisory: its judgments feed deterministic safety gates (Task 9)
and can never bypass freshness or hard-hydraulic interlocks.
"""

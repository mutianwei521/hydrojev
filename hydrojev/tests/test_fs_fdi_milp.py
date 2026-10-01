"""Tests for the localized FS-FDI MILP and its deterministic hydraulic math."""

from __future__ import annotations

import numpy as np
import pytest

from hydrojev.attacks.fs_fdi_milp import (
    FSFDIProblem,
    PipeSpec,
    SolveStatus,
    chi_square_statistic,
    hazen_williams_conductance,
    hazen_williams_pwl,
    select_solver,
    signed_hazen_williams_headloss,
    signed_incidence_matrix,
    solve_fs_fdi,
    vectorized_cusum,
    wls_state_estimate,
)


# --------------------------------------------------------------------------- #
# Hazen-Williams conductance and piecewise linearization
# --------------------------------------------------------------------------- #


def test_conductance_scales_linearly_with_length() -> None:
    single = hazen_williams_conductance(100.0, 0.3048, 100.0)
    double = hazen_williams_conductance(200.0, 0.3048, 100.0)
    assert double == pytest.approx(2.0 * single)
    assert single > 0.0


def test_conductance_rejects_nonphysical_geometry() -> None:
    with pytest.raises(ValueError):
        hazen_williams_conductance(100.0, 0.0, 100.0)
    with pytest.raises(ValueError):
        hazen_williams_conductance(100.0, 0.3, 0.0)


def test_signed_headloss_is_odd_and_matches_conductance_form() -> None:
    k = hazen_williams_conductance(120.0, 0.25, 130.0)
    forward = signed_hazen_williams_headloss(0.04, 120.0, 0.25, 130.0)
    backward = signed_hazen_williams_headloss(-0.04, 120.0, 0.25, 130.0)
    assert forward == pytest.approx(k * 0.04**1.852)
    assert backward == pytest.approx(-forward)
    assert signed_hazen_williams_headloss(0.0, 120.0, 0.25, 130.0) == 0.0


def test_pwl_is_symmetric_and_interpolates_curve_exactly_at_breakpoints() -> None:
    pwl = hazen_williams_pwl(30.48, 0.3048, 100.0, max_flow_m3_s=0.5, n_segments=8)

    assert pwl.n_segments == 8
    assert pwl.breakpoints.shape == (9,)
    # Breakpoints are antisymmetric and strictly increasing through zero.
    assert np.allclose(pwl.breakpoints, -pwl.breakpoints[::-1])
    assert pwl.breakpoints[4] == pytest.approx(0.0)
    assert np.all(np.diff(pwl.breakpoints) > 0)
    # Slopes mirror across zero; intercepts anti-mirror -> the curve stays odd.
    assert np.allclose(pwl.slopes, pwl.slopes[::-1])
    assert np.allclose(pwl.intercepts, -pwl.intercepts[::-1])
    # The PWL passes through the true Hazen-Williams curve at every breakpoint.
    for q in pwl.breakpoints:
        assert pwl.evaluate(float(q)) == pytest.approx(
            signed_hazen_williams_headloss(float(q), 30.48, 0.3048, 100.0), abs=1e-9
        )


def test_pwl_evaluation_is_odd_between_breakpoints() -> None:
    pwl = hazen_williams_pwl(50.0, 0.3, 120.0, max_flow_m3_s=0.4, n_segments=8)
    mid = 0.4 * 0.375  # between two positive breakpoints
    assert pwl.evaluate(mid) == pytest.approx(-pwl.evaluate(-mid), abs=1e-9)


def test_pwl_rejects_odd_segment_count() -> None:
    with pytest.raises(ValueError):
        hazen_williams_pwl(30.0, 0.3, 100.0, max_flow_m3_s=0.5, n_segments=7)


# --------------------------------------------------------------------------- #
# Incidence, WLS, and detector statistics
# --------------------------------------------------------------------------- #


def test_incidence_signs_leaving_positive_entering_negative() -> None:
    junctions = ("4", "5", "6")
    pipes = [("P34", "3", "4"), ("P45", "4", "5"), ("P56", "5", "6")]
    matrix = signed_incidence_matrix(junctions, pipes)

    # Node 4: P34 enters (-1), P45 leaves (+1); boundary node 3 has no row.
    assert matrix[0].tolist() == [-1.0, 1.0, 0.0]
    assert matrix[1].tolist() == [0.0, -1.0, 1.0]
    assert matrix[2].tolist() == [0.0, 0.0, -1.0]


def test_wls_recovers_exact_solution_without_matrix_inverse() -> None:
    x_true = np.array([2.0, -1.0])
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    z = H @ x_true
    weights = np.array([1.0, 4.0, 2.0, 0.5])  # weighting is irrelevant when consistent
    estimate = wls_state_estimate(H, z, weights)
    assert np.allclose(estimate, x_true)


def test_wls_rejects_nonpositive_weights_and_bad_shapes() -> None:
    H = np.eye(3)
    z = np.zeros(3)
    with pytest.raises(ValueError):
        wls_state_estimate(H, z, np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        wls_state_estimate(H, np.zeros(2), np.ones(3))


def test_vectorized_cusum_accumulates_and_resets() -> None:
    series = np.array([[1.0], [1.0], [1.0]])
    history = vectorized_cusum(series, bias=np.array([0.5]))
    assert history[:, 0].tolist() == pytest.approx([0.5, 1.0, 1.5])

    # A small residual under an initialized state floors the statistic at zero.
    reset = vectorized_cusum(
        np.array([[0.1]]), bias=np.array([0.5]), initial=np.array([0.3])
    )
    assert reset[0, 0] == pytest.approx(0.0)


def test_chi_square_uses_covariance_weighting() -> None:
    residual = np.array([3.0, 4.0])
    assert chi_square_statistic(residual, np.eye(2)) == pytest.approx(25.0)
    assert chi_square_statistic(residual, np.diag([1.0, 4.0])) == pytest.approx(13.0)


def test_chi_square_rejects_mismatched_covariance() -> None:
    with pytest.raises(ValueError):
        chi_square_statistic(np.zeros(2), np.eye(3))


# --------------------------------------------------------------------------- #
# Solver probing
# --------------------------------------------------------------------------- #


def test_select_solver_returns_a_usable_backend() -> None:
    solver, name, attempts = select_solver(("gurobi", "highs", "pulp"))
    assert solver is not None
    assert name in {"GUROBI", "HiGHS", "PULP_CBC_CMD"}


def test_select_solver_skips_unknown_names() -> None:
    solver, name, attempts = select_solver(("does_not_exist", "pulp"))
    assert solver is not None
    assert name == "PULP_CBC_CMD"
    assert any("does_not_exist:unknown" in note for note in attempts)


# --------------------------------------------------------------------------- #
# End-to-end MILP: tiny network and a bounded Net1 zone
# --------------------------------------------------------------------------- #


def _single_junction_problem() -> FSFDIProblem:
    return FSFDIProblem(
        junctions=("J1",),
        pipes=(
            PipeSpec(
                name="P1",
                start="R",
                end="J1",
                length_m=30.48,
                diameter_m=0.3048,
                roughness_c=100.0,
                base_flow_m3_s=0.10,
                max_flow_m3_s=0.5,
            ),
        ),
        boundary_heads_m={"R": 213.36},
        base_head_m={"J1": 213.0},
        base_demand_m3_s={"J1": 0.10},
        measured_head_junctions=(),
        impact_pipes=("P1",),
        max_changed_measurements=4,
    )


def test_tiny_network_solution_is_verified_and_sparse() -> None:
    result = solve_fs_fdi(_single_junction_problem(), solver_order=("gurobi", "highs", "pulp"))

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert result.verification["accepted"] is True
    assert result.verification["mass_balance_max_residual"] < 1e-3
    assert 1 <= len(result.changed_measurements) <= 4
    # A stealthy attack that pushes demand to its budget moves the trunk flow.
    assert result.objective is not None and result.objective > 0.0
    assert abs(result.measurement_attack["flow_P1"]) > 0.0
    # Real impact guard: the perceived state must actually move, not fake an
    # absolute-value auxiliary. Demand rises toward its 20% budget.
    assert result.attacked_demands_m3_s["J1"] > 0.10 + 1e-4
    assert result.attacked_flows_m3_s["P1"] == pytest.approx(
        result.attacked_demands_m3_s["J1"], abs=1e-6
    )


def _net1_radial_zone() -> FSFDIProblem:
    """A bounded, radial cut of Net1 with real pipe geometry in SI units.

    Net1 pipes are 100 ft x 12 in, Hazen-Williams C=100. Flows and demands are
    a self-consistent radial supply from a boundary head standing in for the
    pump-fed trunk, so the zero-attack state is feasible by construction.
    """

    length_m = 100.0 * 0.3048
    diameter_m = 12.0 * 0.0254
    # Linearize over the operating envelope (~1.4x the trunk flow), not a wide
    # range, so the 8-segment PWL stays close to the true nonlinear curve.
    common = dict(length_m=length_m, diameter_m=diameter_m, roughness_c=100.0, max_flow_m3_s=0.2)
    return FSFDIProblem(
        junctions=("4", "5", "6"),
        pipes=(
            PipeSpec(name="P34", start="3", end="4", base_flow_m3_s=0.12, **common),
            PipeSpec(name="P45", start="4", end="5", base_flow_m3_s=0.07, **common),
            PipeSpec(name="P56", start="5", end="6", base_flow_m3_s=0.04, **common),
        ),
        boundary_heads_m={"3": 213.36},
        base_head_m={"4": 213.0, "5": 212.7, "6": 212.4},
        base_demand_m3_s={"4": 0.05, "5": 0.03, "6": 0.04},
        measured_head_junctions=(),
        impact_pipes=("P34",),
        max_changed_measurements=6,
        n_segments=8,
    )


def test_net1_zone_smoke_reports_provenance_and_stays_stealthy() -> None:
    result = solve_fs_fdi(_net1_radial_zone(), solver_order=("gurobi", "highs", "pulp"))

    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert result.verification["accepted"] is True
    assert result.verification["stealth_ok"] is True
    assert result.verification["changed_within_budget"] is True
    # Provenance records which backend ran and which relaxations were used.
    assert result.solver in {"GUROBI", "HiGHS", "PULP_CBC_CMD"}
    assert "energy" in result.provenance["relaxations"]
    assert result.provenance["n_segments"] == 8
    # The linearized headloss stays close to the true nonlinear curve.
    assert result.verification["energy_true_hw_max_error"] < 0.05
    # Real impact guard: overstating demand genuinely inflates the trunk flow.
    assert result.attacked_flows_m3_s["P34"] > 0.12 + 1e-4

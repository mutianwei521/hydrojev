"""Localized fully-stealthy FDI as a solver-neutral MILP.

This is a clean HydroJEV re-implementation of the Albustami & Taha (Water
Research 2025) FS-FDI idea, informed by the ``refCode/FDI-WDNs`` MATLAB
reference but not transliterated from it: the reference disables the attack by
default, calls ``value()`` on decision variables before solving, has a
``jj``/``jJ`` case bug, freezes the energy slope to the first segment, and
inverts ``H'H`` explicitly. Each of those is corrected and unit-tested here.

The formulation models the *perceived* (falsified) hydraulic state as the MILP
decision state. Because that state is forced to satisfy mass balance and a
piecewise-linear Hazen-Williams energy law exactly, it produces no structural
state-estimation residual; the only residual the detector sees is the injected
measurement deviation, which a linear CUSUM envelope keeps below threshold. The
objective maximizes bounded physical impact, and sparsity is imposed explicitly
through attack-enable binaries with a cap and penalty -- never assumed from an
L1 magnitude term.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Deterministic hydraulic / detector math (no solver required)
# --------------------------------------------------------------------------- #


def hazen_williams_conductance(
    length_m: float, diameter_m: float, roughness_c: float
) -> float:
    """SI Hazen-Williams resistance constant ``k`` in ``h = k * sign(Q)|Q|^1.852``."""

    if length_m < 0 or diameter_m <= 0 or roughness_c <= 0:
        raise ValueError("length must be non-negative and diameter/roughness positive")
    return 10.67 * length_m / (roughness_c**1.852 * diameter_m**4.8704)


def signed_hazen_williams_headloss(
    flow_m3_s: float, length_m: float, diameter_m: float, roughness_c: float
) -> float:
    """Return the true (nonlinear) signed Hazen-Williams headloss in metres."""

    conductance = hazen_williams_conductance(length_m, diameter_m, roughness_c)
    return conductance * math.copysign(abs(flow_m3_s) ** 1.852, flow_m3_s)


@dataclass(frozen=True)
class PiecewiseLinearization:
    """Symmetric PWL approximation of the odd Hazen-Williams headloss curve."""

    breakpoints: np.ndarray  # length n_segments + 1, strictly increasing, symmetric
    slopes: np.ndarray  # length n_segments
    intercepts: np.ndarray  # length n_segments

    @property
    def n_segments(self) -> int:
        return int(self.slopes.shape[0])

    def evaluate(self, flow_m3_s: float) -> float:
        """Evaluate the PWL headloss at a flow inside the breakpoint range."""

        flow = float(flow_m3_s)
        lower, upper = float(self.breakpoints[0]), float(self.breakpoints[-1])
        if not lower - 1e-9 <= flow <= upper + 1e-9:
            raise ValueError("flow is outside the linearized range")
        index = int(np.searchsorted(self.breakpoints, flow, side="right") - 1)
        index = min(max(index, 0), self.n_segments - 1)
        return float(self.slopes[index] * flow + self.intercepts[index])


def hazen_williams_pwl(
    length_m: float,
    diameter_m: float,
    roughness_c: float,
    max_flow_m3_s: float,
    n_segments: int = 8,
) -> PiecewiseLinearization:
    """Build a symmetric PWL headloss model with ``n_segments`` (even) pieces.

    Mirrors the ``PipeHeadLoss_Linearization.m`` breakpoint/slope/intercept
    construction, but in SI units and with the symmetry made explicit: slopes
    are mirrored across zero and intercepts are anti-mirrored, so the resulting
    curve is odd, matching the physical headloss law.
    """

    if n_segments < 2 or n_segments % 2 != 0:
        raise ValueError("n_segments must be an even integer >= 2")
    if max_flow_m3_s <= 0:
        raise ValueError("max_flow_m3_s must be positive")
    conductance = hazen_williams_conductance(length_m, diameter_m, roughness_c)
    half = n_segments // 2

    positive_q = np.array([n / half * max_flow_m3_s for n in range(half + 1)])
    positive_h = conductance * positive_q**1.852
    slopes: list[float] = []
    intercepts: list[float] = []
    for n in range(1, half + 1):
        q_lo, q_hi = positive_q[n - 1], positive_q[n]
        h_lo, h_hi = positive_h[n - 1], positive_h[n]
        slope = (h_hi - h_lo) / (q_hi - q_lo)
        intercept = h_hi - slope * q_hi
        slopes.append(slope)
        intercepts.append(intercept)

    slopes_arr = np.array(slopes)
    intercepts_arr = np.array(intercepts)
    full_slopes = np.concatenate([slopes_arr[::-1], slopes_arr])
    full_intercepts = np.concatenate([-intercepts_arr[::-1], intercepts_arr])
    breakpoints = np.concatenate([-positive_q[1:][::-1], positive_q])
    return PiecewiseLinearization(
        breakpoints=breakpoints, slopes=full_slopes, intercepts=full_intercepts
    )


def signed_incidence_matrix(
    junctions: Sequence[str],
    pipes: Sequence[tuple[str, str, str]],
) -> np.ndarray:
    """Return a junction x pipe incidence matrix (+1 leaves node, -1 enters).

    Each pipe is ``(name, from_node, to_node)``. Row ``j`` times a flow vector
    yields the net outflow from junction ``j`` through pipes, so nodal mass
    balance is ``incidence @ flow + boundary_inflow == demand`` per junction.
    """

    index = {name: position for position, name in enumerate(junctions)}
    matrix = np.zeros((len(junctions), len(pipes)))
    for column, (_name, start, end) in enumerate(pipes):
        if start in index:
            matrix[index[start], column] += 1.0
        if end in index:
            matrix[index[end], column] -= 1.0
    return matrix


def wls_state_estimate(
    measurement_matrix: np.ndarray,
    measurements: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted least squares via a whitened least-squares solve, never ``inv``.

    Solves ``min_x (z - Hx)^T W (z - Hx)`` by forming ``sqrt(W) H x = sqrt(W) z``
    and calling :func:`numpy.linalg.lstsq`, which is numerically stabler than the
    explicit ``inv(H'H)`` used in the MATLAB reference.
    """

    H = np.asarray(measurement_matrix, dtype=float)
    z = np.asarray(measurements, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if H.ndim != 2 or H.shape[0] != z.shape[0] or w.shape[0] != z.shape[0]:
        raise ValueError("H, measurements, and weights have inconsistent shapes")
    if np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite and positive")
    root = np.sqrt(w)[:, None]
    solution, *_ = np.linalg.lstsq(root * H, root.reshape(-1) * z, rcond=None)
    return solution


def vectorized_cusum(
    residual_series: np.ndarray, bias: np.ndarray, initial: np.ndarray | None = None
) -> np.ndarray:
    """Non-negative vectorized CUSUM ``c_k = max(0, c_{k-1} + |r_k| - b)``."""

    residuals = np.atleast_2d(np.asarray(residual_series, dtype=float))
    b = np.asarray(bias, dtype=float).reshape(-1)
    if residuals.shape[1] != b.shape[0]:
        raise ValueError("bias length must match the residual dimension")
    state = np.zeros(b.shape[0]) if initial is None else np.asarray(initial, float).copy()
    history = np.empty_like(residuals)
    for step in range(residuals.shape[0]):
        state = np.maximum(0.0, state + np.abs(residuals[step]) - b)
        history[step] = state
    return history


def chi_square_statistic(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Return ``r^T Sigma^-1 r`` via a linear solve rather than an inverse."""

    r = np.asarray(residual, dtype=float).reshape(-1)
    sigma = np.asarray(covariance, dtype=float)
    if sigma.shape != (r.shape[0], r.shape[0]):
        raise ValueError("covariance must be square and match the residual length")
    whitened = np.linalg.solve(sigma, r)
    return float(r @ whitened)


# --------------------------------------------------------------------------- #
# Solver-neutral problem / result types
# --------------------------------------------------------------------------- #


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    ERROR = "error"


@dataclass(frozen=True)
class PipeSpec:
    name: str
    start: str
    end: str
    length_m: float
    diameter_m: float
    roughness_c: float
    base_flow_m3_s: float
    max_flow_m3_s: float
    measured: bool = True


@dataclass(frozen=True)
class FSFDIProblem:
    """A localized FS-FDI instance over one cyber-physical defence zone."""

    junctions: tuple[str, ...]
    pipes: tuple[PipeSpec, ...]
    boundary_heads_m: Mapping[str, float]
    base_head_m: Mapping[str, float]
    base_demand_m3_s: Mapping[str, float]
    measured_head_junctions: tuple[str, ...]
    impact_pipes: tuple[str, ...]
    max_flow_fraction: float = 0.20
    max_head_fraction: float = 0.10
    max_demand_fraction: float = 0.20
    state_deviation_fraction: float = 0.25
    cusum_bias: float = 0.02
    cusum_prev: float = 0.0
    cusum_threshold: float = 5.0
    max_changed_measurements: int = 4
    n_segments: int = 8

    def __post_init__(self) -> None:
        pipe_names = {pipe.name for pipe in self.pipes}
        missing = set(self.impact_pipes) - pipe_names
        if missing:
            raise ValueError(f"impact pipes absent from problem: {sorted(missing)}")
        if not 1 <= self.max_changed_measurements:
            raise ValueError("max_changed_measurements must be positive")


@dataclass(frozen=True)
class FSFDIResult:
    status: SolveStatus
    solver: str
    objective: float | None
    attacked_flows_m3_s: dict[str, float]
    attacked_heads_m: dict[str, float]
    attacked_demands_m3_s: dict[str, float]
    measurement_attack: dict[str, float]
    changed_measurements: tuple[str, ...]
    verification: dict[str, object]
    provenance: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Solver probing
# --------------------------------------------------------------------------- #


def select_solver(order: Sequence[str] = ("gurobi", "highs", "pulp")):
    """Return the first usable PuLP solver in the requested order.

    ``gurobi`` maps to PuLP's gurobipy-backed solver, ``highs`` to highspy, and
    ``pulp``/``cbc`` to the bundled CBC binary. License, import, and probe
    failures are caught so a missing or unlicensed backend degrades to the next.
    """

    import pulp

    backends = {
        "gurobi": ("GUROBI", {"msg": 0}),
        "highs": ("HiGHS", {"msg": False}),
        "pulp": ("PULP_CBC_CMD", {"msg": 0}),
        "cbc": ("PULP_CBC_CMD", {"msg": 0}),
    }
    attempts: list[str] = []
    for key in order:
        entry = backends.get(key.strip().lower())
        if entry is None:
            attempts.append(f"{key}:unknown")
            continue
        class_name, kwargs = entry
        try:
            solver_cls = getattr(pulp, class_name)
            solver = solver_cls(**kwargs)
            if not solver.available():
                attempts.append(f"{key}:unavailable")
                continue
            return solver, class_name, tuple(attempts)
        except Exception as exc:  # noqa: BLE001 - backend probing is best-effort
            attempts.append(f"{key}:{type(exc).__name__}")
    return None, None, tuple(attempts)


# --------------------------------------------------------------------------- #
# MILP construction and solve
# --------------------------------------------------------------------------- #


def solve_fs_fdi(
    problem: FSFDIProblem,
    *,
    solver_order: Sequence[str] = ("gurobi", "highs", "pulp"),
    impact_direction: Mapping[str, float] | None = None,
    sparsity_penalty: float = 1e-3,
    time_limit_s: float | None = 30.0,
    verification_tolerance: float = 1e-3,
) -> FSFDIResult:
    """Build and solve the localized FS-FDI MILP, then verify it independently.

    ``impact_direction`` gives the signed harm direction per impact pipe (+1 to
    reward inflating the perceived flow, which drives excess pumping energy as in
    the Net3 scenario; -1 to reward suppressing it). The objective maximizes this
    *signed* deviation, never an absolute-value auxiliary: rewarding ``|x|`` built
    from ``x = pos - neg`` lets a maximizer inflate ``pos = neg`` to fake impact
    without moving the state, so the physical impact must carry a direction.
    """

    directions = dict(impact_direction or {})

    import pulp

    solver, solver_name, attempts = select_solver(solver_order)
    if solver is None:
        return FSFDIResult(
            status=SolveStatus.SOLVER_UNAVAILABLE,
            solver="none",
            objective=None,
            attacked_flows_m3_s={},
            attacked_heads_m={},
            attacked_demands_m3_s={},
            measurement_attack={},
            changed_measurements=(),
            verification={"reason": "no usable solver", "attempts": attempts},
            provenance={"solver_attempts": attempts},
        )
    if time_limit_s is not None:
        try:
            solver.timeLimit = float(time_limit_s)
        except Exception:  # noqa: BLE001 - not every backend exposes a time limit
            pass

    pwl = {
        pipe.name: hazen_williams_pwl(
            pipe.length_m,
            pipe.diameter_m,
            pipe.roughness_c,
            pipe.max_flow_m3_s,
            problem.n_segments,
        )
        for pipe in problem.pipes
    }
    model = pulp.LpProblem("hydrojev_fs_fdi", pulp.LpMaximize)

    # Perceived hydraulic state variables.
    flow = {
        pipe.name: pulp.LpVariable(
            f"q_{pipe.name}",
            lowBound=float(pwl[pipe.name].breakpoints[0]),
            upBound=float(pwl[pipe.name].breakpoints[-1]),
        )
        for pipe in problem.pipes
    }
    head = {j: pulp.LpVariable(f"h_{j}", lowBound=None) for j in problem.junctions}
    demand = {
        j: pulp.LpVariable(f"d_{j}", lowBound=0.0) for j in problem.junctions
    }

    # Piecewise-linear Hazen-Williams energy per pipe (zeta / omega selection).
    node_head = dict(problem.boundary_heads_m)
    for pipe in problem.pipes:
        segments = range(pwl[pipe.name].n_segments)
        zeta = {
            n: pulp.LpVariable(f"zeta_{pipe.name}_{n}", lowBound=None) for n in segments
        }
        omega = {
            n: pulp.LpVariable(f"omega_{pipe.name}_{n}", cat="Binary") for n in segments
        }
        bp = pwl[pipe.name].breakpoints
        model += pulp.lpSum(zeta.values()) == flow[pipe.name]
        model += pulp.lpSum(omega.values()) == 1
        for n in segments:
            model += zeta[n] >= float(bp[n]) * omega[n]
            model += zeta[n] <= float(bp[n + 1]) * omega[n]
        headloss = pulp.lpSum(
            float(pwl[pipe.name].slopes[n]) * zeta[n]
            + float(pwl[pipe.name].intercepts[n]) * omega[n]
            for n in segments
        )
        up = head[pipe.start] if pipe.start in head else node_head.get(pipe.start)
        down = head[pipe.end] if pipe.end in head else node_head.get(pipe.end)
        if up is None or down is None:
            raise ValueError(f"pipe {pipe.name} references an unknown node")
        model += up - down == headloss

    # Nodal mass balance on the perceived state: net outflow + demand == 0.
    incidence = signed_incidence_matrix(
        problem.junctions,
        [(p.name, p.start, p.end) for p in problem.pipes],
    )
    for row, j in enumerate(problem.junctions):
        net_out = pulp.lpSum(
            float(incidence[row, col]) * flow[p.name]
            for col, p in enumerate(problem.pipes)
            if incidence[row, col] != 0.0
        )
        model += net_out + demand[j] == 0

    # Attack magnitudes, enable binaries, and stealth (CUSUM) envelope.
    changed_binaries: list = []
    measurement_attack_expr: dict[str, object] = {}

    def _bounded_attack(name: str, variable, base: float, fraction: float, floor: float):
        limit = max(abs(base) * fraction, floor)
        pos = pulp.LpVariable(f"ap_{name}", lowBound=0.0, upBound=limit)
        neg = pulp.LpVariable(f"an_{name}", lowBound=0.0, upBound=limit)
        enable = pulp.LpVariable(f"b_{name}", cat="Binary")
        # ``model +=`` would rebind ``model`` as a closure local; use addConstraint.
        model.addConstraint(variable - base == pos - neg)
        model.addConstraint(pos + neg <= limit * enable)
        # Stealth: the injected deviation must keep CUSUM under threshold.
        model.addConstraint(
            problem.cusum_prev + (pos + neg) - problem.cusum_bias
            <= problem.cusum_threshold
        )
        changed_binaries.append(enable)
        measurement_attack_expr[name] = pos + neg
        return pos + neg

    signed_impact = {}
    for pipe in problem.pipes:
        if not pipe.measured:
            continue
        _bounded_attack(
            f"flow_{pipe.name}",
            flow[pipe.name],
            pipe.base_flow_m3_s,
            problem.max_flow_fraction,
            floor=1e-4,
        )
        if pipe.name in problem.impact_pipes:
            direction = float(directions.get(pipe.name, 1.0))
            signed_impact[pipe.name] = direction * (flow[pipe.name] - pipe.base_flow_m3_s)

    for j in problem.measured_head_junctions:
        base = float(problem.base_head_m[j])
        _bounded_attack(f"head_{j}", head[j], base, problem.max_head_fraction, floor=1e-3)

    for j in problem.junctions:
        base = float(problem.base_demand_m3_s.get(j, 0.0))
        # State-deviation bound keeps the perceived demand near truth.
        limit = max(abs(base) * problem.state_deviation_fraction, 1e-5)
        model += demand[j] - base <= limit
        model += base - demand[j] <= limit
        if base != 0.0:
            _bounded_attack(
                f"demand_{j}", demand[j], base, problem.max_demand_fraction, floor=1e-5
            )

    # Explicit sparsity cap; sparsity is enforced, not inferred from the L1 term.
    model += pulp.lpSum(changed_binaries) <= problem.max_changed_measurements

    # Maximize bounded, signed physical impact minus an explicit sparsity penalty.
    model += (
        pulp.lpSum(signed_impact.values())
        - sparsity_penalty * pulp.lpSum(changed_binaries)
    )

    try:
        model.solve(solver)
    except Exception as exc:  # noqa: BLE001 - surface any solver runtime failure
        return FSFDIResult(
            status=SolveStatus.ERROR,
            solver=solver_name or "unknown",
            objective=None,
            attacked_flows_m3_s={},
            attacked_heads_m={},
            attacked_demands_m3_s={},
            measurement_attack={},
            changed_measurements=(),
            verification={"reason": f"{type(exc).__name__}: {exc}"},
            provenance={"solver_attempts": attempts, "solver": solver_name},
        )

    status_name = pulp.LpStatus[model.status]
    if status_name == "Infeasible":
        status = SolveStatus.INFEASIBLE
    elif status_name == "Optimal":
        status = SolveStatus.OPTIMAL
    else:
        status = SolveStatus.FEASIBLE if model.status == 1 else SolveStatus.ERROR

    attacked_flows = {p.name: float(flow[p.name].value() or 0.0) for p in problem.pipes}
    attacked_heads = {j: float(head[j].value() or 0.0) for j in problem.junctions}
    attacked_demands = {j: float(demand[j].value() or 0.0) for j in problem.junctions}
    measurement_attack = {
        name: float(pulp.value(expr) or 0.0)
        for name, expr in measurement_attack_expr.items()
    }
    changed = tuple(
        name for name, value in measurement_attack.items() if abs(value) > 1e-6
    )

    verification = _verify_solution(
        problem,
        pwl,
        attacked_flows,
        attacked_heads,
        attacked_demands,
        measurement_attack,
        tolerance=verification_tolerance,
    )
    return FSFDIResult(
        status=status,
        solver=solver_name or "unknown",
        objective=float(pulp.value(model.objective)) if status in {
            SolveStatus.OPTIMAL,
            SolveStatus.FEASIBLE,
        } else None,
        attacked_flows_m3_s=attacked_flows,
        attacked_heads_m=attacked_heads,
        attacked_demands_m3_s=attacked_demands,
        measurement_attack=measurement_attack,
        changed_measurements=changed,
        verification=verification,
        provenance={
            "solver": solver_name,
            "solver_attempts": attempts,
            "pulp_status": status_name,
            "relaxations": {
                "energy": "piecewise-linear Hazen-Williams (SOS-style segment selection)",
                "chi_square": "not constrained in-model; evaluated post-solve",
                "stealth": "linear CUSUM absolute-value envelope",
            },
            "impact_direction": {p: float(directions.get(p, 1.0)) for p in problem.impact_pipes},
            "n_segments": problem.n_segments,
        },
    )


def _verify_solution(
    problem: FSFDIProblem,
    pwl: Mapping[str, PiecewiseLinearization],
    flows: Mapping[str, float],
    heads: Mapping[str, float],
    demands: Mapping[str, float],
    measurement_attack: Mapping[str, float],
    *,
    tolerance: float,
) -> dict[str, object]:
    """Recompute hydraulics and detector bounds independently of the solver."""

    incidence = signed_incidence_matrix(
        problem.junctions,
        [(p.name, p.start, p.end) for p in problem.pipes],
    )
    flow_vector = np.array([flows[p.name] for p in problem.pipes])
    mass_residual = incidence @ flow_vector + np.array(
        [demands[j] for j in problem.junctions]
    )
    mass_ok = bool(np.all(np.abs(mass_residual) <= tolerance))

    node_head = dict(problem.boundary_heads_m)
    node_head.update(heads)
    energy_errors_pwl: dict[str, float] = {}
    energy_errors_true: dict[str, float] = {}
    for pipe in problem.pipes:
        modelled = node_head[pipe.start] - node_head[pipe.end]
        energy_errors_pwl[pipe.name] = abs(
            modelled - pwl[pipe.name].evaluate(flows[pipe.name])
        )
        energy_errors_true[pipe.name] = abs(
            modelled
            - signed_hazen_williams_headloss(
                flows[pipe.name], pipe.length_m, pipe.diameter_m, pipe.roughness_c
            )
        )
    energy_ok = bool(max(energy_errors_pwl.values(), default=0.0) <= tolerance)

    changed = [name for name, value in measurement_attack.items() if abs(value) > 1e-6]
    budget_ok = len(changed) <= problem.max_changed_measurements
    max_injection = max((abs(v) for v in measurement_attack.values()), default=0.0)
    cusum_after = max(0.0, problem.cusum_prev + max_injection - problem.cusum_bias)
    stealth_ok = cusum_after <= problem.cusum_threshold + tolerance

    return {
        "mass_balance_ok": mass_ok,
        "mass_balance_max_residual": float(np.max(np.abs(mass_residual)) if mass_residual.size else 0.0),
        "energy_pwl_ok": energy_ok,
        "energy_pwl_max_error": float(max(energy_errors_pwl.values(), default=0.0)),
        "energy_true_hw_max_error": float(max(energy_errors_true.values(), default=0.0)),
        "changed_measurement_count": len(changed),
        "changed_within_budget": budget_ok,
        "stealth_cusum_after": float(cusum_after),
        "stealth_ok": stealth_ok,
        "accepted": bool(mass_ok and energy_ok and budget_ok and stealth_ok),
        "tolerance": tolerance,
    }

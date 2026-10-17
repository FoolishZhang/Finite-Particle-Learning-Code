"""Bounded physical-coordinate projected optimization, without random draws.

The caller owns the objective and its frozen integration rule.  The gradient
must differentiate that same scalar; a Price expectation score is not accepted
as a substitute for the derivative of a fixed-bank SAA.
"""
from __future__ import annotations

import time
import numpy as np
import signal_proposal_core as legacy
from informative_projection import project_coordinates

SCHEMA = "active_reference_projected_solver_v1"


def gradient_mapping(coordinates, gradient, *, floor=.15, step=1.):
    c, g = np.asarray(coordinates, float), np.asarray(gradient, float)
    if c.shape != (3,) or g.shape != (3,) or not np.isfinite([c, g]).all() or step <= 0:
        raise ValueError("finite m3 coordinates, gradient and positive step required")
    p = project_coordinates(c-step*g, delta=floor)
    return {"mapping": (c-p["coordinates"])/step,
            "norm_inf": float(np.max(np.abs((c-p["coordinates"])/step))),
            "projection_residual": float(p["residual"])}


def solve(objective, initial, *, floor=.15, initial_step=32.,
          mapping_tolerance=1e-5, max_iterations=100, max_evaluations=240,
          max_backtracks=16, armijo=1e-4, step_bounds=(.25, 256.),
          deadline=None):
    """Monotone spectral projected gradient with a bounded Armijo search.

    objective(c) -> {'value': scalar, 'gradient': float[3]}.
    Return the last accepted point, never a risk-selected output. Numerical
    convergence is for this particular frozen scalar only. Every attempted
    function evaluation and every accepted state is recorded.
    """
    c = np.asarray(initial, float).copy()
    if c.shape != (3,) or np.linalg.eigvalsh(legacy.correlation_matrix(c))[0] < floor-1e-12:
        raise ValueError("initial coordinates must satisfy the same spectral floor")
    if not (0 < floor < 1 and 0 < initial_step and 0 < mapping_tolerance and
            max_iterations >= 1 and max_evaluations >= 1 and max_backtracks >= 1 and
            0 < armijo < 1 and 0 < step_bounds[0] <= step_bounds[1]):
        raise ValueError("invalid explicit optimizer limits")
    evaluations, trace = [], []
    started = time.perf_counter()

    def evaluate(x):
        if deadline is not None and time.perf_counter() >= deadline:
            raise RuntimeError("frozen phase wall cap reached before objective call")
        if len(evaluations) >= max_evaluations:
            return None
        result = objective(x)
        f, g = float(result["value"]), np.asarray(result["gradient"], float)
        if not np.isfinite(f) or g.shape != (3,) or not np.isfinite(g).all():
            raise FloatingPointError("nonfinite objective/gradient; no clipping")
        evaluations.append({"coordinates": x.tolist(), "value": f, "gradient": g.tolist()})
        return f, g

    f, g = evaluate(c)
    alpha = float(initial_step)
    status = "max_iterations"
    for iteration in range(max_iterations+1):
        mapping = gradient_mapping(c, g, floor=floor)
        trace.append({"iteration": iteration, "coordinates": c.tolist(), "value": f,
                      "gradient": g.tolist(), "mapping_norm_inf": mapping["norm_inf"],
                      "min_eigenvalue": float(np.linalg.eigvalsh(legacy.correlation_matrix(c))[0]),
                      "evaluation_count": len(evaluations)})
        if mapping["norm_inf"] <= mapping_tolerance:
            status = "mapping_tolerance"
            break
        if iteration == max_iterations:
            break
        accepted = False
        for bt in range(max_backtracks):
            projection = project_coordinates(c-alpha*g, delta=floor)
            trial = np.asarray(projection["coordinates"])
            displacement = trial-c
            slope = float(g @ displacement)
            candidate = evaluate(trial)
            if candidate is None:
                status = "evaluation_cap"
                break
            trial_f, trial_g = candidate
            evaluations[-1].update({"backtrack": bt, "step": alpha,
                                    "projected_slope": slope,
                                    "projection_residual": float(projection["residual"])})
            if trial_f <= f+armijo*slope+1e-13:
                accepted = True
                y = trial_g-g
                sy = float(displacement @ y)
                next_step = float(displacement @ displacement)/sy if sy > 1e-18 else initial_step
                c, f, g = trial.copy(), trial_f, trial_g
                alpha = float(np.clip(next_step, *step_bounds))
                evaluations[-1]["accepted"] = True
                break
            evaluations[-1]["accepted"] = False
            alpha *= .5
        if not accepted:
            if status != "evaluation_cap":
                status = "line_search_limit"
            break
    # A failed final line search/cap leaves the already recorded last point.
    final_mapping = gradient_mapping(c, g, floor=floor)
    return {"schema": SCHEMA, "status": status, "coordinates": c,
            "value": f, "gradient": g, "mapping_norm_inf": final_mapping["norm_inf"],
            "iterations": len(trace)-1, "objective_calls": len(evaluations),
            "trace": trace, "evaluations": evaluations,
            "seconds": time.perf_counter()-started,
            "output_rule": "last_accepted_iterate_no_risk_selection",
            "qualification": "stationarity_of_frozen_numerical_scalar_not_global_optimum_or_true_FK_stationarity"}


def selftest():
    target = np.array([.3, -.2, .1])
    def quadratic(c):
        d = c-target
        return {"value": float(.5*d@d), "gradient": d}
    interior = solve(quadratic, np.zeros(3), mapping_tolerance=1e-9)
    assert np.max(abs(interior["coordinates"]-target)) < 1e-8
    outside = np.array([1.1, 1.1, 1.1])
    boundary = solve(lambda c: {"value": float(.5*np.sum((c-outside)**2)), "gradient": c-outside},
                     np.zeros(3), mapping_tolerance=1e-8)
    assert boundary["mapping_norm_inf"] <= 1e-8
    assert np.linalg.eigvalsh(legacy.correlation_matrix(boundary["coordinates"]))[0] >= .15-1e-10
    assert all(b["value"] <= a["value"]+1e-12 for a,b in zip(boundary["trace"],boundary["trace"][1:]))
    capped = solve(quadratic, np.zeros(3), max_evaluations=1)
    assert capped["status"] == "evaluation_cap" and capped["objective_calls"] == 1
    print("3 manufactured solver cases passed; no scientific RNG")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    selftest()

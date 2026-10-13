"""Saved-grid Gaussian comparison weights and finite R02 error propagation.

Pure functions only: no file I/O, RNG, NPZ, optimizer, target D_n evaluations,
or imports from production noise code. Freeze production inputs before calling
make_weight_grid; propagate evaluates only saved nodes and closed scalars.
"""
from __future__ import annotations

from fractions import Fraction
import math
import time

import flint
from flint import arb, ctx


VERSION = "0.8.0"
K = 100
Q = 98
N_VALUES = (64, 256, 1024, 4096)


def exact(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError("explicit rational required; floats/bools rejected")
    return Fraction(value)


def aq(value):
    value = exact(value)
    return arb(value.numerator) / value.denominator


def _endpoint(value):
    m, e = value.man_exp()
    return {"mantissa": str(m), "exponent_of_two": int(e)}


def endpoint_fraction(data):
    m, e = exact(data["mantissa"]), exact(data["exponent_of_two"])
    if m.denominator != 1 or e.denominator != 1:
        raise ValueError("integer dyadic fields required")
    return Fraction(m.numerator * 2 ** max(e.numerator, 0),
                    2 ** max(-e.numerator, 0))


def record(value):
    if not isinstance(value, arb) or not value.is_finite():
        raise ArithmeticError("finite Arb value required")
    return {"arb_display_only": str(value),
            "lower_exact_dyadic": _endpoint(value.lower()),
            "upper_exact_dyadic": _endpoint(value.upper())}


def _bounds(data):
    lo = endpoint_fraction(data["lower_exact_dyadic"])
    hi = endpoint_fraction(data["upper_exact_dyadic"])
    if lo > hi:
        raise ValueError("inverted interval")
    return lo, hi


def from_record(data):
    lo, hi = _bounds(data)
    return aq(lo).union(aq(hi))


def _lo(value):
    return endpoint_fraction(_endpoint(value.lower()))


def _hi(value):
    return endpoint_fraction(_endpoint(value.upper()))


def _up(value):
    return aq(_hi(value))


def _cap(data):
    """Only the upper endpoint is an input error budget."""
    _, hi = _bounds(data)
    if hi < 0:
        raise ValueError("negative upper budget")
    return aq(hi)


def _minimum_cap(*values):
    return aq(min(_hi(x) for x in values))


def _maximum_cap(*values):
    return aq(max(_hi(x) for x in values))


def _maximum_value(*values):
    """An enclosure for a maximum, not selection by displayed midpoint."""
    return aq(max(_lo(x) for x in values)).union(
        aq(max(_hi(x) for x in values)))


def _check_bits(bits):
    if flint.__version__ != VERSION:
        raise RuntimeError("python-flint 0.8.0 required")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 64:
        raise ValueError("integer precision >=64 required")


def _grid_parameters(step, N):
    step = exact(step)
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError("positive integer N required")
    if step <= 0 or N * step < 2:
        raise ValueError("positive step and L=N*step>=2 required")
    if (1 / step).denominator != 1:
        raise ValueError("the grid must contain -1, 0, and 1")
    return step


def make_weight_grid(step="1/128", N=1536, bits=192, hard_seconds=120):
    """One bounded standard-normal grid, retaining partials on failure.

    The default has 3073 nodes in [-12,12]. Smaller configurations are only
    manufactured interface fixtures, not alternative production grids.
    """
    _check_bits(bits)
    step = _grid_parameters(step, N)
    if (isinstance(hard_seconds, bool) or
            not isinstance(hard_seconds, (int, float)) or
            not math.isfinite(hard_seconds) or hard_seconds < 0):
        raise ValueError("finite nonnegative wall budget required")
    old = ctx.prec
    ctx.prec = bits
    start = time.perf_counter()
    rows, failure, current = [], None, None
    try:
        norm, root2 = (2 * arb.pi()).sqrt(), arb(2).sqrt()
        try:
            for j in range(-N, N + 1):
                current = j
                if time.perf_counter() - start >= hard_seconds:
                    break
                z = aq(j * step)
                phi = (-z * z / 2).exp() / norm
                cdf = (-z / root2).erfc() / 2
                rows.append({"index": j, "z_exact": str(j * step),
                             "phi": record(phi), "Phi": record(cdf)})
        except Exception as exc:
            failure = {"phase": "Gaussian_weight_grid", "index": current,
                       "exception_type": type(exc).__name__, "message": str(exc)}
        elapsed = time.perf_counter() - start
        complete = len(rows) == 2 * N + 1 and elapsed <= hard_seconds and failure is None
        result = {
            "schema": "r02_remainder_weight_grid_v1",
            "status": "complete" if complete else
                      ("runtime_failure" if failure else "budget_stop"),
            "configuration": {"step_exact": str(step), "N": N,
                              "limit_exact": str(N * step),
                              "precision_bits": bits,
                              "python_flint_version": flint.__version__,
                              "node_functions": ["standard_normal_phi", "standard_normal_Phi"],
                              "contains_minus_one_zero_one": True,
                              "purpose": "Gaussian comparison weights, not finite-n D_n"},
            "hard_seconds": hard_seconds, "elapsed_seconds": elapsed,
            "timing_excludes_callers_disk_write": True,
            "nodes_prescribed": 2 * N + 1, "nodes_completed": len(rows),
            "rows": rows,
            "work_counts": {"new_weight_grid_nodes": len(rows),
                            "new_gaussian_phi_evaluations": len(rows),
                            "new_gaussian_Phi_evaluations": len(rows),
                            "new_finite_n_noise_nodes": 0,
                            "new_D_n_evaluations": 0, "new_MC_draws": 0,
                            "new_NPZ_reads": 0}}
        if failure is not None:
            result["runtime_failure"] = failure
        result["elapsed_seconds"] = time.perf_counter() - start
        if result["status"] == "complete" and result["elapsed_seconds"] > hard_seconds:
            result["status"] = "budget_stop"
        return result
    finally:
        ctx.prec = old


def _absolute_bounds(lo, hi):
    if lo > hi:
        raise ValueError("inverted difference interval")
    return (Fraction(0) if lo <= 0 <= hi else min(abs(lo), abs(hi)),
            max(abs(lo), abs(hi)))


def _scale_bounds(c, lo, hi):
    return (c * lo, c * hi) if c >= 0 else (c * hi, c * lo)


def _prepare_grid(grid):
    if grid.get("schema") != "r02_remainder_weight_grid_v1" or grid.get("status") != "complete":
        raise ValueError("complete saved Gaussian grid required")
    config = grid["configuration"]
    if config["python_flint_version"] != VERSION:
        raise ValueError("grid backend version mismatch")
    step, N = exact(config["step_exact"]), config["N"]
    _grid_parameters(step, N)
    if exact(config["limit_exact"]) != N * step:
        raise ValueError("grid limit mismatch")
    if (grid["nodes_prescribed"] != 2 * N + 1 or
            grid["nodes_completed"] != 2 * N + 1 or len(grid["rows"]) != 2 * N + 1):
        raise ValueError("missing Gaussian nodes")
    nodes = []
    for j, row in zip(range(-N, N + 1), grid["rows"]):
        z = j * step
        if row["index"] != j or exact(row["z_exact"]) != z:
            raise ValueError("Gaussian node sequence mismatch")
        pl, ph = _bounds(row["phi"])
        cl, ch = _bounds(row["Phi"])
        if pl <= 0 or ch < 0 or cl > 1:
            raise ValueError("invalid saved normal density/CDF enclosure")
        # The known CDF range is exact even if low precision reaches past it.
        cl, ch = max(Fraction(0), cl), min(Fraction(1), ch)
        nodes.append({"z": z, "phi_lo": pl, "phi_hi": ph,
                      "cdf_lo": cl, "cdf_hi": ch})
    cells = []
    for a, b in zip(nodes, nodes[1:]):
        if any(a["z"] < point < b["z"] for point in (-1, 0, 1)):
            raise ValueError("cell crosses a derivative turning point")
        # Integral |phi'| = |phi(b)-phi(a)| on a cell not crossing 0.
        d1 = _absolute_bounds(b["phi_lo"] - a["phi_hi"],
                              b["phi_hi"] - a["phi_lo"])
        # Integral |phi''| = |phi'(b)-phi'(a)| on a cell not crossing +/-1.
        dl, dh = _scale_bounds(-a["z"], a["phi_lo"], a["phi_hi"])
        el, eh = _scale_bounds(-b["z"], b["phi_lo"], b["phi_hi"])
        d2 = _absolute_bounds(el - dh, eh - dl)
        cells.append({"left": a["z"], "right": b["z"],
                      "cdf_left_lo": a["cdf_lo"], "cdf_right_hi": b["cdf_hi"],
                      "d1_lo": d1[0], "d1_hi": d1[1],
                      "d2_lo": d2[0], "d2_hi": d2[1]})
    return {"cells": cells, "limit": N * step, "step": step,
            "phi_limit_hi": nodes[-1]["phi_hi"], "nodes_count": len(nodes)}


def _weighted_compare(prepared, exponent, delta, kind):
    """Enclose a Gaussian comparison integral using only saved endpoints."""
    if isinstance(exponent, bool) or not isinstance(exponent, int) or exponent < 0:
        raise ValueError("nonnegative integer exponent required")
    delta = exact(delta)
    if delta < 0 or kind not in ("W1", "W1plus", "W2"):
        raise ValueError("nonnegative delta and known weight kind required")
    lower, upper, used = arb(0), arb(0), 0
    mass = "d2" if kind == "W2" else "d1"
    for cell in prepared["cells"]:
        if kind == "W1plus" and cell["left"] < 0:
            continue
        wl = min(Fraction(1), max(Fraction(0), cell["cdf_left_lo"] + delta))
        wh = min(Fraction(1), max(Fraction(0), cell["cdf_right_hi"] + delta))
        lower += aq(cell[mass + "_lo"]) * aq(wl) ** exponent
        upper += aq(cell[mass + "_hi"]) * aq(wh) ** exponent
        used += 1
    L, pL = prepared["limit"], prepared["phi_limit_hi"]
    tail = pL if kind == "W1plus" else (2 * L * pL if kind == "W2" else 2 * pL)
    lower_endpoint = max(Fraction(0), _lo(lower))
    upper_central_endpoint = _hi(upper)
    upper_endpoint = _hi(upper + aq(tail))
    if upper_endpoint < lower_endpoint:
        raise ArithmeticError("inverted Gaussian comparison enclosure")
    enclosure = aq(lower_endpoint).union(aq(upper_endpoint))
    return {"kind": kind, "exponent": exponent, "delta_upper_exact": str(delta),
            "support": "[0,infinity)" if kind == "W1plus" else "real_line",
            "lower_riemann_sum": record(lower), "upper_riemann_sum": record(upper),
            "tail_upper": record(aq(tail)), "enclosure": record(enclosure),
            "upper": record(_up(enclosure)),
            "riemann_width_upper": record(aq(max(Fraction(0), upper_central_endpoint - lower_endpoint))),
            "total_enclosure_width_exact": str(_hi(enclosure) - _lo(enclosure)),
            "saved_cells_used": used, "new_grid_function_evaluations": 0,
            "comparison_integral_lower_is_not_a_lower_bound_on_actual_R02_error": True}


def _hermite_constants():
    pi, root2pi = arb.pi(), (2 * arb.pi()).sqrt()
    phi = lambda z: (-z * z / 2).exp() / root2pi
    root6 = arb(6).sqrt()
    a, b = (3 - root6).sqrt(), (3 + root6).sqrt()
    he3 = lambda z: z ** 3 - 3 * z
    he4 = lambda z: z ** 4 - 6 * z * z + 3
    e_inf = _maximum_value(abs(he3(a) * phi(a)), abs(he3(b) * phi(b)))
    root10 = arb(10).sqrt()
    critical = [arb(0), (5 - root10).sqrt(), (5 + root10).sqrt()]
    ep_inf = _maximum_value(*(abs(he4(z) * phi(z)) for z in critical))
    return {"phi_zero": phi(arb(0)), "phi_prime_Linf": phi(arb(1)),
            "e_Linf": e_inf, "e_prime_Linf": ep_inf,
            "e_prime_L1": 4 * root6 * (a * phi(a) + b * phi(b)),
            "e_L2_squared": 15 / (16 * pi.sqrt()),
            "E1_CDF_Linf": phi(arb(0))}


def _active_constants(constants):
    names = ("v", "kappa3_L", "m_H", "cbar", "V_A", "V_H", "s0", "J_K",
             "second_order_coefficient")
    c = {name: from_record(constants[name]) for name in names}
    if not c["v"] > 0 or not c["V_H"] >= 0 or not c["J_K"] > 0:
        raise ValueError("positive variance/J_K and nonnegative V_H required")
    if not (c["s0"] > 0 and c["s0"] < 1 and c["m_H"] > 0 and c["m_H"] < aq("1/4")):
        raise ValueError("fixed-model spin/mark positivity qualification failed")
    rho, lamc = aq("9/20"), aq("19/10")
    vd, spin, t2 = c["V_A"] - aq("3/4"), 4 * c["m_H"], 1 - c["s0"]
    if not vd > 0:
        raise ValueError("positive active even-score variance required")
    vp, vz = 3 * lamc / 4 + vd, (3 - 2 * rho) / 4 + vd
    mp, mz = c["m_H"] + aq("1/4"), c["m_H"] - aq("1/12")
    cp = c["cbar"] - c["s0"] * (1 + rho) / 8
    cz = c["cbar"] + c["s0"] * (1 + rho) / 24
    mmv = abs(mp) * vp / 4 + 3 * abs(mz) * vz / 4
    mcv = abs(cp) * vp / 4 + 3 * abs(cz) * vz / 4
    b4 = ((27 + 36 * rho * rho) / 16 + (aq("9/2") + 3 * lamc) * vd + vd * vd
          + aq("9/4") * (1 + 2 * rho * rho) * t2
          + aq("9/2") * rho * (2 + rho) * spin)
    m2 = (c["V_H"] * b4).sqrt()
    checks = {"positive_Vd": bool(vd > 0), "positive_class_variances": bool(vp > 0 and vz > 0),
              "positive_fourth_moment_cap": bool(b4 > 0)}
    if "B_K" in constants:
        b2 = from_record(constants["B_K"]) / c["v"] * (
            c["m_H"] * c["kappa3_L"] / (3 * c["v"]) - c["cbar"])
        checks["old_second_order_coefficient_identity_overlaps"] = bool(
            b2.overlaps(c["second_order_coefficient"]))
    if not all(checks.values()):
        raise ValueError("old model constant identities/positivity failed")
    values = {"rho": rho, "lambda_C": lamc, "V_d": vd, "spin_kappa": spin,
              "t2": t2, "v_plus": vp, "v_zero": vz, "m_plus": mp, "m_zero": mz,
              "c_plus": cp, "c_zero": cz, "Mmv": mmv, "Mcv": mcv,
              "B4_bar_upper_formula": b4, "M2_upper_formula": m2,
              "v_max_upper": _maximum_cap(vp, vz)}
    return c, values, checks


def propagate(old_constants, noise_bounds, grid):
    """Compose the fixed four-n finite bound without evaluating new nodes.

    old_constants is CONSTANTS.json['composition']['constants']; noise_bounds
    is the separately saved compose_bounds output. No production noise import
    or function is called here. All reported error lower endpoints are merely
    arithmetic enclosures of upper-budget formulas, not actual-error bounds.
    """
    if noise_bounds.get("schema") != "r02_remainder_noise_bounds_v1":
        raise ValueError("saved finite-noise bounds schema required")
    if [r["n"] for r in noise_bounds["rows"]] != list(N_VALUES):
        raise ValueError("the frozen n sequence must be 64,256,1024,4096")
    bits = grid["configuration"]["precision_bits"]
    _check_bits(bits)
    old = ctx.prec
    ctx.prec = bits
    try:
        prepared = _prepare_grid(grid)
        c, active, active_checks = _active_constants(old_constants)
        hermite = _hermite_constants()
        rows = []
        for noise in noise_bounds["rows"]:
            n = noise["n"]
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError("integer n required")
            if noise.get("all_budgets_are_upper_bounds_not_actual_error_lower_bounds") is not True:
                raise ValueError("noise upper-budget interpretation flag missing")
            s = (n * c["v"]).sqrt()
            lam = c["kappa3_L"] / (6 * c["v"] * s)
            al = _up(abs(lam))
            r0 = _cap(noise["error_density_derivative_Linf"]["0"])
            r1 = _cap(noise["error_density_derivative_Linf"]["1"])
            rcdf = _cap(noise["error_CDF_Linf"])
            r0l2 = _cap(noise["error_density_L2"])
            r1l1 = _cap(noise["error_density_prime_L1"])
            df = _up(al * _up(hermite["e_Linf"]) + r0)
            dp = _up(al * _up(hermite["e_prime_Linf"]) + r1)
            dg = _minimum_cap(arb(1), al * _up(hermite["E1_CDF_Linf"]) + rcdf)
            vdiff = _up(al * _up(hermite["e_prime_L1"]) + r1l1)
            a2sq = _up((al * _up(hermite["e_L2_squared"].sqrt()) + r0l2) ** 2)
            direct = noise["direct_density_derivative_Linf"]
            L0 = _minimum_cap(_cap(direct["0"]), hermite["phi_zero"] + df)
            L1 = _minimum_cap(_cap(direct["1"]), hermite["phi_prime_Linf"] + dp)
            L2 = _cap(direct["2"])
            if min(_lo(L0), _lo(L1), _lo(L2)) < 0:
                raise ValueError("negative density derivative norm cap")
            Delta = _up(dg + L1 * active["v_max_upper"] / (2 * s * s))
            weights = {
                "W1_q_Delta": _weighted_compare(prepared, Q, _hi(Delta), "W1"),
                "W1_q_minus1_Delta": _weighted_compare(prepared, Q - 1, _hi(Delta), "W1"),
                "W2_q_deltaG": _weighted_compare(prepared, Q, _hi(dg), "W2"),
                "W1plus_q_minus1_deltaG": _weighted_compare(prepared, Q - 1, _hi(dg), "W1plus")}
            W1q = _cap(weights["W1_q_Delta"]["upper"])
            W1qm1 = _cap(weights["W1_q_minus1_Delta"]["upper"])
            Wq = W1q + vdiff
            Wqm1 = W1qm1 + vdiff
            W2 = _cap(weights["W2_q_deltaG"]["upper"])
            Wplus = _cap(weights["W1plus_q_minus1_deltaG"]["upper"])
            A = K * (K - 1)
            E1 = K * L2 * active["M2_upper_formula"] / (2 * s ** 3)
            E2 = A * active["Mmv"] / (2 * s ** 3) * (L2 / (K - 1) + L1 * Wq)
            E3 = A * active["Mcv"] / (2 * s ** 4) * (L2 * Wq + Q * L1 * L0 * Wqm1)
            E4 = A * abs(c["cbar"]) / (s * s) * (dp / (K - 1) + dg * W2)
            Ilinear = (2 * r0 + c["J_K"] * rcdf) / (K - 1)
            Iquadratic = a2sq + 2 * df * dg + Q * dg * dg * (Wplus + df) / 2
            E5 = A * abs(c["m_H"]) / s * (Ilinear + Iquadratic)
            budgets = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5}
            if not all(x >= 0 and x.is_finite() for x in budgets.values()):
                raise ArithmeticError("nonnegative finite error budgets required")
            total = sum(budgets.values(), arb(0))
            total_up = _up(total)
            leading = -c["m_H"] * c["J_K"] / s
            correction = c["second_order_coefficient"] / n
            d2 = leading + correction
            dn = d2 + arb(0, total_up)
            negative = bool(dn < 0)
            denom = abs(d2)
            ratios = ({name: record(value / denom) for name, value in budgets.items()}
                      if denom > 0 else None)
            Efirst = abs(correction) + total_up
            formula_inputs = {
                "s": s, "L0": L0, "L1": L1, "L2": L2,
                "deltaG": dg, "deltaf": df, "deltap": dp,
                "Vdiff": vdiff, "a2_squared": a2sq, "R0": r0, "RCDF": rcdf,
                "Delta": Delta, "M2": active["M2_upper_formula"],
                "Mmv": active["Mmv"], "Mcv": active["Mcv"],
                "M2_upper": _up(active["M2_upper_formula"]),
                "Mmv_upper": _up(active["Mmv"]), "Mcv_upper": _up(active["Mcv"]),
                "m_H": c["m_H"], "cbar": c["cbar"], "J_K": c["J_K"],
                "v": c["v"], "second_order_coefficient": c["second_order_coefficient"],
                "W1_q_Delta": W1q, "W1_q_minus1_Delta": W1qm1,
                "W2_q_deltaG": W2, "W1plus_q_minus1_deltaG": Wplus,
                "Wg_q": Wq, "Wg_q_minus1": Wqm1}
            rows.append({
                "n": n, "s": record(s), "lambda": record(lam),
                "propagation_inputs": {k: record(v) for k, v in formula_inputs.items()},
                "s_lower_exact": str(_lo(s)), "s_upper_exact": str(_hi(s)),
                "noise_input_caps": {"r0_Linf": record(r0), "r1_Linf": record(r1),
                                     "R_CDF": record(rcdf), "r0_L2": record(r0l2),
                                     "r1_L1": record(r1l1)},
                "actual_noise_norm_caps": {
                    "delta_f": record(df), "delta_p": record(dp), "delta_G": record(dg),
                    "Vdiff": record(vdiff), "a2_squared": record(a2sq),
                    "L0": record(L0), "L1": record(L1), "L2": record(L2),
                    "Delta": record(Delta), "Wg_q": record(Wq), "Wg_q_minus1": record(Wqm1)},
                "weights": weights, "budgets": {k: record(v) for k, v in budgets.items()},
                "I_linear_error_budget": record(Ilinear),
                "I_quadratic_error_budget": record(Iquadratic),
                "E_total": record(total), "E_total_upper": record(total_up),
                "leading_term": record(leading), "signed_second_order_term": record(correction),
                "D_second_order": record(d2), "D_n_theoretical_enclosure": record(dn),
                "theoretical_D_interval": record(dn),
                "bound_qualification_passed": True,
                "negative_sign_certified": negative,
                "status": "negative" if negative else "unresolved",
                "budget_ratios_to_abs_D_second_order": ratios,
                "total_ratio_to_abs_D_second_order": record(total_up / denom) if denom > 0 else None,
                "E_first_order": record(Efirst),
                "E_first_order_ratio_to_abs_leading": record(Efirst / abs(leading)),
                "first_order_sign_test_passes": bool(Efirst < abs(leading)),
                "all_error_budget_lower_endpoints_are_not_actual_error_lower_bounds": True,
                "target_D_n_evaluated": False})
        return {
            "schema": "r02_remainder_finite_propagation_v1",
            "configuration": {"K": K, "q": Q, "n_values": list(N_VALUES),
                              "precision_bits": bits, "python_flint_version": flint.__version__},
            "hermite_constants": {k: record(v) for k, v in hermite.items()},
            "active_constants": {k: record(v) for k, v in active.items()},
            "active_constant_checks": active_checks, "rows": rows,
            "bound_qualification_passed": True,
            "noise_certificate_must_be_verified_separately": True,
            "negative_certified_n": [r["n"] for r in rows if r["negative_sign_certified"]],
            "unresolved_n": [r["n"] for r in rows if not r["negative_sign_certified"]],
            "saved_grid_nodes_used": prepared["nodes_count"],
            "saved_grid_functions_recomputed": False,
            "new_integral_nodes": 0, "new_grid_function_evaluations": 0,
            "new_fourier_integration_nodes": 0, "new_D_n_evaluations": 0,
            "new_MC_draws": 0, "new_NPZ_reads": 0,
            "closed_scalar_transcendental_evaluations_only": True,
            "target_D_n_evaluated": False,
            "theoretical_bound_not_a_new_target_or_MC_estimate": True}
    finally:
        ctx.prec = old

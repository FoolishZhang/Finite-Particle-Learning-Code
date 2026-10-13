"""Analytic nested-grid L2 error for paired F_K radial gradients.

Scalar budget only: no label/prior/PMF integration, arrays, or optimization.
The fixed F8 use has four paired exact-dyadic lengths and zero bias.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import operator
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/finite_selection_tools"))
import flint
from flint import arb, ctx

PLANNING_CENTER_PAIRS = (
    "4236614515730759/9007199254740992",
    "1274451243427741/2251799813685248",
    "181129089615167/281474976710656",
    "775431105319425/1125899906842624")
TEACHER_ERROR = "2447/500000000000000000000000"


def exact_q(value):
    if isinstance(value, (bool, float)):
        raise ValueError("exact rational str/int/Fraction required")
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid exact rational") from error


def actual_int(value):
    if isinstance(value, bool):
        raise ValueError("actual positive integer required")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError("actual positive integer required") from error
    if value < 1:
        raise ValueError("actual positive integer required")
    return value


def aq(value):
    value = exact_q(value)
    return arb(value.numerator)/value.denominator


def endpoint(value):
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent_of_two": int(exponent)}


def record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite bound")
    return {"arb_display_only": str(value), "lower_exact_dyadic": endpoint(value.lower()),
            "upper_exact_dyadic": endpoint(value.upper())}


def dyadic_lt_q(ep, target):
    q = exact_q(target)
    m, e = int(ep["mantissa"]), operator.index(ep["exponent_of_two"])
    if m < 0 or q < 0:
        raise ValueError("nonnegative comparison operands required")
    left, right = m*q.denominator, q.numerator
    if not left or not right:
        return left < right
    lb, rb = left.bit_length()+max(e, 0), right.bit_length()+max(-e, 0)
    if lb != rb:
        return lb < rb
    return (left << max(e, 0)) < (right << max(-e, 0))


def gradient_l2_bounds(paired_a, *, K=8, hz="1/40", L="11", dz="1/8",
                       hv="1/8", vmin="-32", vmax="64", dv="1/2", T="4",
                       teacher_error_L1_upper=TEACHER_ERROR,
                       teacher_mass_upper=None, precision_bits=192, target="1/100000000"):
    """JSON bounds for mathematical full-weight nested trapezoid, no arithmetic.

    All labels share this bound. Parameters are paired zero-bias lengths with
    Gred_j=z*(c_j-2*sigma(a_j*z))/sqrt(2), c_j in {0,1,2}.
    Teacher mass/error refer to the exact mathematical finite teacher sum;
    errors of saved/computed Arb balls belong to the later compute enclosure.
    """
    lengths = tuple(exact_q(value) for value in paired_a)
    if not lengths:
        raise ValueError("at least one paired length required")
    K, precision_bits = actual_int(K), actual_int(precision_bits)
    if precision_bits < 64:
        raise ValueError("precision_bits>=64 required")
    names = ("hz", "L", "dz", "hv", "vmin", "vmax", "dv", "T")
    inputs = {name: exact_q(value) for name, value in zip(
        names, (hz, L, dz, hv, vmin, vmax, dv, T))}
    if min(inputs[name] for name in ("hz", "L", "dz", "hv", "dv", "T")) <= 0:
        raise ValueError("positive grid/strip/T parameters required")
    if inputs["L"] < 1 or inputs["vmin"] >= inputs["vmax"]:
        raise ValueError("L>=1 and vmin<vmax required")
    Nzq = inputs["L"]/inputs["hz"]
    Nvwidthq = (inputs["vmax"]-inputs["vmin"])/inputs["hv"]
    if Nzq.denominator != 1 or Nvwidthq.denominator != 1:
        raise ValueError("L/hz and outer width/hv must be exact integers")
    Nz, Nvwidth = actual_int(Nzq.numerator), actual_int(Nvwidthq.numerator)
    te = exact_q(teacher_error_L1_upper)
    tm = 1+te if teacher_mass_upper is None else exact_q(teacher_mass_upper)
    targetq = exact_q(target)
    if te < 0 or tm < 0 or targetq <= 0:
        raise ValueError("nonnegative teacher bounds, positive target required")
    previous = ctx.prec; ctx.prec = precision_bits
    try:
        x = {name: aq(value) for name, value in inputs.items()}
        a = [aq(abs(value)) for value in lengths]
        m = 2*len(a)
        e, CG = arb(1).exp(), arb(m).sqrt()
        eg = CG*(2/arb.pi()).sqrt()
        asum = 2*sum(a, arb(0))
        theta = asum*x["dz"]
        if not theta < arb.pi()/2 or not x["dv"] < arb.pi()/2:
            raise ValueError("inner phase and outer Re(u)>0 strip not established")
        cos_theta, cos_v = theta.cos(), x["dv"].cos()
        if not cos_theta > 0 or not cos_v > 0:
            raise ValueError("strict positive sector cosine required")
        Mpsi = (x["dz"]**2/2).exp()
        Ccomplex = (2*sum((1/(aa*x["dz"]/2).cos()**2 for aa in a), arb(0))).sqrt()
        MD = Mpsi*Ccomplex*(1+x["dz"]**2).sqrt()/(e*cos_theta)
        qz = (-2*arb.pi()*x["dz"]/x["hz"]).exp()
        psi_alias, D_alias = 2*Mpsi*qz/(1-qz), 2*MD*qz/(1-qz)
        phi = lambda xx: (-xx*xx/2).exp()/(2*arb.pi()).sqrt()
        psi_tail, D_tail = 2*phi(x["L"])/x["L"], 2*CG*phi(x["L"])/e
        delta_psi, delta_D = psi_alias+psi_tail, D_alias+D_tail
        Dmax = eg/e
        Wv = x["vmax"]-x["vmin"]+x["hv"]
        # Polynomial telescoping; valid also for K=1 without negative powers.
        power_error = arb(0) if K == 1 else (
            (K-1)*(1+delta_psi)**(K-2)*delta_psi*Dmax)
        nested = K*Wv*((1+delta_psi)**(K-1)*delta_D+power_error)
        Mv = K*eg/cos_v
        qv = (-2*arb.pi()*x["dv"]/x["hv"]).exp()
        outer_alias = 2*Mv*qv/(1-qv)
        lower = K*eg*x["hv"]*x["vmin"].exp()/x["hv"].expm1()
        eta = (-x["T"]*asum).exp()/(arb(2)**m)
        Ueta = x["vmax"].exp()*eta
        if not Ueta > 1:
            raise ValueError("good-event discrete upper-tail monotonicity U*eta>1 required")
        pT = 2*phi(x["T"])/x["T"]
        upper_good = CG*arb(K).sqrt()*(-Ueta).exp()
        upper_bad = CG*(1+2*x["hv"]/e)*2*K*phi(x["T"])*pT**(K-1)
        upper = upper_good+upper_bad
        uniform = nested+outer_alias+lower+upper
        global_true_gradient = arb(m*K).sqrt()
        teacher_transfer = aq(te)*global_true_gradient
        total = aq(tm)*uniform+teacher_transfer
        finite_grid_gradient = K*Wv*(1+delta_psi)**(K-1)*(Dmax+delta_D)
        scalars = {"paired_l1_full_asum": asum, "inner_phase_theta": theta,
            "inner_Psi_boundary_L1": Mpsi, "inner_D_boundary_L1_L2": MD,
            "inner_alias_Psi": psi_alias, "inner_alias_D_L2": D_alias,
            "inner_gridtail_Psi": psi_tail, "inner_gridtail_D_L2": D_tail,
            "inner_delta_Psi": delta_psi, "inner_delta_D_L2": delta_D,
            "inner_nested_L2": nested, "outer_boundary_L1_L2": Mv,
            "outer_alias_L2": outer_alias, "outer_gridtail_lower_L2": lower,
            "outer_gridtail_upper_good_L2": upper_good,
            "outer_gridtail_upper_bad_L2": upper_bad,
            "outer_gridtail_upper_L2": upper, "upper_U_eta": Ueta,
            "uniform_per_label_math_L2": uniform,
            "true_per_label_gradient_global_L2": global_true_gradient,
            "finite_grid_per_label_gradient_L2": finite_grid_gradient,
            "teacher_transfer_L2": teacher_transfer,
            "teacher_weighted_math_L2": aq(tm)*uniform,
            "total_without_compute_arithmetic_L2": total,
            "remaining_arithmetic_budget": aq(targetq)-total}
        return {"schema_version": "1.0", "scope": "paired_gradient_nested_trapezoid_L2_scalar_budget",
            "precision_bits": precision_bits, "python_flint_version": flint.__version__,
            "K_actual_integer": K, "pairs_actual_integer": len(a), "m_actual_integer": m,
            "exact_inputs": {**{k: str(v) for k, v in inputs.items()},
                "paired_a": [str(v) for v in lengths],
                "teacher_error_L1_upper": str(te), "teacher_mass_upper": str(tm)},
            "grid": {"inner_min_index": -Nz, "inner_max_index": Nz,
                     "inner_nodes": 2*Nz+1, "outer_nodes": Nvwidth+1,
                     "outer_first": str(inputs["vmin"]), "outer_last": str(inputs["vmax"]),
                     "all_endpoint_weights_full": True, "renormalized": False},
            "uniform_all_labels": True, "factor_label_count_or_orbit_count_used": False,
            "strip_conditions": {"inner_theta_strictly_below_pi_over_2": True,
                "outer_dv_strictly_below_pi_over_2": True, "upper_U_eta_strictly_above_1": True},
            "bounds": {key: record(value) for key, value in scalars.items()},
            "comparison": {"target_exact_rational": str(targetq),
                "strictly_below": dyadic_lt_q(endpoint(total.upper()), targetq),
                "method": "exact_integer_dyadic_vs_rational"},
            "teacher_decomposition": "qfinite*(gtrue-ggrid)+(ptrue-qfinite)*gtrue",
            "finite_teacher_ball_midpoints_interpreted_as_exact": False,
            "compute_arithmetic_included": False, "real_gradient_evaluations": 0,
            "scientific_data_reads": 0, "random_draws": 0, "acb_integral_calls": 0}
    finally:
        ctx.prec = previous

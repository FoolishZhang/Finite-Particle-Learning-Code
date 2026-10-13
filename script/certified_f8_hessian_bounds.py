"""Scalar operator-norm budgets for full Hba and transverse Hp nested grids.

No real Hessian integration, old NPZ access, optimization, or RNG.
"""
from __future__ import annotations
from fractions import Fraction
import operator
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/finite_selection_tools"))
import flint
from flint import arb, ctx

PLANNING_CENTER_PAIRS = (
    "4236614515730759/9007199254740992", "1274451243427741/2251799813685248",
    "181129089615167/281474976710656", "775431105319425/1125899906842624")
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


def hessian_operator_bounds(paired_a, *, K=8, hz="1/40", L="11", dz="1/8",
                            hv="1/8", vmin="-32", vmax="64", dv="1/2", T="4",
                            teacher_error_L1_upper=TEACHER_ERROR,
                            teacher_mass_upper=None, precision_bits=192, target="1/1000"):
    lengths = tuple(exact_q(value) for value in paired_a)
    if not lengths:
        raise ValueError("at least one paired length required")
    K, precision_bits = actual_int(K), actual_int(precision_bits)
    if K < 2 or precision_bits < 64:
        raise ValueError("K>=2,precision>=64 required")
    names = ("hz", "L", "dz", "hv", "vmin", "vmax", "dv", "T")
    inputs = {name: exact_q(value) for name, value in zip(
        names, (hz, L, dz, hv, vmin, vmax, dv, T))}
    if min(inputs[name] for name in ("hz", "L", "dz", "hv", "dv", "T")) <= 0:
        raise ValueError("positive grid/strip/T parameters required")
    if inputs["L"] < 2 or inputs["vmin"] >= inputs["vmax"]:
        raise ValueError("L>=2 and vmin<vmax required")
    Nzq = inputs["L"]/inputs["hz"]
    Nvwidthq = (inputs["vmax"]-inputs["vmin"])/inputs["hv"]
    if Nzq.denominator != 1 or Nvwidthq.denominator != 1:
        raise ValueError("grid widths must be exact integer multiples")
    Nz, Nvwidth = actual_int(Nzq.numerator), actual_int(Nvwidthq.numerator)
    te = exact_q(teacher_error_L1_upper)
    tm = 1+te if teacher_mass_upper is None else exact_q(teacher_mass_upper)
    targetq = exact_q(target)
    if te < 0 or tm < 0 or targetq <= 0:
        raise ValueError("nonnegative teacher bounds,positive target required")
    previous = ctx.prec; ctx.prec = precision_bits
    try:
        x = {name: aq(value) for name, value in inputs.items()}
        a = [aq(abs(value)) for value in lengths]
        m = 2*len(a); quarter = arb(1)/4; e = arb(1).exp()
        asum = 2*sum(a, arb(0)); theta = asum*x["dz"]
        if not theta < arb.pi()/2 or not x["dv"] < arb.pi()/2:
            raise ValueError("inner sector or outer Re(u)>0 strip not established")
        cz, cv = theta.cos(), x["dv"].cos()
        if not cz > 0 or not cv > 0:
            raise ValueError("positive sector cosines required")
        gauss = (x["dz"]**2/2).exp()
        sec2 = [1/(aa*x["dz"]/2).cos()**2 for aa in a]
        cr2 = 2*sum(sec2, arb(0))
        # secant is increasing on these established positive arguments.
        amax = aq(max(abs(value) for value in lengths))
        secmax2 = 1/(amax*x["dz"]/2).cos()**2
        moment = 2+x["dz"]**2
        base = gauss/(e*cz)
        M = {"Psi": gauss, "T0": base, "R": base*(cr2*moment).sqrt(),
             "Q": base*cr2*moment, "B": base*secmax2*moment/4,
             "Qp": base*cr2, "Bp": base*secmax2/4}
        real = {"T0": 1/e, "R": arb(2*m).sqrt()/e, "Q": 2*m/e,
                "B": arb(1)/2/e, "Qp": m/e, "Bp": quarter/e}
        phi = lambda zz: (-zz*zz/2).exp()/(2*arb.pi()).sqrt()
        pL = 2*phi(x["L"])/x["L"]
        momtail = 2*phi(x["L"])*(x["L"]+2/x["L"])
        tails = {"Psi": pL, "T0": pL/e,
                 "R": 2*arb(2*m).sqrt()*phi(x["L"])/e,
                 "Q": m*momtail/e, "B": momtail/4/e,
                 "Qp": m*pL/e, "Bp": pL/4/e}
        qz = (-2*arb.pi()*x["dz"]/x["hz"]).exp()
        aliases = {key: 2*value*qz/(1-qz) for key, value in M.items()}
        delta = {key: aliases[key]+tails[key] for key in M}
        Wv = x["vmax"]-x["vmin"]+x["hv"]
        def power_product(n, b, db):
            perturb = arb(0) if n == 0 else n*(1+delta["Psi"])**(n-1)*delta["Psi"]*b
            return (1+delta["Psi"])**n*db+perturb
        Cmax = real["T0"]*real["Q"]+real["R"]**2
        dC = (delta["T0"]*real["Q"]+(real["T0"]+delta["T0"])*delta["Q"]
              +2*real["R"]*delta["R"]+delta["R"]**2)
        Cpmax = real["T0"]*real["Qp"]
        dCp = delta["T0"]*real["Qp"]+(real["T0"]+delta["T0"])*delta["Qp"]
        inner_ba = Wv*(K*power_product(K-1, real["B"], delta["B"])
                      +K*(K-1)*power_product(K-2, Cmax, dC))
        inner_p = Wv*(K*power_product(K-1, real["Bp"], delta["Bp"])
                     +K*(K-1)*power_product(K-2, Cpmax, dCp))
        Mvp = quarter/cv+m/cv**2; Mvba = (1+K)*Mvp
        qv = (-2*arb.pi()*x["dv"]/x["hv"]).exp()
        outer_ba, outer_p = 2*Mvba*qv/(1-qv), 2*Mvp*qv/(1-qv)
        low1 = x["hv"]*x["vmin"].exp()/x["hv"].expm1()
        low2 = x["hv"]*(2*x["vmin"]).exp()/(2*x["hv"]).expm1()
        lower_p = K*low1/4+m*K*K*low2
        lower_ba = (1+K)*lower_p
        eta = (-x["T"]*asum).exp()/(arb(2)**m)
        Ueta = x["vmax"].exp()*eta
        if not Ueta > 2:
            raise ValueError("U*eta>2 required for both good-event discrete tails")
        pT = 2*phi(x["T"])/x["T"]
        t2T = 2*phi(x["T"])*(x["T"]+1/x["T"])
        weighted_bad = pT**K+K*t2T*pT**(K-1)
        discrete_bad_coefficient = quarter*(1+2*x["hv"]/e)+m*(1+8*x["hv"]/e**2)
        good_p = (quarter+m*(1+Ueta))*(-Ueta).exp()
        good_ba = (1+K)*good_p
        bad_ba, bad_p = discrete_bad_coefficient*weighted_bad, discrete_bad_coefficient*pT**K
        upper_ba, upper_p = good_ba+bad_ba, good_p+bad_p
        uniform_ba = inner_ba+outer_ba+lower_ba+upper_ba
        uniform_p = inner_p+outer_p+lower_p+upper_p
        # Deliberately preserve the preceding geometry/proof's conservative
        # global full-parameter bound, rather than tightening it here.
        global_ba, global_p = (2*m+quarter)*(1+K), m+quarter
        teacher_ba, teacher_p = aq(te)*global_ba, aq(te)*global_p
        total_ba, total_p = aq(tm)*uniform_ba+teacher_ba, aq(tm)*uniform_p+teacher_p
        scalars = {"full_l1_asum": asum, "inner_phase_theta": theta,
            "inner_G_complex_squared_coefficient": cr2,
            "inner_B_complex_secant_squared": secmax2,
            "outer_boundary_Hba_operator_L1": Mvba, "outer_boundary_Hp_operator_L1": Mvp,
            "outer_window_weight": Wv, "upper_U_eta": Ueta,
            "inner_Hba_operator": inner_ba, "inner_Hp_operator": inner_p,
            "outer_alias_Hba_operator": outer_ba, "outer_alias_Hp_operator": outer_p,
            "outer_lower_Hba_operator": lower_ba, "outer_lower_Hp_operator": lower_p,
            "outer_upper_good_Hba_operator": good_ba, "outer_upper_good_Hp_operator": good_p,
            "outer_upper_bad_Hba_operator": bad_ba, "outer_upper_bad_Hp_operator": bad_p,
            "outer_upper_Hba_operator": upper_ba, "outer_upper_Hp_operator": upper_p,
            "uniform_Hba_operator": uniform_ba, "uniform_Hp_operator": uniform_p,
            "teacher_global_Hba_operator": global_ba, "teacher_global_Hp_operator": global_p,
            "teacher_transfer_Hba_operator": teacher_ba, "teacher_transfer_Hp_operator": teacher_p,
            "total_Hba_operator_without_arithmetic": total_ba,
            "total_Hp_operator_without_arithmetic": total_p,
            "remaining_Hba_operator_budget": aq(targetq)-total_ba,
            "remaining_Hp_operator_budget": aq(targetq)-total_p}
        comparisons = [{"matrix": key, "target_exact_rational": str(targetq),
            "strictly_below": dyadic_lt_q(endpoint(value.upper()), targetq),
            "method": "exact_integer_dyadic_vs_rational"}
            for key, value in (("Hba", total_ba), ("Hp", total_p))]
        return {"schema_version": "1.0", "scope": "full_Hba_and_Hp_operator_scalar_budget",
            "precision_bits": precision_bits, "python_flint_version": flint.__version__,
            "K_actual_integer": K, "m_actual_integer": m, "Hba_dimension": 2*m, "Hp_dimension": m,
            "exact_inputs": {**{k: str(v) for k, v in inputs.items()},
                "paired_a": [str(v) for v in lengths], "teacher_error_L1_upper": str(te),
                "teacher_mass_upper": str(tm)},
            "grid": {"inner_nodes": 2*Nz+1, "outer_nodes": Nvwidth+1,
                "full_endpoint_weights": True, "renormalized": False},
            "moments": {key: {"boundary": record(M[key]), "alias": record(aliases[key]),
                        "grid_tail": record(tails[key]), "total_error": record(delta[key])}
                        for key in M},
            "bounds": {key: record(value) for key, value in scalars.items()},
            "comparisons": comparisons, "uniform_all_labels": True,
            "entrywise_errors_used_as_operator_errors": False,
            "teacher_decomposition": "qfinite*(Htrue-Hgrid)+(ptrue-qfinite)*Htrue",
            "finite_teacher_ball_midpoints_interpreted_as_exact": False,
            "compute_arithmetic_included": False, "old_saved_matrix_distance_included": False,
            "real_Hessian_evaluations": 0, "scientific_data_reads": 0,
            "random_draws": 0, "acb_integral_calls": 0}
    finally:
        ctx.prec = previous

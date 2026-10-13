"""Pure frozen-input CF/moment sums and analytic finite-noise bounds.

No file I/O, RNG, optimizer, project imports, finite-n target gradients, or
Fourier quadrature. Production callers must freeze inputs and source first.
"""
from __future__ import annotations

from fractions import Fraction
import math
import time

import flint
from flint import acb, arb, ctx

VERSION = "0.8.0"


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


def record(value):
    if not isinstance(value, arb) or not value.is_finite():
        raise ArithmeticError("finite Arb value required")
    return {"arb_display_only": str(value),
            "lower_exact_dyadic": _endpoint(value.lower()),
            "upper_exact_dyadic": _endpoint(value.upper())}


def endpoint_fraction(data):
    m, e = exact(data["mantissa"]), exact(data["exponent_of_two"])
    if m.denominator != 1 or e.denominator != 1:
        raise ValueError("integer dyadic fields required")
    return Fraction(m.numerator * 2 ** max(e.numerator, 0),
                    2 ** max(-e.numerator, 0))


def from_record(data):
    lo = endpoint_fraction(data["lower_exact_dyadic"])
    hi = endpoint_fraction(data["upper_exact_dyadic"])
    if lo > hi:
        raise ValueError("inverted interval")
    return aq(lo).union(aq(hi))


def _hi(value):
    return aq(endpoint_fraction(_endpoint(value.upper())))


def _lo(value):
    return aq(endpoint_fraction(_endpoint(value.lower())))


def _maximum(*values):
    return aq(max(endpoint_fraction(_endpoint(x.upper())) for x in values))


def _minimum(*values):
    return aq(min(endpoint_fraction(_endpoint(x.upper())) for x in values))


def enlarge(value, error):
    return value + arb(0, _hi(error))


def _check_version_bits(bits):
    if flint.__version__ != VERSION:
        raise RuntimeError("python-flint 0.8.0 required")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 64:
        raise ValueError("integer precision >=64 required")


def _inputs(constants):
    result = {k: from_record(constants[k]) for k in ("v", "d0", "kappa3_L")}
    if not result["v"] > 0 or not result["d0"] >= 0:
        raise ValueError("positive v and nonnegative d0 required")
    return result


def _raw4_bounds(d0, step, N, strip):
    pi, eta, L = arb.pi(), aq(strip), aq(step * N)
    d0hi = _hi(d0)
    logsec = -(eta / 2).cos().log()
    shift = d0hi + eta / 2 + logsec + eta / 2
    mu1 = (2 / pi).sqrt()
    moments = [arb(1), mu1, arb(1), 2 * mu1, arb(3)]
    boundary = (eta * eta / 2).exp() * sum(
        math.comb(4, k) * shift ** (4-k) * moments[k] for k in range(5))
    phiL = (-L*L/2).exp() / (2*pi).sqrt()
    P = [2*phiL/L, 2*phiL, 2*phiL*(L+1/L),
         2*phiL*(L*L+2), 2*phiL*(L**3+3*L+3/L)]
    tail = sum(math.comb(4, k) * d0hi ** (4-k) * P[k] for k in range(5))
    alias = 2 * boundary / ((2*pi*eta/aq(step)).exp()-1)
    return {"strip_B": record(shift), "boundary_single_side": record(boundary),
            "alias": record(alias), "tail": record(tail),
            "total": record(alias+tail)}


def _cf_error(u, v, step, N, strip):
    pi, eta, L = arb.pi(), aq(strip), aq(N*step)
    boundary = (eta*eta/2 + abs(aq(u))*eta/_lo(v).sqrt()).exp()
    alias = 2*boundary / ((2*pi*eta/aq(step)).exp()-1)
    tail = 2*(-L*L/2).exp() / ((2*pi).sqrt()*L)
    return {"boundary_single_side": record(boundary), "alias": record(alias),
            "tail": record(tail), "total": record(alias+tail)}


def finite_noise(constants, *, step="1/16", N=192, u_start="1/4",
                 u_step="1/64", u_count=273, strip="1", bits=192,
                 hard_seconds=600):
    """One bounded raw fourth-moment sum and CF table; partials preserved.

    constants must contain saved dyadic intervals v,d0,kappa3_L for the exact
    Gaussian-logistic nuisance model. Generic small grids are for fixtures;
    only the externally frozen production configuration supports the study.
    """
    _check_version_bits(bits)
    step, u_start, u_step, strip = map(exact, (step, u_start, u_step, strip))
    for value, name in ((N, "N"), (u_count, "u_count")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + " must be a positive integer")
    if step <= 0 or N*step < 2 or u_start <= 0 or u_step <= 0:
        raise ValueError("positive spacings/start and L>=2 required")
    if not 0 < strip <= 1:
        raise ValueError("0<strip<=1 required")
    if isinstance(hard_seconds, bool) or not math.isfinite(hard_seconds) or hard_seconds < 0:
        raise ValueError("finite nonnegative wall budget required")
    old = ctx.prec
    ctx.prec = bits
    start = time.perf_counter()
    try:
        values = _inputs(constants)
        v, d0 = values["v"], values["d0"]
        rootv, h, pi = v.sqrt(), aq(step), arb.pi()
        gaussian, raw4, node_indices = [], arb(0), []
        cf_rows, complex_visits = [], 0
        runtime_failure = None
        phase, current_index = "gaussian_nodes", None
        try:
            for j in range(-N, N+1):
                current_index = j
                if time.perf_counter()-start >= hard_seconds:
                    break
                x = aq(j*step)
                weight = h*(-x*x/2).exp()/(2*pi).sqrt()
                centered = x/2-(x/2).cosh().log()+d0
                raw4 += weight*centered**4
                gaussian.append((weight, centered/rootv))
                node_indices.append(j)
            if len(node_indices) == 2*N+1:
                phase = "frequency_nodes"
                for k in range(u_count):
                    current_index = k
                    if time.perf_counter()-start >= hard_seconds:
                        break
                    u = u_start+k*u_step
                    total, visits = acb(0), 0
                    for weight, standardized in gaussian:
                        if time.perf_counter()-start >= hard_seconds:
                            break
                        total += weight*acb(0, aq(u)*standardized).exp()
                        visits += 1
                        complex_visits += 1
                    complete = visits == len(gaussian)
                    err = _cf_error(u, v, step, N, strip)
                    er = from_record(err["total"])
                    finite_abs = abs(total)
                    row = {"index": k, "u_exact": str(u), "complete": complete,
                           "gaussian_nodes_completed": visits,
                           "finite_real": record(total.real), "finite_imag": record(total.imag),
                           "finite_abs": record(finite_abs), "analytic_error": err}
                    if complete:
                        row.update({"true_real": record(enlarge(total.real, er)),
                                    "true_imag": record(enlarge(total.imag, er)),
                                    "true_abs_upper": record(_minimum(arb(1), finite_abs+er))})
                    cf_rows.append(row)
                    if not complete:
                        break
        except Exception as exc:
            runtime_failure = {"phase": phase, "index": current_index,
                               "exception_type": type(exc).__name__, "message": str(exc)}
            if phase == "frequency_nodes" and "total" in locals():
                runtime_failure["current_partial_frequency"] = {
                    "u_exact": str(u), "gaussian_nodes_completed": visits,
                    "finite_real": record(total.real), "finite_imag": record(total.imag)}
        raw_complete = len(node_indices) == 2*N+1
        raw_error = _raw4_bounds(d0, step, N, strip)
        complete = raw_complete and len(cf_rows) == u_count and all(r["complete"] for r in cf_rows)
        elapsed = time.perf_counter()-start
        complete = complete and elapsed <= hard_seconds and runtime_failure is None
        result = {"schema": "r02_remainder_noise_sums_v1",
                  "status": "complete" if complete else ("runtime_failure" if runtime_failure else "budget_stop"),
                  "configuration": {"step_exact": str(step), "N": N,
                                    "limit_exact": str(N*step), "strip_exact": str(strip),
                                    "u_start_exact": str(u_start), "u_step_exact": str(u_step),
                                    "u_count": u_count, "u_end_exact": str(u_start+(u_count-1)*u_step),
                                    "precision_bits": bits, "python_flint_version": flint.__version__,
                                    "endpoint_weights": "full",
                                    "centered_L": "x/2-logcosh(x/2)+d0", "standardized_by": "sqrt(v)"},
                  "input_constants": {k: constants[k] for k in values},
                  "hard_seconds": hard_seconds, "elapsed_seconds": elapsed,
                  "timing_excludes_callers_disk_write": True,
                  "gaussian_node_indices_completed": node_indices,
                  "raw4_finite": record(raw4), "raw4_analytic_error": raw_error,
                  "cf_rows": cf_rows,
                  "work_counts": {"new_gaussian_nodes": len(node_indices),
                                  "new_logcosh_evaluations": len(node_indices),
                                  "new_complex_exp_nodes": complex_visits,
                                  "new_scalar_finite_sums_completed": int(raw_complete)+2*sum(r["complete"] for r in cf_rows),
                                  "new_fourier_integration_nodes": 0, "new_D_n_evaluations": 0,
                                  "new_MC_draws": 0, "new_NPZ_reads": 0}}
        if runtime_failure is not None:
            result["runtime_failure"] = runtime_failure
        if raw_complete:
            result["raw4_true"] = record(enlarge(raw4, from_record(raw_error["total"])))
            result["mu4_true"] = record(from_record(result["raw4_true"])/(v*v))
        if complete:
            result["composition"] = compose_saved(result)
        result["elapsed_seconds"] = time.perf_counter()-start
        result["timing_includes_saved_composition"] = True
        if result["status"] == "complete" and result["elapsed_seconds"] > hard_seconds:
            result["status"] = "budget_stop"
        return result
    finally:
        ctx.prec = old


def compose_saved(saved):
    """Reconstruct true fourth moment and all segment caps from saved sums."""
    if saved["status"] != "complete":
        raise ValueError("complete finite sums required")
    config = saved["configuration"]
    old = ctx.prec
    ctx.prec = config["precision_bits"]
    try:
        _check_version_bits(ctx.prec)
        values = _inputs(saved["input_constants"])
        step, strip = exact(config["step_exact"]), exact(config["strip_exact"])
        N, count = config["N"], config["u_count"]
        u0, du = exact(config["u_start_exact"]), exact(config["u_step_exact"])
        if saved["gaussian_node_indices_completed"] != list(range(-N, N+1)):
            raise ValueError("missing Gaussian nodes")
        if len(saved["cf_rows"]) != count:
            raise ValueError("missing frequency rows")
        error = _raw4_bounds(values["d0"], step, N, strip)
        raw4 = enlarge(from_record(saved["raw4_finite"]), from_record(error["total"]))
        mu4 = raw4/(values["v"]**2)
        caps, rows = [], []
        for k, row in enumerate(saved["cf_rows"]):
            if row["index"] != k or exact(row["u_exact"]) != u0+k*du or not row["complete"]:
                raise ValueError("frequency sequence mismatch")
            if row["gaussian_nodes_completed"] != 2*N+1:
                raise ValueError("partial CF row")
            z = acb(from_record(row["finite_real"]), from_record(row["finite_imag"]))
            er = from_record(_cf_error(u0+k*du, values["v"], step, N, strip)["total"])
            caps.append(_minimum(arb(1), abs(z)+er))
        for j in range(count-1):
            rho = _minimum(arb(1), _maximum(caps[j], caps[j+1])+aq(du/2))
            rows.append({"index": j, "u_left_exact": str(u0+j*du),
                         "u_right_exact": str(u0+(j+1)*du), "rho_upper": record(rho)})
        width = endpoint_fraction(_endpoint(mu4.upper()))-endpoint_fraction(_endpoint(mu4.lower()))
        return {"mu4_true": record(mu4), "raw4_true": record(raw4),
                "cf_abs_caps": [record(x) for x in caps], "segments": rows,
                "checks": {"mu4_positive": bool(mu4 > 0),
                           "mu4_width_below_1e_minus_8": width < Fraction(1,10**8),
                           "all_segment_rho_strictly_below_one": all(from_record(r["rho_upper"]) < 1 for r in rows)},
                "new_integral_nodes": 0, "new_D_n_evaluations": 0}
    finally:
        ctx.prec = old


# Positive envelopes are lists (coefficient, integer power, positive decay).
def _shift(terms, power=0, factor=None):
    factor = arb(1) if factor is None else factor
    return [(a*factor, p+power, c) for a, p, c in terms]


def _gauss_moment(power, decay, lower=None):
    if power < 0 or not isinstance(power, int) or not decay > 0:
        raise ValueError("nonnegative integer power and positive decay required")
    if lower is None:
        values = [arb.pi().sqrt()/(2*decay.sqrt()), 1/(2*decay)]
        for k in range(2, power+1):
            values.append((k-1)*values[k-2]/(2*decay))
    else:
        exponential = (-decay*lower*lower).exp()
        values = [arb.pi().sqrt()*(decay.sqrt()*lower).erfc()/(2*decay.sqrt()),
                  exponential/(2*decay)]
        for k in range(2, power+1):
            values.append((lower**(k-1)*exponential+(k-1)*values[k-2])/(2*decay))
    return values[power]


def _integrate(terms, lower=None):
    return sum((a*_gauss_moment(p,c,lower) for a,p,c in terms), arb(0))


def _square(terms, lower=None):
    return sum((a*b*_gauss_moment(p+q,c+d,lower)
                for a,p,c in terms for b,q,d in terms), arb(0))


def _tail_power_moment(p, k, n, B, cutoff, segments):
    """Integral of t^p rho(t)^k on [delta sqrt(n), infinity)."""
    if p < 0 or k <= p+1:
        raise ValueError("tail power not integrable")
    sn = arb(n).sqrt()
    total = arb(0)
    for row in segments:
        left, right = aq(row["u_left_exact"])*sn, aq(row["u_right_exact"])*sn
        rho = _hi(from_record(row["rho_upper"]))
        total += rho**k*(right**(p+1)-left**(p+1))/(p+1)
    # n^((p+1)/2) is formed as sqrt(n)^(p+1), no floating exponent.
    total += sn**(p+1)*B**k*aq(cutoff)**(p-k+1)/(k-p-1)
    return total


def _cf_square(terms, n, B, cutoff, segments):
    # CF terms: (coefficient, t power, exponent of the CF modulus cap).
    return sum((a*b*_tail_power_moment(p+q,k+l,n,B,cutoff,segments)
                for a,p,k in terms for b,q,l in terms), arb(0))


def compose_bounds(saved, n_values=(64,256,1024,4096)):
    """Only saved sums and closed scalar arithmetic; zero new integrations.

    Returned dyadic intervals enclose the *reported upper budgets*. Use their
    upper endpoint as a bound, not their lower endpoint as an actual error.
    """
    config = saved["configuration"]
    old = ctx.prec
    ctx.prec = config["precision_bits"]
    try:
        base = compose_saved(saved)
        if not all(base["checks"].values()):
            raise ValueError("saved moment/CF qualification failed")
        delta = exact(config["u_start_exact"])
        cutoff = exact(config["u_end_exact"])
        if not 0 < delta < cutoff or len(base["segments"]) < 1:
            raise ValueError("nonempty frequency interval required")
        values = _inputs(saved["input_constants"])
        v, mu4, dd = values["v"], _hi(from_record(base["mu4_true"])), aq(delta)
        mu3 = values["kappa3_L"]/(v*v.sqrt())
        alpha = _hi(abs(mu3))/6
        q = arb(1)/2+alpha*dd+mu4*dd*dd/24
        zz = dd*dd*q
        if not zz < 1:
            raise ValueError("low-frequency logarithm gate failed")
        C = _hi(mu4/24+q*q/(2*(1-zz)))
        D = _hi((mu4/6+(1+3*alpha*dd)*q)/(1-zz))
        c = _lo(arb(1)/2-C*dd*dd)
        cm = _lo((1-(2*mu4+6)*dd*dd/24)/2)
        if not c > 0 or not cm > 0:
            raise ValueError("low-frequency positive Gaussian decay required")
        # Select only lower endpoints: upward rounding of a decay rate could
        # shrink the Gaussian envelope and invalidate an upper error bound.
        decay = aq(max(endpoint_fraction(_endpoint(x.lower())) for x in (c,cm)))
        B = _hi(2*_hi(v).sqrt()*(1+(arb(1)/2).exp())/(2*arb.pi()).sqrt())
        if not aq(cutoff) > B:
            raise ValueError("high-frequency cutoff must exceed BV cap")
        rows = []
        for n in n_values:
            if isinstance(n,bool) or not isinstance(n,int) or n < 8:
                raise ValueError("integer n>=8 required")
            sn, half = arb(n).sqrt(), arb(1)/2
            gum = _hi(arb.pi()**2/(12*n*_lo(v)))
            A = [(C/n,4,c),(alpha*alpha/(2*n),6,half)]
            Ap = (_shift(A,1)+[(3*alpha*alpha/n,5,half),
                   (3*alpha*C/(n*sn),6,c),(D/n,3,c)])
            Edge = [(arb(1),0,half),(alpha/sn,3,half)]
            Edgep = _shift(Edge,1)+[(3*alpha/sn,2,half)]
            Ag = A+_shift(Edge,2,gum)
            Agp = Ap+[(2*gum,1,decay)]+_shift(Edgep,2,gum)
            t0 = dd*sn
            Pmoment = lambda p,k=n: _tail_power_moment(p,k,n,B,cutoff,base["segments"])
            direct = {}
            error_inf = {}
            for j in range(3):
                low_direct = _gauss_moment(j,decay)
                direct[str(j)] = record((low_direct+Pmoment(j))/arb.pi())
                error_inf[str(j)] = record((_integrate(_shift(Ag,j))+Pmoment(j)
                                           +_integrate(_shift(Edge,j),t0))/arb.pi())
            # CDF CF tail /t: t >= t0 allows P/t <= P/t0; Gaussian tail integrates explicitly.
            # The Edge constant term /t is bounded by 1/t0; cubic term remains a polynomial.
            cdf_tail = Pmoment(0)/t0+_gauss_moment(0,half,t0)/t0+alpha/sn*_gauss_moment(2,half,t0)
            cdf = (_integrate(_shift(Ag,-1))+cdf_tail)/arb.pi()
            pnorm = _cf_square([(arb(1),0,n)],n,B,cutoff,base["segments"]).sqrt()
            enorm = _square(Edge,t0).sqrt()
            tail0_sq = (pnorm+enorm)**2
            L2 = ((_square(Ag)+tail0_sq)/arb.pi()).sqrt()
            # r1 Sobolev vector: (t R, R+t R').
            cf_first = [(arb(1),1,n)]
            cf_second = [(arb(1),0,n),(sn,1,n-1),(2*gum,2,n)]
            cf_vector = (_cf_square(cf_first,n,B,cutoff,base["segments"])
                         +_cf_square(cf_second,n,B,cutoff,base["segments"])).sqrt()
            eg_first, eg_second = _shift(Edge,1), Edge+_shift(Edgep,1)
            eg_vector = (_square(eg_first,t0)+_square(eg_second,t0)).sqrt()
            tail1_sq = (cf_vector+eg_vector)**2
            low1_sq = _square(_shift(Ag,1))+_square(Ag+_shift(Agp,1))
            L1prime = (low1_sq+tail1_sq).sqrt()
            rows.append({"n": n,"error_density_derivative_Linf":error_inf,
                         "error_CDF_Linf":record(cdf),"error_density_L2":record(L2),
                         "error_density_prime_L1":record(L1prime),
                         "direct_density_derivative_Linf":direct,
                         "diagnostics":{"low_error_r0_integral":record(_integrate(Ag)/arb.pi()),
                           "tail_error_r0_integral":record((Pmoment(0)+_integrate(Edge,t0))/arb.pi()),
                           "tail_r0_L2_squared_positive_frequency":record(tail0_sq),
                           "tail_r1_Sobolev_squared_positive_frequency":record(tail1_sq),
                           "low_r1_Sobolev_squared_positive_frequency":record(low1_sq)},
                         "all_budgets_are_upper_bounds_not_actual_error_lower_bounds":True})
        return {"schema":"r02_remainder_noise_bounds_v1", "saved_composition":base,
                "low_frequency":{"delta_exact":str(delta),"mu3":record(mu3),
                    "mu4_upper":record(mu4),"alpha_absolute_upper":record(alpha),
                    "z_delta":record(zz),"C_delta":record(C),"D_delta":record(D),
                    "c_log_lower":record(c),"c_mod_lower":record(cm),
                    "direct_decay_lower":record(decay)},
                "BV_constant_upper":record(B),"high_frequency_cutoff_exact":str(cutoff),
                "rows":rows,"new_integral_nodes":0,"new_fourier_integration_nodes":0,
                "new_D_n_evaluations":0,"finite_noise_bounds_only":True,
                "finite_n_gradient_sign_certified":False}
    finally:
        ctx.prec = old

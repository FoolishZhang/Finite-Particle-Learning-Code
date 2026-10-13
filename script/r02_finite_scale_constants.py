"""Frozen-input Gaussian constants for R02, with rigorous strip/lattice errors.

Pure functions only: no file I/O, RNG, optimization, prior arrays, or finite-n
score calculation.  Production callers must freeze source and inputs first.
The only nested integral is one conditional Gaussian integral followed by one
outer Gaussian integral.  Finite sums and analytic error bounds remain separate.
"""
from __future__ import annotations

from fractions import Fraction
import json
import math
import time

import flint
from flint import arb, ctx


VERSION = "0.8.0"
INNER_NAMES = ("T", "B", "C", "V")
OUTER_NAMES = ("OT2", "OTC", "OT2B", "OV2", "OB2")


def exact(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError("explicit rational required; floats and bools rejected")
    return Fraction(value)


def aq(value):
    value = exact(value)
    return arb(value.numerator) / value.denominator


def _endpoint(value):
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent_of_two": int(exponent)}


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
    m, e = m.numerator, e.numerator
    return Fraction(m * 2 ** max(e, 0), 2 ** max(-e, 0))


def from_record(data):
    lo = endpoint_fraction(data["lower_exact_dyadic"])
    hi = endpoint_fraction(data["upper_exact_dyadic"])
    if lo > hi:
        raise ValueError("inverted interval")
    return aq(lo).union(aq(hi))


def enlarge(value, error):
    if not isinstance(error, arb):
        error = aq(error)
    if not error.is_finite() or not error >= 0:
        raise ValueError("certified nonnegative finite error required")
    return value + arb(0, error.upper())


def _params(rho, K, step, N, strip, bits):
    rho, step, strip = map(exact, (rho, step, strip))
    if not 0 < rho < 1:
        raise ValueError("this factorization requires 0 < rho < 1")
    if isinstance(K, bool) or not isinstance(K, int) or K < 2:
        raise ValueError("integer K >= 2 required")
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError("positive integer N required")
    if step <= 0 or N * step < 2:
        raise ValueError("positive step and L=N*step >= 2 required")
    if not 0 < strip <= 1:
        raise ValueError("proved strip range is 0 < delta <= 1")
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 64:
        raise ValueError("integer precision >= 64 required")
    if flint.__version__ != VERSION:
        raise RuntimeError("requires python-flint 0.8.0")
    return rho, step, strip


def _config(rho, K, step, N, strip, bits):
    return {"m": 3, "teacher": "independent zero-mean signs; eight labels uniform",
            "rho_exact": str(rho), "K": K, "step_exact": str(step), "N": N,
            "limit_exact": str(step * N), "strip_exact": str(strip),
            "precision_bits": bits, "python_flint_version": flint.__version__,
            "endpoint_weights": "full", "node_count": 2 * N + 1,
            "candidate": "Xi=sqrt(rho)*U+sqrt(1-rho)*Vi"}


def _bound_values(rho, K, step, N, strip):
    """All arithmetic at caller precision; no target grid evaluation."""
    pi = arb.pi()
    alpha, beta = aq(rho).sqrt(), aq(1-rho).sqrt()
    delta, h, L = aq(strip), aq(step), aq(N*step)
    mu1 = (2/pi).sqrt()
    moments = [arb(1), mu1, arb(1), 2*mu1]
    shift = (delta*delta/2).exp()
    denom = (2*pi*delta/h).exp()-1
    phiL = (-L*L/2).exp()/(2*pi).sqrt()
    P = [2*phiL/L, 2*phiL, 2*phiL*(L+1/L), 2*phiL*(L*L+2)]

    def c(y):
        return -(y/2).cos().log()+y/2

    def error(M, tail):
        alias = 2*M/denom
        return {"boundary_single_side": M, "alias": alias, "tail": tail,
                "total": alias+tail}

    univariate = {}
    for k in (1, 2, 3):
        M = shift*sum((math.comb(k,j)*moments[j]*(aq(1)/2)**j*c(delta)**(k-j)
                       for j in range(k+1)), arb(0))
        univariate["d"+str(k)] = error(M, P[k]/(2**k))
    univariate["t2"] = error(shift, P[0])
    univariate["gaussian_mass"] = error(shift, P[0])
    ac = alpha/2
    bc = beta*mu1/2+c(alpha*delta)
    br = beta*mu1/2
    outer = {
        "OT2": error(shift, P[0]),
        "OV2": error(shift, P[0]),
        "OTC": error(shift*(ac*mu1+bc), ac*P[1]+br*P[0]),
        "OT2B": error(shift*(ac*mu1+bc), ac*P[1]+br*P[0]),
        "OB2": error(shift*(ac*ac+2*ac*bc*mu1+bc*bc),
                      ac*ac*P[2]+2*ac*br*P[1]+br*br*P[0]),
    }
    CF = 1+delta*shift/(2*pi).sqrt()
    common = K*(K-1)*(delta*delta).exp()*CF**(K-2)
    max_tail_common = K*(K-1)*(-L*L).exp()/(2*pi)
    maximum = {
        "J_K": error(common/(2*pi.sqrt()), max_tail_common/L),
        "B_K": error(common*(1/(2*pi)+delta/(2*pi.sqrt())), max_tail_common),
    }
    return {"alpha": alpha, "beta": beta, "mu1": mu1, "delta": delta,
            "shift": shift, "alias_denominator": denom, "P": P,
            "inner_c": c(beta*delta), "CF": CF,
            "univariate": univariate, "outer": outer, "maximum": maximum}


def _inner_errors(bounds, abs_u):
    a, b = bounds["alpha"], bounds["beta"]
    P, shift, denom = bounds["P"], bounds["shift"], bounds["alias_denominator"]
    tv_M = shift
    bc_M = shift*(a*abs_u/2+b*bounds["mu1"]/2+bounds["inner_c"])
    tv_tail = P[0]
    bc_tail = a*abs_u*P[0]/2+b*P[1]/2
    tv = 2*tv_M/denom+tv_tail
    bc = 2*bc_M/denom+bc_tail
    return {"T": tv, "V": tv, "B": bc, "C": bc}


def _record_tree(data):
    if isinstance(data, arb):
        return record(data)
    if isinstance(data, dict):
        return {k: _record_tree(v) for k,v in data.items()}
    if isinstance(data, list):
        return [_record_tree(v) for v in data]
    return data


def analytic_bounds(rho="9/20", K=100, step="1/16", N=160, strip="3/4", bits=192):
    """Return rigorous strip/tail balls without evaluating an integral grid."""
    rho, step, strip = _params(rho,K,step,N,strip,bits)
    old = ctx.prec
    ctx.prec = bits
    try:
        return {"schema": "r02_constant_analytic_bounds_v1",
                "configuration": _config(rho,K,step,N,strip,bits),
                "bounds": _record_tree(_bound_values(rho,K,step,N,strip)),
                "new_integral_nodes": 0}
    finally:
        ctx.prec = old


def _products(inner):
    T, B, C, V = (inner[k] for k in INNER_NAMES)
    return {"OT2": T*T, "OTC": T*C, "OT2B": T*T*B,
            "OV2": V*V, "OB2": B*B}


def _interval_contains(outer, inner):
    return (endpoint_fraction(outer["lower_exact_dyadic"]) <=
            endpoint_fraction(inner["lower_exact_dyadic"]) and
            endpoint_fraction(outer["upper_exact_dyadic"]) >=
            endpoint_fraction(inner["upper_exact_dyadic"]))


def _overlap(a, b):
    return (endpoint_fraction(a["lower_exact_dyadic"]) <=
            endpoint_fraction(b["upper_exact_dyadic"]) and
            endpoint_fraction(b["lower_exact_dyadic"]) <=
            endpoint_fraction(a["upper_exact_dyadic"]))


def compose_saved(result):
    """Reconstruct constants from saved finite sums, without target integrands.

    Rebuilds the 321-term outer weighted sums from serialized inner finite balls
    plus analytic errors.  This is saved-evidence arithmetic, not an independent
    replay of any transcendental quadrature.  Saved weights are bound by the
    frozen producer; no tanh/logcosh/erf target evaluations occur here.
    """
    if result.get("status") != "complete":
        raise ValueError("partial target sums cannot be composed into a certificate")
    c = result["configuration"]
    rho, step, strip = _params(c["rho_exact"],c["K"],c["step_exact"],c["N"],
                               c["strip_exact"],c["precision_bits"])
    old = ctx.prec
    ctx.prec = c["precision_bits"]
    try:
        bounds = _bound_values(rho,c["K"],step,c["N"],strip)
        rows = result["inner_rows"]
        if len(rows) != 2*c["N"]+1:
            raise ValueError("incomplete nested outer rows")
        outer = {k: arb(0) for k in OUTER_NAMES}
        inner_true_overlap = True
        for j,row in zip(range(-c["N"],c["N"]+1), rows):
            if row["j"] != j or exact(row["u_exact"]) != j*step:
                raise ValueError("wrong or duplicate outer node")
            if row["inner_nodes_completed"] != 2*c["N"]+1:
                raise ValueError("incomplete conditional finite sum")
            errors = _inner_errors(bounds, aq(abs(j*step)))
            inner = {k: enlarge(from_record(row["finite"][k]),errors[k])
                     for k in INNER_NAMES}
            for k in INNER_NAMES:
                inner_true_overlap &= _overlap(record(inner[k]),row["true"][k])
            weight = from_record(row["outer_weight"])
            if not weight > 0:
                raise ArithmeticError("nonpositive saved Gaussian weight")
            for k,v in _products(inner).items():
                outer[k] += weight*v
        checks = {"all_inner_true_intervals_overlap_saved": inner_true_overlap,
                  "all_reconstructed_outer_finite_intervals_overlap_saved": all(
                      _overlap(record(outer[k]),result["outer_finite_with_inner_error"][k])
                      for k in OUTER_NAMES)}
        if not all(checks.values()):
            raise ArithmeticError("saved nested arithmetic mismatch")
        uni = {k: enlarge(from_record(v),bounds["univariate"][k]["total"])
               for k,v in result["univariate_finite"].items()}
        outer_true = {k: enlarge(v,bounds["outer"][k]["total"])
                      for k,v in outer.items()}
        maximum = {k: enlarge(from_record(v),bounds["maximum"][k]["total"])
                   for k,v in result["maximum_finite"].items()}
        d0, d2, d3, t2 = (uni[k] for k in ("d1","d2","d3","t2"))
        v = aq("1/4")+d2-d0*d0
        s0 = 1-t2
        kappa3 = -3*s0/16-(d3-3*d0*d2+2*d0*d0*d0)
        spin = outer_true["OT2"]
        mH = spin/4
        cbar = -aq(rho)*s0/8-(2*outer_true["OTC"]+outer_true["OT2B"]-3*d0*spin)/4
        VH = (2*t2+outer_true["OV2"]-spin*spin)/16
        VA = aq("3/4")+3*(v-aq("1/4"))+6*(outer_true["OB2"]-d0*d0)
        constants = {"d0":d0, "v":v, "s0":s0, "kappa3_L":kappa3,
                     "kappa_spin":spin,"m_H":mH,"cbar":cbar,"V_H":VH,"V_A":VA,
                     **maximum}
        checks["positive_variance_v"] = bool(v > 0)
        checks["positive_VH_VA"] = bool(VH > 0 and VA > 0)
        if checks["positive_variance_v"]:
            constants["second_order_coefficient"] = maximum["B_K"]/v*(mH*kappa3/(3*v)-cbar)
        if checks["positive_variance_v"] and checks["positive_VH_VA"]:
            constants["marked_budget_numerator"] = c["K"]/(2*arb.pi()*arb(1).exp()).sqrt()*(VH*VA).sqrt()/v
        widths = {k: endpoint_fraction(record(x)["upper_exact_dyadic"])-
                     endpoint_fraction(record(x)["lower_exact_dyadic"])
                  for k,x in constants.items()}
        # The deliberately tiny manufactured grid can fail this qualification.
        checks["all_constant_widths_below_1e_minus_8"] = all(w < Fraction(1,10**8) for w in widths.values())
        return {"schema":"r02_constant_composition_v1","configuration":c,
                "univariate_true":_record_tree(uni),"outer_true":_record_tree(outer_true),
                "reconstructed_outer_finite_with_inner_error":_record_tree(outer),
                "constants":_record_tree(constants),"checks":checks,
                "widths_exact":{k:str(w) for k,w in widths.items()},
                "target_integrand_evaluations":0,"new_integral_nodes":0,
                "saved_inner_rows_used":len(rows),
                "finite_n_D_computed":False,"finite_n_remainder_certified":False}
    finally:
        ctx.prec = old


def compute_constants(rho="9/20", K=100, step="1/16", N=160, strip="3/4", bits=192,
                      hard_seconds=600):
    """Single bounded execution; a stopped result contains prefixes, no certificate.

    Budget includes all target nodes, interval composition and a JSON serialization
    validation pass.  The returned timing/status fields incur only final bookkeeping
    after that pass.  Caller's disk writes are outside this pure function's budget.
    """
    rho, step, strip = _params(rho,K,step,N,strip,bits)
    if isinstance(hard_seconds,bool) or not isinstance(hard_seconds,(int,float)) or not math.isfinite(hard_seconds) or hard_seconds < 0:
        raise ValueError("finite nonnegative wall budget required")
    old = ctx.prec
    ctx.prec = bits
    started = time.monotonic()
    try:
        config = _config(rho,K,step,N,strip,bits)
        bounds = _bound_values(rho,K,step,N,strip)
        uni = {k:arb(0) for k in ("d1","d2","d3","t2","gaussian_mass")}
        maximum = {k:arb(0) for k in ("J_K","B_K")}
        outer = {k:arb(0) for k in OUTER_NAMES}
        rows, nodes, uni_indices = [], [], []
        counts = {"gaussian_coordinate_evaluations":0,"univariate_nodes":0,
                  "normal_cdf_evaluations":0,"nested_nodes":0,
                  "tanh_evaluations":0,"logcosh_evaluations":0,
                  "outer_nodes_completed":0,"finite_scalar_sums_completed":0,
                  "new_D_n_scores":0,"new_MC_draws":0,"new_NPZ_reads":0,
                  "optimizer_steps":0,"new_teacher_PMFs":0}
        def expired():
            return time.monotonic()-started >= hard_seconds
        # A shared one-dimensional bank is also the deterministic inner/outer mesh.
        for j in range(-N,N+1):
            if expired():
                break
            x = aq(j*step)
            phi = (-x*x/2).exp()/(2*arb.pi()).sqrt()
            weight = aq(step)*phi
            t, d = (x/2).tanh(), (x/2).cosh().log()
            Phi = (1+(x/arb(2).sqrt()).erf())/2
            uni["d1"] += weight*d
            uni["d2"] += weight*d*d
            uni["d3"] += weight*d*d*d
            uni["t2"] += weight*t*t
            uni["gaussian_mass"] += weight
            max_weight = aq(step)*K*(K-1)*phi*phi*Phi**(K-2)
            maximum["J_K"] += max_weight
            maximum["B_K"] += x*max_weight
            nodes.append((j,x,weight))
            uni_indices.append(j)
            counts["gaussian_coordinate_evaluations"] += 1
            counts["univariate_nodes"] += 1
            counts["normal_cdf_evaluations"] += 1
            counts["tanh_evaluations"] += 1
            counts["logcosh_evaluations"] += 1
        if len(nodes)==2*N+1:
            for j,u,wu in nodes:
                if expired():
                    break
                inner = {k:arb(0) for k in INNER_NAMES}
                inner_count = 0
                for _,z,wz in nodes:
                    if expired():
                        break
                    x = bounds["alpha"]*u+bounds["beta"]*z
                    t,d = (x/2).tanh(),(x/2).cosh().log()
                    inner["T"] += wz*t
                    inner["B"] += wz*d
                    inner["C"] += wz*t*d
                    inner["V"] += wz*t*t
                    inner_count += 1
                    counts["nested_nodes"] += 1
                    counts["tanh_evaluations"] += 1
                    counts["logcosh_evaluations"] += 1
                row = {"j":j,"u_exact":str(j*step),"outer_weight":record(wu),
                       "inner_nodes_completed":inner_count,"finite":_record_tree(inner)}
                if inner_count == 2*N+1:
                    errors = _inner_errors(bounds,aq(abs(j*step)))
                    true = {k:enlarge(inner[k],errors[k]) for k in INNER_NAMES}
                    row["error"] = _record_tree(errors)
                    row["true"] = _record_tree(true)
                    for k,vv in _products(true).items():
                        outer[k] += wu*vv
                    counts["outer_nodes_completed"] += 1
                rows.append(row)
                if inner_count != 2*N+1:
                    break
        complete = counts["outer_nodes_completed"]==2*N+1 and len(nodes)==2*N+1
        if complete:
            # 5 single moments, 2 max moments, four sums at each conditional
            # node, and 5 outer sums. These are finite sums, not independent laws.
            counts["finite_scalar_sums_completed"] = 5+2+4*(2*N+1)+5
        result = {"schema":"r02_finite_scale_constant_sums_v1",
                  "status":"complete" if complete else "budget_stop",
                  "configuration":config,"hard_seconds":hard_seconds,
                  "univariate_node_indices_completed":uni_indices,
                  "univariate_finite":_record_tree(uni),"maximum_finite":_record_tree(maximum),
                  "inner_rows":rows,"outer_finite_with_inner_error":_record_tree(outer),
                  "analytic_bounds":_record_tree(bounds),"counts":counts,
                  "finite_n_D_computed":False,"finite_n_remainder_certified":False}
        if complete and not expired():
            result["composition"] = compose_saved(result)
        json.dumps(result,ensure_ascii=False,sort_keys=True)
        elapsed = time.monotonic()-started
        if elapsed > hard_seconds or "composition" not in result:
            result["status"] = "budget_stop"
            result.pop("composition",None)
        result["elapsed_seconds_including_serialization_check"] = elapsed
        result["timing_excludes_callers_disk_write"] = True
        return result
    finally:
        ctx.prec = old

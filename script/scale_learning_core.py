"""Pure numerical kernels for the separately frozen scalar-scale learning study.

No file writes, random draws, mutable scientific configuration, or old-package
imports.  Callers own freeze, stopping, fitting, and reference-sensitivity rules.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import (
    eval_hermitenorm, expit, gammaln, ndtr, roots_hermitenorm,
)


class NumericalReferenceError(RuntimeError):
    """A deterministic integrator did not meet its declared stopping contract."""


def numerics_contract() -> dict[str, Any]:
    """Return a fresh description; callers may serialize it in their new freeze."""
    return {
        "reference_variants": ["GH256", "GH512", "adaptive"],
        "primary_reference": "adaptive",
        "GH_orders": [256, 512],
        "inner_standard_limit": 12.0,
        "inner_epsabs": 2e-13,
        "inner_epsrel": 2e-13,
        "inner_limit": 300,
        "outer_standard_limit": 9.0,
        "outer_epsabs": 2e-12,
        "outer_epsrel": 2e-12,
        "outer_limit": 4000,
        "outer_error_norm": "PMF max norm",
        "mass_gate": 1e-12,
        "identity_gate": 1e-10,
        "variant_L1_gate": 1e-8,
        "variant_function_difference_gate": 1e-5,
        "outer_error_estimate_gate": 1e-10,
        "inner_propagated_error_estimate_gate": 1e-9,
        "moment_gate": 1e-8,
        "hermite_odd_orders": [1, 3, 5, 7, 9, 11, 13],
        "hermite_standard_limit": 12.0,
        "hermite_epsabs": 2e-13,
        "hermite_epsrel": 2e-13,
        "hermite_limit": 300,
        "dtype": "float64",
        "no_probability_floor": True,
        "no_negative_probability_clipping": True,
        "qualified_is_not_a_rigorous_all_error_certificate": True,
    }


def _domain(d, scales, c):
    if isinstance(d, bool) or int(d) != d or d < 2 or d % 2:
        raise ValueError("d must be an even integer >= 2")
    scales = np.asarray(scales, dtype=np.float64)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("scales must contain two finite nonnegative values")
    if not math.isfinite(c) or not 0 <= c < 1:
        raise ValueError("c must satisfy 0 <= c < 1")
    return int(d), scales, float(c)


def geometry(d):
    """Old seven-field contract: na, nb, total, logmult, logchoose, labels, indicator.

    For d=16: na/nb/total/logmult/logchoose [81], labels [81,16],
    indicator [81,17].  labels contains one canonical pattern per count cell;
    it is not all 2**d patterns of a fixed raw particle bank.
    """
    d, _, _ = _domain(d, [0.0, 0.0], 0.0)
    m = d // 2
    a, b = np.meshgrid(np.arange(m+1), np.arange(m+1), indexing="ij")
    a, b = a.ravel(), b.ravel()
    n = a+b
    lg = (gammaln(m+1)-gammaln(a+1)-gammaln(m-a+1)
          +gammaln(m+1)-gammaln(b+1)-gammaln(m-b+1))
    lt = gammaln(d+1)-gammaln(n+1)-gammaln(d-n+1)
    labels = np.concatenate(
        [np.arange(m)[None, :] < a[:, None],
         np.arange(m)[None, :] < b[:, None]], axis=1).astype(np.float64)
    return a, b, n, lg, lt, labels, np.eye(d+1, dtype=np.float64)[n]


def eta_from_z(z, scales, c=.9):
    """Map supplied standard Gaussian coordinates [...,d] to two-group logits."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim < 1 or not np.isfinite(z).all():
        raise ValueError("z must be a finite array with a final coordinate axis")
    d, scales, c = _domain(z.shape[-1], scales, c)
    rho = c/(d-1)
    a = math.sqrt(1+rho)
    u = a*z-(a-math.sqrt(1-c))*z.mean(axis=-1, keepdims=True)
    return u*np.repeat(scales, d//2)


def pb(eta):
    """Poisson-binomial probabilities [particles,coordinates+1], no clipping."""
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 2 or not np.isfinite(eta).all():
        raise ValueError("eta must have shape [particles, coordinates] and be finite")
    p, q = expit(eta), expit(-eta)
    mass = np.zeros((len(eta), eta.shape[1]+1), dtype=np.float64)
    mass[:, 0] = 1
    for j in range(eta.shape[1]):
        mass[:, 1:j+2] = mass[:, 1:j+2]*q[:, j, None]+mass[:, :j+1]*p[:, j, None]
        mass[:, 0] *= q[:, j]
    return mass


def joint_mass(eta):
    """Conditional two-group count PMF [particles,(d/2+1)**2]."""
    eta = np.asarray(eta, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[1] < 2 or eta.shape[1] % 2:
        raise ValueError("eta must have shape [particles, even d >= 2]")
    m = eta.shape[1]//2
    aa, bb = pb(eta[:, :m]), pb(eta[:, m:])
    return (aa[:, :, None]*bb[:, None, :]).reshape(len(eta), -1)


def functionals(p, d):
    """Own-distribution H(P), D(P||S(P)), total PMF, symmetrized group PMF.

    These are not cross-risks of a different teacher against this candidate.
    """
    _, _, n, lg, lt, _, ind = geometry(d)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != n.shape or not np.isfinite(p).all() or not np.all(p > 0):
        raise ValueError("reference PMF must be strictly positive and have the count-grid shape")
    pn = p @ ind
    ps = pn[n]*np.exp(lg-lt)
    return -float(p @ (np.log(p)-lg)), float(p @ np.log(p/ps)), pn, ps


def _pairwise_max(values, l1=False):
    values = np.asarray(values)
    return max(float(np.abs(values[i]-values[j]).sum() if l1 else
                     np.max(np.abs(values[i]-values[j])))
               for i in range(len(values)) for j in range(i))


def independent_moment_reference(d, scales, c=.9):
    """Mehler/Hermite pair moments, computed independently of Fourier factors."""
    d, scales, c = _domain(d, scales, c)
    policy = numerics_contract()
    sources = {}
    orders = policy["hermite_odd_orders"]
    for scale in dict.fromkeys(float(s) for s in scales):
        coefficients, estimates = [], []
        neval = 0
        for order in orders:
            if scale == 0:
                coefficients.append(0.0)
                estimates.append(0.0)
                continue
            denominator = math.sqrt(math.factorial(order))*math.sqrt(2*math.pi)

            def integrand(z):
                return (math.tanh(scale*z/2)*float(eval_hermitenorm(order, z))
                        *math.exp(-z*z/2)/denominator)

            ans = quad(integrand, 0., policy["hermite_standard_limit"],
                       epsabs=policy["hermite_epsabs"], epsrel=policy["hermite_epsrel"],
                       limit=policy["hermite_limit"], full_output=1)
            if len(ans) != 3:
                raise NumericalReferenceError("independent Hermite quadrature failed: "+str(ans[3]))
            coefficients.append(2*ans[0])
            estimates.append(2*ans[1])
            neval += int(ans[2]["neval"])
        sources[scale] = dict(scale=scale, coefficients=coefficients,
                              quadrature_error_estimates=estimates, evaluations=neval)
    rho = c/(d-1)
    powers = rho**np.asarray(orders)
    remainder = rho**15
    coefficient_tail = math.sqrt(2*float(ndtr(-policy["hermite_standard_limit"])))
    magnitudes, errors = [], []
    for s, tscale in ((scales[0], scales[0]), (scales[1], scales[1]),
                     (scales[0], scales[1])):
        if s == 0 or tscale == 0 or c == 0:
            magnitudes.append(0.0)
            errors.append(0.0)
            continue
        aa = np.array(sources[float(s)]["coefficients"])
        bb = np.array(sources[float(tscale)]["coefficients"])
        ea = np.array(sources[float(s)]["quadrature_error_estimates"])+coefficient_tail
        eb = np.array(sources[float(tscale)]["quadrature_error_estimates"])+coefficient_tail
        magnitudes.append(float((aa*bb) @ powers))
        errors.append(float((abs(aa)*eb+abs(bb)*ea+ea*eb) @ powers+remainder))
    m = d//2
    aa, bb, ab = magnitudes
    expected = [m-m*(m-1)*aa, m-m*(m-1)*bb, -m*m*ab]
    moment_errors = np.array(errors)*[m*(m-1), m*(m-1), m*m]
    return dict(expected_spin_moments=expected,
                pair_magnitudes_AA_BB_AB=magnitudes,
                pair_error_estimates_plus_analytic_tail=errors,
                moment_error_estimates_plus_tail=moment_errors.tolist(),
                mehler_analytic_remainder_bound=remainder,
                coefficient_analytic_tail_bound=coefficient_tail,
                coefficient_sources=list(sources.values()),
                quadrature_estimates_not_rigorous_certificates=True)


def _independent_reference(d, scales, c):
    """Exact mathematical independent case, represented in float64."""
    na, nb, _, _, _, _, _ = geometry(d)
    m = d//2
    denominator = float(2**d)
    p = np.array([math.comb(m, int(a))*math.comb(m, int(b))/denominator
                  for a, b in zip(na, nb)], dtype=np.float64)
    pn = np.array([math.comb(d, j)/denominator for j in range(d+1)], dtype=np.float64)
    moment_ref = dict(
        expected_spin_moments=[float(m), float(m), 0.0],
        pair_magnitudes_AA_BB_AB=[0.0, 0.0, 0.0],
        pair_error_estimates_plus_analytic_tail=[0.0, 0.0, 0.0],
        moment_error_estimates_plus_tail=[0.0, 0.0, 0.0],
        mehler_analytic_remainder_bound=0.0,
        coefficient_analytic_tail_bound=0.0,
        coefficient_sources=[],
        analytic_independence=True,
        quadrature_estimates_not_rigorous_certificates=True)
    variants = np.repeat(p[None, :], 3, axis=0)
    return dict(d=d, scales=scales.tolist(), c=c, mean=p, variants=variants,
                raw_variants=variants.copy(), H=d*math.log(2), model_KL_single=0.0,
                total_p=pn, symmetrized_group_p=p.copy(),
                variant_names=["GH256", "GH512", "adaptive"], primary="adaptive",
                checks=[dict(name=name, analytic_bypass=True, raw_mass_error=abs(float(p.sum())-1),
                             raw_min_probability=float(p.min()))
                        for name in ("GH256", "GH512", "adaptive")],
                stats=dict(inner_calls=0, inner_evaluations=0, inner_max_error_estimate=0.,
                           max_factor_abs=.5, max_integrand_mass_identity_error=0.),
                independent_Hermite_moments=moment_ref,
                variant_functionals=np.array([[d*math.log(2), 0.]]*3),
                variant_spin_moments=np.array([[m, m, 0.]]*3),
                independent_moment_abs_errors=np.zeros((3, 3)),
                independent_moment_max_error=0., pairwise_L1=0.,
                pairwise_max_function_difference=0.,
                inner_u_analytic_tail_bound=0., outer_exact_L1_analytic_tail_bound=0.,
                inner_L1_error_estimate_plus_tail=0., approximate_outer_L1_tail_estimate=0.,
                probability_finite=True, raw_mass_pass=True, probability_strict_positive=True,
                deterministic_variants_are_not_independent_replicates=True,
                reference_standard_error=None, analytic_independence=True,
                numerical_qualification_not_rigorous_all_error_certificate=True,
                complex_cancellation_not_positive_product_mixture=True,
                qualified=True, numerical_pass=True)


def reference(d, scales, c=.9):
    """Three deterministic Fourier PMFs and independent pair-moment qualification.

    Never writes files or samples labels/banks. An unqualified raw PMF is returned
    without clipping; callers must save its diagnostics and stop. A quad stopping
    failure raises NumericalReferenceError and is not repaired automatically.
    """
    d, scales, c = _domain(d, scales, c)
    if c == 0 or np.all(scales == 0):
        return _independent_reference(d, scales, c)
    policy = numerics_contract()
    rho = c/(d-1)
    a = 1+rho
    tau = rho/(a*(1-c))
    normalizer = math.sqrt(a/(1-c))
    na, nb, _, lg, _, _, _ = geometry(d)
    m = d//2
    multiplicity = np.exp(lg)
    invsqrt2pi = 1/math.sqrt(2*math.pi)
    inner_tail_u = float(ndtr(-policy["inner_standard_limit"]))
    outer_tail = normalizer*2*float(ndtr(-policy["outer_standard_limit"]))
    cache = {}
    stats = dict(inner_calls=0, inner_evaluations=0, inner_max_error_estimate=0.,
                 max_factor_abs=0., max_integrand_mass_identity_error=0.)

    def uvalue(scale, w):
        key = (float(scale), float(abs(w)))
        if key not in cache:
            ws = key[1]
            if scale == 0 or ws == 0:
                integral, error, neval = 0.0, 0.0, 0
            else:
                def integrand(x):
                    return (math.tanh(scale*math.sqrt(a)*x/2)
                            *math.sin(ws*math.sqrt(a)*x)*math.exp(-x*x/2)*invsqrt2pi)

                ans = quad(integrand, 0., policy["inner_standard_limit"],
                           epsabs=policy["inner_epsabs"], epsrel=policy["inner_epsrel"],
                           limit=policy["inner_limit"], full_output=1)
                if len(ans) != 3:
                    raise NumericalReferenceError("inner Fourier quadrature failed: "+str(ans[3]))
                integral, error, info = ans
                neval = int(info["neval"])
            cache[key] = complex(.5*math.exp(-a*ws*ws/2), integral)
            stats["inner_calls"] += 1
            stats["inner_evaluations"] += neval
            stats["inner_max_error_estimate"] = max(stats["inner_max_error_estimate"], float(error))
        u = cache[key] if w >= 0 else cache[key].conjugate()
        stats["max_factor_abs"] = max(stats["max_factor_abs"], abs(u))
        return u

    def outer_integrand(t0):
        w = math.sqrt(tau)*t0
        ua, ub = uvalue(scales[0], w), uvalue(scales[1], w)
        value = (normalizer*multiplicity*ua**na*ua.conjugate()**(m-na)
                 *ub**nb*ub.conjugate()**(m-nb))
        expected_mass = normalizer*math.exp(-a*d*w*w/2)
        stats["max_integrand_mass_identity_error"] = max(
            stats["max_integrand_mass_identity_error"], float(abs(value.sum()-expected_mass)))
        return value

    variants, checks = [], []
    for order in policy["GH_orders"]:
        nodes, weights = roots_hermitenorm(order)
        weights = weights/math.sqrt(2*math.pi)
        raw = np.zeros(len(na), dtype=complex)
        for node, weight in zip(nodes, weights):
            if weight != 0:
                raw += weight*outer_integrand(float(node))
        variants.append(raw.real)
        checks.append(dict(name="GH"+str(order), outer_nodes=order,
                           maximum_imaginary_residual=float(np.max(abs(raw.imag))),
                           raw_mass_error=abs(float(raw.real.sum())-1),
                           raw_min_probability=float(raw.real.min())))

    def density_integrand(t0):
        return outer_integrand(float(t0)).real*math.exp(-t0*t0/2)*invsqrt2pi

    qvalue, qerror, qinfo = quad_vec(
        density_integrand, -policy["outer_standard_limit"], policy["outer_standard_limit"],
        epsabs=policy["outer_epsabs"], epsrel=policy["outer_epsrel"], norm="max",
        limit=policy["outer_limit"], quadrature="gk21", full_output=True)
    variants.append(qvalue)
    checks.append(dict(name="adaptive", outer_evaluations=int(qinfo.neval),
                       status=int(qinfo.status), success=bool(qinfo.success), message=qinfo.message,
                       error_estimate=float(qerror), error_norm="PMF max norm",
                       L1_error_estimate=float(qerror)*len(na),
                       raw_mass_error=abs(float(qvalue.sum())-1),
                       raw_min_probability=float(qvalue.min())))
    raw = np.array(variants)
    finite = bool(np.isfinite(raw).all())
    positive = bool(np.all(raw > 0))
    validmass = all(x["raw_mass_error"] < policy["mass_gate"] for x in checks)
    delta = stats["inner_max_error_estimate"]+inner_tail_u
    inner_L1 = normalizer*2*d*delta*(1+2*delta)**(d-1)
    approximate_outer_L1 = outer_tail*(1+2*delta)**d
    moment_ref = independent_moment_reference(d, scales, c)
    moment_weights = np.stack([(2*na-m)**2, (2*nb-m)**2,
                               (2*na-m)*(2*nb-m)], axis=1)
    result = dict(
        d=d, scales=scales.tolist(), c=c, primary="adaptive",
        variant_names=policy["reference_variants"], raw_variants=raw,
        variants=raw.copy(), mean=raw[2].copy(),
        H=None, model_KL_single=None, total_p=None, symmetrized_group_p=None,
        checks=checks, stats=stats,
        inner_u_analytic_tail_bound=inner_tail_u,
        outer_exact_L1_analytic_tail_bound=outer_tail,
        inner_L1_error_estimate_plus_tail=inner_L1,
        approximate_outer_L1_tail_estimate=approximate_outer_L1,
        probability_finite=finite, raw_mass_pass=validmass,
        probability_strict_positive=positive,
        independent_Hermite_moments=moment_ref,
        deterministic_variants_are_not_independent_replicates=True,
        reference_standard_error=None, analytic_independence=False,
        numerical_qualification_not_rigorous_all_error_certificate=True,
        complex_cancellation_not_positive_product_mixture=True,
        qualified=False, numerical_pass=False)
    if not (finite and positive and validmass):
        return result
    probs = raw/raw.sum(axis=1, keepdims=True)  # only after roundoff-mass qualification
    funcs = np.array([functionals(p, d)[:2] for p in probs])
    p = probs[2]
    H, D, pn, ps = functionals(p, d)
    spin_moments = probs @ moment_weights
    moment_errors = abs(spin_moments-np.array(moment_ref["expected_spin_moments"]))
    l1 = _pairwise_max(probs, l1=True)
    function_difference = _pairwise_max(funcs)
    max_moment_error = float(moment_errors.max())
    qualified = (
        l1 <= policy["variant_L1_gate"]
        and function_difference <= policy["variant_function_difference_gate"]
        and bool(qinfo.success) and qerror <= policy["outer_error_estimate_gate"]
        and inner_L1 <= policy["inner_propagated_error_estimate_gate"]
        and max(x.get("maximum_imaginary_residual", 0.) for x in checks) < policy["mass_gate"]
        and stats["max_integrand_mass_identity_error"] < policy["identity_gate"]
        and max_moment_error <= policy["moment_gate"]
        and max(moment_ref["moment_error_estimates_plus_tail"]) <= policy["moment_gate"])
    result.update(mean=p, variants=probs, H=H, model_KL_single=D, total_p=pn,
                  symmetrized_group_p=ps, variant_functionals=funcs,
                  variant_spin_moments=spin_moments,
                  independent_moment_abs_errors=moment_errors,
                  independent_moment_max_error=max_moment_error,
                  pairwise_L1=l1, pairwise_max_function_difference=function_difference,
                  qualified=bool(qualified), numerical_pass=bool(qualified))
    return result


def pair_loss_vectors(ref, *, return_details=False):
    """All-pairs average NLL for each group-count cell.

    Uses independent Hermite pair probabilities. For d=16 the default result is
    [81], averaging 28 AA, 28 BB and 64 AB pairs (120 total).  It is a composite
    loss in nats per pair, not a full-vector joint NLL.
    """
    if not ref.get("qualified", False):
        raise ValueError("pair loss requires a qualified reference")
    d, _, _ = _domain(ref["d"], ref["scales"], ref["c"])
    if d < 4:
        raise ValueError("pair-class audit requires at least two coordinates per group")
    na, nb, _, _, _, _, _ = geometry(d)
    m = d//2
    types = np.array([math.comb(m, 2), math.comb(m, 2), m*m], dtype=np.int64)
    total_pairs = math.comb(d, 2)
    sa, sb = 2*na-m, 2*nb-m
    spin_sums = np.stack([(sa*sa-m)//2, (sb*sb-m)//2, sa*sb], axis=1)
    if not np.all((types[None, :]+spin_sums) % 2 == 0):
        raise ValueError("inconsistent same-spin pair counts")
    same = (types[None, :]+spin_sums)//2
    different = types[None, :]-same
    if np.any(same < 0) or np.any(different < 0):
        raise ValueError("negative pair counts")
    moments = ref["independent_Hermite_moments"]
    correlations = -np.asarray(moments["pair_magnitudes_AA_BB_AB"], dtype=np.float64)
    probabilities = np.column_stack([
        (1+correlations)/4, (1-correlations)/4,
        (1-correlations)/4, (1+correlations)/4])
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0):
        raise NumericalReferenceError("nonpositive pair probability; no probability floor")
    vectors = (total_pairs*math.log(4)
               -same @ np.log1p(correlations)
               -different @ np.log1p(-correlations))/total_pairs
    p = np.asarray(ref["mean"], dtype=np.float64)
    spin_reference = np.array([
        (float(p @ (sa*sa))-m)/(m*(m-1)),
        (float(p @ (sb*sb))-m)/(m*(m-1)),
        float(p @ (sa*sb))/(m*m)])
    difference = float(np.max(abs(spin_reference-correlations)))
    # This preserves the old stronger sum-moment qualification; no new gate chosen.
    if ref.get("independent_moment_max_error", math.inf) > numerics_contract()["moment_gate"]:
        raise NumericalReferenceError("reference/pair spin-moment check was not qualified")
    if not np.isfinite(vectors).all():
        raise NumericalReferenceError("nonfinite composite loss")
    if not return_details:
        return vectors
    return dict(loss_vector=vectors, pair_class_names=["AA", "BB", "AB"],
                pair_outcome_order=["00", "01", "10", "11"],
                pair_probabilities=probabilities, spin_correlations=correlations,
                pair_multiplicities=types, total_pairs=total_pairs,
                same_spin_pair_counts=same, different_spin_pair_counts=different,
                correlations_from_Fourier_PMF=spin_reference,
                independent_pair_vs_Fourier_max_error=difference,
                pair_probability_error_estimates_plus_tail=
                    np.asarray(moments["pair_error_estimates_plus_analytic_tail"])/4,
                qualified=True, unit="nats per pair (all-pairs average)",
                independent_pairs_are_not_independent_observations=True)

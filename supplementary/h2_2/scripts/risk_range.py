"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

from fractions import Fraction

import operator

import flint

from flint import arb,ctx

def exact(value):
    if isinstance(value,(bool,float)):raise ValueError("exact rational str/int/Fraction required")
    return Fraction(value)

def integer(value):
    if isinstance(value,bool):raise ValueError("actual integer precision required")
    try:value=operator.index(value)
    except TypeError as error:raise ValueError("actual integer precision required") from error
    if value<64:raise ValueError("precision>=64 required")
    return value

def aq(value):
    value=exact(value)
    return arb(value.numerator)/value.denominator

def endpoint(value):
    m,e=value.man_exp()
    return dict(mantissa=str(m),exponent_of_two=int(e))

def record(value):
    if not value.is_finite():raise ArithmeticError("nonfinite Arb result")
    return dict(arb_display_only=str(value),lower_exact_dyadic=endpoint(value.lower()),
                upper_exact_dyadic=endpoint(value.upper()))

def from_endpoint(value):
    m=int(value["mantissa"]);e=operator.index(value["exponent_of_two"])
    return arb(m)*(arb(2)**e)

def from_record(value):
    lo,hi=from_endpoint(value["lower_exact_dyadic"]),from_endpoint(value["upper_exact_dyadic"])
    if lo>hi:raise ValueError("inverted exact dyadic endpoints")
    return lo.union(hi)

def ball(value):
    if isinstance(value,dict):return from_record(value)
    if isinstance(value,arb):
        if not value.is_finite():raise ValueError("finite Arb value required")
        return value
    return aq(value)

def risk_range(q_true,Q_true,root_distance_upper,risk_Hessian_upper,*,
               precision_bits=192,exact_unit_mass=True):
    """Pure saved-box arithmetic, full-physical-gradient Fisher bound 1/4.

    exact_unit_mass is a mathematical model premise, not inferred from a
    numeric sum enclosing 1. Generic fixtures must satisfy that premise too.
    """
    precision_bits=integer(precision_bits)
    if exact_unit_mass is not True:raise ValueError("exact mathematical unit mass required")
    q_true=list(q_true);Q_true=list(Q_true)
    if not q_true or len(q_true)!=len(Q_true):raise ValueError("equally sized complete alphabets required")
    previous=ctx.prec;ctx.prec=precision_bits
    try:
        q=[ball(value) for value in q_true];Q=[ball(value) for value in Q_true]
        if any(not x.is_finite() or not x.lower()>0 for x in q+Q):
            raise ValueError("all true probability balls must have strictly positive lower endpoints")
        qmass=sum(q,arb(0));Qmass=sum(Q,arb(0))
        if not qmass.contains(1) or not Qmass.contains(1):
            raise ValueError("saved probability boxes inconsistent with exact unit mass")
        rr,mm=ball(root_distance_upper),ball(risk_Hessian_upper)
        if not rr.lower()>=0 or not mm.lower()>=0:
            raise ValueError("nonnegative distance/Hessian enclosures required")
        r,M=rr.upper(),mm.upper()
        KL=arb(0);chi=arb(0);terms=[]
        for qi,Qi in zip(q,Q):
            kt=qi*(qi/Qi).log()
            ct=(qi-Qi)**2/Qi
            KL+=kt;chi+=ct
            terms.append(dict(KL=record(kt),chi_square=record(ct)))
        chiupper=chi.upper()
        if not chiupper>=0:raise ArithmeticError("chi-square upper endpoint unexpectedly negative")
        # Use an exact positive endpoint, not sqrt of a ball whose outward
        # lower endpoint could cross zero. This never clips probability balls.
        G=chiupper.sqrt()/2
        first=G.upper()*r
        second=M*r*r/2
        allowance=first+second
        rootKL=KL+arb(0,allowance.upper())
        return dict(schema="candidate_v18_saved_probability_risk_range_v1",
            precision_bits=precision_bits,python_flint_version=flint.__version__,label_count=len(q),
            scope="saved_true_probability_endpoint_arithmetic_only",
            exact_unit_mass_model_premise=True,marginal_mass_enclosures_consistent=True,
            teacher_probability_mass_enclosure=record(qmass),candidate_probability_mass_enclosure=record(Qmass),
            all_true_probability_lower_endpoints_positive=True,probability_balls_renormalized=False,
            KL_definition="sum_y q_y log(q_y/Q_y)",KL_units="nats",center_KL_enclosure=record(KL),
            chi_square_definition="sum_y (q_y-Q_y)^2/Q_y",
            chi_square_enclosure=record(chi),chi_square_upper_exact_dyadic=endpoint(chiupper),
            observed_Fisher_operator_upper_exact="1/4",
            full_physical_gradient_norm_upper_enclosure=record(G),
            root_distance_upper_exact_dyadic=endpoint(r),
            risk_Hessian_operator_upper_exact_dyadic=endpoint(M),
            first_order_root_transfer_upper=record(first),second_order_root_transfer_upper=record(second),
            total_root_transfer_upper=record(allowance),root_KL_enclosure=record(rootKL),
            center_KL_lower_positive=bool(KL.lower()>0),root_KL_lower_positive=bool(rootKL.lower()>0),
            full_gradient_bound_not_directional_derivative=True,legacy_float_gradient_used=False,
            label_terms=terms,new_grid_evaluations=0,new_integrals=0,new_PMF_evaluations=0,
            new_random_draws=0,new_optimizer_updates=0,scientific_NPZ_reads=0,
            true_root_explicitly_solved_or_optimized=False)
    finally:ctx.prec=previous

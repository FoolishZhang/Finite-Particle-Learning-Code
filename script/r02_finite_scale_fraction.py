"""Independent exact-rational reconstruction of the saved R02 constants.

Standard library only.  This checks saved finite-sum arithmetic and consumes the
saved analytic error upper bounds as mathematical inputs; it does not re-prove
those bounds or replay tanh/logcosh/erf quadrature, MC, arrays, or optimization.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts/r02_finite_scale_v1_20261010"
DEFAULT_INPUT = PACKAGE / "CONSTANTS.json"
DEFAULT_OUTPUT = PACKAGE / "FRACTION_CHECK.json"
INNER = ("T", "B", "C", "V")
OUTER = ("OT2", "OTC", "OT2B", "OV2", "OB2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value):
    require(not isinstance(value, (float, bool)), "explicit rational required")
    return Fraction(value)


def endpoint(data):
    m, e = exact(data["mantissa"]), exact(data["exponent_of_two"])
    require(m.denominator == e.denominator == 1, "integer dyadic fields required")
    m, e = m.numerator, e.numerator
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << -e)


def interval(data):
    lo, hi = endpoint(data["lower_exact_dyadic"]), endpoint(data["upper_exact_dyadic"])
    require(lo <= hi, "inverted saved interval")
    return lo, hi


def point(x):
    x = exact(x)
    return x, x


def add(a, b):
    return a[0]+b[0], a[1]+b[1]


def neg(a):
    return -a[1], -a[0]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    p = [x*y for x in a for y in b]
    return min(p), max(p)


def square(a):
    if a[0] <= 0 <= a[1]:
        return Fraction(0), max(a[0]*a[0], a[1]*a[1])
    return min(a[0]*a[0], a[1]*a[1]), max(a[0]*a[0], a[1]*a[1])


def scale(a, factor):
    return mul(a, point(factor))


def divide_positive(a, b):
    require(b[0] > 0, "positive denominator not established")
    return mul(a, (1/b[1], 1/b[0]))


def enlarge(a, radius):
    require(radius >= 0, "negative error radius")
    return a[0]-radius, a[1]+radius


def contains(outer, inner):
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def fraction_record(x):
    return {"numerator_hex":hex(x.numerator), "denominator_hex":hex(x.denominator)}


def interval_record(x):
    return {"lower_exact_fraction":fraction_record(x[0]),
            "upper_exact_fraction":fraction_record(x[1]),
            "width_exact_fraction":fraction_record(x[1]-x[0]),
            "display_only_float": [float(x[0]),float(x[1])]}


def comparison(independent, arb_record):
    saved = interval(arb_record)
    return {"independent_exact_interval":interval_record(independent),
            "saved_Arb_contains_independent":contains(saved,independent),
            "independent_contains_saved_Arb":contains(independent,saved),
            "intervals_intersect":overlap(independent,saved)}


def error_upper(record):
    e = interval(record)
    require(e[0] >= 0, "error bound not nonnegative")
    return e[1]


def audit(data, source_sha256, implementation_sha256):
    require(data.get("status") == "complete", "partial sums cannot certify constants")
    c = data["configuration"]
    require(c["m"] == 3 and exact(c["rho_exact"]) == Fraction(9,20) and c["K"] == 100,
            "wrong target model")
    require(c["N"] == 160 and exact(c["step_exact"]) == Fraction(1,16)
            and exact(c["strip_exact"]) == Fraction(3,4)
            and c["precision_bits"] == 192 and c["endpoint_weights"] == "full",
            "wrong frozen quadrature configuration")
    step, N = exact(c["step_exact"]), c["N"]
    require(len(data["inner_rows"]) == 2*N+1, "incomplete outer nodes")
    bounds = data["analytic_bounds"]
    composition = data["composition"]
    outer = {k:point(0) for k in OUTER}
    inner_contained = 0
    inner_intersect = 0
    inner_failures = []
    for j,row in zip(range(-N,N+1),data["inner_rows"]):
        require(row["j"] == j and exact(row["u_exact"]) == j*step,
                "wrong, missing or duplicate outer node")
        require(row["inner_nodes_completed"] == 2*N+1, "partial inner node")
        w = interval(row["outer_weight"])
        require(w[0] > 0, "nonpositive Gaussian weight")
        inner = {k:enlarge(interval(row["finite"][k]),error_upper(row["error"][k]))
                 for k in INNER}
        for k in INNER:
            comp = comparison(inner[k],row["true"][k])
            inner_contained += int(comp["saved_Arb_contains_independent"])
            inner_intersect += int(comp["intervals_intersect"])
            if not comp["saved_Arb_contains_independent"]:
                inner_failures.append({"j":j,"moment":k,**comp})
        T,B,C,V = (inner[k] for k in INNER)
        prod = {"OT2":square(T),"OTC":mul(T,C),"OT2B":mul(square(T),B),
                "OV2":square(V),"OB2":square(B)}
        for k in OUTER:
            outer[k] = add(outer[k],mul(w,prod[k]))
    outer_finite_comparison = {k:comparison(outer[k],data["outer_finite_with_inner_error"][k])
                               for k in OUTER}
    outer_true = {k:enlarge(outer[k],error_upper(bounds["outer"][k]["total"])) for k in OUTER}
    outer_comparison = {k:comparison(outer_true[k],composition["outer_true"][k]) for k in OUTER}
    uni = {k:enlarge(interval(v),error_upper(bounds["univariate"][k]["total"]))
           for k,v in data["univariate_finite"].items()}
    maximum = {k:enlarge(interval(v),error_upper(bounds["maximum"][k]["total"]))
               for k,v in data["maximum_finite"].items()}
    univariate_comparison = {k:comparison(v,composition["univariate_true"][k]) for k,v in uni.items()}
    d0,d2,d3,t2 = (uni[k] for k in ("d1","d2","d3","t2"))
    variance = add(point(Fraction(1,4)), sub(d2,square(d0)))
    s0 = sub(point(1),t2)
    centered_d3 = add(sub(d3,scale(mul(d0,d2),3)),scale(mul(square(d0),d0),2))
    kappa3 = sub(scale(s0,Fraction(-3,16)),centered_d3)
    spin = outer_true["OT2"]
    mH = scale(spin,Fraction(1,4))
    cbar = sub(scale(s0,Fraction(-9,160)),
                scale(sub(add(scale(outer_true["OTC"],2),outer_true["OT2B"]),
                          scale(mul(d0,spin),3)),Fraction(1,4)))
    VH = scale(sub(add(scale(t2,2),outer_true["OV2"]),square(spin)),Fraction(1,16))
    VA = add(add(point(Fraction(3,4)),scale(sub(variance,point(Fraction(1,4))),3)),
             scale(sub(outer_true["OB2"],square(d0)),6))
    constants = {"d0":d0,"v":variance,"s0":s0,"kappa3_L":kappa3,
                 "kappa_spin":spin,"m_H":mH,"cbar":cbar,"V_H":VH,"V_A":VA,
                 **maximum}
    require(variance[0] > 0, "independent v interval is not strictly positive")
    second = mul(divide_positive(maximum["B_K"],variance),
                 sub(divide_positive(mul(mH,kappa3),scale(variance,3)),cbar))
    constants["second_order_coefficient"] = second
    const_comparison = {k:comparison(v,composition["constants"][k]) for k,v in constants.items()}
    comparisons = [*outer_finite_comparison.values(),*outer_comparison.values(),
                   *univariate_comparison.values(),*const_comparison.values()]
    checks = {
        "all_1284_inner_expansions_contained_by_saved_Arb":inner_contained==4*(2*N+1),
        "all_1284_inner_expansions_intersect_saved_Arb":inner_intersect==4*(2*N+1),
        "all_outer_and_constant_intervals_contained_by_saved_Arb":all(
            r["saved_Arb_contains_independent"] for r in comparisons),
        "all_outer_and_constant_intervals_intersect_saved_Arb":all(
            r["intervals_intersect"] for r in comparisons),
        "v_positive":variance[0]>0,"V_H_positive":VH[0]>0,"V_A_positive":VA[0]>0,
        "J_K_positive":maximum["J_K"][0]>0,"B_K_positive":maximum["B_K"][0]>0,
        "m_H_positive":mH[0]>0,
        "second_order_coefficient_sign_agrees_with_saved_Arb":(
            ("positive" if second[0]>0 else "negative" if second[1]<0 else "unresolved") ==
            ("positive" if interval(composition["constants"]["second_order_coefficient"])[0]>0 else
             "negative" if interval(composition["constants"]["second_order_coefficient"])[1]<0 else "unresolved")),
        "all_independent_constant_widths_below_1e_minus_8":all(
            x[1]-x[0] < Fraction(1,10**8) for x in constants.values()),
        "gaussian_mass_interval_contains_one":uni["gaussian_mass"][0]<=1<=uni["gaussian_mass"][1],
    }
    # Positivity and sign conclusions stand on the independent intervals even
    # if a containing Arb range happened to be narrower because of dependence.
    # Such containment failures remain failures here, never hidden by tolerance.
    return {
        "schema":"r02_independent_fraction_check_v1",
        "source_sha256":source_sha256,"implementation_sha256":implementation_sha256,
        "configuration":c,"checks":checks,"passed":all(checks.values()),
        "inner_expansions_checked":4*(2*N+1),"inner_containment_failures":inner_failures,
        "second_order_coefficient_sign":("positive" if second[0]>0 else "negative" if second[1]<0 else "unresolved"),
        "outer_finite":outer_finite_comparison,"outer_true":outer_comparison,
        "univariate_true":univariate_comparison,"constants":const_comparison,
        "budget_positive_prefactor_prerequisites":{
            k:interval_record(constants[k]) for k in ("v","V_H","V_A","J_K","m_H")},
        "scope":{
            "standard_library_only":True,"saved_analytic_error_bounds_consumed_as_inputs":True,
            "analytic_error_proof_independently_recomputed":False,
            "target_tanh_logcosh_erf_evaluations":0,"new_integral_nodes":0,
            "new_MC_draws":0,"new_NPZ_reads":0,"new_D_n_scores":0,
            "new_optimization_steps":0,"scientific_external_review":False,
            "Arb_runtime_used":False,"project_module_imports":False,
            "numerical_tolerance_added":False,
            "marked_budget_transcendental_prefactor_recomputed":False,
            "finite_n_remainder_certified":False,
        },
    }


def self_test():
    require(square((Fraction(-2),Fraction(3))) == (Fraction(0),Fraction(9)), "cross-zero square")
    require(square((Fraction(-3),Fraction(-2))) == (Fraction(4),Fraction(9)), "negative square")
    require(square((Fraction(2),Fraction(3))) == (Fraction(4),Fraction(9)), "positive square")
    require(mul((Fraction(-2),Fraction(3)),(Fraction(-4),Fraction(5))) ==
            (Fraction(-12),Fraction(15)), "interval product")
    require(divide_positive((Fraction(-2),Fraction(3)),(Fraction(2),Fraction(4))) ==
            (Fraction(-1),Fraction(3,2)), "positive division")
    require(endpoint({"mantissa":"-3","exponent_of_two":-2}) == Fraction(-3,4), "dyadic decode")
    require(not contains((Fraction(0),Fraction(1)),(Fraction(-1,10**100),Fraction(1))),
            "no hidden containment tolerance")
    return {"passed":True,"exact_manufactured_checks":7,"target_nodes":0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write",action="store_true")
    modes.add_argument("--check",action="store_true")
    modes.add_argument("--self-test",action="store_true")
    p.add_argument("--input",type=Path,default=DEFAULT_INPUT)
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),sort_keys=True))
        return
    raw=args.input.read_bytes()
    data=json.loads(raw.decode("utf-8-sig"))
    result=audit(data,hashlib.sha256(raw).hexdigest(),
                 hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.write:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8",newline="\n") as handle:
            json.dump(result,handle,ensure_ascii=False,indent=2,sort_keys=True)
            handle.write("\n")
    else:
        saved=json.loads(args.output.read_text(encoding="utf-8"))
        require(result==saved,"saved independent Fraction check changed")
    print(json.dumps({"mode":"write" if args.write else "check","passed":result["passed"],
                      "checks":result["checks"],"target_nodes":0},sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()

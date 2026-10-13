"""Strict saved-dyadic matrix checks and conditional witness budgets.

This does not certify Gaussian/Laplace quadrature, the exact F8 expectation,
or existence of a stationary point. --run is freeze-gated; --check only
recomputes saved JSON binary64 data with Arb elementary operations and LDL.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/local_witness_feasibility_v1_20261008"
sys.path.insert(0, str(ROOT / ".cache/finite_selection_tools"))
from flint import arb, ctx
import flint

BITS = 192
BASELINE = "e924b13643e09a9ad190b71c03cabeac501b3971"
RADIUS = Fraction(1, 2000000)
H_ERROR = Fraction(1, 1000)
G_TOTAL = Fraction(1, 100000000)
RADIAL_FLOOR = Fraction(1, 20)
TRANSVERSE_FLOOR = Fraction(7, 500)
RISK_CENTER_UPPER = Fraction(-1, 50)
L_SAFE = 14069
SUMMARY = ART / "NOMINAL_SUMMARY.json"
OUTPUT = ART / "SAVED_MATRIX_CERTIFICATE.json"
FREEZE = ART / "FREEZE.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rational(value):
    if isinstance(value, (bool, float, np.floating)):
        raise ValueError("explicit rational input required")
    return Fraction(value)


def binary64(value):
    """The JSON number denotes its round-tripped binary64, not decimal exactness."""
    if isinstance(value, bool):
        raise ValueError("boolean is not a binary64 datum")
    if isinstance(value, str):
        # Float.hex is supported only when explicitly marked by a hex literal.
        v = float.fromhex(value) if value.lower().startswith(("0x", "-0x", "+0x")) else float(value)
    else:
        v = float(value)
    if not math.isfinite(v):
        raise ValueError("nonfinite binary64 datum")
    return Fraction.from_float(v)


def aq(value):
    value = rational(value)
    return arb(value.numerator) / value.denominator


def endpoint(value):
    mantissa, exponent = value.man_exp()
    return {"mantissa": str(mantissa), "exponent_of_two": int(exponent)}


def ball(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite Arb result")
    return {"arb_display_only": str(value), "lower_exact_dyadic": endpoint(value.lower()),
            "upper_exact_dyadic": endpoint(value.upper())}


def exact_record(value):
    value = rational(value)
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def norm(values):
    return aq(sum((v*v for v in values), Fraction(0))).sqrt()


def dyadic_matrix(values, dimension):
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (dimension, dimension) or not np.isfinite(a).all():
        raise ValueError(f"finite {dimension} by {dimension} matrix required")
    raw = [[binary64(x) for x in row] for row in a.tolist()]
    with np.errstate(over="raise", invalid="raise"):
        rounded = (a+a.T)*np.float64(.5)
    if not np.isfinite(rounded).all():
        raise ValueError("matrix symmetrization produced nonfinite values")
    sym = [[binary64(x) for x in row] for row in rounded.tolist()]
    exact_sym = [[(raw[i][j]+raw[j][i])/2 for j in range(dimension)]
                 for i in range(dimension)]
    rounding = [sym[i][j]-exact_sym[i][j] for i in range(dimension)
                for j in range(dimension)]
    raw_difference = [sym[i][j]-raw[i][j] for i in range(dimension)
                      for j in range(dimension)]
    return sym, {"dimension": dimension,
        "raw_binary64_hex": [[float(x).hex() for x in row] for row in raw],
        "symmetrized_binary64_hex": [[float(x).hex() for x in row] for row in sym],
        "symmetrization_operation": "numpy float64 (A + A.T) * 0.5; saved result treated as exact dyadic",
        "sym_rounding_error_Frobenius_bound": ball(norm(rounding)),
        "sym_minus_raw_Frobenius_bound": ball(norm(raw_difference)),
        "future_error_budget_note": "Certified true-Hessian error must cover original quadrature/arithmetic error and this symmetrization rounding; C1/D1 difference is not such a certificate."}


def ldlt_positive(matrix):
    """No-pivot Arb LDL: positive enclosures certify exact symmetric matrix PD.

    Unresolved pivot signs and negative/zero pivots all reject certification.
    Inputs may be exact Fraction matrices for manufactured tests.
    """
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("nonempty square matrix required")
    exact = [[rational(value) for value in row] for row in matrix]
    for i in range(n):
        for j in range(i):
            if exact[i][j] != exact[j][i]:
                raise ValueError("symmetric matrix required")
    a = [[aq(value) for value in row] for row in exact]
    lower = [[arb(0) for _ in range(n)] for _ in range(n)]
    pivots, records = [], []
    for i in range(n):
        lower[i][i] = arb(1)
        pivot = a[i][i]-sum((lower[i][k]*lower[i][k]*pivots[k]
                            for k in range(i)), arb(0))
        positive = bool(pivot.is_finite() and pivot > 0)
        records.append({"index": i, "pivot": ball(pivot), "strictly_positive": positive})
        if not positive:
            return {"positive_definite_certified": False, "first_rejected_pivot": i,
                    "dimension": n, "pivots": records,
                    "method": "192-bit Arb no-pivot LDL; unresolved sign rejects"}
        pivots.append(pivot)
        for j in range(i+1, n):
            lower[j][i] = (a[j][i]-sum((lower[j][k]*lower[i][k]*pivots[k]
                                       for k in range(i)), arb(0)))/pivot
    return {"positive_definite_certified": True, "first_rejected_pivot": None,
            "dimension": n, "pivots": records,
            "method": "192-bit Arb no-pivot LDL; unresolved sign rejects"}


def shift(matrix, floor):
    return [[value-(floor if i == j else 0) for j, value in enumerate(row)]
            for i, row in enumerate(matrix)]


def fixed_slice_basis(a):
    if len(a) != 8 or a[7] == 0:
        raise ValueError("eight-vector with nonzero last coordinate required")
    return [[Fraction(int(i == j)) if i != 7 else -a[j]/a[7]
             for j in range(7)] for i in range(8)]


def congruence(matrix, basis):
    n, p = len(matrix), len(basis[0])
    if len(basis) != n or any(len(row) != p for row in basis):
        raise ValueError("basis shape mismatch")
    return [[sum((basis[i][s]*matrix[i][j]*basis[j][t]
                  for i in range(n) for j in range(n)), Fraction(0))
             for t in range(p)] for s in range(p)]


def projected_gradient_norm(gradient):
    if len(gradient) != 8:
        raise ValueError("eight radial gradient coordinates required")
    # U+ has columns (ei+e7-i)/sqrt(2), so this formula is exact without sqrt2 rounding.
    return aq(sum(((gradient[i]+gradient[7-i])**2/2 for i in range(4)), Fraction(0))).sqrt()


def safe_lipschitz_proof():
    s, t = Fraction(2828428, 1000000), Fraction(75836587, 1000000)
    product = (Fraction(1, 4)+Fraction(131, 2)*s)*t
    checks = [s*s > 8, t**4 > 321**3, product < L_SAFE]
    if not all(checks):
        raise AssertionError("safe Lipschitz integer proof failed")
    return {"L_F_safe_integer": L_SAFE,
        "sqrt8_upper": exact_record(s), "moment_upper": exact_record(t),
        "product_upper": exact_record(product), "exact_fraction_checks_passed": checks}


def conditional_budget(a, nominal_gradient):
    a_norm = norm(a)
    r = aq(RADIUS)
    radial_margin = aq(RADIAL_FLOOR-H_ERROR-L_SAFE*RADIUS)
    transverse_margin = aq(TRANSVERSE_FLOOR-H_ERROR-L_SAFE*RADIUS)
    outward_capacity = radial_margin*r
    remaining = aq(G_TOTAL)-nominal_gradient
    # On the full physical parameter r-ball: ||b||1 <= sqrt8 r <= 3r,
    # ||b||2 <= r, and ||W||F <= ||a0||2+r. Independent bounds are conservative.
    qmin = (-3*r/2-(r*r+(a_norm+r)**2)/8).exp()/256
    risk_curvature = 24/qmin
    risk_at_root_upper = aq(RISK_CENTER_UPPER)+risk_curvature*r
    nu_lower = -risk_at_root_upper
    return {"scope": "conditional antecedents; true-center integration errors not certified",
        "physical_radius_exact": str(RADIUS), "future_H_operator_error_max_exact": str(H_ERROR),
        "future_total_restricted_gradient_max_exact": str(G_TOTAL),
        "future_true_center_risk_direction_upper_exact": str(RISK_CENTER_UPPER),
        "L_F_safe_integer": L_SAFE, "L_Fr_exact": str(L_SAFE*RADIUS),
        "radial_curvature_lower": ball(radial_margin),
        "fixed_transverse_curvature_lower": ball(transverse_margin),
        "outward_gradient_capacity": ball(outward_capacity),
        "nominal_restricted_gradient_norm": ball(nominal_gradient),
        "remaining_gradient_error_budget": ball(remaining),
        "a0_norm": ball(a_norm), "uniform_Qy_lower_bound": ball(qmin),
        "risk_Hessian_operator_bound": ball(risk_curvature),
        "conditional_root_risk_direction_upper": ball(risk_at_root_upper),
        "conditional_root_negative_derivative_margin": ball(nu_lower),
        "checks": {"positive_radial_margin": bool(radial_margin > 0),
                   "positive_transverse_margin": bool(transverse_margin > 0),
                   "gradient_outward_condition": bool(outward_capacity > aq(G_TOTAL)),
                   "positive_future_gradient_error_budget": bool(remaining > 0),
                   "fixed_slice_transverse_neighbourhood": bool(a_norm > r),
                   "conditional_risk_margin_exceeds_point01": bool(nu_lower > aq(Fraction(1, 100)))},
        "conditions_not_established_here": ["true-center gradient error within remaining budget",
            "true-center Hessian operator errors at most .001 including sym rounding",
            "true-center derivative along the exact unit direction at most -.020",
            "finite-interval quadrature, arithmetic, teacher PMF and truncation bounds combined"]}


def summary_fields(summary):
    center = summary["center"]
    theta = center.get("theta", center.get("theta16"))
    if theta is None:
        theta = [0.]*8+center["a"]
    original = center.get("original_theta", center.get("original_theta16"))
    if original is None:
        raise ValueError("summary center.original_theta is required for projection distance")
    theta, original = [binary64(v) for v in theta], [binary64(v) for v in original]
    if len(theta) != 16 or len(original) != 16 or any(theta[:8]):
        raise ValueError("center must contain 16 coordinates with exact zero biases")
    a = theta[8:]
    if a != list(reversed(a)) or not all(v > 0 for v in a):
        raise ValueError("exact bit-matched positive reversal-symmetric center required")
    expected_pairs = [binary64(.5*(float(original[8+i])+float(original[8+7-i])))
                      for i in range(4)]
    if a != expected_pairs+list(reversed(expected_pairs)):
        raise ValueError("center must use frozen float64 old-P pair means with copied reversal bits")
    blocks = summary["rules"]["D1"]["blocks"]
    if isinstance(blocks, dict):
        blocks = [{"K": k, **v} for k, v in blocks.items()]
    finite = [b for b in blocks if str(b["K"]) == "8"]
    integrated = [b for b in blocks if str(b["K"]).lower() in ("inf", "infinity")]
    if len(finite) != 1 or len(integrated) != 1:
        raise ValueError("exactly one D1 K8 and one D1 inf block required")
    return theta, original, a, finite[0], integrated[0]


def make_certificate(summary, *, input_bindings=None):
    old_precision = ctx.prec
    ctx.prec = BITS
    try:
        theta, original, a, finite, integrated = summary_fields(summary)
        hba, hba_record = dyadic_matrix(finite["H_ba"], 16)
        hp, hp_record = dyadic_matrix(finite["Hperp"], 8)
        basis = fixed_slice_basis(a)
        radial = ldlt_positive(shift(hba, RADIAL_FLOOR))
        transverse = ldlt_positive(congruence(shift(hp, TRANSVERSE_FLOOR), basis))
        g = [binary64(v) for v in finite["gradient"]]
        gr = [binary64(v) for v in integrated["gradient"]]
        if len(g) != 16 or len(gr) != 16:
            raise ValueError("K8 and inf gradients must have 16 b,a coordinates")
        gnorm = projected_gradient_norm(g[8:])
        direction = summary["risk_direction"]
        records = direction["h_dyadic"]
        if len(records) != 8:
            raise ValueError("eight saved dyadic risk direction coordinates required")
        risk_h = []
        for record in records:
            exact = Fraction(int(record["numerator"]), int(record["denominator"]))
            if exact != binary64(record["hex"]) or exact != binary64(record["value"]):
                raise ValueError("risk h ratio/hex/value mismatch")
            risk_h.append(exact)
        expected_h = []
        for i in range(4):
            rounded = binary64(.5*(float(gr[8+i])+float(gr[8+7-i])))
            expected_h.append(rounded)
        if risk_h != expected_h+list(reversed(expected_h)):
            raise ValueError("risk h must be frozen float64 pair-average with copied reversal bits")
        risk_h_norm = norm(risk_h)
        if not risk_h_norm > 0:
            raise ValueError("nonzero reversal-projected risk gradient required")
        exact_projection = [(gr[8+i]+gr[8+7-i])/2 for i in range(8)]
        direction_rounding = norm([x-y for x, y in zip(risk_h, exact_projection)])
        nominal_directional = -aq(sum((x*y for x, y in zip(gr[8:], risk_h)), Fraction(0)))/risk_h_norm
        budget = conditional_budget(a, gnorm)
        exact_basis_orthogonal = all(sum((a[i]*basis[i][j] for i in range(8)), Fraction(0)) == 0
                                     for j in range(7))
        projection_distance = norm([x-y for x, y in zip(theta, original)])
        all_saved_checks = radial["positive_definite_certified"] and transverse["positive_definite_certified"]
        result = {"schema": "local_witness_saved_matrix_certificate_v1", "precision_bits": BITS,
            "python_flint_version": flint.__version__,
            "scope": "strict saved binary64 matrix certificate plus conditional budgets; not an exact F8 certificate",
            "input_bindings": input_bindings or {}, "baseline": BASELINE,
            "center": {"theta_exact_dyadics": [exact_record(v) for v in theta],
                "original_theta_exact_dyadics": [exact_record(v) for v in original],
                "exact_old_to_center_projection_distance": ball(projection_distance),
                "bias_exact_zero": True, "radial_exact_reversal_symmetric": True,
                "center_matches_frozen_float64_old_pair_means": True},
            "saved_Hba": hba_record, "saved_Hperp": hp_record,
            "radial_floor_exact": str(RADIAL_FLOOR), "radial_shifted_LDL": radial,
            "transverse_floor_exact": str(TRANSVERSE_FLOOR), "transverse_shifted_generalized_LDL": transverse,
            "fixed_slice": {"basis_exact_rationals": [[exact_record(v) for v in row] for row in basis],
                "exact_a0_dot_V_zero": exact_basis_orthogonal,
                "generalized_test": "V.T @ (Hperp_sym - .014 I8) @ V; equivalent to physical-norm curvature > .014 on a0-perp",
                "basis_assumed_orthonormal": False,
                "naive_VtHV_eigenvalue_is_not_physical_curvature": True},
            "nominal_gradient": {"radial_exact_dyadics": [exact_record(v) for v in g[8:]],
                "restricted_norm": ball(gnorm), "below_point1e_minus10": bool(gnorm < aq(Fraction(1, 10**10))),
                "norm_formula": "sqrt(sum_i=0..3 (g_i+g_7-i)^2 / 2); U+=pairlift/sqrt2"},
            "fixed_risk_direction": {"h_exact_rationals": [exact_record(v) for v in risk_h],
                "h_norm": ball(risk_h_norm), "definition": "d=(0,-h/||h||2,0) in (b,u,v); h is the frozen float64 pair-average of saved D1 inf radial gradient, then treated as exact dyadic",
                "unit_direction_component_enclosures": [ball(-aq(v)/risk_h_norm) for v in risk_h],
                "pair_average_rounding_difference_norm": ball(direction_rounding),
                "nominal_directional_derivative": ball(nominal_directional),
                "nominal_derivative_computation": "exact saved dyadic g_inf dot mathematical unit d; not automatically -||h|| when pair means were rounded",
                "nominal_not_true_integral_directional_derivative": True,
                "floating_normalization_used_as_exact": False},
            "safe_L_F_proof": safe_lipschitz_proof(), "conditional_budget": budget,
            "strict_saved_matrix_checks_passed": bool(all_saved_checks and exact_basis_orthogonal),
            "conditional_budget_arithmetic_passed": all(budget["checks"].values()),
            "exact_stationary_point_existence_certified": False,
            "true_F8_Hessian_positive_certified": False,
            "true_risk_derivative_certified": False,
            "finite_interval_quadrature_certified": False,
            "C1_D1_difference_used_as_true_error": False,
            "new_Gaussian_Laplace_integrals": 0, "new_kernel_calls": 0,
            "new_random_draws": 0, "Arb_operations": "elementary exp/sqrt and matrix LDL only"}
        return result
    finally:
        ctx.prec = old_precision


def require_freeze(freeze_path=FREEZE):
    if not Path(freeze_path).is_file():
        raise RuntimeError("FREEZE.json required before reading nominal scientific output")
    frozen = read(freeze_path)
    if frozen.get("baseline") != BASELINE:
        raise RuntimeError("freeze baseline mismatch")
    sources = frozen.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("freeze sources mapping required")
    for path in (Path(__file__).resolve(), ROOT/"script/local_witness_feasibility_certificate_test.py"):
        relative = path.relative_to(ROOT).as_posix()
        if sources.get(relative) != sha(path):
            raise RuntimeError(f"frozen source hash mismatch: {relative}")
    return frozen


def computed_from_files(summary_path=SUMMARY, freeze_path=FREEZE):
    require_freeze(freeze_path)
    # Freeze is checked before any nominal scientific output is opened.
    summary = read(summary_path)
    bindings = {"summary_path": Path(summary_path).resolve().relative_to(ROOT).as_posix(),
                "summary_sha256": sha(summary_path), "freeze_sha256": sha(freeze_path),
                "certificate_source_sha256": sha(__file__)}
    return make_certificate(summary, input_bindings=bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.run and args.output.exists():
        raise FileExistsError("saved matrix certificate exists; no overwrite allowed")
    result = computed_from_files(args.summary)
    if args.run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    else:
        if read(args.output) != result:
            raise AssertionError("saved certificate does not match exact recomputation")
    passed = result["strict_saved_matrix_checks_passed"] and result["conditional_budget_arithmetic_passed"]
    print(json.dumps({"mode": "run" if args.run else "check", "passed": passed,
        "strict_saved_matrix_checks_passed": result["strict_saved_matrix_checks_passed"],
        "conditional_budget_arithmetic_passed": result["conditional_budget_arithmetic_passed"],
        "exact_stationary_point_existence_certified": False,
        "new_kernel_calls": 0, "new_Gaussian_Laplace_integrals": 0}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

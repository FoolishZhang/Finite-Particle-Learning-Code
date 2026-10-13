"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from flint import arb, ctx

L_SAFE = 14069

def rational(value):
    if isinstance(value, (bool, float, np.floating)):
        raise ValueError("explicit rational input required")
    return Fraction(value)

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

def safe_lipschitz_proof():
    s, t = Fraction(2828428, 1000000), Fraction(75836587, 1000000)
    product = (Fraction(1, 4)+Fraction(131, 2)*s)*t
    checks = [s*s > 8, t**4 > 321**3, product < L_SAFE]
    if not all(checks):
        raise AssertionError("safe Lipschitz integer proof failed")
    return {"L_F_safe_integer": L_SAFE,
        "sqrt8_upper": exact_record(s), "moment_upper": exact_record(t),
        "product_upper": exact_record(product), "exact_fraction_checks_passed": checks}

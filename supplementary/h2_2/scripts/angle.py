"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

from fractions import Fraction

def require(condition, message):
    if not condition:
        raise ValueError(message)

def exact(value):
    require(not isinstance(value, (bool, float)), "explicit exact rational required")
    return Fraction(value)

def endpoint(data):
    mantissa, exponent = exact(data["mantissa"]), exact(data["exponent_of_two"])
    require(mantissa.denominator == exponent.denominator == 1, "integer dyadic fields required")
    m, e = mantissa.numerator, exponent.numerator
    return Fraction(m << e) if e >= 0 else Fraction(m, 1 << -e)

def interval(record):
    lower = endpoint(record["lower_exact_dyadic"])
    upper = endpoint(record["upper_exact_dyadic"])
    require(lower <= upper, "inverted exact saved endpoints")
    return lower, upper

def add(left, right):
    return left[0] + right[0], left[1] + right[1]

def multiply(left, right):
    products = [x * y for x in left for y in right]
    return min(products), max(products)

def divide_positive(numerator, denominator):
    require(denominator[0] > 0, "denominator must be strictly positive")
    return multiply(numerator, (1 / denominator[1], 1 / denominator[0]))

def enlarge(value, radius):
    radius = exact(radius)
    require(radius >= 0, "nonnegative exact error radius required")
    return value[0] - radius, value[1] + radius

def overlap(left, right):
    return left[0] <= right[1] and right[0] <= left[1]

def contained(inner, outer):
    return outer[0] <= inner[0] and inner[1] <= outer[1]

def fraction_record(value):
    # Different positive Q endpoints create large rational denominators when
    # 256 quotients are summed. Hex is exact and avoids Python's decimal digit
    # conversion limit without modifying that process-wide security setting.
    return {"numerator_hex": hex(value.numerator), "denominator_hex": hex(value.denominator)}

def self_test():
    F = Fraction
    require(multiply((F(-2), F(3)), (F(-4), F(5))) == (F(-12), F(15)), "mixed-sign multiply")
    require(divide_positive((F(-6), F(4)), (F(2), F(3))) == (F(-3), F(2)), "mixed-sign divide")
    require(divide_positive((F(-6), F(-4)), (F(2), F(3))) == (F(-3), F(-4, 3)), "negative divide")
    require(enlarge((F(1, 3), F(2, 3)), F(1, 6)) == (F(1, 6), F(5, 6)), "exact enlargement")
    require(add((F(-1, 3), F(2, 5)), (F(1, 7), F(3, 8))) == (F(-4, 21), F(31, 40)), "exact sum")
    require(endpoint({"mantissa": "-7", "exponent_of_two": -3}) == F(-7, 8), "dyadic negative")
    require(endpoint({"mantissa": "3", "exponent_of_two": 5}) == F(96), "dyadic positive exponent")
    for rational in (F(-1, 3), F(7, 8), F(0)):
        encoded = fraction_record(rational)
        require(F(int(encoded["numerator_hex"], 16), int(encoded["denominator_hex"], 16)) == rational,
                "hex rational round trip")
    require(overlap((F(0), F(1)), (F(1), F(2))), "closed endpoint intersection")
    require(contained((F(0), F(1)), (F(-1), F(2))), "interval containment")
    try:
        divide_positive((F(1), F(2)), (F(0), F(1)))
    except ValueError:
        pass
    else:
        raise ValueError("zero denominator accepted")
    return {"passed": True, "arithmetic": "stdlib fractions.Fraction only", "new_integrals": 0}

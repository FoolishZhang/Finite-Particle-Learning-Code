"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

from fractions import Fraction

from flint import arb,ctx

def exact(value):
    if isinstance(value,(float,bool)):raise ValueError('explicit exact rational required')
    return Fraction(value)

def aq(value):
    value=exact(value);return arb(value.numerator)/value.denominator

def endpoint(value):
    mantissa,exponent=value.man_exp()
    return dict(mantissa=str(mantissa),exponent_of_two=int(exponent))

def record(value):
    if not value.is_finite():raise ArithmeticError('nonfinite Arb finite-sum enclosure')
    return dict(arb_display_only=str(value),lower_exact_dyadic=endpoint(value.lower()),upper_exact_dyadic=endpoint(value.upper()))

def from_endpoint(ep):
    m=int(ep['mantissa']);e=int(ep['exponent_of_two'])
    return arb(m)*(arb(2)**e)

def from_record(data):
    return from_endpoint(data['lower_exact_dyadic']).union(from_endpoint(data['upper_exact_dyadic']))

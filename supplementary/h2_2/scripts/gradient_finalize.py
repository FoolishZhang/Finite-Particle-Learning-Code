"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

import common

from flint import arb,ctx

def certify_vector(vector,error):
    if not vector or not error>=0:raise ValueError('finite vector and nonnegative error')
    finite_norm=sum((abs(x).upper()**2 for x in vector),arb(0)).sqrt()
    total_norm=finite_norm+error
    enclosure=(-error.upper()).union(error.upper())
    return dict(finite_vector=[common.record(x) for x in vector],finite_box_norm_upper=common.record(finite_norm),
        arithmetic_box_L2_radius_upper=common.record(sum(((x.upper()-x.lower())**2/4 for x in vector),arb(0)).sqrt()),
        true_component_enclosures=[common.record(x+enclosure) for x in vector],
        mathematical_vector_error_upper=common.record(error),true_gradient_L2_upper=common.record(total_norm),
        gradient_L2_strictly_below_1e8_inverse=bool(total_norm<common.aq('1/100000000')),
        norm_uses_triangle_vector_error_once=True)

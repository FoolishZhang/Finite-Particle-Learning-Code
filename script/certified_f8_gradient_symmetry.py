"""Pure paired-count mappings and weight aggregation for collinear F8.

No files, arrays, quadrature, RNG or science-kernel calls. Labels are integer
0..2^m-1 with coordinate 0 as MSB. Original weights are summed, never replaced
by multiplicities, normalized, or treated as within-pair exchangeable.
"""
from __future__ import annotations

from fractions import Fraction
import itertools
import math
import operator


def even_dimension(m):
    if isinstance(m, bool):
        raise ValueError("actual positive even integer required")
    try:
        m = operator.index(m)
    except TypeError as exc:
        raise ValueError("actual positive even integer required") from exc
    if m <= 0 or m % 2:
        raise ValueError("positive even dimension required")
    return m


def label_bits(label, m=8):
    m = even_dimension(m)
    if isinstance(label, bool):
        raise ValueError("actual integer label required")
    try:
        label = operator.index(label)
    except TypeError as exc:
        raise ValueError("actual integer label required") from exc
    if not 0 <= label < 2**m:
        raise ValueError("label outside complete alphabet")
    return tuple((label >> (m-1-j)) & 1 for j in range(m))


def paired_counts(label, m=8):
    bits = label_bits(label, m)
    return tuple(bits[j]+bits[m-1-j] for j in range(m//2))


def complement_count(count):
    count = tuple(count)
    if not count or any(isinstance(x, bool) for x in count):
        raise ValueError("nonempty ternary count vector required")
    try:
        count = tuple(operator.index(x) for x in count)
    except TypeError as exc:
        raise ValueError("actual integer ternary counts required") from exc
    if any(x not in (0, 1, 2) for x in count):
        raise ValueError("ternary counts must be 0, 1 or 2")
    return tuple(2-x for x in count)


def representative(count):
    complement = complement_count(count)
    count = tuple(2-x for x in complement)
    return min(count, complement)


def symmetry_map(m=8):
    m = even_dimension(m)
    pairs = m//2
    counts = list(itertools.product(range(3), repeat=pairs))
    representatives = sorted({representative(c) for c in counts})
    count_index = {c: i for i, c in enumerate(counts)}
    rep_index = {c: i for i, c in enumerate(representatives)}
    label_counts = [paired_counts(label, m) for label in range(2**m)]
    labels_to_count = [count_index[c] for c in label_counts]
    labels_to_rep = [rep_index[representative(c)] for c in label_counts]
    count_labels = [[] for _ in counts]
    rep_labels = [[] for _ in representatives]
    for label, ci, ri in zip(range(2**m), labels_to_count, labels_to_rep):
        count_labels[ci].append(label)
        rep_labels[ri].append(label)
    count_to_rep = [rep_index[representative(c)] for c in counts]
    rep_counts = [[] for _ in representatives]
    for ci, ri in enumerate(count_to_rep):
        rep_counts[ri].append(ci)
    return {"schema": "certified_f8_paired_count_mapping_v1", "m": m,
        "pair_count": pairs, "label_count": 2**m, "count_group_count": 3**pairs,
        "representative_count": (3**pairs+1)//2,
        "pair_indices": [[j, m-1-j] for j in range(pairs)],
        "label_order": "integer label 0..2^m-1; observation 0 is MSB",
        "count_order": "lexicographic ternary tuples",
        "representative_rule": "lexicographic min(c, 2-c)",
        "count_groups": [list(c) for c in counts],
        "representatives": [list(c) for c in representatives],
        "label_to_count_index": labels_to_count,
        "label_to_representative_index": labels_to_rep,
        "count_to_representative_index": count_to_rep,
        "count_group_labels": count_labels, "representative_labels": rep_labels,
        "representative_count_group_indices": rep_counts,
        "complement_fixed_count": [1]*pairs,
        "within_pair_teacher_exchangeability_assumed": False,
        "gradient_equivalence_after_prior_sign_average_not_at_one_unreflected_bank": True}


def aggregate_weights(weights, m=8):
    """Accept Fractions/Arb/etc.; return sums with input arithmetic untouched."""
    mapping = symmetry_map(m)
    weights = list(weights)
    if len(weights) != mapping["label_count"]:
        raise ValueError("complete original label weight vector required")
    count_weights = [sum((weights[label] for label in labels), 0)
                     for labels in mapping["count_group_labels"]]
    representative_weights = [sum((weights[label] for label in labels), 0)
                              for labels in mapping["representative_labels"]]
    return {"count_weights": count_weights, "representative_weights": representative_weights,
            "weights_renormalized": False, "multiplicities_used_as_weights": False}


def exact_binary64_or_rational(value):
    if isinstance(value, bool):
        raise ValueError("boolean parameter not allowed")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("finite parameter required")
        return Fraction.from_float(value)
    try:
        return Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError("exact rational or finite binary64 parameter required") from exc


def exact_pair_parameters(theta, m=8):
    """Return exact physical a-pair Fractions; xi = sqrt2*a is symbolic."""
    m = even_dimension(m)
    theta = [exact_binary64_or_rational(x) for x in theta]
    if len(theta) != 2*m or any(theta[:m]):
        raise ValueError("theta=(zero biases, full m radial parameters) required")
    a = theta[m:]
    if any(x <= 0 for x in a) or a != a[::-1]:
        raise ValueError("positive exact reversal-paired radial parameters required")
    return tuple(a[:m//2])


def transfer_constants(m=8, K=8):
    m = even_dimension(m)
    if isinstance(K, bool):
        raise ValueError("positive actual integer K required")
    try:
        K = operator.index(K)
    except TypeError as exc:
        raise ValueError("positive actual integer K required") from exc
    if K <= 0:
        raise ValueError("positive K required")
    return {"m": m, "K": K,
        "per_label_gradient_bound_formula": "sqrt(m*K) for restricted paired-count gradient",
        "gradient_coefficient_squared_exact": str(m*K),
        "teacher_L1_transfer_formula": "restricted_gradient_norm_error <= sqrt(m*K)*teacher_total_L1",
        "legacy_full_radial_coefficient_squared_exact": str(m*(1+K)**2),
        "legacy_full_radial_coefficient_used_in_this_batch": False,
        "finite_target_full_gradient_error_not_required": True,
        "grouping_L1_contraction": True,
        "root_sum_dimension_factor_not_needed_for_vector_bounds": True,
        "uncertified_float_gradient_used_as_bound": False}

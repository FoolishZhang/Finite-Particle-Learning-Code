"""Pure folded label moments and complete Hessian-block contractions.

Numeric entries may be Fraction/Arb/etc. No I/O, arrays, quadrature or kernels.
Complement folding preserves Hbb/Haa/Hperp, not the mixed bias-radial block.
All 8 coordinates, including antisymmetric radial directions, are retained.
"""
from __future__ import annotations

import operator
from certified_f8_gradient_symmetry import even_dimension, label_bits, paired_counts, symmetry_map


def fold_teacher_moments(weights, m=8):
    """Sum original weights with effective y; no division by class mass W."""
    m = even_dimension(m)
    mapping = symmetry_map(m)
    weights = list(weights)
    if len(weights) != 2**m:
        raise ValueError("all original label weights required")
    reps = [tuple(c) for c in mapping["representatives"]]
    moments = [{"W": 0, "M": [0]*m, "S": [[0]*m for _ in range(m)]} for _ in reps]
    reflected = []
    for label, weight in enumerate(weights):
        ri = mapping["label_to_representative_index"][label]
        bits = label_bits(label, m)
        reflect = paired_counts(label, m) != reps[ri]
        reflected.append(reflect)
        if reflect:
            bits = tuple(1-y for y in bits)
        entry = moments[ri]
        entry["W"] += weight
        ones = [i for i, y in enumerate(bits) if y]
        for i in ones:
            entry["M"][i] += weight
            for j in ones:
                entry["S"][i][j] += weight
    return {"schema": "certified_f8_hessian_folded_moments_v1", "m": m,
        "representatives": [list(c) for c in reps], "moments": moments,
        "label_to_representative_index": mapping["label_to_representative_index"],
        "label_reflected_to_representative": reflected,
        "effective_label_rule": "if count is not the lex representative, y_effective=1-y; self all-ones count unchanged",
        "teacher_weights_renormalized": False, "within_pair_exchangeability_assumed": False,
        "mixed_bias_radial_block_preserved_by_this_fold": False}


def pair_index(m=8):
    m = even_dimension(m)
    return tuple(min(i, m-1-i) for i in range(m))


def upper_pairs(pair_count=4):
    if isinstance(pair_count, bool):
        raise ValueError("positive actual pair count required")
    pair_count = operator.index(pair_count)
    if pair_count < 1:
        raise ValueError("positive actual pair count required")
    return tuple((i, j) for i in range(pair_count) for j in range(i, pair_count))


def expand_pair_moments(T, P, PP0_upper, PP2_upper, m=8):
    """Expand four unique p moments into full m vectors/matrices exactly.

    T=(T0,T1,T2); P[l][j] for l=0,1,2 and j=0..m/2-1.
    PP0/PP2 use lex upper-triangle (i,j), i<=j. Ψ is separate.
    """
    m = even_dimension(m)
    pairs = m//2
    T, P = list(T), [list(row) for row in P]
    if len(T) != 3 or len(P) != 3 or any(len(row) != pairs for row in P):
        raise ValueError("T has 3 values and P has shape (3,pair_count)")
    uppers = upper_pairs(pairs)
    if len(PP0_upper) != len(uppers) or len(PP2_upper) != len(uppers):
        raise ValueError("PP0/PP2 complete upper triangles required")
    index = pair_index(m)
    def expand(values):
        lookup = {ij: value for ij, value in zip(uppers, values)}
        return [[lookup[tuple(sorted((index[i], index[j])))] for j in range(m)] for i in range(m)]
    return {"T": T, "P": [[row[j] for j in index] for row in P],
            "PP0": expand(PP0_upper), "PP2": expand(PP2_upper)}


def moment_columns(m=8):
    """Unambiguous scalar column order; 36 total, 35 excluding Ψ for m=8."""
    m = even_dimension(m)
    pairs = m//2
    columns = ["Psi", "T0", "T1", "T2"]
    columns += [f"P{l}_{i}" for l in (0, 1, 2) for i in range(pairs)]
    columns += [f"PP{l}_{i}_{j}" for l in (0, 2) for i, j in upper_pairs(pairs)]
    return {"columns": columns, "total_scalar_columns": len(columns),
        "columns_excluding_Psi": len(columns)-1,
        "columns_excluding_separate_Psi_T0": len(columns)-2,
        "unique_probability_count": pairs,
        "expanded_matrix_dimension": m}


def contractions(moment, candidate):
    """Weighted B0/B2 and same-minus-cross, with Hbb M/S cancellation."""
    W, M, S = moment["W"], list(moment["M"]), moment["S"]
    T, P, PP0, PP2 = candidate["T"], candidate["P"], candidate["PP0"], candidate["PP2"]
    m = len(M)
    if not m or len(S) != m or any(len(row) != m for row in S):
        raise ValueError("full original-coordinate S matrix required")
    if len(T) != 3 or len(P) != 3 or any(len(row) != m for row in P):
        raise ValueError("expanded P and T shapes required")
    if any(len(mat) != m or any(len(row) != m for row in mat) for mat in (PP0, PP2)):
        raise ValueError("expanded full PP matrices required")
    t0, t1, t2 = T
    p0, p1, p2 = P
    B0 = [[W*(p0[i]-PP0[i][i]) if i == j else 0 for j in range(m)] for i in range(m)]
    B2 = [[W*(p2[i]-PP2[i][i]) if i == j else 0 for j in range(m)] for i in range(m)]
    Dbb = [[W*(t0*PP0[i][j]-p0[i]*p0[j]) for j in range(m)] for i in range(m)]
    Daa = [[(t0*t2-t1*t1)*S[i][j]
            -t0*(M[i]*p2[j]+p2[i]*M[j])
            +t1*(M[i]*p1[j]+p1[i]*M[j])
            +W*(t0*PP2[i][j]-p1[i]*p1[j])
            for j in range(m)] for i in range(m)]
    Dperp = [[t0*(t0*S[i][j]-M[i]*p0[j]-p0[i]*M[j]+W*PP0[i][j])
              for j in range(m)] for i in range(m)]
    return {"B0": B0, "B2": B2, "Dbb": Dbb, "Daa": Daa, "Dperp": Dperp}


def hessian_integrand(Psi, moment, candidate, K=8):
    """Matrices integrated in v=log(u), t=u*A*exp(-u*A), scale one."""
    if isinstance(K, bool):
        raise ValueError("actual positive integer K required")
    K = operator.index(K)
    if K < 1:
        raise ValueError("positive K required")
    parts = contractions(moment, candidate)
    first = K*Psi**(K-1)
    second = K*(K-1)*Psi**(K-2) if K > 1 else 0
    combine = lambda b, d: [[first*x-second*y for x, y in zip(br, dr)]
                             for br, dr in zip(b, d)]
    return {"Hbb": combine(parts["B0"], parts["Dbb"]),
        "Haa": combine(parts["B2"], parts["Daa"]),
        "Hperp": combine(parts["B0"], parts["Dperp"]),
        "mixed_bias_radial_constructed": False}


def teacher_transfer_constants(m=8, K=8):
    m = even_dimension(m)
    if isinstance(K, bool):
        raise ValueError("actual positive K required")
    K = operator.index(K)
    if K < 1:
        raise ValueError("positive K required")
    from fractions import Fraction
    return {"m": m, "K": K,
        "full_ba_invariant_block_diagonal_coefficient_exact": str((2*m+Fraction(1, 4))*(1+K)),
        "transverse_coefficient_exact": str(m+Fraction(1, 4)),
        "true_ba_mixed_block_zero_by_exact_teacher_not_by_finite_weights": True,
        "teacher_total_L1_used_once_per_matrix_norm": True,
        "uncertified_float_matrix_used_as_bound": False}

"""Exact count-space mapping for fixed exchanges between two equal groups."""
from __future__ import annotations

import math
import numpy as np


def count_geometry(d=16):
    if d < 2 or d % 2:
        raise ValueError("d must be even and at least two")
    m = d//2
    cells = [(a, b) for a in range(m+1) for b in range(m+1)]
    a = np.array([x for x, _ in cells], dtype=np.int64)
    b = np.array([y for _, y in cells], dtype=np.int64)
    n = a+b
    mult = np.array([math.comb(m, int(x))*math.comb(m, int(y))
                     for x, y in cells], dtype=np.int64)
    total_mult = np.array([math.comb(d, int(x)) for x in n], dtype=np.int64)
    indicator = (n[:, None] == np.arange(d+1)[None, :]).astype(np.int64)
    return dict(d=d, m=m, na=a, nb=b, total=n, multiplicity=mult,
                log_multiplicity=np.log(mult.astype(float)),
                log_total_multiplicity=np.log(total_mult.astype(float)),
                total_multiplicity=total_mult, indicator=indicator)


def _choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def swap_kernel(r, m=8):
    """Return exact-numerator W with row-vector convention mapped_p = p @ W.

    Cell index is (m+1)*nA+nB.  r original-A coordinates and r original-B
    coordinates exchange group membership.  r=0..m is supported.
    """
    if isinstance(r, bool) or int(r) != r or not 0 <= r <= m:
        raise ValueError("r must be an integer from zero through m")
    if int(m) != m or not 1 <= m <= 8:
        raise ValueError("exact int64 implementation supports integer m=1..8")
    r, m = int(r), int(m)
    size = (m+1)**2
    denominator = math.comb(m, r)**2
    numerator = np.zeros((size, size), dtype=np.int64)
    for a in range(m+1):
        for b in range(m+1):
            row = (m+1)*a+b
            for i in range(r+1):
                wa = _choose(a, i)*_choose(m-a, r-i)
                for j in range(r+1):
                    wb = _choose(b, j)*_choose(m-b, r-j)
                    if wa and wb:
                        u, v = a-i+j, b+i-j
                        numerator[row, (m+1)*u+v] += wa*wb
    geo = count_geometry(2*m)
    assert np.array_equal(numerator.sum(axis=1), np.full(size, denominator))
    assert not np.any(numerator[geo["total"][:, None] != geo["total"][None, :]])
    weighted = geo["multiplicity"][:, None]*numerator
    assert np.array_equal(weighted, weighted.T)
    if r == 0:
        assert np.array_equal(numerator, np.eye(size, dtype=np.int64))
    if r == m:
        for a in range(m+1):
            for b in range(m+1):
                assert numerator[(m+1)*a+b, (m+1)*b+a] == 1
    return dict(r=r, m=m, kernel=numerator.astype(float)/denominator,
                numerator=numerator, denominator=denominator,
                convention="row_vector: mapped_p = p @ kernel",
                total_count_preserved=True, exact_integer_checks_passed=True)


def pattern_geometry(d=16, r_values=(0, 1, 2, 4)):
    """Pattern id = sum_i C_i*2**i; true A occupies low d/2 bits."""
    if d < 2 or d % 2 or d > 16:
        raise ValueError("pattern indexing supports even d from 2 through 16")
    m = d//2
    if any(int(r) != r or not 0 <= r <= m for r in r_values):
        raise ValueError("invalid exchange count")
    ids = np.arange(1 << d, dtype=np.int64)
    popcount = np.fromiter((int(i).bit_count() for i in ids), dtype=np.int64, count=len(ids))
    original_A = (1 << m)-1
    full_mask = (1 << d)-1
    original_B = full_mask ^ original_A
    true_cells = (m+1)*popcount[ids & original_A]+popcount[ids & original_B]
    masks, cells = [], []
    for r in r_values:
        lower = (1 << int(r))-1
        mask_A = (original_A ^ lower) | (lower << m)
        masks.append(mask_A)
        cells.append((m+1)*popcount[ids & mask_A]+popcount[ids & (full_mask ^ mask_A)])
    return dict(d=d, m=m, r_values=list(r_values), pattern_ids=ids,
                true_A_mask=original_A, assumed_A_masks=masks,
                true_count_indices=true_cells,
                count_indices_by_r=np.array(cells, dtype=np.int64),
                pattern_convention="id=sum_i C_i*2**i; bit0..m-1=trueA",
                cell_convention="cell=(m+1)*nA+nB")


def integer_group_histograms(pattern_histograms, pattern_map):
    """Actual integer regrouping of saved patterns, never h @ W fractions."""
    hist = np.asarray(pattern_histograms)
    d, m = pattern_map["d"], pattern_map["m"]
    if hist.ndim != 2 or hist.shape[1] != 1 << d:
        raise ValueError("expected [prefix, 2**d] pattern histograms")
    if np.any(hist < 0) or not np.array_equal(hist, hist.astype(np.int64)):
        raise ValueError("pattern histograms must contain nonnegative integers")
    result = np.zeros((len(pattern_map["r_values"]), len(hist), (m+1)**2), dtype=np.int64)
    for ri, indices in enumerate(pattern_map["count_indices_by_r"]):
        for ni, row in enumerate(hist):
            np.add.at(result[ri, ni], indices, row.astype(np.int64))
    assert np.array_equal(result.sum(axis=-1), np.broadcast_to(hist.sum(axis=1), result.shape[:2]))
    return result


def pattern_probabilities(group_count_pmf, pattern_map):
    geo = count_geometry(pattern_map["d"])
    p = np.asarray(group_count_pmf, dtype=float)
    if p.shape != geo["na"].shape or np.any(p <= 0) or not np.isfinite(p).all():
        raise ValueError("group-count PMF must be finite and strictly positive")
    cell = pattern_map["true_count_indices"]
    raw = p[cell]/geo["multiplicity"][cell]
    error = abs(float(raw.sum())-1)
    if error >= 1e-12:
        raise ValueError("pattern probability mass fails the fixed gate")
    return raw/raw.sum(), dict(raw_mass_error=error,
                              roundoff_normalization=float(raw.sum()),
                              pattern_probabilities_strict_positive=bool(np.all(raw > 0)))


def orbit_entropy(group_count_pmf, d=16):
    geo = count_geometry(d)
    p = np.asarray(group_count_pmf, dtype=float)
    if p.shape != geo["na"].shape or np.any(p <= 0):
        raise ValueError("expected positive count PMF")
    return -float(p @ (np.log(p)-geo["log_multiplicity"]))


def full_pattern_cross_entropy(mapped_teacher, candidate_count_pmf, d=16):
    geo = count_geometry(d)
    p = np.asarray(mapped_teacher, dtype=float)
    q = np.asarray(candidate_count_pmf, dtype=float)
    if p.shape != geo["na"].shape or q.shape[-1] != len(p) or np.any(q <= 0):
        raise ValueError("incompatible or nonpositive count PMFs")
    return (-np.log(q)+geo["log_multiplicity"]) @ p

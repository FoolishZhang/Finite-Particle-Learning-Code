"""Exact, non-project fixtures for the label-conditional R02 pair identity.

No numerical integration, random sampling, target kernel or NPZ I/O.
The document defines all fixture parameters before execution.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
from itertools import product
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/R02FiniteScale_v1_20261010/AVERAGED_KERNEL.md"
OUT = ROOT / "artifacts/r02_finite_scale_v1_20261010/FIXTURE_CHECK.json"
LABEL_P = (Q(2, 5), Q(3, 5))
STATES = (
    ((Q(1, 3), Q(-1), Q(-2, 5)), (Q(2, 3), Q(-1, 4), Q(3, 4))),
    ((Q(3, 5), Q(-3, 4), Q(2, 3)), (Q(2, 5), Q(0), Q(-1, 2))),
)
EXP_WEIGHTS = ((Q(1), Q(2)), (Q(3), Q(1)))
G_DENSITY = tuple(map(Q, (0, 0, 0, 140, -420, 420, -140)))
G_CDF = tuple(map(Q, (0, 0, 0, 0, 35, -84, 70, -20)))


def trim(p):
    p = list(p)
    while len(p) > 1 and not p[-1]:
        p.pop()
    return tuple(p)


def add(a, b):
    out = [Q(0)] * max(len(a), len(b))
    for i, x in enumerate(a):
        out[i] += x
    for i, x in enumerate(b):
        out[i] += x
    return trim(out)


def scale(a, c):
    return trim([x * c for x in a])


def mul(a, b):
    out = [Q(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return trim(out)


def power(a, n):
    out = (Q(1),)
    for _ in range(n):
        out = mul(out, a)
    return out


def compose(p, inner):
    out = (Q(0),)
    for c in reversed(p):
        out = add(mul(out, inner), (c,))
    return out


def derivative(p):
    return trim([Q(i) * p[i] for i in range(1, len(p))] or [Q(0)])


def integral(p, left, right):
    return sum((c * (right ** (i + 1) - left ** (i + 1)) / (i + 1)
                for i, c in enumerate(p)), Q(0))


def frac(x):
    x = Q(x)
    return {"numerator": str(x.numerator), "denominator": str(x.denominator)}


def intervals(states, s):
    cuts = sorted({a / s + t for _, a, _ in states for t in (Q(0), Q(1))})
    return list(zip(cuts[:-1], cuts[1:]))


def pieces(states, s, mid):
    ans = []
    for p, a, h in states:
        x = mid - a / s
        if x <= 0:
            den, cdf = (Q(0),), (Q(0),)
        elif x >= 1:
            den, cdf = (Q(0),), (Q(1),)
        else:
            var = (-a / s, Q(1))
            den, cdf = compose(G_DENSITY, var), compose(G_CDF, var)
        ans.append((p, h, den, cdf))
    return ans


def beta_average(states, k, s):
    marked = unmarked = centered = Q(0)
    mean_h = sum((p * h for p, _, h in states), Q(0))
    for left, right in intervals(states, s):
        vals = pieces(states, s, (left + right) / 2)
        f = F = hfun = (Q(0),)
        for p, h, den, cdf in vals:
            f = add(f, scale(den, p))
            F = add(F, scale(cdf, p))
            hfun = add(hfun, scale(den, p * h))
        weight = mul(f, power(F, k - 2))
        marked += integral(mul(hfun, weight), left, right)
        unmarked += integral(mul(f, weight), left, right)
        centered += integral(mul(add(hfun, scale(f, -mean_h)), weight), left, right)
    assert marked == centered + mean_h * unmarked
    return marked, unmarked, centered


def beta_banks(states, k, s):
    """Enumerate ordered pairs in each bank, without the averaged identity."""
    value = Q(0)
    for left, right in intervals(states, s):
        vals = pieces(states, s, (left + right) / 2)
        for indices in product(range(len(states)), repeat=k):
            bank = [vals[i] for i in indices]
            prob = Q(1)
            for p, _, _, _ in bank:
                prob *= p
            pair_sum = (Q(0),)
            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    term = (bank[i][1],)
                    for z in range(k):
                        term = mul(term, bank[z][2 if z in (i, j) else 3])
                    pair_sum = add(pair_sum, term)
            value -= prob * integral(pair_sum, left, right) / s
    return value


def gumbel_banks(states, k):
    """States hold exact exp(A), not a rounded logarithm."""
    total = Q(0)
    for bank in product(states, repeat=k):
        prob = Q(1)
        for p, _, _ in bank:
            prob *= p
        weight = sum((w for _, w, _ in bank), Q(0))
        marked = Q(0)
        for i in range(k):
            for j in range(k):
                if i != j:
                    marked += bank[i][2] * bank[i][1] * bank[j][1] / weight ** 2
        total -= prob * marked
    return total


def exponential_product(a, b):
    out = {}
    for rate_a, coeff_a in a.items():
        for rate_b, coeff_b in b.items():
            rate = rate_a + rate_b
            out[rate] = out.get(rate, Q(0)) + coeff_a * coeff_b
    return out


def gumbel_average(states, k, s):
    cdf = {}
    density = {}
    mark_density = {}
    for p, w, h in states:
        cdf[w] = cdf.get(w, Q(0)) + p
        density[w] = density.get(w, Q(0)) + p * w
        mark_density[w] = mark_density.get(w, Q(0)) + p * h * w
    coeffs = exponential_product(mark_density, density)
    for _ in range(k - 2):
        coeffs = exponential_product(coeffs, cdf)
    # g_s contributes s twice and dz contributes 1/s: kernel = s * sum c/rate^2.
    kernel = s * sum((c / rate ** 2 for rate, c in coeffs.items()), Q(0))
    return -Q(k * (k - 1)) * kernel / s


def mixed(states_by_label):
    return tuple((py * p, a, h) for py, states in zip(LABEL_P, states_by_label)
                 for p, a, h in states)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute():
    beta_rows = []
    gumbel_rows = []
    residual_rows = []
    beta_bank_evaluations = 0
    gumbel_bank_evaluations = 0
    density_prime = derivative(G_DENSITY)
    density_second = derivative(density_prime)
    # Safe rational global derivative caps on [0,1]; derivatives vanish outside.
    L0 = sum(map(abs, G_DENSITY), Q(0))
    L1 = sum(map(abs, density_prime), Q(0))
    L2 = sum(map(abs, density_second), Q(0))
    V1 = L1
    for k in (2, 3, 4):
        r = k - 2
        Ig = integral(mul(power(G_DENSITY, 2), power(G_CDF, r)), Q(0), Q(1))
        Jg = integral(mul(mul(density_prime, G_DENSITY), power(G_CDF, r)), Q(0), Q(1))
        J_ibp = (Q(0) if r == 0 else -Q(r, 2) * integral(
            mul(power(G_DENSITY, 3), power(G_CDF, r - 1)), Q(0), Q(1)))
        assert Jg == J_ibp
        for s in (Q(1), Q(2)):
            direct = factorized = decomposition = translated = centered_value = Q(0)
            mean_mark = covariance = remainder_bound = Q(0)
            for index, (py, states) in enumerate(zip(LABEL_P, STATES)):
                d = beta_banks(states, k, s)
                main, bare, coupling = beta_average(states, k, s)
                mean = sum((p * h for p, _, h in states), Q(0))
                active_mean = sum((p * a for p, a, _ in states), Q(0))
                v = sum((p * (a - active_mean) ** 2 for p, a, _ in states), Q(0))
                c = sum((p * (h - mean) * (a - active_mean) for p, a, h in states), Q(0))
                M2 = sum((p * abs(h - mean) * (a - active_mean) ** 2
                          for p, a, h in states), Q(0))
                rhs = -Q(k * (k - 1)) * main / s
                assert d == rhs
                direct += py * d
                factorized += py * rhs
                decomposition += -py * Q(k * (k - 1)) * (coupling + mean * (bare - Ig)) / s
                shift = (Q(7, 5), Q(-3, 7))[index]
                translated_states = tuple((p, a + shift, h) for p, a, h in states)
                centered_states = tuple((p, a - active_mean, h) for p, a, h in states)
                translated += -py * Q(k * (k - 1)) * beta_average(translated_states, k, s)[0] / s
                centered_value += -py * Q(k * (k - 1)) * beta_average(centered_states, k, s)[0] / s
                mean_mark += py * mean
                covariance += py * c
                remainder_bound += py * (
                    Q(k) * L2 * M2 / (2 * s ** 3)
                    + Q(k * (k - 1)) * abs(mean) * v * (2 * L2 + r * L0 * L1) / (2 * s ** 3)
                    + Q(k * (k - 1)) * abs(c) * v * V1 * (L2 + r * L0 * L1) / (2 * s ** 4))
                beta_bank_evaluations += len(states) ** k
            assert direct == factorized == translated == centered_value
            assert direct + Q(k * (k - 1)) * mean_mark * Ig / s == decomposition
            wrong = -Q(k * (k - 1)) * beta_average(mixed(STATES), k, s)[0] / s
            assert wrong != direct
            leading = -Q(k * (k - 1)) * mean_mark * Ig / s
            correction = Q(k * (k - 1)) * covariance * Jg / s ** 2
            remainder = direct - leading - correction
            assert abs(remainder) <= remainder_bound
            beta_rows.append({
                "K": k, "s": frac(s), "direct": frac(direct), "factorized": frac(factorized),
                "wrong_mix_labels_first": frac(wrong), "wrong_mix_difference": frac(wrong - direct),
                "centered_equal": centered_value == direct, "translated_equal": translated == direct,
                "residual_decomposition_equal": True,
            })
            residual_rows.append({"K": k, "s": frac(s), "I_g": frac(Ig), "J_g": frac(Jg),
                                  "J_integration_by_parts_equal": True, "signed_correction": frac(correction),
                                  "exact_remainder": frac(remainder), "explicit_bound": frac(remainder_bound),
                                  "bound_pass": True})
            exp_states = tuple(tuple((p, w, h) for (p, _, h), w in zip(states, weights))
                               for states, weights in zip(STATES, EXP_WEIGHTS))
            direct_g = sum((py * gumbel_banks(states, k) for py, states in zip(LABEL_P, exp_states)), Q(0))
            averaged_g = sum((py * gumbel_average(states, k, s) for py, states in zip(LABEL_P, exp_states)), Q(0))
            wrong_g = gumbel_average(mixed(exp_states), k, s)
            assert direct_g == averaged_g
            assert wrong_g != direct_g
            gumbel_bank_evaluations += sum(len(states) ** k for states in exp_states)
            gumbel_rows.append({"K": k, "s": frac(s), "direct_softmax": frac(direct_g),
                                "factorized_gumbel_integral": frac(averaged_g),
                                "wrong_mix_labels_first": frac(wrong_g),
                                "wrong_mix_difference": frac(wrong_g - direct_g)})
    return {
        "schema": "r02_label_averaged_kernel_exact_fixtures_v1",
        "qualification": "manufactured algebra and finite-bound tests; no project target values or finite-n certificate",
        "source_sha256": {"script/r02_finite_scale_fixture.py": digest(Path(__file__)),
                           str(DOC.relative_to(ROOT)).replace("\\", "/"): digest(DOC)},
        "arithmetic": "Python fractions.Fraction, exact polynomial integration and exact Gumbel Laplace identity",
        "fixed_K": [2, 3, 4], "fixed_coordinate_scales": ["1", "2"],
        "beta_density_C2": True, "beta_cases": beta_rows, "gumbel_cases": gumbel_rows,
        "signed_correction_fixture_checks": residual_rows,
        "beta_safe_norm_caps": {"L0": frac(L0), "L1": frac(L1), "L2": frac(L2), "V1": frac(V1)},
        "checks": {"all_12_exact_factorizations": True, "all_12_wrong_label_mixtures_rejected": True,
                   "all_6_beta_translations_and_centerings": True, "all_6_exact_residual_decompositions": True,
                   "all_6_signed_correction_finite_bounds": True, "all_3_J_integration_by_parts": True},
        "work_counts": {"beta_direct_banks": beta_bank_evaluations, "gumbel_direct_banks": gumbel_bank_evaluations,
                        "fixture_cases": 12, "new_project_target_integrals": 0, "new_Gaussian_grids": 0,
                        "new_random_samples": 0, "new_NPZ_reads": 0, "new_optimizations": 0},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.write and OUT.exists():
        raise FileExistsError(f"Exclusive receipt already exists: {OUT}")
    result = compute()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    else:
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        if result != saved:
            raise AssertionError("Saved fixture receipt differs from read-only reconstruction")
    print(json.dumps({"status": "pass", "mode": "write" if args.write else "check",
                      "receipt": str(OUT.relative_to(ROOT)), "checks": result["checks"],
                      "work_counts": result["work_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

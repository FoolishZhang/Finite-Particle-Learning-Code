"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

from copy import deepcopy

from dataclasses import dataclass

from fractions import Fraction as F

import hashlib

NS = [64, 256, 1024, 4096]

WEIGHTS = {
    "W1_q_Delta": ("W1", 98, "Delta"),
    "W1_q_minus1_Delta": ("W1", 97, "Delta"),
    "W2_q_deltaG": ("W2", 98, "deltaG"),
    "W1plus_q_minus1_deltaG": ("W1plus", 97, "deltaG"),
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def exact(x):
    require(isinstance(x, (str, int, F)) and not isinstance(x, bool),
            "explicit rational required; floats/bools rejected")
    return F(x)

def endpoint(x):
    m, e = exact(x["mantissa"]), exact(x["exponent_of_two"])
    require(m.denominator == e.denominator == 1, "integer dyadic fields required")
    m, e = m.numerator, e.numerator
    return F(m << e) if e >= 0 else F(m, 1 << -e)

@dataclass(frozen=True)
class I:
    lo: F
    hi: F

    def __post_init__(self):
        object.__setattr__(self, "lo", exact(self.lo))
        object.__setattr__(self, "hi", exact(self.hi))
        require(self.lo <= self.hi, "inverted interval")

    def __add__(self, x):
        x = point(x)
        return I(self.lo + x.lo, self.hi + x.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, x):
        return self + -point(x)

    def __rsub__(self, x):
        return point(x) + -self

    def __mul__(self, x):
        x = point(x)
        vals = [a * b for a in (self.lo, self.hi) for b in (x.lo, x.hi)]
        return I(min(vals), max(vals))

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = point(x)
        require(x.lo > 0, "strictly positive denominator required")
        return self * I(1 / x.hi, 1 / x.lo)

    def __pow__(self, power):
        require(isinstance(power, int) and not isinstance(power, bool) and power >= 0,
                "nonnegative integer power required")
        if not power:
            return point(1)
        if power % 2:
            return I(self.lo ** power, self.hi ** power)
        return abs(self) ** (power - 1) * abs(self)

    def __abs__(self):
        return I(0 if self.lo <= 0 <= self.hi else min(abs(self.lo), abs(self.hi)),
                 max(abs(self.lo), abs(self.hi)))

    def contains(self, x):
        x = point(x)
        return self.lo <= x.lo and x.hi <= self.hi

    def overlaps(self, x):
        x = point(x)
        return self.lo <= x.hi and x.lo <= self.hi

def point(x):
    return x if isinstance(x, I) else I(exact(x), exact(x))

def saved(x):
    return I(endpoint(x["lower_exact_dyadic"]), endpoint(x["upper_exact_dyadic"]))

def digest_q(x):
    return hashlib.sha256((hex(x.numerator) + "/" + hex(x.denominator)).encode("ascii")).hexdigest()

def rational_summary(x):
    """Short, outward decimal endpoints plus hashes of the exact Fractions."""
    x = point(x)
    scale = 10 ** 35
    lower = (x.lo.numerator * scale) // x.lo.denominator
    upper = -((-x.hi.numerator * scale) // x.hi.denominator)
    def decimal(integer):
        sign = "-" if integer < 0 else ""
        a, b = divmod(abs(integer), scale)
        return f"{sign}{a}.{b:035d}"
    return {"lower_decimal_outward": decimal(lower),
            "upper_decimal_outward": decimal(upper),
            "lower_exact_fraction_sha256": digest_q(x.lo),
            "upper_exact_fraction_sha256": digest_q(x.hi)}

def relation(stored, rebuilt):
    return {"overlap": stored.overlaps(rebuilt),
            "saved_contains_fraction_interval": stored.contains(rebuilt),
            "fraction_contains_saved_interval": rebuilt.contains(stored),
            "saved_upper_ge_fraction_upper": stored.hi >= rebuilt.hi,
            "fraction_reconstruction": rational_summary(rebuilt)}

def prepare_grid(grid):
    require(grid.get("schema") == "r02_remainder_weight_grid_v1" and
            grid.get("status") == "complete", "complete saved Gaussian grid required")
    cfg = grid["configuration"]
    step, n = exact(cfg["step_exact"]), cfg["N"]
    require(isinstance(n, int) and not isinstance(n, bool) and n > 0,
            "positive integer grid N required")
    require(step > 0 and n * step >= 2 and (1 / step).denominator == 1,
            "grid must cover [-2,2] and contain derivative turning points")
    require(exact(cfg["limit_exact"]) == n * step, "grid limit mismatch")
    require(grid["nodes_prescribed"] == grid["nodes_completed"] ==
            len(grid["rows"]) == 2 * n + 1, "incomplete node list")
    nodes = []
    for j, row in zip(range(-n, n + 1), grid["rows"]):
        z = j * step
        require(row["index"] == j and exact(row["z_exact"]) == z,
                "grid index/location mismatch")
        phi, cdf = saved(row["phi"]), saved(row["Phi"])
        require(phi.lo > 0 and cdf.hi >= 0 and cdf.lo <= 1,
                "invalid Gaussian density/CDF range")
        cdf = I(max(F(0), cdf.lo), min(F(1), cdf.hi))
        nodes.append((z, phi, cdf))
    cells = []
    for left, right in zip(nodes, nodes[1:]):
        a, pa, ca = left
        b, pb, cb = right
        require(not any(a < t < b for t in (-1, 0, 1)), "turning point crossed")
        cells.append({"left": a, "right": b, "cdf_lower": ca.lo,
                      "cdf_upper": cb.hi, "W1": abs(pb - pa),
                      "W2": abs((-b) * pb - (-a) * pa)})
    return {"nodes": nodes, "cells": cells, "limit": n * step,
            "phi_limit_upper": nodes[-1][1].hi}

def weight(prepared, kind, exponent, delta):
    require(kind in ("W1", "W1plus", "W2"), "unknown comparison weight")
    require(isinstance(exponent, int) and not isinstance(exponent, bool) and exponent >= 0,
            "nonnegative integer exponent required")
    delta = exact(delta)
    require(delta >= 0, "negative CDF shift")
    lower, upper, count = F(0), F(0), 0
    for cell in prepared["cells"]:
        if kind == "W1plus" and cell["left"] < 0:
            continue
        mass = cell["W2" if kind == "W2" else "W1"]
        wl = min(F(1), max(F(0), cell["cdf_lower"] + delta)) ** exponent
        wh = min(F(1), max(F(0), cell["cdf_upper"] + delta)) ** exponent
        lower += mass.lo * wl
        upper += mass.hi * wh
        count += 1
    tail = prepared["phi_limit_upper"] * (
        1 if kind == "W1plus" else 2 * prepared["limit"] if kind == "W2" else 2)
    return {"lower_sum": lower, "upper_sum": upper, "tail": tail,
            "enclosure": I(lower, upper + tail), "cells": count}

def five_budgets(p):
    """P5-P9, rational interval operations independently of Arb evaluation."""
    s = p["s"]
    require(s.lo > 0, "scale must be strictly positive")
    nonnegative = ("L0", "L1", "L2", "deltaG", "deltaf", "deltap", "Vdiff",
                   "a2_squared", "R0", "RCDF", "M2", "Mmv", "Mcv",
                   "W1_q_Delta", "W1_q_minus1_Delta", "W2_q_deltaG",
                   "W1plus_q_minus1_deltaG")
    require(all(p[k].lo >= 0 for k in nonnegative), "negative norm/active/weight input")
    l0, l1, l2 = (p[k] for k in ("L0", "L1", "L2"))
    dg, df, dp = (p[k] for k in ("deltaG", "deltaf", "deltap"))
    w98 = p["W1_q_Delta"] + p["Vdiff"]
    w97 = p["W1_q_minus1_Delta"] + p["Vdiff"]
    linear = (2 * p["R0"] + p["J_K"] * p["RCDF"]) / 99
    quadratic = p["a2_squared"] + 2 * df * dg + 49 * dg ** 2 * (
        p["W1plus_q_minus1_deltaG"] + df)
    return {
        "E1": 50 * l2 * p["M2"] / s ** 3,
        "E2": 4950 * p["Mmv"] / s ** 3 * (l2 / 99 + l1 * w98),
        "E3": 4950 * p["Mcv"] / s ** 4 * (l2 * w98 + 98 * l1 * l0 * w97),
        "E4": 9900 * abs(p["cbar"]) / s ** 2 * (dp / 99 + dg * p["W2_q_deltaG"]),
        "E5": 9900 * abs(p["m_H"]) / s * (linear + quadratic),
    }, linear, quadratic

def fixtures(grid):
    """Five saved production nodes; no new Gaussian function evaluations."""
    source = {exact(row["z_exact"]): row for row in grid["rows"]}
    mini = deepcopy(grid)
    mini["configuration"].update(step_exact="1", N=2, limit_exact="2")
    mini["nodes_prescribed"] = mini["nodes_completed"] = 5
    mini["rows"] = [dict(deepcopy(source[F(j)]), index=j) for j in range(-2, 3)]
    prep = prepare_grid(mini)
    phi = {z: p for z, p, _ in prep["nodes"]}
    expected = {"W1": 2 * phi[F(0)], "W1plus": phi[F(0)], "W2": 4 * phi[F(1)]}
    identities = []
    for kind in expected:
        for power, delta in ((0, F(0)), (98, F(1))):
            w = weight(prep, kind, power, delta)
            ok = w["enclosure"].contains(expected[kind])
            require(ok, "unit-weight Gaussian mass identity failed")
            identities.append({"kind": kind, "exponent": power, "delta_exact": str(delta),
                               "contains_closed_mass_interval": ok})
    invalid = []
    def rejects(name, fn):
        try:
            fn()
        except (ValueError, KeyError):
            invalid.append(name)
        else:
            raise ValueError("invalid-input fixture accepted: " + name)
    rejects("float_rational", lambda: exact(0.5))
    rejects("bool_rational", lambda: exact(True))
    rejects("inverted_interval", lambda: I(2, 1))
    rejects("zero_denominator", lambda: point(1) / point(0))
    rejects("negative_weight_shift", lambda: weight(prep, "W1", 98, -1))
    rejects("bool_weight_exponent", lambda: weight(prep, "W1", True, 0))
    bad = deepcopy(mini)
    bad["rows"][0]["index"] += 1
    rejects("node_sequence", lambda: prepare_grid(bad))
    bad = deepcopy(mini)
    bad["configuration"]["step_exact"] = "2/3"
    rejects("grid_turning_points", lambda: prepare_grid(bad))
    p = {k: point(0) for k in ("L0", "L1", "L2", "deltaG", "deltaf", "deltap", "Vdiff",
         "a2_squared", "R0", "RCDF", "M2", "Mmv", "Mcv", "W1_q_Delta",
         "W1_q_minus1_Delta", "W2_q_deltaG", "W1plus_q_minus1_deltaG", "m_H", "cbar", "J_K")}
    p.update(s=point(2), m_H=point(F(1, 8)), cbar=point(F(-1, 16)), J_K=point(3))
    all_zero, _, _ = five_budgets(p)
    require(all(x == point(0) for x in all_zero.values()), "all-zero budget fixture")
    p.update({k: point(1) for k in ("L0", "L1", "L2", "M2", "Mmv", "Mcv")})
    zero_noise, _, _ = five_budgets(p)
    require(zero_noise["E4"] == zero_noise["E5"] == point(0), "zero-noise fixture")
    require(all(zero_noise[k].lo > 0 for k in ("E1", "E2")), "active terms accidentally removed")
    p["deltaf"] = point(-1)
    rejects("negative_norm_budget", lambda: five_budgets(p))
    return {"saved_nodes_in_fixture": 5, "new_manufactured_nodes": 0,
            "mass_identity_checks": identities, "invalid_inputs_rejected": invalid,
            "all_zero_budget_fixture_passed": True, "zero_noise_E4_E5_fixture_passed": True}

def validate(results, grid):
    """Return a deterministic receipt; unresolved sign is never a failure."""
    require(results.get("schema") == "r02_remainder_results_v1", "results schema")
    data = results["propagation"]
    require(data.get("schema") == "r02_remainder_finite_propagation_v1", "propagation schema")
    require(data["configuration"]["K"] == 100 and data["configuration"]["q"] == 98,
            "fixed model K/q required")
    require([r["n"] for r in data["rows"]] == NS, "fixed n sequence required")
    prep = prepare_grid(grid)
    require(grid["configuration"]["step_exact"] == "1/128" and
            grid["configuration"]["N"] == 1536 and prep["limit"] == 12,
            "frozen production comparison grid required")
    rows, weight_cell_uses = [], 0
    for row in data["rows"]:
        n = row["n"]
        p = {k: saved(v) for k, v in row["propagation_inputs"].items()}
        require(p["s"].lo == exact(row["s_lower_exact"]) and
                p["s"].hi == exact(row["s_upper_exact"]), "scale endpoint aliases")
        require((p["s"] ** 2).contains(n * p["v"]), "scale square must contain n*v")
        require(p["deltaG"].hi <= 1 and p["deltaG"].lo >= 0, "CDF cap range")
        weights = {}
        for key, (kind, exponent, shift) in WEIGHTS.items():
            stored = row["weights"][key]
            delta = exact(stored["delta_upper_exact"])
            require(stored["kind"] == kind and stored["exponent"] == exponent and
                    delta == p[shift].hi, "weight metadata/input binding")
            w = weight(prep, kind, exponent, delta)
            checks = {
                "lower_sum_contains_exact_fraction": saved(stored["lower_riemann_sum"]).contains(w["lower_sum"]),
                "upper_sum_contains_exact_fraction": saved(stored["upper_riemann_sum"]).contains(w["upper_sum"]),
                "tail_contains_exact_fraction": saved(stored["tail_upper"]).contains(w["tail"]),
                "enclosure_contains_fraction_riemann_bounds": saved(stored["enclosure"]).contains(w["enclosure"]),
                "upper_cap_covers_fraction_upper": saved(stored["upper"]).hi >= w["enclosure"].hi,
                "propagation_weight_covers_saved_upper_cap": p[key].contains(saved(stored["upper"]).hi),
                "cell_count_matches": stored["saved_cells_used"] == w["cells"],
                "no_new_grid_evaluations": stored["new_grid_function_evaluations"] == 0,
            }
            require(all(checks.values()), f"n={n} {key} conservative weight check: {checks}")
            weights[key] = {"checks": checks, "cells_used": w["cells"],
                            "fraction_riemann_enclosure": rational_summary(w["enclosure"])}
            weight_cell_uses += w["cells"]
        for name in ("M2", "Mmv", "Mcv"):
            require(p[name + "_upper"].hi >= p[name].hi, "active upper endpoint direction")
        require(p["Wg_q"].overlaps(p["W1_q_Delta"] + p["Vdiff"]) and
                p["Wg_q_minus1"].overlaps(p["W1_q_minus1_Delta"] + p["Vdiff"]),
                "Gaussian plus L1 weight reconstruction")
        built, linear, quadratic = five_budgets(p)
        budget_checks = {k: relation(saved(row["budgets"][k]), v) for k, v in built.items()}
        require(all(x["overlap"] for x in budget_checks.values()), f"n={n} budget disjoint")
        require(saved(row["I_linear_error_budget"]).overlaps(linear) and
                saved(row["I_quadratic_error_budget"]).overlaps(quadratic), "E5 component disjoint")
        total = sum(built.values(), point(0))
        total_check = relation(saved(row["E_total"]), total)
        require(total_check["overlap"], "total budget disjoint")
        cap = saved(row["E_total_upper"])
        require(cap.hi >= saved(row["E_total"]).hi, "total upper cap direction")
        d2 = -p["m_H"] * p["J_K"] / p["s"] + p["second_order_coefficient"] / n
        d2_check = relation(saved(row["D_second_order"]), d2)
        require(d2_check["overlap"], "second-order predictor disjoint")
        # These three objects intentionally remain distinct. Tiny endpoint
        # differences between natural interval arithmetic and Arb do not imply
        # one includes the other. The final outer enclosure is the hard gate.
        independent = I(d2.lo - total.hi, d2.hi + total.hi)
        from_saved = saved(row["D_second_order"]) + I(-cap.hi, cap.hi)
        final = saved(row["theoretical_D_interval"])
        require(final == saved(row["D_n_theoretical_enclosure"]), "final enclosure aliases")
        require(final.contains(independent), "final enclosure misses independent conservative reconstruction")
        require(final.contains(from_saved), "final enclosure misses saved predictor plus saved total cap")
        sign = final.hi < 0
        require(row["negative_sign_certified"] == sign and
                row["status"] == ("negative" if sign else "unresolved"), "sign/status mismatch")
        require(row["bound_qualification_passed"] is True and
                row["target_D_n_evaluated"] is False and
                row["all_error_budget_lower_endpoints_are_not_actual_error_lower_bounds"] is True,
                "qualification/interpretation flags")
        rows.append({"n": n, "weights": weights, "budgets": budget_checks,
                     "total_budget": total_check,
                     "saved_total_upper_covers_fraction_total_upper": cap.hi >= total.hi,
                     "D_second_order": d2_check,
                     "fraction_D_n_enclosure": rational_summary(independent),
                     "final_enclosure_contains_fraction_reconstruction": True,
                     "final_enclosure_contains_saved_D2_plus_saved_cap": True,
                     "negative_sign_certified": sign, "sign_gate_is_not_validation_gate": True})
    negative = [x["n"] for x in rows if x["negative_sign_certified"]]
    unresolved = [x["n"] for x in rows if not x["negative_sign_certified"]]
    require(data["negative_certified_n"] == negative and data["unresolved_n"] == unresolved,
            "top-level sign classification")
    fixture_report = fixtures(grid)
    return {"schema": "r02_remainder_fraction_check_v1", "passed": True,
            "method": "Independent Python Fraction arithmetic; no production imports",
            "rows": rows, "negative_certified_n": negative, "unresolved_n": unresolved,
            "fixtures": fixture_report,
            "counts": {"saved_Gaussian_nodes_checked": len(prep["nodes"]),
                       "saved_cells_checked": len(prep["cells"]),
                       "comparison_integrals_reconstructed": 16,
                       "weighted_cell_terms_reconstructed": weight_cell_uses,
                       "budget_formulas_reconstructed": 20, "D2_rows_reconstructed": 4,
                       "final_conservative_enclosures_checked": 4,
                       "unit_weight_mass_identity_fixtures": 6,
                       "invalid_input_rejection_fixtures": len(fixture_report["invalid_inputs_rejected"]),
                       "zero_residual_fixtures": 2,
                       "new_production_nodes": 0, "new_manufactured_nodes": 0,
                       "new_Gaussian_function_evaluations": 0, "new_CF_evaluations": 0,
                       "new_D_n_evaluations": 0, "new_MC_draws": 0, "new_NPZ_reads": 0},
            "limits": ["Noise certificate and old active constants are supplied certified inputs, not re-proved here.",
                       "Arb/Fraction overlap is recorded separately from interval containment.",
                       "The final saved enclosure must contain the independent rational reconstruction.",
                       "Budget-expression lower endpoints are not lower bounds on actual errors.",
                       "No direct finite-n D_n or production integral is evaluated."]}

"""Exact algebra and rational caps for AngularWitness-v1; no quadrature.

The manufactured p values are independent rationals.  Two formal values of the
second latent coordinate apply only its moments E[X]=0 and E[X**2]=1; they are
not a Gaussian quadrature rule.  No project angle direction or target curvature
is selected or evaluated here.  Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/08_research_研究工作/AngularWitness_v1_20261010/MATH.md"
REPORT = ROOT / "artifacts/angular_witness_v1_20261010/MATH_CHECK.json"
BASE_COMMIT = "fd19c05ee181e95e8c2ce4040943b01ee18907e5"
HANDOFF_SHA256 = "762dd8160e2abad43e110a7f60c5e055a8f025bc98edd4ab542226acea88b7f2"


@dataclass(frozen=True)
class Jet:
    """Value, first derivative and second derivative (not Taylor coefficients)."""

    value: F
    first: F = F(0)
    second: F = F(0)

    def __add__(self, other):
        if not isinstance(other, Jet):
            other = Jet(F(other))
        return Jet(self.value + other.value, self.first + other.first,
                   self.second + other.second)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, -self.first, -self.second)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Jet) else -F(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, Jet):
            other = Jet(F(other))
        return Jet(self.value * other.value,
                   self.first * other.value + self.value * other.first,
                   self.second * other.value + 2 * self.first * other.first
                   + self.value * other.second)

    __rmul__ = __mul__


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def product(values):
    result = 1
    for value in values:
        result = result * value
    return result


def sigmoid_jet(p, eta_first, eta_second):
    variance = p * (1 - p)
    return Jet(p, variance * eta_first,
               variance * eta_second + variance * (1 - 2 * p) * eta_first**2)


def likelihood_jet(probabilities, label):
    return product(p if bit else 1 - p for p, bit in zip(probabilities, label))


def pair_polynomial(a, d, residual):
    return sum((a[i] * a[j] * (d[i] - d[j])**2 * residual[i] * residual[j]
                for i in range(len(a)) for j in range(i + 1, len(a))), F(0))


CASES = (
    {"name": "one_coordinate", "a": (F(3, 5),), "p": (F(2, 7),),
     "d": (F(1),), "h": (F(1),)},
    {"name": "three_coordinate", "a": (F(1, 3), F(2, 5), F(7, 8)),
     "p": (F(1, 5), F(3, 7), F(5, 6)),
     "d": (F(2, 3), F(-2, 3), F(1, 3)),
     "h": (F(-2, 3), F(1, 3), F(2, 3))},
    {"name": "four_coordinate_signed_loading",
     "a": (F(-4, 5), F(2, 3), F(1, 7), F(5, 6)),
     "p": (F(1, 5), F(2, 7), F(3, 8), F(4, 9)),
     "d": (F(1, 2), F(-1, 2), F(1, 2), F(-1, 2)),
     "h": (F(-1, 2), F(1, 2), F(1, 2), F(-1, 2))},
)
Z_VALUES = (F(-2, 3), F(0), F(5, 4))


def exact_fixture_suite():
    counts = {"label_z_cases": 0, "complete_product_second_derivative": 0,
              "first_derivative_reflection": 0, "pair_identity": 0,
              "gaussian_total_derivative_identity": 0,
              "loading_direction_derivative": 0, "loading_derivative_bound": 0,
              "common_rotation_pair_zero": 0, "own_target_conditional_zero": 0,
              "wrong_curvature_sign_rejected": 0,
              "omitted_path_acceleration_rejected": 0}
    case_records = []
    for case in CASES:
        a, p, d, h = (case[key] for key in ("a", "p", "d", "h"))
        m = len(a)
        require(sum(x*x for x in d) == 1, "fixture d must have exact unit norm")
        require(sum(x*x for x in h) == 1, "fixture h must have exact unit norm")
        require(all(0 < x < 1 for x in p), "fixture probabilities must be interior")
        csum = sum((d[i]-d[j])**2 for i in range(m) for j in range(i+1, m))
        require(csum == m-sum(d)**2 and csum <= m, "pair-weight identity")
        labels = tuple(itertools.product((0, 1), repeat=m))
        own_target_zero = F(0)
        for y in labels:
            r = tuple(F(bit)-prob for bit, prob in zip(y, p))
            A = product(prob if bit else 1-prob for prob, bit in zip(p, y))
            P = pair_polynomial(a, d, r)
            own_target_zero += A * P
            require(pair_polynomial(a, (F(1),)*m, r) == 0, "common rotation")
            counts["common_rotation_pair_zero"] += 1
            for z in Z_VALUES:
                counts["label_z_cases"] += 1
                D = tuple(prob*(1-prob) for prob in p)
                U = sum(ai*di*ri for ai, di, ri in zip(a, d, r))
                V = sum(ai*di*di*ri for ai, di, ri in zip(a, d, r))
                S = sum(ai*ri for ai, ri in zip(a, r))
                T = sum(ai*ai*di*di*vi for ai, di, vi in zip(a, d, D))
                path_jets, missing_jets = [], []
                # This exact moment functional is applied only to polynomials
                # of degree <=2 in X, after differentiation at t=0.
                for x in (F(-1), F(1)):
                    path_jets.append(likelihood_jet(
                        [sigmoid_jet(pi, ai*di*x, -ai*di*di*z)
                         for pi, ai, di in zip(p, a, d)], y))
                    missing_jets.append(likelihood_jet(
                        [sigmoid_jet(pi, ai*di*x, F(0))
                         for pi, ai, di in zip(p, a, d)], y))
                first = sum(j.first for j in path_jets)/2
                second = sum(j.second for j in path_jets)/2
                missing = sum(j.second for j in missing_jets)/2
                B = A*(U*U-T-z*V)
                require(first == 0, "reflection must remove integrated first derivative")
                counts["first_derivative_reflection"] += 1
                require(second == B, "independent product derivative disagrees")
                counts["complete_product_second_derivative"] += 1
                require(P == S*V-U*U, "pair expansion disagrees")
                counts["pair_identity"] += 1
                Az = likelihood_jet([sigmoid_jet(pi, ai, F(0))
                                     for pi, ai in zip(p, a)], y).first
                Vz = -sum(ai*ai*di*di*vi for ai, di, vi in zip(a, d, D))
                gaussian_derivative = Az*V + A*Vz-z*A*V
                require(second+A*P == gaussian_derivative,
                        "B+AP must be (AV)'-zAV")
                counts["gaussian_total_derivative_identity"] += 1
                wrong_sign_residual = second-A*P-gaussian_derivative
                missing_residual = missing+A*P-gaussian_derivative
                require(wrong_sign_residual == -2*A*P, "wrong-sign control drift")
                require(missing_residual == A*z*V, "acceleration control drift")
                counts["wrong_curvature_sign_rejected"] += int(wrong_sign_residual != 0)
                counts["omitted_path_acceleration_rejected"] += int(missing_residual != 0)

                aj = [Jet(ai, hi) for ai, hi in zip(a, h)]
                rj = [F(bit)-sigmoid_jet(pi, z*hi, F(0))
                      for bit, pi, hi in zip(y, p, h)]
                Pj = pair_polynomial(aj, d, rj)
                if not isinstance(Pj, Jet):  # Empty pair sum at m=1.
                    Pj = Jet(Pj)
                direct = sum(((d[i]-d[j])**2 * (
                    (h[i]*a[j]+a[i]*h[j])*r[i]*r[j]
                    - z*a[i]*a[j]*(D[i]*h[i]*r[j]+r[i]*D[j]*h[j]))
                    for i in range(m) for j in range(i+1, m)), F(0))
                require(Pj.first == direct, "loading derivative product rule")
                counts["loading_direction_derivative"] += 1
                cap_a = max(map(abs, a))
                require(abs(direct) <= 2*m*cap_a+F(m, 2)*cap_a**2*abs(z),
                        "pointwise loading derivative bound")
                counts["loading_derivative_bound"] += 1
        require(own_target_zero == 0, "conditional own-target curvature cancellation")
        counts["own_target_conditional_zero"] += 1
        case_records.append({"name": case["name"], "m": m,
                             **{key: list(map(str, case[key])) for key in ("a", "p", "d", "h")}})
    require(counts["wrong_curvature_sign_rejected"] > 0, "sign control never exercised")
    require(counts["omitted_path_acceleration_rejected"] > 0, "acceleration control never exercised")
    return {"cases": case_records, "z_exact": list(map(str, Z_VALUES)),
            "counts": counts, "all_required_identities_exact": True,
            "formal_X_moment_functional": "mean at -1 and +1, only degree<=2; not quadrature"}


def rational_caps():
    sqrt8_upper, sqrt68_upper = F(2829, 1000), F(8247, 1000)
    require(sqrt8_upper > 0 and sqrt8_upper**2 > 8, "sqrt(8) rational cap")
    require(sqrt68_upper > 0 and sqrt68_upper**2 > 68, "sqrt(68) rational cap")
    M1_upper = (8+sqrt68_upper)/2
    L_upper = 16+(4+16*sqrt8_upper)*M1_upper
    require(L_upper < 417, "L_ang integer cap")
    eta = F("1.805e-8")
    transfer = 417*eta
    require(transfer == F("7.52685e-6"), "transfer product")
    # Public K.21 transcription, used only for an amplitude envelope; no target
    # likelihood, posterior or angle curvature is evaluated from these values.
    center_first_four = (F(4236614515730759, 9007199254740992),
                         F(1274451243427741, 2251799813685248),
                         F(181129089615167, 281474976710656),
                         F(775431105319425, 1125899906842624))
    require(all(0 < value-eta and value+eta < 1 for value in center_first_four),
            "positive row lengths and A=1 must hold through root segment")
    return {"m": 8, "A_exact": "1", "sqrt8_upper_exact": str(sqrt8_upper),
            "sqrt68_upper_exact": str(sqrt68_upper),
            "sqrt8_square_excess_exact": str(sqrt8_upper**2-8),
            "sqrt68_square_excess_exact": str(sqrt68_upper**2-68),
            "M1_strict_upper_exact": str(M1_upper),
            "L_ang_strict_upper_exact": str(L_upper),
            "L_ang_upper_below_417": True, "integer_cap": 417,
            "root_distance_strict_upper_exact": str(eta),
            "transfer_strict_upper_exact": str(transfer),
            "transfer_strict_upper_decimal_exact": "0.00000752685",
            "center_first_four_exact_from_K21": list(map(str, center_first_four)),
            "positive_A1_root_segment_verified": True,
            "full_24_dimensional_risk_Hessian_Lipschitz_certified": False}


def make_report():
    return {"schema": "angular_witness_exact_math_check_v1",
            "status": "passed", "role": "author_internal_math_execution_check",
            "base_commit": BASE_COMMIT, "feedback_sha256_at_read": HANDOFF_SHA256,
            "source_bindings": {
                Path(__file__).relative_to(ROOT).as_posix(): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                DOC.relative_to(ROOT).as_posix(): hashlib.sha256(DOC.read_bytes()).hexdigest()},
            "fixture_suite": exact_fixture_suite(), "rational_caps": rational_caps(),
            "scope": {"standard_library_only": True, "exact_fraction_arithmetic": True,
                      "new_Gaussian_integrals": 0, "Gaussian_density_evaluations": 0,
                      "sigmoid_or_transcendental_evaluations": 0,
                      "project_direction_selections": 0, "project_target_curvature_evaluations": 0,
                      "scientific_NPZ_or_NPY_reads": 0, "new_random_draws": 0,
                      "new_optimizer_updates": 0, "external_review_runs": 0,
                      "scientific_negative_curvature_certified_by_this_check": False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="save a new exact check receipt; refuses overwrite")
    mode.add_argument("--check", action="store_true", help="recompute algebra and compare the saved receipt")
    args = parser.parse_args()
    result = make_report()
    if args.write:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        saved = json.loads(REPORT.read_text(encoding="utf-8"))
        require(result == saved, "saved exact math receipt or source bindings differ")
    print(json.dumps({"status": "passed", "mode": "write" if args.write else "check",
                      "counts": result["fixture_suite"]["counts"],
                      "L_ang_below_417": True, "transfer_strict_upper": "7.52685e-6",
                      "new_Gaussian_integrals": 0, "project_target_curvature_evaluations": 0,
                      "scientific_NPZ_or_NPY_reads": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

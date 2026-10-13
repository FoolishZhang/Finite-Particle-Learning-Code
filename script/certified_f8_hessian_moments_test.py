"""Manufactured finite-measure moment identities, no scientific integration."""
from __future__ import annotations

from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/".cache/finite_selection_tools"))
from flint import arb, ctx
import certified_f8_hessian_moments as h
from certified_f8_gradient_symmetry import paired_counts, label_bits

OUT = ROOT/"artifacts/certified_f8_hessian_v1_20261008"


def aq(value):
    value = Q(value)
    return arb(value.numerator)/value.denominator


def matrix_zero(m):
    return [[arb(0) for _ in range(m)] for _ in range(m)]


def manufacture_nodes():
    # Logistic loadings log(2),log(3),log(3),log(2); rational odds at z=-1,0,1.
    nodes = []
    for z, weight in ((-1, Q(1, 4)), (0, Q(1, 2)), (1, Q(1, 4))):
        odds = [Q(base**z) if z >= 0 else Q(1, base**(-z)) for base in (2, 3)]
        ps = [aq(x/(1+x)) for x in odds]
        nodes.append((arb(z), aq(weight), [ps[0], ps[1], ps[1], ps[0]]))
    return nodes


def candidate_for_count(count, nodes, u):
    m = 4
    T = [arb(0) for _ in range(3)]
    P = [[arb(0) for _ in range(m)] for _ in range(3)]
    PP = {l: matrix_zero(m) for l in (0, 2)}
    Psi = arb(0)
    for z, w, p in nodes:
        A = arb(1)
        for j, c in enumerate(count):
            A *= p[j]**c*(1-p[j])**(2-c)
        exponential = (-u*A).exp()
        t = u*A*exponential
        Psi += w*exponential
        for l in range(3):
            wt = w*t*z**l
            T[l] += wt
            for i in range(m):
                P[l][i] += wt*p[i]
            if l in PP:
                for i in range(m):
                    for j in range(m):
                        PP[l][i][j] += wt*p[i]*p[j]
    return Psi, {"T": T, "P": P, "PP0": PP[0], "PP2": PP[2]}


def direct_label(label, nodes, u, K=8):
    m, y = 4, label_bits(label, 4)
    T0, Psi = arb(0), arb(0)
    R = [[arb(0) for _ in range(m)] for _ in range(2)]
    BB = [matrix_zero(m) for _ in range(3)]
    GG = [matrix_zero(m) for _ in range(3)]
    for z, w, p in nodes:
        A = arb(1)
        for yi, pi in zip(y, p):
            A *= pi if yi else 1-pi
        expval = (-u*A).exp()
        wt = w*u*A*expval
        T0 += wt; Psi += w*expval
        r = [arb(yi)-pi for yi, pi in zip(y, p)]
        for s in (0, 1):
            for i in range(m):
                R[s][i] += wt*z**s*r[i]
        for l in range(3):
            for i in range(m):
                BB[l][i][i] += wt*z**l*p[i]*(1-p[i])
                for j in range(m):
                    GG[l][i][j] += wt*z**l*r[i]*r[j]
    first, second = K*Psi**(K-1), K*(K-1)*Psi**(K-2)
    samecross = lambda l, s, t: [[first*BB[l][i][j]-second*(T0*GG[l][i][j]-R[s][i]*R[t][j])
                                 for j in range(m)] for i in range(m)]
    perpendicular = [[first*BB[0][i][j]-second*T0*GG[0][i][j] for j in range(m)] for i in range(m)]
    return {"Hbb": samecross(0, 0, 0), "Haa": samecross(2, 1, 1),
        "Hperp": perpendicular, "mixed": samecross(1, 0, 1)}


def weights(complement_symmetric=False):
    vals = [(label+1)**2 for label in range(16)]
    if complement_symmetric:
        vals = [x+vals[15-label] for label, x in enumerate(vals)]
    return [Q(x, sum(vals)) for x in vals]


class Manufactured(unittest.TestCase):
    def setUp(self):
        self.previous = ctx.prec
        ctx.prec = 192

    def tearDown(self):
        ctx.prec = self.previous

    def assert_same_matrix(self, a, b):
        self.assertEqual(len(a), len(b))
        for ar, br in zip(a, b):
            for x, y in zip(ar, br):
                difference = x-y
                if isinstance(difference, arb):
                    self.assertTrue(difference.contains(0), str(difference))
                else:
                    self.assertEqual(difference, 0)

    def test_folded_moments_mass_binary_diagonal_and_no_double_self_count(self):
        q = weights()
        folded = h.fold_teacher_moments(q, 4)
        self.assertEqual(sum(x["W"] for x in folded["moments"]), 1)
        self.assertEqual(len(folded["moments"]), 5)
        self.assertFalse(folded["within_pair_exchangeability_assumed"])
        self.assertFalse(folded["mixed_bias_radial_block_preserved_by_this_fold"])
        for label in range(16):
            if paired_counts(label, 4) == (1, 1):
                self.assertFalse(folded["label_reflected_to_representative"][label])
        for moments in folded["moments"]:
            self.assertEqual([moments["S"][i][i] for i in range(4)], moments["M"])

    def test_nonexchangeable_label_weights_preserved(self):
        q = weights(True)
        self.assertNotEqual(q[1], q[8])
        folded = h.fold_teacher_moments(q, 4)
        for ri, moments in enumerate(folded["moments"]):
            expected = Q(0)
            for label, weight in enumerate(q):
                if folded["label_to_representative_index"][label] == ri:
                    y = label_bits(label, 4)
                    if folded["label_reflected_to_representative"][label]:
                        y = tuple(1-x for x in y)
                    expected += weight*y[0]*y[1]
            self.assertEqual(moments["S"][0][1], expected)

    def test_full_label_direct_integrands_match_folded_moments_arbitrary_q(self):
        nodes, u = manufacture_nodes(), aq(Q(9, 7))
        for symmetric in (False, True):
            q = weights(symmetric)
            folded = h.fold_teacher_moments([aq(x) for x in q], 4)
            direct = {name: matrix_zero(4) for name in ("Hbb", "Haa", "Hperp")}
            for label, weight in enumerate(q):
                values = direct_label(label, nodes, u)
                for name in direct:
                    for i in range(4):
                        for j in range(4):
                            direct[name][i][j] += aq(weight)*values[name][i][j]
            grouped = {name: matrix_zero(4) for name in direct}
            for count, moments in zip(folded["representatives"], folded["moments"]):
                Psi, candidate = candidate_for_count(count, nodes, u)
                values = h.hessian_integrand(Psi, moments, candidate)
                for name in grouped:
                    for i in range(4):
                        for j in range(4):
                            grouped[name][i][j] += values[name][i][j]
            for name in direct:
                self.assert_same_matrix(direct[name], grouped[name])

    def test_complement_blocks_invariant_but_mixed_changes_sign(self):
        nodes, u = manufacture_nodes(), aq(Q(9, 7))
        for label in range(16):
            a, b = direct_label(label, nodes, u), direct_label(15-label, nodes, u)
            for name in ("Hbb", "Haa", "Hperp"):
                self.assert_same_matrix(a[name], b[name])
            self.assert_same_matrix(a["mixed"], [[-x for x in row] for row in b["mixed"]])

    def test_exact_teacher_complement_symmetry_mixed_zero_not_for_arbitrary_weights(self):
        nodes, u = manufacture_nodes(), aq(Q(9, 7))
        results = []
        for symmetry in (True, False):
            mixed = matrix_zero(4)
            for label, weight in enumerate(weights(symmetry)):
                values = direct_label(label, nodes, u)["mixed"]
                for i in range(4):
                    for j in range(4):
                        mixed[i][j] += aq(weight)*values[i][j]
            results.append(mixed)
        self.assertTrue(all(x.contains(0) for row in results[0] for x in row))
        self.assertTrue(any(not x.contains(0) for row in results[1] for x in row))

    def test_Hbb_M_S_cancel_exact_rational_algebra(self):
        W, M, S = Q(7), [Q(2), Q(3)], [[Q(2), Q(1)], [Q(1), Q(3)]]
        t0, p = Q(3, 2), [Q(1, 4), Q(2, 5)]
        pp = [[Q(1, 9), Q(1, 11)], [Q(1, 11), Q(2, 13)]]
        for i in range(2):
            for j in range(2):
                GG = S[i][j]*t0-M[j]*p[i]-M[i]*p[j]+W*pp[i][j]
                RR = S[i][j]*t0*t0-M[i]*t0*p[j]-M[j]*p[i]*t0+W*p[i]*p[j]
                self.assertEqual(t0*GG-RR, W*(t0*pp[i][j]-p[i]*p[j]))

    def test_general_Haa_same_cross_matches_simplification(self):
        moment = {"W": Q(7), "M": [Q(2), Q(3)], "S": [[Q(2), Q(1)], [Q(1), Q(3)]]}
        candidate = {"T": [Q(3, 2), Q(2, 3), Q(5, 4)],
            "P": [[Q(1, 4), Q(2, 5)], [Q(1, 6), Q(1, 7)], [Q(2, 9), Q(1, 8)]],
            "PP0": [[Q(1, 9), Q(1, 11)], [Q(1, 11), Q(2, 13)]],
            "PP2": [[Q(1, 12), Q(1, 15)], [Q(1, 15), Q(1, 10)]]}
        parts = h.contractions(moment, candidate)
        W, M, S, t = moment["W"], moment["M"], moment["S"], candidate["T"]
        p1, p2, pp2 = candidate["P"][1], candidate["P"][2], candidate["PP2"]
        for i in range(2):
            for j in range(2):
                GG = S[i][j]*t[2]-M[j]*p2[i]-M[i]*p2[j]+W*pp2[i][j]
                RR = S[i][j]*t[1]*t[1]-M[i]*t[1]*p1[j]-M[j]*p1[i]*t[1]+W*p1[i]*p1[j]
                self.assertEqual(parts["Daa"][i][j], t[0]*GG-RR)

    def test_full_coordinate_pair_expansion_preserves_cross_pair_covariances(self):
        expanded = h.expand_pair_moments([Q(1), Q(2), Q(3)],
            [[Q(l+i) for i in range(4)] for l in range(3)], list(map(Q, range(10))), list(map(Q, range(10, 20))))
        self.assertEqual(len(expanded["PP0"]), 8)
        self.assertEqual(expanded["P"][0], [Q(0), Q(1), Q(2), Q(3), Q(3), Q(2), Q(1), Q(0)])
        self.assertEqual(expanded["PP0"][0][7], expanded["PP0"][0][0])
        self.assertEqual(expanded["PP2"][2][6], expanded["PP2"][6][2])

    def test_zero_class_mass_safe_without_division(self):
        candidate = {"T": [Q(1), Q(2), Q(3)], "P": [[Q(1, 2)]*2 for _ in range(3)],
                     "PP0": [[Q(1, 4)]*2 for _ in range(2)], "PP2": [[Q(1, 4)]*2 for _ in range(2)]}
        empty = {"W": Q(0), "M": [Q(0)]*2, "S": [[Q(0)]*2 for _ in range(2)]}
        result = h.hessian_integrand(Q(1, 2), empty, candidate)
        for name in ("Hbb", "Haa", "Hperp"):
            self.assertEqual(result[name], [[0, 0], [0, 0]])

    def test_K1_does_not_compute_negative_power_for_cross_coefficient(self):
        candidate = {"T": [Q(1), Q(2), Q(3)], "P": [[Q(1, 2)]*2 for _ in range(3)],
                     "PP0": [[Q(1, 4)]*2 for _ in range(2)], "PP2": [[Q(1, 4)]*2 for _ in range(2)]}
        moment = {"W": Q(1), "M": [Q(0)]*2, "S": [[Q(0)]*2 for _ in range(2)]}
        result = h.hessian_integrand(Q(0), moment, candidate, K=1)
        self.assertEqual(result["Hbb"], [[Q(1, 4), 0], [0, Q(1, 4)]])

    def test_column_and_teacher_transfer_contract(self):
        columns = h.moment_columns()
        self.assertEqual(columns["total_scalar_columns"], 36)
        self.assertEqual(columns["columns_excluding_Psi"], 35)
        self.assertEqual(columns["columns_excluding_separate_Psi_T0"], 34)
        transfer = h.teacher_transfer_constants()
        self.assertEqual(transfer["full_ba_invariant_block_diagonal_coefficient_exact"], "585/4")
        self.assertEqual(transfer["transverse_coefficient_exact"], "33/4")
        self.assertFalse(transfer["uncertified_float_matrix_used_as_bound"])


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Manufactured))
    source_paths = ("script/certified_f8_hessian_moments.py", "script/certified_f8_hessian_moments_test.py")
    receipt = {"schema": "certified_f8_hessian_moments_manufactured_v1", "passed": result.wasSuccessful(),
        "tests_run": result.testsRun, "failures": [(str(case), text) for case, text in result.failures],
        "errors": [(str(case), text) for case, text in result.errors],
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths},
        "finite_non_Gaussian_manufactured_measure_only": True,
        "all_labels_compared_with_folded_moments_for_nonexchangeable_q": True,
        "moment_columns": h.moment_columns(), "teacher_transfer_constants": h.teacher_transfer_constants(),
        "real_scientific_array_reads": 0, "real_integrals": 0, "new_kernel_calls": 0,
        "new_random_draws": 0, "true_Hessian_certificate_created": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"MOMENTS_MANUFACTURED_CHECK.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "real_integrals": 0, "true_Hessian_certificate_created": False}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

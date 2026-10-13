"""Manufactured rational identities and nonexchangeable label-weight tests."""
from __future__ import annotations

from fractions import Fraction as Q
import hashlib
import itertools
import json
from pathlib import Path
import unittest

import certified_f8_gradient_symmetry as s

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/certified_f8_gradient_v1_20261008"


def probability_and_scaled_score(count, probabilities, z):
    """Return A and sqrt2*Gxi with exact rational logistic values."""
    likelihood = Q(1)
    for c, p in zip(count, probabilities):
        likelihood *= p**c*(1-p)**(2-c)
    score = tuple(z*(c-2*p) for c, p in zip(count, probabilities))
    return likelihood, score


def bank(count, zs, bases):
    values, scores = [], []
    for z in zs:
        ps = [Q(base**z) if z >= 0 else Q(1, base**(-z)) for base in bases]
        ps = [odds/(1+odds) for odds in ps]
        a, g = probability_and_scaled_score(count, ps, Q(z))
        values.append(a); scores.append(g)
    denominator = sum(values)
    return tuple(sum((a*g[j] for a, g in zip(values, scores)), Q(0))/denominator
                 for j in range(len(count)))


class Manufactured(unittest.TestCase):
    def test_default_mapping_has_complete_256_81_41_partition(self):
        mapping = s.symmetry_map()
        self.assertEqual(mapping["label_count"], 256)
        self.assertEqual(mapping["count_group_count"], 81)
        self.assertEqual(mapping["representative_count"], 41)
        self.assertEqual(mapping["representatives"], sorted(mapping["representatives"]))
        self.assertEqual(sorted(sum(mapping["representative_labels"], [])), list(range(256)))
        self.assertEqual(mapping["representatives"][-1], [1, 1, 1, 1])
        self.assertEqual(sum(len(x) == 1 for x in mapping["representative_count_group_indices"]), 1)

    def test_m4_mapping_and_MSB_reverse_pairs(self):
        mapping = s.symmetry_map(4)
        self.assertEqual(mapping["representative_count"], 5)
        self.assertEqual(mapping["pair_indices"], [[0, 3], [1, 2]])
        self.assertEqual(s.label_bits(8, 4), (1, 0, 0, 0))
        self.assertEqual(s.paired_counts(8, 4), (1, 0))
        self.assertEqual(s.paired_counts(1, 4), (1, 0))

    def test_count_multiplicity_is_enum_only(self):
        mapping = s.symmetry_map(8)
        for count, labels in zip(mapping["count_groups"], mapping["count_group_labels"]):
            self.assertEqual(len(labels), 2**count.count(1))

    def test_pointwise_count_and_sign_identity(self):
        probabilities, z = (Q(2, 3), Q(3, 4)), Q(3, 5)
        for c in itertools.product(range(3), repeat=2):
            a, g = probability_and_scaled_score(c, probabilities, z)
            aa, gg = probability_and_scaled_score(s.complement_count(c), tuple(1-p for p in probabilities), -z)
            self.assertEqual(a, aa)
            self.assertEqual(g, gg)

    def test_individual_labels_same_counts_same_A_and_projected_score(self):
        ps, z = (Q(2, 3), Q(3, 4)), Q(2, 5)
        for label in range(16):
            bits = s.label_bits(label, 4)
            full_ps = (ps[0], ps[1], ps[1], ps[0])
            A = Q(1)
            for y, p in zip(bits, full_ps):
                A *= p if y else 1-p
            scaled = tuple(z*(bits[j]+bits[3-j]-2*ps[j]) for j in range(2))
            ac, gc = probability_and_scaled_score(s.paired_counts(label, 4), ps, z)
            self.assertEqual(A, ac)
            self.assertEqual(scaled, gc)

    def test_normalized_particle_score_identity_under_bank_sign_flip(self):
        for c in itertools.product(range(3), repeat=2):
            self.assertEqual(bank(c, [-2, 0, 1], (2, 3)),
                bank(s.complement_count(c), [2, 0, -1], (2, 3)))

    def test_complement_equality_requires_prior_average_not_fixed_bank(self):
        c = (0, 0)
        self.assertNotEqual(bank(c, [1], (2, 3)), bank(s.complement_count(c), [1], (2, 3)))
        reflected = lambda count: tuple((x+y)/2 for x, y in zip(
            bank(count, [-2, 0, 1], (2, 3)), bank(count, [2, 0, -1], (2, 3))))
        self.assertEqual(reflected(c), reflected(s.complement_count(c)))

    def test_nonexchangeable_teacher_weights_are_summed_without_renormalization(self):
        weights = [Q((label+1)**2, 7) for label in range(16)]
        result = s.aggregate_weights(weights, 4)
        mapping = s.symmetry_map(4)
        self.assertNotEqual(weights[1], weights[8])
        self.assertEqual(s.paired_counts(1, 4), s.paired_counts(8, 4))
        for labels, total in zip(mapping["representative_labels"], result["representative_weights"]):
            self.assertEqual(total, sum((weights[label] for label in labels), Q(0)))
        self.assertEqual(sum(result["representative_weights"]), sum(weights))
        self.assertFalse(result["weights_renormalized"])
        self.assertFalse(result["multiplicities_used_as_weights"])

    def test_reduced_weighted_gradient_matches_full_labels_for_asymmetric_q(self):
        weights = [Q((label+1)**2, sum((i+1)**2 for i in range(16))) for label in range(16)]
        averaged = lambda c: tuple(-(x+y)/2 for x, y in zip(
            bank(c, [-2, 0, 1], (2, 3)), bank(c, [2, 0, -1], (2, 3))))
        full = tuple(sum((weights[label]*averaged(s.paired_counts(label, 4))[j]
                          for label in range(16)), Q(0)) for j in range(2))
        mapping, aggregates = s.symmetry_map(4), s.aggregate_weights(weights, 4)
        reduced = tuple(sum((weight*averaged(tuple(c))[j] for c, weight in zip(
            mapping["representatives"], aggregates["representative_weights"])), Q(0)) for j in range(2))
        self.assertEqual(full, reduced)

    def test_grouping_contracts_L1_of_arbitrary_signed_weight_errors(self):
        errors = [Q((-1)**label*(label+1), 13) for label in range(16)]
        aggregated = s.aggregate_weights(errors, 4)
        self.assertLessEqual(sum(map(abs, aggregated["representative_weights"])), sum(map(abs, errors)))

    def test_isometric_full_gradient_norm_equals_reduced_norm_conditionally(self):
        # Full ga=S*pair values; ||ga||²=2||pair||²=||gxi||².
        pair = (Q(2), Q(3))
        full = pair+tuple(reversed(pair))
        self.assertEqual(sum(x*x for x in full), 2*sum(x*x for x in pair))
        antisymmetric = (Q(1), Q(0), Q(0), Q(-1))
        self.assertEqual(tuple(antisymmetric[j]+antisymmetric[3-j] for j in range(2)), (0, 0))
        self.assertGreater(sum(x*x for x in antisymmetric), 0)

    def test_exact_pair_input_mapping_and_rejections(self):
        theta = [0.]*4+[.5, .6, .6, .5]
        self.assertEqual(s.exact_pair_parameters(theta, 4), (Q.from_float(.5), Q.from_float(.6)))
        for bad in ([1.]+theta[1:], [0.]*4+[.5, .6, .6, .51], [0.]*4+[0., .6, .6, 0.]):
            with self.assertRaises(ValueError):
                s.exact_pair_parameters(bad, 4)

    def test_invalid_mapping_dimensions_labels_and_weights_rejected(self):
        for dimension in (0, 3, True, 4.):
            with self.assertRaises(ValueError):
                s.symmetry_map(dimension)
        for label in (-1, 16, True, 1.):
            with self.assertRaises(ValueError):
                s.label_bits(label, 4)
        with self.assertRaises(ValueError):
            s.aggregate_weights([Q(1)]*15, 4)
        for count in ((), (0, 3), (True, 1), (1., 0)):
            with self.assertRaises(ValueError):
                s.representative(count)

    def test_exact_safe_transfer_constant_metadata(self):
        self.assertEqual(s.transfer_constants()["gradient_coefficient_squared_exact"], "64")
        self.assertEqual(s.transfer_constants()["legacy_full_radial_coefficient_squared_exact"], "648")
        self.assertFalse(s.transfer_constants()["legacy_full_radial_coefficient_used_in_this_batch"])
        self.assertFalse(s.transfer_constants()["uncertified_float_gradient_used_as_bound"])


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Manufactured))
    source_paths = ("script/certified_f8_gradient_symmetry.py", "script/certified_f8_gradient_symmetry_test.py")
    receipt = {"schema": "certified_f8_gradient_symmetry_manufactured_v1", "passed": result.wasSuccessful(),
        "tests_run": result.testsRun, "failures": [(str(case), text) for case, text in result.failures],
        "errors": [(str(case), text) for case, text in result.errors],
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths},
        "mapping": s.symmetry_map(), "safe_transfer_constant": s.transfer_constants(),
        "manufactured_rational_values_only": True, "real_scientific_array_reads": 0,
        "real_integrals": 0, "new_kernel_calls": 0, "new_random_draws": 0,
        "within_pair_teacher_exchangeability_assumed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"SYMMETRY_MANUFACTURED_CHECK.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "representatives": 41, "real_integrals": 0, "new_kernel_calls": 0}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

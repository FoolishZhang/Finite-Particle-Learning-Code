"""Small m2, h1/2, 25-point manufactured sums; no actual teacher grid."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import unittest
from unittest.mock import patch
import certified_teacher_compute as compute


class ManufacturedTests(unittest.TestCase):
    def test_orbits_partition_and_prefix_work(self):
        groups=compute.orbits(8)
        self.assertEqual(len(groups),72)
        self.assertEqual(sorted(x for group in groups for x in group),list(range(256)))
        self.assertLess(sum(map(len,compute.prefix_schedule([g[0] for g in groups],8))),510)

    def test_manufactured_finite_sum_mass_and_symmetry(self):
        rows=range(-2,3)
        block=compute.finite_grid_block(('-1/3','1/3'),'4/5','1/2',-2,2,rows,precision_bits=192,label_mode='all_labels')
        outcome=compute.aggregate_blocks([block],2,'all_labels')
        self.assertTrue(outcome['probability_sum_overlaps_grid_mass'])
        self.assertTrue(outcome['complement_symmetry_balls_overlap']);self.assertTrue(outcome['reversal_symmetry_balls_overlap'])
        self.assertEqual(block['work_counts']['grid_points'],25)
        self.assertFalse(outcome['true_PMF_enclosed'])
        global LAST_MANUFACTURED
        LAST_MANUFACTURED=dict(block=block,outcome=outcome)

    def test_exact_input_validation(self):
        with self.assertRaises(ValueError):compute.aq(.8)
        with self.assertRaises(ValueError):compute.validate_model(('-1/3','1/2'),'4/5','complement_reversal_orbits')
        with self.assertRaises(ArithmeticError):compute.record(compute.arb('nan'))

    def test_expired_budget_keeps_partial_and_restores_precision(self):
        before=compute.ctx.prec
        with self.assertRaises(compute.GridBudgetStop) as captured:
            compute.finite_grid_block(('-1/3','1/3'),'4/5','1/2',-2,2,range(-2,3),deadline=0.)
        self.assertEqual(compute.ctx.prec,before)
        self.assertFalse(captured.exception.partial['completed'])
        self.assertEqual(captured.exception.partial['work_counts']['grid_points'],0)
        self.assertIsNone(captured.exception.partial['last_completed_point'])

    def test_partitioned_sum_overlaps_complete_manufacture(self):
        args=(('-1/3','1/3'),'4/5','1/2',-2,2)
        first=compute.finite_grid_block(*args,(-2,-1),label_mode='all_labels')
        second=compute.finite_grid_block(*args,(0,1,2),label_mode='all_labels')
        full=compute.finite_grid_block(*args,range(-2,3),label_mode='all_labels')
        aggregate=compute.aggregate_blocks([first,second],2,'all_labels')
        full_sum=compute.aggregate_blocks([full],2,'all_labels')
        previous=compute.ctx.prec;compute.ctx.prec=192
        try:
            for left,right in zip(aggregate['pmf_finite_sum'],full_sum['pmf_finite_sum']):
                self.assertTrue(compute.from_record(left).overlaps(compute.from_record(right)))
        finally:compute.ctx.prec=previous


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests))
    if not outcome.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:
        compute.OUT.mkdir(parents=True,exist_ok=True)
        paths=[Path(compute.__file__).resolve(),Path(__file__).resolve()]
        receipt=dict(schema='certified_teacher_compute_manufactured_check_v1',created_utc=datetime.now(timezone.utc).isoformat(),
            status='passed',tests_run=outcome.testsRun,source_sha256={p.relative_to(compute.ROOT).as_posix():compute.sha(p) for p in paths},
            manufactured_finite_grid_calls=5,manufactured_grid_points=75,manufactured_m=2,manufactured_h='1/2',
            expired_budget_manufactured_calls=1,expired_budget_points=0,new_kernel_calls=0,acb_integral_calls=0,
            real_teacher_grid_calls=0,real_teacher_grid_points=0,new_random_draws=0,new_optimizer_updates=0,
            source_input_floats_used_as_certified_coefficients=False,manufactured_readout=LAST_MANUFACTURED)
        compute.write_new(compute.OUT/'COMPUTE_MANUFACTURED_CHECK.json',receipt)


if __name__=='__main__':main()

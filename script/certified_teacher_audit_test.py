"""Manufactured small teacher grids; zero scientific grid evaluations."""
import argparse
import unittest
import certified_teacher_audit as audit
import certified_teacher_compute as common

class Tests(unittest.TestCase):
    def test_distinct_algebra_all_small_labels(self):
        phi=('-1/5','1/10','4/5')
        direct=audit.selected_grid(phi,'3/5','1/2',2,range(8))
        produced=common.finite_grid_block(phi,'3/5','1/2',-2,2,range(-2,3))
        fixture=dict(pmf_finite_sum=produced['representative_balls'])
        self.assertEqual(len(audit.compare(direct,fixture)),8)
        fixture['pmf_finite_sum'][3]=common.record(common.arb(0))
        with self.assertRaises(ValueError):audit.compare(direct,fixture)
    def test_budget_retains_zero_point_partial(self):
        with self.assertRaises(audit.AuditBudgetStop) as caught:
            audit.selected_grid(('1/5',),'1','1/2',2,[0],deadline=-1)
        self.assertEqual(caught.exception.partial['points'],0)
        self.assertFalse(caught.exception.partial['completed'])
    def test_duplicate_label_rejected(self):
        with self.assertRaises(ValueError):audit.selected_grid(('1/5',),'1','1/2',2,[0,0])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');a=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():raise SystemExit(1)
    if a.write_receipt:common.write_new(common.OUT/'AUDIT_MANUFACTURED_CHECK.json',dict(passed=True,
        tests_run=result.testsRun,source_sha256={name:common.sha(common.ROOT/name) for name in audit.SOURCES},
        manufactured_m3_grid_points_per_method=25,manufactured_grid_methods=2,
        real_teacher_grid_points=0,new_random_draws=0,scientific_array_reads=0))

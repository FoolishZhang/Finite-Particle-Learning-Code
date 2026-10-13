"""Manufactured norm conversion and implication-gate checks, no science reads."""
import argparse,unittest
import certified_teacher_compute as common
from flint import arb,ctx
import certified_f8_hessian_finalize as final

class Tests(unittest.TestCase):
    def test_frobenius_not_max_entry(self):
        self.assertTrue(final.frobenius_distance_upper([[arb(3),arb(0)],[arb(0),arb(4)]],[[arb(0)]*2 for _ in range(2)]).contains(5))
    def test_interval_uncertainty_retained(self):
        x=arb(1).union(arb(2));bound=final.frobenius_distance_upper([[x]],[[arb(0)]])
        self.assertTrue(bound>=2)
    def test_block_diagonal_keeps_directions(self):
        m=final.block_diagonal([[arb(1)]],[[arb(2)]])
        self.assertEqual(m,[[arb(1),arb(0)],[arb(0),arb(2)]])
    def test_declared_conditions_suffice(self):
        w=final.witness_logic(arb(1)/100000,arb(1)/100000,arb(7)/10**10,-arb(3)/100,arb(1),saved_floors_verified=True)
        self.assertTrue(w['fixed_step_objective_risk_conflict_certified'])
    def test_bad_Hessian_or_gradient_blocks_existence(self):
        for h,g in ((arb(2)/1000,arb(7)/10**10),(arb(1)/100000,arb(1)/100000)):
            w=final.witness_logic(h,arb(1)/100000,g,-arb(3)/100,arb(1),saved_floors_verified=True)
            self.assertFalse(w['true_F8_stationary_existence_certified'])
    def test_risk_failure_does_not_erase_existence(self):
        w=final.witness_logic(arb(1)/100000,arb(1)/100000,arb(7)/10**10,arb(1),arb(1),saved_floors_verified=True)
        self.assertTrue(w['true_F8_stationary_existence_certified'])
        self.assertFalse(w['positive_risk_at_stationary_point_certified'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write-receipt',action='store_true');args=p.parse_args();ctx.prec=192
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not outcome.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:common.write_new(final.OUT/'FINALIZE_MANUFACTURED_CHECK.json',dict(passed=True,tests_run=outcome.testsRun,
        sources={n:common.sha(common.ROOT/n) for n in final.SOURCES},real_Hessian_evaluations=0,scientific_data_reads=0,new_random_draws=0))

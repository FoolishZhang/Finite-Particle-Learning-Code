"""Manufactured checks for arithmetic and vector-L1 error composition."""
import argparse
import unittest
import certified_teacher_finalize as final
import certified_teacher_compute as common
from flint import arb,ctx

class Tests(unittest.TestCase):
    def test_global_error_is_added_once(self):
        old=ctx.prec;ctx.prec=192
        try:
            result=final.compose([common.aq('1/4'),common.aq('3/4')],[common.aq('1/5'),common.aq('4/5')],common.aq('1/1000'))
            self.assertTrue(result[2].contains(common.aq('101/1000')))
            self.assertFalse(result[2].contains(common.aq('102/1000')))
            self.assertTrue(result[4].contains(1))
        finally:ctx.prec=old
    def test_missing_probability_mass_rejected(self):
        with self.assertRaises(ValueError):final.compose([arb(1)/4],[arb(1)/4],arb(1)/100)
    def test_arithmetic_uncertainty_is_retained(self):
        a=(arb(1)/5).union(arb(3)/10);b=(arb(7)/10).union(arb(4)/5)
        result=final.compose([a,b],[arb(1)/4,arb(3)/4],arb(1)/1000)
        self.assertTrue(result[0]>arb(9)/100)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');a=parser.parse_args()
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not outcome.wasSuccessful():raise SystemExit(1)
    if a.write_receipt:
        common.write_new(common.OUT/'FINALIZE_MANUFACTURED_CHECK.json',dict(passed=True,tests_run=outcome.testsRun,
            source_sha256={name:common.sha(common.ROOT/name) for name in (final.SOURCE,'script/certified_teacher_finalize_test.py')},
            real_teacher_grid_points=0,scientific_data_reads=0,new_random_draws=0))

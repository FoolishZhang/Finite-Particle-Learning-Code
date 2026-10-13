import argparse,unittest
import certified_f8_gradient_finalize as final
import certified_teacher_compute as common
from flint import arb,ctx

class Tests(unittest.TestCase):
    def test_nonuniform_weights_are_summed(self):
        self.assertEqual(final.weighted_gradient([arb(1)/4,arb(3)/4],[[arb(1),arb(2)],[arb(3),arb(6)]]),[arb(5)/2,arb(5)])
    def test_vector_error_added_once(self):
        out=final.certify_vector([arb(0)]*4,arb(1)/2)
        self.assertTrue(common.from_record(out['true_gradient_L2_upper']).contains(arb(1)/2))
        self.assertFalse(out['gradient_L2_strictly_below_1e8_inverse'])
    def test_arithmetic_radius_retained(self):
        x=arb(0).union(arb(1)/1000);out=final.certify_vector([x]*4,arb(1)/10000)
        self.assertTrue(common.from_record(out['true_gradient_L2_upper'])>arb(2)/1000)
    def test_bad_shape(self):
        with self.assertRaises(ValueError):final.weighted_gradient([arb(1)],[[arb(1)],[arb(2)]])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    ctx.prec=192;res=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not res.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:common.write_new(final.OUT/'FINALIZE_MANUFACTURED_CHECK.json',dict(passed=True,tests_run=res.testsRun,
        sources={name:common.sha(common.ROOT/name) for name in final.SOURCES},real_gradient_evaluations=0,new_random_draws=0))

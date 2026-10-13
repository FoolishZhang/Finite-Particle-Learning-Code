"""Small non-scientific direct/producer finite-sum identity tests."""
import argparse,unittest
import certified_f8_gradient_audit as audit
import certified_f8_gradient_compute as producer
import certified_teacher_compute as common
from flint import ctx

class Tests(unittest.TestCase):
    def test_independent_cosh_tanh_and_producer(self):
        old=ctx.prec;ctx.prec=192
        try:
            a=('1/3','2/5');cache=producer.precompute_inner(a,'1/2','2')
            for c,K in (((0,2),1),((1,0),2),((1,1),8)):
                row=audit.direct_representative(a,c,'1/2','2','1/2','-2','2',K)
                ref=producer.finite_representative(cache,c,'1/2','-2','2',K=K,backend='loop',saturation_threshold=128)
                self.assertEqual(len(audit.compare(row,ref)),2)
            ref['conditional_gradient_finite'][0]=common.record(common.arb(100))
            with self.assertRaises(ValueError):audit.compare(row,ref)
        finally:ctx.prec=old
    def test_budget_stop_retains_partial(self):
        with self.assertRaises(audit.AuditBudgetStop) as caught:audit.direct_representative(('1/3',),(0,),'1/2','2','1/2','-2','2',deadline=-1)
        self.assertEqual(caught.exception.partial['factors'],0)
    def test_invalid_count(self):
        with self.assertRaises(ValueError):audit.direct_representative(('1/3',),(3,),'1/2','2','1/2','-2','2')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');a=parser.parse_args()
    run=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not run.wasSuccessful():raise SystemExit(1)
    if a.write_receipt:common.write_new(producer.OUT/'AUDIT_MANUFACTURED_CHECK.json',dict(passed=True,tests_run=run.testsRun,
        sources={name:common.sha(common.ROOT/name) for name in audit.SOURCES},
        manufactured_m4_factor_evaluations_per_method=3*81,real_center_factor_evaluations=0,new_random_draws=0))

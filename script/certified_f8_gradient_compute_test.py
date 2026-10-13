"""m2 paired manufacture and scalar large-exponent tests; no real F8 point."""
import argparse
from pathlib import Path
import time
import unittest
import certified_f8_gradient_compute as compute
from flint import arb,ctx

RESULTS=[]


class ManufacturedTests(unittest.TestCase):
    def setUp(self):self.previous=ctx.prec;ctx.prec=192
    def tearDown(self):ctx.prec=self.previous
    def test_exact_zero_score_case(self):
        cache=compute.precompute_inner(('0',),'1/4','2')
        result=compute.finite_representative(cache,(1,),'1/4','-2','2',K=8)
        self.assertTrue(compute.common.from_record(result['conditional_gradient_finite'][0]).contains(arb(0)))
        self.assertTrue(cache['normal_mass']<1)
        RESULTS.append(dict(test='zero_score',result=result))

    def test_positive_middle_and_complement_scores(self):
        cache=compute.precompute_inner(('1/3',),'1/4','2')
        middle=compute.finite_representative(cache,(1,),'1/4','-2','2',K=8)
        self.assertTrue(compute.common.from_record(middle['conditional_gradient_finite'][0])>0)
        left=compute.finite_representative(cache,(0,),'1/4','-2','2',K=8)
        right=compute.finite_representative(cache,(2,),'1/4','-2','2',K=8)
        self.assertTrue(compute.common.from_record(left['conditional_gradient_finite'][0]).overlaps(compute.common.from_record(right['conditional_gradient_finite'][0])))
        RESULTS.append(dict(test='positive_middle_and_complement',middle=middle,left=left,right=right))

    def test_loop_and_matrix_manufactured_performance(self):
        cache=compute.precompute_inner(('1/3',),'1/4','2');records={}
        for backend in ('loop','arb_mat'):
            records[backend]=compute.finite_representative(cache,(1,),'1/4','-2','2',K=8,backend=backend)
        a,b=[compute.common.from_record(records[name]['conditional_gradient_finite'][0]) for name in records]
        self.assertTrue(a.overlaps(b));RESULTS.append(dict(test='loop_matrix_performance',records=records))

    def test_saturation_outer_enclosure_and_huge_exponent(self):
        enclosures=compute.saturation_enclosures(128)
        self.assertTrue(enclosures[0].contains(0));self.assertTrue(enclosures[0].contains(arb(2)**-184))
        self.assertTrue(enclosures[1].contains(0));self.assertTrue(enclosures[1].contains(arb(2)**-177))
        self.assertTrue((-arb(128)).exp()<arb(2)**-184);self.assertTrue(128*(-arb(128)).exp()<arb(2)**-177)
        cases=[]
        for t in (arb(128),arb(129),arb(1000),arb(64).exp()/4):
            tick=time.perf_counter();e=(-t).exp();ordinary_seconds=time.perf_counter()-tick
            tick=time.perf_counter();outer_e,outer_te,used=compute.exponential_factors(t,threshold=128,enclosures=enclosures);enclosure_seconds=time.perf_counter()-tick
            self.assertTrue(used);self.assertTrue(outer_e.contains(e));self.assertTrue(outer_te.contains(t*e))
            self.assertTrue(outer_e.contains(0));self.assertTrue(outer_te.contains(0))
            cases.append(dict(t=compute.common.record(t),ordinary_exp_seconds=ordinary_seconds,
                interval_replace_seconds=enclosure_seconds,ordinary_exp_upper_exponent=int(e.upper().man_exp()[1]),
                encloses_ordinary_factors=True))
        cache=compute.precompute_inner(('0',),'1/4','2')
        exact=compute.finite_representative(cache,(0,),'1/4','6','8',K=8)
        outer=compute.finite_representative(cache,(0,),'1/4','6','8',K=8,saturation_threshold=128)
        self.assertTrue(compute.common.from_record(exact['conditional_gradient_finite'][0]).overlaps(compute.common.from_record(outer['conditional_gradient_finite'][0])))
        self.assertGreater(outer['resources']['saturated_factor_enclosures'],0)
        RESULTS.append(dict(test='saturation_outer_enclosure',cases=cases,finite_exact=exact,finite_outer=outer))

    def test_budget_keeps_only_completed_outer_sum(self):
        cache=compute.precompute_inner(('1/3',),'1/4','2')
        with self.assertRaises(compute.EngineBudgetStop) as captured:
            compute.finite_representative(cache,(1,),'1/4','-2','2',deadline=0.)
        self.assertEqual(captured.exception.partial['outer_grid']['nodes_completed'],0)
        self.assertEqual(captured.exception.partial['status'],'budget_stopped')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:
        compute.OUT.mkdir(parents=True,exist_ok=True)
        sources=[Path(compute.__file__).resolve(),Path(__file__).resolve()]
        compute.common.write_new(compute.OUT/'COMPUTE_MANUFACTURED_CHECK.json',dict(schema='certified_f8_gradient_compute_manufactured_v1',
            created_utc=compute.common.now(),status='passed',tests_run=result.testsRun,results=RESULTS,
            source_sha256={p.relative_to(compute.ROOT).as_posix():compute.common.sha(p) for p in sources},
            real_center_F8_evaluations=0,scientific_NPZ_reads=0,new_kernel_calls=0,new_random_draws=0,new_optimizer_updates=0))


if __name__=='__main__':main()

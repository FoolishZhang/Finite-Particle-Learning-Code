"""Small paired m4 Hessians against explicit per-label old same/cross form."""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import time
import unittest
import certified_f8_hessian_compute as compute
from flint import arb,ctx

RESULTS=[]


def probabilities(m,complement_symmetric=False):
    raw=[Fraction(1+y*y+((2**m-1-y)**2 if complement_symmetric else y),1) for y in range(2**m)]
    mass=sum(raw);return [compute.common.aq(x/mass) for x in raw]


def folded_producer(cache,q,hv,lo,hi,K=8):
    m=2*cache['pair_count'];fold=compute.moments_helper.fold_teacher_moments(q,m);blocks={name:compute.zero_matrix(m) for name in ('Hbb','Haa','Hperp')};rows=[]
    for c,entry in zip(fold['representatives'],fold['moments']):
        row=compute.finite_representative(cache,c,entry['W'],entry['M'],entry['S'],hv,lo,hi,K=K,saturation_threshold=None)
        rows.append(row)
        for name in blocks:
            for i in range(m):
                for j in range(m):blocks[name][i][j]+=compute.common.from_record(row['matrix_blocks'][name][i][j])
    return blocks,rows


def direct_labels(cache,q,hv,lo,hi,K=8):
    """No folded teacher moments or probability-moment contraction helper."""
    tick=time.perf_counter();m=2*cache['pair_count'];index=[min(i,m-1-i) for i in range(m)]
    totals={name:compute.zero_matrix(m) for name in ('Hbb','Haa','Hperp','Hba')};ga=[arb(0) for _ in range(m)]
    hv,lo,hi=map(compute.common.exact,(hv,lo,hi))
    for label,weight_q in enumerate(q):
        if weight_q==0:continue
        y=[(label>>(m-1-i))&1 for i in range(m)]
        A=[];residual=[];sp=[]
        for x,probs in zip(cache['z'],cache['probabilities']):
            yes=[probs[j][1] for j in index];no=[probs[j][0] for j in index]
            product=arb(1)
            for i in range(m):product*=yes[i] if y[i] else no[i]
            A.append(product);residual.append([y[i]-yes[i] for i in range(m)]);sp.append([yes[i]*no[i] for i in range(m)])
        for vi in range((lo/hv).numerator,(hi/hv).numerator+1):
            u=(compute.common.aq(hv)*vi).exp();psi=arb(0);T0=arb(0)
            rb,ra,b0,b1,b2=([arb(0) for _ in range(m)] for _ in range(5))
            GG0,GG1,GG2=(compute.zero_matrix(m) for _ in range(3))
            for x,w,a,r,ss in zip(cache['z'],cache['w'],A,residual,sp):
                t=u*a;e=(-t).exp();f=w*t*e;psi+=w*e;T0+=f
                for i in range(m):
                    rb[i]+=f*r[i];ra[i]+=f*x*r[i]
                    b0[i]+=f*ss[i];b1[i]+=f*x*ss[i];b2[i]+=f*x*x*ss[i]
                    for j in range(m):
                        rr=f*r[i]*r[j];GG0[i][j]+=rr;GG1[i][j]+=x*rr;GG2[i][j]+=x*x*rr
            first=K*psi**(K-1);second=K*(K-1)*psi**(K-2);scale=compute.common.aq(hv)*weight_q
            for i in range(m):
                ga[i]-=scale*first*ra[i]
                for j in range(m):
                    totals['Hbb'][i][j]+=scale*(-second*(T0*GG0[i][j]-rb[i]*rb[j])+(first*b0[i] if i==j else 0))
                    totals['Haa'][i][j]+=scale*(-second*(T0*GG2[i][j]-ra[i]*ra[j])+(first*b2[i] if i==j else 0))
                    totals['Hperp'][i][j]+=scale*(-second*T0*GG0[i][j]+(first*b0[i] if i==j else 0))
                    totals['Hba'][i][j]+=scale*(-second*(T0*GG1[i][j]-rb[i]*ra[j])+(first*b1[i] if i==j else 0))
    return totals,ga,time.perf_counter()-tick


class ManufacturedTests(unittest.TestCase):
    def setUp(self):self.previous=ctx.prec;ctx.prec=192
    def tearDown(self):ctx.prec=self.previous
    def test_folded_full_blocks_match_direct_nonexchangeable_weights(self):
        cache=compute.precompute_inner(('1/3','2/5'),'1/4','2');q=probabilities(4)
        started=time.perf_counter();prod,rows=folded_producer(cache,q,'1/4','-2','2');producer_seconds=time.perf_counter()-started
        direct,_,seconds=direct_labels(cache,q,'1/4','-2','2')
        overlaps=0
        for name in prod:
            for i in range(4):
                for j in range(4):
                    self.assertTrue(prod[name][i][j].overlaps(direct[name][i][j]),(name,i,j))
                    self.assertTrue(prod[name][i][j].overlaps(prod[name][j][i]));overlaps+=1
        RESULTS.append(dict(test='full_blocks_vs_direct_label',m=4,teacher_within_pair_exchangeability=False,teacher_complement_symmetry=False,
            class_count=5,overlapping_entries=overlaps,producer_wall_seconds=producer_seconds,direct_wall_seconds=seconds,
            representative_rows=rows))

    def test_zero_weight_no_division(self):
        cache=compute.precompute_inner(('1/3','2/5'),'1/4','2');zero=arb(0)
        row=compute.finite_representative(cache,(0,0),zero,[zero]*4,[[zero]*4 for _ in range(4)],'1/4','-2','2')
        for matrix in row['matrix_blocks'].values():
            for values in matrix:
                for value in values:self.assertTrue(compute.common.from_record(value).contains(0))
        self.assertFalse(row['class_weight_division_used'])

    def test_complement_teacher_mixed_and_Ward_small_grid(self):
        cache=compute.precompute_inner(('1/3','2/5'),'1/4','6');q=probabilities(4,True)
        matrices,ga,seconds=direct_labels(cache,q,'1/4','-24','32')
        for row in matrices['Hba']:
            for value in row:self.assertTrue(value.contains(0))
        aa=list(map(compute.common.aq,('1/3','2/5','2/5','1/3')))
        ward=[sum((v*a for v,a in zip(row,aa)),arb(0))-g for row,g in zip(matrices['Hperp'],ga)]
        self.assertTrue(all(abs(x)<compute.common.aq('1/1000000') for x in ward))
        RESULTS.append(dict(test='complement_mixed_Ward',m=4,inner_nodes=49,outer_nodes=225,
            mixed_ba_zero_in_manufactured_arithmetic_enclosure=True,
            Ward_residuals=[compute.common.record(x) for x in ward],Ward_diagnostic_threshold_exact='1/1000000',
            Ward_not_a_certified_real_center_bound=True,direct_wall_seconds=seconds))

    def test_budget_preserves_partial_without_retry(self):
        cache=compute.precompute_inner(('1/3','2/5'),'1/4','2');q=probabilities(4);entry=compute.moments_helper.fold_teacher_moments(q,4)['moments'][0]
        with self.assertRaises(compute.HessianBudgetStop) as caught:
            compute.finite_representative(cache,(0,0),entry['W'],entry['M'],entry['S'],'1/4','-2','2',deadline=0.)
        self.assertEqual(caught.exception.partial['outer_grid']['nodes_completed'],0)
        self.assertFalse(caught.exception.partial['folded_mixed_bias_radial_block_provided'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ManufacturedTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.write_receipt:
        compute.OUT.mkdir(parents=True,exist_ok=True);sources=[Path(compute.__file__).resolve(),Path(__file__).resolve()]
        compute.common.write_new(compute.OUT/'COMPUTE_MANUFACTURED_CHECK.json',dict(schema='certified_f8_hessian_compute_manufactured_v1',
            created_utc=compute.common.now(),status='passed',tests_run=result.testsRun,results=RESULTS,
            source_sha256={p.relative_to(compute.ROOT).as_posix():compute.common.sha(p) for p in sources},
            real_center_Hessian_evaluations=0,scientific_NPZ_reads=0,new_kernel_calls=0,new_random_draws=0,new_optimizer_updates=0))


if __name__=='__main__':main()

"""Manufactured m2/m3 fixtures only; no old study array is opened."""
from __future__ import annotations
import copy
import io
import json
import math
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit, roots_hermitenorm
import direction_validation_compute as compute


RULES = ((12,0.),(16,0.),(20,0.),(18,math.pi/7))


def fixture(m=2, t_values=compute.T_VALUES, real_hessian=False):
    signs=np.array([1,-1,1][:m],dtype=np.int8)
    b=np.array([.12,-.18,.07][:m]); a=np.array([.32,-.42,.27][:m])
    d=np.array([.6,-.8,.2][:m]); d-=d.mean(); d/=math.hypot(*d)
    q=np.arange(1.,(1<<m)+1); q/=q.sum()
    teachers=np.array([np.arange(1.,(1<<m)+1)[::-1] for _ in range(4)])
    teachers/=teachers.sum(axis=1)[:,None]
    # Deliberately non-unit raw teacher mass for normalization-accounting tests.
    teachers*=np.array([1-1e-13,1+2e-13,1.,1+1e-13])[:,None]
    permutation=compute.recoded_indices(signs)
    if real_hessian:
        nodes,weights=roots_hermitenorm(48); weights/=math.sqrt(2*math.pi)
        br=b*signs; ar=a*signs
        Y=compute.joint.patterns(m)
        p=expit(br[None,:]+nodes[:,None]*ar[None,:])
        H=[]
        for y in Y:
            A=np.prod(np.where(y[None,:],p,1-p),axis=1)
            post=weights*A/np.dot(weights,A)
            r=y[None,:]-p
            edges=ar[:,None]*ar[None,:]*np.einsum('l,li,lj->ij',post,r,r)
            np.fill_diagonal(edges,0.)
            H.append(np.diag(edges.sum(axis=1))-edges)
        H=np.array(H)
    else:
        projection=np.eye(m)-np.ones((m,m))/m
        H=np.arange(-2.,(1<<m)-2)[:,None,None]*.01*projection[None,:,:]
    H=np.stack([H,H*(1+1e-11)])
    s=np.einsum('i,ryij,j->ry',d,H,d)[:,permutation]
    source=dict(unit=dict(id=f'manufactured_m{m}',teacher=0,N=20,rep=0),theta_signed=np.r_[b,a],
        theta_recoded=np.r_[b*signs,a*signs],signs=signs,q=q,teacher_pmfs=teachers,
        teacher_rules=[list(r) for r in RULES],fixed_d=d,label_Hangle=H,
        old_population_curvature=s@teachers.T,old_empirical_curvature=s@q,
        A=True,B=True,full_level='D3',saved_delta_R=.002,original_numeric_risk_qualification=True)
    candidate=compute.generate_candidate_pmfs(b,a,d,t_values=t_values,rules=RULES)
    return source,candidate


class DirectionValidationTests(unittest.TestCase):
    diagnostics={}

    def test_recoding_is_bijective_original_label_pullback(self):
        signs=np.array([-1,1,-1]); perm=compute.recoded_indices(signs)
        self.assertEqual(sorted(perm),list(range(8)))
        np.testing.assert_array_equal(perm[perm],np.arange(8))
        values=np.arange(8.)**2; P=np.arange(1.,9.)*.02
        recoded=np.empty_like(P);recoded[perm]=P
        self.assertAlmostEqual(float(P@values[perm]),float(recoded@values),delta=1e-14)

    def test_path_original_signed_loading_and_copy(self):
        b=np.array([.2,-.3]); a=np.array([.7,-.4]); d=np.array([.6,-.8])
        before=[v.copy() for v in (b,a,d)]
        br,ar,dr,W=compute.path_models(b,a,d)
        np.testing.assert_array_equal(W[0,:,0],a);np.testing.assert_array_equal(W[0,:,1],np.zeros(2))
        np.testing.assert_allclose(np.sum(W*W,axis=2),np.tile(a*a,(4,1)),rtol=0,atol=4*np.finfo(float).eps)
        for lhs,rhs in zip((b,a,d),before):np.testing.assert_array_equal(lhs,rhs)
        br[0]=9;ar[0]=9;dr[0]=9
        for lhs,rhs in zip((b,a,d),before):np.testing.assert_array_equal(lhs,rhs)

    def test_baseline_once_per_rule_and_fixed_call_counts(self):
        calls=[]
        def pmf(b,W,order,angle):
            calls.append((W.copy(),order,angle));return np.full(4,.25)
        candidate=compute.generate_candidate_pmfs([.1,.2],[.3,-.2],[1.,-1.],rules=RULES,pmf_fn=pmf)
        self.assertEqual(len(calls),16)
        self.assertEqual(sum(np.all(W[:,1]==0) for W,_,_ in calls),4)
        self.assertEqual(candidate['resources']['PMF_calls'],16)
        self.assertEqual(candidate['resources']['node_pattern_products'],4*sum(o*o for o,_ in RULES)*4)

    def test_manufactured_PMF_against_direct_probability_product(self):
        source,cand=fixture()
        Y=compute.joint.patterns(2)
        errors=[]
        for mi,W in enumerate(cand['model_W']):
            for ri,(order,angle) in enumerate(RULES):
                nodes,w=roots_hermitenorm(order);w/=math.sqrt(2*math.pi)
                grid=np.array([(x,y) for x in nodes for y in nodes])
                R=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
                grid=grid@R.T;weights=np.outer(w,w).reshape(-1)
                p=expit(cand['b'][None,:]+grid@W.T)
                direct=np.array([np.dot(weights,np.prod(np.where(y,p,1-p),axis=1)) for y in Y])
                errors.append(float(np.max(abs(direct-cand['pmfs'][mi,ri]))))
        self.assertLess(max(errors),8*np.finfo(float).eps)
        self.diagnostics['manufactured_PMF_product_max_error']=max(errors)

    def test_raw_curvature_identity_and_explicit_normalization(self):
        source,cand=fixture(3);original=copy.deepcopy(source)
        result=compute.derive_unit(source,cand)
        np.testing.assert_allclose(result['raw_population_curvature'],source['old_population_curvature'],atol=1e-17,rtol=0)
        compute.same_tree(source,original)
        np.testing.assert_array_equal(result['teacher_raw_pmfs'],source['teacher_pmfs'])
        self.assertNotEqual(result['curvature']['main']['moments']['raw_mass'],1.)
        moments=result['curvature']['main']['moments']
        self.assertAlmostEqual(moments['normalization_correction'],moments['mean']-moments['raw_mean'],delta=1e-20)
        self.assertEqual(result['source']['fixed_d'].tobytes(),source['fixed_d'].tobytes())

    def test_derivation_calls_no_quadrature_and_returns_snapshots(self):
        source,cand=fixture();original=copy.deepcopy(cand)
        with patch.object(compute.joint,'model_pmf',side_effect=AssertionError('new quadrature forbidden')):
            result=compute.derive_unit(source,cand)
        compute.same_tree(cand,original)
        result['source']['fixed_d'][0]=100
        self.assertNotEqual(source['fixed_d'][0],100)
        result['teacher_raw_pmfs'][0,0]=0
        self.assertGreater(source['teacher_pmfs'][0,0],0)

    def test_all_raw_finite_gain_identities(self):
        source,cand=fixture();result=compute.derive_unit(source,cand)
        errors=[]
        for ti in range(3):
            for ri in range(4):
                Q0,Qt=cand['pmfs'][0,ri],cand['pmfs'][ti+1,ri]
                for pi,P in enumerate(source['teacher_pmfs']):
                    risk0=np.sum(P*(np.log(P)-np.log(Q0)))
                    riskt=np.sum(P*(np.log(P)-np.log(Qt)))
                    raw=result['finite_gain'][ti]['variants'][ri][pi]['moments']['raw_mean']
                    errors.append(abs(raw-(risk0-riskt)))
        self.assertLess(max(errors),1e-15)
        self.diagnostics['raw_risk_gain_identity_max_error']=max(errors)

    def test_smallstep_bridge_with_independent_label_Hessian(self):
        source,cand=fixture(t_values=(.01,.04,.08),real_hessian=True)
        result=compute.derive_unit(source,cand)
        errors=[r['maxabs_label_residual'] for r in result['bridge']]
        self.assertLess(errors[0],2e-5)
        self.assertLess(errors[0],errors[2]/15)
        self.assertGreater(result['bridge'][0]['correlation']['value'],.999999)
        self.diagnostics['smallstep_label_residuals']=errors

    def test_numeric_failure_retained_without_call_or_escalation(self):
        source,cand=fixture()
        cand['pmfs'][1,0,0]+=1e-4;cand['pmfs'][1,0,1]-=1e-4
        with patch.object(compute.joint,'model_pmf',side_effect=AssertionError('no retry')):
            result=compute.derive_unit(source,cand)
        self.assertFalse(result['finite_qualification'][0]['qualified'])
        self.assertFalse(result['all_numerically_qualified'])
        self.assertEqual(len(result['finite_gain']),3)
        self.assertEqual(result['source']['A'],source['A'])

    def test_zero_candidate_probability_keeps_undefined_record(self):
        source,cand=fixture();cand['pmfs'][2,1,0]=0.
        with np.errstate(invalid='ignore'):
            result=compute.derive_unit(source,cand)
        self.assertFalse(result['finite_qualification'][1]['qualified'])
        self.assertEqual(result['finite_gain'][1]['main']['status'],'undefined_nonfinite_score')
        self.assertEqual(result['transfer'][1]['status'],'undefined_nonfinite_score')
        self.assertEqual(len(result['finite_gain']),3)

    def test_call_failure_record_and_no_retry(self):
        counter=[0]
        def pmf(*args,**kwargs):
            counter[0]+=1
            if counter[0]==5:raise FloatingPointError('manufactured nonfinite')
            return np.full(4,.25)
        c=compute.generate_candidate_pmfs([.1,.2],[.3,-.2],[1.,-1.],rules=RULES,pmf_fn=pmf)
        self.assertEqual(counter[0],16);self.assertEqual(c['calls'][4]['status'],'PMF_call_failed')
        self.assertTrue(np.isnan(c['pmfs'][1,0]).all())

    def test_transfer_bound_preserves_negative_correction(self):
        X=np.array([.1,.3,-.2,.4]);P=np.array([.1,.2,.3,.4])*1.03;t=.1
        curvature=np.tile(X,(2,1));z=np.tile(.5*t*t*(X+.02),(4,1))
        cr=compute._record(X,P,{'zero':0.,'useful':1e-4})
        zr=compute._record(z[1],P,{'zero':0.,'matched_curvature':.5*t*t*1e-4})
        result=compute.transfer_readout(curvature,z,P,cr,zr,t)
        self.assertLess(result['main_e_minus'],0)
        self.assertAlmostEqual(result['main_e_minus'],-.02,delta=1e-16)
        self.assertGreaterEqual(result['normalized_mean_slack'],-1e-17)
        self.assertLess(result['scaling_identity_maxabs_residual'],1e-18)
        self.assertEqual(result['new_score_families'],0)
        self.assertFalse(result['strict_Gaussian_likelihood_certificate'])
        self.diagnostics['transfer_scaling_residual']=result['scaling_identity_maxabs_residual']

    def test_constant_score_and_zero_denominator_explicit(self):
        P=np.full(4,.25);X=np.full(4,.001);z=.005*X
        cr=compute._record(X,P,{'zero':0.,'useful':1e-4});zr=compute._record(z,P,{'zero':0.})
        b=compute.bridge_readout(X,X,P,cr,zr,.1)
        self.assertIsNone(b['correlation']['value'])
        self.assertEqual(b['correlation']['status'],'undefined_zero_denominator')
        self.assertIsNone(cr['standardized']['signal_to_SD']['value'])

    def test_oracle_filtering_is_exact_retrospective_algebra(self):
        rows=[dict(unit=str(i),N=20,A=A,B=B,old_population_curvature=k,
                   old_empirical_curvature=k-.003,saved_delta_R=g)
              for i,(A,B,k,g) in enumerate([(True,True,-.1,.004),(True,True,.1,-.002),(False,True,-.1,.006)])]
        result=compute.oracle_retrospective(rows);p=result['all_units']['policies']
        self.assertEqual(p['A_star']['selected'],1);self.assertEqual(p['B_star']['selected'],2)
        self.assertAlmostEqual(p['A_star']['mean_risk_difference_from_always_HIGH'],.004/3,delta=1e-18)
        self.assertAlmostEqual(result['all_units']['hP_minus_hemp_mean'],.003,delta=1e-16)
        self.assertFalse(result['new_policy_executed'])

    def test_missing_freeze_fails_before_any_kernel_or_source_read(self):
        with patch.object(compute,'verify_freeze',side_effect=FileNotFoundError('freeze required')):
            with patch.object(compute.joint,'model_pmf',side_effect=AssertionError('no kernel')):
                with patch.object(compute,'load_sources',side_effect=AssertionError('no source read')):
                    with self.assertRaises(FileNotFoundError):compute.main('run')

    def test_source_mean_mismatch_rejected_not_reinterpreted(self):
        source,cand=fixture();source['old_population_curvature'][1,1]+=.001
        with self.assertRaisesRegex(AssertionError,'old curvature mean mismatch'):compute.derive_unit(source,cand)

    def test_summary_and_readable_include_all_rows_and_null_budgets(self):
        source,cand=fixture();result=compute.derive_unit(source,cand)
        summary=compute.summarize([result],[cand]);text=compute.readable(summary)
        self.assertEqual(len(summary['rows']),1)
        self.assertIn('manufactured_m2',text)
        self.assertIn('not a confidence interval',text)
        self.assertTrue(summary['oracle_planning_is_not_confidence_or_power'])
        for item in summary['rows'][0]['curvature']['planning']['minimum_n'].values():
            for record in item.values():
                if record['n'] is not None:
                    self.assertLess(record['radius_at_n'],record['margin'])
                    if record['previous_n'] is not None:self.assertGreaterEqual(record['radius_at_previous_n'],record['margin'])


def run_tests():
    tick=time.perf_counter()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(DirectionValidationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        source_sha256=compute.sha(Path(compute.__file__)),test_sha256=compute.sha(Path(__file__)),
        diagnostics=DirectionValidationTests.diagnostics,seconds=time.perf_counter()-tick,
        real_study_objects_loaded=False,real_teacher_PMFS_computed=False,new_random_draws=0,new_training=0,
        manufactured_only=True)
    print(json.dumps(report,indent=2));return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(run_tests())

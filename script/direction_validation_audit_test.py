"""Manufactured-only direction-validation audit tests; no study files or RNG."""
from __future__ import annotations

import math
import copy
from unittest.mock import patch
import numpy as np
import direction_validation_audit as audit
import direction_validation_statistics as statistics


def run_tests():
    checks=audit.Checks();tests=[]
    x=np.array([-1.,0.,2.,3.]);P=np.array([.1,.2,.3,.4])*.97
    m=audit.moments(x,P);prod=statistics.weighted_moments(x,P)
    for key,value in m.items():checks.close(prod[key],value,'manufactured.moments',key)
    score=statistics.planning_table(m['mean'],m['variance'],m['range'],{'zero':0.,'positive':.1})
    audit.check_planning(score,m,{'zero':0.,'positive':.1},checks,'manufactured/planning')
    tests.append('raw/nonunit Pbar moments and full six-look planning')
    for mean,target,V,R in ((.1,.1,.2,1.),(.001,0.,1.,10.),(1.,0.,0.,0.),(.1,0.,0.,1.)):
        for method in ('hoeffding','eb_proxy'):
            result=statistics.minimum_n(mean,target,V,R,method)
            audit.check_minimum(result,mean,target,V,R,method,checks,'manufactured/minimum')
    equality=statistics.hoeffding_radius(8,1.)
    n,status=audit.required_n(equality,0.,.1,1.,'hoeffding')
    checks.same(n,9,'strict_equality_requires_next_integer')
    tests.append('nonpositive margins, cap, constant score, zero variance and strict integer boundary')
    Q0=np.array([.1,.2,.3,.4])*1.001;Qt=np.array([.15,.2,.3,.35])*.999
    identity=audit.score_kl_identity(P,Q0,Qt)
    checks.close(identity['raw_residual'],0.,'raw_identity','manufactured')
    checks.close(identity['pbar_residual'],0.,'pbar_identity','manufactured')
    checks.close(identity['normalized_model_mean'],identity['normalized_model_KL_difference'],'model_mass_correction','manufactured')
    tests.append('raw and explicitly normalized score/KL identities')
    import direction_validation_compute as producer
    d=np.array([1.,-1.])/math.sqrt(2);signs=np.array([-1,1]);theta=np.array([.1,-.2,-.4,.6])
    s=np.array([-.2,.1,.4,-.3]);H=np.array([[v*np.outer(d,d) for v in s],[v*np.outer(d,d) for v in s+1e-8]])
    q=np.array([.2,.3,.25,.25]);teachers=np.tile(np.array([.12,.24,.31,.33]),(4,1))
    mask=2;idx=np.arange(4)^mask
    recoded=np.array([[float(d@h@d) for h in rule] for rule in H]);original=recoded[:,idx]
    source=dict(unit=dict(id='manufactured',teacher=0,N=4,rep=0),theta_signed=theta,signs=signs,q=q,
        teacher_pmfs=teachers,fixed_d=d,label_Hangle=H,old_population_curvature=original@teachers.T,
        old_empirical_curvature=original@q,A=True,B=True,full_level='D3',saved_delta_R=.002)
    b,a,dd,W=producer.path_models(theta[:2],theta[2:],d)
    base=np.array([.1,.2,.3,.4]);slope=np.array([.01,-.02,.03,-.02])
    models=np.array([base]+[base+t*t*slope for t in audit.TS])
    calls=[]
    def fixed_pmf(b,W,order,angle):
        index=len(calls)//4;calls.append(index);return models[index].copy()
    cand=producer.generate_candidate_pmfs(b,a,dd,pmf_fn=fixed_pmf)
    derived=producer.derive_unit(source,cand)
    sr,so,perm=audit.fixed_curvature(source)
    checks.close(derived['label_curvature_recoded'],sr,'label_loop_recoded','manufactured')
    checks.close(derived['label_curvature_original'],so,'label_loop_original','manufactured')
    checks.same(derived['recoded_index_of_original_label'],perm,'exact_original_label_reindex')
    audit.check_score_records(derived['curvature'],-derived['label_curvature_original'],teachers,{'zero':0.,'useful':1e-4},checks,'manufactured/curvature')
    for i,t in enumerate(audit.TS):
        audit.check_score_records(derived['finite_gain'][i],derived['z'][i],teachers,audit.finite_targets(t),checks,'manufactured/finite')
        audit.check_bridge(derived['bridge'][i],derived['Xc'],derived['Xt_norm'][i,1],teachers[1],derived['finite_gain'][i]['main'],t,checks,'manufactured/bridge')
        audit.check_transfer(derived['transfer'][i],-derived['label_curvature_original'],derived['z'][i],teachers[1],t,checks,'manufactured/transfer')
    tests.append('complete curvature/finite/bridge/transfer schema and critical-n reconstruction')
    report=audit.check_unit(derived,cand,checks)
    checks.same(report['unit'],'manufactured','whole_unit_interface')
    checks.same(len(report['raw_and_explicitly_normalized_KL_identities']),48,'all48_variant_KL_identities')
    gl_summary=audit.gl_compare(models,cand,teachers)
    checks.require(gl_summary['qualified'],'stored_independent_array_comparison_pass')
    perturbed=models.copy();perturbed[1,0]+=1e-5
    checks.require(not audit.gl_compare(perturbed,cand,teachers)['qualified'],'numerical_limit_retained_without_execution_failure')
    bad=models.copy();bad[1,0]=np.nan
    checks.require(not audit.gl_compare(bad,cand,teachers)['qualified'],'nonfinite_independent_point_retained')
    tests.append('whole-unit/call-schema reconstruction and independent PMF qualification including failure')
    # Exercise the complete orchestration on twelve copied manufactured m2
    # cells. All source loaders/hashes are replaced; there is no study I/O.
    manufactured_units=[];analyses=[];candidates=[];lookup={}
    for i in range(12):
        unit=dict(id=f'fixture_{i}',teacher=i//4,N=(1024,4096)[(i//2)%2],rep=i%2)
        aa=copy.deepcopy(derived);aa['source']['unit']=unit;cc=copy.deepcopy(cand)
        manufactured_units.append(unit);analyses.append(aa);candidates.append(cc)
        lookup[unit['id']]={'analysis':aa,'candidate_pmfs':cc}
    produced=dict(unit_artifacts=[dict(unit=u['id']) for u in manufactured_units])
    summary=producer.summarize(analyses,candidates)
    class FixtureTree:
        def __init__(self,stem,*args):self.value=lookup[stem.parent.name][stem.name]
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def get(self,*args):return self.value
    mock_calls=[]
    def gl_fixture(*args,**kwargs):
        checks.same(kwargs,dict(order=128,limit=10.,rotation=0.),'frozen_GL_arguments')
        idx=len(mock_calls)%4;mock_calls.append(idx);return dict(pmf=models[idx].copy())
    storage={}
    with patch.object(audit,'verify_inputs',return_value=({},produced,manufactured_units,{})), \
         patch.object(audit,'old_source_check'),patch.object(audit.independent,'SavedTree',FixtureTree), \
         patch.object(audit,'read',return_value=dict(derived=summary)),patch.object(audit,'sha',return_value='manufactured_hash'), \
         patch.object(audit.independent,'gaussian_legendre_pmf',side_effect=gl_fixture):
        initial=audit.audit(storage,True)
        checks.require(initial['passed'],'manufactured_run_orchestration')
        checks.same(len(mock_calls),48,'exact48_mocked_PMF_dispatches')
        with patch.object(audit.independent,'gaussian_legendre_pmf',side_effect=AssertionError('check must not integrate')):
            repeat=audit.audit(storage,False)
        checks.same(audit.native(initial),audit.native(repeat),'stable_run_check_receipt_without_integrations')
    checks.same(len(storage),24,'twelve_GL_PMF_and_status_arrays')
    tests.append('complete mocked 12-unit dispatch and stable no-integration --check algebra')
    # An explicitly negative e_minus must remain negative, never clipped to zero.
    cs=np.array([[.1,.2,.3,.4],[.1,.2,.3,.4]])
    zs=np.repeat((.5*.1**2*(cs[1]+.02))[None,:],4,axis=0)
    curv=producer.score_records(cs,teachers,{'zero':0.,'useful':1e-4})
    finite=producer.score_records(zs,teachers,audit.finite_targets(.1))
    transfer=producer.transfer_readout(cs,zs,teachers[1],curv['main'],finite['main'],.1)
    audit.check_transfer(transfer,cs,zs,teachers[1],.1,checks,'manufactured/negative_transfer')
    checks.require(transfer['main_e_minus']<0,'negative_e_minus_retained')
    tests.append('negative transfer correction preserved')
    # One manufactured PMF verifies reuse of the independent GL evaluator only.
    gl=audit.independent.gaussian_legendre_pmf([.1,-.2],np.zeros((2,2)),order=12,limit=3.,rotation=0.)
    p=1/(1+np.exp(-np.array([.1,-.2])));Y=audit.independent.patterns(2)
    expected=np.prod(np.where(Y==1,p,1-p),axis=1)*gl['normal_mass']
    checks.close(gl['pmf'],expected,'GL_raw_mass_constant_logits','manufactured',tol=1e-14)
    tests.append('independent GL PMF with unnormalized finite mass')
    return dict(passed=not checks.failures,manufactured_only=True,scientific_inputs_read=False,
        new_random_draws=0,checks=checks.count,planning_rows=checks.planning_rows,critical_n_checks=checks.minimum_n_checks,
        mocked_panel_run_and_check_checks=initial['checks']+repeat['checks'],mocked_PMF_dispatches=48,
        actual_manufactured_GL_PMF_calls=1,scientific_GL_PMF_calls=0,
        failures=checks.failures,test_groups=tests)


if __name__=='__main__':
    import json
    result=run_tests();print(json.dumps(audit.native(result),indent=2,allow_nan=False))
    raise SystemExit(0 if result['passed'] else 1)

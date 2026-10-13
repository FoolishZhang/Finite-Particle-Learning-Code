"""Bounded independent execution/numerical audit of DirectionValidation-v1.

The first --run makes exactly48 GL128/T10 PMF calls, never validation samples.
--check reuses saved GL arrays and repeats only source/score/statistics algebra.
The numerical comparison gates are not rigorous integration-error bounds.
"""
from __future__ import annotations

import os
for _thread in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_thread]="1"

import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import prospective_explanation_independent as independent

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/direction_validation_v1_20261007'
OLD=ROOT/'artifacts/prospective_diagnostic_dev_v1_20261007'
EXPLANATION=ROOT/'artifacts/prospective_explanation_v1_20261007'
TS=(.02,.1,.25)
GRID=(1024,4096,16384,65536,262144,1048576)
DELTA=.05/288
CAP=2**30
read,sha,rel,native=independent.read,independent.sha,independent.rel,independent.native


class Checks(independent.Checks):
    def __init__(self):
        super().__init__();self.minimum_n_checks=0;self.planning_rows=0
    def scalar(self,a,b,field,context,atol=2e-12,rtol=2e-12):
        if a is None or b is None:self.same(a,b,context+':'+field);return
        self.close(a,b,field,context,tol=atol+rtol*max(abs(float(a)),abs(float(b))))


def ratio(n,d):
    if n is None or d is None or not math.isfinite(float(n)) or not math.isfinite(float(d)):return None
    if d==0:return None
    v=float(n/d)
    return v if math.isfinite(v) else None


def moments(x,P):
    x=np.asarray(x,dtype=float);P=np.asarray(P,dtype=float)
    if x.shape!=P.shape or x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(P).all() or np.any(P<0):
        raise ValueError('finite matching score/probability arrays required')
    M=math.fsum(float(w) for w in P)
    if M<=0:raise ValueError('positive raw probability mass required')
    pb=np.array([float(w)/M for w in P])
    raw=math.fsum(float(w)*float(v) for w,v in zip(P,x))
    mean=math.fsum(float(w)*float(v) for w,v in zip(pb,x))
    variance=math.fsum(float(w)*(float(v)-mean)**2 for w,v in zip(pb,x))
    rawvar=math.fsum(float(w)*(float(v)-mean)**2 for w,v in zip(P,x))
    lo,hi=min(map(float,x)),max(map(float,x))
    if lo==hi:mean,variance,rawvar=lo,0.,0.
    return dict(raw_mass=M,raw_mass_error=M-1,raw_mean=raw,mean=mean,variance=variance,std=math.sqrt(variance),
        raw_centered_second_moment=rawvar,score_min=lo,score_max=hi,range=hi-lo,
        normalization_correction=mean-raw,normalization_correction_absolute=abs(mean-raw),
        normalization_correction_algebraic_bound=abs(1-M)*max(abs(lo),abs(hi)),
        pbar=pb,pbar_mass=math.fsum(float(w) for w in pb),score_count=len(x))


def radius(n,V,R,method,factor=1.):
    if n<2 or V<0 or R<0:raise ValueError('invalid independent planning arguments')
    if method=='hoeffding':return factor*R*math.sqrt(-math.log(DELTA)/(2*n))
    if method=='eb_proxy':
        L=math.log(2.)-math.log(DELTA)
        return factor*(math.sqrt(2*V*L/n)+7*R*L/(3*(n-1)))
    raise ValueError('unknown planning method')


def required_n(mu,target,V,R,method,factor=1.,cap=CAP):
    """Full-range binary search, independently of producer doubling search."""
    gap=mu-target
    if gap<=0:return None,'nonpositive_oracle_margin'
    if not radius(cap,V,R,method,factor)<gap:return None,'not_reached_by_cap'
    lo,hi=2,cap
    while lo<hi:
        mid=(lo+hi)//2
        if radius(mid,V,R,method,factor)<gap:hi=mid
        else:lo=mid+1
    return lo,'strict_planning_inequality_met'


def check_ratio(saved,num,den,checks,context):
    value=ratio(num,den)
    checks.scalar(saved.get('value'),value,'ratio',context)
    if 'defined' in saved:
        checks.same(saved['defined'],value is not None,context+':ratio_defined')
        status='finite' if value is not None else 'zero_denominator' if den==0 else 'floating_overflow'
    else:
        invalid=num is None or den is None or not math.isfinite(float(num)) or not math.isfinite(float(den))
        status='undefined_nonfinite_or_missing' if invalid else 'undefined_zero_denominator' if den==0 else 'defined' if value is not None else 'undefined_ratio_overflow'
    checks.same(saved['status'],status,context+':ratio_status')


def check_minimum(saved,mu,target,V,R,method,checks,context,factor=1.):
    checks.minimum_n_checks+=1
    n,status=required_n(mu,target,V,R,method,factor)
    checks.same(saved['n'],n,context+':critical_n')
    checks.same(saved['status'],status,context+':critical_n_status')
    checks.scalar(saved['margin'],mu-target,'planning_gap',context)
    checks.same(saved['delta'],DELTA,context+':delta')
    checks.same(saved['max_n'],CAP,context+':planning_cap')
    checks.same(saved['radius_multiplier'],factor,context+':radius_factor')
    checks.scalar(saved['radius_at_cap'],radius(CAP,V,R,method,factor),'radius_cap',context)
    if n is not None:
        rn=radius(n,V,R,method,factor);checks.require(rn<mu-target,context+':strict_at_n')
        checks.scalar(saved['radius_at_n'],rn,'radius_at_n',context)
        previous=n-1 if n>2 else None;checks.same(saved['previous_n'],previous,context+':previous_n')
        if previous is not None:
            rp=radius(previous,V,R,method,factor)
            checks.require(not rp<mu-target,context+':previous_does_not_pass')
            checks.scalar(saved['radius_at_previous_n'],rp,'radius_at_previous',context)
    for key,value in dict(strict_inequality=True,oracle_planning=True,confidence_bound=False,power_guarantee=False,necessary_sample_size=False).items():
        checks.same(saved[key],value,context+':scope_'+key)


def check_planning(saved,m,targets,checks,context):
    mu,V,R=m['mean'],m['variance'],m['range']
    checks.same(saved['targets'],targets,context+':fixed_margins')
    checks.same(saved['delta'],DELTA,context+':family_delta')
    checks.same([r['n'] for r in saved['rows']],GRID,context+':six_looks')
    for row in saved['rows']:
        n=row['n'];h=radius(n,V,R,'hoeffding');e=radius(n,V,R,'eb_proxy');checks.planning_rows+=1
        checks.scalar(row['hoeffding_radius'],h,'hoeffding_radius',context)
        checks.scalar(row['eb_proxy_radius'],e,'EB_population_proxy_radius',context)
        check_ratio(row['oracle_mean_over_standard_error'],mu,math.sqrt(V/n),checks,context+':mean_over_SE')
        for key,t in targets.items():
            expected=dict(target=t,oracle_gap=mu-t,mean_minus_hoeffding_exceeds_target=h<mu-t,
                mean_minus_eb_proxy_exceeds_target=e<mu-t,twice_hoeffding_below_gap=2*h<mu-t)
            checks.same(row['target_checks'][key],expected,context+':target_checks')
    for key,t in targets.items():
        for method in ('hoeffding','eb_proxy'):
            check_minimum(saved['minimum_n'][key][method],mu,t,V,R,method,checks,context+'/'+key+'/'+method)
        check_minimum(saved['twice_hoeffding_sufficient_n'][key],mu,t,V,R,'hoeffding',checks,context+'/'+key+'/two_H',2.)
    for key,value in dict(oracle_planning=True,confidence_bound=False,power_guarantee=False,targets_share_same_score_event=True,observed_validation_samples=0).items():
        checks.same(saved[key],value,context+':planning_scope_'+key)


def check_score_records(saved,scores,teachers,targets,checks,context):
    checks.same(saved['targets'],targets,context+':targets')
    checks.same(saved['main_index'],[1,1],context+':main')
    checks.same(saved['main'],saved['variants'][1][1],context+':main_alias')
    for ri,x in enumerate(scores):
        for pi,P in enumerate(teachers):
            record=saved['variants'][ri][pi];where=context+f'/{ri}/{pi}'
            if not np.isfinite(x).all():
                checks.same(record['status'],'undefined_nonfinite_score',where+':nonfinite_retained');continue
            checks.same(record['status'],'defined',where+':defined')
            own=moments(x,P)
            for key,value in own.items():
                if key=='pbar':checks.close(record['moments'][key],value,'explicit_pbar',where)
                else:checks.scalar(record['moments'][key],value,'moment.'+key,where)
            checks.same(record['moments']['normalization_explicit'],True,where+':explicit_normalization')
            checks.same(record['moments']['range_includes_zero_probability_labels'],True,where+':whole_score_range')
            check_ratio(record['moments']['mean_over_sd'],own['mean'],own['std'],checks,where+':moment_SNR')
            check_ratio(record['standardized']['signal_to_SD'],own['mean'],own['std'],checks,where+':SNR')
            for key,t in targets.items():check_ratio(record['standardized']['excess_signal_to_SD'][key],own['mean']-t,own['std'],checks,where+'/'+key+':excess_SNR')
            check_planning(record['planning'],own,targets,checks,where)


def fixed_curvature(source):
    """Direct per-label double loop, then exact original-label reindex."""
    H=np.asarray(source['label_Hangle']);d=np.asarray(source['fixed_d']);signs=source['signs'];m=len(d)
    s=np.empty(H.shape[:2])
    for ri in range(2):
        for y in range(1<<m):s[ri,y]=math.fsum(float(d[i])*float(H[ri,y,i,j])*float(d[j]) for i in range(m) for j in range(m))
    mask=sum(1<<(m-1-j) for j in range(m) if signs[j]<0)
    index=np.arange(1<<m)^mask
    return s,s[:,index],index


def score_kl_identity(P,Q0,Qt):
    P=np.asarray(P);Q0=np.asarray(Q0);Qt=np.asarray(Qt)
    M=math.fsum(float(w) for w in P);pbar=P/M
    z=np.log(Qt)-np.log(Q0)
    raw=math.fsum(float(w)*float(v) for w,v in zip(P,z))
    norm=math.fsum(float(w)*float(v) for w,v in zip(pbar,z))
    raw_kl=independent.raw_kl(P,Q0)-independent.raw_kl(P,Qt)
    normalized_teacher_kl=independent.raw_kl(pbar,Q0)-independent.raw_kl(pbar,Qt)
    m0=math.fsum(float(x) for x in Q0);mt=math.fsum(float(x) for x in Qt)
    correction=math.log(m0)-math.log(mt)
    return dict(raw_mean=raw,raw_KL_difference=raw_kl,raw_residual=raw-raw_kl,
        pbar_mean=norm,pbar_KL_difference=normalized_teacher_kl,pbar_residual=norm-normalized_teacher_kl,
        Q0_raw_mass=m0,Qt_raw_mass=mt,model_normalization_score_shift=correction,
        normalized_model_mean=norm+correction,
        normalized_model_KL_difference=independent.raw_kl(pbar,Q0/m0)-independent.raw_kl(pbar,Qt/mt),
        model_normalization_is_explicit_secondary_identity=True)


def finite_targets(t):
    return dict(zero=0.,matched_curvature=t*t*1e-4/2,gain_1e_5=1e-5,gain_1e_4=1e-4)


def check_bridge(saved,Xc,Xt,P,finite_record,t,checks,context):
    if not np.isfinite(Xt).all():
        checks.same(saved['status'],'undefined_nonfinite_score',context+':bridge_undefined');return
    c,z=moments(Xc,P),moments(Xt,P);pb=c['pbar'];residual=Xt-Xc
    covariance=math.fsum(float(w)*(float(x)-c['mean'])*(float(y)-z['mean']) for w,x,y in zip(pb,Xc,Xt))
    checks.scalar(saved['maxabs_label_residual'],max(abs(residual)),'bridge.maxabs',context)
    checks.scalar(saved['weighted_RMS_residual'],math.sqrt(math.fsum(float(w)*float(x)*float(x) for w,x in zip(pb,residual))),'bridge.RMS',context)
    for key,num,den in (('range_ratio_Xt_over_Xc',z['range'],c['range']),('SD_ratio_Xt_over_Xc',z['std'],c['std']),
        ('range_to_SD_curvature',c['range'],c['std']),('range_to_SD_finite_normalized',z['range'],z['std']),
        ('correlation',covariance,c['std']*z['std']),('curvature_SNR',c['mean'],c['std']),('finite_normalized_SNR',z['mean'],z['std']),
        ('SNR_ratio_Xt_over_Xc',ratio(z['mean'],z['std']),ratio(c['mean'],c['std']))):
        check_ratio(saved[key],num,den,checks,context+'/'+key)
    fm=finite_record['moments']
    for method in ('hoeffding','eb_proxy'):
        row=saved['matched_margin_required_n'][method]
        check_minimum(row['curvature'],c['mean'],1e-4,c['variance'],c['range'],method,checks,context+'/bridge_curv_n')
        check_minimum(row['finite_gain'],fm['mean'],t*t*1e-4/2,fm['variance'],fm['range'],method,checks,context+'/bridge_finite_n')
        check_ratio(row['finite_over_curvature'],row['finite_gain']['n'],row['curvature']['n'],checks,context+'/bridge_n_ratio')


def check_transfer(saved,curvature_scores,zscores,P,t,checks,context):
    if not np.isfinite(zscores).all():
        checks.same(saved['status'],'undefined_nonfinite_score',context+':transfer_undefined');return
    Xt=2*zscores/(t*t);e=np.empty((2,4));slack=np.empty_like(e);scaled_slack=np.empty_like(e);roundoff=0.
    for cr in range(2):
        for mr in range(4):
            e[cr,mr]=max(float(c)-float(x) for c,x in zip(curvature_scores[cr],Xt[mr]))
            residual=[float(x)-float(c)+e[cr,mr] for c,x in zip(curvature_scores[cr],Xt[mr])]
            finite=[float(z)-.5*t*t*(float(c)-e[cr,mr]) for c,z in zip(curvature_scores[cr],zscores[mr])]
            slack[cr,mr]=min(residual);scaled_slack[cr,mr]=min(finite)
            roundoff=max(roundoff,max(abs(f-.5*t*t*r) for f,r in zip(finite,residual)))
    checks.close(saved['e_minus'],e,'transfer.e_minus',context)
    checks.close(saved['minimum_pointwise_slack'],slack,'transfer.pointwise_slack',context)
    checks.close(saved['minimum_scaled_pointwise_slack'],scaled_slack,'transfer.scaled_slack',context)
    checks.scalar(saved['scaling_identity_maxabs_residual'],roundoff,'transfer.scaling_residual',context)
    c=moments(curvature_scores[1],P);z=moments(zscores[1],P);lower=.5*t*t*(c['mean']-e[1,1])
    for key,value in dict(main_e_minus=e[1,1],normalized_mean_gain=z['mean'],normalized_mean_gain_lower_from_curvature=lower,
        normalized_mean_slack=z['mean']-lower,raw_mean_gain=z['raw_mean'],raw_mass=c['raw_mass'],
        raw_mean_gain_lower_from_curvature=.5*t*t*(c['raw_mean']-c['raw_mass']*e[1,1])).items():
        checks.scalar(saved[key],value,'transfer.'+key,context)
    targets={key:float(e[1,1]+2*eta/(t*t)) for key,eta in finite_targets(t).items()}
    check_planning(saved['curvature_event_planning'],c,targets,checks,context+'/transfer_planning')
    for key,eta in finite_targets(t).items():
        row=saved['thresholds'][key];threshold=targets[key]
        for field,value in dict(eta=eta,curvature_threshold=threshold,oracle_curvature_gap=c['mean']-threshold).items():
            checks.scalar(row[field],value,'transfer.threshold_'+field,context+'/'+key)
        checks.same(row['transfer_gives_no_support_even_with_unlimited_data'],c['mean']<=threshold,context+':transfer_no_support')
        for method in ('hoeffding','eb_proxy'):
            details=row['methods'][method]
            check_minimum(details['transferred'],c['mean'],threshold,c['variance'],c['range'],method,checks,context+'/transfer/'+key)
            check_minimum(details['direct'],z['mean'],eta,z['variance'],z['range'],method,checks,context+'/direct/'+key)
            check_ratio(details['transferred_over_direct_n'],details['transferred']['n'],details['direct']['n'],checks,context+'/transfer_ratio/'+key)
    checks.same(saved['negative_e_minus_preserved'],True,context+':e_minus_not_clipped')
    checks.same(saved['new_score_families'],0,context+':no_extra_transfer_family')
    checks.same(saved['same_curvature_event_reused_for_all_candidates_and_margins'],True,context+':event_reuse')


def compare_array(saved,expected,field,checks,context,tol=1e-12):
    """Matching numerical failures are retained, not interpreted as evidence."""
    a,b=np.asarray(saved,dtype=float),np.asarray(expected,dtype=float)
    checks.same(a.shape,b.shape,context+':'+field+':shape')
    if a.shape!=b.shape:return
    checks.same(np.isnan(a),np.isnan(b),context+':'+field+':NaN_mask')
    checks.same(np.isposinf(a),np.isposinf(b),context+':'+field+':posinf_mask')
    checks.same(np.isneginf(a),np.isneginf(b),context+':'+field+':neginf_mask')
    mask=np.isfinite(a)&np.isfinite(b)
    if mask.any():checks.close(a[mask],b[mask],field,context,tol=tol)


def differences(a):
    a=np.asarray(a,dtype=float)
    if not np.isfinite(a).all():return dict(max_pairwise=None,max_from_main=None,finite=False)
    return dict(max_pairwise=float(np.max(np.max(a,axis=0)-np.min(a,axis=0))),
                max_from_main=float(np.max(abs(a-a[1]))),finite=True)


def compare_difference(saved,expected,checks,context):
    checks.same(saved['finite'],expected['finite'],context+':finite')
    for key in ('max_pairwise','max_from_main'):checks.scalar(saved[key],expected[key],key,context)


def check_unit(a,c,checks):
    """Pure arrays: independent loops, fsum and closed-form planning rules."""
    s=a['source'];uid=s['unit']['id'];P=np.asarray(s['teacher_pmfs']);Q=np.asarray(c['pmfs'])
    theta=np.asarray(s['theta_signed']);d=np.asarray(s['fixed_d']);m=len(d)
    checks.same(c['t_values'],TS,uid+':fixed_t')
    checks.same(c['rules'],[[40,0.],[60,0.],[80,0.],[72,math.pi/7]],uid+':fixed_GH_rules')
    checks.same(c['b'],theta[:m],uid+':b');checks.same(c['signed_a'],theta[m:],uid+':signed_a')
    checks.same(np.asarray(c['fixed_d']).tobytes(),d.tobytes(),uid+':fixed_d_bytes')
    W=np.asarray([np.column_stack((theta[m:]*np.cos(t*d),theta[m:]*np.sin(t*d))) for t in (0.,*TS)])
    checks.same(c['model_W'],W,uid+':fixed_path_no_alignment')
    checks.same(Q.shape,(4,4,1<<m),uid+':PMF_axes')
    checks.close(np.linalg.norm(W,axis=2),np.broadcast_to(abs(theta[m:]),(4,m)),'fixed_row_norm',uid)
    recoded,original,index=fixed_curvature(s)
    checks.same(a['recoded_index_of_original_label'],index,uid+':original_label_reindex')
    checks.close(a['label_curvature_recoded'],recoded,'direct_label_curvature',uid)
    checks.close(a['label_curvature_original'],original,'original_label_curvature',uid)
    checks.close(a['Xc'],-original[1],'Xc',uid)
    raw_curv=np.array([[math.fsum(float(w)*float(x) for w,x in zip(p,v)) for p in P] for v in original])
    empirical=np.array([math.fsum(float(w)*float(x) for w,x in zip(s['q'],v)) for v in original])
    checks.close(a['raw_population_curvature'],raw_curv,'raw_curvature',uid)
    checks.close(a['raw_empirical_curvature'],empirical,'raw_empirical_curvature',uid)
    checks.close(s['old_population_curvature'],raw_curv,'unchanged_old_hP',uid)
    checks.close(s['old_empirical_curvature'],empirical,'unchanged_old_hemp',uid)
    for name,err in dict(population=float(np.max(abs(raw_curv-s['old_population_curvature']))),
                         empirical=float(np.max(abs(empirical-s['old_empirical_curvature'])))).items():
        checks.scalar(a['source_mean_identity_errors'][name],err,'old_mean_residual.'+name,uid)
    checks.same(a['teacher_raw_pmfs'],P,uid+':raw_P_preserved')
    pb=np.array([p/math.fsum(map(float,p)) for p in P])
    checks.same(a['teacher_pbar'],pb,uid+':explicit_pbar')
    checks.close(a['teacher_normalization_maxabs'],np.max(abs(pb-P),axis=1),'Pbar_correction',uid)
    pmass=np.array([math.fsum(map(float,p)) for p in P]);qmass=np.array([[math.fsum(map(float,v)) for v in row] for row in Q])
    checks.close(a['teacher_masses'],pmass,'raw_P_mass',uid)
    compare_array(a['candidate_raw_masses'],qmass,'raw_Q_mass',checks,uid)
    pd=differences(P);qd=[differences(q) for q in Q]
    compare_difference(a['teacher_PMF_difference'],pd,checks,uid+':teacher_PMF')
    for i in range(4):compare_difference(a['candidate_PMF_differences'][i],qd[i],checks,uid+f':model_PMF_{i}')
    # Recompute qualification using the declared raw sums, never renormalizing.
    # Floating sum vs fsum disagreements at a gate boundary are retained as audit failures.
    teacher_mass=bool(max(abs(pmass-1))<=2e-12);teacher_PMF=bool(pd['finite'] and pd['max_pairwise']<=2e-8)
    base_mass=bool(np.isfinite(qmass[0]).all() and max(abs(qmass[0]-1))<=2e-12)
    base_PMF=bool(qd[0]['finite'] and qd[0]['max_pairwise']<=2e-8)
    curvdiff=float(max(abs(original[0]-original[1])))
    cq=dict(qualified=bool(teacher_mass and teacher_PMF and curvdiff<=2e-7),label_C_D_maxdiff=curvdiff,
            teacher_mass_gate=teacher_mass,teacher_PMF_gate=teacher_PMF,rigorous_error_bound=False)
    for k,v in cq.items():
        if k=='label_C_D_maxdiff':checks.scalar(a['curvature_qualification'][k],v,'label_C_D_maxdiff',uid)
        else:checks.same(a['curvature_qualification'][k],v,uid+':curvature_gate_'+k)
    with np.errstate(invalid='ignore',divide='ignore'):
        z=np.log(Q[1:])-np.log(Q[0]);Xt=2*z/np.square(TS)[:,None,None]
    for name,value in dict(z=z,Xt_norm=Xt,Xt_norm_minus_Xc=Xt+original[1][None,None,:]).items():
        compare_array(a[name],value,name,checks,uid)
    # Integer crossing belongs to the exact saved score table. The independent
    # double-loop reconstruction is checked above; do not silently substitute
    # its differently rounded dot products into a near-integer crossing.
    stored_curvature=-np.asarray(a['label_curvature_original'])
    check_score_records(a['curvature'],stored_curvature,P,dict(zero=0.,useful=1e-4),checks,uid+'/curvature')
    identities=[];quals=[]
    for ti,t in enumerate(TS):
        where=uid+f'/t={t}';saved=a['finite_qualification'][ti];zd=differences(z[ti])
        means=np.array([[math.fsum(float(w)*float(x) for w,x in zip(p,v)) for p in P] for v in z[ti]])
        finite=bool(np.isfinite(means).all());md=float(np.ptp(means)) if finite else None
        gates=dict(raw_teacher_mass=teacher_mass,raw_teacher_PMF=teacher_PMF,base_mass=base_mass,base_PMF=base_PMF,
            perturbed_mass=bool(np.isfinite(qmass[ti+1]).all() and max(abs(qmass[ti+1]-1))<=2e-12),
            perturbed_PMF=bool(qd[ti+1]['finite'] and qd[ti+1]['max_pairwise']<=2e-8),
            z=bool(zd['finite'] and zd['max_pairwise']<=1e-9),paired_mean=bool(finite and md<=1e-10),
            strictly_positive_candidate_PMFS=bool(np.isfinite(Q[[0,ti+1]]).all() and np.all(Q[[0,ti+1]]>0)))
        checks.same(saved['gates'],gates,where+':qualification_gates')
        checks.same(saved['qualified'],all(gates.values()),where+':qualification_conjunction');quals.append(all(gates.values()))
        compare_difference(saved['z_rule_difference'],zd,checks,where+':paired_logscore_difference')
        compare_array(saved['paired_raw_means'],means,'paired_raw_means',checks,where)
        for k,v in dict(paired_mean_max_pairwise=md,paired_mean_max_from_main=float(np.max(abs(means-means[1,1]))) if finite else None,
                       candidate_only_mean_maxdiff=float(np.max(np.ptp(means,axis=0))) if finite else None,
                       teacher_only_mean_maxdiff=float(np.max(np.ptp(means,axis=1))) if finite else None).items():
            checks.scalar(saved[k],v,k,where)
        check_score_records(a['finite_gain'][ti],z[ti],P,finite_targets(t),checks,where)
        check_bridge(a['bridge'][ti],stored_curvature[1],Xt[ti,1],P[1],a['finite_gain'][ti]['main'],t,checks,where)
        check_transfer(a['transfer'][ti],stored_curvature,z[ti],P[1],t,checks,where)
        for ri in range(4):
            for pi in range(4):
                if np.isfinite(Q[[0,ti+1],ri]).all() and np.all(Q[[0,ti+1],ri]>0):
                    identity=score_kl_identity(P[pi],Q[0,ri],Q[ti+1,ri])
                    for key in ('raw_residual','pbar_residual'):checks.scalar(identity[key],0.,'KL_identity.'+key,where,atol=1e-12,rtol=0.)
                    checks.scalar(identity['normalized_model_mean'],identity['normalized_model_KL_difference'],'normalized_model_KL_identity',where,atol=1e-12,rtol=0.)
                    identities.append(dict(t=t,candidate_rule=ri,teacher_rule=pi,**identity))
    checks.same(a['all_numerically_qualified'],cq['qualified'] and all(quals),uid+':overall_numeric_qualification')
    for field in ('original_direction_bytes_preserved','old_alarm_policy_unchanged'):checks.same(a[field],True,uid+':'+field)
    checks.same(a['new_independent_validation'],False,uid+':no_validation_data')
    checks.same(c['resources']['PMF_calls'],16,uid+':producer_16_calls')
    checks.same(len(c['calls']),16,uid+':producer_call_records')
    for mi in range(4):
        for ri,(order,angle) in enumerate(c['rules']):
            call=c['calls'][mi*4+ri]
            for k,v in dict(model_index=mi,rule_index=ri,order=order,angle=angle,t=(0.,*TS)[mi],
                            quadrature_nodes=order**2,node_pattern_products=order**2*(1<<m)).items():
                checks.same(call[k],v,uid+':call_'+k)
    return dict(unit=uid,producer_numerically_qualified=cq['qualified'] and all(quals),
                raw_and_explicitly_normalized_KL_identities=identities,old_A=s['A'],old_B=s['B'],
                old_hP=float(raw_curv[1,1]),old_hemp=float(empirical[1]),saved_delta_R=s['saved_delta_R'])


def old_source_check(s,checks,locked):
    uid=s['unit']['id'];folder=OLD/'units'/uid
    with independent.SavedTree(folder/'anchor',checks,locked) as a, independent.SavedTree(folder/'diagnosis',checks,locked) as d, \
         independent.SavedTree(EXPLANATION/'units'/uid/'explanation',checks) as e, \
         independent.SavedTree(OLD/f"teacher_D{s['unit']['teacher']+1}",checks,locked) as p:
        for key in ('theta_signed','theta_recoded','signs'):
            checks.same(s[key],a.get(key),uid+':old_anchor_'+key);checks.same(s[key],e.get(key),uid+':old_explanation_'+key)
        checks.same(s['q'],a.get('q_original'),uid+':old_raw_q')
        checks.same(s['q'],e.get('empirical_frequency'),uid+':old_explanation_q')
        checks.same(s['teacher_pmfs'],p.get('pmfs'),uid+':old_raw_teacher')
        checks.same(s['teacher_rules'],p.get('rules'),uid+':old_teacher_rules')
        direction=d.get('numerics','direction')
        checks.same(np.asarray(s['fixed_d']).tobytes(),np.asarray(direction).tobytes(),uid+':old_direction_bytes')
        checks.same(s['fixed_d'],e.get('fixed_empirical_direction'),uid+':old_explanation_direction')
        for i,rule in enumerate(('C','D')):
            checks.same(s['label_Hangle'][i],d.get('reference_only','raw',rule,'label_Hangle')[0],uid+':old_label_H_'+rule)
            checks.same(d.get('reference_only','raw',rule,'patterns'),independent.patterns(len(direction)),uid+':label_order_'+rule)
        for key,dk,ek in (('A','full_raise','original_A'),('B','ref_raise','original_B'),('full_level','full_level','original_full_level')):
            checks.same(s[key],d.get(dk),uid+':old_diagnostic_'+key);checks.same(s[key],e.get(ek),uid+':old_explanation_'+key)
        checks.same(s['old_population_curvature'],e.get('population_direction_curvature'),uid+':old_hP')
        checks.same(s['old_empirical_curvature'],e.get('empirical_fixed_direction_curvature'),uid+':old_hemp')
        checks.same(s['saved_delta_R'],e.get('saved_paired_gain')[1],uid+':old_saved_gain')
        checks.same(s['original_numeric_risk_qualification'],e.get('original_risk_qualified'),uid+':old_risk_qualification')
    row=next(r for r in read(OLD/'SUMMARY.json')['rows'] if r['unit']==uid)
    for name,old in (('A','A'),('B','B'),('full_level','full_level'),('saved_delta_R','delta_R')):
        checks.same(s[name],row[old],uid+':old_summary_'+name)


def verify_inputs(checks):
    f=read(OUT/'FREEZE.json');checks.bind(OUT/'FREEZE.json')
    checks.same(f['schema'],'direction_validation_freeze_v1','freeze_schema')
    checks.same(f['baseline'],'df31ecd3d93c50f72859d364f1e5fd22faf0eed4','baseline')
    checks.same(f['handoff_sha256'],'11c9d0d8518c8f7fff6b1abd74bde35abe4e00a2ff49d75228d1126c9c1f8f89','feedback')
    checks.same(f['production_not_started'],True,'freeze_pre_execution_snapshot')
    required=dict(GL_PMF_calls=48,GL_order=128,GL_limit=10.,GL_rotation=0.,PMF_mass_tolerance=2e-12,
                  PMF_difference_tolerance=2e-8,paired_logscore_tolerance=1e-9,paired_mean_tolerance=1e-10,automatic_order_escalation=False)
    checks.same(f['independent_config'],required,'independent_configuration')
    for name in (rel(Path(__file__)),rel(Path(__file__).with_name('direction_validation_audit_test.py')),rel(Path(independent.__file__))):
        checks.require(name in f['sources'],'frozen_dependency:'+name)
    for path,expected in f['sources'].items():checks.bind(ROOT/path,expected)
    checks.bind(OUT/'HANDOFF_INPUT.md',f['handoff_sha256'])
    checks.bind(OLD/'DELIVERY_MANIFEST.json','2b537dd6e86f9547a1e540dd429b14a86a46c04d327ddb97c5b58fb783a9abd8')
    checks.bind(EXPLANATION/'DELIVERY_MANIFEST.json','8a3ca2e301540e1e89f41dc7d69cc0feafc84125899ffb0c69f07fdb203feabd')
    config=f.get('config',f.get('config_compute'));checks.require(config is not None,'producer_config_present')
    expected_units=[dict(id=f'D{d}_N{n}_r{r}',teacher=d-1,N=n,rep=r) for d in (1,2,3) for n in (1024,4096) for r in (0,1)]
    # Unit IDs are legacy-defined; match all factor cells independently and
    # compare their exact IDs to the original public CONFIG rather than inventing aliases.
    units=config['original_units'];oldunits=read(OLD/'CONFIG.json')['units']
    checks.same(units,oldunits,'unchanged_all12_units')
    checks.same(sorted((u['teacher'],u['N'],u['rep']) for u in units),sorted((u['teacher'],u['N'],u['rep']) for u in expected_units),'fixed_12_factor_cells')
    checks.same(config['t_values'],TS,'fixed_amplitudes')
    for key,value in dict(PMF_rules=[[40,0.],[60,0.],[80,0.],[72,math.pi/7]],main_PMF_index=1,main_teacher_index=1,
                         main_curvature_index=1,conditional_rule_order=['C','D'],PMF_model_order=[0.,*TS],candidate_PMF_calls=192,
                         candidate_models=48,statistical_alpha=.05,score_families=48,looks=6,per_score_per_look_delta=DELTA,
                         N_grid=list(GRID),maximum_planning_n=CAP,transfer_new_families=0,new_observations=0,new_random_draws=0,
                         new_training=0,new_teacher_PMF_calls=0,new_curvature_integrals=0,independent_validation=False,
                         automatically_escalate_quadrature=False).items():checks.same(config[key],value,'frozen_config_'+key)
    checks.same(config['gates'],dict(PMF_mass=2e-12,PMF_maxdiff=2e-8,z_maxdiff=1e-9,paired_mean_maxdiff=1e-10,
                                   label_curvature_maxdiff=2e-7,source_curvature_identity=1e-12),'frozen_producer_gates')
    locked={}
    for name in ('INPUTS_LOCK','DIAGNOSIS_LOCK','ENDPOINTS_LOCK','EVALUATION_LOCK'):
        path=OLD/(name+'.json');checks.bind(path);lock=read(path)
        checks.same(lock['freeze_sha256'],sha(OLD/'FREEZE.json'),'old_lock_freeze:'+name)
        for key,binding in lock['files'].items():
            checks.require((ROOT/key).resolve().is_relative_to(OLD.resolve()),'old_locked_path_boundary')
            if key in locked:checks.same(locked[key],binding,'consistent_old_lock:'+key)
            locked[key]=binding
    for path,binding in locked.items():checks.bind(ROOT/path,binding)
    for path,binding in read(EXPLANATION/'RESULTS.json')['artifacts'].items():checks.bind(ROOT/path,binding)
    result=read(OUT/'RESULTS.json');checks.bind(OUT/'RESULTS.json')
    checks.same(result['status'],'completed_fixed_object_feasibility','producer_completed_before_audit')
    checks.same(result['freeze_sha256'],sha(OUT/'FREEZE.json'),'producer_freeze')
    checks.same(result['source_sha256'],sha(ROOT/'script/direction_validation_compute.py'),'producer_source')
    for path,binding in result['artifacts'].items():checks.bind(ROOT/path,binding)
    checks.same([u['unit'] for u in result['unit_artifacts']],[u['id'] for u in units],'all12_no_selection')
    if checks.failures:raise ValueError('source/freeze binding failure: '+str(checks.failures[:8]))
    return f,result,units,locked


def gl_compare(gl,c,P):
    Q=np.asarray(c['pmfs'])[:,1];finite=bool(np.isfinite(gl).all() and np.all(gl>0))
    masses=np.array([math.fsum(map(float,v)) for v in gl]);pmfdiff=float(np.max(abs(gl-Q))) if finite and np.isfinite(Q).all() else None
    model_gates=dict(finite_positive=finite,raw_mass=bool(np.isfinite(masses).all() and max(abs(masses-1))<=2e-12),
                     PMF_vs_GH60=pmfdiff is not None and pmfdiff<=2e-8)
    rows=[]
    for i,t in enumerate(TS):
        valid=finite and bool(np.isfinite(Q[[0,i+1]]).all() and np.all(Q[[0,i+1]]>0))
        if valid:
            z=np.log(gl[i+1])-np.log(gl[0]);ref=np.log(Q[i+1])-np.log(Q[0]);diff=float(max(abs(z-ref)))
            rawmeans=np.array([math.fsum(float(w)*float(x) for w,x in zip(p,z)) for p in P])
            meansref=np.array([math.fsum(float(w)*float(x) for w,x in zip(p,ref)) for p in P])
            md=float(max(abs(rawmeans-meansref)))
            ids=[score_kl_identity(p,gl[0],gl[i+1]) for p in P]
        else:diff=md=None;rawmeans=meansref=None;ids=[]
        gates=dict(**model_gates,paired_logscore=diff is not None and diff<=1e-9,paired_raw_mean=md is not None and md<=1e-10)
        rows.append(dict(t=t,gates=gates,qualified=all(gates.values()),paired_logscore_maxdiff=diff,
            paired_raw_mean_maxdiff=md,GL_raw_means=rawmeans,GH60_raw_means=meansref,raw_normalized_KL_identities=ids))
    return dict(raw_masses=masses,PMF_maxdiff_vs_GH60=pmfdiff,model_gates=model_gates,finite_amplitudes=rows,
                qualified=all(row['qualified'] for row in rows),strict_error_certificate=False)


def check_oracle(summary,analyses,checks):
    rows=[dict(unit=a['source']['unit']['id'],N=a['source']['unit']['N'],A=a['source']['A'],B=a['source']['B'],
            saved_delta_R=a['source']['saved_delta_R'],old_population_curvature=float(a['source']['old_population_curvature'][1,1]),
            old_empirical_curvature=float(a['source']['old_empirical_curvature'][1])) for a in analyses]
    def group(rs):
        policies={}
        for name in ('A','B','A_star','B_star'):
            mask=[bool(r[name[0]]) and (not name.endswith('star') or r['old_population_curvature']<0) for r in rs]
            policies[name]=dict(selected=sum(mask),mean_risk_difference_from_always_HIGH=math.fsum((1-int(x))*r['saved_delta_R'] for x,r in zip(mask,rs))/len(rs),
                captured_gain_above_0p001=sum(x and r['saved_delta_R']>.001 for x,r in zip(mask,rs)),total_gain_above_0p001=sum(r['saved_delta_R']>.001 for r in rs),
                selected_HIGH_disadvantages=sum(x and r['saved_delta_R']<0 for x,r in zip(mask,rs)))
        gaps=[r['old_population_curvature']-r['old_empirical_curvature'] for r in rs]
        return dict(units=len(rs),policies=policies,hP_minus_hemp_mean=math.fsum(gaps)/len(gaps),hP_minus_hemp_min=min(gaps),
                    hP_minus_hemp_max=max(gaps),hP_above_hemp_count=sum(x>0 for x in gaps))
    expected=dict(all_units=group(rows),by_N={str(n):group([r for r in rows if r['N']==n]) for n in (1024,4096)},
        raw_main_population_curvature_used=True,retrospective_oracle_only=True,new_policy_executed=False,
        not_an_upper_bound_over_all_validation_policies=True,bias_causal_components_identified=False)
    checks.same(summary['oracle_retrospective'],expected,'oracle_retrospective_algebra_only')
    checks.same([r['unit'] for r in summary['rows']],[r['unit'] for r in rows],'summary_all12')
    for saved,own in zip(summary['rows'],rows):
        for key,value in own.items():checks.same(saved[key],value,'summary_original:'+key)
    checks.same(summary['resources']['PMF_calls'],192,'producer_total192_calls')
    checks.same(summary['resources']['score_functions'],48,'48_score_families')
    checks.same(summary['resources']['candidate_models'],48,'48_model_points')
    checks.same(summary['numerical_qualified_units'],sum(a['all_numerically_qualified'] for a in analyses),'qualified_unit_count')
    checks.same(summary['numerical_qualified_curvature_scores'],sum(a['curvature_qualification']['qualified'] for a in analyses),'qualified_curvature_count')
    checks.same(summary['numerical_qualified_finite_scores'],sum(q['qualified'] for a in analyses for q in a['finite_qualification']),'qualified_finite_count')


def audit(arrays,integrate,counter=None):
    checks=Checks();f,produced,units,locked=verify_inputs(checks);reports=[];analyses=[];calls=0
    for unit,item in zip(units,produced['unit_artifacts']):
        uid=unit['id'];expected={v['path']:v for key,v in item.items() if key!='unit'}
        with independent.SavedTree(OUT/'units'/uid/'analysis',checks,expected) as tree:a=tree.get()
        with independent.SavedTree(OUT/'units'/uid/'candidate_pmfs',checks,expected) as tree:c=tree.get()
        checks.same(a['source']['unit'],unit,uid+':unit_identity');old_source_check(a['source'],checks,locked)
        report=check_unit(a,c,checks);analyses.append(a);glkey=uid+'__GL_PMFS'
        statuskey=uid+'__GL_STATUS'
        if checks.failures:raise ValueError('source or algebra discrepancy before independent integration: '+str(checks.failures[:8]))
        if integrate:
            arrays[glkey]=np.full((4,len(c['pmfs'][0,0])),np.nan)
            arrays[statuskey]=np.full(4,'not_attempted',dtype='<U512')
            for mi,W in enumerate(c['model_W']):
                calls+=1
                if counter is not None:counter['GL_PMF_calls']=calls
                try:
                    result=independent.gaussian_legendre_pmf(c['b'],W,order=128,limit=10.,rotation=0.)
                    arrays[glkey][mi]=result['pmf']
                    arrays[statuskey][mi]='completed'
                except (FloatingPointError,ValueError,OverflowError) as exc:
                    # This point remains nonfinite and unqualified, with no retry.
                    # The frozen finite inputs and fixed helper are already bound.
                    arrays[statuskey][mi]=type(exc).__name__+': '+str(exc)[:480]
        gl=np.asarray(arrays[glkey]);checks.same(gl.shape,(4,len(a['source']['q'])),uid+':GL_array_shape')
        checks.same(np.asarray(arrays[statuskey]).shape,(4,),uid+':GL_status_shape')
        checks.require(all(str(s)!='not_attempted' for s in arrays[statuskey]),uid+':all_four_GL_calls_attempted')
        report['GL_call_status']=np.asarray(arrays[statuskey]).tolist()
        report['GL_qualification']=gl_compare(gl,c,a['source']['teacher_pmfs'])
        for row in report['GL_qualification']['finite_amplitudes']:
            for identity in row['raw_normalized_KL_identities']:
                for key in ('raw_residual','pbar_residual'):checks.scalar(identity[key],0.,'GL_KL_identity.'+key,uid,atol=1e-12,rtol=0.)
                checks.scalar(identity['normalized_model_mean'],identity['normalized_model_KL_difference'],'GL_normalized_model_KL',uid,atol=1e-12,rtol=0.)
        reports.append(report)
    checks.same(sorted(arrays),sorted(u['id']+suffix for u in units for suffix in ('__GL_PMFS','__GL_STATUS')),'exact_saved48_GLPmf_arrays')
    checks.same(calls,48 if integrate else 0,'actual_integration_count')
    summary=read(OUT/'SUMMARY.json')['derived'];check_oracle(summary,analyses,checks)
    qualified=all(r['GL_qualification']['qualified'] for r in reports)
    return dict(schema='direction_validation_audit_v1',passed=not checks.failures,
        status='completed' if qualified else 'completed_with_numerical_qualification_limits',
        failures=checks.failures,checks=checks.count,critical_n_checks=checks.minimum_n_checks,planning_rows=checks.planning_rows,
        numerical_maxima=checks.maxima,source_sha256=sha(Path(__file__)),test_sha256=sha(Path(__file__).with_name('direction_validation_audit_test.py')),
        reused_independent_sha256=sha(Path(independent.__file__)),freeze_sha256=sha(OUT/'FREEZE.json'),producer_results_sha256=sha(OUT/'RESULTS.json'),
        input_hashes=checks.hashes,unit_checks=reports,numerical_qualification=dict(GL_all_qualified=qualified,
            GL_qualified_units=sum(r['GL_qualification']['qualified'] for r in reports),producer_qualified_units=sum(r['producer_numerically_qualified'] for r in reports),
            numerical_qualification_required_for_execution_pass=False),
        initial_run_actual_counts=dict(GL_PMF_calls=48,new_teacher_PMF_calls=0,new_curvature_integrals=0,new_random_draws=0,new_training=0,new_validation_observations=0),
        scope=dict(all_original_units=12,independent_models=48,GL_rule=dict(order=128,limit=10.,rotation=0.,raw_mass_preserved=True),
            per_label_curvature='explicit double loops and original-label XOR reindex',moments='independent fsum with explicit raw and Pbar quantities',
            critical_n='independent full-range integer binary search and strict previous-n comparison',
            check_new_integrals=0,source_arrays='only saved anchor, direction, Hessian, teacher and explanation arrays; no training-prior or trajectory replay',
            strict_integration_error_certificate=False,statistical_confidence_claim=False,power_claim=False,oracle_reweighting_not_validation_data=True,
            transfer_additional_families=0,transfer_is_score_table_algebra_not_Taylor_certificate=True,
            original_diagnostic_direction_not_training_kick=True,no_new_policy_or_population_mode_selected=True))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as file:json.dump(native(value),file,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    for flag in ('self-test','run','check'):mode.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.self_test:
        import direction_validation_audit_test as tests
        result=tests.run_tests();print(json.dumps(native(result),indent=2,allow_nan=False));return 0 if result['passed'] else 1
    receipt=OUT/'AUDIT.json';raw=OUT/'AUDIT_ARRAYS.npz';failure=OUT/'AUDIT_FAILURE.json';partial=OUT/'AUDIT_PARTIAL_ARRAYS.npz'
    if args.check:
        old=read(receipt)
        if sha(raw)!=old['arrays_artifact']['sha256']:raise ValueError('saved GL array hash mismatch')
        with np.load(raw,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
        current=audit(arrays,False)
        expected={k:v for k,v in old.items() if k not in ('created_utc','seconds','environment','arrays_artifact')}
        matched=native(current)==expected
        result=dict(passed=bool(current['passed'] and matched),failures=current['failures']+([] if matched else ['stable audit receipt differs']),
                    checks=current['checks'],GL_PMF_calls_this_check=0,stored_GL_PMF_calls=48,numerical_qualification=current['numerical_qualification'])
        print(json.dumps(native(result),indent=2,allow_nan=False));return 0 if result['passed'] else 1
    if any(p.exists() for p in (receipt,raw,failure,partial)):raise FileExistsError('audit attempt exists; no overwrite or automatic retry')
    tick=time.perf_counter();arrays={};counter=dict(GL_PMF_calls=0)
    try:
        result=audit(arrays,True,counter)
        with raw.open('xb') as file:np.savez_compressed(file,**arrays)
        result.update(created_utc=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-tick,
            environment=dict(python=platform.python_version(),numpy=np.__version__,BLAS_threads=1),
            arrays_artifact=dict(path=rel(raw),sha256=sha(raw),bytes=raw.stat().st_size))
        write_new(receipt,result)
        print(json.dumps(native(dict(passed=result['passed'],failures=result['failures'],checks=result['checks'],
            initial_run_actual_counts=result['initial_run_actual_counts'],numerical_qualification=result['numerical_qualification'])),indent=2))
        return 0 if result['passed'] else 1
    except Exception:
        if arrays:
            with partial.open('xb') as file:np.savez_compressed(file,**arrays)
        write_new(failure,dict(status='failed_no_retry',source_sha256=sha(Path(__file__)),traceback=traceback.format_exc(),
            actual_GL_calls_attempted=counter['GL_PMF_calls'],saved_GL_unit_arrays=sum(k.endswith('__GL_PMFS') for k in arrays),seconds=time.perf_counter()-tick))
        raise


if __name__=='__main__':raise SystemExit(main())

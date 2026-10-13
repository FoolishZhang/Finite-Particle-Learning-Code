"""Prespecified conditional scoring contrasts; no new RNG or model integration."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_six_frozen_targets'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def contrast_vectors(means,weights):
    """[group,profile,candidate,pattern] -> [11,group]."""
    a=np.asarray(means,dtype=np.float64)
    assert a.shape[1:]==(2,35,16) and np.isfinite(a).all()
    p=np.asarray(weights['population_weights']);d=np.asarray([r['weights'] for r in weights['data_weights']])
    assert p.shape==(16,) and d.shape==(16,16)
    records=[]
    for profile in range(2):
        fixed=a[:,profile,2:18,:];fact=a[:,profile,18:34,:];truth=a[:,profile,0,None,:]
        for w in [p[None,:],d]:
            records.append(((fixed-truth)*w).sum(axis=2).mean(axis=1))
            records.append(((fixed-fact)*w).sum(axis=2).mean(axis=1))
    records.append(((a[:,0,1,:]-a[:,0,0,:])*p).sum(axis=1))
    for profile in range(2):
        records.append(((a[:,profile,2:18,:].mean(axis=1)-a[:,profile,34,:])*p).sum(axis=1))
    return np.stack(records)
def summarize(values,config):
    x=np.asarray(values);assert x.shape==(11,config['groups'])
    critical=float(t.ppf(1-config['family_alpha']/(2*config['family_size']),config['df']))
    rows=[]
    for name,sign,v in zip(config['primary_order'],config['primary_expected_sign'],x):
        mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)));lo=mean-critical*se;hi=mean+critical*se
        rows.append(dict(name=name,group_values=v.tolist(),mean=mean,SE=se,lower=lo,upper=hi,expected_sign=sign,effect_passed=bool(hi<0 if sign=='negative' else lo>0),precision_passed=bool(se<=config['SE_threshold']),resolved_sign='negative' if hi<0 else 'positive' if lo>0 else 'unresolved'))
    return critical,rows
def verify_freeze():
    f=read(OUT/'FREEZE.json');assert f['run_authorized']
    for p,h in f['inputs'].items():assert sha(ROOT/p)==h,p
    return f
def run():
    verify_freeze();cfg=read(OUT/'CONFIG.json');c=read(OUT/'CANDIDATES.json')
    mc=read(OUT/'MC/RESULTS.json');ref=read(OUT/'REFERENCE/RESULTS.json')
    groups=mc['groups'];assert [r['group'] for r in groups]==list(range(32))
    means=np.array([r['means'] for r in groups]);v=contrast_vectors(means,c);critical,rows=summarize(v,cfg)
    refs=ref['records'];assert [r['index'] for r in refs]==list(range(35))
    pmf=np.array([r['main_pmf'] for r in refs]);assert pmf.shape==(35,16) and (pmf>0).all()
    exact=-np.log(pmf);pseudo=np.broadcast_to(exact,(1,2,35,16));integrated=contrast_vectors(pseudo,c)[:,0]
    for r,delta in zip(rows,integrated):
        r['integrated_loglikelihood_contrast']=float(delta)
        r['MC_distortion_contrast']=r['mean']-float(delta)
        r['MC_distortion_interval']=[r['lower']-float(delta),r['upper']-float(delta)]
        r['shifted_interval_scope']='Same original interval translated by a fixed integral contrast; validity uses reported PMF qualification. No independent family or causal share.'
    p=np.array(c['population_weights']);d=np.array([r['weights'] for r in c['data_weights']]);lossP=exact@p;lossD=exact@d.T
    kl=(p*np.log(p/pmf)).sum(axis=1);Vp=np.array([r['V_main'] for r in refs])
    candidates=[]
    for j,row in enumerate(c['candidates']):
        sc=(means[:,:,j,:]*p).sum(axis=-1)
        candidates.append(dict(index=j,name=row['name'],rho=row['rho'],reference_joint_KL=float(kl[j]),integrated_P=float(lossP[j]),integrated_D_by_stream=lossD[j].tolist(),V=float(Vp[j]),mean_score_P=sc.mean(axis=0).tolist(),SE_score_P=(sc.std(axis=0,ddof=1)/np.sqrt(32)).tolist()))
    per_model=[]
    for s in range(16):
        for profile in range(2):
            for target,j in [('truth',0),('FACT',18+s)]:
                for kind,w in [('P',p),('D',d[s])]:
                    z=((means[:,profile,2+s]-means[:,profile,j])*w).sum(axis=1)
                    per_model.append(dict(stream=s,profile=cfg['profiles'][profile],weights=kind,comparison='FIXED_minus_'+target,mean=float(z.mean()),SE=float(z.std(ddof=1)/np.sqrt(32)),integrated_difference=float((exact[2+s]-exact[j])@w),V_population_reference_difference=float(Vp[2+s]-Vp[j]),scope='Descriptive; no per-model significance test. V is always the population P potential, including when the score weights are D.'))
    vread=dict(FIXED_minus_truth=float(Vp[2:18].mean()-Vp[0]),FIXED_minus_FACT=float((Vp[2:18]-Vp[18:34]).mean()),high_minus_truth=float(Vp[1]-Vp[0]),FIXED_mean_minus_MEAN_FIXED=float(Vp[2:18].mean()-Vp[34]))
    record=dict(schema='scbm6eval-results-v1',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [OUT/'MC/RESULTS.json',OUT/'REFERENCE/RESULTS.json',OUT/'CONFIG.json',OUT/'CANDIDATES.json']},critical_t=critical,primary_family=rows,primary_effects_passed=sum(r['effect_passed'] for r in rows),primary_precision_passed=sum(r['precision_passed'] for r in rows),primary_combined_passed=sum(r['effect_passed'] and r['precision_passed'] for r in rows),candidates=candidates,per_model_descriptive=per_model,V_panel_reference=vread,mean_fixed_risk_comparison=dict(mean_individual_KL=float(kl[2:18].mean()),mean_matrix_KL=float(kl[34]),difference=float(kl[2:18].mean()-kl[34])),new_RNG=0,new_model_integrals=0,new_training=0,scope=cfg['conditional_scope'])
    record['reference_qualification']=dict(PMF=ref['PMF_qualification_passed'],V=ref['V_qualification_passed'],all=ref['qualification_passed'])
    record['native_numerical_qualification']=mc['numerical_qualification']
    record['qualification_policy']='MC statistical effects and precision are not retroactively changed by reference/numerical qualification failures. Failed qualification blocks the corresponding interpretation; no automatic refinement.'
    save(OUT/'RESULTS.json',record)
    print(json.dumps(dict(primary=[{k:r[k] for k in ['name','mean','SE','lower','upper','effect_passed','precision_passed']} for r in rows],risk=record['mean_fixed_risk_comparison'],V=vread),ensure_ascii=False))
def selftest():
    # Deterministic manufactured losses: scalar candidate index + pattern term.
    a=np.empty((32,2,35,16))
    for g in range(32):
        for h in range(2):
            a[g,h]=np.arange(35)[:,None]+np.arange(16)[None,:]/16+g/32+h
    c=dict(population_weights=[1/16]*16,data_weights=[dict(weights=[1/16]*16) for _ in range(16)])
    expected=np.array([9.5,-16,9.5,-16]*2+[1,-24.5,-24.5])
    assert np.array_equal(contrast_vectors(a,c),np.broadcast_to(expected[:,None],(11,32)))
    # Unequal data weights and pattern/candidate interactions expose axis swaps.
    c['population_weights']=(np.arange(1,17)/136).tolist()
    for s in range(16):c['data_weights'][s]['weights']=(np.roll(np.arange(1,17),s)/136).tolist()
    a=a+np.arange(35)[None,None,:,None]*np.arange(16)[None,None,None,:]/1000
    actual=contrast_vectors(a,c);manual=[]
    for h in range(2):
        for kind in ['P','D']:
            for target in ['truth','FACT']:
                row=[]
                for g in range(32):
                    z=0.
                    for s in range(16):
                        w=c['population_weights'] if kind=='P' else c['data_weights'][s]['weights']
                        j=0 if target=='truth' else 18+s
                        z+=sum(w[y]*(a[g,h,2+s,y]-a[g,h,j,y]) for y in range(16))/16
                    row.append(z)
                manual.append(row)
    assert np.allclose(actual[:8],manual,rtol=0,atol=1e-13)
    print(json.dumps(dict(passed=True,new_RNG=0,new_model_integrals=0,contrasts=11)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['selftest','run']);a=p.parse_args()
    selftest() if a.mode=='selftest' else run()

"""Extract fixed old candidates and weights; no new scientific random draws."""
from pathlib import Path
import hashlib,itertools,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_six_frozen_targets'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    paths={k:ROOT/v for k,v in dict(train='artifacts/scbm_multicorrelation/TRAIN/RESULTS.json',reference='artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json',truth='artifacts/scbm_invariance/REFERENCE/MODEL.json',paper='artifacts/paper_v7_20261003/MANIFEST.json',paper_lock='artifacts/paper_v7_20261003/VERSION_LOCK.json').items()}
    train,reference,truth=[read(paths[k]) for k in ['train','reference','truth']]
    lock=read(paths['paper_lock']);assert lock['manifest_sha256']==sha(paths['paper'])
    pairs=list(itertools.combinations(range(4),2));rho0=[.25,-.15,.1,.1,-.15,.25]
    records=[]
    def add(name,rho,p,origin,stream=None):
        C=np.eye(4)
        for (i,j),v in zip(pairs,rho):C[i,j]=C[j,i]=v
        assert np.linalg.eigvalsh(C).min()>=.09999
        records.append(dict(index=len(records),name=name,rho=rho,covariance=C.tolist(),minimum_eigenvalue=float(np.linalg.eigvalsh(C).min()),reused_pmf=p,origin=origin,stream=stream))
    for name,t in [('truth',0.),('high',.75)]:
        r=next(r for r in truth['records'] if r['t']==t)
        add(name,[(1+t)*v for v in rho0],r['main_pmf'],dict(path=paths['truth'].relative_to(ROOT).as_posix(),t=t))
    for method in ['FIXED','FACT']:
        for s,r in enumerate(train['records']):
            assert r['stream']==s
            rho=r['outputs'][method]['TAIL']['rho']
            ref=next(x for x in reference['records'] if x['rho']==rho)
            add(f'{method}_TAIL_{s:02d}',rho,ref['main_pmf'],dict(path=paths['reference'].relative_to(ROOT).as_posix(),key=ref['key'],method=method,kind='TAIL'),s)
    mean=[math.fsum(records[2+s]['rho'][j] for s in range(16))/16 for j in range(6)]
    add('MEAN_FIXED',mean,None,dict(type='arithmetic mean of 16 fixed TAIL parameter vectors; geometric probe, not new fitted method'))
    weights=[];inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths.values()}
    for s,row in enumerate(train['records']):
        p=ROOT/row['input_file'];assert sha(p)==row['input_sha256'];inputs[row['input_file']]=sha(p)
        with np.load(p,allow_pickle=False) as f:y=f['active_labels'].astype(np.int64)
        assert y.shape==(4096,4) and np.isin(y,[0,1]).all()
        counts=np.bincount(y@np.array([8,4,2,1]),minlength=16)
        weights.append(dict(stream=s,counts=counts.tolist(),weights=(counts/4096).tolist()))
    candidates=dict(schema='scbm6eval-candidates-v1',mean=[-.6,-.2,.2,.6],mu=[-.6,-.2,.2,.6],pair_order=[list(p) for p in pairs],patterns=list(map(list,itertools.product([0,1],repeat=4))),candidates=records,population_weights=records[0]['reused_pmf'],data_weights=weights,inputs=inputs,scope='Frozen TAIL deployment means, not optimizer LAST; P uses old numerically qualified truth weights without renormalization.')
    save(OUT/'CANDIDATES.json',candidates)
    config=dict(schema='scbm6eval-config-v1',groups=32,banks_per_group=1024,K=100,nuisance=256,chunk=64,seed_base=202610032100,roles=dict(active=0,noise=1),profiles=['raw','FACT'],saved_prefix_banks=4,numerical_tolerance=1e-4,native_control_candidate_indices=[0,1,2,18,17,33],dtype='float64',workers=2,canonical_nuisance_label=0,inner_particle_reduction='negative log arithmetic mean of 100 likelihood products, separately per bank',outer_reduction='arithmetic bank mean, then independent-group statistics',family_alpha=.05,family_size=11,df=31,SE_threshold=.0002,
        primary_order=['raw_P_FIXED_minus_truth','raw_P_FIXED_minus_FACT','raw_D_FIXED_minus_truth','raw_D_FIXED_minus_FACT','FACT_P_FIXED_minus_truth','FACT_P_FIXED_minus_FACT','FACT_D_FIXED_minus_truth','FACT_D_FIXED_minus_FACT','raw_P_high_minus_truth','raw_P_FIXED_mean_minus_MEAN_FIXED','FACT_P_FIXED_mean_minus_MEAN_FIXED'],
        primary_expected_sign=['negative']*4+['positive']*4+['negative','negative','positive'],
        reference=dict(pmf_orders=[24,32,40],pmf_main_order=32,pmf_maximum_change=5e-8,normalization_tolerance=2e-12,V_orders=[[16,24],[24,32],[32,40]],V_main_orders=[24,32],V_maximum_change=1e-8,V_center='fixed old truth PMF spin marginal means and true spin covariances; candidate pair moment by GH'),
        independent_reference=dict(V_candidate_indices=[0,1,2,17,34],V_GL=40,V_GH=48,root='symmetric',mean_pmf_order=36,maximum_difference=1e-8,pmf_maximum_difference=5e-8),new_observations=0,new_training=0,new_scientific_selection=0,conditional_scope='Intervals condition on frozen old 16 data sets, all candidates, and qualified reference P weights; development t approximation, not new training-population inference.')
    save(OUT/'CONFIG.json',config)
    save(OUT/'BASELINE.json',dict(base_commit='cf4f8f51f14083634bc55375d0fc372fad581d20',paper_v7_manifest_sha256=sha(paths['paper']),paper_v7_lock_sha256=sha(paths['paper_lock']),paper_sealed_utc=lock['created_utc'],paper_immutable=True,handoff_sha256=sha(ROOT/'HANDOFF.md'),new_results_excluded_from_paper=True,external_sandbox_obtained=False,preparation_source_sha256=sha(__file__),inputs=inputs))
    print(json.dumps(dict(candidates=len(records),old_data_sets=len(weights),new_RNG=0,new_integrals=0,new_training=0)))
if __name__=='__main__':main()

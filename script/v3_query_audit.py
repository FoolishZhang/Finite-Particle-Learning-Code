"""Independent enumeration of conditional risks and count-blind information floors."""
import os
for _key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[_key]='2'
from pathlib import Path
import argparse, hashlib, itertools, json, math
from functools import lru_cache
import numpy as np
from scipy.special import xlogy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/v3_query'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def dump(p,obj):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def states(d):return ((np.arange(2**d)[:,None]>>np.arange(d))&1)
def bern_entropy(p):return -xlogy(p,p)-xlogy(1-p,1-p)
@lru_cache(None)
def design(d,k):
    c=states(d);res=[]
    for obs in itertools.combinations(range(d),k):
        eid=c[:,obs]@(2**np.arange(k));count=c[:,obs].sum(1)
        for target in set(range(d))-set(obs):res.append((eid,count,c[:,target]))
    return res
def independent_metrics(p,q,k):
    d=int(round(math.log2(len(p))));logrisk=0.;brier=0.;blind=0.;blindb=0.
    for eid,count,tgt in design(d,k):
        pe=np.bincount(eid,weights=p,minlength=2**k);p1=np.bincount(eid,weights=p*tgt,minlength=2**k)
        qe=np.bincount(eid,weights=q,minlength=2**k);q1=np.bincount(eid,weights=q*tgt,minlength=2**k)
        pp=p1/pe;qp=q1/qe
        logrisk+=np.sum(pe*(xlogy(pp,pp/qp)+xlogy(1-pp,(1-pp)/(1-qp))))
        brier+=np.sum(pe*(pp-qp)**2)
        pn=np.bincount(count,weights=p,minlength=k+1);pt=np.bincount(count,weights=p*tgt,minlength=k+1)
        pn1=pt/pn
        blind+=np.sum(pn*bern_entropy(pn1))-np.sum(pe*bern_entropy(pp))
        ec=states(k).sum(1)
        blindb+=np.sum(pe*(pp-pn1[ec])**2)
    denom=len(design(d,k))
    return dict(conditional_excess_KL=float(logrisk/denom),excess_Brier=float(brier/denom),
                count_blind_KL_floor=float(blind/denom),count_blind_Brier_floor=float(blindb/denom))
def analytic_checks():
    c=states(3);n=c.sum(1);p=(2.**(n*(n-1)/2))/18;p=p[np.arange(8)^1]
    t=(math.sqrt(41)-5)/8;pn=np.bincount(n,weights=p,minlength=4)
    w=t**c[:,0];zn=np.bincount(n,weights=w,minlength=4);q=pn[n]*w/zn[n]
    eid=c[:,0]+2*c[:,1];pe=np.bincount(eid,weights=p);p1=np.bincount(eid,weights=p*c[:,2]);qp=np.bincount(eid,weights=q*c[:,2])/np.bincount(eid,weights=q)
    pc=p1/pe;en=states(2).sum(1);nt=np.bincount(en,weights=p1)/np.bincount(en,weights=pe)
    blindb=float(np.sum(pe*(pc-nt[en])**2));assert abs(blindb-1/120)<1e-14
    joint_b=float(np.sum(pe*(pc-qp)**2));assert joint_b>=blindb-1e-14
    c4=states(4);n4=c4.sum(1);p4=8.**(c4[:,0]*c4[:,1]+c4[:,2]*c4[:,3])/121
    pn4=np.bincount(n4,weights=p4);q4=pn4[n4]/np.array([math.comb(4,int(j)) for j in n4])
    d4=float(np.sum(p4*np.log(p4/q4)));formula=(20/121)*(.8*math.log(12/5)+.2*math.log(.3))
    assert abs(d4-formula)<1e-14 and np.max(abs(p4@c4-9/11))<1e-14
    return dict(d3_t=t,d3_joint_opt_KL=float(np.sum(p*np.log(p/q))),d3_query_only_Brier_floor=blindb,
        d3_joint_opt_query_excess_Brier=joint_b,d3_context_probabilities=pe.tolist(),d3_target_given_context=pc.tolist(),
        d4_joint_KL_floor=d4,d4_same_count_pair_products=[.4*.4,.05*.05])
def audit_a():
    r=read(OUT/'A_RESULTS.json');a={key:np.asarray(value) for key,value in read(OUT/'A_PMFS.json').items()}
    if (ROOT/r['array_file']).exists():
        assert sha(ROOT/r['array_file'])==r['array_sha256']
        local=np.load(ROOT/r['array_file'])
        for key in a:assert np.array_equal(a[key],local[key])
    maxerr=0.;diagnostics=[]
    for row in r['rows']:
        name,code,family=row['teacher'],row['encoding'],row['family'];p=a[f'truth_{name}_{code}'];q=a[f'fit_{name}_{code}_{family}']
        for k in [3,7]:
            m=independent_metrics(p,q,k)
            for key in ['conditional_excess_KL','excess_Brier']:maxerr=max(maxerr,abs(m[key]-row['queries'][str(k)][key]))
            if family=='cardinality':assert m['conditional_excess_KL']>=m['count_blind_KL_floor']-1e-11
            if family=='product':diagnostics.append(dict(teacher=name,encoding=code,k=k,**m))
        assert abs(float(np.sum(p*np.log(p/q)))-row['joint_KL'])<1e-11
        if code=='flip21' and family in ['product','ising','linear_ar']:
            native=next(v for v in r['rows'] if v['teacher']==name and v['encoding']=='native' and v['family']==family)
            assert abs(native['joint_KL']-row['joint_KL'])<=native['gap']+row['gap']+1e-10
    n=states(8).sum(1);p=a['truth_allocation_native'];s=a['truth_allocation_sym_native']
    count_err=float(np.max(abs(np.bincount(n,weights=p)-np.bincount(n,weights=s))))
    marginal_err=float(np.max(abs((p-s)@states(8))))
    assert maxerr<1e-10 and count_err<1e-12 and marginal_err<1e-12
    return dict(passed=True,max_metric_error=maxerr,same_count_error=count_err,same_marginal_error=marginal_err,
        qualified=r['qualified'],fit_count=r['population_fits'],blindness_diagnostics=diagnostics,analytic=analytic_checks())
def audit_b():
    from v3_query_experiments import conditional_teachers
    from v3_query_core import pmf_from_params
    from scipy.stats import t
    r=read(OUT/'B_RESULTS.json');a=None;ps=conditional_teachers()
    if (ROOT/r['array_file']).exists():
        assert sha(ROOT/r['array_file'])==r['array_sha256'];a=np.load(ROOT/r['array_file'])
    maxerr=0.;floors=[]
    for world,px in ps.items():
        for code,mask in [('native',0),('flip21',21)]:
            for x,p0 in enumerate(px):
                p=p0[np.arange(256)^mask]
                for k in [3,7]:floors.append(dict(world=world,encoding=code,x=x,k=k,**independent_metrics(p,p,k)))
    for row in r['rows']:
        world,code,x,rep,family=(row[key] for key in ['world','encoding','x','rep','family'])
        p=ps[world][x][np.arange(256)^(0 if code=='native' else 21)];q=pmf_from_params(row['params'])
        if a is not None:assert np.max(abs(q-a[f'q_{world}_{rep}_{code}_{x}_{family}']))<1e-13
        for k in [3,7]:
            m=independent_metrics(p,q,k)
            for key in ['conditional_excess_KL','excess_Brier']:maxerr=max(maxerr,abs(m[key]-row['queries'][str(k)][key]))
        assert abs(float(np.sum(p*np.log(p/q)))-row['joint_KL'])<1e-11
    cierr=0.;critical=t.ppf(1-.05/64,15)
    for con in r['contrasts']:
        v=[]
        qualified=True
        for rep in range(16):
            s=[row for row in r['rows'] if row['world']==con['world'] and row['encoding']==con['encoding'] and row['rep']==rep]
            qualified=qualified and all(z['qualified'] for z in s if z['family'] in ['product',con['family']])
            k=str(con['k']);v.append(sum(z['queries'][k]['conditional_excess_KL'] for z in s if z['family']=='product')/2-
                                    sum(z['queries'][k]['conditional_excess_KL'] for z in s if z['family']==con['family'])/2)
        mean=np.mean(v);se=np.std(v,ddof=1)/4;ci=[mean-critical*se,mean+critical*se]
        assert con['contrast_qualified']==qualified and con['practical_gain']==bool(qualified and ci[0]>1e-3)
        cierr=max(cierr,float(np.max(abs(np.array(ci)-con['simultaneous_t_interval']))))
    meanmatch=max(float(np.max(abs((ps['dependent'][x]-ps['independent'][x])@states(8)))) for x in [0,1])
    assert maxerr<1e-10 and cierr<1e-12 and meanmatch<1e-12
    return dict(passed=True,all_finite_rows_recomputed=len(r['rows']),max_metric_error=maxerr,max_ci_error=cierr,
        conditional_mean_match=meanmatch,blindness_diagnostics=floors,qualified=r['qualified'],total_fits=r['total_fits'])
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('package',choices=['A','B']);pa.add_argument('--check',action='store_true');args=pa.parse_args()
    obj=audit_a() if args.package=='A' else audit_b();path=OUT/f'{args.package}_AUDIT.json'
    if args.check:
        saved=read(path);assert saved['passed'] and obj['passed']
        for key in ['qualified','fit_count','all_finite_rows_recomputed','total_fits']:
            if key in obj:assert saved[key]==obj[key]
    else:dump(path,obj)
    print(json.dumps({key:value for key,value in obj.items() if key not in ['blindness_diagnostics','analytic']},ensure_ascii=False))

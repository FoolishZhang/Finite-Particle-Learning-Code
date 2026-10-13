"""Independent qualification and audits for the fixed public Optdigits workflow."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,itertools,json,math,platform,time,traceback
import numpy as np
import scipy
from scipy.special import log_expit,logsumexp,expit
from scipy.optimize import minimize
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/use_digits_v3_20261005'
NAMES=('IND','L001','L010','L100');METHODS=('RAW100','OUT16','INNER1600','JVI100','AGH')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def close(a,b,tol=2e-10):
    e=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert e<=tol,(e,tol);return e

def direct_logq(theta,profiles,order=24):
    d=profiles.shape[1];b=np.array(theta[:d]);W=np.array(theta[d:]).reshape(d,2)
    x,w=np.polynomial.hermite.hermgauss(order);x*=math.sqrt(2);w/=math.sqrt(math.pi)
    z=np.array(list(itertools.product(x,x)));lw=np.log(np.outer(w,w).ravel());eta=z@W.T+b
    return np.array([logsumexp(lw+log_expit((2*y-1)*eta).sum(axis=1)) for y in profiles])

def direct_scores(theta,profiles,prior):
    d=profiles.shape[1];eta=prior@np.array(theta[d:]).reshape(d,2).T+theta[:d];result=np.empty((4,len(profiles)))
    for j,y in enumerate(profiles):
        a=log_expit((2*y-1)*eta).sum(axis=1);low=math.log(100)-logsumexp(a[:100])
        result[0,j]=low;result[1,j]=np.mean([math.log(100)-logsumexp(block) for block in a.reshape(16,100)])
        result[2,j]=math.log(1600)-logsumexp(a)
        loo=[math.log(99)-logsumexp(np.delete(a[:100],i)) for i in range(100)]
        result[3,j]=100*low-99*float(np.mean(loo))
    return result

def qualify():
    import udigits_compute as producer
    start=time.perf_counter()
    try:
        errors=[]
        for d,coords in ((6,list(range(18))),(64,[0,7,31,63,64,79,95,127,128,159,175,191])):
            profiles=np.array(list(itertools.product((0.,1.),repeat=6)))
            if d==64:profiles=np.tile(profiles,(1,11))[:,:64]
            theta=np.concatenate((np.linspace(-.3,.3,d),.1*np.sin(np.arange(d*2))))
            freq=np.arange(1.,len(profiles)+1);freq/=freq.sum();v,g=producer.objective(theta,profiles,freq,.01)
            for index in coords:
                plus=theta.copy();minus=theta.copy();plus[index]+=1e-5;minus[index]-=1e-5
                def value(x):return -float(freq@direct_logq(x,profiles))+.005*float(np.sum(x[d:]**2))
                numeric=(value(plus)-value(minus))/2e-5;errors.append(abs(numeric-g[index]))
            q=producer.logprob_prior(theta,profiles,80)
            agh,variants,meta=producer.reference(theta,profiles);assert meta['qualified']
            close(agh,q,2e-8)
            close(producer.logprob_prior(theta,profiles[:3],80),q[:3])
            if d==6:close(np.exp(q).sum(),1.)
        assert max(errors)<=5e-7
        prior=np.sin(np.arange(3200).reshape(1600,2)*.137)
        score_error=close(producer.score_patterns(theta,profiles[:6],prior),direct_scores(theta,profiles[:6],prior),3e-10)
        payload={'yes':np.bool_(True),'no':np.bool_(False),'count':np.int64(3),'value':np.float64(1.25)}
        converted=json.loads(json.dumps(payload,default=producer.encoder));assert converted==dict(yes=True,no=False,count=3,value=1.25)
        save(OUT/'NUMERICAL_QUALIFICATION.json',dict(passed=True,producer_sha256=sha(producer.__file__),audit_sha256=sha(__file__),
            max_gradient_error=max(errors),MC_JVI_error=score_error,subset_probability_invariance=True,AGH_vs_prior_qualified=True,
            JSON_numpy_scalars_checked=True,seconds=time.perf_counter()-start,
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
            scope='Manufactured profiles only; no Optdigits fit or score pilot'))
        print(json.dumps(dict(passed=True,max_gradient_error=max(errors),score_error=score_error,seconds=time.perf_counter()-start)))
    except Exception:
        save(OUT/'QUALIFICATION_FAILURE.json',dict(traceback=traceback.format_exc(),source_sha256=sha(__file__)));raise

def inputs():
    f=read(OUT/'FREEZE.json')
    for group in ('sources','inputs'):
        for p,h in f[group].items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'NUMERICAL_QUALIFICATION.json')==f['qualification_sha256']

def data():
    train=(np.loadtxt(OUT/'DATA_SOURCE/optdigits.tra',delimiter=',')[:,:64]>=8).astype(float)
    test=(np.loadtxt(OUT/'DATA_SOURCE/optdigits.tes',delimiter=',')[:,:64]>=8).astype(float)
    return train,test

def arithmetic():
    inputs();r=read(OUT/'RESULTS.json');train,test=data();direct_error=0
    for row in r['records']:
        folder=ROOT/row['folder'];tr=read(folder/'TRAINED.json');sel=read(folder/'SELECTION.json')
        assert sha(folder/'TRAINED.json')==row['trained_sha256']==sel['trained_sha256']
        assert sha(folder/'SELECTION.json')==row['selection_sha256']
        assert sha(ROOT/tr['trace_file'])==tr['trace_sha256'] and sha(ROOT/row['evaluation_file'])==row['evaluation_sha256']
        ti=np.array(tr['train_indices']);vi=np.array(tr['validation_indices'])
        assert len(ti)==3058 and len(vi)==765 and np.array_equal(np.sort(np.r_[ti,vi]),np.arange(3823))
        for x,h in ((train[ti],tr['train_sha256']),(train[vi],tr['validation_sha256']),(test,tr['test_sha256'])):
            assert hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()==h
        close(tr['candidates'][0]['p'],(train[ti].sum(axis=0)+.5)/3059)
        with np.load(ROOT/tr['trace_file'],allow_pickle=False) as f:
            for fit in tr['fits']:
                prefix=fit['name']+'_start'+str(fit['start'])
                assert len(f[prefix+'_loss'])==fit['nfev']<=300
                assert np.isfinite(f[prefix+'_x']).all() and np.isfinite(f[prefix+'_gradient']).all()
        for c in tr['candidates'][1:]:
            ends=[x for x in tr['fits'] if x['name']==c['name']];chosen=min(range(2),key=lambda j:(ends[j]['objective'],j))
            assert chosen==c['selected_start'] and c['theta']==ends[chosen]['theta']
        with np.load(ROOT/row['evaluation_file'],allow_pickle=False) as f:
            vp,vc=np.unique(train[vi],axis=0,return_counts=True);tp,tc=np.unique(test,axis=0,return_counts=True)
            close(vp,f['val_profiles'],0);close(tp,f['test_profiles'],0);vf=vc/765;tf=tc/1797
            close(vf,f['val_frequency']);close(tf,f['test_frequency'])
            scores=np.empty((5,4));losses=[]
            for j,name in enumerate(NAMES):
                scores[:4,j]=f['score_'+name]@vf
                vo=sel['validation_references'][j]['main_order'];to=row['test_reference'][j]['main_order']
                scores[4,j]=-f[f'val_{name}_{vo}']@vf;losses.append(float(-f[f'test_{name}_{to}']@tf))
                if row['rep']==0 and j>0:
                    direct_error=max(direct_error,close(direct_scores(tr['candidates'][j]['theta'],vp[:16],f['prior']),f['score_'+name][:,:16],3e-10))
            close(scores,sel['scores']);choices=[int(np.flatnonzero(x<=x.min()+1e-12)[0]) for x in scores]
            assert choices==sel['choices']==row['choices']
            close(losses,row['candidate_test_NLL']);close(np.array(losses)[choices],row['selected_test_NLL'])
            close(np.array(losses)[choices]-min(losses),row['test_relative_best'])
    x=np.array([z['selected_test_NLL'] for z in r['records']]);ind=np.array([z['candidate_test_NLL'][0] for z in r['records']]);crit=float(t.ppf(1-.05/10,31))
    for cell in r['family']:
        values={'RAW100-AGH':x[:,0]-x[:,4],'OUT16-INNER1600':x[:,1]-x[:,2],'INNER1600-AGH':x[:,2]-x[:,4],
                'JVI100-AGH':x[:,3]-x[:,4],'AGH-IND':x[:,4]-ind}[cell['contrast']]
        mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(32));ci=[mean-crit*se,mean+crit*se]
        close(values,cell['paired_values']);close(mean,cell['mean']);close(se,cell['SE']);close(ci,cell['interval'])
        assert cell['material_excess']==(ci[0]>0 and mean>=.05)
        assert cell['below_practical_excess']==(ci[1]<.05)
    return dict(passed=True,source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),workflow_runs=32,candidates=128,
        choices=160,primary_cells=5,max_direct_score_error=direct_error,
        scope='All splits, candidate locks, stored reference/MC choices, test losses and conditional statistics; no retraining')

def alternate(theta,profiles,order):
    d=profiles.shape[1];b=np.array(theta[:d]);W=np.array(theta[d:]).reshape(d,2)
    x,w=np.polynomial.hermite.hermgauss(order);x*=math.sqrt(2);w/=math.sqrt(math.pi)
    u=np.array(list(itertools.product(x,x)));lw=np.log(np.outer(w,w).ravel());a=math.pi/7
    rotation=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    values=[]
    for y in profiles:
        def fun(z):
            eta=b+W@z;p=expit(eta)
            return .5*float(z@z)-float(log_expit((2*y-1)*eta).sum()),z-W.T@(y-p)
        result=minimize(fun,np.zeros(2),jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':100})
        z0=result.x;eta=b+W@z0;p=expit(eta);H=np.eye(2)+(W.T*(p*(1-p)))@W
        S=1.5*np.linalg.inv(H);L=np.linalg.cholesky(S)@rotation;z=z0+u@L.T
        loga=log_expit((2*y-1)*(b+z@W.T)).sum(axis=1)
        logratio=loga-.5*np.sum(z*z,axis=1)+.5*np.sum(u*u,axis=1)+.5*np.linalg.slogdet(S)[1]
        values.append(float(logsumexp(lw+logratio)))
    return np.array(values)

def independent():
    inputs();tick=time.perf_counter();r=read(OUT/'RESULTS.json');train,test=data();records=[]
    save(OUT/'INDEPENDENT_EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    try:
        for row in r['records']:
            if row['rep'] not in (0,8,16,24):continue
            folder=ROOT/row['folder'];tr=read(folder/'TRAINED.json');sel=read(folder/'SELECTION.json')
            val=train[np.sort(tr['validation_indices'])[:64]];target=test[:64]
            with np.load(ROOT/row['evaluation_file'],allow_pickle=False) as f:
                for j,c in enumerate(tr['candidates'][1:],start=1):
                    for label,profiles,meta in (('val',val,sel['validation_references'][j]),('test',target,row['test_reference'][j])):
                        all_profiles=f['val_profiles' if label=='val' else 'test_profiles']
                        lookup={tuple(p):i for i,p in enumerate(all_profiles)};ix=[lookup[tuple(p)] for p in profiles]
                        reference=f[f'{label}_{c["name"]}_{meta["main_order"]}'][ix]
                        low=alternate(c['theta'],profiles,64);high=alternate(c['theta'],profiles,80)
                        error=float(np.max(abs(high-reference)));change=float(np.max(abs(high-low)))
                        records.append(dict(rep=row['rep'],candidate=c['name'],part=label,max_logprob_error=error,order_change=change,qualified=bool(error<=1e-4 and change<=1e-4)))
            print(json.dumps(dict(rep=row['rep'],independently_audited=True)),flush=True)
        save(OUT/'INDEPENDENT_AUDIT.json',dict(completed=True,qualified=all(x['qualified'] for x in records),records=records,
            profile_evaluations=1536,nodes=1536*(64**2+80**2),source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),seconds=time.perf_counter()-tick))
    except Exception:save(OUT/'INDEPENDENT_ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('qualify','arithmetic','check','independent'));cmd=parser.parse_args().command
    if cmd=='qualify':qualify()
    elif cmd=='independent':independent()
    else:
        r=arithmetic()
        if cmd=='arithmetic':save(OUT/'ARITHMETIC_AUDIT.json',r)
        else:assert r==read(OUT/'ARITHMETIC_AUDIT.json')
        print(json.dumps(r))

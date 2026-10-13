"""USE-v2 numerical qualification and independent saved-evidence checks."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime, timezone
from functools import lru_cache
import argparse, hashlib, itertools, json, math, platform, time, traceback
import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/use_selection_v2_20261005'
Y=np.array(list(itertools.product((0,1),repeat=8)));POWERS=2**np.arange(7,-1,-1)
CASES=('NULL','WEAK','STRONG');METHODS=('RAW100','OUT16','INNER1600','JVI100','GH')
NAMES=('IND','L0','L002','L020')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def close(a,b,tol=2e-11):
    error=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert error<=tol,(error,tol);return error

@lru_cache(None)
def nodes(order,rotated):
    x,w=np.polynomial.hermite.hermgauss(order);x*=math.sqrt(2);w/=math.sqrt(math.pi)
    z=np.array(list(itertools.product(x,x)))
    if rotated:
        a=math.pi/7;z=z@np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    return z,np.outer(w,w).ravel()

def direct_pmf(theta,order=20,rotated=False):
    z,w=nodes(order,rotated);theta=np.array(theta);logits=z@theta[8:].reshape(8,2).T+theta[:8]
    q=np.array([math.fsum(w*np.prod(expit((2*y-1)*logits),axis=1)) for y in Y])
    mass=float(q.sum());return q/mass,abs(mass-1)

def direct_objective(theta,freq,penalty):
    q,_=direct_pmf(theta);return -float(np.asarray(freq)@np.log(q))+penalty*float(np.sum(np.array(theta[8:])**2))/2

def numerical_gradient(theta,freq,penalty):
    theta=np.array(theta);values=[]
    for j in range(24):
        plus=theta.copy();minus=theta.copy();plus[j]+=1e-5;minus[j]-=1e-5
        values.append((direct_objective(plus,freq,penalty)-direct_objective(minus,freq,penalty))/2e-5)
    return np.array(values)

def direct_scores(theta,prior):
    theta=np.array(theta);logits=prior@theta[8:].reshape(8,2).T+theta[:8];out=np.empty((4,256))
    for j,y in enumerate(Y):
        a=np.prod(expit((2*y-1)*logits),axis=1);low=-math.log(math.fsum(a[:100])/100)
        out[0,j]=low;out[1,j]=np.mean([-math.log(math.fsum(block)/100) for block in a.reshape(16,100)])
        out[2,j]=-math.log(math.fsum(a)/1600)
        loo=[-math.log(math.fsum(np.delete(a[:100],i))/99) for i in range(100)]
        out[3,j]=100*low-99*float(np.mean(loo))
    return out

def qualify():
    import uselect_compute as producer
    tick=time.perf_counter()
    try:
        theta=np.concatenate((np.linspace(-.4,.4,8),.2*np.cos(np.arange(16))));freq=np.arange(1.,257.);freq/=freq.sum()
        value,g=producer.objective(theta,freq,.002)
        dg=numerical_gradient(theta,freq,.002)
        gradient_error=close(g,dg,5e-7);objective_error=close(value,direct_objective(theta,freq,.002))
        prior=np.sin(np.arange(3200).reshape(1600,2)*.137)
        got=producer.validation_scores(theta,prior);want=direct_scores(theta,prior)
        score_error=close(got,want,2e-11)
        zero=theta.copy();zero[8:]=0
        constant=producer.validation_scores(zero,prior)
        constant_error=close(constant,np.broadcast_to(constant[0],constant.shape),2e-11)
        q=producer.pmf(theta,60);ind,err=direct_pmf(theta,72,True)
        pmf_error=close(q['pmf'],ind,2e-10);assert err<2e-12
        save(OUT/'NUMERICAL_QUALIFICATION.json',dict(passed=True,producer_sha256=sha(producer.__file__),audit_sha256=sha(__file__),
            gradient_max_error=gradient_error,objective_error=objective_error,score_error=score_error,independent_score_error=constant_error,
            rotated_pmf_error=pmf_error,seconds=time.perf_counter()-tick,
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
            scope='Manufactured fixed arrays only; all 24 derivatives, direct scoring and all 100 leave-one-out terms; no scientific teacher pilot'))
        print(json.dumps(dict(passed=True,gradient_error=gradient_error,score_error=score_error,pmf_error=pmf_error,seconds=time.perf_counter()-tick)))
    except Exception:
        number=len(list(OUT.glob('QUALIFICATION_FAILURE_*.json')))
        save(OUT/f'QUALIFICATION_FAILURE_{number:02}.json',dict(traceback=traceback.format_exc(),source_sha256=sha(__file__)))
        raise

def inputs():
    f=read(OUT/'FREEZE.json')
    for group in ('sources','inputs'):
        for name,digest in f[group].items():assert sha(ROOT/name)==digest,name
    assert sha(OUT/'NUMERICAL_QUALIFICATION.json')==f['qualification_sha256']

def arithmetic():
    inputs();result=read(OUT/'RESULTS.json');teachers=read(OUT/'TEACHERS.json')['records'];checked=[];score_error=0.
    for record in result['records']:
        folder=ROOT/record['folder'];trained=read(folder/'TRAINED.json');selection=read(folder/'SELECTION.json')
        assert sha(folder/'TRAINED.json')==record['trained_sha256']==selection['trained_sha256']
        assert sha(folder/'SELECTION.json')==record['selection_sha256']
        assert sha(ROOT/trained['data_file'])==trained['data_sha256'] and sha(ROOT/trained['trace_file'])==trained['trace_sha256']
        assert sha(ROOT/record['evaluation_file'])==record['evaluation_sha256']
        with np.load(ROOT/trained['data_file'],allow_pickle=False) as f:data={key:f[key].astype(np.int64) for key in f.files}
        frequencies={key:np.bincount(value@POWERS,minlength=256)/8192 for key,value in data.items()}
        close(frequencies['train'],trained['train_frequency']);close(frequencies['validation'],selection['validation_frequency']);close(frequencies['test'],record['test_frequency'])
        p=(data['train'].sum(axis=0)+.5)/8193;close(trained['candidates'][0]['p'],p)
        with np.load(ROOT/trained['trace_file'],allow_pickle=False) as tapes:
            for fit in trained['fits']:
                key=fit['name']+'_start'+str(fit['start'])
                assert len(tapes[key+'_loss'])==fit['saved_calls']==fit['nfev']<=450
                assert np.isfinite(tapes[key+'_x']).all() and np.isfinite(tapes[key+'_gradient']).all()
            for candidate in trained['candidates'][1:]:
                candidates=[f for f in trained['fits'] if f['name']==candidate['name']]
                best=min(range(2),key=lambda j:(candidates[j]['objective'],j))
                assert best==candidate['selected_start'] and candidates[best]['theta']==candidate['theta']
        with np.load(ROOT/record['evaluation_file'],allow_pickle=False) as f:patterns=f['pattern_scores'];prior=f['prior']
        assert hashlib.sha256(prior.tobytes()).hexdigest()==selection['validation_prior_sha256']
        score=np.empty((5,4));score[:4]=patterns@frequencies['validation']
        refs=selection['references'];score[4]=[-frequencies['validation']@np.log(r['main_pmf']) for r in refs]
        close(score,selection['score_matrix']);choices=[int(np.flatnonzero(row<=row.min()+1e-12)[0]) for row in score]
        assert choices==record['choices']==selection['choices']
        ci=CASES.index(record['case']);perm=np.array(trained['permutation']);mapping=(Y[:,np.argsort(perm)]@POWERS).astype(int)
        P=np.array(teachers[ci]['main_pmf'])[mapping];close(P,record['teacher_P'])
        risks=[];test=[]
        for r in refs:
            q=np.array(r['main_pmf']);assert q.min()>0;close(q.sum(),1.)
            risks.append(math.fsum(float(a)*(math.log(float(a))-math.log(float(b))) for a,b in zip(P,q)))
            test.append(-float(frequencies['test']@np.log(q)))
        close(risks,record['candidate_risks']);close(test,record['candidate_test_NLL'])
        assert record['oracle_candidate']==int(np.argmin(risks))
        close(np.array(risks)[choices],record['selected_risks']);close(np.array(risks)[choices]-min(risks),record['selection_regret'])
        close(np.array(test)[choices],record['selected_test_NLL'])
        if record['rep']==0:
            for j,c in enumerate(trained['candidates'][1:],start=1):score_error=max(score_error,close(direct_scores(c['theta'],prior),patterns[:,j],3e-11))
        checked.append(record)
    critical=float(t.ppf(1-.05/30,31))
    for row in result['family']:
        records=[r for r in checked if r['case']==row['case']];risk=np.array([r['selected_risks'] for r in records]);ind=np.array([r['candidate_risks'][0] for r in records])
        values={'RAW100-GH':risk[:,0]-risk[:,4],'OUT16-INNER1600':risk[:,1]-risk[:,2],'INNER1600-GH':risk[:,2]-risk[:,4],
                'JVI100-GH':risk[:,3]-risk[:,4],'GH-IND':risk[:,4]-ind}[row['contrast']]
        mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(32));interval=[mean-critical*se,mean+critical*se]
        close(values,row['paired_values']);close(mean,row['mean']);close(se,row['SE']);close(interval,row['interval'])
        assert row['positive_material_loss']==(interval[0]>0 and mean>=.0002)
        assert row['below_practical_excess']==(interval[1]<.0002)
        assert row['useful_joint_gain']==(interval[1]<-.0002)
    return dict(passed=True,source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),streams_checked=96,
        fits_bound=576,candidates_checked=384,selectors_checked=480,primary_cells=15,max_direct_score_error=score_error,
        scope='All data counts, stored training traces/bindings, choices, oracle/test risks and paired statistics; direct MC/JVI scores for rep0 only; no full optimizer replay')

def independent():
    inputs();tick=time.perf_counter();result=read(OUT/'RESULTS.json');rows=[];gradients=[]
    save(OUT/'INDEPENDENT_EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    try:
        for record in result['records']:
            if record['rep'] not in (0,8,16,24):continue
            folder=ROOT/record['folder'];trained=read(folder/'TRAINED.json');selection=read(folder/'SELECTION.json');P=np.array(record['teacher_P'])
            for j,candidate in enumerate(trained['candidates'][1:],start=1):
                q,mass=direct_pmf(candidate['theta'],72,True);base=np.array(selection['references'][j]['main_pmf'])
                error=float(np.max(abs(q-base)));risk=float(P@np.log(P/q));risk_error=abs(risk-record['candidate_risks'][j])
                ok=error<=2e-7 and risk_error<=2e-6 and mass<=2e-12
                rows.append(dict(case=record['case'],rep=record['rep'],candidate=candidate['name'],max_PMF_error=error,
                    risk_error=risk_error,mass_error=mass,qualified=ok,pmf=q.tolist()))
                if record['rep']==0:
                    g=numerical_gradient(candidate['theta'],trained['train_frequency'],candidate['penalty'])
                    fit=next(x for x in trained['fits'] if x['name']==candidate['name'] and x['start']==candidate['selected_start'])
                    ge=float(np.max(abs(g-fit['gradient'])))
                    gradients.append(dict(case=record['case'],candidate=candidate['name'],gradient_error=ge,qualified=ge<=5e-7))
            print(json.dumps(dict(case=record['case'],rep=record['rep'],audited=True)),flush=True)
        save(OUT/'INDEPENDENT_AUDIT.json',dict(completed=True,qualified=all(x['qualified'] for x in rows+gradients),
            references=rows,gradients=gradients,source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),
            reference_nodes=36*72**2,finite_difference_objective_calls=9*24*2,seconds=time.perf_counter()-tick))
    except Exception:
        save(OUT/'INDEPENDENT_ERROR.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('qualify','arithmetic','check','independent'))
    cmd=parser.parse_args().command
    if cmd=='qualify':qualify()
    elif cmd=='independent':independent()
    else:
        result=arithmetic()
        if cmd=='arithmetic':save(OUT/'ARITHMETIC_AUDIT.json',result)
        else:assert result==read(OUT/'ARITHMETIC_AUDIT.json')
        print(json.dumps(result))

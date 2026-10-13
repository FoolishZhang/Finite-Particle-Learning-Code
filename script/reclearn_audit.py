"""REC-v4 independent loss/gradient, Adam arithmetic and GH reference audits."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,platform,time,traceback,itertools
import numpy as np
import scipy
from scipy.special import expit,logsumexp,roots_hermitenorm
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rec_learning_v4_20261005'
PAIRS=tuple(itertools.combinations(range(4),2));Y=np.array(list(itertools.product((0,1),repeat=4)))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def close(a,b,tol=2e-10):
    e=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert e<=tol,(e,tol);return e
def correlation(rho):
    C=np.eye(4)
    for r,(i,j) in zip(rho,PAIRS):C[i,j]=C[j,i]=r
    return C
def direct_loss(v,eps,labels,outer):
    z=v[:4]+(eps@np.linalg.cholesky(correlation(v[8:14])).T)*v[4:8]
    loga=-np.logaddexp(0.,-(2*labels[None]-1)*z).sum(axis=2)
    if outer:return float(np.mean(math.log(100)-logsumexp(loga.reshape(4,100,len(labels)),axis=1)))
    return float(np.mean(math.log(400)-logsumexp(loga,axis=0)))
def finite_gradient(v,eps,labels,outer):
    result=[]
    for j in range(14):
        plus=v.copy();minus=v.copy();plus[j]+=1e-5;minus[j]-=1e-5
        result.append((direct_loss(plus,eps,labels,outer)-direct_loss(minus,eps,labels,outer))/2e-5)
    return np.array(result)
def repair(v):
    x=v.copy();x[:4]=np.clip(x[:4],-2,2);x[4:8]=np.clip(x[4:8],.001,2)
    minimum=float(np.linalg.eigvalsh(correlation(x[8:])).min());alpha=1.
    if minimum<.1:alpha=(1-.1001)/(1-minimum);x[8:]*=alpha
    return x,alpha

def qualify():
    import reclearn_compute as producer
    import torch
    start=time.perf_counter()
    try:
        v=np.array([-.3,-.1,.1,.3,.4,.6,.8,1.,.1,-.07,.05,.07,-.08,.13])
        eps=np.sin(np.arange(400*7*4).reshape(400,7,4)*.037);labels=(np.arange(28).reshape(7,4)%3==0).astype(int)
        errors=[];losses=[]
        for outer,K in ((True,100),(False,400)):
            m=producer.model_from(v,K);value=producer.loss(m,eps,labels,outer);value.backward()
            g=np.concatenate([p.grad.detach().numpy().ravel() for p in m.parameters()])
            errors.append(close(g,finite_gradient(v,eps,labels,outer),2e-7));close(float(value.detach()),direct_loss(v,eps,labels,outer),2e-12)
            losses.append(float(value.detach()))
        assert losses[1]<=losses[0]+1e-12
        edge=np.r_[np.full(4,1.995),np.full(4,1.995),np.full(6,.895)]
        m=producer.model_from(edge,100);opt=producer.kernel.make_optimizer(m,.01)
        for p in m.parameters():p.grad=torch.full_like(p,-1.)
        opt.step();proposal=producer.vector(m);expected=edge+.01/(1+1e-8);close(proposal,expected)
        m.restore_feasibility();after,alpha=repair(expected);close(producer.vector(m),after)
        assert alpha<1 and max(after[:8])<=2
        mu,info=producer.initial_mu(np.array([.3,.45,.55,.7]));x,w=roots_hermitenorm(80);w/=math.sqrt(2*math.pi)
        close([w@expit(m+x) for m in mu],[.3,.45,.55,.7],2e-11)
        save(OUT/'QUALIFICATION.json',dict(passed=True,producer_sha256=sha(producer.__file__),audit_sha256=sha(__file__),
            max_gradient_error=max(errors),same_state_Jensen_gap=losses[0]-losses[1],Adam_and_projection_passed=True,
            initial_observation_marginals_passed=True,seconds=time.perf_counter()-start,
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__),
            scope='Manufactured arrays only; no scientific data/prior or training pilot'))
        print(json.dumps(dict(passed=True,max_gradient_error=max(errors),seconds=time.perf_counter()-start)))
    except Exception:save(OUT/'QUALIFICATION_FAILURE.json',dict(traceback=traceback.format_exc()));raise

def inputs():
    f=read(OUT/'FREEZE.json')
    for group in ('sources','inputs'):
        for p,h in f[group].items():assert sha(ROOT/p)==h,p
    assert sha(OUT/'QUALIFICATION.json')==f['qualification_sha256']

def arithmetic():
    inputs();result=read(OUT/'RESULTS.json');max_update=0.;max_risk=0.
    for record in result['records']:
        path=ROOT/record['trace_file'];assert sha(path)==record['trace_sha256']
        with np.load(path,allow_pickle=False) as f:
            params=f['params'];grads=f['gradient'];m=f['adam_m'];v=f['adam_v'];proposal=f['proposal'];labels=f['labels']
            assert params.shape==(2,513,14) and labels.shape==(8192,4)
            assert labels.sum(axis=0).tolist()==record['data_counts']
            p=(labels.sum(axis=0)+.5)/8193.;close(p,record['independent_p'])
            close(params[0,0],params[1,0],0.)
            close(m[:,1:],.9*m[:,:-1]+.1*grads);close(v[:,1:],.999*v[:,:-1]+.001*grads*grads)
            steps=np.arange(1,513)[None,:,None]
            expected=params[:,:-1]-.01*(m[:,1:]/(1-.9**steps))/(np.sqrt(v[:,1:]/(1-.999**steps))+1e-8)
            max_update=max(max_update,close(expected,proposal))
            for arm in range(2):
                for step in range(512):
                    fixed,alpha=repair(proposal[arm,step]);close(fixed,params[arm,step+1]);close(alpha,f['alpha'][arm,step])
                tail=params[arm,-64:];mu=tail[:,:4].mean(axis=0)
                covariance=np.mean([np.outer(x[4:8],x[4:8])*correlation(x[8:]) for x in tail],axis=0)
                row=record['rows'][arm];close(mu,row['model']['mu']);close(covariance,row['model']['covariance'])
                P=np.array(record['teacher_P']);q=np.array(row['main_pmf'])
                risk=math.fsum(float(a)*(math.log(float(a))-math.log(float(b))) for a,b in zip(P,q))
                max_risk=max(max_risk,close(risk,row['risk']))
            product=np.prod(np.where(Y,p,1-p),axis=1);P=np.array(record['teacher_P'])
            close(float(P@np.log(P/product)),record['independent_risk'])
    critical=float(t.ppf(1-.05/12,31))
    for row in result['family']:
        records=[r for r in result['records'] if r['case']==row['case']]
        values=np.array([r['rows'][1]['risk']-r['rows'][0]['risk'] if row['contrast']=='INNER-OUTER' else
                         r['rows'][1 if row['contrast']=='INNER-IND' else 0]['risk']-r['independent_risk'] for r in records])
        mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(32));ci=[mean-critical*se,mean+critical*se]
        close(values,row['paired_values']);close(mean,row['mean']);close(se,row['SE']);close(ci,row['interval'])
    return dict(passed=True,source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),updates_checked=65536,
        models=128,primary_cells=6,max_update_error=max_update,max_risk_error=max_risk,scope='All saved Adam/projection/TAIL arithmetic and statistics, not full gradient replay')

def independent():
    inputs();from rec1_verify import independent_pmf
    start=time.perf_counter();result=read(OUT/'RESULTS.json');refs=[];gradients=[]
    save(OUT/'INDEPENDENT_EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),automatic_retry=False))
    try:
        for record in result['records']:
            if record['rep'] in (0,16):
                with np.load(ROOT/record['trace_file'],allow_pickle=False) as f:
                    labels=f['labels'].astype(int)
                    for step in (0,255,511):
                        eps=f[f'prior_{step}'];assert hashlib.sha256(eps.tobytes()).hexdigest()==record['prior_hashes'][step]
                        begin=(step*64)%8192
                        for arm in range(2):
                            g=finite_gradient(f['params'][arm,step],eps,labels[begin:begin+64],arm==0)
                            error=float(np.max(abs(g-f['gradient'][arm,step])))
                            gradients.append(dict(case=record['case'],rep=record['rep'],step=step,arm=arm,error=error,qualified=error<=2e-7))
            if record['rep'] in (0,8,16,24):
                row=record['rows'][1];m=row['model'];own=independent_pmf(m['mu'],np.array(m['covariance']),44)
                P=np.array(record['teacher_P']);q=np.array(own['pmf']);risk=float(P@np.log(P/q));error=float(np.max(abs(q-row['main_pmf'])))
                re=abs(risk-row['risk']);ok=error<=2e-7 and re<=2e-6 and own['mass_error']<=2e-12
                refs.append(dict(case=record['case'],rep=record['rep'],max_PMF_error=error,risk_error=re,qualified=bool(ok)))
                print(json.dumps(dict(case=record['case'],rep=record['rep'],audited=True)),flush=True)
        save(OUT/'INDEPENDENT_AUDIT.json',dict(completed=True,qualified=all(x['qualified'] for x in refs+gradients),references=refs,gradients=gradients,
            source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'),reference_nodes=8*44**4,seconds=time.perf_counter()-start))
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

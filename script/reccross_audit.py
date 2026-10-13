"""REC-v5 direct-probability score audit and independently aggregated MC statistics."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,itertools,json,math,platform,time
import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rec_cross_v5_20261005'
Y=np.array(list(itertools.product((0,1),repeat=4)),dtype=int)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def save(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def close(a,b,tol=2e-12):
    error=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert error<=tol,error
    return error

def direct(model,eps):
    z=np.asarray(eps)@np.linalg.cholesky(model['covariance']).T+model['mu']
    plus=expit(z);minus=expit(-z);out=np.empty((len(eps),2,16))
    for j,y in enumerate(Y):
        probabilities=np.prod(np.where(y,plus,minus),axis=2)
        for bank in range(len(eps)):
            out[bank,0,j]=math.fsum(-math.log(math.fsum(float(v) for v in probabilities[bank,k:k+100])/100) for k in range(0,400,100))/4
            out[bank,1,j]=-math.log(math.fsum(float(v) for v in probabilities[bank])/400)
    return out

def qualify():
    import reccross_compute as production
    import jointscale_kernel as kernel
    tick=time.perf_counter()
    mu=np.array([-.4,-.1,.2,.5]);s=np.array([.4,.6,.8,1.]);C=.75*np.eye(4)+.25*np.ones((4,4))
    model=dict(mu=mu.tolist(),covariance=(s[:,None]*C*s[None,:]).tolist())
    eps=np.sin(np.arange(3*400*4).reshape(3,400,4)*.173)
    actual=production.scores(model,eps);manual=direct(model,eps);direct_error=close(actual,manual)
    api_errors=[]
    for bank in range(3):
        for ki,k in enumerate((100,400)):
            m=kernel.ScoreProgram('ZERO',mu,s,K=k,n=1,tau0=0.,path='math64')
            m.load_physical_state(dict(mu=mu.tolist(),s=s.tolist(),rho=[.25]*6,tau=0.))
            values=[]
            for start in (range(0,400,100) if k==100 else [0]):
                e=np.repeat(eps[bank,start:start+k,None,:],16,axis=1)
                batch=kernel.prepare_batch(e,np.zeros((k,16,1)),Y,np.zeros((16,1),dtype=int),path='math64')
                values.append(float(m(batch).detach())-math.log(2))
            api_errors.append(close(actual[bank,ki].mean(),math.fsum(values)/len(values)))
    constant=production.scores(model,np.zeros((2,400,4)))
    constant_error=close(constant[:,0],constant[:,1])
    fixture=np.sin(np.arange(4096)*.13)*.01+.003
    cell=production.mc_cell(fixture);mean=math.fsum(fixture)/4096
    se=math.sqrt(math.fsum((float(v)-mean)**2 for v in fixture)/(4095*4096))
    stat_error=max(close(cell['mean'],mean),close(cell['SE'],se))
    save(OUT/'QUALIFICATION.json',dict(passed=True,producer_sha256=sha(production.__file__),audit_sha256=sha(__file__),
        direct_score_error=direct_error,kernel_score_error=max(api_errors),constant_particle_error=constant_error,
        manufactured_statistics_error=stat_error,seconds=time.perf_counter()-tick,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        scope='Manufactured fixed arrays only; no scientific endpoint scoring or RNG pilot'))
    print(json.dumps(dict(qualified=True,direct_error=direct_error,kernel_error=max(api_errors))))

def arithmetic(public_only=False):
    panel=read(OUT/'INPUT_PANEL.json')['rows'];result=read(OUT/'RESULTS.json');values={};maxerr=0.;referenceerr=0.
    for row,record in zip(panel,result['records']):
        key=(row['case'],row['rep']);assert key==(record['case'],record['rep'])
        source=read(ROOT/row['source']);assert sha(ROOT/row['source'])==row['source_sha256']
        if not public_only:
            assert sha(ROOT/row['trace_file'])==row['trace_sha256']
            with np.load(ROOT/row['trace_file'],allow_pickle=False) as f:labels=f['labels']
            assert ah(labels)==row['labels_sha256']
            counts=np.zeros(16,dtype=int)
            for j,y in enumerate(Y):counts[j]=int(np.sum(np.all(labels==y,axis=1)))
            close(counts,row['counts'],0);close(counts/8192,row['empirical'],0)
        weights=np.stack((row['empirical'],row['teacher_P']),axis=1)
        for j,model in enumerate(row['models']):
            assert model['model']==source['rows'][j]['model']
            assert model['pmf']==source['rows'][j]['main_pmf']
            exact=[-math.fsum(float(w)*math.log(float(q)) for w,q in zip(weights[:,i],model['pmf'])) for i in range(2)]
            referenceerr=max(referenceerr,close(exact,model['exact_losses']))
        if public_only:continue
        assert sha(ROOT/record['score_file'])==record['score_sha256']
        with np.load(ROOT/record['score_file'],allow_pickle=False) as f:
            pattern=f['pattern_scores'];weighted=f['weighted_scores']
        assert pattern.shape==(4096,2,2,16) and weighted.shape==(4096,2,2,2)
        maxerr=max(maxerr,close(np.einsum('baky,yt->bakt',pattern,weights),weighted))
        assert float(np.min(pattern[:,:,0]-pattern[:,:,1]))>=-2e-12
        close(pattern[0],record['first_bank_pattern_scores'],0)
        close(pattern[:,:,0].min()-pattern[:,:,0].min(),0)
        close(float(np.min(pattern[:,:,0]-pattern[:,:,1])),record['minimum_Jensen_gap'],0)
        close(weighted.mean(axis=0),record['means']);delta=weighted[:,1]-weighted[:,0]
        close(delta.mean(axis=0),record['delta_means']);close(delta.std(axis=0,ddof=1)/64,record['delta_SE'])
        values[key]=delta
    if not public_only:
        for cell in result['family']:
            ti=('F_D','F_P').index(cell['target']);case=cell['case']
            data=[]
            for bank in range(4096):
                row=[]
                for rep in range(32):
                    d=values[(case,rep)][bank]
                    row.append(float(d[0,ti] if cell['contrast']=='K100' else d[1,ti] if cell['contrast']=='K400' else d[1,ti]-d[0,ti]))
                data.append(math.fsum(row)/32)
            mean=math.fsum(data)/4096;se=math.sqrt(math.fsum((v-mean)**2 for v in data)/(4095*4096))
            crit=float(t.ppf(1-.05/(2*12),4095));interval=[mean-crit*se,mean+crit*se]
            maxerr=max(maxerr,close(mean,cell['mean']),close(se,cell['SE']),close(interval,cell['interval']))
            assert cell['precision_passed']==(se<=1e-5)
            assert cell['direction']==('INNER' if interval[1]<0 else 'OUTER' if interval[0]>0 else 'unresolved')
    return dict(passed=True,public_only=public_only,streams=64,models=128,full_bank_count=0 if public_only else 262144,
                primary_cells=0 if public_only else 12,max_score_stat_error=maxerr,max_reference_loss_error=referenceerr,
                result_sha256=sha(OUT/'RESULTS.json'),source_sha256=sha(__file__),
                scope='All saved label/score/Jensen/statistic arithmetic, not fresh integration or training')

def independent():
    tick=time.perf_counter();panel=read(OUT/'INPUT_PANEL.json')['rows'];results=read(OUT/'RESULTS.json')['records'];errors=[];records=[]
    for row,record in zip(panel,results):
        if row['rep'] not in (0,8,16,24):continue
        ci=('NULL','MIXED').index(row['case'])
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610055010,ci,row['rep'],0])))
        eps=rng.standard_normal((2,400,4));assert ah(eps)==record['first_two_bank_prior_sha256']
        with np.load(ROOT/record['score_file'],allow_pickle=False) as f:target=f['pattern_scores'][:2]
        for arm,model in enumerate(row['models']):
            error=close(direct(model['model'],eps),target[:,arm]);errors.append(error)
            records.append(dict(case=row['case'],rep=row['rep'],arm=model['arm'],max_score_error=error))
    return dict(passed=True,records=records,scores_checked=1024,max_error=max(errors),replayed_vectors=6400,replayed_Gaussian=25600,
                new_independent_units=0,seconds=time.perf_counter()-tick,
                source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('qualify','arithmetic','independent','check'));cmd=p.parse_args().command
    if cmd=='qualify':qualify()
    elif cmd=='check':
        assert arithmetic()==read(OUT/'ARITHMETIC_AUDIT.json')
        print(json.dumps(dict(passed=True,full_saved_array_arithmetic=True)))
    else:
        result=arithmetic() if cmd=='arithmetic' else independent();save(OUT/('ARITHMETIC_AUDIT.json' if cmd=='arithmetic' else 'INDEPENDENT_AUDIT.json'),result)
        print(json.dumps({k:v for k,v in result.items() if k!='records'}))

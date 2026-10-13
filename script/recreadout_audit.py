"""Independent extraction, score, quadrature and fixed-panel statistics for REC-v6."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,hashlib,json,math,platform,time,itertools
import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import t
import reccross_audit as score_audit

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rec_readout_v6_20261005'
Y=np.array(list(itertools.product((0,1),repeat=4)),dtype=int)
read=score_audit.read;sha=score_audit.sha;ah=score_audit.ah;save=score_audit.save;close=score_audit.close

def covariance(v):
    C=np.eye(4);n=8
    for i in range(4):
        for j in range(i+1,4):C[i,j]=C[j,i]=v[n];n+=1
    return np.diag(v[4:8])@C@np.diag(v[4:8])

def qualify():
    import recreadout_compute as producer
    import jointscale_compute as reference
    import rec1_verify as alternate
    tick=time.perf_counter()
    vectors=np.array([[-.4,-.1,.2,.5,.3,.4,.5,.6]+[0.]*6,[-.3,0.,.1,.4,.6,.7,.8,.9]+[0.]*6])
    last=producer.physical(vectors[-1]);tail=producer.average(vectors)
    extraction=max(close(last['mu'],vectors[-1,:4]),close(last['covariance'],covariance(vectors[-1])),
                   close(tail['mu'],vectors[:,:4].mean(axis=0)),close(tail['covariance'],sum(covariance(v) for v in vectors)/2))
    assert np.max(abs(np.array(tail['covariance'])-covariance(vectors.mean(axis=0))))>.01
    q=reference.pmf(np.array(last['mu']),np.array(last['covariance']),24)
    nodes,weights=np.polynomial.hermite.hermgauss(96);nodes*=math.sqrt(2);weights/=math.sqrt(math.pi)
    p=np.array([weights@expit(last['mu'][j]+math.sqrt(last['covariance'][j][j])*nodes) for j in range(4)])
    exact=np.prod(np.where(Y,p,1-p),axis=1);qerror=close(q['pmf'],exact)
    alternate_q=alternate.independent_pmf(last['mu'],np.array(last['covariance']),28)
    alt_error=close(alternate_q['pmf'],q['pmf'])
    eps=np.cos(np.arange(2*400*4).reshape(2,400,4)*.173);errors=[]
    for model in (last,tail):errors.append(close(producer.cross.scores(model,eps),score_audit.direct(model,eps)))
    fixture=np.cos(np.arange(4096)*.17)*.02+.003;cell=producer.mc_cell(fixture)
    mean=math.fsum(fixture)/4096;se=math.sqrt(math.fsum((float(v)-mean)**2 for v in fixture)/(4095*4096));critical=float(t.ppf(1-.05/48,4095))
    stat_error=max(close(cell['mean'],mean),close(cell['SE'],se),close(cell['interval'],[mean-critical*se,mean+critical*se]))
    save(OUT/'QUALIFICATION.json',dict(passed=True,producer_sha256=sha(producer.__file__),audit_sha256=sha(__file__),
        extraction_error=extraction,diagonal_PMF_error=qerror,alternate_PMF_error=alt_error,score_error=max(errors),statistics_error=stat_error,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),seconds=time.perf_counter()-tick,
        scope='Manufactured tensors only; no saved LAST risks or scientific prior pilot'))
    print(json.dumps(dict(qualified=True,PMF_error=qerror,score_error=max(errors))))

def arithmetic(public_only=False):
    panel=read(OUT/'INPUT_PANEL.json')['rows'];result=read(OUT/'RESULTS.json')
    assert len(panel)==len(result['records'])==64 and len(result['family'])==24
    values={};maxerror=0.;referror=0.;allqualified=True
    for row,record in zip(panel,result['records']):
        key=(row['case'],row['rep']);assert key==(record['case'],record['rep'])
        source=read(ROOT/row['source']);assert sha(ROOT/row['source'])==row['source_sha256']
        if not public_only:
            assert sha(ROOT/row['trace_file'])==row['trace_sha256']
            with np.load(ROOT/row['trace_file'],allow_pickle=False) as f:labels=f['labels'];params=f['params']
            assert ah(labels)==row['labels_sha256']
            counts=np.array([np.sum(np.all(labels==y,axis=1)) for y in Y]);close(counts,row['counts'],0);close(counts/8192,row['empirical'],0)
            for ai in range(2):
                close(params[ai,512],row['models'][0][ai]['physical_vector'],0)
                close(params[ai,-64:,:4].mean(axis=0),row['models'][1][ai]['model']['mu'])
                close(sum(covariance(v) for v in params[ai,-64:])/64,row['models'][1][ai]['model']['covariance'])
        weights=np.stack((row['empirical'],row['teacher_P']),axis=1);stream_qualified=True
        for ri in range(2):
            for ai in range(2):
                m=row['models'][ri][ai];ref=record['references'][ri][ai]
                assert ref['readout']==('LAST','TAIL')[ri] and ref['arm']==('OUTER100x4','INNER400')[ai]
                if ri==0:
                    v=np.array(m['physical_vector']);close(m['model']['mu'],v[:4],0);close(m['model']['covariance'],covariance(v))
                else:assert m['model']==source['rows'][ai]['model'] and ref['pmf']==source['rows'][ai]['main_pmf']
                q=np.array(ref['pmf']);assert len(q)==16 and min(q)>0
                exact=np.array([-math.fsum(float(w)*math.log(float(x)) for w,x in zip(weights[:,ti],q)) for ti in range(2)])
                risk=math.fsum(float(w)*math.log(float(w)/float(x)) for w,x in zip(weights[:,1],q))
                referror=max(referror,close(exact,ref['exact_losses']),close(risk,ref['risk']))
                flags=[]
                for variant,check in zip(ref['variants'],ref['qualification']):
                    qv=np.array(variant['pmf']);pe=float(np.max(abs(qv-q)));le=float(np.max(abs(-np.log(qv)@weights-exact)))
                    re=abs(float(weights[:,1]@np.log(weights[:,1]/qv))-risk)
                    close(pe,check['pmf_error']);close(le,check['loss_error']);close(re,check['risk_error'])
                    passed=pe<=2e-7 and le<=2e-6 and re<=2e-6 and variant['mass_error']<=2e-12 and min(qv)>0
                    assert bool(passed)==check['passed'];flags.append(passed)
                assert ref['qualified']==all(flags);stream_qualified=stream_qualified and all(flags)
        assert record['qualified']==stream_qualified;allqualified=allqualified and stream_qualified
        if public_only:continue
        assert sha(ROOT/record['score_file'])==record['score_sha256']
        with np.load(ROOT/record['score_file'],allow_pickle=False) as f:p=f['pattern_scores'];w=f['weighted_scores']
        assert p.shape==(4096,2,2,2,16) and w.shape==(4096,2,2,2,2)
        maxerror=max(maxerror,close(np.einsum('braky,yt->brakt',p,weights),w))
        assert np.min(p[:,:,:,0]-p[:,:,:,1])>=-2e-12
        close(np.min(p[:,:,:,0]-p[:,:,:,1]),record['minimum_Jensen_gap'],0);close(p[0],record['first_bank_pattern_scores'],0)
        close(w.mean(axis=0),record['means']);delta=w[:,:,1]-w[:,:,0]
        close(delta.mean(axis=0),record['delta_means']);close(delta.std(axis=0,ddof=1)/64,record['delta_SE']);values[key]=delta
    assert result['qualified']==allqualified
    if not public_only:
        for cell in result['family']:
            ti=('F_D','F_P').index(cell['target']);ki=(100,400).index(cell['K']);case=cell['case'];data=[]
            for bank in range(4096):
                terms=[]
                for rep in range(32):
                    v=values[(case,rep)][bank,:,ki,ti]
                    terms.append(float(v[0] if cell['contrast']=='LAST' else v[1] if cell['contrast']=='TAIL' else v[0]-v[1]))
                data.append(math.fsum(terms)/32)
            mean=math.fsum(data)/4096;se=math.sqrt(math.fsum((v-mean)**2 for v in data)/(4095*4096));critical=float(t.ppf(1-.05/48,4095))
            ci=[mean-critical*se,mean+critical*se];maxerror=max(maxerror,close(mean,cell['mean']),close(se,cell['SE']),close(ci,cell['interval']))
            assert cell['precision_passed']==(se<=1e-5)
            assert cell['direction']==('negative' if ci[1]<0 else 'positive' if ci[0]>0 else 'unresolved')
    for exact in result['exact']:
        subset=[r for r in result['records'] if r['case']==exact['case']]
        loss=np.array([[[m['exact_losses'] for m in x] for x in r['references']] for r in subset])
        risk=np.array([[[m['risk'] for m in x] for x in r['references']] for r in subset])
        close(loss.mean(axis=0),exact['mean_losses']);close(risk.mean(axis=0),exact['mean_risks'])
        close((loss[:,:,1]-loss[:,:,0]).mean(axis=0),exact['arm_loss_deltas']);close((risk[:,:,1]-risk[:,:,0]).mean(axis=0),exact['arm_risk_deltas'])
        close((loss[:,0]-loss[:,1]).mean(axis=0),exact['readout_loss_deltas']);close((risk[:,0]-risk[:,1]).mean(axis=0),exact['readout_risk_deltas'])
        close(np.array(exact['arm_loss_deltas'])[:,1],exact['arm_risk_deltas'])
    return dict(passed=True,public_only=public_only,reference_qualified=allqualified,streams=64,readout_models=256,
        full_banks_checked=0 if public_only else 262144,primary_cells_checked=0 if public_only else 24,max_score_stat_error=maxerror,
        max_reference_loss_error=referror,source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'))

def independent():
    import rec1_verify as alternate
    tick=time.perf_counter();panel=read(OUT/'INPUT_PANEL.json')['rows'];result=read(OUT/'RESULTS.json');scores=[];references=[]
    for row,record in zip(panel,result['records']):
        if row['rep'] not in (0,8,16,24):continue
        ci=('NULL','MIXED').index(row['case']);rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610056010,ci,row['rep'],0])))
        eps=rng.standard_normal((2,400,4));assert ah(eps)==record['first_two_bank_prior_sha256']
        with np.load(ROOT/record['score_file'],allow_pickle=False) as f:target=f['pattern_scores'][:2]
        for ri in range(2):
            for ai in range(2):
                m=row['models'][ri][ai];error=close(score_audit.direct(m['model'],eps),target[:,ri,ai])
                scores.append(dict(case=row['case'],rep=row['rep'],readout=m['readout'],arm=m['arm'],error=error))
                if ri==0 and row['rep'] in (0,16):
                    ref=record['references'][0][ai];alternate_q=alternate.independent_pmf(m['model']['mu'],np.array(m['model']['covariance']),44)
                    q=np.array(alternate_q['pmf']);P=np.array(row['teacher_P']);T=np.array(row['empirical'])
                    pe=float(np.max(abs(q-ref['pmf'])));le=float(np.max(abs(np.array([-T@np.log(q),-P@np.log(q)])-ref['exact_losses'])))
                    re=abs(float(P@np.log(P/q))-ref['risk']);passed=pe<=2e-7 and le<=2e-6 and re<=2e-6 and alternate_q['mass_error']<=2e-12
                    references.append(dict(case=row['case'],rep=row['rep'],arm=m['arm'],pmf_error=pe,loss_error=le,risk_error=re,passed=bool(passed)))
    return dict(completed=True,passed=all(r['passed'] for r in references),scores_checked=2048,score_records=scores,
        max_score_error=max(x['error'] for x in scores),reference_records=references,reference_nodes=29984768,
        replay_vectors=6400,replay_Gaussian=25600,new_independent_MC_units=0,seconds=time.perf_counter()-tick,
        source_sha256=sha(__file__),result_sha256=sha(OUT/'RESULTS.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('qualify','arithmetic','independent','check'));cmd=p.parse_args().command
    if cmd=='qualify':qualify()
    elif cmd=='check':
        assert arithmetic()==read(OUT/'ARITHMETIC_AUDIT.json');print('{"passed":true,"full_arithmetic":true}')
    else:
        r=arithmetic() if cmd=='arithmetic' else independent();save(OUT/('ARITHMETIC_AUDIT.json' if cmd=='arithmetic' else 'INDEPENDENT_AUDIT.json'),r)
        print(json.dumps({k:v for k,v in r.items() if not k.endswith('_records')}))

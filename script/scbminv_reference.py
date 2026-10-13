"""Frozen deterministic four-dimensional observation reference, no RNG."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,itertools,time
import numpy as np
from scipy.special import roots_hermitenorm,expit
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_invariance/REFERENCE'
DOC=ROOT/'docs/08_research_研究工作/真实联合评分接口与无关扩展_20261003'
MU=np.array([-.6,-.2,.2,.6])
COV=np.array([[1.,.25,-.15,.10],[.25,1.,.10,-.15],[-.15,.10,1.,.25],[.10,-.15,.25,1.]])
TS=[-.75,0.,.75]
CFG=dict(mean=MU.tolist(),covariance=COV.tolist(),candidates=TS,orders=[24,32,40],main_order=32,rotated_order=32,
         maximum_PMF_change=5e-8,normalization_tolerance=2e-12,minimum_eigenvalue=.12,chunk=65536)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pmf(cov,order,rotation):
    nodes,weights=roots_hermitenorm(order);weights=weights/np.sqrt(2*np.pi)
    ids=np.indices((order,)*4).reshape(4,-1).T
    signs=np.array(list(itertools.product([0,1],repeat=4)),dtype=int)
    if rotation=='cholesky':root=np.linalg.cholesky(cov)
    else:
        eig,vec=np.linalg.eigh(cov);root=vec@np.diag(np.sqrt(eig))
    total=np.zeros(16,dtype=np.longdouble)
    for start in range(0,len(ids),CFG['chunk']):
        sub=ids[start:start+CFG['chunk']];x=nodes[sub]@root.T+MU;w=np.prod(weights[sub],axis=1)
        plus,minus=expit(x),expit(-x)
        for j,y in enumerate(signs):
            value=np.prod(np.where(y,plus,minus),axis=1)*w
            total[j]+=np.sum(value,dtype=np.longdouble)
    return dict(order=order,rotation=rotation,pmf=np.asarray(total,dtype=float).tolist(),gaussian_nodes=order**4,
                covariance=cov.tolist(),eigenvalues=np.linalg.eigvalsh(cov).tolist())
def run():
    f=read(OUT/'FREEZE.json');assert f['source_sha256']==sha(__file__) and f['config']==CFG and f['integrals_authorized']
    assert f['plan_sha256']==sha(ROOT/f['plan_path'])
    assert not (OUT/'EXECUTION.json').exists();save(OUT/'EXECUTION.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json')))
    started=time.perf_counter();records=[];qual=[]
    for t in TS:
        cov=np.eye(4)+(1+t)*(COV-np.eye(4));assert np.linalg.eigvalsh(cov).min()>CFG['minimum_eigenvalue']
        rows=[pmf(cov,o,'cholesky') for o in CFG['orders']]+[pmf(cov,CFG['rotated_order'],'eigen')]
        main=np.asarray(next(r['pmf'] for r in rows if r['order']==32 and r['rotation']=='cholesky'))
        for row in rows:
            p=np.asarray(row['pmf']);delta=float(np.max(abs(p-main)));norm=abs(float(p.sum())-1.)
            qual.append(dict(t=t,order=row['order'],rotation=row['rotation'],max_change=delta,normalization_error=norm,
                             passed=bool(delta<=CFG['maximum_PMF_change'] and norm<=CFG['normalization_tolerance'] and np.all(p>0))))
        records.append(dict(t=t,main_pmf=main.tolist(),variants=rows))
    truth=np.asarray(next(r['main_pmf'] for r in records if r['t']==0.))
    for row in records:
        p=np.asarray(row['main_pmf']);row['reference_joint_KL']=float(np.sum(truth*np.log(truth/p)))
        row['exact_cross_entropy_reference']=float(-np.sum(truth*np.log(p)))
    result=dict(config=CFG,source_sha256=sha(__file__),freeze_sha256=sha(OUT/'FREEZE.json'),
        patterns=list(map(list,itertools.product([0,1],repeat=4))),records=records,qualifications=qual,
        qualification_passed=all(q['passed'] for q in qual),accounting=dict(PMF_integrations=12,
        scalar_pattern_integrals=12*16,Gaussian_nodes=sum(x['gaussian_nodes'] for r in records for x in r['variants']),
        new_random_draws=0,new_fits=0),seconds=time.perf_counter()-started,
        scope='Numerical quadrature qualification only, not a rigorous full-integration remainder certificate; exact-task references independent of nuisance padding.')
    save(OUT/'MODEL.json',result);print(json.dumps(dict(passed=result['qualification_passed'],risks=[dict(t=r['t'],KL=r['reference_joint_KL']) for r in records],accounting=result['accounting'],seconds=result['seconds'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['schema','run']);a=p.parse_args()
    if a.mode=='schema':print(json.dumps(dict(source_sha256=sha(__file__),config=CFG)))
    else:run()

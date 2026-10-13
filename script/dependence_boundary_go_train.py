"""Frozen GoEmotions iid/antithetic covariance learning, with no test interface."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, itertools, json, os, time, traceback
import numpy as np
import torch
import torch.nn.functional as F
from script.shared_uncertainty_holdout_train import source_loss, SRC

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/dependence_boundary_v1/GN'
GDATA=ROOT/'artifacts/dependence_boundary_v1/GDATA'
PLAN=ROOT/'docs/08_research_研究工作/边际依赖接口_20260929/PLAN_GN.md'
SEEDS=[20269701,20269702,20269703]
CONFIG=dict(heads=['word','word_char'],roles=['train','association_fit'],K=[10,100],samplers=['iid','antithetic'],
            seeds=SEEDS,d=27,batch=128,initial_sd=1.,phases=[1500,500],threads=2,models=48,
            rater=0,factor_floor=1e-6,mean_scaling=1.,reg_precision=None)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj,exclusive=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def factor(raw):
    return torch.tril(raw,-1)+torch.diag_embed(F.softplus(torch.diagonal(raw,dim1=-2,dim2=-1))+1e-6)
def cells():
    for head,role,K in itertools.product(CONFIG['heads'],CONFIG['roles'],CONFIG['K']):yield dict(head=head,role=role,K=K)
def ident(c):return f"{c['head']}_{c['role']}_K{c['K']}"
def check_seal(p):
    for name,h in read(p)['files'].items():assert sha(ROOT/name)==h,name
def freeze():
    assert not OUT.exists();check_seal(GDATA/'DATA_SEAL.json');check_seal(GDATA/'PREDICTION_SEAL.json')
    paths=[Path(__file__),PLAN,ROOT/'script/shared_uncertainty_holdout_train.py',ROOT/'script/shared_uncertainty_core.py',SRC/'models/losses.py',
           GDATA/'FREEZE.json',GDATA/'COMPLETE.json',GDATA/'DATA_SEAL.json',GDATA/'PREDICTION_SEAL.json']
    paths += [GDATA/'cache'/f'{ro}.npz' for ro in CONFIG['roles']]
    paths += [GDATA/'cache'/f'{h}_{ro}_means.npz' for h in CONFIG['heads'] for ro in CONFIG['roles']]
    write(OUT/'FREEZE.json',dict(time_utc=now(),config=CONFIG,files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},test_opened=False))
def check_freeze():
    assert read(OUT/'FREEZE.json')['config']==CONFIG;check_seal(OUT/'FREEZE.json')
def snapshot(raw):
    L=factor(raw.detach()).numpy().astype(float);return dict(raw=raw.detach().tolist(),factors=L.tolist(),covariance=(L@L.transpose(0,2,1)).tolist())
def train(c):
    name=ident(c);assert not (OUT/'initial'/f'{name}.json').exists()
    with np.load(GDATA/'cache'/f"{c['head']}_{c['role']}_means.npz") as z:mu=torch.tensor(z['means'],dtype=torch.float32);ids=z['ids']
    with np.load(GDATA/'cache'/f"{c['role']}.npz") as z:assert np.array_equal(z['id'],ids);Y=torch.tensor(z['labels'][:,0],dtype=torch.float32)
    assert mu.shape==Y.shape and mu.shape[1]==27
    raw=torch.zeros(6,27,27);raw[:,torch.arange(27),torch.arange(27)]=float(np.log(np.expm1(1-1e-6)));raw=torch.nn.Parameter(raw)
    write(OUT/'initial'/f'{name}.json',dict(config=c,**snapshot(raw)))
    lossfun=source_loss();trace=[];states=[];start=time.perf_counter();K=c['K']
    for phase,length in enumerate([1500,500]):
        opt=torch.optim.Adam([raw],lr=.002 if phase==0 else .0005)
        rng=[np.random.default_rng(s+(2000 if phase==0 else 3000)) for s in SEEDS*2]
        for step in range(1,length+1):
            ix=[];noise=[]
            for j,r in enumerate(rng):
                ix.append(r.integers(0,len(mu),128));z=r.normal(size=(128,K,27));r.integers(0,27)
                if j>=3:z=np.concatenate([z[:,:K//2],-z[:,:K//2]],axis=1)
                noise.append(z)
            index=torch.tensor(np.array(ix));z=torch.tensor(np.array(noise),dtype=torch.float32);m=mu[index];y=Y[index];L=factor(raw)
            eta=m[:,:,None]+torch.einsum('sbkj,sij->sbki',z,L);probs=torch.sigmoid(eta).permute(0,1,3,2)
            values=lossfun(probs.reshape(-1,27,K),y.reshape(-1,27),torch.zeros(768,1),torch.zeros(768),L);loss=values[-1]*6
            opt.zero_grad();loss.backward();assert torch.isfinite(loss) and torch.isfinite(raw.grad).all()
            opt.param_groups[0]['lr']=.002 if phase==0 and step<=1000 else .0005;opt.step();assert torch.isfinite(raw).all()
            if step%250==0:
                with torch.no_grad():
                    bce=F.binary_cross_entropy(probs,y[...,None].expand_as(probs),reduction='none')
                    nll=-torch.logsumexp(-bce.sum(2),dim=2).mean(1)+np.log(K)
                    tr=(factor(raw.detach())**2).sum((1,2))
                trace.append(dict(phase=phase,step=step,total_step=phase*1500+step,pre_update_concept_NLL=nll.tolist(),
                                  post_update_tracevariance=tr.tolist(),sigmoid_zero=(probs==0).sum((1,2,3)).tolist(),
                                  sigmoid_one=(probs==1).sum((1,2,3)).tolist(),probabilities_per_rep=128*27*K))
                print(name,phase,step,round(time.perf_counter()-start,1),'trace',round(float(tr.mean()),4),flush=True)
        state=[hashlib.sha256(json.dumps(r.bit_generator.state,sort_keys=True).encode()).hexdigest() for r in rng]
        assert state[:3]==state[3:];states.append(state)
        write(OUT/'checkpoints'/f'{name}_phase{phase}.json',dict(config=c,phase=phase,rng_states=state,**snapshot(raw)))
    write(OUT/'models'/f'{name}.json',dict(time_utc=now(),config=c,seeds=SEEDS*2,samplers=['iid']*3+['antithetic']*3,
          replicates=[0,1,2]*2,steps=2000,phases=2,trace=trace,rng_states=states,elapsed_seconds=time.perf_counter()-start,
          finite_loss_gradient_raw_every_step=True,**snapshot(raw)))
def audit():
    rows=[];traces=[]
    for c in cells():
        model=read(OUT/'models'/f'{ident(c)}.json');assert model['config']==c and len(model['trace'])==8 and model['steps']==2000
        raw=np.array(model['raw']);L=np.array(model['factors']);derived=np.tril(raw,-1)
        derived[:,np.arange(27),np.arange(27)]=np.logaddexp(0,np.diagonal(raw,axis1=1,axis2=2))+1e-6
        assert np.max(abs(L-derived))<5e-7
        C=L@L.transpose(0,2,1);assert np.allclose(C,model['covariance'],atol=1e-13,rtol=1e-13)
        initial=read(OUT/'initial'/f'{ident(c)}.json');assert np.allclose(initial['covariance'],np.broadcast_to(np.eye(27),(6,27,27)),rtol=2e-6,atol=1e-12)
        assert all(s[:3]==s[3:] for s in model['rng_states'])
        assert read(OUT/'checkpoints'/f'{ident(c)}_phase1.json')['raw']==model['raw']
        for j in range(6):
            ev=np.linalg.eigvalsh(C[j]);assert ev[0]>=-1e-10*max(1.,ev[-1]);assert np.all(np.diag(L[j])>0)
            rows.append(dict(**c,sampler=model['samplers'][j],rep=j%3,seed=SEEDS[j%3],tracevariance=float(np.trace(C[j])),
                             eigen_min=float(ev[0]),eigen_max=float(ev[-1]),factor_reconstruction_error=float(np.max(abs(L[j]-derived[j]))),
                             offdiagonal_norm=float(np.linalg.norm(C[j]-np.diag(np.diag(C[j]))))))
            for t in model['trace']:traces.append(dict(**c,sampler=model['samplers'][j],rep=j%3,phase=t['phase'],step=t['step'],
                                                     NLL=t['pre_update_concept_NLL'][j],tracevariance=t['post_update_tracevariance'][j],
                                                     sigmoid_zero=t['sigmoid_zero'][j],sigmoid_one=t['sigmoid_one'][j]))
    assert len(rows)==48 and len(traces)==384;csvwrite(OUT/'MODEL_AUDIT.csv',rows);csvwrite(OUT/'TRAJECTORIES.csv',traces)
    write(OUT/'AUDIT.json',dict(time_utc=now(),passed=True,models=48,matched_sampler_rng_states=True,
          factor_max_error=max(r['factor_reconstruction_error'] for r in rows),test_accessed=False,new_mean_fits=0))
def run():
    check_freeze();torch.set_num_threads(2)
    for c in cells():
        if (OUT/'models'/f'{ident(c)}.json').exists():continue
        write(OUT/'RUN_STATE.json',dict(status='running',pid=os.getpid(),cell=ident(c),time_utc=now()),exclusive=False);train(c)
    audit();check_freeze();models=sorted((OUT/'models').glob('*.json'));assert len(models)==8
    write(OUT/'MODELS_FREEZE.json',dict(time_utc=now(),models=48,files={p.relative_to(ROOT).as_posix():sha(p) for p in models},test_opened=False))
    write(OUT/'RUN_STATE.json',dict(status='complete_models_sealed',time_utc=now(),active_jobs=[]),exclusive=False)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run','check']);args=parser.parse_args()
    try:
        if args.phase=='check':check_freeze();check_seal(OUT/'MODELS_FREEZE.json');print('pass_GN_frozen_48_models')
        else:globals()[args.phase]()
    except Exception:
        if OUT.exists():write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=now(),traceback=traceback.format_exc()))
        raise

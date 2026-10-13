"""Matched K100 nuisance-weight intervention; frozen, bounded scientific run."""
from pathlib import Path
from datetime import datetime,timezone
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,hashlib,json,math,time,traceback
import numpy as np
from scipy.special import logsumexp,expit
import finite_selection_compute as old
from flint import arb,ctx
ctx.prec=128
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nuisance_contrast_v1_20261006'
DOC=ROOT/'docs/08_research_研究工作/NuisanceContrast_v1_20261006'
K=100; B=16384; L=262144; BLOCK=256; R0=1e-6
YS=old.YS;PAIRS=old.PAIRS;CB=old.CB
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def native(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [native(v) for v in x]
 return x
def save(p,o):Path(p).write_text(json.dumps(native(o),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def now():return datetime.now(timezone.utc).isoformat()
def rng(role):return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610060201,role])))
def logweights(s):
 c=s-np.max(s,axis=1)[:,None]
 return c-logsumexp(c,axis=1)[:,None]
def score_pair(c,z,s,q,gradient=False):
 x=z@old.root(c);base=-np.logaddexp(x/2,-x/2).sum(axis=2)
 ws=[np.full(s.shape,-math.log(s.shape[1])),logweights(s)]
 totals=np.zeros((len(z),2));grads=np.zeros((len(z),2,3))
 for y,p in zip(YS,q):
  logs=base+sum(float(y[j])*.5*x[:,:,j] for j in range(3))
  for ni,w in enumerate(ws):
   logits=w+logs;norm=logsumexp(logits,axis=1);totals[:,ni]-=p*norm
   if gradient:
    assert np.array_equal(c,np.eye(3))
    pi=np.exp(logits-norm[:,None]);gx=y*expit(-x*y)
    for ci,(i,j) in enumerate(PAIRS):grads[:,ni,ci]-=p*np.sum(pi*(gx[:,:,i]*z[:,:,j]+gx[:,:,j]*z[:,:,i])/2,axis=1)
 return (totals,grads) if gradient else totals
def direct(c,z,s,q):
 x=z@old.root(c);out=np.zeros((len(z),2))
 for y,p in zip(YS,q):
  logs=-np.logaddexp(0,-x*y).sum(axis=2)
  for ni,w in enumerate([np.full(s.shape,-math.log(s.shape[1])),logweights(s)]):out[:,ni]-=p*logsumexp(w+logs,axis=1)
 return out
def bounds(U):
 beta=(-arb(1)/2).exp()/4;d=(12*arb('0.000001')).sqrt()/beta;lam=1-d
 h=(arb(13)/4)*U/(4*lam)+(3*U).sqrt()/(4*lam**arb('1.5'))
 return d,h,h*d*d/2
def population_U():return arb(8)/3*(arb(K).log()+3*arb(2).log())
def interval_pair(t,z,s,counts,gradient=False,deadline=None):
 sr=old.root(old.matrix(t));sr=(sr+sr.T)/2;ar,rooterr=old.root_error(t,sr)
 kk=z.shape[1];qq=[arb(int(x))/int(sum(counts)) for x in counts]
 totals=[arb(0),arb(0)];grads=[[arb(0) for _ in PAIRS] for _ in range(2)];maxnorm2=arb(0)
 for bi,(zz,ss) in enumerate(zip(z,s)):
  az=[[arb(float(v)) for v in row] for row in zz]
  upper=max(old.rounded_upper(sum(v*v for v in row)) for row in az);maxnorm2+=arb(upper)
  x=[[sum(ar[i][j]*az[k][j] for j in range(3)) for i in range(3)] for k in range(kk)]
  plus=[[1/(1+(-v).exp()) for v in row] for row in x]
  mx=arb(float(np.max(ss)));weight=[(arb(float(v))-mx).exp() for v in ss]
  lognorm=[arb(kk).log(),sum(weight).log()]
  for yi,y in enumerate(YS):
   prob=[[plus[k][j] if y[j]>0 else 1-plus[k][j] for j in range(3)] for k in range(kk)]
   lk=[p[0]*p[1]*p[2] for p in prob];mix=[sum(lk),sum(w*v for w,v in zip(weight,lk))]
   for ni in range(2):totals[ni]+=qq[yi]*(lognorm[ni]-mix[ni].log())
   if gradient:
    assert t==(0,0,0)
    for ci,(i,j) in enumerate(PAIRS):
     deriv=[(int(y[i])*(1-prob[k][i])*az[k][j]+int(y[j])*(1-prob[k][j])*az[k][i])/2 for k in range(kk)]
     numer=[sum(v*g for v,g in zip(lk,deriv)),sum(w*v*g for w,v,g in zip(weight,lk,deriv))]
     for ni in range(2):grads[ni][ci]-=qq[yi]*numer[ni]/mix[ni]
  if (bi+1)%512==0:
   assert deadline is None or time.monotonic()<deadline,'Interval deadline exceeded'
   print(json.dumps(dict(stage='interval',candidate=t,banks=bi+1,total=len(z))),flush=True)
 U=maxnorm2/len(z);allowance=(3*U).sqrt()*rooterr
 values=[a/len(z)+arb(0,old.rounded_upper(allowance)) for a in totals]
 return values,[[g/len(z) for g in row] for row in grads],U,allowance
def qualify():
 errors=[];flat=[];fd=[];checks=[]
 for kk in (2,5,100):
  z=np.sin(np.arange(6*kk*3)).reshape(6,kk,3);s=np.linspace(-7,3,6*kk).reshape(6,kk)
  counts=np.arange(1,9);q=counts/counts.sum();a,g=score_pair(np.eye(3),z,s,q,True)
  errors.append(float(np.max(np.abs(a-direct(np.eye(3),z,s,q)))))
  v,ig,u,_=interval_pair((0,0,0),z,s,counts,True)
  errors.extend(abs(float(v[j])-a[:,j].mean()) for j in range(2))
  errors.extend(abs(float(ig[j][i])-g[:,j,i].mean()) for j in range(2) for i in range(3))
  for i,j in PAIRS:
   h=np.zeros((3,3));h[i,j]=h[j,i]=1e-5
   diff=(score_pair(np.eye(3)+h,z,s,q)-score_pair(np.eye(3)-h,z,s,q))/2e-5
   fd.append(float(np.max(np.abs(diff.mean(axis=0)-g[:,:,PAIRS.index((i,j))].mean(axis=0)))))
  const=np.repeat(np.array([-1000.,1000.,.3,-.4,0.,55.])[:,None],kk,axis=1)
  ans=score_pair(CB,z,const,q);flat.append(float(np.max(np.abs(ans[:,0]-ans[:,1]))))
  d,h,cost=bounds(u);checks.append(float(h))
 assert max(errors)<1e-12 and max(fd)<1e-8 and max(flat)<1e-12
 save(OUT/'QUALIFICATION.json',dict(created_utc=now(),manufactured_only=True,max_score_gradient_error=max(errors),max_gradient_fd_error=max(fd),constant_weight_identity_error=max(flat),K_cases=[2,5,100],population_norm_bound=old.ball(population_U()),population_region_cost=old.ball(bounds(population_U())[2]),numpy_scalar_serialization_test=np.bool_(True),passed=True))
 print(json.dumps(read(OUT/'QUALIFICATION.json')),flush=True)
def freeze():
 assert not (OUT/'FREEZE.json').exists() and not (OUT/'TRAIN_BANKS.npz').exists()
 names=[Path(__file__),Path(old.__file__),DOC/'CONTRACT.md',OUT/'QUALIFICATION.json',OUT/'HANDOFF_INPUT.md',ROOT/'artifacts/finite_selection_v1_20261006/TRAIN_BANKS.npz',ROOT/'artifacts/finite_selection_v1_20261006/DATA_COUNTS.json']
 save(OUT/'FREEZE.json',dict(created_utc=now(),baseline='a97896f243d257537a862479fe1e2e46bcca8bbd',sources={p.relative_to(ROOT).as_posix():sha(p) for p in names},K=K,B=B,L=L,block=BLOCK,r0=R0,alpha=.01,no_scientific_prior_before_freeze=True,data_reused=True))
def run():
 frozen=read(OUT/'FREEZE.json')
 for p,h in frozen['sources'].items():assert sha(ROOT/p)==h,p
 assert not (OUT/'TRAIN_BANKS.npz').exists(),'No silent repeat'
 started=time.monotonic();save(OUT/'RUN_START.json',dict(utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')))
 previous=np.load(ROOT/'artifacts/finite_selection_v1_20261006/TRAIN_BANKS.npz');counts=previous['counts'];q=counts/counts.sum()
 z=rng(1).standard_normal((B,K,3));s=np.empty((B,K));sr=rng(2)
 for lo in range(0,B,BLOCK):s[lo:lo+BLOCK]=-np.logaddexp(0,-sr.standard_normal((BLOCK,K,256))).sum(axis=2)
 np.savez(OUT/'TRAIN_BANKS.npz',z=z,s=s,counts=counts)
 candidates=old.grid();values=[]
 for ci,t in enumerate(candidates):
  val=score_pair(old.matrix(t),z,s,q).mean(axis=0);values.append(val)
  if (ci+1)%20==0:print(json.dumps(dict(stage='grid',candidates=ci+1,total=len(candidates))),flush=True)
  assert time.monotonic()-started<3600,'Main budget exceeded'
 values=np.array(values);selected=[]
 for ni,n in enumerate((0,256)):
  ix=int(np.flatnonzero(values[:,ni]<=values[:,ni].min()+1e-12)[0]);selected.append(dict(n=n,index=ix,candidate=list(candidates[ix]),score=float(values[ix,ni])))
 np.savez(OUT/'GRID.npz',candidates=np.array(candidates),scores=values)
 save(OUT/'SELECTED_OUTPUTS.json',dict(utc=now(),selected=selected,risk_used=False,fallback_rule=False))
 save(OUT/'PUBLIC_GRID.json',dict(candidates=candidates,scores=values,condition_order=[0,256]))
 print(json.dumps(dict(stage='selected',selected=selected)),flush=True)
 ez=np.empty((L,K,3));es=np.empty((L,K));diff=np.empty((L,2));zr,gr=rng(3),rng(4)
 for lo in range(0,L,BLOCK):
  zz=zr.standard_normal((BLOCK,K,3));ss=-np.logaddexp(0,-gr.standard_normal((BLOCK,K,256))).sum(axis=2)
  ez[lo:lo+BLOCK]=zz;es[lo:lo+BLOCK]=ss
  diff[lo:lo+BLOCK]=score_pair(np.eye(3),zz,ss,np.ones(8)/8)-score_pair(CB,zz,ss,np.ones(8)/8)
  if (lo+BLOCK)%(BLOCK*32)==0:print(json.dumps(dict(stage='population',banks=lo+BLOCK,total=L)),flush=True)
  assert time.monotonic()-started<3600,'Main budget exceeded'
 np.savez(OUT/'EVALUATION_BANKS.npz',z=ez,s=es,difference=diff)
 trunc=diff.copy();outside=np.max(np.abs(ez),axis=(1,2))>8;trunc[outside]=0
 A=max(1-math.sqrt(.1),math.sqrt(2.8)-1);bound=3*A*8;u=float(population_U());tau=math.sqrt(3)*A*math.sqrt(u*2*K*3*math.exp(-32)/math.sqrt(2*math.pi)/8)
 pop=[];log=math.log(4/(.01/3));cost=bounds(population_U())[2]
 for name,arr,factor in [('g0',trunc[:,0],1),('g256',trunc[:,1],1),('interaction',trunc[:,1]-trunc[:,0],2)]:
  v=float(arr.var(ddof=1));half=math.sqrt(2*v*log/L)+7*(2*bound*factor)*log/(3*(L-1))+factor*(tau+1e-9);mean=float(arr.mean())
  pop.append(dict(contrast=name,mean=mean,variance=v,interval=[mean-half,mean+half],halfwidth=half,range=2*bound*factor,tail=factor*tau,numerical_allowance=factor*1e-9))
 save(OUT/'POPULATION.json',dict(alpha=.01,point_alpha=.01/3,banks=L,results=pop,population_norm_bound=old.ball(population_U()),region_cost=old.ball(cost),n256_region_lower=pop[1]['interval'][0]-old.rounded_upper(cost),truncated_banks=int(outside.sum())))
 # Audit both branches with shared active arithmetic, then test each returned
 # candidate without changing selection on the basis of the audit.
 deadline=time.monotonic()+2400;audited={}
 todo=[(0,0,0),(9,9,9)]+[tuple(x['candidate']) for x in selected]
 for t in dict.fromkeys(todo):
  v,g,U,e=interval_pair(t,z,s,counts,gradient=(t==(0,0,0)),deadline=deadline)
  audited[t]=(v,g,U,e)
  save(OUT/'INTERVAL_PROGRESS.json',dict(completed=[dict(candidate=t,scores=[old.ball(x) for x in v],gradient=[[old.ball(x) for x in row] for row in g],norm_upper=old.ball(U),root_allowance=old.ball(e)) for t,(v,g,U,e) in audited.items()]))
 vI,g,U,_=audited[(0,0,0)];vB=audited[(9,9,9)][0];d,h,quad=bounds(U)
 empirical=[]
 for ni,n in enumerate((0,256)):
  t=tuple(selected[ni]['candidate']);vs=audited[t][0][ni];linear=(sum(x*x for x in g[ni])/2).sqrt()*d
  region=vI[ni]-vB[ni]-linear-quad
  at_reference=t==(9,9,9);reference_comparison=at_reference or bool(vB[ni]-vs>=0)
  empirical.append(dict(n=n,selected=list(t),score_I=old.ball(vI[ni]),score_Cb=old.ball(vB[ni]),score_selected=old.ball(vs),point_gap=old.ball(vI[ni]-vB[ni]),gradient_I=[old.ball(x) for x in g[ni]],norm_upper=old.ball(U),H=old.ball(h),linear_cost=old.ball(linear),quadratic_cost=old.ball(quad),region_margin=old.ball(region),region_excluded=bool(region>0 and reference_comparison),selected_not_worse_than_Cb=reference_comparison))
 save(OUT/'EMPIRICAL_CERTIFICATES.json',dict(results=empirical,r0=R0))
 ids=np.linspace(0,L-1,32,dtype=int);errs=[]
 for ix in ids:
  ai,_,_,_=interval_pair((0,0,0),ez[ix:ix+1],es[ix:ix+1],np.ones(8,dtype=int))
  ab,_,_,_=interval_pair((9,9,9),ez[ix:ix+1],es[ix:ix+1],np.ones(8,dtype=int))
  errs.extend(abs(float(ai[j]-ab[j])-diff[ix,j]) for j in range(2))
 risks=[]
 for row in selected:
  c=old.matrix(row['candidate']);pp=[old.pmf(c,o) for o in (32,48,64)];rr=[old.risk(c,o) for o in (32,48,64)];ip=[old.pmf3(c,o) for o in (32,48)]
  pe=max(float(np.max(np.abs(p-pp[-1]))) for p in pp+ip);re=max(abs(x-rr[-1]) for x in rr);me=max(abs(x.sum()-1) for x in pp+ip)
  risks.append(dict(n=row['n'],candidate=row['candidate'],risk=rr[-1],risk_by_order=rr,eigenvalues=np.linalg.eigvalsh(c),probability_error=pe,risk_error=re,mass_error=me,passed=pe<=1e-9 and re<=1e-8 and me<=1e-12))
 gates=dict(n0_returns_I=selected[0]['candidate']==[0,0,0],n256_empirical_region=empirical[1]['region_excluded'],n0_reference_order=bool(vB[0]-vI[0]>0),population_n0_order=pop[0]['interval'][1]<0,population_n256_region=pop[1]['interval'][0]>old.rounded_upper(cost),population_interaction=pop[2]['interval'][0]>0,numerical_qualification=max(errs)<1e-10 and all(x['passed'] for x in risks))
 result=dict(start=read(OUT/'RUN_START.json'),finished_utc=now(),wall_seconds=time.monotonic()-started,freeze_sha256=sha(OUT/'FREEZE.json'),selected=selected,empirical=empirical,population=read(OUT/'POPULATION.json'),risks=risks,audit_arithmetic=dict(indices=ids.tolist(),max_error=max(errs),all_population_nodes_interval_checked=False),gates=gates,all_primary_gates_passed=all(gates.values()),resources=dict(new_observations=0,reused_observations=65536,training_banks=B,evaluation_banks=L,K=K,prior_vectors=(B+L)*K,prior_Gaussians=(B+L)*K*259,learning_programs=2,candidates=len(candidates),optimizer_updates=0),arrays={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.glob('*.npz')},claims=dict(fixed_program_weight_intervention_only=True,general_optimizer_claim=False,asymptotic_remainder_verified=False,population_global_numerical_error_bound_proved=False))
 save(OUT/'RESULTS.json',result);print(json.dumps(native(dict(completed=True,wall_seconds=result['wall_seconds'],gates=gates,selected=selected,risks=risks))),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','freeze','run']);a=p.parse_args()
 try:globals()[a.mode]()
 except BaseException as e:
  save(OUT/f'FAILURE_{a.mode}_{time.time_ns()}.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()));raise

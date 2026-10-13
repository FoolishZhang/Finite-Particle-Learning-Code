"""Frozen three-correlation finite selection and continuous low-risk exclusion.

Raw scientific arrays remain local. No changes to previous studies. The learner
uses only empirical labels and its particle scores; risk is evaluated afterward.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, itertools, json, math, sys, time, traceback
import numpy as np
from scipy.special import expit, logsumexp, roots_hermitenorm, roots_legendre

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/finite_selection_v1_20261006'
DOC=ROOT/'docs/08_research_研究工作/FiniteSelection_v1_20261006'
sys.path.insert(0,str(ROOT/'.cache/finite_selection_tools'))
from flint import arb, ctx
ctx.prec=128
M=3; K=2; N=65536; B=65536; EVAL=2097152; BLOCK=4096; R0=1e-6
YS=np.array(list(itertools.product((-1,1),repeat=3)),dtype=np.int8)
PAIRS=[(0,1),(0,2),(1,2)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def now():return datetime.now(timezone.utc).isoformat()
def rng(role):return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610060101,role])))
def matrix(t):
 c=np.eye(3)
 for (i,j),x in zip(PAIRS,t):c[i,j]=c[j,i]=int(x)/10
 return c
CB=matrix((9,9,9))
def grid():
 return [t for t in itertools.product((-9,-6,-3,0,3,6,9),repeat=3)
         if 729+2*t[0]*t[1]*t[2]-9*sum(x*x for x in t)>=0]
def root(c):
 w,v=np.linalg.eigh(c);assert w.min()>.099999999999
 return (v*np.sqrt(w))@v.T
def scores(c,z,s,q,gradient=False):
 x=z@root(c);a=s-logsumexp(s,axis=1)[:,None]
 total=np.zeros(len(z));g=np.zeros((len(z),3))
 for y,p in zip(YS,q):
  logs=-np.logaddexp(0,-x*y).sum(axis=2)
  logits=a+logs;normal=logsumexp(logits,axis=1)
  total-=p*normal
  if gradient:
   assert np.array_equal(c,np.eye(3))
   pi=np.exp(logits-normal[:,None]);gx=y*expit(-x*y)
   for k,(i,j) in enumerate(PAIRS):g[:,k]-=p*(pi*(gx[:,:,i]*z[:,:,j]+gx[:,:,j]*z[:,:,i])/2).sum(axis=1)
 return (total,g) if gradient else total
def rounded_upper(a):return math.nextafter(float(a.upper()),math.inf)
def ball(a):return dict(mid=float(a.mid()),lower=math.nextafter(float(a.lower()),-math.inf),upper=rounded_upper(a),enclosure=str(a))
def root_error(t,s):
 a=[[arb(float(s[i,j])) for j in range(3)] for i in range(3)]
 # Sylvester's criterion certifies positivity of the symmetric rounded root.
 assert a[0][0]>0 and a[0][0]*a[1][1]-a[0][1]**2>0
 det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]**2)-a[0][1]*(a[0][1]*a[2][2]-a[1][2]*a[0][2])+a[0][2]*(a[0][1]*a[1][2]-a[1][1]*a[0][2])
 assert det>0
 c=[[arb(int(i==j)) for j in range(3)] for i in range(3)]
 for (i,j),v in zip(PAIRS,t):c[i][j]=c[j][i]=arb(int(v))/10
 residual=sum((sum(a[i][k]*a[k][j] for k in range(3))-c[i][j])**2 for i in range(3) for j in range(3)).sqrt()
 return a,residual/(arb(1)/10).sqrt()
def interval_score(t,z,s,counts,gradient=False,deadline=None):
 start=time.monotonic();sr=root(matrix(t));sr=(sr+sr.T)/2
 ar,rooterr=root_error(t,sr)
 q=[arb(int(v))/int(sum(counts)) for v in counts]
 total=arb(0);grad=[arb(0) for _ in PAIRS];a2=arb(0)
 for bi,(zs,ss) in enumerate(zip(z,s)):
  az=[[arb(float(v)) for v in row] for row in zs]
  a2+=sum(v*v for row in az for v in row)
  x=[[sum(ar[i][j]*az[k][j] for j in range(3)) for i in range(3)] for k in range(2)]
  pos=[[1/(1+(-v).exp()) for v in row] for row in x]
  a0=1/(1+(arb(float(ss[1]))-arb(float(ss[0]))).exp());weights=[a0,1-a0]
  for yi,y in enumerate(YS):
   prob=[[pos[k][j] if y[j]>0 else 1-pos[k][j] for j in range(3)] for k in range(2)]
   ls=[prob[k][0]*prob[k][1]*prob[k][2] for k in range(2)]
   mix=weights[0]*ls[0]+weights[1]*ls[1]
   total-=q[yi]*mix.log()
   if gradient:
    assert t==(0,0,0)
    pi=[weights[k]*ls[k]/mix for k in range(2)]
    for ci,(i,j) in enumerate(PAIRS):
     grad[ci]-=q[yi]*sum(pi[k]*(int(y[i])*(1-prob[k][i])*az[k][j]+int(y[j])*(1-prob[k][j])*az[k][i])/2 for k in range(2))
  if (bi+1)%4096==0:
   assert time.monotonic()-start<1800,'Interval audit exceeded fixed wall budget'
   assert deadline is None or time.monotonic()<deadline,'Full interval phase exceeded fixed wall budget'
   print(json.dumps(dict(stage='interval',candidate=t,banks=bi+1,total=len(z))),flush=True)
 a2/=len(z);err=(3*a2).sqrt()*rooterr
 enclosure=total/len(z)+arb(0,rounded_upper(err))
 return enclosure,[v/len(z) for v in grad],a2,err
def bounds(a2=arb(6)):
 beta=(-arb(1)/2).exp()/4;r=arb('0.000001');d=(12*r).sqrt()/beta;lam=1-d
 h=(arb(13)/4)*a2/(4*lam)+(3*a2).sqrt()/(4*lam**arb('1.5'))
 return d,h,h*d*d/2
def pair_moment(c,order,mu=(0.,0.)):
 x,w=roots_hermitenorm(order);w=w/math.sqrt(2*math.pi)
 a=mu[0]+x[:,None];b=mu[1]+c*x[:,None]+math.sqrt(1-c*c)*x[None,:]
 return float(np.sum(w[:,None]*w[None,:]*np.tanh(a/2)*np.tanh(b/2)))
def pmf(c,order):
 r=[pair_moment(c[i,j],order) for i,j in PAIRS]
 return (1+sum(v*YS[:,i]*YS[:,j] for v,(i,j) in zip(r,PAIRS)))/8
def risk(c,order):
 p=pmf(c,order);assert np.min(p)>0
 return float(np.mean(-np.log(p*8)))
def pmf3(c,order):
 x,w=roots_hermitenorm(order);w=w/math.sqrt(2*math.pi)
 a=np.array(list(itertools.product(x,repeat=3)));ww=(w[:,None,None]*w[None,:,None]*w[None,None,:]).ravel()
 a=a@root(c)
 return np.array([np.sum(ww*expit(a*y).prod(axis=1)) for y in YS])
def identity():
 rows=[];mu=(-.6,.4)
 for order in (32,48,64):
  x,w=roots_hermitenorm(order);w=w/math.sqrt(2*math.pi);wx=w[:,None]*w[None,:]
  means=[float(w@np.tanh((u+x)/2)) for u in mu]
  tau=pair_moment(.25,order,mu);kap=tau-means[0]*means[1]
  def var(c):
   xx=np.stack(np.broadcast_arrays(mu[0]+x[:,None],mu[1]+c*x[:,None]+math.sqrt(1-c*c)*x[None,:]),axis=-1)
   answer=0.
   for y in itertools.product((-1,1),repeat=2):
    py=(1+y[0]*means[0]+y[1]*means[1]+y[0]*y[1]*tau)/4
    a=-np.logaddexp(0,-xx*np.array(y)).sum(axis=2);ea=np.sum(wx*a)
    answer+=py*np.sum(wx*(a-ea)**2)
   return answer
  baseline=var(0);nodes,weights=roots_legendre(64)
  for c in (-.8,-.3,0.,.4,.8):
   integ=c/2*sum(v*(pair_moment(c*(u+1)/2,order,mu)-means[0]*means[1]) for u,v in zip(nodes,weights))
   potential=-(kap*c+integ)/4;diff=var(c)-baseline
   rows.append(dict(order=order,c=c,T_difference=diff,V=potential,error=abs(potential+diff/2)))
 return dict(rows=rows,maximum_error=max(v['error'] for v in rows),passed=max(v['error'] for v in rows)<1e-9)
def qualify():
 z=np.arange(48,dtype=float).reshape(8,2,3)/13-1.5;s=np.arange(16,dtype=float).reshape(8,2)/5
 counts=np.arange(1,9);q=counts/counts.sum();v,g=scores(np.eye(3),z,s,q,True)
 iv,ig,_,_=interval_score((0,0,0),z,s,counts,True)
 errs=[abs(float(iv)-v.mean())]+[abs(float(a)-g.mean(axis=0)[i]) for i,a in enumerate(ig)]
 fd=[]
 for i in range(3):
  h=np.zeros((3,3));a,b=PAIRS[i];h[a,b]=h[b,a]=1e-5
  fd.append(abs((scores(np.eye(3)+h,z,s,q).mean()-scores(np.eye(3)-h,z,s,q).mean())/2e-5-g[:,i].mean()))
 assert max(errs)<1e-12 and max(fd)<1e-8
 cb,_,_,e=interval_score((9,9,9),z,s,counts)
 assert abs(float(cb)-scores(CB,z,s,q).mean())<1e-12
 out=dict(created_utc=now(),manufactured_inputs_only=True,score_gradient_arb_error=max(errs),gradient_fd_error=max(fd),reference_score_difference=abs(float(cb)-scores(CB,z,s,q).mean()),root_error_allowance=ball(e),grid_candidates=len(grid()),passed=True)
 save(OUT/'QUALIFICATION.json',out);print(json.dumps(out),flush=True)
def freeze():
 assert not (OUT/'FREEZE.json').exists() and not (OUT/'TRAIN_BANKS.npz').exists()
 assert read(OUT/'QUALIFICATION.json')['passed']
 names=[Path(__file__),DOC/'CONTRACT.md',OUT/'HANDOFF_INPUT.md',OUT/'MULTIMODEL_REVIEW_INPUT.md',OUT/'QUALIFICATION.json']
 save(OUT/'FREEZE.json',dict(created_utc=now(),sources={p.relative_to(ROOT).as_posix():sha(p) for p in names},N=N,B_train=B,B_evaluation=EVAL,r0=R0,n=[0,256],K=K,alpha=.01,python_flint='0.8.0',private_dependency_directory='.cache/finite_selection_tools',no_scientific_draw_before_freeze=True))
def run():
 frozen=read(OUT/'FREEZE.json')
 for p,h in frozen['sources'].items():assert sha(ROOT/p)==h,p
 assert not (OUT/'TRAIN_BANKS.npz').exists(),'No silent repeat'
 started=time.monotonic();started_utc=now()
 labels=rng(0).integers(0,8,size=N,dtype=np.uint8);counts=np.bincount(labels,minlength=8);q=counts/N
 z=rng(1).standard_normal((B,2,3));s=np.empty((B,2));nr=rng(2)
 for lo in range(0,B,BLOCK):s[lo:lo+BLOCK]=-np.logaddexp(0,-nr.standard_normal((BLOCK,2,256))).sum(axis=2)
 np.savez(OUT/'TRAIN_BANKS.npz',labels=labels,counts=counts,z=z,s=s)
 candidates=grid();learning=[]
 for n in (0,256):
  ss=np.zeros_like(s) if n==0 else s
  values=np.array([scores(matrix(t),z,ss,q).mean() for t in candidates])
  index=int(np.flatnonzero(values<=values.min()+1e-12)[0]);t=candidates[index]
  learning.append(dict(n=n,proposal=list(t),score=float(values[index]),reference_score=float(values[candidates.index((9,9,9))])))
  np.savez(OUT/f'GRID_N{n}.npz',candidates=np.array(candidates),scores=values)
  print(json.dumps(dict(stage='grid',**learning[-1])),flush=True)
 save(OUT/'LEARNER_PROPOSALS.json',dict(created_utc=now(),learning=learning,risk_accessed=False))
 # Independent point audit. All nuisance sums/active banks and per-bank paired
 # differences are retained; no stopping/adaptation by intermediate outcomes.
 ez=np.empty((EVAL,2,3));es=np.empty((EVAL,2));diff=np.empty((EVAL,2));er,ns=rng(3),rng(4)
 for lo in range(0,EVAL,BLOCK):
  zz=er.standard_normal((BLOCK,2,3));ss=-np.logaddexp(0,-ns.standard_normal((BLOCK,2,256))).sum(axis=2)
  ez[lo:lo+BLOCK]=zz;es[lo:lo+BLOCK]=ss
  for ni,n in enumerate((0,256)):
   tt=np.zeros_like(ss) if n==0 else ss
   diff[lo:lo+BLOCK,ni]=scores(np.eye(3),zz,tt,np.ones(8)/8)-scores(CB,zz,tt,np.ones(8)/8)
  if (lo+BLOCK)%(BLOCK*32)==0:print(json.dumps(dict(stage='population',banks=lo+BLOCK,total=EVAL)),flush=True)
  assert time.monotonic()-started<1800,'Main computation exceeded fixed budget'
 np.savez(OUT/'EVALUATION_BANKS.npz',z=ez,s=es,difference=diff)
 trunc=diff.copy();outside=np.max(np.abs(ez),axis=(1,2))>8;trunc[outside]=0
 A=max(1-math.sqrt(.1),math.sqrt(2.8)-1);bound=3*A*8
 tail=math.sqrt(3)*A*math.sqrt(6*12*math.exp(-32)/math.sqrt(2*math.pi)/8)
 d,h,popcost=bounds();population=[]
 for ni,n in enumerate((0,256)):
  v=trunc[:,ni];sd2=float(np.var(v,ddof=1));log=math.log(4/.005)
  width=math.sqrt(2*sd2*log/EVAL)+7*2*bound*log/(3*(EVAL-1))+tail+1e-9
  mean=float(np.mean(v));population.append(dict(n=n,mean=mean,sample_variance=sd2,interval=[mean-width,mean+width],halfwidth=width,tail_bound=tail,range_bound=bound,truncated_banks=int(outside.sum()),region_cost=ball(popcost),region_gap_lower=mean-width-rounded_upper(popcost),region_excluded=mean-width>rounded_upper(popcost)))
 save(OUT/'POPULATION.json',dict(created_utc=now(),family_alpha=.01,point_alpha=.005,B=EVAL,method='two-sided scaled Maurer-Pontil plus tail and numerical allowance',results=population))
 # Full ball replay of the realized finite criterion, independently of the
 # population audit. Input sums and labels are now exact saved program inputs.
 interval=[];interval_deadline=time.monotonic()+1800
 for row in learning:
  n=row['n'];ss=np.zeros_like(s) if n==0 else s;t=tuple(row['proposal'])
  si,gi,a2,_=interval_score((0,0,0),z,ss,counts,True,interval_deadline)
  sb,_,_,_=interval_score((9,9,9),z,ss,counts,deadline=interval_deadline)
  if t==(9,9,9):sp=sb
  elif t==(0,0,0):sp=si
  else:sp,_,_,_=interval_score(t,z,ss,counts,deadline=interval_deadline)
  selected=t if t==(9,9,9) or sb-sp>=0 else (9,9,9)
  d,hh,cost=bounds(a2);linear=(sum(v*v for v in gi)/2).sqrt()*d
  gap=si-sb-linear-cost
  interval.append(dict(n=n,proposal=list(t),selected=list(selected),comparison_verified=selected==(9,9,9) or bool(sb-sp>=0),fallback_used=selected!=t,score_I=ball(si),score_reference=ball(sb),proposal_improvement=ball(sb-sp),gradient_I=[ball(v) for v in gi],a2_mean=ball(a2),radius=ball(d),curvature=ball(hh),linear_cost=ball(linear),quadratic_cost=ball(cost),empirical_region_gap=ball(gap),empirical_region_excluded=bool(gap>0)))
  save(OUT/'INTERVAL_PROGRESS.json',dict(completed=interval))
 save(OUT/'SELECTED_OUTPUTS.json',dict(created_utc=now(),results=interval,risk_used_for_selection=False))
 # Prespecified sampled population arithmetic qualification, not full interval
 # integration. The distinction is retained in the result and manuscript.
 ids=np.linspace(0,EVAL-1,32,dtype=int);numeric=[]
 for ni,n in enumerate((0,256)):
  checks=[]
  for ix in ids:
   ss=es[ix:ix+1] if n else np.zeros((1,2))
   ai,_,_,_=interval_score((0,0,0),ez[ix:ix+1],ss,np.ones(8,dtype=int))
   ab,_,_,_=interval_score((9,9,9),ez[ix:ix+1],ss,np.ones(8,dtype=int))
   checks.append(abs(float(ai-ab)-diff[ix,ni]))
  numeric.append(dict(n=n,indices=ids.tolist(),max_error=max(checks),passed=max(checks)<1e-10))
 risks=[]
 for row in interval:
  c=matrix(row['selected']);ps=[pmf(c,o) for o in (32,48,64)];rr=[risk(c,o) for o in (32,48,64)]
  independent=[pmf3(c,o) for o in (32,48)];pe=max(float(np.max(np.abs(p-ps[-1]))) for p in ps+independent)
  re=max(abs(v-rr[-1]) for v in rr);mass=max(abs(p.sum()-1) for p in ps+independent)
  risks.append(dict(n=row['n'],selected=row['selected'],eigenvalues=np.linalg.eigvalsh(c).tolist(),reference_risk=rr[-1],orders=[32,48,64],risk_by_order=rr,maximum_probability_change=pe,maximum_risk_change=re,mass_error=float(mass),passed=pe<=1e-9 and re<=1e-8 and mass<=1e-12))
 ident=identity();save(OUT/'VARIANCE_IDENTITY.json',ident)
 arrays={p.relative_to(ROOT).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in OUT.glob('*.npz')}
 result=dict(started_utc=started_utc,finished_utc=now(),wall_seconds=time.monotonic()-started,freeze_sha256=sha(OUT/'FREEZE.json'),learning=learning,empirical_certificates=interval,population=population,population_arithmetic_qualification=numeric,risks=risks,variance_identity_passed=ident['passed'],arrays=arrays,resources=dict(active_observations=N,training_banks=B,evaluation_banks=EVAL,shared_K=2,shared_nuisance_dimension=256,total_prior_vectors=(B+EVAL)*2,total_prior_Gaussians=(B+EVAL)*2*259,grid_candidates=len(candidates),learning_programs=2,optimizer_updates=0,retained_sufficient_nuisance_sums=True),claims=dict(realized_certificate_for_saved_inputs=True,population_interval_requires_iid_Gaussian_model=True,population_all_node_interval_validation=False,continuous_global_optimality_claimed=False,Adam_claimed=False))
 save(OUT/'RESULTS.json',result);print(json.dumps(dict(completed=True,wall_seconds=result['wall_seconds'],empirical=[(r['n'],r['empirical_region_excluded']) for r in interval],population=[(r['n'],r['region_excluded']) for r in population],risks=risks)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','freeze','run']);a=p.parse_args()
 try:globals()[a.mode]()
 except BaseException as e:
  save(OUT/f'FAILURE_{a.mode}_{time.time_ns()}.json',dict(time=now(),error=repr(e),traceback=traceback.format_exc(),mode=a.mode));raise

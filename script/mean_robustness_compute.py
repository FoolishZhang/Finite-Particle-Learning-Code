"""Frozen asymmetric-known-mean derivative audit; no fitting or manuscript edits."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction as Q
from math import factorial
import argparse,hashlib,json,math,os,sys,time,traceback
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.special import expit,roots_hermitenorm
import mean_robustness_kernel as kernel
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/mean_robustness_v1_20261006';DOC=ROOT/'docs/08_research_研究工作/MeanRobustness_v1_20261006'
sys.path.insert(0,str(ROOT/'.cache/finite_selection_tools'))
from flint import arb,acb,ctx
ctx.prec=128
BASE='830b64c3dcad1babdd187740c982fa77cf3a2729';L=262144;K=100;BLOCK=256;MU=np.array([-.2,.1,.3]);NAMES=['g12','g13','g23','h12_12','h13_13','h23_23','h12_13','h12_23','h13_23'];INDICES=[(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ball(x):return dict(mid=float(x.mid()),lower=math.nextafter(float(x.lower()),-math.inf),upper=math.nextafter(float(x.upper()),math.inf),enclosure=str(x))
def analytic():
 def partition_sum(n):
  def rec(k,blocks):
   if k==n:return factorial(len(blocks)-1)*math.prod(factorial(b) for b in blocks)
   return sum(rec(k+1,blocks[:i]+[b+1]+blocks[i+1:]) for i,b in enumerate(blocks))+rec(k+1,blocks+[1])
  return rec(1,[1])
 t=[partition_sum(n) for n in range(1,7)];assert t==[1,3,14,90,744,7560]
 M=Q(1,10**7);r=Q(3,10**8);eta=2367*M+22680*Q(1733,1000)*r;assert Q(1733,1000)**2>3 and eta<Q(1416,10**6)
 a0=Q(2684,10**6);a1=Q(2202,10**6);g=Q(419,4)*M*M;shift=g/a1
 assert g<a1*r and shift<Q(48,10**11)
 eps=Q(1415,1000)*Q(48,10**11);risk_upper=Q(48,10**11)**2/(2*(1-eps)**2)
 assert Q(1415,1000)**2>2 and risk_upper<Q(1153,10**22)
 eu=sum(Q(1,factorial(k)) for k in range(8))+Q(1,factorial(8))/(1-Q(1,9));assert eu<Q(2719,1000)
 risk_lower=Q(1000,2719)*(1-M*M/2)*r*r/96;assert risk_lower>Q(344,10**20)
 old=read(ROOT/'artifacts/local_geometry_v1_20261006/RESULTS.json')['statistics']['results'][1]['interval'][1]
 assert Q.from_float(-old)-eta>a1 and Q(41,10000)-eta>a0
 encode=lambda x:dict(numerator=str(x.numerator),denominator=str(x.denominator),display=float(x))
 return dict(derivative_constants=t,mean_l1_radius=1e-7,correlation_l2_radius=3e-8,hessian_deviation_upper=encode(eta),n0_curvature_lower=float(a0),n256_negative_curvature_magnitude_lower=float(a1),gradient_norm_upper=encode(g),stationary_displacement_upper=encode(shift),stationary_risk_upper=encode(risk_upper),n256_global_risk_lower=encode(risk_lower),old_event_reused=True,new_statistical_contrasts=0,new_alpha=0)
def probability(mu):
 calls=0;normal=(2*arb.pi()).sqrt();am=arb(float(mu))
 def f(z,analytic):
  nonlocal calls
  calls+=1;return (-z*z/2).exp()/((1+(-am-z).exp())*normal)
 val=acb.integral(f,-12,12,rel_tol=arb('1e-25'),abs_tol=arb('1e-25'),eval_limit=100000)
 assert val.is_finite() and val.imag.contains(0)
 tail=2*(-arb(72)).exp()/(normal*12);p=val.real+arb(0,float(tail.upper()))
 b=ball(p);gh=[]
 for order in (64,96):
  x,w=roots_hermitenorm(order);gh.append(float(np.dot(w,expit(float(mu)+x))/math.sqrt(2*math.pi)))
 assert b['upper']-b['lower']<1e-12 and max(abs(v-b['mid']) for v in gh)<1e-10
 return dict(mu=float(mu),probability=b,tail=ball(tail),GH_orders=[64,96],GH=gh,integrand_callbacks=calls,qualification_passed=True)
def teacher():
 rows=[probability(mu) for mu in MU];p=[r['probability']['mid'] for r in rows]
 errors=[max(abs(p[i]-r['probability']['lower']),abs(r['probability']['upper']-p[i])) for i,r in enumerate(rows)]
 tv_l1=2*sum(errors)
 assert tv_l1*7<1e-12
 return dict(mu=MU.tolist(),p=p,components=rows,teacher_distribution_l1_rounding_bound=tv_l1,integrand_callbacks=sum(r['integrand_callbacks'] for r in rows),GH_node_visits=3*(64+96),passed=True)
def pack(g,h):return np.concatenate([g,np.stack([h[:,:,i,j] for i,j in INDICES],axis=-1)],axis=-1)
def statistics(values,teacher):
 rows=[];ranges=[2]*3+[133/16]*3+[14]*3;maxima=[1]*3+[67/16]*3+[7]*3
 for condition in range(2):
  for j,name in enumerate(NAMES):
   x=values[:,condition,j];mean=float(x.mean());v=float(x.var(ddof=1));allowance=1e-9+maxima[j]*teacher['teacher_distribution_l1_rounding_bound'];half=math.sqrt(2*v*math.log(7200)/L)+7*ranges[j]*math.log(7200)/(3*(L-1))+allowance
   rows.append(dict(n=(0,256)[condition],component=name,mean=mean,variance=v,interval=[mean-half,mean+half],halfwidth=half,range_length=ranges[j],numerical_allowance=allowance,observed_min=float(x.min()),observed_max=float(x.max())))
 matrices=[]
 for condition in range(2):
  arm=rows[condition*9:(condition+1)*9];lo=np.zeros((3,3));hi=lo.copy();mid=lo.copy()
  for row,(i,j) in zip(arm[3:],INDICES):lo[i,j]=lo[j,i]=row['interval'][0];hi[i,j]=hi[j,i]=row['interval'][1];mid[i,j]=mid[j,i]=row['mean']
  off=np.maximum(np.abs(lo),np.abs(hi));np.fill_diagonal(off,0)
  matrices.append(dict(n=(0,256)[condition],mean_hessian=mid.tolist(),descriptive_mean_eigenvalues=np.linalg.eigvalsh(mid).tolist(),eigenvalue_lower=float(np.min(np.diag(lo)-off.sum(axis=1))),eigenvalue_upper=float(np.max(np.diag(hi)+off.sum(axis=1))),gradient_zero_excluded=[row['interval'][0]>0 or row['interval'][1]<0 for row in arm[:3]]))
 return dict(alpha=.01,point_alpha=.01/18,banks=L,contrasts=18,results=rows,matrices=matrices,combined_with_local_geometry_lower_coverage=.98)
def freeze():
 OUT.mkdir(parents=True,exist_ok=True);save(OUT/'ANALYTIC_CERTIFICATES.json',analytic());q=kernel.qualification()
 q['teacher_integrator_fixture']=probability(0.0);assert q['teacher_integrator_fixture']['probability']['lower']<.5<q['teacher_integrator_fixture']['probability']['upper']
 save(OUT/'QUALIFICATION.json',q);(OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
 files=[Path(__file__),ROOT/'script/mean_robustness_kernel.py',*[DOC/x for x in ('CONTRACT.md','KERNEL_DERIVATION.md','QUALITATIVE_PROOF.md','QUANTITATIVE_PROOF.md')],*[OUT/x for x in ('QUALIFICATION.json','ANALYTIC_CERTIFICATES.json','HANDOFF_INPUT.md')],ROOT/'artifacts/local_geometry_v1_20261006/DELIVERY_MANIFEST.json']
 save(OUT/'FREEZE.json',dict(created_utc=now(),baseline=BASE,L=L,K=K,block=BLOCK,mu=MU.tolist(),alpha=.01,seed_family=202610060401,new_scientific_banks_before_freeze=0,analytic_feasibility_done_before_freeze=True,sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
def run():
 start=time.monotonic();f=read(OUT/'FREEZE.json')
 for p,h in f['sources'].items():assert sha(ROOT/p)==h,p
 save(OUT/'RUN_START.json',dict(utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')))
 try:
  t=teacher();save(OUT/'TEACHER.json',t);z=np.empty((L,K,3));s=np.empty((L,K));values=np.empty((L,2,9))
  rz=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610060401,1])));rg=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610060401,2])))
  for first in range(0,L,BLOCK):
   last=first+BLOCK;z[first:last]=rz.standard_normal((BLOCK,K,3));g=rg.standard_normal((BLOCK,K,256));s[first:last]=-np.logaddexp(0,-g).sum(axis=2)
   values[first:last]=pack(*kernel.mean_derivatives(z[first:last],s[first:last],MU,t['p']))
   assert time.monotonic()-start<45*60,'main deadline exceeded'
   if last%16384==0:print(json.dumps(dict(stage='full_derivatives',banks=last,total=L,wall_seconds=time.monotonic()-start)),flush=True)
  assert np.isfinite(values).all();array=OUT/'BANKS.npz';assert not array.exists();np.savez_compressed(array,z=z,s=s,values=values)
  stats=statistics(values,t);save(OUT/'STATISTICS.json',stats);audit_start=time.monotonic();indices=np.linspace(0,L-1,32,dtype=int);checks=[]
  for index in indices:
   ag,ah=kernel.interval_derivatives(z[index],s[index],MU,t['p'],bits=128);av=[[*ag[a],*[ah[a][i][j] for i,j in INDICES]] for a in range(2)]
   err=max(abs(float(av[a][j].mid())-float(values[index,a,j])) for a in range(2) for j in range(9));checks.append(dict(index=int(index),max_error=err,values=[[str(v) for v in row] for row in av]));assert time.monotonic()-audit_start<15*60,'interval deadline exceeded'
  error=max(x['max_error'] for x in checks);audit=dict(indices=indices.tolist(),comparisons=576,max_error=error,passed=bool(error<=1e-10),precision_bits=128,all_nodes_interval_checked=False,checks=checks);save(OUT/'ARITHMETIC_AUDIT.json',audit)
  gates=dict(n0_positive_definite=stats['matrices'][0]['eigenvalue_lower']>0,n256_negative_definite=stats['matrices'][1]['eigenvalue_upper']<0,teacher_quadrature=t['passed'],numerical_qualification=audit['passed'])
  result=dict(finished_utc=now(),freeze_sha256=sha(OUT/'FREEZE.json'),statistics=stats,gates=gates,all_primary_gates_passed=all(gates.values()),teacher_sha256=sha(OUT/'TEACHER.json'),analytic_certificates_sha256=sha(OUT/'ANALYTIC_CERTIFICATES.json'),audit_sha256=sha(OUT/'ARITHMETIC_AUDIT.json'),wall_seconds=time.monotonic()-start,array=dict(path=array.relative_to(ROOT).as_posix(),sha256=sha(array),bytes=array.stat().st_size),resources=dict(new_observations=0,prior_vectors=L*K,Gaussian_scalars=L*K*259,bank_derivative_components=L*18,internal_label_derivative_components=L*18*8,score_calls=0,optimizer_updates=0,teacher_integrand_callbacks=t['integrand_callbacks'],teacher_GH_node_visits=t['GH_node_visits'],statistical_contrasts=18,interval_derivative_checks=576),claims=dict(known_matched_means=True,means_learned=False,truth_stationary_claimed=False,global_n0_optimum_proved=False,manuscript_modified=False,new_family_only=True))
  save(OUT/'RESULTS.json',result);print(json.dumps(dict(completed=True,gates=gates,matrices=stats['matrices'],wall_seconds=result['wall_seconds'])),flush=True)
 except BaseException as e:
  save(OUT/'FAILURE.json',dict(utc=now(),error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','run'));a=p.parse_args();freeze() if a.action=='freeze' else run()

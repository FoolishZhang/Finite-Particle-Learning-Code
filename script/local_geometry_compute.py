"""Frozen matched local-curvature audit plus exact analytic n0 certificates."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction as Q
from math import factorial
import argparse,hashlib,json,math,os,sys,time,traceback
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import local_geometry_kernel as kernel
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/local_geometry_v1_20261006'
DOC=ROOT/'docs/08_research_研究工作/LocalGeometry_v1_20261006'
BASE='325138f65f958d445b2b523f6154c0164fd3e93d'
L=262144;K=100;BLOCK=256;ALPHA=.01
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def exact(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator),display=float(x))
def analytic_certificates():
 r=Q(57,32);a=Q(204124,10**6)
 assert 24*a*a<1 and sum(Q(694,1000)**j/factorial(j) for j in range(7))>2
 assert sum(Q(4,3)**j/factorial(j) for j in range(6))>Q(15,4)
 eu=sum(r**j/factorial(j) for j in range(13))+(r**13/factorial(13))/(1-r/14)
 qu=1-Q(1,6)+Q(8,9)*(Q(1,2)-a)**3
 pl=Q(100*99,2304)*Q(7,8)**2/128/eu;nu=200*qu**99*(3*(Q(694,1000)+1)+Q(1,5))
 assert pl-nu>Q(41,10000)
 x=Q(1,198);ell=sum(x**j/j for j in range(1,5))+x**5/(5*(1-x));z=Q(297,2)*ell+Q(3,197)
 assert 0<z<1
 pu=sum(z**j/factorial(j) for j in range(13))+z**13/factorial(13)/(1-z/14)
 u=Q(140627215,10**10);s=Q(13758517,10**9);v=Q(61,4096);e=Q(5,16)*(Q(1,72)+Q(3,112))
 assert Q(64,9801)*pu<u and v*e<s*s
 ratio=Q(2025,103376)-100*s*u;assert ratio>Q(240,10**6)
 return dict(laplace=dict(K=100,lower=exact(pl-nu),certified_threshold=exact(Q(41,10000))),ratio=dict(K_min=100,lower_at_100=exact(ratio),certified_threshold=exact(Q(240,10**6))),strict_local_minimum=True,global_minimum_proved=False,new_random_draws=0)
def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 save(OUT/'ANALYTIC_CERTIFICATES.json',analytic_certificates())
 q=kernel.qualification();errors=[]
 for row in q['fixtures']:
  a=kernel.interval_curvature(row['active_z'],row['log_nuisance'],bits=128)
  errors.extend(abs(float(a[i][j].mid())-row['kernel'][i][j]) for i in range(2) for j in range(3))
 q['interval_fixture_max_error']=max(errors);assert max(errors)<1e-10
 save(OUT/'QUALIFICATION.json',q)
 (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
 sources=[Path(__file__),ROOT/'script/local_geometry_kernel.py',*[DOC/n for n in ('CONTRACT.md','KERNEL_DERIVATION.md','N0_LAPLACE_PROOF.md','N0_RATIO_PROOF.md')],*[OUT/n for n in ('QUALIFICATION.json','ANALYTIC_CERTIFICATES.json','HANDOFF_INPUT.md')],ROOT/'artifacts/joint_risk_refinement_v1_20261006/DELIVERY_MANIFEST.json']
 save(OUT/'FREEZE.json',dict(created_utc=now(),baseline=BASE,L=L,K=K,block=BLOCK,alpha=ALPHA,seed_family=202610060301,new_curvature_scientific_draws_before_freeze=0,analytic_feasibility_done_before_freeze=True,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
def intervals(h):
 average=h.mean(axis=2);values=[average[:,0],average[:,1],average[:,1]-average[:,0]]
 rows=[]
 for name,x,ran,allowance in zip(('h0','h256','interaction'),values,(133/16,133/16,133/8),(1e-9,1e-9,2e-9)):
  var=float(x.var(ddof=1));mean=float(x.mean());radius=math.sqrt(2*var*math.log(1200)/L)+7*ran*math.log(1200)/(3*(L-1))+allowance
  rows.append(dict(contrast=name,mean=mean,variance=var,range_length=ran,numerical_allowance=allowance,halfwidth=radius,interval=[mean-radius,mean+radius],observed_min=float(x.min()),observed_max=float(x.max())))
 return dict(alpha=ALPHA,point_alpha=ALPHA/3,banks=L,estimator='mean of three correlated pair kernels per bank',results=rows,pair_means=h.mean(axis=0).tolist(),new_family_only=True,combined_with_prior_99_family_lower_coverage=.98)
def run():
 start=time.monotonic();f=read(OUT/'FREEZE.json')
 for name,digest in f['sources'].items():assert sha(ROOT/name)==digest,name
 save(OUT/'RUN_START.json',dict(utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')))
 try:
  z=np.empty((L,K,3));s=np.empty((L,K));h=np.empty((L,2,3))
  rz=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610060301,1])))
  rg=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([202610060301,2])))
  for first in range(0,L,BLOCK):
   last=min(first+BLOCK,L);z[first:last]=rz.standard_normal((last-first,K,3));g=rg.standard_normal((last-first,K,256))
   s[first:last]=-np.logaddexp(0,-g).sum(axis=2);h[first:last]=kernel.price_curvature(z[first:last],s[first:last])
   assert time.monotonic()-start<45*60,'fixed main deadline exceeded'
   if last%16384==0:print(json.dumps(dict(stage='curvature',banks=last,total=L,wall_seconds=time.monotonic()-start)),flush=True)
  assert np.isfinite(h).all() and h.min()>=kernel.KERNEL_LOWER and h.max()<=kernel.KERNEL_UPPER
  array=OUT/'BANKS.npz';assert not array.exists();np.savez_compressed(array,z=z,s=s,kernels=h)
  summary=intervals(h);save(OUT/'STATISTICS.json',summary)
  interval_start=time.monotonic();indices=np.linspace(0,L-1,32,dtype=int);checks=[]
  for index in indices:
   balls=kernel.interval_curvature(z[index],s[index],bits=128)
   errors=[[abs(float(balls[i][j].mid())-float(h[index,i,j])) for j in range(3)] for i in range(2)]
   checks.append(dict(index=int(index),max_error=max(max(row) for row in errors),interval_values=[[str(x) for x in row] for row in balls]))
   assert time.monotonic()-interval_start<15*60,'fixed interval deadline exceeded'
  maxerror=max(c['max_error'] for c in checks);numerical=bool(maxerror<=1e-10)
  audit=dict(indices=indices.tolist(),comparisons=192,precision_bits=128,max_error=maxerror,passed=numerical,all_population_nodes_interval_checked=False,checks=checks)
  save(OUT/'ARITHMETIC_AUDIT.json',audit)
  rows=summary['results'];gates=dict(n0_positive=rows[0]['interval'][0]>0,n256_negative=rows[1]['interval'][1]<0,interaction_negative=rows[2]['interval'][1]<0,numerical_qualification=numerical)
  result=dict(finished_utc=now(),freeze_sha256=sha(OUT/'FREEZE.json'),statistics=summary,gates=gates,all_primary_gates_passed=all(gates.values()),analytic_certificates_sha256=sha(OUT/'ANALYTIC_CERTIFICATES.json'),arithmetic_audit_sha256=sha(OUT/'ARITHMETIC_AUDIT.json'),wall_seconds=time.monotonic()-start,array=dict(path=array.relative_to(ROOT).as_posix(),sha256=sha(array),bytes=array.stat().st_size),resources=dict(new_observations=0,prior_vectors=L*K,Gaussian_scalars=L*K*259,bank_condition_direction_evaluations=L*2*3,score_calls=0,optimizer_updates=0,quadrature_nodes=0,statistical_contrasts=3,additional_interval_direction_checks=192),claims=dict(qualified_population_curvature_only=True,no_nuisance_global_optimum_proved=False,general_optimizer_guarantee=False,quantitative_neighborhood_proved=False,manuscript_modified=False,original_nuisance_contrast_failed_gate_preserved=True))
  save(OUT/'RESULTS.json',result);print(json.dumps(dict(completed=True,statistics=summary,gates=gates,wall_seconds=result['wall_seconds'])),flush=True)
 except BaseException as error:
  save(OUT/'FAILURE.json',dict(utc=now(),error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','run'));a=p.parse_args();freeze() if a.action=='freeze' else run()

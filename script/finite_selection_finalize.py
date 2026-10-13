"""Frozen same-input completion after NumPy scalar JSON serialization failure.

Does not call run(), any RNG, training, or full interval selection. Repeats only
the already specified numerical diagnostics that were not saved at the failure.
"""
from pathlib import Path
import argparse, json, time
import numpy as np
import finite_selection_compute as c
O=c.OUT
def native(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):native(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [native(v) for v in x]
 return x
def save(p,x):p.write_text(json.dumps(native(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def freeze():
 assert not (O/'RECOVERY_FREEZE.json').exists() and not (O/'RESULTS.json').exists()
 paths=[Path(__file__),Path(c.__file__),O/'FREEZE.json']+list(O.glob('*.npz'))+[O/n for n in ('SELECTED_OUTPUTS.json','POPULATION.json','LEARNER_PROPOSALS.json')]+list(O.glob('FAILURE_run_*.json'))
 save(O/'RECOVERY_FREEZE.json',dict(created_utc=c.now(),reason='NumPy bool in variance identity could not be JSON serialized after scientific outputs and selections were saved',sources={p.relative_to(c.ROOT).as_posix():c.sha(p) for p in paths},new_random_draws=0,new_learning=0,new_selection=0,repeated_diagnostics='32 fixed population banks per n, chosen-output quadrature and variance identity; all same inputs and criteria'))
def finish():
 start=time.monotonic();f=c.read(O/'RECOVERY_FREEZE.json')
 assert not (O/'RESULTS.json').exists()
 for p,h in f['sources'].items():assert c.sha(c.ROOT/p)==h,p
 arr=np.load(O/'EVALUATION_BANKS.npz');ez,es,diff=arr['z'],arr['s'],arr['difference']
 selected=c.read(O/'SELECTED_OUTPUTS.json')['results'];ids=np.linspace(0,c.EVAL-1,32,dtype=int);numeric=[]
 for ni,n in enumerate((0,256)):
  checks=[]
  for ix in ids:
   ss=es[ix:ix+1] if n else np.zeros((1,2))
   ai,_,_,_=c.interval_score((0,0,0),ez[ix:ix+1],ss,np.ones(8,dtype=int))
   ab,_,_,_=c.interval_score((9,9,9),ez[ix:ix+1],ss,np.ones(8,dtype=int))
   checks.append(abs(float(ai-ab)-diff[ix,ni]))
  numeric.append(dict(n=n,indices=ids.tolist(),max_error=max(checks),passed=max(checks)<1e-10))
 risks=[]
 for row in selected:
  mat=c.matrix(row['selected']);ps=[c.pmf(mat,o) for o in (32,48,64)];rr=[c.risk(mat,o) for o in (32,48,64)]
  independent=[c.pmf3(mat,o) for o in (32,48)];pe=max(float(np.max(np.abs(p-ps[-1]))) for p in ps+independent)
  re=max(abs(v-rr[-1]) for v in rr);mass=max(abs(p.sum()-1) for p in ps+independent)
  risks.append(dict(n=row['n'],selected=row['selected'],eigenvalues=np.linalg.eigvalsh(mat).tolist(),reference_risk=rr[-1],orders=[32,48,64],risk_by_order=rr,maximum_probability_change=pe,maximum_risk_change=re,mass_error=float(mass),passed=pe<=1e-9 and re<=1e-8 and mass<=1e-12))
 ident=c.identity();save(O/'VARIANCE_IDENTITY.json',ident)
 arrays={p.relative_to(c.ROOT).as_posix():dict(sha256=c.sha(p),bytes=p.stat().st_size) for p in O.glob('*.npz')}
 result=dict(finished_utc=c.now(),original_run_start_not_recorded=True,recovery_wall_seconds=time.monotonic()-start,freeze_sha256=c.sha(O/'FREEZE.json'),recovery_freeze_sha256=c.sha(O/'RECOVERY_FREEZE.json'),learning=c.read(O/'LEARNER_PROPOSALS.json')['learning'],empirical_certificates=selected,population=c.read(O/'POPULATION.json')['results'],population_arithmetic_qualification=numeric,risks=risks,variance_identity_passed=ident['passed'],arrays=arrays,resources=dict(active_observations=c.N,training_banks=c.B,evaluation_banks=c.EVAL,shared_K=2,shared_nuisance_dimension=256,total_prior_vectors=(c.B+c.EVAL)*2,total_prior_Gaussians=(c.B+c.EVAL)*2*259,grid_candidates=len(c.grid()),learning_programs=2,optimizer_updates=0,retained_sufficient_nuisance_sums=True,recovery_new_random_draws=0,recovery_new_learning=0,repeated_quadrature_diagnostics=True),claims=dict(realized_certificate_for_saved_inputs=True,population_interval_requires_iid_Gaussian_model=True,population_all_node_interval_validation=False,continuous_global_optimality_claimed=False,Adam_claimed=False,n0_also_excludes_low_risk=True,nuisance_necessary_for_this_finite_failure_claimed=False))
 save(O/'RESULTS.json',result)
 for p,h in f['sources'].items():assert c.sha(c.ROOT/p)==h,p
 print(json.dumps(native(dict(completed=True,recovery_wall_seconds=result['recovery_wall_seconds'],empirical=[(r['n'],r['empirical_region_gap']['lower']) for r in selected],population=result['population'],risks=risks,identity=ident['maximum_error']))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','finish']);a=p.parse_args();globals()[a.mode]()

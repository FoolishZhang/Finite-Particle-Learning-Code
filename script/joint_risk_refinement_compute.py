"""Post-hoc deterministic joint-risk refinement; no new scientific samples."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
import argparse,hashlib,itertools,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/joint_risk_refinement_v1_20261006'
DOC=ROOT/'docs/08_research_研究工作/JointRiskRefinement_v1_20261006'
OLD=ROOT/'artifacts/nuisance_contrast_v1_20261006'
sys.path.insert(0,str(ROOT/'.cache/finite_selection_tools'))
from flint import arb,ctx
BASE='cbd42515215ddcb2cac56049b9ffd370ace17565'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ball(x):return dict(mid=float(x.mid()),lower=math.nextafter(float(x.lower()),-math.inf),upper=math.nextafter(float(x.upper()),math.inf),enclosure=str(x))
def qualify():
 ys=list(itertools.product((-1,1),repeat=3));v=[[y[0]*y[1],y[0]*y[2],y[1]*y[2]] for y in ys]
 assert all(sum(row[i] for row in v)==0 for i in range(3))
 assert all(sum(row[i]*row[j] for row in v)==8*int(i==j) for i in range(3) for j in range(3))
 k=[Fraction(1,10),Fraction(-1,20),Fraction(1,25)];u=[sum(a*b for a,b in zip(k,row)) for row in v]
 assert sum(u)==0 and sum(x*x for x in u)/8==sum(x*x for x in k)
 assert all(u[i]==u[7-i] for i in range(8))
 assert all(abs(((1+x)/8)*2-Fraction(1,4))==abs(x)/4 for x in u)
 ctx.prec=128
 for text in ('-.4','-.01','.01','.4'):
  x=arb(text);a=arb('.5');assert - (1+x).log()+x-x*x/(2*(1+a)**2)>0
 assert 2*(1+(8*arb(0)).sqrt())*arb(0).sqrt()==0
 return dict(exact_Walsh_orthogonality=True,exact_second_moment_identity=True,exact_inversion_event_mass=True,positive_and_negative_Taylor_fixtures=4,zero_radius_limit=True,scientific_draws=0)
def calculate(bits):
 ctx.prec=bits;r=arb('0.000001');beta=(-arb(1)/2).exp()/4;a=(8*r).sqrt();d=2*(1+a)*r.sqrt()/beta;lam=1-d
 assert a<1 and lam>0
 U=arb(8)/3*(arb(100).log()+3*arb(2).log())
 H=lambda u,l:(arb(13)/4)*u/(4*l)+(3*u).sqrt()/(4*l**arb('1.5'))
 cost=H(U,lam)*d*d/2;old_d=(12*r).sqrt()/beta;old_cost=H(U,1-old_d)*old_d*old_d/2
 source=read(OLD/'RESULTS.json');lower=arb(source['population']['results'][1]['interval'][0]);margin=lower-cost
 constants=dict(beta=beta,a=a,d_new=d,lambda_new=lam,d_old=old_d,squared_radius_ratio=d*d/(old_d*old_d),population_norm_bound=U,population_H=H(U,lam),new_population_cost=cost,old_population_cost=old_cost,population_gap_lower=lower,new_population_region_lower=margin,cost_reduction=old_cost-cost)
 empirical=[]
 for row in source['empirical']:
  u=arb(row['norm_upper']['upper']);grad=[arb(max(abs(g['lower']),abs(g['upper']))) for g in row['gradient_I']]
  linear=sum(g*g for g in grad).sqrt()*d/arb(2).sqrt();quad=H(u,lam)*d*d/2
  empirical.append(dict(n=row['n'],linear_cost=linear,quadratic_cost=quad,region_lower=arb(row['point_gap']['lower'])-linear-quad))
 return constants,empirical
def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 save(OUT/'QUALIFICATION.json',qualify())
 (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
 sources=[Path(__file__),DOC/'CONTRACT.md',DOC/'PROOF.md',OUT/'QUALIFICATION.json',OUT/'HANDOFF_INPUT.md',OLD/'RESULTS.json',OLD/'POPULATION.json',OLD/'DELIVERY_MANIFEST.json',ROOT/'artifacts/submission_candidate_v1_3_20261006/DELIVERY_MANIFEST.json']
 save(OUT/'FREEZE.json',dict(created_utc=now(),baseline=BASE,post_hoc=True,pre_freeze_feasibility_disclosed=True,r0=1e-6,K=100,n=256,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources}))
def run():
 start=time.monotonic();f=read(OUT/'FREEZE.json')
 for name,h in f['source_sha256'].items():assert sha(ROOT/name)==h,name
 save(OUT/'RUN_START.json',dict(utc=now(),freeze_sha256=sha(OUT/'FREEZE.json')))
 c,e=calculate(128);hi,he=calculate(256)
 for k,x in c.items():assert arb(ball(x)['lower'])<=hi[k] and hi[k]<=arb(ball(x)['upper']),k
 for lo,high in zip(e,he):
  for k in ('linear_cost','quadratic_cost','region_lower'):assert arb(ball(lo[k])['lower'])<=high[k] and high[k]<=arb(ball(lo[k])['upper'])
 result=dict(created_utc=now(),freeze_sha256=sha(OUT/'FREEZE.json'),constants={k:ball(v) for k,v in c.items()},empirical=[{k:(v if k=='n' else ball(v)) for k,v in row.items()} for row in e],verification=dict(primary_bits=128,independent_precision_bits=256,high_precision_inside_outward_intervals=True),claims=dict(post_hoc_analytic_refinement=True,original_all_primary_gates_passed=False,original_passed_gates=6,original_total_gates=7,qualified_population_region_exclusion=bool(c['new_population_region_lower']>0),same_original_joint_event=True,new_alpha_spent=0,global_population_numerical_error_bound_proved=False,general_optimizer_guarantee=False,asymptotic_remainder_certified=False,manuscript_modified=False),resources=dict(new_observations=0,new_prior_vectors=0,new_Gaussians=0,new_score_calls=0,optimizer_updates=0,new_quadrature_nodes=0,new_statistical_contrasts=0),wall_seconds=time.monotonic()-start)
 assert result['wall_seconds']<300
 save(OUT/'RESULTS.json',result);print(json.dumps(result,ensure_ascii=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','run'));a=p.parse_args();freeze() if a.action=='freeze' else run()

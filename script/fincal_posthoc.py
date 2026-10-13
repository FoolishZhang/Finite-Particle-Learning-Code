"""Explicitly exploratory replays; never change the original training family."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,math
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator,PPoly

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/finite_calibration'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def risk(t,r):return float(max(0.,.5*((1+t)*np.log1p((t-r)/(1+r))+(1-t)*np.log1p((r-t)/(1-r)))))
def upper(frac):
    f=float(frac)
    while Fraction.from_float(f)<frac:f=float(np.nextafter(f,np.inf))
    return f
def f(x):return Fraction.from_float(float(x))


def main():
    model=read(OUT/'CAL/MODEL.json');val=read(OUT/'CAL/VALIDATION.json');train=read(OUT/'TRAIN/RESULTS.json')
    p=OUT/'POSTHOC_QUADRATIC';protocol=read(p/'PROTOCOL.json')
    inputs={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in [OUT/'CAL/MODEL.json',OUT/'CAL/VALIDATION.json',OUT/'TRAIN/RESULTS.json']}
    save(p/'EXECUTION_PROTOCOL.json',dict(source_sha256=sha(__file__),definition_protocol_sha256=sha(p/'PROTOCOL.json'),
        inputs=inputs,new_scientific_inputs=0,original_primary_family_modified=False))
    rho=np.array(model['reference']['rho801']);rfun=PchipInterpolator(rho,model['reference']['r801'])
    case=next(c for c in model['cases'] if c['n']==254)
    gp=PPoly(case['gprime_spline_c'],case['gprime_spline_x'])
    coefficient=float(gp.derivative()(0)/2)
    assert coefficient==protocol['coefficient']
    rows=[]
    for stream in train['records']:
        with np.load(ROOT/stream['file'],allow_pickle=False) as arr:
            raw=arr['raw'].mean(axis=0)
            for ti,theta in enumerate(train['config']['teachers']):
                for ni,N in enumerate(train['config']['sample_sizes']):
                    old=next(x for x in stream['outputs'] if x['theta']==theta and x['N']==N and x['method']=='finitecal')
                    scores=raw[ti,ni]+coefficient*rho*rho;index=int(np.argmin(scores));q=float(rho[index])
                    D=upper(f(stream['base_curvature_D'])+2*abs(f(coefficient)))
                    mesh=upper(f(D)*f(old['mesh_step'])**2/8)
                    R=risk(float(rfun(theta)),float(rfun(q)))
                    rows.append(dict(stream=stream['stream'],theta=theta,N=N,method='finite_taylor_quad',rho=q,grid_index=index,
                        reference_joint_KL=R,full_finite_rho=old['rho'],full_finite_risk=old['reference_joint_KL'],
                        paired_risk_difference=R-old['reference_joint_KL'],same_point=index==old['grid_index'],
                        curvature_D=D,continuous_mesh_bound_real_arithmetic=mesh))
    summary=[]
    for theta in [0.,.3,.6]:
        for N in [1024,4096]:
            x=[r for r in rows if r['theta']==theta and r['N']==N]
            summary.append(dict(theta=theta,N=N,mean_reference_KL=float(np.mean([r['reference_joint_KL'] for r in x])),
                mean_rho=float(np.mean([r['rho'] for r in x])),mean_minus_full_finite=float(np.mean([r['paired_risk_difference'] for r in x])),
                exact_same_grid_point_count=sum(r['same_point'] for r in x)))
    save(p/'RESULTS.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(p/'PROTOCOL.json'),
        execution_sha256=sha(p/'EXECUTION_PROTOCOL.json'),coefficient=coefficient,rows=rows,summary=summary,
        additional_grid_selections=96,new_data_or_random_draws=0,original_primary_family_modified=False,
        interpretation='Exploratory reuse of saved training curves, conditional on same calibration; no new independent confirmation or success gate.'))

    cdir=OUT/'CONDITIONAL_TARGET';cdir.mkdir(exist_ok=True)
    save(cdir/'PROTOCOL.json',dict(source_sha256=sha(__file__),inputs=inputs,posthoc=True,
        rule='pooled independent-validation derivative spline Uprime+t_N Vprime, with/without frozen finite gprime',
        scope='conditional-bank-expectation numerical proxy; not actual empirical objective or strict population certificate',
        candidates='all real derivative roots plus endpoints',tie_tolerance=1e-12,tie_rule='smallest abs rho then rho',
        methods=['raw','finitecal'],target_assignments=192,new_random_draws=0,new_Gaussian_integrals=0,
        no_causal_percentage_decomposition=True))
    cv=next(c for c in val['cases'] if c['n']==254);x=np.array(cv['rho101']);u=np.array(cv['U_prime']);v=np.array(cv['V_prime'])
    cache={};nroots=0
    def select(t,method):
        nonlocal nroots
        key=(t,method)
        if key not in cache:
            derivative=CubicSpline(x,u+t*v+(gp(x) if method=='finitecal' else 0))
            potential=derivative.antiderivative();roots=derivative.roots(extrapolate=False)
            candidates=sorted(set([-.75,.75]+[float(q) for q in roots if np.isfinite(q) and -.75<=q<=.75]))
            vals=np.array([potential(q)-potential(0) for q in candidates]);minimum=float(vals.min())
            tied=[candidates[i] for i,z in enumerate(vals) if z<=minimum+1e-12];q=min(tied,key=lambda q:(abs(q),q))
            cache[key]=dict(rho=q,derivative=float(derivative(q)),candidate_count=len(candidates),tied=tied)
            nroots+=1
        return cache[key]
    conditional=[]
    for stream in train['records']:
        for theta in [0.,.3,.6]:
            for N in [1024,4096]:
                exact=next(r for r in stream['outputs'] if r['theta']==theta and r['N']==N and r['method']=='exact')
                for method in ['raw','finitecal']:
                    chosen=select(exact['empirical_pair'],method)
                    actual=next(r for r in stream['outputs'] if r['theta']==theta and r['N']==N and r['method']==method)
                    conditional.append(dict(stream=stream['stream'],theta=theta,N=N,method=method,t_N=exact['empirical_pair'],
                        proxy=chosen,proxy_reference_KL=risk(float(rfun(theta)),float(rfun(chosen['rho']))),
                        actual_rho=actual['rho'],exact_MLE_rho=exact['rho'],
                        proxy_minus_MLE=chosen['rho']-exact['rho'],actual_minus_proxy=actual['rho']-chosen['rho']))
    condsum=[]
    for theta in [0.,.3,.6]:
        for N in [1024,4096]:
            s=[r for r in conditional if r['theta']==theta and r['N']==N and r['method']=='finitecal']
            condsum.append(dict(theta=theta,N=N,proxy_mean_KL=float(np.mean([r['proxy_reference_KL'] for r in s])),
                mean_abs_proxy_minus_MLE=float(np.mean([abs(r['proxy_minus_MLE']) for r in s])),
                mean_abs_actual_minus_proxy=float(np.mean([abs(r['actual_minus_proxy']) for r in s]))))
    save(cdir/'RESULTS.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(cdir/'PROTOCOL.json'),rows=conditional,
        summary=condsum,assignments=192,unique_derivative_spline_optimizations=nroots,
        validation_streams=4,new_data_or_random_draws=0,new_Gaussian_integrals=0,
        conditional_expectation_proxy_not_real_empirical_training=True,no_causal_percentage_claim=True))
    print(json.dumps(dict(posthoc_quadratic=summary,conditional_proxy=condsum)))


if __name__=='__main__':main()

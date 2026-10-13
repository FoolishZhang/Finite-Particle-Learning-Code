"""Q: unbounded RC LOO convex fitting and exact rational query certificates."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='2'
import argparse
import csv
from fractions import Fraction as R
import hashlib
import itertools
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import rc_quant_certificate as exact

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/rc_quant/Q'
DOC=ROOT/'docs/08_research_研究工作/RC定量界与多块查询_20261001'
OLD=ROOT/'artifacts/v3_select/T'
CFG=dict(d=6,epsilon_numerator=9,epsilon_denominator=10,tree_count=945,
    features='binary C unary and internal count indicators at levels2..size',
    query='uniform six targets, all other five observed',parameter_box=None,
    precision_bits=[100,160],log_terms=96,gradient_gate=1e-10,
    lbfgs=dict(maxiter=3000,ftol=1e-15,gtol=1e-12,maxls=50),
    newton_maxiter=100,backtracks=60,armijo=1e-4,roundoff_eps_multiplier=16,
    damping=1e-12,desired_global_width='1/100000000',numeric_tie_tolerance=1e-12,
    new_population_query_fits=945,new_samples=0,new_RNG_draws=0,
    old_joint_fits_unchanged=945,analytic_query_lower='27/14080')


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,R):return [x.numerator,x.denominator]
    raise TypeError(type(x).__name__)
def write(p,x,replace=False):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w' if replace else 'x',encoding='utf-8',newline='\n') as f:
        json.dump(x,f,default=serial,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def hashes(paths):return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
def verify(mapping):
    for path,value in mapping.items():assert sha(ROOT/path)==value,path
def enc(x):return [R(x).numerator,R(x).denominator]
def dec(x):return R(*x)
def pack_vector(v):
    denominator=math.lcm(*(x.denominator for x in v))
    return dict(denominator=denominator,numerators=[int(x*denominator) for x in v])
def unpack_vector(v):return [R(x,v['denominator']) for x in v['numerators']]
def csvwrite(path,rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def canonical_nodes(d,nodes):
    nodes=tuple(sorted((tuple(sorted(n)) for n in nodes),key=lambda n:(len(n),n)))
    assert len(nodes)==len(set(nodes)) and tuple(range(d)) in nodes
    for node in nodes:
        assert len(node)>=2 and len(node)==len(set(node)) and min(node)>=0 and max(node)<d
        for other in nodes:
            a,b=set(node),set(other);assert not a&b or a<=b or b<=a
    return nodes


def design(d,nodes,numerators=None):
    nodes=canonical_nodes(d,nodes)
    states=((np.arange(1<<d)[:,None]>>np.arange(d))&1).astype(np.int64)
    features=[states]
    for node in nodes:
        count=states[:,node].sum(1)
        features.append((count[:,None]==np.arange(2,len(node)+1)).astype(np.int64))
    F=np.column_stack(features)
    if numerators is None:
        assert d==6
        h=np.prod(states[:,::2]-states[:,1::2],axis=1);numerators=(10+9*h).tolist()
    assert len(numerators)==1<<d and min(numerators)>0
    n=np.array(numerators,dtype=np.int64);denominator=int(n.sum())*d
    events=[];rows=[];n0=[];n1=[];summary=[]
    for j in range(d):
        others=[i for i in range(d) if i!=j]
        prior={j};siblings=[]
        for node in sorted((set(s) for s in nodes if j in s),key=len):
            siblings.append(tuple(sorted(node-prior)));prior=node
        assert prior==set(range(d)) and sorted(i for block in siblings for i in block)==others
        for code in range(1<<(d-1)):
            c0=np.zeros(d,dtype=np.int64)
            c0[others]=[(code>>k)&1 for k in range(d-1)]
            idx0=int(c0@(1<<np.arange(d)));idx1=idx0+(1<<j)
            rows.append(F[idx1]-F[idx0]);n0.append(int(n[idx0]));n1.append(int(n[idx1]))
            events.append((j,idx0,idx1));summary.append((j,)+tuple(int(c0[list(block)].sum()) for block in siblings))
    A=np.asarray(rows,dtype=np.int64);n0=np.array(n0,dtype=np.int64);n1=np.array(n1,dtype=np.int64);m=n0+n1
    assert int(m.sum())==denominator
    w=m/denominator;p=n1/m
    entropy=float(-np.sum(w*(p*np.log(p)+(1-p)*np.log1p(-p))))
    return dict(d=d,nodes=nodes,states=states,F=F,A=A,n0=n0,n1=n1,m=m,w=w,p=p,
        denominator=denominator,events=events,summary=summary,H_cond=entropy)


def objective(data,theta,hessian=False):
    A=data['A'];u=A@theta;s=expit(u)
    value=float(np.sum(data['w']*(np.logaddexp(0.,u)-data['p']*u)))
    gradient=A.T@(data['w']*(s-data['p']))
    if not hessian:return value,gradient
    H=A.T@((data['w']*s*(1-s))[:,None]*A)
    return value,gradient,.5*(H+H.T)


def fit_query(data):
    started=time.perf_counter();calls=0;hessians=0
    def call(theta,hessian=False):
        nonlocal calls,hessians
        calls+=1;hessians+=int(hessian);out=objective(data,theta,hessian)
        assert all(np.isfinite(x).all() for x in out)
        return out
    lbfgs=minimize(call,np.zeros(data['A'].shape[1]),jac=True,method='L-BFGS-B',bounds=None,options=CFG['lbfgs'])
    theta=np.asarray(lbfgs.x).copy();assert np.isfinite(theta).all();trace=[]
    for iteration in range(CFG['newton_maxiter']):
        value,g,H=call(theta,True);norm=float(np.max(abs(g)))
        if norm<=CFG['gradient_gate']:break
        damping=max(0.,CFG['damping']*max(1.,float(np.max(np.diag(H))))-float(np.linalg.eigvalsh(H)[0]))
        kind='damped_Newton'
        try:direction=np.linalg.solve(H+damping*np.eye(len(theta)),-g)
        except np.linalg.LinAlgError:direction=-g;kind='gradient_fallback'
        if not np.isfinite(direction).all() or g@direction>=0:direction=-g;kind='gradient_fallback'
        step=1.;accepted=False;relaxed=False
        for backtrack in range(CFG['backtracks']):
            candidate=theta+step*direction;fv,fg=call(candidate)
            armijo=value+CFG['armijo']*float(g@(candidate-theta))
            slack=CFG['roundoff_eps_multiplier']*np.finfo(float).eps*max(1.,abs(value),abs(fv))
            strict=fv<=armijo
            relaxed=bool(not strict and fv<=armijo+slack and np.max(abs(fg))<norm)
            if strict or relaxed:accepted=True;break
            step*=.5
        trace.append(dict(iteration=iteration,objective=value,max_gradient=norm,damping=damping,
            direction=kind,step=step,backtracks=backtrack,accepted=accepted,roundoff_relaxed=relaxed,roundoff_slack=slack))
        if not accepted or np.array_equal(candidate,theta):break
        theta=candidate
    value,g=call(theta)
    return dict(theta=theta,conditional_NLL=value,query_KL=value-data['H_cond'],gradient=g,
        max_gradient=float(np.max(abs(g))),numeric_qualified=bool(np.max(abs(g))<=CFG['gradient_gate']),
        parameter_count=len(theta),lbfgs_iterations=int(lbfgs.nit),lbfgs_success=bool(lbfgs.success),
        lbfgs_message=str(lbfgs.message),Newton_trace=trace,objective_calls=calls,hessian_calls=hessians,
        elapsed_seconds=time.perf_counter()-started,parameter_box=None,gradient_is_not_a_rigorous_gap=True)


def log_linear_interval(terms,bits):
    """Exact signed linear combination of exact outward log intervals."""
    lo=hi=R(0)
    for coefficient,x in terms:
        coefficient=R(coefficient);x=R(x)
        if coefficient==0:continue
        assert x>0
        lower,upper=exact.rational_log_bounds(x,bits)
        if coefficient>=0:lo+=coefficient*lower;hi+=coefficient*upper
        else:lo+=coefficient*upper;hi+=coefficient*lower
    assert lo<=hi
    return lo,hi


def verify_dual(data,v):
    assert len(v)==len(data['m']) and all(0<a<int(m) for a,m in zip(v,data['m']))
    for row in data['A'].T.tolist():
        assert sum((R(a)*x for a,x in zip(row,v)),R(0))==sum(int(a)*int(b) for a,b in zip(row,data['n1']))


def dual_interval(data,v,bits):
    verify_dual(data,v);terms=[]
    for a,b,m,x in zip(data['n0'],data['n1'],data['m'],v):
        terms.extend([(int(a),int(a)),(int(b),int(b)),(-x,x),(-(R(int(m))-x),R(int(m))-x)])
    lo,hi=log_linear_interval(terms,bits);return lo/data['denominator'],hi/data['denominator']


def primal_probabilities(data,weight_numerators,bits):
    assert len(weight_numerators)==data['A'].shape[1] and all(isinstance(x,int) and x>0 for x in weight_numerators)
    weights=[R(x,1<<bits) for x in weight_numerators];out=[]
    for row in data['A']:
        odds=R(1)
        for delta,w in zip(row,weights):
            if delta:odds*=w**int(delta)
        out.append(odds/(1+odds))
    return out


def primal_interval(data,weight_numerators,bits):
    probabilities=primal_probabilities(data,weight_numerators,bits);terms=[]
    for a,b,m,p in zip(data['n0'],data['n1'],data['m'],probabilities):
        terms.extend([(int(a),R(int(a),int(m))/(1-p)),(int(b),R(int(b),int(m))/p)])
    lo,hi=log_linear_interval(terms,bits);return lo/data['denominator'],hi/data['denominator']


def summary_dual(data):
    totals={}
    for key,a,m in zip(data['summary'],data['n1'],data['m']):
        old=totals.get(key,(0,0));totals[key]=(old[0]+int(a),old[1]+int(m))
    v=[R(int(m)*totals[key][0],totals[key][1]) for key,m in zip(data['summary'],data['m'])]
    verify_dual(data,v)
    return v


def certificate(data,fit,tree_id,bits):
    start=time.perf_counter();A=data['A'].T.tolist()
    target=[sum(int(a)*int(b) for a,b in zip(row,data['n1'])) for row in A]
    proposal=data['m']*expit(data['A']@np.asarray(fit['theta']))
    v,metadata=exact.dual_feasible_projection(A,target,proposal.tolist(),pbits=bits)
    verify_dual(data,v)
    scale=1<<bits
    weights=[int(round(R.from_float(math.exp(float(x)))*scale)) for x in fit['theta']]
    assert min(weights)>0
    lower=dual_interval(data,v,bits);upper=primal_interval(data,weights,bits)
    sv=summary_dual(data);summary=dual_interval(data,sv,bits)
    assert lower[0]<=upper[1] and summary[0]<=upper[1]
    return dict(tree_id=tree_id,d=data['d'],nodes=data['nodes'],bits=bits,log_terms=exact.LOG_TERMS,
        event_order='target increasing, remaining-bit binary code increasing',event_denominator=data['denominator'],
        dual_eventmass=pack_vector(v),dual_basis=metadata,dual_risk_interval=[enc(x) for x in lower],
        primal_weight_numerators=weights,primal_risk_interval=[enc(x) for x in upper],
        summary_eventmass=pack_vector(sv),summary_CMI_interval=[enc(x) for x in summary],
        exact_dual_equations=True,strictly_interior_eventmass=True,unbounded_parameter_class=True,
        exp_is_untrusted_proposal_only=True,elapsed_seconds=time.perf_counter()-start)


def verify_certificate(cert,data):
    bits=cert['bits'];assert bits in CFG['precision_bits'] and cert['log_terms']==96
    assert cert['event_denominator']==data['denominator'] and canonical_nodes(data['d'],cert['nodes'])==data['nodes']
    v=unpack_vector(cert['dual_eventmass']);verify_dual(data,v)
    sv=unpack_vector(cert['summary_eventmass']);assert sv==summary_dual(data)
    lower=dual_interval(data,v,bits);upper=primal_interval(data,cert['primal_weight_numerators'],bits);summary=dual_interval(data,sv,bits)
    assert [enc(x) for x in lower]==cert['dual_risk_interval']
    assert [enc(x) for x in upper]==cert['primal_risk_interval']
    assert [enc(x) for x in summary]==cert['summary_CMI_interval']
    assert lower[0]<=upper[1] and summary[0]<=upper[1]
    return lower,upper,summary


def toyqa():
    """No optimizer call; exact d2 model with integer positive weights 2,3,5."""
    data=design(2,[(0,1)],[1,2,3,30]);theta=np.log([2.,3.,5.])
    value,g,H=objective(data,theta,True);assert np.max(abs(g))<1e-14 and np.linalg.eigvalsh(H)[0]>0
    query_probs=primal_probabilities(data,[2*(1<<100),3*(1<<100),5*(1<<100)],100)
    assert all(q==R(int(n),int(m)) for q,n,m in zip(query_probs,data['n1'],data['m']))
    checks=[]
    for bits in CFG['precision_bits']:
        cert=certificate(data,dict(theta=theta),-1,bits);lo,up,su=verify_certificate(cert,data)
        assert lo[0]<=0<=up[1] and su[0]<=0<=su[1] and up[1]-lo[0]<R(1,10**20)
        checks.append(dict(bits=bits,certificate_bracket=[enc(lo[0]),enc(up[1])]))
    return dict(passed=True,gradient_at_known_solution=float(np.max(abs(g))),certificates=checks,
        optimizer_calls=0,scientific_fits=0,new_samples=0,positive_rational_model_constructions=2)


def registry():
    trees=read(OLD/'TREE_REGISTRY.json');assert len(trees)==945
    seen=set()
    for i,spec in enumerate(trees):
        assert spec['tree_id']==i
        nodes=canonical_nodes(6,spec['nodes']);assert len(nodes)==5 and nodes not in seen;seen.add(nodes)
        allsets=[set(s) for s in nodes]+[{j} for j in range(6)]
        for node in nodes:
            s=set(node);strict=[t for t in allsets if t<s]
            children=[t for t in strict if not any(t<u<s for u in strict)]
            assert len(children)==2 and not children[0]&children[1] and children[0]|children[1]==s
    return trees


def freeze():
    assert not OUT.exists()
    oldfreeze=read(OLD/'FREEZE.json');assert sha(OLD/'TREE_REGISTRY.json')==oldfreeze['inputs']['artifacts/v3_select/T/TREE_REGISTRY.json']
    assert sha(OLD/'TREE_REGISTRY.json')==read(OLD/'RESULTS_FREEZE.json')['files']['artifacts/v3_select/T/TREE_REGISTRY.json']
    assert exact.LOG_TERMS==96 and tuple(exact.LEVELS)==(100,160)
    registry();OUT.mkdir(parents=True)
    write(OUT/'CONFIG.json',CFG);write(OUT/'ENVIRONMENT.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),threads=2))
    (OUT/'HANDOFF_INPUT.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    paths=[Path(__file__).resolve(),Path(exact.__file__).resolve(),DOC/'PLAN_Q.md',DOC/'QUERY_DUAL.md',OLD/'TREE_REGISTRY.json',OLD/'FREEZE.json',OLD/'RESULTS_FREEZE.json',OUT/'CONFIG.json',OUT/'ENVIRONMENT.json',OUT/'HANDOFF_INPUT.md']
    write(OUT/'FREEZE.json',dict(time_utc=now(),files=hashes(paths),before_new_query_fits=True,scientific_fits_completed=0))
    write(OUT/'STATE.json',dict(status='frozen',time_utc=now()))
    print(json.dumps(dict(frozen=True,source_sha256=sha(__file__),shared_library_sha256=sha(exact.__file__),freeze_sha256=sha(OUT/'FREEZE.json'))),flush=True)


def checkfreeze():
    assert read(OUT/'CONFIG.json')==CFG;verify(read(OUT/'FREEZE.json')['files'])


def aggregate():
    specs=registry();fits=[read(OUT/'fits'/f'tree_{i:03}.json') for i in range(945)]
    minimum=min(f['query_KL'] for f in fits);numeric_tree=next(i for i,f in enumerate(fits) if f['query_KL']<=minimum+CFG['numeric_tie_tolerance'])
    rows=[];levels={}
    for bits in CFG['precision_bits']:
        certs=[read(OUT/'certificates'/f'tree_{i:03}_p{bits}.json') for i in range(945)]
        lower=[dec(c['dual_risk_interval'][0]) for c in certs];upper=[dec(c['primal_risk_interval'][1]) for c in certs]
        sl=[dec(c['summary_CMI_interval'][0]) for c in certs];su=[dec(c['summary_CMI_interval'][1]) for c in certs]
        best_upper=min(range(945),key=lambda i:upper[i]);best_lower=min(range(945),key=lambda i:lower[i])
        low=min(lower);high=min(upper);assert low<=high
        levels[str(bits)]=dict(global_lower=enc(low),global_upper=enc(high),width=enc(high-low),
            desired_width_pass=bool(high-low<=R(CFG['desired_global_width'])),lowest_dual_tree=best_lower,best_primal_tree=best_upper,
            global_summary_min_interval=[enc(min(sl)),enc(min(su))],
            summary_at_numeric_query_best_tree=certs[numeric_tree]['summary_CMI_interval'],
            summary_at_best_primal_tree=certs[best_upper]['summary_CMI_interval'],
            numeric_best_tree=numeric_tree,summary_minima_not_causal_attribution=True)
        for i,(fit,cert) in enumerate(zip(fits,certs)):
            rows.append(dict(tree_id=i,bits=bits,parameter_count=fit['parameter_count'],numeric_query_KL=fit['query_KL'],
                max_gradient=fit['max_gradient'],numeric_qualified=fit['numeric_qualified'],
                rigorous_query_lower=float(lower[i]),rigorous_query_upper=float(upper[i]),rigorous_width=float(upper[i]-lower[i]),
                summary_CMI_lower=float(sl[i]),summary_CMI_upper=float(su[i]),
                fit_seconds=fit['elapsed_seconds'],certificate_seconds=cert['elapsed_seconds']))
    csvwrite(OUT/'TREE_RESULTS.csv',rows)
    low=max(dec(v['global_lower']) for v in levels.values());high=min(dec(v['global_upper']) for v in levels.values());assert low<=high
    summary=dict(new_population_query_fits=945,old_joint_fits=945,new_samples=0,new_RNG_draws=0,
        numeric_qualified=sum(f['numeric_qualified'] for f in fits),numeric_failed_tree_ids=[i for i,f in enumerate(fits) if not f['numeric_qualified']],
        numeric_minimum_query_KL=minimum,numeric_best_representative_tree=numeric_tree,numeric_tie_tolerance=CFG['numeric_tie_tolerance'],
        precision_levels=levels,intersection_global_interval=[enc(low),enc(high)],intersection_width=enc(high-low),
        analytic_query_lower=enc(R(27,14080)),all_precisions_width_pass=all(v['desired_width_pass'] for v in levels.values()),
        rigorous_dual_certificates=1890,rational_primal_certificates=1890,summary_certificates=1890,
        full_unbounded_class=True,external_review_number_not_input=True)
    write(OUT/'SUMMARY.json',summary);return summary


def audit():
    checkfreeze();levels={str(bits):dict(lower=[],upper=[],summary_lower=[],summary_upper=[]) for bits in CFG['precision_bits']}
    for spec in registry():
        i=spec['tree_id'];data=design(6,spec['nodes']);fit=read(OUT/'fits'/f'tree_{i:03}.json')
        value,g=objective(data,np.asarray(fit['theta']));assert abs(value-fit['conditional_NLL'])<1e-13 and np.max(abs(g-np.asarray(fit['gradient'])))<1e-13
        for bits in CFG['precision_bits']:
            cert=read(OUT/'certificates'/f'tree_{i:03}_p{bits}.json');lo,up,su=verify_certificate(cert,data)
            level=levels[str(bits)];level['lower'].append(lo[0]);level['upper'].append(up[1]);level['summary_lower'].append(su[0]);level['summary_upper'].append(su[1])
    summary=read(OUT/'SUMMARY.json')
    for bits,level in levels.items():
        saved=summary['precision_levels'][bits]
        assert dec(saved['global_lower'])==min(level['lower']) and dec(saved['global_upper'])==min(level['upper'])
        assert [dec(v) for v in saved['global_summary_min_interval']]==[min(level['summary_lower']),min(level['summary_upper'])]
    return dict(passed=True,time_utc=now(),exact_dual_and_primal_certificates_replayed=1890,summary_certificates_replayed=1890,
        fits_reoptimized=0,new_samples=0,new_RNG_draws=0,strict_integer_constraints_and_log_bounds=True,
        scope='stored exact witnesses and log bounds; objective arithmetic replay; producer verifier and shared library reused',timing_not_replayed=True)


def run():
    checkfreeze();assert read(OUT/'STATE.json')['status']=='frozen';write(OUT/'TOY_QA.json',toyqa())
    started=now();write(OUT/'START.json',dict(first_new_population_query_fit_utc=started,freeze_sha256=sha(OUT/'FREEZE.json')))
    for spec in registry():
        i=spec['tree_id'];write(OUT/'STATE.json',dict(status='running',tree_id=i,time_utc=now(),pid=os.getpid()),replace=True)
        data=design(6,spec['nodes']);fit=fit_query(data);fit.update(tree_id=i,nodes=spec['nodes'],source_sha256=sha(__file__))
        write(OUT/'fits'/f'tree_{i:03}.json',fit)
        for bits in CFG['precision_bits']:
            cert=certificate(data,fit,i,bits);write(OUT/'certificates'/f'tree_{i:03}_p{bits}.json',cert)
        if (i+1)%25==0 or i==944:print(json.dumps(dict(completed_query_fits=i+1,max_gradient=fit['max_gradient'])),flush=True)
    summary=aggregate();write(OUT/'AUDIT.json',audit())
    paths=list((OUT/'fits').glob('*.json'))+list((OUT/'certificates').glob('*.json'))+[OUT/name for name in ['TOY_QA.json','START.json','TREE_RESULTS.csv','SUMMARY.json','AUDIT.json']]
    write(OUT/'RESULTS_FREEZE.json',dict(time_utc=now(),source_freeze_sha256=sha(OUT/'FREEZE.json'),files=hashes(paths)))
    write(OUT/'STATE.json',dict(status='complete',time_utc=now(),active_jobs=[]),replace=True)
    print(json.dumps(dict(complete=True,numeric_qualified=summary['numeric_qualified'],global_interval=summary['intersection_global_interval'],all_precisions_width_pass=summary['all_precisions_width_pass'])),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['toyqa','freeze','run','check']);args=parser.parse_args()
    try:
        if args.phase=='toyqa':print(json.dumps(toyqa(),default=serial))
        elif args.phase=='freeze':freeze()
        elif args.phase=='run':run()
        else:
            verify(read(OUT/'RESULTS_FREEZE.json')['files']);assert read(OUT/'STATE.json')['status']=='complete';print(json.dumps(audit()))
    except Exception as error:
        if args.phase=='run' and (OUT/'FREEZE.json').exists() and not (OUT/'RESULTS_FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=now(),error=repr(error),traceback=traceback.format_exc(),state=read(OUT/'STATE.json')))
            write(OUT/'STATE.json',dict(status='stopped_failure',time_utc=now(),active_jobs=[]),replace=True)
        raise


if __name__=='__main__':main()

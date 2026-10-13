"""Two fixed count-class replays using cosh/tanh and direct contractions."""
import argparse
from fractions import Fraction
import time
import certified_teacher_compute as common
import certified_f8_gradient_compute as producer
from flint import arb,ctx

OUT=producer.OUT
SELECTED=((0,0,0,0),(1,1,1,1))
SOURCES=('script/certified_f8_gradient_audit.py','script/certified_f8_gradient_audit_test.py')

class AuditBudgetStop(RuntimeError):
    def __init__(self,partial):
        super().__init__('fixed 600-second independent replay budget exhausted')
        self.partial=partial

def direct_representative(a,c,hz,L,hv,lo,hi,K=8,*,deadline=None):
    """Different likelihood/score algebra; no producer evaluation function."""
    a=tuple(map(common.exact,a));hz,L,hv,lo,hi=map(common.exact,(hz,L,hv,lo,hi));c=tuple(c)
    if len(a)!=len(c) or any(type(x) is not int or x not in (0,1,2) for x in c):raise ValueError('counts')
    if min(hz,L,hv)<=0 or lo>=hi or any(q.denominator!=1 for q in (L/hz,lo/hv,hi/hv)):raise ValueError('grid')
    if type(K) is not int or K<1:raise ValueError('K')
    p=len(a);aa=list(map(common.aq,a));nz=(L/hz).numerator
    z=[common.aq(hz)*i for i in range(-nz,nz+1)]
    w=[common.aq(hz)*(-x*x/2).exp()/(2*arb.pi()).sqrt() for x in z]
    slope=sum((common.aq(aj)*(cj-1) for aj,cj in zip(a,c)),arb(0))
    A=[];G=[]
    for x in z:
        denominator=arb(1)
        for aj in aa:denominator*=(2*(aj*x/2).cosh())**2
        A.append((slope*x).exp()/denominator)
        G.append([x*(cj-1-(aj*x/2).tanh())/arb(2).sqrt() for aj,cj in zip(aa,c)])
    ecap,scap=arb(2)**-184,arb(2)**-177
    if not (-arb(128)).exp()<ecap or not 128*(-arb(128)).exp()<scap:raise ArithmeticError('cap')
    eball,sball=arb(0).union(ecap),arb(0).union(scap)
    total=[arb(0) for _ in a];completed=0;saturated=0;factors=0
    def output(complete):
        return dict(completed=complete,pair_counts=list(c),gradient_finite=[common.record(q) for q in total],
            inner_nodes=len(z),outer_nodes_completed=completed,factors=factors,saturated_factors=saturated,
            implementation='cosh likelihood, tanh score, direct scalar contractions',new_random_draws=0)
    for vi in range((lo/hv).numerator,(hi/hv).numerator+1):
        if deadline is not None and time.perf_counter()>=deadline:raise AuditBudgetStop(output(False))
        u=(common.aq(hv)*vi).exp();psi=arb(0);D=[arb(0) for _ in a]
        for likelihood,weight,score in zip(A,w,G):
            t=u*likelihood
            if t.lower()>=128:e,te=eball,sball;saturated+=1
            else:e=(-t).exp();te=t*e
            psi+=weight*e
            for j in range(p):D[j]+=weight*te*score[j]
            factors+=1
        prefactor=-K*common.aq(hv)*psi**(K-1)
        for j in range(p):total[j]+=prefactor*D[j]
        completed+=1
    return output(True)

def compare(row,saved):
    previous=ctx.prec;ctx.prec=192
    try:
        if row['pair_counts']!=saved['pair_counts']:raise ValueError('pair count mismatch')
        pairs=zip(row['gradient_finite'],saved['conditional_gradient_finite'])
        result=[]
        for a,b in pairs:
            av,bv=common.from_record(a),common.from_record(b)
            if not av.overlaps(bv):raise ValueError('independent finite gradient enclosures do not overlap')
            result.append(dict(overlap=True,difference=common.record(av-bv)))
        if len(result)!=len(row['pair_counts']):raise ValueError('gradient shape')
        return result
    finally:ctx.prec=previous

def bindings():
    paths=(*SOURCES,'artifacts/certified_f8_gradient_v1_20261008/FREEZE.json','artifacts/certified_f8_gradient_v1_20261008/FINITE_GRADIENTS.json')
    return {name:common.sha(common.ROOT/name) for name in paths}

def run():
    inputs=producer.verify_freeze()
    if isinstance(inputs,tuple):inputs=inputs[0]
    freeze=common.read(producer.FREEZE)
    for name in SOURCES:
        if freeze['sources'].get(name)!=common.sha(common.ROOT/name):raise ValueError('audit not frozen')
    common.write_new(OUT/'AUDIT_START.json',dict(created_utc=common.now(),bindings=bindings(),maximum_seconds=600,selected_counts=SELECTED))
    tick=time.perf_counter();previous=ctx.prec;ctx.prec=192;results=[]
    try:
        source=common.read(common.ROOT/producer.NOMINAL)
        a=[Fraction.from_float(x) for x in source['center']['theta'][8:12]]
        summary=common.read(OUT/'FINITE_GRADIENTS.json')
        for count,index in zip(SELECTED,(0,40)):
            try:row=direct_representative(a,count,'1/40','11','1/8','-32','64',deadline=tick+600)
            except AuditBudgetStop as error:
                common.write_new(OUT/'AUDIT_PARTIAL.json',dict(completed_replays=results,partial=error.partial));raise
            entry=summary['representative_files'][index];saved=common.read(common.ROOT/entry['path'])
            row.update(producer_rep=entry,comparisons=compare(row,saved));results.append(row)
        result=dict(schema='certified_f8_gradient_independent_replay_v1',created_utc=common.now(),bindings=bindings(),
            replays=results,all_eight_components_overlap=True,replayed_representatives=2,total_producer_representatives=41,
            wall_seconds=time.perf_counter()-tick,new_factors=sum(x['factors'] for x in results),
            new_random_draws=0,new_optimizer_updates=0,scientific_NPZ_reads=0,
            scope='two preselected representatives, full same grid; not a complete independent replay')
        common.write_new(OUT/'AUDIT.json',result)
        return dict(passed=True,replayed_representatives=2,new_factors=result['new_factors'],wall_seconds=result['wall_seconds'])
    finally:ctx.prec=previous

def check():
    producer.verify_freeze();saved=common.read(OUT/'AUDIT.json')
    if saved['bindings']!=bindings() or saved['replayed_representatives']!=2:raise ValueError('replay bindings')
    for count,row in zip(SELECTED,saved['replays']):
        if row['pair_counts']!=list(count) or row['outer_nodes_completed']!=769 or row['inner_nodes']!=881 or row['factors']!=881*769:raise ValueError('replay grid drift')
        source=common.read(common.ROOT/row['producer_rep']['path'])
        if row['comparisons']!=compare(row,source):raise ValueError('replay comparison drift')
    return dict(passed=True,new_grid_evaluations=0,new_integrals=0,new_random_draws=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    print(common.json.dumps(run() if args.run else check(),indent=2))

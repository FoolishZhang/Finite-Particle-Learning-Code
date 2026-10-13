"""Fixed eight-label independent finite-grid replay; no optimization or RNG."""
from __future__ import annotations
import argparse
from fractions import Fraction
import time
import certified_teacher_compute as common
from flint import arb, ctx

OUT=common.OUT
LABELS=(0,15,85,90,165,170,240,255)
SOURCES=('script/certified_teacher_audit.py','script/certified_teacher_audit_test.py')

class AuditBudgetStop(RuntimeError):
    def __init__(self,partial):
        super().__init__('fixed 300-second audit budget exhausted; partial retained')
        self.partial=partial

def selected_grid(angles,amplitude,h,N,labels,*,deadline=None):
    """Direct signed-logistic products and a combined 2D Gaussian density.

    The producer uses a probability prefix tree and separable Gaussian weights.
    Only endpoint serialization is shared; the integration algebra is separate.
    """
    if isinstance(N,bool) or not isinstance(N,int) or N<1: raise ValueError('positive integer N required')
    angles=tuple(common.exact(a) for a in angles); amplitude=common.exact(amplitude);h=common.exact(h)
    if h<=0 or amplitude<=0 or not angles:raise ValueError('positive model and step')
    m=len(angles);labels=tuple(labels)
    if len(set(labels))!=len(labels) or any(type(y) is not int or not 0<=y<2**m for y in labels):raise ValueError('unique integer labels')
    previous=ctx.prec;ctx.prec=192
    try:
        step=common.aq(h);amp=common.aq(amplitude)
        trig=[(common.aq(x).cos(),common.aq(x).sin()) for x in angles]
        totals={y:arb(0) for y in labels};points=0;last=None
        def output(complete):
            return dict(completed=complete,labels=list(labels),points=points,last_point=last,
                finite_probabilities={str(y):common.record(totals[y]) for y in labels},
                precision_bits=192,full_endpoint_weights=True,
                density='h^2 exp(-(x^2+y^2)/2)/(2*pi)',
                likelihood='direct product of 1/(1+exp(-spin*logit)); no prefix recursion')
        for i in range(-N,N+1):
            x=step*i
            for j in range(-N,N+1):
                if deadline is not None and time.perf_counter()>=deadline:raise AuditBudgetStop(output(False))
                ycoord=step*j
                weight=step*step*(-(x*x+ycoord*ycoord)/2).exp()/(2*arb.pi())
                logits=[amp*(x*c+ycoord*s) for c,s in trig]
                for label in labels:
                    product=arb(1)
                    for index,logit in enumerate(logits):
                        spin=1 if (label>>(m-1-index))&1 else -1
                        product*=1/(1+(-spin*logit).exp())
                    totals[label]+=weight*product
                points+=1;last=[i,j]
        return output(True)
    finally:ctx.prec=previous

def compare(replay,producer):
    previous=ctx.prec;ctx.prec=192
    try:
        comparisons=[]
        for label in replay['labels']:
            a=common.from_record(replay['finite_probabilities'][str(label)])
            b=common.from_record(producer['pmf_finite_sum'][label])
            ok=a.overlaps(b)
            if not ok:raise ValueError('independent finite-sum enclosure mismatch at label '+str(label))
            comparisons.append(dict(label=label,overlapping_enclosures=True,difference=common.record(a-b)))
        return comparisons
    finally:ctx.prec=previous

def bindings():
    paths=(*SOURCES,'artifacts/certified_teacher_v1_20261008/FREEZE.json',
           'artifacts/certified_teacher_v1_20261008/FINITE_GRID_SUM.json')
    return {name:common.sha(common.ROOT/name) for name in paths}

def run():
    inputs=common.verify_freeze()
    # verify_freeze returns inputs; no science is executed by validation.
    if isinstance(inputs,tuple):inputs=inputs[0]
    frozen=common.read(common.FREEZE)
    for name in SOURCES:
        if frozen['sources'].get(name)!=common.sha(common.ROOT/name):raise ValueError('audit source not frozen')
    common.write_new(OUT/'AUDIT_START.json',dict(created_utc=common.now(),bindings=bindings(),
        labels=list(LABELS),grid_points=25921,maximum_wall_seconds=300,new_random_draws=0))
    tick=time.perf_counter()
    try: result=selected_grid(common.PHI,'4/5','1/8',80,LABELS,deadline=tick+300)
    except AuditBudgetStop as error:
        common.write_new(OUT/'AUDIT_PARTIAL.json',error.partial);raise
    result.update(schema='certified_teacher_selected_grid_audit_v1',created_utc=common.now(),bindings=bindings(),
        comparisons=compare(result,common.read(OUT/'FINITE_GRID_SUM.json')),
        wall_seconds=time.perf_counter()-tick,all_eight_replays_passed=True,
        new_teacher_grid_point_evaluations=25921,selected_probability_summands=25921*8,
        new_independent_samples=0,new_random_draws=0,new_optimizer_updates=0,
        audit_scope='eight fixed labels on the full same grid; not all-256 independent replay')
    common.write_new(OUT/'AUDIT.json',result)
    return dict(passed=True,grid_points=result['points'],labels=len(LABELS),wall_seconds=result['wall_seconds'])

def check():
    common.verify_freeze(); saved=common.read(OUT/'AUDIT.json')
    if saved['bindings']!=bindings() or saved['labels']!=list(LABELS) or saved['points']!=25921:raise ValueError('audit binding/count drift')
    if saved['comparisons']!=compare(saved,common.read(OUT/'FINITE_GRID_SUM.json')):raise ValueError('audit comparison drift')
    return dict(passed=True,new_grid_evaluations=0,new_integrals=0,new_random_draws=0,
        scope='saved independent enclosures; no replay performed by check')

if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');a=parser.parse_args()
    print(common.json.dumps(run() if a.run else check(),indent=2))

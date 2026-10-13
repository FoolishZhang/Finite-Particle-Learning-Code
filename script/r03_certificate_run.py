"""Frozen R03 teacher, risk-box and candidate-box execution."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction as F
import argparse,hashlib,json,os,sys,time,traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import flint
from flint import arb,ctx
import r03_candidate_interval as candidate
import r03_teacher_derivative as teacher
import r03_risk_interval as risk
import r03_continuous_compose as synthesis
import r03_development_run as dev
ROOT,OUT,DOC=dev.ROOT,dev.OUT,dev.DOC
CFG=dict(K=8,precision_bits=192,hz='1/32',L='10',dz='1/6',hv='1/4',vmin='-28',vmax='56',dv='1',T='4',saturation_threshold=128,label_mode='complement')
IDS=('h01','h001','h0001')
HS=('1/100','1/1000','1/10000')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):dev.write(p,x)
def now():return datetime.now(timezone.utc).isoformat()
def freeze():
    dev.verify();assert flint.__version__=='0.8.0'
    data=read(OUT/'DEVELOPMENT.json');assert data['all_points_qualified']
    a0=read(ROOT/dev.A)['a_exact'];av=[F(x['numerator'])/F(x['denominator']) for x in a0]
    H=np.array(data['points'][0]['evaluations']['D']['result']['H_ba'])[0,0,8:,8:]
    A=np.linalg.inv(H);A=(A+A.T)/2
    ctx.prec=192
    boxes={}
    for key,h in zip(IDS,HS):
        radii=[F(3,4)*F(h)]+[F(3,20)*F(h)]*7
        boxes[key]=dict(h_exact=h,radii_exact=[str(x) for x in radii],a0_exact=[str(x) for x in av],
            a_input_boxes=[candidate.record(candidate.aq(a-r).union(candidate.aq(a+r))) for a,r in zip(av,radii)])
    inp=dict(schema='r03_box_inputs_v1',baseline=dev.BASE,feedback_sha256=dev.HANDOFF_SHA,
        boxes=boxes,box_order=list(IDS),candidate_config=CFG,
        A_exact=[[str(F.from_float(float(x))) for x in row] for row in A],
        direction=read(ROOT/dev.D),teacher_config=teacher.configuration(),teacher_seconds=120,
        risk_step='1/8',risk_N=88,risk_seconds_per_box=120,candidate_seconds_per_box=900,
        old_gradient_L2_cap_exact='0.0000000007572',old_root_distance_cap_exact='0.00000001805',
        selected_on_development_not_statistically_blinded=True,stability_lower_targets=dict(Hbb='.1',Haa='.02',Hperp_fixed_slice='.005'))
    write(OUT/'BOX_INPUTS.json',inp)
    checks=dict(schema='r03_box_manufactured_checks_v1',candidate=candidate.self_test(),teacher=teacher.self_test(),
                composition=synthesis.manufactured_checks(),risk=risk.self_test(),new_project_target_calls=0)
    write(OUT/'BOX_IMPLEMENTATION_CHECK.json',checks)
    names=[dev.A,dev.D,dev.PAPER,
      'artifacts/certified_f8_hessian_v1_20261008/HESSIAN_AND_WITNESS_CERTIFICATE.json',
      'script/r03_certificate_run.py','script/r03_candidate_interval.py','script/r03_teacher_derivative.py','script/r03_risk_interval.py','script/r03_continuous_compose.py']
    names += [DOC+x for x in ('BOX_CONTRACT.md','CANDIDATE_CERTIFICATE_METHOD.md','TEACHER_DERIVATIVE_METHOD.md','RISK_BOX_METHOD.md','BOX_CERTIFICATE_ARGUMENT.md','CONTINUATION_ARGUMENT.md')]
    names += ['artifacts/r03_teacher_family_v1_20261011/'+x for x in ('DEV_FREEZE.json','DEVELOPMENT.json','BOX_INPUTS.json','BOX_IMPLEMENTATION_CHECK.json','CANDIDATE_BUDGET_DESIGN.json')]
    write(OUT/'BOX_FREEZE.json',dict(schema='r03_box_freeze_v1',frozen_utc=now(),bindings={x:sha(ROOT/x) for x in names},new_project_box_nodes_before_freeze=0))
def verify():
    for name,digest in read(OUT/'BOX_FREEZE.json')['bindings'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed: '+name)
    return read(OUT/'BOX_INPUTS.json')
def marker(name):
    write(OUT/(name+'_START.json'),dict(started_utc=now(),freeze_sha256=sha(OUT/'BOX_FREEZE.json'),executable=sys.executable,python_flint=flint.__version__,automatic_retry=False))
def run_teacher():
    inp=verify();marker('TEACHER');tick=time.perf_counter()
    b=teacher.finite_grid_block('0',config=inp['teacher_config'],deadline=tick+inp['teacher_seconds'])
    write(OUT/'TEACHER_FINITE_BLOCK.json',b)
    if not b['completed']:raise RuntimeError('Teacher budget stopped, partial retained')
    f=teacher.aggregate_blocks([b]);c=teacher.compose(f)
    write(OUT/'TEACHER_FINITE.json',f);write(OUT/'TEACHER_CERTIFICATE.json',c)
    print(json.dumps(dict(teacher_complete=True,seconds=time.perf_counter()-tick,counts=f['work_counts']),indent=2))
def risk_screen(raw,t,b):
    if not raw['composition']['all_Q_positive']:return dict(passed=False,reason='Q positivity unresolved')
    ctx.prec=192;q,p,_=synthesis._teacher(t);h=F(b['h_exact']);a0=list(map(F,b['a0_exact']));r=list(map(F,b['radii_exact']))
    return synthesis._risk(raw['composition'],q,p,synthesis.aq('0.2211615')*synthesis.aq(h)**2,list(zip([a-r for a,r in zip(a0,r)],[a+r for a,r in zip(a0,r)])),synthesis.aq(h))
def run_risk():
    inp=verify();t=read(OUT/'TEACHER_CERTIFICATE.json');marker('RISK')
    screens={}
    for key in inp['box_order']:
        b=inp['boxes'][key];raw=risk.finite_risk(b['a_input_boxes'],inp['direction'],step=inp['risk_step'],N=inp['risk_N'],hard_seconds=inp['risk_seconds_per_box'])
        write(OUT/('RISK_'+key+'.json'),raw)
        if raw['status']!='complete':raise RuntimeError('Risk prefix retained; incomplete box')
        screens[key]=risk_screen(raw,t,b)
        print(json.dumps(dict(box=key,passed=screens[key]['passed'],curvature=screens[key].get('true_curvature')),ensure_ascii=True),flush=True)
    write(OUT/'RISK_SCREENS.json',screens)
def compose_box(key):
    inp=verify();b=inp['boxes'][key];blocks=[read(OUT/key/f'LABELS_{k:03}.json') for k in range(0,128,8)]
    compact={k:b[k] for k in ('h_exact','radii_exact','a0_exact')}
    compact.update(A_exact=inp['A_exact'],precision_bits=192,candidate_analytic_bounds=read(OUT/key/'ANALYTIC_BOUNDS.json'),
        old_gradient_L2_cap_exact=inp['old_gradient_L2_cap_exact'],old_root_distance_cap_exact=inp['old_root_distance_cap_exact'])
    return synthesis.compute(compact,blocks,read(OUT/'TEACHER_CERTIFICATE.json'),read(OUT/('RISK_'+key+'.json'))['composition'])
def run_candidate(key):
    inp=verify();assert key in inp['box_order'];screens=read(OUT/'RISK_SCREENS.json')
    assert screens[key]['passed'],'Risk screen did not qualify this costly candidate box'
    for earlier in inp['box_order'][:inp['box_order'].index(key)]:
        if screens[earlier]['passed']:
            prior=read(OUT/earlier/'CERTIFICATE.json');assert not prior['passed'],'Larger box already passed: stop smaller runs'
    marker('CANDIDATE_'+key);tick=time.perf_counter();budget=inp['candidate_seconds_per_box'];deadline=tick+budget
    cache=candidate.precompute(inp['boxes'][key]['a_input_boxes'],inp['candidate_config'])
    bounds=candidate.analytic_error_bounds(cache['a_enclosures'],inp['candidate_config'],teacher_error_L1_upper='0',teacher_mass_upper='1')
    write(OUT/key/'ANALYTIC_BOUNDS.json',bounds)
    write(OUT/key/'PRECOMPUTE.json',{k:v for k,v in cache.items() if k in ('config','a_input_audit','a_enclosures','signature','moment_columns','scalar_moments_including_Psi','normal_nodes','precompute_logistic_exponentials','precompute_gaussian_exponentials')})
    try:
        for first in range(0,128,8):
            block=candidate.finite_label_block(cache,list(range(first,first+8)),deadline=deadline)
            write(OUT/key/f'LABELS_{first:03}.json',block)
            print(json.dumps(dict(box=key,completed_labels=first+8,seconds=time.perf_counter()-tick)),flush=True)
    except candidate.CandidateBudgetStop as error:
        write(OUT/key/'PARTIAL.json',error.partial)
        write(OUT/key/'STOP.json',dict(reason='budget_stop',seconds=time.perf_counter()-tick,automatic_retry=False));raise
    c=compose_box(key);write(OUT/key/'CERTIFICATE.json',c)
    write(OUT/key/'RUN.json',dict(completed=True,seconds=time.perf_counter()-tick,budget_seconds=budget,new_laplace_factors=128*641*337,precompute_counts=read(OUT/key/'PRECOMPUTE.json')))
    print(json.dumps(dict(box=key,status=c['status'],root=c['contraction']['passed'],stability={k:v['passed'] for k,v in c['stability'].items()},risk=c['risk']['passed']),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('freeze','teacher','risk','check-freeze'):g.add_argument('--'+flag,action='store_true')
    g.add_argument('--candidate',choices=IDS);g.add_argument('--check-box',choices=IDS)
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.teacher:run_teacher()
    elif a.risk:run_risk()
    elif a.candidate:run_candidate(a.candidate)
    elif a.check_box:assert compose_box(a.check_box)==read(OUT/a.check_box/'CERTIFICATE.json');print('Saved box arithmetic unchanged.')
    else:verify();print('Frozen box inputs unchanged.')

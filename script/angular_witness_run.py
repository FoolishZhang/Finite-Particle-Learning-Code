"""One-shot frozen R01 arithmetic; --check never calls a quadrature kernel."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json
import platform
import sys
import traceback
import flint
from flint import arb, ctx
import angular_witness_integrals as pair
import angular_witness_path_audit as audit
from angular_witness_setup import ROOT, PACKAGE, DOC, SOURCES, read, sha, now, write_new, require, verify_bindings

FILES = ['script/angular_witness_setup.py', 'script/angular_witness_run.py',
         'script/angular_witness_integrals.py', 'script/angular_witness_path_audit.py',
         'script/angular_witness_math_check.py'] + [DOC+n for n in
         ('CONTRACT.md','MATH.md','NUMERICAL_BOUNDS.md','AUDIT_METHOD.md')]
PRE = ('SELECTION_FREEZE.json','DIRECTION.json','INPUTS.json','MATH_CHECK.json',
       'HANDOFF_INPUT.md','PARENT_EXECUTION.json','PARENT_STDOUT.txt','PARENT_STDERR.txt',
       'MANUFACTURED_PAIR_TEST.py','MANUFACTURED_PAIR_LOG.txt','MANUFACTURED_PATH_LOG.txt')

def lo(rec):
    return pair.endpoint_fraction(rec['lower_exact_dyadic'])

def hi(rec):
    return pair.endpoint_fraction(rec['upper_exact_dyadic'])

def overlaps(a, b):
    return lo(a) <= hi(b) and lo(b) <= hi(a)

def rational(item):
    return F(int(item['numerator']), int(item['denominator']))

def validate_inputs():
    inp=read(PACKAGE/'INPUTS.json')
    source_freeze=read(PACKAGE/'SELECTION_FREEZE.json')
    # HANDOFF is a mutable user-owned inbox; bind its archived original here.
    verify_bindings({p:s for p,s in source_freeze['bindings'].items() if p != 'HANDOFF.md'})
    require(sha(PACKAGE/'HANDOFF_INPUT.md') == source_freeze['bindings']['HANDOFF.md'], 'Archived feedback')
    require(sha(PACKAGE/'SELECTION_FREEZE.json') == inp['selection_freeze_sha256'], 'Selection freeze')
    direction=read(PACKAGE/'DIRECTION.json')
    require(sha(PACKAGE/'DIRECTION.json') == inp['direction_sha256'], 'Direction binding')
    require(inp['source_paths'] == SOURCES, 'Source mapping')
    a=list(map(F,inp['a_exact'])); u=inp['u_integer']; S=sum(x*x for x in u)
    require(len(a)==len(u)==8 and all(type(x) is int for x in u), 'Exact eight variables')
    require(sum(u)==0 and S>0 and S==inp['S_integer']==direction['S_integer'], 'Unit nonrotation direction')
    require(u==direction['u_integer'] and inp['b_exact']==['0']*8, 'Direction/b identity')
    require(a == list(map(rational,read(ROOT/SOURCES['center'])['a_exact'])), 'Candidate centre')
    full=list(map(rational,read(ROOT/SOURCES['center_identity'])['center_theta_dyadic']))
    require(full == [F(0)]*8+a, 'LocalWitness exact centre')
    for name in ('gradient_inputs','hessian_inputs'):
        first=list(map(F,read(ROOT/SOURCES[name])['a_pairs_exact']))
        require(first+first[::-1] == a, name+' centre')
    require(inp['step_exact']=='1/8' and inp['N']==88 and inp['bits']==192 and inp['strip_exact']=='1' and inp['hard_seconds']==600, 'Fixed numerical parameters')
    require(F(inp['transfer_cap_exact']) == F('7.52685e-6') == 417*F('1.805e-8'), 'Exact transfer cap')
    require(all(0 < x-F('1.805e-8') and x+F('1.805e-8') < 1 for x in a), 'Root row-length envelope')
    root=read(ROOT/SOURCES['root_certificate'])
    require(root['same_exact_center_across_gradient_Hessian_risk'], 'Predecessor same centre')
    witness=root['witness']
    require(witness['true_F8_stationary_existence_certified'] and witness['strict_local_minimum_modulo_rotations_certified'] and all(witness['checks'].values()), 'Predecessor witness')
    require(hi(witness['root_distance_upper']) < F('1.805e-8'), 'Certified root radius')
    teacher=read(ROOT/SOURCES['teacher_certificate'])
    require(hi(teacher['total_L1_error_from_finite_midpoints']) < F(inp['teacher_L1_cap_exact']) == F('4.894e-21'), 'Teacher L1 cap')
    parent=read(PACKAGE/'PARENT_EXECUTION.json')
    require(parent['passed'] and parent['returncode']==0 and parent['target']=='certified_f8_hessian_delivery', 'Parent public check passed')
    require(parent['stdout_sha256']==sha(PACKAGE/'PARENT_STDOUT.txt') and parent['stderr_sha256']==sha(PACKAGE/'PARENT_STDERR.txt'), 'Parent logs')
    require(not parent['new_scientific_grid_run'] and not parent['scientific_array_reads_allowed'], 'Parent check scope')
    require(flint.__version__=='0.8.0' and sys.flags.optimize==0, 'Public backend and assertions')
    return inp,a,u

def freeze():
    inp,a,u=validate_inputs()
    files=FILES+[str((PACKAGE/n).relative_to(ROOT)).replace('\\','/') for n in PRE]
    write_new(PACKAGE/'PRODUCTION_FREEZE.json', dict(schema='angular_production_freeze_v1',
        frozen_utc=now(), bindings={p:sha(ROOT/p) for p in files},
        source_freeze_sha256=sha(PACKAGE/'SELECTION_FREEZE.json'),
        direction_sha256=sha(PACKAGE/'DIRECTION.json'), target_nodes_before_freeze=0,
        environment=dict(executable=sys.executable, python=platform.python_version(),flint=flint.__version__)))

def frozen_inputs():
    lock=read(PACKAGE/'PRODUCTION_FREEZE.json')
    verify_bindings(lock['bindings'])
    return validate_inputs()

def run(kind):
    inp,a,u=frozen_inputs()
    marker=PACKAGE/(kind+'_START.json')
    write_new(marker, dict(schema='angular_run_start_v1', kind=kind, started_utc=now(),
        freeze_sha256=sha(PACKAGE/'PRODUCTION_FREEZE.json'), input_sha256=sha(PACKAGE/'INPUTS.json'),
        executable=sys.executable, python=platform.python_version(),flint=flint.__version__,
        attempt=1, automatic_retry=False))
    try:
        function=pair.finite_pair_sum if kind=='PAIR' else audit.finite_path_audit
        data=function(a,u,step=inp['step_exact'],N=inp['N'],bits=inp['bits'],hard_seconds=inp['hard_seconds'])
        data['run_start_sha256']=sha(marker)
        data['input_sha256']=sha(PACKAGE/'INPUTS.json')
        data['production_freeze_sha256']=sha(PACKAGE/'PRODUCTION_FREEZE.json')
        data['finished_utc']=now()
        write_new(PACKAGE/(kind+'_FINITE.json'),data)
        print(json.dumps({k:data[k] for k in ('status','nodes_completed','wall_seconds','finite_scalar_integrals_completed')}))
        require(data['status']=='completed', kind+' budget stopped; retained prefix, no retry')
    except Exception:
        write_new(PACKAGE/(kind+'_FAILURE.json'),dict(failed_utc=now(),traceback=traceback.format_exc(),automatic_retry=False))
        raise

def validate_finite(data, kind, inp):
    require(data['status']=='completed' and data['nodes_completed']==data['nodes_prescribed']==177, kind+' completion')
    require(data['precision_bits']==192 and data['step_exact']=='1/8' and data['N']==88 and data['endpoint_weights']=='full', kind+' parameters')
    require(data['hard_budget_seconds']==600 and 0<=data['wall_seconds']<=600, kind+' budget')
    require(data['label_count']==256 and data['m']==8, kind+' labels')
    require(data['run_start_sha256']==sha(PACKAGE/(kind+'_START.json')), kind+' start binding')
    require(data['input_sha256']==sha(PACKAGE/'INPUTS.json') and data['production_freeze_sha256']==sha(PACKAGE/'PRODUCTION_FREEZE.json'), kind+' frozen identity')
    marker=read(PACKAGE/(kind+'_START.json'))
    require(marker['freeze_sha256']==sha(PACKAGE/'PRODUCTION_FREEZE.json') and marker['input_sha256']==sha(PACKAGE/'INPUTS.json'), kind+' marker')
    require(data['real_sigmoid_evaluations']==1416 and data['real_Gaussian_density_evaluations']==177, kind+' actual node work')
    require(all(data[k]==0 for k in ('new_random_draws','new_optimizer_updates','scientific_NPZ_reads')), kind+' scope')
    if kind=='PAIR':
        require(list(map(rational,data['a_exact']))==list(map(F,inp['a_exact'])), 'Primary a identity')
        require(list(map(int,data['u_exact']))==inp['u_integer'] and int(data['S_exact'])==inp['S_integer'], 'Primary direction')
        require(data['node_indices_completed']==list(range(-88,89)), 'Primary node indices')
        require(data['finite_scalar_integrals_completed']==256 and data['new_Q_denominator_integrals']==0, 'Primary integral count')
    else:
        require(data['u_integer']==inp['u_integer'] and data['S_integer']==inp['S_integer'], 'Audit direction')
        require(data['path_acceleration_included'] and data['finite_scalar_integrals_completed']==512, 'Audit path and count')

def compose():
    inp,a,u=frozen_inputs()
    ctx.prec=192
    p=read(PACKAGE/'PAIR_FINITE.json'); v=read(PACKAGE/'AUDIT_FINITE.json')
    validate_finite(p,'PAIR',inp); validate_finite(v,'AUDIT',inp)
    pb=pair.pair_bounds(a,u); vb=audit.path_bounds(a,u)
    qrecords=read(ROOT/SOURCES['teacher_finite'])['pmf_finite_sum']
    Qrecords=read(ROOT/SOURCES['Q_certificate'])['candidate_Q_true_enclosures']
    oldQfinite=read(ROOT/SOURCES['center'])['finite_sum']['Q_finite_sum']
    require(all(len(z)==256 for z in (qrecords,Qrecords,oldQfinite,p['N_finite_sum'],v['N_path_finite_sum'],v['Q_finite_sum'])), 'Full labels')
    nerror=pair.from_record(pb['errors']['N_total'])
    verror=pair.from_record(vb['errors']['N_path_total'])
    qerror=pair.from_record(vb['errors']['Q_total'])
    total=arb(0); audit_total=arb(0); rows=[]
    finite_mass=arb(0); audit_mass=arb(0)
    for i in range(256):
        qhat=(lo(qrecords[i])+hi(qrecords[i]))/2
        require(qhat>=0 and lo(Qrecords[i])>0, 'Positive qhat/Q')
        Q=pair.from_record(Qrecords[i])
        n=pair.enlarge(pair.from_record(p['N_finite_sum'][i]),nerror)
        nv=pair.enlarge(pair.from_record(v['N_path_finite_sum'][i]),verror)
        Qv=pair.enlarge(pair.from_record(v['Q_finite_sum'][i]),qerror)
        nr=pair.record(n); vr=pair.record(nv); qr=pair.record(Qv)
        flags=dict(numerator_true_interval_overlap=overlaps(nr,vr),
            denominator_true_interval_overlap=overlaps(qr,Qrecords[i]),
            denominator_same_finite_sum_overlap=overlaps(v['Q_finite_sum'][i],oldQfinite[i]))
        require(all(flags.values()), 'Independent expression disagreement at label '+str(i))
        ratio=n/Q; audit_ratio=nv/Q
        total+=pair.aq(qhat)*ratio; audit_total+=pair.aq(qhat)*audit_ratio
        finite_mass+=pair.from_record(p['N_finite_sum'][i])
        audit_mass+=pair.from_record(v['N_path_finite_sum'][i])
        rows.append(dict(label=i,teacher_midpoint_exact=str(qhat),N_true=nr,N_path_true=vr,
            audit_Q_true=qr,Q_source_label=i,ratio=pair.record(ratio),audit_ratio=pair.record(audit_ratio),**flags))
    require(finite_mass.contains(0) and audit_mass.contains(0), 'All-label finite numerator mass zero')
    # For the TRUE ratio N/Q=E_pi P, |P|<=8; this is independent of interval division width.
    teacher_allowance=8*F(inp['teacher_L1_cap_exact'])
    center=pair.enlarge(total,teacher_allowance)
    audit_center=pair.enlarge(audit_total,teacher_allowance)
    root=pair.enlarge(center,F(inp['transfer_cap_exact']))
    audit_root=pair.enlarge(audit_center,F(inp['transfer_cap_exact']))
    summary=dict(center=pair.record(center),audit_center=pair.record(audit_center),
        root=pair.record(root),audit_root=pair.record(audit_root))
    checks=dict(all_256_numerator_intervals_overlap=True,all_256_denominator_intervals_overlap=True,
        all_256_same_denominator_finite_sums_overlap=True,finite_numerator_masses_contain_zero=True,
        center_compositions_overlap=overlaps(summary['center'],summary['audit_center']),
        primary_root_strictly_negative=hi(summary['root'])<0,
        audit_root_strictly_negative=hi(summary['audit_root'])<0,
        primary_center_width_below_gate=hi(summary['center'])-lo(summary['center'])<F(inp['center_width_gate_exact']),
        audit_center_width_below_gate=hi(summary['audit_center'])-lo(summary['audit_center'])<F(inp['center_width_gate_exact']))
    return dict(schema='angular_same_root_certificate_v1',precision_bits=192,
        input_sha256=sha(PACKAGE/'INPUTS.json'),freeze_sha256=sha(PACKAGE/'PRODUCTION_FREEZE.json'),
        pair_finite_sha256=sha(PACKAGE/'PAIR_FINITE.json'),audit_finite_sha256=sha(PACKAGE/'AUDIT_FINITE.json'),
        pair_bounds=pb,audit_bounds=vb,teacher_L1_cap_exact=inp['teacher_L1_cap_exact'],
        teacher_curvature_allowance_exact=str(teacher_allowance),transfer_cap_exact=inp['transfer_cap_exact'],
        teacher_midpoints_renormalized=False,normalization='||d||2=1; parameter t is angle-path scale',
        summary=summary,checks=checks,passed=all(checks.values()),rows=rows,
        scope=dict(same_existing_F8_root=True,new_optimizer_updates=0,new_random_draws=0,
            new_teacher_integrals=0,new_F8_grid_nodes=0,scientific_NPZ_reads=0,
            execution_target_nodes=354,execution_scalar_integrals=768,
            new_target_nodes_during_composition_or_check=0,finite_step_certified=False,
            manuscript_changed=False,external_scientific_review=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for flag in ('freeze','run-pair','run-audit','compose','check'):
        group.add_argument('--'+flag,action='store_true')
    args=parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run_pair or args.run_audit:
        run('PAIR' if args.run_pair else 'AUDIT')
    else:
        result=compose()
        if args.compose:
            write_new(PACKAGE/'CERTIFICATE.json',result)
        else:
            require(result==read(PACKAGE/'CERTIFICATE.json'), 'Saved certificate differs from reconstruction')
        print(json.dumps(dict(passed=result['passed'],summary=result['summary'],checks=result['checks'],new_target_nodes=0),indent=2))
        require(result['passed'],'R01 gates not passed; preserve result and stop')

if __name__=='__main__':
    main()

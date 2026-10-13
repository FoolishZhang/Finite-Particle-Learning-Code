"""Read-only, self-contained saved-arithmetic checks. Never integrate or sample."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))


def require(ok,message):
    if not ok: raise ValueError(message)


def read(pkg,name):
    return json.loads((ROOT/'artifacts'/pkg/name).read_text(encoding='utf8'))


def identity():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text(encoding='utf8'))
    files={p.relative_to(ROOT).as_posix():p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json'}
    require(set(files)==set(manifest['files']),'Supplement file inventory differs')
    for name,p in files.items():
        require(hashlib.sha256(p.read_bytes()).hexdigest()==manifest['files'][name]['sha256'],'Supplement hash mismatch: '+name)
        require(p.stat().st_size==manifest['files'][name]['bytes'],'Supplement byte count mismatch: '+name)
        if p.suffix in ('.json','.py','.md','.txt'):
            t=p.read_text(encoding='utf8')
            require(not re.search(r'https?://github\.com/|(?<![A-Za-z])[A-Za-z]:[\\/]',t),'Identity metadata found: '+name)
    return dict(files_checked=len(files),bytes_checked=sum(p.stat().st_size for p in files.values()),package_identity_only=True)


def angular():
    import angle as a
    import angle_math as m
    a.self_test(); fixtures=m.exact_fixture_suite(); caps=m.rational_caps()
    pkg='angular_witness_v1_20261010'; cert=read(pkg,'CERTIFICATE.json'); inp=read(pkg,'INPUTS.json')
    direction=read(pkg,'DIRECTION.json');u=direction['u_integer'];S=direction['S_integer']
    require(S==281474961915412==sum(x*x for x in u) and sum(u)==0,'Exact direction identity')
    require(inp['u_integer']==u and inp['S_integer']==S,'Direction/input agreement')
    teacher=read('certified_teacher_v1_20261008','FINITE_GRID_SUM.json')['pmf_finite_sum']
    tc=read('certified_teacher_v1_20261008','TEACHER_CERTIFICATE.json')
    Q=read('certified_teacher_v1_20261008','TEACHER_RISK_CERTIFICATE.json')['candidate_Q_true_enclosures']
    oldQ=read('certified_teacher_v1_20261008','CANDIDATE_RISK_INTEGRALS.json')['finite_sum']['Q_finite_sum']
    pair=read(pkg,'PAIR_FINITE.json');audit=read(pkg,'AUDIT_FINITE.json')
    e=a.interval(cert['pair_bounds']['errors']['N_total'])[1]
    en=a.interval(cert['audit_bounds']['errors']['N_path_total'])[1];eq=a.interval(cert['audit_bounds']['errors']['Q_total'])[1]
    te=F(inp['teacher_L1_cap_exact']);transfer=F(inp['transfer_cap_exact'])
    require(a.interval(tc['total_L1_error_from_finite_midpoints'])[1]<te,'Teacher allowance')
    require(transfer==F('7.52685e-6')==417*F('1.805e-8'),'Root transfer constant')
    root_cert=read('certified_f8_hessian_v1_20261008','HESSIAN_AND_WITNESS_CERTIFICATE.json')
    require(a.interval(root_cert['witness']['root_distance_upper'])[1]<F('1.805e-8'),'Saved same-root distance below angle transfer cap')
    totals=[(F(0),F(0)),(F(0),F(0))]; masses=[(F(0),F(0)),(F(0),F(0))]
    for y in range(256):
        row=cert['rows'][y];q=sum(a.interval(teacher[y]))/2;den=a.interval(Q[y]);require(den[0]>0,'Positive Q')
        np=a.enlarge(a.interval(pair['N_finite_sum'][y]),e)
        na=a.enlarge(a.interval(audit['N_path_finite_sum'][y]),en)
        qa=a.enlarge(a.interval(audit['Q_finite_sum'][y]),eq)
        vals={'N_true':np,'N_path_true':na,'audit_Q_true':qa,'ratio':a.divide_positive(np,den),'audit_ratio':a.divide_positive(na,den)}
        require(q==F(row['teacher_midpoint_exact']) and all(a.contained(v,a.interval(row[k])) for k,v in vals.items()),'Row containment '+str(y))
        require(a.overlap(np,na) and a.overlap(qa,den) and a.overlap(a.interval(audit['Q_finite_sum'][y]),a.interval(oldQ[y])),'Two-formula agreement '+str(y))
        for j,key in enumerate(('ratio','audit_ratio')):totals[j]=a.add(totals[j],a.multiply((q,q),vals[key]))
        for j,obj in enumerate((pair['N_finite_sum'][y],audit['N_path_finite_sum'][y])):masses[j]=a.add(masses[j],a.interval(obj))
    values=[a.enlarge(t,8*te) for t in totals];values += [a.enlarge(t,transfer) for t in values]
    for name,v in zip(('center','audit_center','root','audit_root'),values):
        require(a.contained(v,a.interval(cert['summary'][name])) and v[1]<0,'Strict summary '+name)
    require(values[2][1]<F('-.00450377'),'Paper angle-curvature margin')
    require(all(lo<=0<=hi for lo,hi in masses),'Finite numerator mass check')
    return dict(passed=True,labels=256,row_intervals=1280,negative_summary_intervals=4,
                exact_math_fixtures=fixtures,transfer_cap=caps['transfer_strict_upper_exact'],
                analytic_error_bounds_are_supplied_inputs=True,new_integrals=0)


def remainder():
    import remainder as r
    pkg='r02_remainder_v1_20261010'
    result=r.validate(read(pkg,'RESULTS.json'),read(pkg,'WEIGHTS.json'))
    expected=read(pkg,'FRACTION_CHECK.json')
    for k,v in result.items(): require(expected[k]==v,'R02 saved Fraction result differs: '+k)
    require(result['negative_certified_n']==[4096] and result['unresolved_n']==[64,256,1024],'Paper R02 sign classification')
    import angle
    lo,hi=angle.interval(read(pkg,'RESULTS.json')['propagation']['rows'][-1]['theoretical_D_interval'])
    require(F('-.002045393')<lo<=hi<F('-.0003519373'),'Paper R02 finite-n derivative enclosure')
    return dict(passed=True,negative_certified_n=result['negative_certified_n'],unresolved_n=result['unresolved_n'],counts=result['counts'],noise_and_active_bounds_are_supplied_inputs=True)


def teacher_family():
    import teacher_family as t
    import consequences as c
    pkg='r03_teacher_family_v1_20261011';inp=read(pkg,'BOX_INPUTS.json');ref=read(pkg,'SPECTRAL_REFINEMENT.json')
    spectra=[]
    for key,row in zip(('h01','h001','h0001'),ref['rows']):
        spectra.append(t.audit_spectrum(read(pkg,key+'/CERTIFICATE.json'),row))
    require([x['core_passed'] for x in spectra]==[False,True,True],'Preserved success/unresolved classes')
    cert=read(pkg,'h001/CERTIFICATE.json');teacher=read(pkg,'TEACHER_CERTIFICATE.json')
    box=inp['boxes']['h001']
    require(all(cert[k]==box[k] for k in ('h_exact','radii_exact','a0_exact')) and cert['A_exact']==inp['A_exact'],'R03 claimed box and exact preconditioner identity')
    blocks=[read(pkg,f'h001/LABELS_{i:03d}.json') for i in range(0,128,8)]
    bounds=read(pkg,'h001/ANALYTIC_BOUNDS.json')
    require(bounds['config']==inp['candidate_config'] and all(b['config']==inp['candidate_config'] for b in blocks),'R03 label/bound configuration identity')
    raw_risk=read(pkg,'RISK_h001.json')
    require(all(len(raw_risk[k])==256 for k in ('Q_finite','N_finite')) and all(len(raw_risk['composition'][k])==256 for k in ('Q_true_intervals','N_true_intervals','rho_true_intervals')),'R03 complete 256-label risk arrays')
    q,p,teacher_report=t.audit_teacher(teacher,read(pkg,'TEACHER_FINITE.json'))
    require(teacher_report['true_component_enclosures_and_zero_mass_declaration_checked'],'R03 teacher transfers and zero-mass prerequisite')
    rows=t.read_conditional(blocks)
    G,H,combined=t.signed_rebuild(rows,q,p,F('1/1000'),cert)
    root=t.audit_root(cert,rows,G,H,teacher,bounds)
    risk=t.audit_risk(raw_risk,cert,q,p,F('1/1000'))
    require(root['passed'] is True and risk['passed'] is True,'R03 root/risk prerequisites')
    quant=c.compute_from_saved(teacher,cert,ref,read('angular_witness_v1_20261010','DIRECTION.json'))
    require(quant==read(pkg,'QUANTITATIVE_CONSEQUENCES.json'),'Quantitative consequences differ')
    require(quant['lambda_interval_exact']==['-1/1000','1/1000'] and F(quant['F8_fixed_angle_curvature_strict_lower_exact'])>F('.0011'),'Paper R03 interval and positive objective curvature')
    require(F(quant['true_risk_fixed_angle_curvature']['upper_exact'])<F('-.00435107'),'Paper R03 negative risk curvature')
    return dict(passed=True,lambda_interval_exact=quant['lambda_interval_exact'],teacher=teacher_report,
                signed_combination=combined,root_passed=root['passed'],risk_passed=risk['passed'],
                preserved_core_classes=[x['core_passed'] for x in spectra],
                alias_tail_and_global_analytic_caps_are_supplied_inputs=True,new_integrals=0)


def theorem3():
    from flint import arb,ctx
    import common as c
    import hessian_aggregate as agg
    import hessian_finalize as hf
    import hessian_bounds as hb
    import saved_matrix as sm
    import gradient_finalize as gf
    import risk_range as rr
    ctx.prec=192
    pkg='certified_f8_hessian_v1_20261008';inp=read(pkg,'INPUTS.json');cert=read(pkg,'HESSIAN_AND_WITNESS_CERTIFICATE.json')
    rows=[read(pkg,f'rep_{i:02d}.json') for i in range(41)]
    require(all(x['status']=='completed' and x['representative_index']==i for i,x in enumerate(rows)),'All 41 classes')
    matrices=agg.aggregate_records(rows)
    require(matrices==read(pkg,'FINITE_HESSIANS.json')['matrix_blocks'],'Saved matrix aggregation')
    saved=read('local_witness_feasibility_v1_20261008','SAVED_MATRIX_CERTIFICATE.json')
    a=list(map(F,inp['a_pairs_exact']));full=a+a[::-1];basis=sm.fixed_slice_basis(full)
    gradient_input=read('certified_f8_gradient_v1_20261008','INPUTS.json')
    risk_input=read('certified_teacher_v1_20261008','CANDIDATE_RISK_INTEGRALS.json')
    theta=[F(int(x['numerator']),int(x['denominator'])) for x in saved['center']['theta_exact_dyadics']]
    risk_a=[F(int(x['numerator']),int(x['denominator'])) for x in risk_input['a_exact']]
    require(a==list(map(F,gradient_input['a_pairs_exact'])) and full==risk_a and theta==[F(0)]*8+full,'Exact same centre in gradient/Hessian/risk and saved matrices')
    require(all(sum(full[i]*basis[i][j] for i in range(8))==0 for j in range(7)),'Exact transverse slice')
    ba=[[F.from_float(float.fromhex(x)) for x in row] for row in saved['saved_Hba']['symmetrized_binary64_hex']]
    hp=[[F.from_float(float.fromhex(x)) for x in row] for row in saved['saved_Hperp']['symmetrized_binary64_hex']]
    require(sm.ldlt_positive(sm.shift(ba,F('1/20')))['positive_definite_certified'],'Radial saved floor')
    require(sm.ldlt_positive(sm.congruence(sm.shift(hp,F('7/500')),basis))['positive_definite_certified'],'Physical transverse saved floor')
    require(sm.safe_lipschitz_proof()==saved['safe_L_F_proof'],'Rational Lipschitz cap')
    grad=read('certified_f8_gradient_v1_20261008','GRADIENT_CERTIFICATE.json')
    require(grad['true_F8_gradient_enclosed'] is True and grad['exact_teacher_symmetry_lifts_four_norm_to_full24'] is True,'True full-24 gradient prerequisite')
    rebuilt=gf.certify_vector([c.from_record(x) for x in grad['finite_vector']],c.from_endpoint(grad['mathematical_vector_error_upper']['upper_exact_dyadic']))
    # A saved true error endpoint can be slightly wider than the original Arb
    # expression; the fixed conservative gradient gate is the relevant test.
    require(c.from_record(rebuilt['true_gradient_L2_upper']).upper()<c.aq('1893/2500000000000') and c.from_record(grad['true_gradient_L2_upper']).upper()<c.aq('1893/2500000000000'),'Gradient norm gate')
    budget=hb.hessian_operator_bounds(inp['a_pairs_exact']);require(budget==cert['analytic_budget'],'Closed-form Hessian budget')
    H={k:hf.matrix_from_records(matrices[k],8) for k in matrices}
    distba=hf.frobenius_distance_upper(hf.block_diagonal(H['Hbb'],H['Haa']),hf.exact_saved(saved['saved_Hba']['symmetrized_binary64_hex'],16))
    distp=hf.frobenius_distance_upper(H['Hperp'],hf.exact_saved(saved['saved_Hperp']['symmetrized_binary64_hex'],8))
    eba=distba+c.from_endpoint(budget['bounds']['total_Hba_operator_without_arithmetic']['upper_exact_dyadic'])
    ep=distp+c.from_endpoint(budget['bounds']['total_Hp_operator_without_arithmetic']['upper_exact_dyadic'])
    risk=read('certified_teacher_v1_20261008','POSTHOC_RISK_BALL.json')
    require(F(risk['physical_radius_exact'])==F('1/2000000') and risk['parameter_dimension']==24,'Same full physical risk ball')
    require(all(x-F('1/2000000')>F('1/20') and x+F('1/2000000')<2 for x in full),'Witness ball inside original parameter box')
    witness=hf.witness_logic(eba,ep,c.from_endpoint(grad['true_gradient_L2_upper']['upper_exact_dyadic']),
                            c.from_endpoint(risk['risk_direction_uniform_upper_expression']['upper_exact_dyadic']),
                            sum((c.aq(x*x) for x in full),arb(0)).sqrt(),saved_floors_verified=True)
    require(witness==cert['witness'],'Root/local-minimum witness arithmetic differs')
    require(all(witness['checks'].values()),'All root/minimum/risk witness prerequisites must pass')
    tc=read('certified_teacher_v1_20261008','TEACHER_CERTIFICATE.json');qc=read('certified_teacher_v1_20261008','TEACHER_RISK_CERTIFICATE.json')
    range_result=rr.risk_range(tc['true_PMF_enclosures'],qc['candidate_Q_true_enclosures'],witness['root_distance_upper'],risk['risk_Hessian_operator_uniform_upper_expression'])
    stored=read('submission_candidate_v1_8_20261008','RISK_RANGE_RESULT.json')
    for key,value in range_result.items(): require(stored[key]==value,'KL synthesis differs: '+key)
    require(c.from_record(range_result['root_KL_enclosure']).lower()>c.aq('.01043356') and c.from_record(range_result['root_KL_enclosure']).upper()<c.aq('.01043358'),'Paper KL about 0.01043')
    return dict(passed=True,classes_aggregated=41,saved_matrix_floors_rechecked=True,
                root_and_local_minimum_modulo_rotation=all(witness['checks'].values()),
                root_KL_interval=range_result['root_KL_enclosure'],
                gradient_finite_sum_and_true_error_are_supplied_inputs=True,new_integrals=0)


def failure_probability():
    from flint import ctx
    import failure as f
    ctx.prec=128
    pkg='failure_probability_v1_20261006';scalar=read(pkg,'SCALAR_STATS.json');saved=read(pkg,'RESULTS.json')
    require(scalar['exact_binary64_integer_moments'] and scalar['records']==saved['scalar_statistics'],'Scalar evidence agreement')
    result=f.infer(scalar['records']);require(result==saved['inference'],'Exact moments/confidence/Cantelli recomposition')
    require(result['mathematical_program']['probability_lower']>=.7035,'Paper qualified program probability at least .7035')
    return dict(passed=True,qualified_mathematical_probability_lower=result['mathematical_program']['probability_lower'],
                risk_grid=result['constants']['risk_grid'],audit_alpha=result['audit_alpha'],
                saved_exact_moments_recomputed=True,random_banks_replayed=False,
                uniform_input_or_future_native_score_error_certified=False,new_integrals=0)


def figure_data():
    gap=read('nuisance_contrast_v1_20261006','POPULATION.json');curvature=read('local_geometry_v1_20261006','STATISTICS.json')
    selected=read('nuisance_contrast_v1_20261006','RESULTS.json');bridge=read('asymptotic_bridge_v1_20261008','AUTHOR_READOUT.json')
    require([x['contrast'] for x in gap['results'][:2]]==['g0','g256'],'Figure 1 gaps')
    require([x['contrast'] for x in curvature['results'][:2]]==['h0','h256'],'Figure 1 curvature')
    require(selected['selected'][0]['candidate']==[0,0,0] and selected['selected'][1]['candidate']==[9,9,9],'Matched selection record')
    require([x['n'] for x in bridge['per_n']]==[64,256,1024,4096],'Figure 2 dimensions')
    for row in gap['results'][:2]+curvature['results'][:2]:require(row['interval'][0]<=row['mean']<=row['interval'][1],'Plot endpoint order')
    for row in bridge['per_n']:
        for key in ('A','Bgrad','omega_over_w','O','J'):
            v=row[key];require(v['interval'][0]<=v['mean']<=v['interval'][1],'Bridge endpoint order')
    return dict(passed=True,plot_data_only=True,statistical_intervals_not_reestimated=True,matched_selection_not_rerun=True)


def spectral_Arb():
    import spectral as s
    def numeric(value):
        if isinstance(value,dict):return {k:numeric(v) for k,v in value.items() if 'sha256' not in k}
        if isinstance(value,list):return [numeric(v) for v in value]
        return value
    pkg='r03_teacher_family_v1_20261011'
    rebuilt=s.compose([read(pkg,key+'/CERTIFICATE.json') for key in ('h01','h001','h0001')])
    stored=read(pkg,'SPECTRAL_REFINEMENT.json')
    for k,v in rebuilt.items():require(numeric(v)==numeric(stored[k]),'Arb spectral recomposition '+k)
    tests=s.manufactured_checks();require(tests['passed'],'Arb spectral manufactured checks')
    return dict(passed=True,lambda_interval_exact=rebuilt['claimed_lambda_interval_exact'],all_three_boxes_rebuilt=True,new_integrals=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--level',choices=('fraction','all'),default='all');args=parser.parse_args()
    require(sys.flags.optimize==0,'Do not disable assertions with -O/-OO')
    start=time.monotonic(); result={'package':identity()}
    jobs=[('angular',angular),('R02',remainder),('R03_fraction',teacher_family),('figures',figure_data)]
    if args.level=='all':jobs += [('theorem3_Arb',theorem3),('R03_spectral_Arb',spectral_Arb),('failure_probability_Arb',failure_probability)]
    for name,fn in jobs:
        tick=time.monotonic();result[name]=fn();result[name]['wall_seconds']=round(time.monotonic()-tick,3)
        print(json.dumps(dict(check=name,passed=True,wall_seconds=result[name]['wall_seconds'])),flush=True)
    print(json.dumps(dict(passed=True,level=args.level,results=result,total_wall_seconds=round(time.monotonic()-start,3),
                         scope='saved arithmetic and package identity; no scientific production replay',new_integrals=0,
                         new_random_draws=0,new_optimizer_updates=0,scientific_arrays_read=0),indent=2))


if __name__=='__main__':main()

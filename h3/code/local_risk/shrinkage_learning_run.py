"""Fixed-program local-alternative risk study; imports never sample or fit.

The unit is an independent dataset/numerical-bank case at a fixed cell teacher.
All case outputs freeze before independent evaluation banks. Reference mapping
is a diagnostic, never an inclusion rule for the algorithm-risk comparison.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.special import expit

import shrinkage_learning_common as common
import group_posterior_objective as group
from group_posterior_target import qualify_marginal

old=common.old
SCHEMA='local_shrinkage_learning_v1'
METHODS=('integrated','group8','ridge')
CELLS={'weak_N2048':(2048,0),'mid_N2048':(2048,1000),
       'strong_N2048':(2048,2000),'mid_N256':(256,3000)}


def validate(contract):
    phase=contract['phase'];old.require(phase in ('development','validation'),'explicit local-learning phase required')
    old.require(not contract.get('draft_not_authorized_to_run',False),'draft cannot execute')
    cases=contract['cases'];n=4 if phase=='development' else 96
    old.require(len(cases)==n and len({c['case_id'] for c in cases})==n,'complete unique frozen case schedule required')
    old.require(contract['main_outputs']==6*n and contract['expected_risk_points']==9*n,'six outputs/nine risk rows per case required')
    old.require(contract['case_count']==n and contract['data_observations']==sum(c['N'] for c in cases) and
        contract['teacher_generation_draws']==0 and contract['methods']==list(METHODS),'declared case/data/method schedule differs')
    old.require(contract['workers']==4 and 'floor' not in contract['solver'],'four threads and one model floor required')
    old.require(contract['initial_coordinates']==[[0.,0.,0.],[.2,-.15,.1]],'both original starts required')
    old.require(contract['numerical_control_extension']=='centered_origin_hessian' and contract['group_control']=='uniform_logratio','fixed quadratic group scalar required')
    old.require(contract['caps'].get('control_construction_particle_pattern_terms',0)>0,'control construction cap required')
    J=contract['ridge']['J'];cells={c['cell_id']:c for c in contract['cells']}
    old.require(len(cells)==len(contract['cells'])==4 and set(cells)==set(CELLS),'four declared fixed cells required')
    for case in cases:
        old.require(case['cell_id'] in CELLS,'unknown frozen cell')
        N,base=CELLS[case['cell_id']];spec=cells[case['cell_id']];S=spec['S']
        old.require(np.isfinite(S) and S>0 and case['N']==spec['N']==N and case['S']==S,'case must use frozen cell N/S')
        for name in ('h','coordinates','Cstar'):old.require(case[name]==spec[name],'case differs from fixed cell '+name)
        h=np.asarray(case['h'],float);c=np.asarray(case['coordinates'],float);C=np.asarray(case['Cstar'],float)
        expected=np.sqrt(S/J)*np.array([1.,1.,0.])/np.sqrt(2)
        old.require(h.shape==c.shape==(3,) and np.allclose(h,expected,rtol=0,atol=1e-12) and np.allclose(c,h/np.sqrt(N),rtol=0,atol=1e-12),'local h/sqrtN identity differs')
        old.require(C.shape==(3,3) and np.array_equal(C,old.legacy.correlation_matrix(c)) and np.linalg.eigvalsh(C)[0]>=contract['model']['spectral_floor'],'fixed cell teacher outside domain')
        if phase=='validation':old.require(case['replicate'] in range(24) and case['case_id']==base+case['replicate'],'independent cell/replicate identity differs')
    if phase=='validation':
        old.require(all(sum(c['cell_id']==cell for c in cases)==24 for cell in CELLS),'24 independent datasets per cell required')
    else:
        old.require({c['case_id'] for c in cases}==set(range(10000,10004)) and
            all(sum(c['cell_id']==cell for c in cases)==2 for cell in ('strong_N2048','mid_N256')),'single four-case development scope required')
    return contract


def prepare_samples(samples,patterns):
    cls=importlib.import_module('group_posterior_quadratic').QuadraticPreparedGroupBanks
    return cls(samples['x_group'],samples['log_marginal_weight'],patterns['state_indices'],patterns['weights'],
        proposal_kind='C_independent_observed_group_rejection',proposal_mean=samples['proposal_mean'],
        proposal_second_moment=samples['proposal_second_moment'])


def group_value(prepared,c,budget,contract):
    miss=getattr(prepared,'_origin_hessian_cache',None) is None
    requested=prepared.R*prepared.B*prepared.K*len(prepared.weights) if miss and prepared.K>1 else 0
    budget.check('control_construction_particle_pattern_terms',requested)
    if requested:budget.add('control_construction_particle_pattern_terms',requested)
    info=prepared.origin_hessian_control(pattern_chunk=contract['pattern_chunk'])
    actual=info.get('control_construction_particle_pattern_terms',info.get('particle_pattern_terms_this_call'))
    old.require(actual==requested,'control construction work receipt differs')
    result=common.group_value(prepared,c,budget,contract)
    old.require(result['control']=='uniform_logratio_plus_centered_origin_hessian','actual selected control label required')
    result['control_construction_particle_pattern_terms_before_evaluation']=actual
    result['control_preconstruction_cache_hit']=info['cache_hit']
    return result


def generate_data(contract,case,streams,budget,destination):
    """Only this generator and offline risk use the fixed cell's true matrix."""
    N=case['N'];m=contract['model'];partial={};rng=streams.get('data',case['case_id'])
    def draw(kind,shape,counter):
        size=int(np.prod(shape));budget.check(counter,size);value=getattr(rng,kind)(shape)
        budget.counts[counter]=budget.counts.get(counter,0)+size;budget.check();return value
    try:
        x=draw('standard_normal',(N,3),'gaussian_scalars')@np.linalg.cholesky(case['Cstar']).T;partial['Xstar']=x
        generation_terms=N*3*m['generation_inner_order']
        budget.add('generation_private_node_terms',generation_terms)
        h=old.competition.h_probabilities(x*np.asarray(m['epsilon']),tau=m['tau'],inner_order=m['generation_inner_order'])
        budget.check('binomial_scalars',N*3);counts=rng.binomial(m['M'],h['h'])
        budget.counts['binomial_scalars']=budget.counts.get('binomial_scalars',0)+counts.size;partial['counts']=counts;budget.check()
        uniform=draw('random',(N,3),'uniform_scalars');Y=np.where(uniform<expit(x),1,-1).astype(np.int8)
        common.write_npz(destination/'VISIBLE_DATA.npz',{'counts':counts,'observed_labels':Y})
        common.write_npz(destination/'GENERATION_PROVENANCE.npz',{'Xstar':x,'label_uniforms':uniform,
            'count_success_probability':h['h'],'count_failure_probability':h['complement']})
        common.write_json(destination/'DATA_RECEIPT.json',{'case_id':case['case_id'],'N':N,'cell_id':case['cell_id'],
            'nested_data':False,'independent_case_stream':True,'generation_inner_order':m['generation_inner_order'],
            'actual_requested_generation_private_node_terms':generation_terms,
            'generation_private_node_terms_scope':'requested N*3*inner_order node locations; not measured CPU operations or posterior rejection work',
            'actual_counts_Y_only_for_fit':True,'rng_state':rng.bit_generator.state})
        return {'counts':counts,'observed_labels':Y}
    except Exception:
        common.write_npz(destination/'FAILED_DATA_PREFIX.npz',partial);raise


def point(case,method,init,c,fit=None):
    c=np.asarray(c,float)
    return {'case_id':case['case_id'],'cell_id':case['cell_id'],'replicate':case['replicate'],'N':case['N'],
        'point_id':f'{method}_init{init}','method':method,'initialization_index':init,'coordinates':c,
        'C':old.legacy.correlation_matrix(c),'is_fit_output':fit is not None,
        'fit_status':None if fit is None else fit['status'],'fit_mapping_norm_inf':None if fit is None else fit['mapping_norm_inf'],
        'output_rule':'fixed_reference_state' if fit is None else 'last_accepted_per_start_no_risk_selection',
        'unit':'dataset_case_not_random_teacher','teacher_information_used_for_empirical_fit':False}


def fit_case(state,sampler,contract):
    case,budget,folder=state['case'],state['budget'],state['folder']
    visible=generate_data(contract,case,state['streams'],budget,folder);state['visible']=visible
    old.require(visible['counts'].shape==visible['observed_labels'].shape==(case['N'],3),'exact case actual-observation count required')
    patterns=group.empirical_patterns(visible['counts'],visible['observed_labels'],M=contract['model']['M'])
    ml_patterns=old.integrated.prepare_observations(visible['counts'],visible['observed_labels'],M=contract['model']['M'])
    samples=common.make_banks(contract,sampler,contract['group']['fit'],'fit',case['case_id'],folder/'FIT_BANKS',budget)
    prepared=prepare_samples(samples,patterns);model=old.engine(contract,contract['fit_gh']);outputs=[]
    for method in METHODS:
        destination=folder/('FIT_'+method);destination.mkdir(exist_ok=False)
        for init,initial in enumerate(contract['initial_coordinates']):
            control=None
            with (destination/f'init{init}_OBJECTIVE_TRACE.jsonl').open('x',encoding='utf-8') as log:
                def objective(c):
                    nonlocal control
                    if method=='group8':result=group_value(prepared,c,budget,contract);control=result['control']
                    else:
                        base=old.ml_value(model,old.legacy.correlation_matrix(c),ml_patterns,None,budget)
                        lam=contract['ridge']['lambda'] if method=='ridge' else 0.
                        result={'value':base['value']+.5*lam*np.dot(c,c),'gradient':base['gradient']+lam*c}
                    log.write(json.dumps({'coordinates':c,'value':result['value'],'gradient':result['gradient'],
                        'control':result.get('control'),'raw_value':result.get('raw_value'),'budget':budget.snapshot()},default=common.serial,allow_nan=False)+'\n');log.flush()
                    return {'value':result['value'],'gradient':result['gradient']}
                fit=old.optimize.solve(objective,initial,floor=contract['model']['spectral_floor'],
                    deadline=budget.started+contract['caps']['seconds'],**contract['solver'])
            common.write_json(destination/f'init{init}_FIT.json',fit)
            outputs.append({**point(case,method,init,fit['coordinates'],fit),
                'selected_scalar':control+'_fixed_group_bank' if method=='group8' else 'integrated_NLL_plus_fixed_ridge' if method=='ridge' else 'integrated_joint_NLL'})
    state['outputs']=outputs
    common.write_json(folder/'FIT_COMPLETE.json',{'outputs':outputs,'budget':budget.snapshot(),'streams':state['streams'].states()})
    return state


def risk_evaluation(contract,case,p,budget,models,truth,destination):
    N=case['N'];base=contract['qualification'];tol=min(base['kl_tol'],base['scaled_risk_tol']/N)
    risk_contract={**contract,'qualification':{**base,'kl_tol':tol}}
    result=old.risk_evaluation(risk_contract,budget,{'id':case['case_id'],'Cstar':case['Cstar']},p,models,truth,destination)
    result.update(scaled_joint_kl=N*result['joint_kl'],scaled_error=N*result['precision_diagnostics']['KL_order_difference'],
        risk_kl_tolerance=tol,risk_pmf_tolerance=risk_contract['qualification']['pmf_tol'],risk_N=N)
    return result


def evaluate_case(state,sampler,contract):
    case,budget,folder=state['case'],state['budget'],state['folder'];visible=state['visible']
    destination=folder/'EVALUATION';destination.mkdir(exist_ok=False)
    samples=common.make_banks(contract,sampler,contract['group']['eval'],'evaluation',case['case_id'],folder/'EVALUATION_BANKS',budget)
    prepared=prepare_samples(samples,group.empirical_patterns(visible['counts'],visible['observed_labels'],M=contract['model']['M']))
    patterns=old.integrated.prepare_observations(visible['counts'],visible['observed_labels'],M=contract['model']['M'])
    models=[old.engine(contract,order) for order in contract['risk_orders']];truth={};risks={};baselines={};rows=[]
    fixed=[point(case,'initial',init,c) for init,c in enumerate(contract['initial_coordinates'])]
    fixed.append({**point(case,'teacher',None,case['coordinates']),'point_id':'teacher_Cstar'})
    for p in [*state['outputs'],*fixed]:
        key=np.asarray(p['coordinates'],float).tobytes()
        if key not in risks:risks[key]=(p['point_id'],risk_evaluation(contract,case,p,budget,models,truth,destination))
        origin,risk=risks[key];own=None;diagnostic=None
        if p['method']=='group8':
            init=p['initialization_index']
            if init not in baselines:
                baselines[init]=group_value(prepared,contract['initial_coordinates'][init],budget,contract)
                common.save_group_result(destination/f'BASELINE_group8_init{init}',baselines[init])
            result=group_value(prepared,p['coordinates'],budget,contract)
            common.save_group_result(destination/(p['point_id']+'_GROUP'),result)
            paired=group.paired_statistics(result,baselines[init])
            raw=group.paired_statistics({'replicate_values':result['raw_replicate_values'],'replicate_gradients':result['raw_replicate_gradients']},
                {'replicate_values':baselines[init]['raw_replicate_values'],'replicate_gradients':baselines[init]['raw_replicate_gradients']})
            common.save_group_result(destination/(p['point_id']+'_MINUS_MATCHING_INITIAL'),{**paired,**{'raw_'+k:v for k,v in raw.items()},
                'selected_bank_delta':result['bank_values']-baselines[init]['bank_values'],
                'raw_bank_delta':result['raw_bank_values']-baselines[init]['raw_bank_values']})
            mapping=old.optimize.gradient_mapping(p['coordinates'],result['gradient'],floor=contract['model']['spectral_floor'])
            diagnostic=bool(p['fit_status']=='mapping_tolerance' and mapping['norm_inf']<=contract['qualification']['mapping_tol'] and
                np.max(result['replicate_gradient_se'])<=contract['qualification']['finite_gradient_SE_tol'])
            own={'target':'group8','control':result['control'],'value':result['value'],'raw_value':result['raw_value'],
                'gradient':result['gradient'],'gradient_scramble_SE':result['replicate_gradient_se'],'projected_mapping':mapping,
                'output_minus_matching_initial':paired['value'],'paired_scramble_SE':paired['replicate_se'],
                'raw_output_minus_matching_initial':raw['value'],'raw_paired_scramble_SE':raw['replicate_se']}
        elif p['method'] in ('integrated','ridge'):
            c=p['coordinates'];lam=contract['ridge']['lambda'] if p['method']=='ridge' else 0.;levels=[]
            for model in models[:2]:
                r=old.ml_value(model,p['C'],patterns,None,budget)
                levels.append({'value':r['value']+.5*lam*np.dot(c,c),'gradient':r['gradient']+lam*c})
            high=levels[-1];dv=abs(high['value']-levels[0]['value']);dg=float(np.max(np.abs(high['gradient']-levels[0]['gradient'])))
            mapping=old.optimize.gradient_mapping(c,high['gradient'],floor=contract['model']['spectral_floor'])
            diagnostic=bool(p['fit_status']=='mapping_tolerance' and mapping['norm_inf']<=contract['qualification']['mapping_tol'] and
                dv<=contract['qualification']['nll_order_tol'] and dg<=contract['qualification']['gradient_order_tol'])
            own={**high,'target':p['method'],'lambda':lam,'levels':levels,'value_order_difference':dv,
                'gradient_order_max_difference':dg,'projected_mapping':mapping}
        row={**p,**risk,'own_context':own,'reference_qualified':diagnostic,'reference_diagnostic_only':True,
            'risk_reused_from':origin,'included_in_algorithm_risk_comparison':True}
        common.write_json(destination/(p['point_id']+'_EVALUATION.json'),row);rows.append(row)
    old.require(len(rows)==9,'all six outputs plus three fixed-risk records required')
    common.write_json(folder/'EVALUATION_COMPLETE.json',{'evaluations':rows,'budget':budget.snapshot(),'streams':state['streams'].states()})
    return {'case_id':case['case_id'],'evaluations':rows}


def parallel(states,fn,stage,workers):
    results=[];failures=[]
    def guarded(state):
        try:return fn(state)
        except Exception:
            common.write_json(state['folder']/('FAILED_'+stage.upper()+'.json'),{'schema':SCHEMA,'case_id':state['case']['case_id'],
                'stage':stage,'traceback':traceback.format_exc(),'budget':state['budget'].snapshot(),'streams':state['streams'].states(),'automatic_retry':False})
            raise
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='local-shrinkage-'+stage) as pool:
        futures={pool.submit(guarded,s):s for s in states}
        for future in as_completed(futures):
            try:results.append(future.result());status='completed'
            except Exception:failures.append(traceback.format_exc());status='failed'
            print(json.dumps({'stage':stage,'case_id':futures[future]['case']['case_id'],'status':status}),flush=True)
    if failures:raise RuntimeError('local-learning case failure(s): '+json.dumps(failures))
    return results


def run(contract_path,out):
    contract_path,out=Path(contract_path).resolve(),Path(out).resolve();cfg=validate(common.read(contract_path))
    cfg,root=common.start(contract_path,out,phase=cfg['phase']);states=[];stage='marginal'
    try:
        sampler,moments=common.make_marginal(cfg,out/'MARGINAL')
        common.write_json(out/'MARGINAL_QUALIFICATION.json',qualify_marginal(cfg,moments))
        common.write_json(out/'CASES.json',{'cases':cfg['cases'],'fixed_cell_teachers_no_teacher_RNG':True,
            'data_not_nested_across_N':True,'before_any_case_data_or_fit':True})
        n=len(cfg['cases']);caps={k:v if k=='seconds' else (int(v)-root.counts.get(k,0))//n for k,v in cfg['caps'].items()}
        for case in cfg['cases']:
            folder=out/f"case_{case['case_id']:05d}";folder.mkdir(exist_ok=False);budget=common.new_budget(cfg,root.started,caps)
            states.append({'case':case,'folder':folder,'budget':budget,'streams':old.Streams(cfg,budget)})
        stage='fit';parallel(states,lambda s:fit_case(s,sampler,cfg),'fit',cfg['workers'])
        outputs=[p for s in states for p in s['outputs']]
        old.require(len(outputs)==cfg['main_outputs'],'all fixed case outputs required before evaluation RNG')
        common.write_json(out/'FROZEN_OUTPUTS.json',{'outputs':outputs,'before_any_independent_evaluation_bank_generation':True,
            'last_accepted_outputs_not_risk_selected':True,'reference_mapping_not_an_inclusion_gate':True})
        stage='evaluation';nested=parallel(states,lambda s:evaluate_case(s,sampler,cfg),'eval',cfg['workers'])
        order={c['case_id']:i for i,c in enumerate(cfg['cases'])};nested.sort(key=lambda v:order[v['case_id']])
        rows=[p for v in nested for p in v['evaluations']];old.require(len(rows)==cfg['expected_risk_points'],'all declared risk rows retained')
        budget=common.combine_budgets(cfg,root,[s['budget'] for s in states])
        common.write_json(out/'PER_CASE_BUDGETS.json',{'root':root.snapshot(),'caps_per_case':caps,
            'cases':[{'case_id':s['case']['case_id'],'budget':s['budget'].snapshot(),'streams':s['streams'].states()} for s in states]})
        report={'schema':SCHEMA,'status':'completed','phase':cfg['phase'],'cases':cfg['cases'],'outputs':outputs,'evaluations':rows,'budget':budget,
            'risk_qualified_count':sum(r['precision_qualified'] for r in rows),
            'reference_diagnostic_pass_count':sum(r['reference_qualified'] is True for r in rows),
            'all_risks_qualified':all(r['precision_qualified'] for r in rows),'reference_all_pass_not_required':True,
            'new_teacher_draws':0,'new_visible_observations':sum(c['N'] for c in cfg['cases']),
            'unit':'independent dataset/numerical-execution case within fixed cell, not random teacher',
            'all_outputs_and_failures_retained':True,'no_prior_or_SGD_replays':True}
        common.write_json(out/'SUMMARY.json',report);common.manifest(out);return report
    except Exception:
        common.write_json(out/'FAILED.json',{'schema':SCHEMA,'stage':stage,'traceback':traceback.format_exc(),'root_budget':root.snapshot(),
            'case_budgets':[{'case_id':s['case']['case_id'],'budget':s['budget'].snapshot(),'streams':s['streams'].states()} for s in states],
            'elapsed_seconds':time.perf_counter()-root.started,'automatic_retry':False})
        common.manifest(out);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true',required=True)
    parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();r=run(args.contract,args.out);print(json.dumps({'schema':SCHEMA,'status':r['status'],'outputs':len(r['outputs'])}))

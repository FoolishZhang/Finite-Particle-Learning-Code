"""Teacher-weighted finite F8 Hessian blocks; freeze-gated scientific use.

Same/cross moment formulas were specified by root and geometry. No division
by class weight, mass normalization, Hessian clipping, optimizer or RNG.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from pathlib import Path
import time
import certified_teacher_compute as common
import certified_f8_gradient_compute as gradient
import certified_f8_gradient_symmetry as symmetry
import certified_f8_hessian_moments as moments_helper
from flint import arb,arb_mat,ctx

ROOT=common.ROOT;OUT=ROOT/'artifacts/certified_f8_hessian_v1_20261008'
INPUTS=OUT/'INPUTS.json';FREEZE=OUT/'FREEZE.json'
NOMINAL='artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json'
TEACHER='artifacts/certified_teacher_v1_20261008/FINITE_GRID_SUM.json'
BITS=192


def precompute_inner(a,h,L):
    cache=gradient.precompute_inner(a,h,L);p=cache['pair_count'];columns=[]
    for x,w,probabilities in zip(cache['z'],cache['w'],cache['probabilities']):
        yes=[value[1] for value in probabilities]
        row=[w,w*x,w*x*x]
        for power in (0,1,2):row.extend(w*x**power*pi for pi in yes)
        for power in (0,2):
            row.extend(w*x**power*yes[i]*yes[j] for i in range(p) for j in range(i,p))
        columns.append(row)
    cache.update(hessian_moment_matrix=arb_mat(columns),moment_column_count=3+3*p+p*(p+1),
        moment_column_order='T0,T1,T2; P0/P1/P2; PP0/PP2 uppertri lex',total_scalar_moments=4+3*p+p*(p+1))
    return cache


def unpack_moments(cache,values):
    p=cache['pair_count'];offset=3;P=[]
    for _ in range(3):P.append(values[offset:offset+p]);offset+=p
    PP={}
    for power in (0,2):
        matrix=[[None]*p for _ in range(p)]
        for i in range(p):
            for j in range(i,p):matrix[i][j]=matrix[j][i]=values[offset];offset+=1
        PP[power]=matrix
    if offset!=len(values):raise ValueError('moment contraction shape drift')
    return dict(T=values[:3],P=P,PP=PP)


def zero_matrix(m):return [[arb(0) for _ in range(m)] for _ in range(m)]


def weighted_blocks(W,M,S,moments,psi,K):
    """Use geometry's agreed contraction API; no competing formula copy."""
    p=len(moments['P'][0]);upper=moments_helper.upper_pairs(p)
    expanded=moments_helper.expand_pair_moments(moments['T'],moments['P'],
        [moments['PP'][0][i][j] for i,j in upper],[moments['PP'][2][i][j] for i,j in upper],m=len(M))
    result=moments_helper.hessian_integrand(psi,dict(W=W,M=M,S=S),expanded,K)
    return {name:result[name] for name in ('Hbb','Haa','Hperp')}


def as_arb(value):
    return value if isinstance(value,arb) else common.aq(value)


def matrix_records(matrix):return [[common.record(as_arb(value)) for value in row] for row in matrix]


class HessianBudgetStop(RuntimeError):
    def __init__(self,partial):super().__init__('fixed wall budget exhausted; no retry');self.partial=partial


def finite_representative(cache,c,W,M,S,hv,lo,hi,*,K=8,deadline=None,saturation_threshold=128):
    K=gradient.integer(K,'K');hv,lo,hi=map(common.exact,(hv,lo,hi))
    if K<2 or hv<=0 or hi<=lo or (lo/hv).denominator!=1 or (hi/hv).denominator!=1:raise ValueError('K>=2 and exact outer grid required')
    if cache['precision_bits']!=int(ctx.prec):raise ValueError('precision mismatch')
    W=as_arb(W);M=[as_arb(x) for x in M];S=[[as_arb(x) for x in row] for row in S]
    m=len(M);total={name:zero_matrix(m) for name in ('Hbb','Haa','Hperp')}
    A,_=gradient.precompute_count(cache,c);caps=gradient.saturation_enclosures(128) if saturation_threshold is not None else None
    completed=factors=saturated=0;tick=time.perf_counter();psimin=psimax=None;last=None;partial_inner=0
    def output(complete):
        return dict(status='completed' if complete else 'budget_stopped',pair_counts=list(c),K=K,precision_bits=int(ctx.prec),
            matrix_blocks={name:matrix_records(matrix) for name,matrix in total.items()},
            teacher_weight_moments=dict(W=common.record(W),M=[common.record(x) for x in M],S=matrix_records(S)),
            inner_grid=dict(h_exact=cache['h_exact'],L_exact=cache['L_exact'],nodes=cache['normal_nodes'],full_weights=True),
            outer_grid=dict(h_exact=str(hv),vmin_exact=str(lo),vmax_exact=str(hi),nodes_prescribed=int((hi-lo)/hv)+1,nodes_completed=completed,full_weights=True),
            raw_normal_grid_mass=common.record(cache['normal_mass']),normal_mass_strictly_below_one=bool(cache['normal_mass']<1),
            psi_completed_lower_endpoint=None if psimin is None else common.record(psimin),psi_completed_upper_endpoint=None if psimax is None else common.record(psimax),
            backend='arb_mat',moment_columns=cache['moment_column_count'],scalar_moments_including_Psi_T0=cache['total_scalar_moments'],
            saturation_threshold=saturation_threshold,saturation_caps=None if saturation_threshold is None else dict(exp_upper_exact='2^-184',t_exp_upper_exact='2^-177',
                extra_t_squared_exponential_needed=False,Arb_outward_rounding_retained=True,numerical_negative_lower_not_clipped=True),
            last_completed_outer_index=last,current_partial_inner_factors=partial_inner,
            resources=dict(laplace_factors_evaluated=factors,saturated_factor_enclosures=saturated,ordinary_exponential_calls=factors-saturated,
                outer_nodes_completed=completed,new_random_draws=0,new_optimizer_updates=0,new_kernel_calls=0),
            wall_seconds=time.perf_counter()-tick,teacher_or_Gaussian_mass_renormalized=False,class_weight_division_used=False,
            folded_mixed_bias_radial_block_provided=False,finite_sum_arithmetic_enclosed=True,analytic_errors_added=False,
            true_F8_Hessian_certified=False,true_F8_stationary_existence_certified=False,true_F8_local_minimum_certified=False)
    for index in range((lo/hv).numerator,(hi/hv).numerator+1):
        if deadline is not None and time.perf_counter()>=deadline:raise HessianBudgetStop(output(False))
        u=(common.aq(hv)*index).exp();psi=arb(0);coeffs=[];partial_inner=0
        for j,(likelihood,w) in enumerate(zip(A,cache['w'])):
            if deadline is not None and j%32==0 and time.perf_counter()>=deadline:raise HessianBudgetStop(output(False))
            e,te,used=gradient.exponential_factors(u*likelihood,threshold=saturation_threshold,enclosures=caps)
            psi+=w*e;coeffs.append(te);factors+=1;saturated+=used;partial_inner+=1
        dot=arb_mat([coeffs])*cache['hessian_moment_matrix'];values=[dot[0,j] for j in range(cache['moment_column_count'])]
        blocks=weighted_blocks(W,M,S,unpack_moments(cache,values),psi,K)
        for name in total:
            for i in range(m):
                for j in range(m):total[name][i][j]+=common.aq(hv)*blocks[name][i][j]
        completed+=1;partial_inner=0;last=index
        psimin=psi.lower() if psimin is None else min(psimin,psi.lower());psimax=psi.upper() if psimax is None else max(psimax,psi.upper())
    return output(True)


def teacher_inputs():
    """Saved finite teacher enclosures -> weighted class moments, no grid."""
    finite=common.read(ROOT/TEACHER)
    if len(finite['pmf_finite_sum'])!=256:raise ValueError('all256 teacher endpoints required')
    weights=[common.from_record(x) for x in finite['pmf_finite_sum']]
    folded=moments_helper.fold_teacher_moments(weights,8)
    records=[dict(W=common.record(as_arb(entry['W'])),M=[common.record(as_arb(x)) for x in entry['M']],S=matrix_records(entry['S'])) for entry in folded['moments']]
    return dict(representatives=folded['representatives'],teacher_moments=records,
        label_to_representative_index=folded['label_to_representative_index'],label_reflected=folded['label_reflected_to_representative'],
        representative_original_labels=symmetry.symmetry_map(8)['representative_labels'],
        sum_W=common.record(sum((as_arb(entry['W']) for entry in folded['moments']),arb(0))),
        original_teacher_finite_mass=common.record(sum(weights,arb(0))),
        teacher_weights_renormalized=False,within_pair_exchangeability_assumed=False,mixed_bias_radial_preserved_by_fold=False)


def prepared_inputs():
    previous=ctx.prec;ctx.prec=BITS
    try:
        nominal=common.read(ROOT/NOMINAL);a=symmetry.exact_pair_parameters(nominal['center']['theta'])
        teacher=teacher_inputs()
        paths=(NOMINAL,TEACHER,'script/certified_teacher_compute.py','script/certified_f8_gradient_compute.py',
            'script/certified_f8_gradient_symmetry.py','script/certified_f8_hessian_moments.py')
        return dict(schema='certified_f8_hessian_inputs_v1',a_pairs_exact=[str(x) for x in a],K=8,precision_bits=192,
            inner_grid=dict(h_exact='1/40',L_exact='11',nodes=881,full_weights=True),
            outer_grid=dict(h_exact='1/8',vmin_exact='-32',vmax_exact='64',nodes=769,full_weights=True),
            backend='arb_mat',saturation_threshold=128,hard_total_wall_seconds=3600,
            moment_columns=35,total_scalar_moments_including_Psi_T0=36,
            saturation_caps=dict(exp_upper_exact='2^-184',t_exp_upper_exact='2^-177',extra_t_squared_exponential_needed=False,
                Arb_outward_rounding_retained=True,numerical_negative_lower_not_clipped=True),
            teacher=teacher,input_source_sha256={p:common.sha(ROOT/p) for p in paths},
            weighted_class_matrices_include_teacher_moments=True,class_weight_division_used=False,
            folded_mixed_bias_radial_block_provided=False,
            true_F8_Hessian_certified=False,true_F8_stationary_existence_certified=False,true_F8_local_minimum_certified=False,
            new_random_draws=0,new_optimizer_updates=0,automatic_retry_or_extension=False)
    finally:ctx.prec=previous


def verify_freeze():
    saved=common.read(INPUTS)
    if saved!=prepared_inputs():raise ValueError('fixed Hessian input/public source drift')
    sources=common.read(FREEZE).get('sources')
    required=('script/certified_f8_hessian_compute.py','script/certified_f8_hessian_compute_test.py',INPUTS.relative_to(ROOT).as_posix())
    if not isinstance(sources,dict) or any(p not in sources for p in required):raise ValueError('freeze omits compute/test/INPUTS')
    for name,digest in sources.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or any(s.lower() in ('.npz','.npy') for s in path.suffixes) or common.sha(path)!=digest:raise ValueError('freeze source drift: '+name)
    return saved


def restore_moment(entry):
    return common.from_record(entry['W']),[common.from_record(x) for x in entry['M']],[[common.from_record(x) for x in row] for row in entry['S']]


def aggregate_records(rows):
    total={name:zero_matrix(8) for name in ('Hbb','Haa','Hperp')}
    for row in rows:
        for name in total:
            matrix=row['matrix_blocks'][name]
            if len(matrix)!=8 or any(len(values)!=8 for values in matrix):raise ValueError('full8x8 checkpoint blocks required')
            for i in range(8):
                for j in range(8):total[name][i][j]+=common.from_record(matrix[i][j])
    return {name:matrix_records(matrix) for name,matrix in total.items()}


def run():
    inputs=verify_freeze();start=OUT/'RUN_START.json'
    if start.exists():raise FileExistsError('RUN_START exists; no overwrite or automatic retry')
    tick=time.perf_counter();deadline=tick+3600
    locks=dict(inputs_sha256=common.sha(INPUTS),freeze_sha256=common.sha(FREEZE))
    common.write_new(start,dict(created_utc=common.now(),**locks,hard_total_wall_seconds=3600))
    previous=ctx.prec;ctx.prec=192
    try:
        cache=precompute_inner(inputs['a_pairs_exact'],'1/40','11');entries=[];rows=[]
        for index,(c,entry) in enumerate(zip(inputs['teacher']['representatives'],inputs['teacher']['teacher_moments'])):
            W,M,S=restore_moment(entry)
            try:row=finite_representative(cache,c,W,M,S,'1/8','-32','64',K=8,deadline=deadline,saturation_threshold=128)
            except HessianBudgetStop as error:
                partial=error.partial;partial.update(representative_index=index,**locks)
                path=OUT/f'partial_rep_{index:02d}.json';common.write_new(path,partial)
                stop=dict(status='budget_stopped',completed_representatives=entries,partial_rep=path.relative_to(ROOT).as_posix(),
                    wall_seconds=time.perf_counter()-tick,automatic_retry=False)
                common.write_new(OUT/'RUN_STOP.json',stop);return stop
            row.update(representative_index=index,**locks,original_labels=inputs['teacher']['representative_original_labels'][index],
                weighted_class_matrices_include_teacher_moments=True)
            path=OUT/f'rep_{index:02d}.json';common.write_new(path,row);rows.append(row)
            entries.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=common.sha(path),bytes=path.stat().st_size))
            if time.perf_counter()>=deadline:
                stop=dict(status='budget_stopped',completed_representatives=entries,wall_seconds=time.perf_counter()-tick,automatic_retry=False,
                    all_rep_points_saved=len(entries)==41,final_summary_written=False)
                common.write_new(OUT/'RUN_STOP.json',stop);return stop
        summary=dict(schema='certified_f8_hessian_finite_summary_v1',**locks,run_start_sha256=common.sha(start),representative_count=41,
            representative_files=entries,matrix_blocks=aggregate_records(rows),matrix_dimension=8,block_names=['Hbb','Haa','Hperp'],
            weighted_class_blocks_summed_without_second_teacher_factor=True,sum_W=inputs['teacher']['sum_W'],
            resources=dict(laplace_factors_evaluated=sum(row['resources']['laplace_factors_evaluated'] for row in rows),
                saturated_factor_enclosures=sum(row['resources']['saturated_factor_enclosures'] for row in rows),
                ordinary_exponential_calls=sum(row['resources']['ordinary_exponential_calls'] for row in rows),
                wall_seconds=time.perf_counter()-tick,representatives=41,new_kernel_calls=0,new_random_draws=0,new_optimizer_updates=0),
            folded_mixed_bias_radial_block_provided=False,finite_matrix_arithmetic_enclosed=True,analytic_errors_added=False,
            true_F8_Hessian_certified=False,true_F8_stationary_existence_certified=False,true_F8_local_minimum_certified=False)
        common.write_new(OUT/'FINITE_HESSIANS.json',summary)
        return dict(status='completed',representatives=41,wall_seconds=summary['resources']['wall_seconds'])
    finally:ctx.prec=previous


def check():
    inputs=verify_freeze();summary=common.read(OUT/'FINITE_HESSIANS.json');rows=[]
    if summary['inputs_sha256']!=common.sha(INPUTS) or summary['freeze_sha256']!=common.sha(FREEZE) or summary['run_start_sha256']!=common.sha(OUT/'RUN_START.json'):
        raise ValueError('summary input/source locks drift')
    if summary['representative_count']!=41 or len(summary['representative_files'])!=41:raise ValueError('fixed41 class coverage required')
    previous=ctx.prec;ctx.prec=192
    try:
        for index,entry in enumerate(summary['representative_files']):
            path=ROOT/entry['path']
            if path!=OUT/f'rep_{index:02d}.json' or common.sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:raise ValueError('checkpoint hash/size/path drift')
            row=common.read(path)
            if row['status']!='completed' or row['pair_counts']!=inputs['teacher']['representatives'][index] or row['representative_index']!=index:raise ValueError('checkpoint order/completeness drift')
            if row['inputs_sha256']!=common.sha(INPUTS) or row['freeze_sha256']!=common.sha(FREEZE):raise ValueError('checkpoint source drift')
            if row['inner_grid']!=inputs['inner_grid'] or row['outer_grid']!=dict(h_exact='1/8',vmin_exact='-32',vmax_exact='64',nodes_prescribed=769,nodes_completed=769,full_weights=True):raise ValueError('fixed grid drift')
            if row['K']!=8 or row['precision_bits']!=192 or row['moment_columns']!=35 or row['backend']!='arb_mat' or row['saturation_threshold']!=128:raise ValueError('fixed policy drift')
            if row['original_labels']!=inputs['teacher']['representative_original_labels'][index] or row['weighted_class_matrices_include_teacher_moments'] is not True or row['class_weight_division_used'] is not False:raise ValueError('teacher weight policy drift')
            counts=row['resources']
            if counts['laplace_factors_evaluated']!=881*769 or counts['saturated_factor_enclosures']+counts['ordinary_exponential_calls']!=881*769:raise ValueError('factor counts drift')
            for key in ('new_kernel_calls','new_random_draws','new_optimizer_updates'):
                if counts[key]!=0:raise ValueError('unexpected extension')
            rows.append(row)
        if summary['matrix_blocks']!=aggregate_records(rows):raise ValueError('weighted checkpoint matrix aggregation drift')
        if summary['sum_W']!=inputs['teacher']['sum_W'] or summary['weighted_class_blocks_summed_without_second_teacher_factor'] is not True:raise ValueError('teacher factors repeated')
        for key in ('true_F8_Hessian_certified','true_F8_stationary_existence_certified','true_F8_local_minimum_certified','folded_mixed_bias_radial_block_provided','analytic_errors_added'):
            if summary[key] is not False:raise ValueError('finite matrix scope upgrade')
    finally:ctx.prec=previous
    return dict(status='passed',new_grid_evaluations=0,new_integrals=0,new_kernel_calls=0,new_random_draws=0,
        validation_scope='source/checkpoint endpoints/counts and weighted Arb matrix aggregation; no independent grid replay')


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-inputs',action='store_true');mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.prepare_inputs:
        OUT.mkdir(parents=True,exist_ok=True);common.write_new(INPUTS,prepared_inputs());result=dict(status='prepared',real_center_Hessian_evaluations=0)
    elif args.run:result=run()
    else:result=check()
    print(common.json.dumps(result,indent=2))


if __name__=='__main__':main()

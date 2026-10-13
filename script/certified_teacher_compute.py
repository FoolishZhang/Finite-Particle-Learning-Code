"""Exact-input Arb finite uniform-grid teacher sums, freeze gated.

This encloses a finite sum only. Infinite-grid discretization and truncation
errors must be added separately before claiming a true teacher PMF enclosure.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/finite_selection_tools'))
from flint import arb,ctx
import flint

OUT=ROOT/'artifacts/certified_teacher_v1_20261008'
INPUTS=OUT/'INPUTS.json';FREEZE=OUT/'FREEZE.json'
OLD_SOURCE='artifacts/rankone_stationary_v1_20261007/SOURCE.json'
BITS=192
PHI=('-7/10','-1/2','-3/10','-1/10','1/10','3/10','1/2','7/10')
SOURCE_NAMES=('script/certified_teacher_compute.py','script/certified_teacher_compute_test.py',
              'artifacts/certified_teacher_v1_20261008/INPUTS.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def write_new(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def exact(value):
    if isinstance(value,(float,bool)):raise ValueError('explicit exact rational required')
    return Fraction(value)


def aq(value):
    value=exact(value);return arb(value.numerator)/value.denominator


def endpoint(value):
    mantissa,exponent=value.man_exp()
    return dict(mantissa=str(mantissa),exponent_of_two=int(exponent))


def record(value):
    if not value.is_finite():raise ArithmeticError('nonfinite Arb finite-sum enclosure')
    return dict(arb_display_only=str(value),lower_exact_dyadic=endpoint(value.lower()),upper_exact_dyadic=endpoint(value.upper()))


def from_endpoint(ep):
    m=int(ep['mantissa']);e=int(ep['exponent_of_two'])
    return arb(m)*(arb(2)**e)


def from_record(data):
    return from_endpoint(data['lower_exact_dyadic']).union(from_endpoint(data['upper_exact_dyadic']))


def reverse_label(label,m):
    answer=0
    for _ in range(m):answer=(answer<<1)|(label&1);label>>=1
    return answer


def orbits(m):
    mask=(1<<m)-1;left=set(range(1<<m));output=[]
    while left:
        label=min(left);rev=reverse_label(label,m)
        orbit=sorted({label,mask-label,rev,mask-rev});output.append(orbit);left.difference_update(orbit)
    return output


def prefix_schedule(labels,m):
    return [sorted({label>>(m-level) for label in labels}) for level in range(1,m+1)]


class GridBudgetStop(RuntimeError):
    def __init__(self,partial):
        super().__init__('fixed wall budget reached; partial finite sum retained, no automatic resume')
        self.partial=partial


def validate_model(angles,row_length,mode):
    angles=tuple(exact(x) for x in angles);length=exact(row_length)
    if not angles or length<=0:raise ValueError('nonempty positive-row-length teacher required')
    if mode not in ('all_labels','complement_reversal_orbits'):raise ValueError('unknown label mode')
    if mode=='complement_reversal_orbits' and angles!=tuple(-x for x in angles[::-1]):
        raise ValueError('orbit mode requires exact opposite reversed angles')
    return angles,length


def finite_grid_block(angles,row_length,h,index_min,index_max,outer_indices,*,precision_bits=192,label_mode='all_labels',deadline=None):
    """Generic finite sum. All grid endpoints have full weights. No file I/O."""
    angles,length=validate_model(angles,row_length,label_mode);h=exact(h)
    if h<=0 or isinstance(precision_bits,bool) or not isinstance(precision_bits,int) or precision_bits<64:
        raise ValueError('positive step and integer precision >=64 required')
    if any(isinstance(x,bool) or not isinstance(x,int) for x in (index_min,index_max)) or index_min>index_max:
        raise ValueError('integer grid range required')
    outer_indices=tuple(outer_indices)
    if not outer_indices or any(isinstance(i,bool) or not isinstance(i,int) or not index_min<=i<=index_max for i in outer_indices):
        raise ValueError('outer indices outside grid')
    m=len(angles);groups=orbits(m);labels=list(range(1<<m)) if label_mode=='all_labels' else [group[0] for group in groups]
    schedule=prefix_schedule(labels,m);previous=ctx.prec;ctx.prec=precision_bits;tick=time.perf_counter()
    try:
        length_ball=aq(length);W=[(length_ball*aq(phi).cos(),length_ball*aq(phi).sin()) for phi in angles]
        step=aq(h);coords={i:step*i for i in range(index_min,index_max+1)}
        weights={i:(-coords[i]*coords[i]/2).exp()*step/(2*arb.pi()).sqrt() for i in coords}
        accum={label:arb(0) for label in labels};mass=arb(0);points=0;last_point=None
        def collect(completed):
            if any(not value.is_finite() for value in accum.values()) or not mass.is_finite():raise ArithmeticError('nonfinite finite sum')
            counts=dict(grid_points=points,teacher_dimension=m,
                label_probabilities_accumulated_per_point=len(labels),prefix_multiplications_per_point=sum(map(len,schedule)),
                logistic_exponentials_per_point=2*m,gaussian_1d_exponentials=index_max-index_min+1,
                new_kernel_calls=0,new_random_draws=0,acb_integral_calls=0)
            return dict(label_representatives=labels,probability_orbits=groups,
                representative_balls=[record(accum[label]) for label in labels],
                grid_gaussian_mass=record(mass),outer_indices=list(outer_indices),inner_index_min=index_min,inner_index_max=index_max,
                completed=completed,last_completed_point=last_point,
                precision_bits=precision_bits,label_mode=label_mode,work_counts=counts,wall_seconds=time.perf_counter()-tick,
                scope='strict Arb enclosure of finite full-weight uniform-grid sum only')
        for i in outer_indices:
            for j in range(index_min,index_max+1):
                if deadline is not None and time.perf_counter()>=deadline:raise GridBudgetStop(collect(False))
                weight=weights[i]*weights[j];mass+=weight
                probabilities=[]
                for wx,wy in W:
                    logit=wx*coords[i]+wy*coords[j]
                    probabilities.append((1/(1+logit.exp()),1/(1+(-logit).exp())))
                products={0:arb(1)}
                for level,prefixes in enumerate(schedule):
                    products={prefix:products[prefix>>1]*probabilities[level][prefix&1] for prefix in prefixes}
                for label in labels:accum[label]+=weight*products[label]
                points+=1;last_point=[i,j]
        return collect(True)
    finally:ctx.prec=previous


def prepared_inputs(mode):
    if mode!='all_labels':raise ValueError('author fixed all256 labels; orbit policy is metadata only')
    validate_model(PHI,'4/5',mode)
    return dict(schema='certified_teacher_inputs_v1',baseline_commit='4aa755b6163a2db95d9be94f74c4c9d3ccb40e9c',
        exact_teacher=dict(bias=['0']*8,row_length='4/5',angles=list(PHI),dimension=2,observations=8,
            coordinate_definition='W_j=(4/5*cos(phi_j),4/5*sin(phi_j)); all rational inputs interpreted exactly'),
        grid=dict(h='1/8',index_min=-80,index_max=80,points=25921,full_weight_at_every_point=True,
            definition='h^2 sum_{i,j=-80}^{80} normal_density_2(ih,jh)*A_y(ih,jh)',outer_rows_per_block=16),
        precision_bits=BITS,label_mode=mode,all_256_labels_returned=True,hard_wall_budget_seconds=600,
        budget_checks='between grid points and after each saved block; an in-progress Arb point is atomic',
        budget_stop_saves_partial=True,automatic_resume_or_reexecution=False,
        probability_orbits=orbits(8),representative_count=256 if mode=='all_labels' else len(orbits(8)),
        old_teacher_source=dict(path=OLD_SOURCE,sha256=sha(ROOT/OLD_SOURCE),role='future comparison only; no saved GH60 value used as certified coefficient'),
        true_PMF_enclosed=False,finite_grid_ball_requires_analytic_discretization_and_tail_error=True,
        new_optimizer_updates=0,new_random_draws=0)


def verify_freeze():
    inputs=read(INPUTS)
    if inputs!=prepared_inputs(inputs['label_mode']):raise ValueError('fixed teacher/grid/input drift')
    sources=read(FREEZE).get('sources')
    if not isinstance(sources,dict) or any(name not in sources for name in SOURCE_NAMES):raise ValueError('FREEZE omits compute/test/INPUTS')
    for name,digest in sources.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path)!=digest:raise ValueError('frozen source drift: '+name)
    return inputs


def aggregate_blocks(blocks,m,mode):
    previous=ctx.prec;ctx.prec=BITS
    try:
        if mode=='complement_reversal_orbits':
            indices=[i for block in blocks for i in block['outer_indices']]
            lo,hi=blocks[0]['inner_index_min'],blocks[0]['inner_index_max']
            if lo!=-hi or sorted(indices)!=list(range(lo,hi+1)) or any(block.get('completed') is not True for block in blocks):
                raise ValueError('orbit lifting requires the complete reflection-invariant square, not a partial block')
        labels=list(range(1<<m)) if mode=='all_labels' else [group[0] for group in orbits(m)]
        sums={label:arb(0) for label in labels};mass=arb(0)
        for block in blocks:
            if block['label_representatives']!=labels or block['label_mode']!=mode:raise ValueError('block label policy drift')
            for label,data in zip(labels,block['representative_balls']):sums[label]+=from_record(data)
            mass+=from_record(block['grid_gaussian_mass'])
        values=[None]*(1<<m)
        if mode=='all_labels':
            for label in labels:values[label]=sums[label]
        else:
            for group in orbits(m):
                for label in group:values[label]=sums[group[0]]
        return dict(pmf_finite_sum=[record(value) for value in values],grid_gaussian_mass=record(mass),
            sum_of_probability_enclosures=record(sum(values,arb(0))),
            probability_sum_overlaps_grid_mass=sum(values,arb(0)).overlaps(mass),
            complement_symmetry_balls_overlap=all(values[label].overlaps(values[(1<<m)-1-label]) for label in range(1<<m)),
            reversal_symmetry_balls_overlap=all(values[label].overlaps(values[reverse_label(label,m)]) for label in range(1<<m)),
            true_PMF_enclosed=False,analytic_discretization_and_tail_error_added=False)
    finally:ctx.prec=previous


def run():
    inputs=verify_freeze();start=OUT/'RUN_START.json'
    if start.exists():raise FileExistsError('RUN_START exists; no overwrite/resume')
    tick=time.perf_counter();deadline=tick+inputs['hard_wall_budget_seconds']
    write_new(start,dict(created_utc=now(),inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE),python_flint_version=flint.__version__,hard_wall_budget_seconds=600))
    spec=inputs['grid'];teacher=inputs['exact_teacher'];blocks=[];artifacts=[]
    rows=list(range(spec['index_min'],spec['index_max']+1))
    for number,offset in enumerate(range(0,len(rows),spec['outer_rows_per_block'])):
        try:
            block=finite_grid_block(teacher['angles'],teacher['row_length'],spec['h'],spec['index_min'],spec['index_max'],
                rows[offset:offset+spec['outer_rows_per_block']],precision_bits=BITS,label_mode=inputs['label_mode'],deadline=deadline)
        except GridBudgetStop as stopped:
            partial=stopped.partial;partial.update(schema='certified_teacher_partial_grid_block_v1',block_index=number,
                inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE),not_a_complete_grid=True)
            path=OUT/f'partial_block_{number:02d}.json';write_new(path,partial)
            stop=dict(status='budget_stopped',created_utc=now(),completed_blocks=artifacts,
                partial_block=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size),
                inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE),wall_seconds=time.perf_counter()-tick,
                automatic_resume=False,real_teacher_grid_completed=False)
            write_new(OUT/'RUN_STOP.json',stop);return stop
        block.update(schema='certified_teacher_grid_block_v1',block_index=number,inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE))
        path=OUT/f'block_{number:02d}.json';write_new(path,block);blocks.append(block)
        artifacts.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),bytes=path.stat().st_size))
        if time.perf_counter()>=deadline:
            stop=dict(status='budget_stopped',created_utc=now(),completed_blocks=artifacts,
                inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE),wall_seconds=time.perf_counter()-tick,
                automatic_resume=False,real_teacher_grid_completed=False)
            write_new(OUT/'RUN_STOP.json',stop);return stop
    output=aggregate_blocks(blocks,8,inputs['label_mode']);output.update(schema='certified_teacher_finite_grid_sum_v1',
        created_utc=now(),inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE),run_start_sha256=sha(start),blocks=artifacts,
        probability_orbits=inputs['probability_orbits'],label_mode=inputs['label_mode'],precision_bits=BITS,
        resources=dict(scientific_finite_grid_calls=len(blocks),grid_points=sum(b['work_counts']['grid_points'] for b in blocks),
            new_random_draws=0,new_optimizer_updates=0,acb_integral_calls=0,wall_seconds=time.perf_counter()-tick))
    if not all(output[key] for key in ('probability_sum_overlaps_grid_mass','complement_symmetry_balls_overlap','reversal_symmetry_balls_overlap')):
        raise ArithmeticError('mass/symmetry finite-grid consistency check failed')
    write_new(OUT/'FINITE_GRID_SUM.json',output)
    return dict(status='completed',**output['resources'])


def check():
    inputs=verify_freeze();saved=read(OUT/'FINITE_GRID_SUM.json');blocks=[];rows=[]
    if saved['inputs_sha256']!=sha(INPUTS) or saved['freeze_sha256']!=sha(FREEZE) or saved['run_start_sha256']!=sha(OUT/'RUN_START.json'):raise ValueError('saved locks drift')
    for number,entry in enumerate(saved['blocks']):
        path=ROOT/entry['path']
        if sha(path)!=entry['sha256'] or path.stat().st_size!=entry['bytes']:raise ValueError('block hash drift')
        block=read(path)
        if block['inputs_sha256']!=sha(INPUTS) or block['freeze_sha256']!=sha(FREEZE) or block['block_index']!=number:raise ValueError('block lock drift')
        if block.get('completed') is not True or block['inner_index_min']!=-80 or block['inner_index_max']!=80:raise ValueError('incomplete or altered grid block')
        if block['work_counts']['grid_points']!=len(block['outer_indices'])*161:raise ValueError('point count drift')
        rows.extend(block['outer_indices']);blocks.append(block)
    if rows!=list(range(-80,81)) or len(blocks)!=11:raise ValueError('grid partition drift')
    expected=aggregate_blocks(blocks,8,inputs['label_mode'])
    for key,value in expected.items():
        if saved[key]!=value:raise ValueError('saved finite-sum aggregation drift: '+key)
    if not all(expected[key] for key in ('probability_sum_overlaps_grid_mass','complement_symmetry_balls_overlap','reversal_symmetry_balls_overlap')):
        raise ValueError('saved finite-grid consistency check failed')
    return dict(status='passed',new_scientific_grid_calls=0,new_integrals=0,new_random_draws=0,
        validation_scope='frozen sources/block endpoints and Arb endpoint aggregation; not independent grid replay')


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-inputs',choices=('all_labels','complement_reversal_orbits'));mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.prepare_inputs:OUT.mkdir(parents=True,exist_ok=True);write_new(INPUTS,prepared_inputs(args.prepare_inputs));result=dict(status='prepared',scientific_grid_calls=0)
    elif args.run:result=run()
    else:result=check()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

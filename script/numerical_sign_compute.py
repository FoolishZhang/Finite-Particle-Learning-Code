"""Development qualification for the saved-input numerical-sign backend.

No full scientific-bank execution is performed by the qualification action.
The primary agent must inspect its receipt and freeze sources/inputs before
authorizing a separate full pass. Existing scientific files are never edited.
"""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import itertools
import json
import math
import sys
import time
import traceback

import numpy as np
import numerical_sign_kernel as kernel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/numerical_sign_budget_v1_20261006'
DOC=ROOT/'docs/08_research_研究工作/NumericalSignBudget_v1_20261006'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save_new(path,record):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def qualify():
    OUT.mkdir(parents=True,exist_ok=True)
    attempt=1
    while (OUT/f'QUALIFICATION_ATTEMPT_{attempt:03d}.json').exists():
        attempt+=1
    target=OUT/f'QUALIFICATION_ATTEMPT_{attempt:03d}.json'
    sources=[Path(__file__),ROOT/'script/numerical_sign_kernel.py',DOC/'CONTRACT.md',DOC/'ERROR_BUDGET.md']
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),action='manufactured_qualification',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    try:
        record['qualification']=kernel.qualification()
        record['summary_qualification']=summary_qualification()
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        save_new(target,record)
        raise
    record['status']='passed'
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    save_new(target,record)
    print(json.dumps(dict(status='passed',receipt=target.relative_to(ROOT).as_posix(),benchmark=record['qualification']['benchmark'],maximum_fixture_error_enclosed=record['qualification']['maximum_fixture_error_enclosed']),ensure_ascii=False),flush=True)


def reliable_summary(reference,errors):
    """All reductions outward; only a few scalar roots/logs use 192-bit Arb.

    reference and errors are interpreted as exact saved binary64 numbers;
    errors must already be outward upper bounds. This propagates all sample
    errors, but does not certify input generation or an iid sampling model.
    """
    reference=np.asarray(reference,dtype=np.float64);errors=np.asarray(errors,dtype=np.float64)
    if reference.ndim!=1 or errors.shape!=reference.shape or len(reference)<2:
        raise ValueError('one-dimensional equally sized samples of length at least two required')
    if not np.isfinite(reference).all() or not np.isfinite(errors).all() or np.any(errors<0):
        raise ValueError('nonfinite values or negative error bound')
    N=len(reference)
    mean=kernel.positive_div(kernel.pairwise_sum(kernel.point(reference)),kernel.point(float(N)))
    centered=kernel.sub(kernel.point(reference),mean)
    squares=kernel.known_range(kernel.mul(centered,centered),0.,np.inf)
    variance=kernel.positive_div(kernel.pairwise_sum(squares),kernel.point(float(N-1)))
    mean_error=kernel.positive_div(kernel.pairwise_sum(kernel.point(errors)),kernel.point(float(N)))
    squared_error=kernel.mul(kernel.point(errors),kernel.point(errors))
    variance_error=kernel.positive_div(kernel.pairwise_sum(squared_error),kernel.point(float(N-1)))
    private_tools=ROOT/'.cache/finite_selection_tools'
    if str(private_tools) not in sys.path:sys.path.insert(0,str(private_tools))
    from flint import arb,ctx
    previous=ctx.prec;ctx.prec=192
    try:
        logterm=arb(1200).log();c=(2*logterm/N).sqrt();range_term=7*(arb(133)/16)*logterm/(3*(N-1))
        sd_upper=arb(float(variance.hi)).sqrt()
        sd_error_upper=arb(float(variance_error.hi)).sqrt()
        half=c*(sd_upper+sd_error_upper)+range_term
        upper=arb(float(mean.hi))+arb(float(mean_error.hi))+half
        lower=arb(float(mean.lo))-arb(float(mean_error.hi))-half
        # Conversion to binary64 is itself enclosed by another outward step.
        upper_float=math.nextafter(float(upper.upper()),math.inf)
        lower_float=math.nextafter(float(lower.lower()),-math.inf)
        multiplier=1+(2*logterm/(N-1)).sqrt()
        old_upper=arb('-0.0036181078957699285');old_allowance=arb('0.000000001')
        epsilon_threshold=(-old_upper+old_allowance)/multiplier
        return dict(N=N,alpha_family=.01,alpha_component=.01/3,range_length=133/16,original_family_reused=True,new_statistical_comparisons=0,reference_mean_enclosure=[float(mean.lo),float(mean.hi)],reference_variance_enclosure=[float(variance.lo),float(variance.hi)],mean_error_enclosure=[float(mean_error.lo),float(mean_error.hi)],sample_sd_upper_arb=str(sd_upper),sample_sd_perturbation_upper_arb=str(sd_error_upper),range_term_arb=str(range_term),candidate_interval_under_input_and_sampling_assumptions=[lower_float,upper_float],upper_arb=str(upper),lower_arb=str(lower),uniform_error_multiplier_arb=str(multiplier),epsilon_threshold_illustration_arb=str(epsilon_threshold),numerical_qualification_remaining=['saved S versus exact nuisance transformations','Gaussian generation and ideal iid sampling model'],saved_input_only=True)
    finally:
        ctx.prec=previous


def summary_qualification():
    """Independent exact-rational moments and all corners of a fixed error box."""
    private_tools=ROOT/'.cache/finite_selection_tools'
    if str(private_tools) not in sys.path:sys.path.insert(0,str(private_tools))
    from flint import arb,ctx
    previous=ctx.prec;ctx.prec=192
    try:
        samples=[[-.75,.25,.5,0.],[.25,.25,.25,.25]]
        errors=np.asarray([.125,.25,0.,.0625])
        count=0
        for values in samples:
            y=np.asarray(values);summary=reliable_summary(y,errors);N=len(y)
            exact_y=[Fraction.from_float(float(v)) for v in y]
            mean=sum(exact_y)/N;variance=sum((v-mean)**2 for v in exact_y)/(N-1)
            ml,mh=summary['reference_mean_enclosure'];vl,vh=summary['reference_variance_enclosure']
            assert Fraction.from_float(ml)<=mean<=Fraction.from_float(mh)
            assert Fraction.from_float(vl)<=variance<=Fraction.from_float(vh)
            for signs in itertools.product((-1,1),repeat=N):
                x=[v+sgn*Fraction.from_float(float(e)) for v,sgn,e in zip(exact_y,signs,errors)]
                xm=sum(x)/N;xv=sum((v-xm)**2 for v in x)/(N-1)
                ar=lambda q:arb(q.numerator)/q.denominator
                logterm=arb(1200).log()
                exact_bound=ar(xm)+(2*logterm/N).sqrt()*ar(xv).sqrt()+7*(arb(133)/16)*logterm/(3*(N-1))
                assert exact_bound < arb(summary['candidate_interval_under_input_and_sampling_assumptions'][1])
                count+=1
        N=262144;factor=1+(2*arb(1200).log()/(N-1)).sqrt()
        margin=arb('0.0036181078957699285')+arb('1e-9')
        budget=dict(multiplier=str(factor),epsilon_threshold=str(margin/factor),upper_if_epsilon_1e_4=str(-margin+factor*arb('1e-4')),upper_if_epsilon_1e_3=str(-margin+factor*arb('1e-3')),illustrative_only=True)
        return dict(status='passed',exact_rational_moment_cases=len(samples),error_box_corners=count,random_draws=0,analytic_budget=budget)
    finally:ctx.prec=previous


def run(freeze_path):
    """Execute only after the primary agent has frozen and authorized inputs."""
    started=time.monotonic();completed=0
    fpath=Path(freeze_path).resolve();freeze=json.loads(fpath.read_text(encoding='utf-8'))
    required=dict(scope='exact_saved_binary64_Z_S_n256_only',N=262144,K=100,block=256,timeout_seconds=1200,epsilon_limit=1e-4)
    for key,value in required.items():
        if freeze.get(key)!=value:raise ValueError(f'freeze {key} must equal {value!r}')
    for name,digest in freeze['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'changed frozen source: {name}')
    qualification_path=ROOT/freeze['qualification_path']
    if sha(qualification_path)!=freeze['qualification_sha256']:raise ValueError('qualification hash mismatch')
    qualification=json.loads(qualification_path.read_text(encoding='utf-8'))
    if qualification['status']!='passed':raise ValueError('qualification did not pass')
    for name in ('script/numerical_sign_kernel.py','script/numerical_sign_compute.py'):
        if qualification['source_sha256'].get(name)!=sha(ROOT/name):raise ValueError(f'qualification predates code: {name}')
    input_path=(ROOT/freeze['input_path']).resolve()
    expected=(ROOT/'artifacts/local_geometry_v1_20261006/BANKS.npz').resolve()
    if input_path!=expected:raise ValueError('only the original local-geometry saved bank is allowed')
    if sha(input_path)!=freeze['input_sha256']:raise ValueError('scientific input hash mismatch')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('RUN_START.json','RESULTS.json','INTERVALS.npz','FAILURE.json','PARTIAL_INTERVALS.npz'):
        if (OUT/name).exists():raise FileExistsError(f'refusing to overwrite {name}')
    save_new(OUT/'RUN_START.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_path=str(fpath),freeze_sha256=sha(fpath),input_sha256=freeze['input_sha256'],N=freeze['N'],K=freeze['K'],random_draws=0))
    N=freeze['N'];K=freeze['K'];block=freeze['block']
    pair_lo=np.empty((N,3));pair_hi=np.empty((N,3));mean_lo=np.empty(N);mean_hi=np.empty(N);reference=np.empty(N);errors=np.empty(N)
    try:
        with np.load(input_path,allow_pickle=False) as original:
            z=original['z'];s=original['s'];old=original['kernels']
        if z.shape!=(N,K,3) or s.shape!=(N,K) or old.shape!=(N,2,3):raise ValueError('frozen shapes changed')
        for first in range(0,N,block):
            last=min(first+block,N)
            pairs,average=kernel.bound_curvature(z[first:last],s[first:last])
            pair_lo[first:last]=pairs.lo;pair_hi[first:last]=pairs.hi
            mean_lo[first:last]=average.lo;mean_hi[first:last]=average.hi
            # Any floating averaging here is included by comparison to the
            # exact saved-input average enclosure, not assumed exact.
            y=old[first:last,1,:].mean(axis=1)
            reference[first:last]=y
            difference=kernel.sub(average,kernel.point(y))
            errors[first:last]=np.maximum(np.abs(difference.lo),np.abs(difference.hi))
            completed=last
            elapsed=time.monotonic()-started
            if elapsed>freeze['timeout_seconds']:raise TimeoutError('fixed 20-minute full-pass budget exceeded')
            if last%32768==0:print(json.dumps(dict(stage='saved_input_interval',completed=last,total=N,wall_seconds=elapsed)),flush=True)
        summary=reliable_summary(reference,errors)
        if time.monotonic()-started>freeze['timeout_seconds']:raise TimeoutError('fixed 20-minute full-pass budget exceeded before save')
        output=OUT/'INTERVALS.npz'
        np.savez_compressed(output,index=np.arange(N,dtype=np.int64),pair_lower=pair_lo,pair_upper=pair_hi,mean_lower=mean_lo,mean_upper=mean_hi,reference=reference,error_bound=errors)
        maximum_error=float(np.max(errors));upper=summary['candidate_interval_under_input_and_sampling_assumptions'][1]
        elapsed=time.monotonic()-started
        passed=bool(maximum_error<=freeze['epsilon_limit'] and upper<0 and elapsed<=freeze['timeout_seconds'])
        result=dict(status='passed' if passed else 'failed_gate',finished_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(fpath),input_sha256=freeze['input_sha256'],completed_banks=completed,total_banks=N,pair_components=3*N,random_draws=0,new_scientific_banks=0,new_statistical_comparisons=0,wall_seconds=elapsed,maximum_saved_kernel_error_enclosed=maximum_error,error_limit=freeze['epsilon_limit'],maximum_pair_interval_width=float(np.max(pair_hi-pair_lo)),all_saved_inputs_enclosed=True,saved_nuisance_transform_certified=False,Gaussian_generation_certified=False,PRNG_certified=False,deterministic_population_sign_proved=False,summary=summary,gates=dict(error_budget=maximum_error<=freeze['epsilon_limit'],conditional_negative_upper=upper<0,deadline=elapsed<=freeze['timeout_seconds']),output=dict(path=output.relative_to(ROOT).as_posix(),sha256=sha(output),bytes=output.stat().st_size))
        save_new(OUT/'RESULTS.json',result)
        print(json.dumps(dict(status=result['status'],completed=completed,max_error=maximum_error,conditional_upper=upper,wall_seconds=elapsed)),flush=True)
    except BaseException as error:
        if completed:
            np.savez_compressed(OUT/'PARTIAL_INTERVALS.npz',pair_lower=pair_lo[:completed],pair_upper=pair_hi[:completed],mean_lower=mean_lo[:completed],mean_upper=mean_hi[:completed],reference=reference[:completed],error_bound=errors[:completed])
        save_new(OUT/'FAILURE.json',dict(status='failed',completed_banks=completed,total_banks=N,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started,full_coverage_claimed=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('qualify','run'));parser.add_argument('--freeze');args=parser.parse_args()
    if args.action=='qualify':qualify()
    else:
        if args.freeze is None:parser.error('run requires --freeze from the primary-agent source/input freeze')
        run(args.freeze)

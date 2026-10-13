"""Timing only: shared-input 16-model evaluation, with no scientific RNG."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'script'))
import numpy as np
import scbmeval_kernel as kernel


def trial():
    B,K,n=64,100,256
    active,noise,base_tau=kernel.deterministic_inputs(B,K,n)
    pmf=np.full(16,1/16)
    empirical=[]
    for i in range(16):
        p=(np.arange(1,17,dtype=float)+i)%17+1;p/=p.sum();empirical.append(p)
    empirical=np.asarray(empirical)
    contexts=[];tick=time.perf_counter()
    for model in range(16):
        for kind in ('TAIL','LAST'):
            tau=base_tau*(1.+.03*np.sin(model+.7 if kind=='TAIL' else model+.1))
            alphas=(1.,.5) if kind=='TAIL' else (1.,)
            cache=kernel.prepare_noise(noise,tau,alphas)
            contexts.append(dict(model=model,kind=kind,t=.017*(model-7)+(0.001 if kind=='LAST' else 0.),cache=cache))
    preparation=time.perf_counter()-tick;tick=time.perf_counter();checksum=0.
    zero=np.zeros((B,K));fact_weights=np.column_stack((pmf,empirical.T))
    for t in np.linspace(-.75,.75,21):
        terms=kernel.active_terms(active,float(t),need_path=True)
        factor=kernel.from_terms(terms,zero,n)
        # One FACT bank-pattern array, separately weighted for population and
        # all 16 empirical active distributions.
        for array in factor.values():checksum+=float((array@fact_weights).sum())
        for context in contexts:
            weights=np.column_stack((pmf,empirical[context['model']]))
            for noise_weight in context['cache']['centered_logweights'].values():
                answer=kernel.from_terms(terms,noise_weight,n)
                for array in answer.values():checksum+=float((array@weights).sum())
    grid_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    for context in contexts:
        terms=kernel.active_terms(active,context['t'],need_path=True)
        weights=np.column_stack((pmf,empirical[context['model']]))
        for noise_weight in [zero,*context['cache']['centered_logweights'].values()]:
            answer=kernel.from_terms(terms,noise_weight,n)
            for array in answer.values():checksum+=float((array@weights).sum())
    point_seconds=time.perf_counter()-tick
    return dict(noise_preparation_seconds=preparation,shared_grid_seconds=grid_seconds,
                saved_output_t_seconds=point_seconds,total_seconds=preparation+grid_seconds+point_seconds,checksum=checksum)


if __name__=='__main__':
    warmup=trial();trials=[trial() for _ in range(3)]
    assert len({r['checksum'] for r in [warmup,*trials]})==1
    median=float(np.median([r['total_seconds'] for r in trials]))
    active,noise,tau=kernel.deterministic_inputs(B=4,K=100,n=256)
    controls=[]
    for _ in range(3):
        start=time.perf_counter();result=kernel.official32_pattern_means(active,noise,tau,.13,1.)
        controls.append(time.perf_counter()-start)
        assert result['official_API_calls']==16
    output=dict(source_sha256=kernel.sha(kernel.SOURCE),benchmark_helper_sha256=kernel.sha(__file__),
        banks_per_block=64,K=100,n=256,models=16,contexts=['TAIL','LAST'],
        nuisance_profiles=48,shared_grid_candidates=21,shared_FACT_grid_arrays=21,
        total_grid_bank_pattern_evaluations=21*49,additional_output_t_candidates=32,
        additional_output_t_bank_pattern_evaluations=80,
        trials=trials,median_block_seconds=median,
        projected_32rep_B512_seconds=median*32*512/64,
        projected_32rep_B512_minutes=median*32*512/64/60,
        projected_32rep_B256_minutes=median*32*256/64/60,
        projected_32rep_B1024_minutes=median*32*1024/64/60,
        official32_control=dict(prefix_banks=4,K=100,one_candidate_alpha_calls=16,
            seconds=controls,median_seconds=float(np.median(controls))),
        scope='All tensors are fixed sine/cosine arrays and artificial frozen-scale/weight vectors. Shared inputs, grid active terms and FACT are cached across models. Price and path both timed. No science outcome read, no convergence pilot, no scientific RNG or integral.',
        exclusions='Scientific RNG, IO, aggregation/selection and independent official32 controls excluded from total projection; controls reported separately.',
        new_scientific_RNG_draws=0,new_model_integrals=0,new_fits=0)
    kernel.save_new('SHARED_BENCHMARK.json',output)
    print(json.dumps(output,indent=2))

"""Posthoc exact-sign diagnostic; no new simulated inputs or hypothesis test."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.interpolate import PPoly

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/finite_calibration'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    dest=OUT/'ENDPOINT_DIAGNOSTIC';dest.mkdir(exist_ok=True)
    paths=[OUT/'TRAIN/RESULTS.json',OUT/'CAL/MODEL.json']
    save(dest/'PROTOCOL.json',dict(source_sha256=sha(__file__),posthoc=True,
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        claim='For any fixed data, exact conditional raw endpoint gap is 2*t_N*V(R), V(R)<0. Compare its known sign to actual saved raw endpoint gap.',
        positive_target_ties='t_N=0 is an expectation tie, excluded from opposite-sign count',
        arithmetic_near_tie=1e-12,new_random_draws=0,new_tests=False,no_causal_percentage_claim=True))
    train,model=map(read,paths);case=next(c for c in model['cases'] if c['n']==254)
    gp=PPoly(case['gprime_spline_c'],case['gprime_spline_x']);g=gp.antiderivative()
    parity=float(g(.75)-g(-.75));assert abs(parity)<1e-12
    rows=[]
    for stream in train['records']:
        with np.load(ROOT/stream['file'],allow_pickle=False) as arr:
            means=arr['raw'].mean(axis=0)
            for ti,theta in enumerate(train['config']['teachers']):
                for ni,N in enumerate(train['config']['sample_sizes']):
                    pair=int(np.sum(arr['y1'][:N].astype(np.int64)*arr['y2'][ti,:N]))
                    raw=next(r for r in stream['outputs'] if r['theta']==theta and r['N']==N and r['method']=='raw')
                    t=pair/N;assert t==raw['empirical_pair']
                    gap=float(means[ti,ni,-1]-means[ti,ni,0])
                    expected_sign=-int(np.sign(pair));actual_sign=int(np.sign(gap)) if abs(gap)>1e-12 else 0
                    rows.append(dict(stream=stream['stream'],theta=theta,N=N,pair_sum=pair,t_N=t,
                        actual_raw_endpoint_gap=gap,expected_conditional_gap_sign=expected_sign,
                        actual_gap_sign=actual_sign,opposite=bool(expected_sign*actual_sign<0),
                        expectation_tie=pair==0,numeric_gap_tie=abs(gap)<=1e-12,raw_rho=raw['rho']))
    summary=[]
    for theta in [0.,.3,.6]:
        for N in [1024,4096]:
            rr=[r for r in rows if r['theta']==theta and r['N']==N]
            summary.append(dict(theta=theta,N=N,opposite_count=sum(r['opposite'] for r in rr),
                expectation_ties=sum(r['expectation_tie'] for r in rr),numeric_ties=sum(r['numeric_gap_tie'] for r in rr)))
    save(dest/'RESULTS.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(dest/'PROTOCOL.json'),rows=rows,
        summary=summary,finite_spline_endpoint_penalty_difference=parity,
        not_a_population_frequency_or_new_independent_confirmation=True,new_random_draws=0))
    print(json.dumps(dict(summary=summary,positive_theta_highN_opposite=[r for r in rows if r['theta']>0 and r['N']==4096 and r['opposite']])))


if __name__=='__main__':main()

"""Independent log-space DP audit on existing first batches; no new streams."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import numpy as np
from scipy.special import expit, log_expit, logsumexp, gammaln
from scipy.stats import t

HERE=Path(__file__).resolve().parent


def projected_gaussian(z,d):
    a=math.sqrt(1+.9/(d-1));b=math.sqrt(.1)
    matrix=4.*(a*np.eye(d)+(b-a)*np.ones((d,d))/d)
    return z@matrix.T


def log_poisson_binomial(eta):
    n,d=eta.shape
    mass=np.full((n,d+1),-np.inf);mass[:,0]=0.
    logp=log_expit(eta);logq=log_expit(-eta)
    for j in range(d):
        mass[:,1:j+2]=np.logaddexp(mass[:,1:j+2]+logq[:,j,None],mass[:,:j+1]+logp[:,j,None])
        mass[:,0]+=logq[:,j]
    return mass


def main():
    rows=list(csv.DictReader((HERE/'STREAM_LOSSES.csv').open(encoding='utf-8')))
    primary=list(csv.DictReader((HERE/'PRIMARY.csv').open(encoding='utf-8')))
    tcrit=t.ppf(1-.05/40,7);stat_error=0.;stored_mean_error=0.
    for p in primary:
        a=np.array([float(r['mean_F_minus_F0']) for r in rows if (r['family'],r['d'],r['method'])==(p['family'],p['d'],p['method'])])
        assert len(a)==8
        mu=a.mean();se=a.std(ddof=1)/np.sqrt(8)
        stat_error=max(stat_error,abs(mu-float(p['mean'])),abs(mu-tcrit*se-float(p['lower'])),abs(mu+tcrit*se-float(p['upper'])))
    methods=['original_IID10','original_IID100','original_ANTI100','count_RB_IID1','count_RB_IID10']
    for p in sorted((HERE/'scores').glob('*.npz')):
        with np.load(p) as archive:
            d=int(p.stem.split('_d')[1].split('_')[0]);family=p.stem.split('_d')[0];stream=int(p.stem.split('stream')[1])
            difference=archive['losses']-d*np.log(2)
            assert np.array_equal(difference,archive['differences'])
            for j,method in enumerate(methods):
                r=next(r for r in rows if (r['family'],int(r['d']),int(r['stream']),r['method'])==(family,d,stream,method))
                stored_mean_error=max(stored_mean_error,abs(difference[:,j].mean()-float(r['mean_F_minus_F0'])))
    detail=[]
    for d in [24,128]:
        for stream in range(8):
            seed=20271901+100*d+stream;teacher=np.random.default_rng(seed);prior=np.random.default_rng(seed+20000000)
            eta_teacher=projected_gaussian(teacher.normal(size=(64,d)),d)
            count=(teacher.random((64,d))<expit(eta_teacher)).sum(1)
            eta=projected_gaussian(prior.normal(size=(64*10,d)),d)
            logmass=log_poisson_binomial(eta)
            selected=logmass[np.arange(64*10),np.repeat(count,10)].reshape(64,10)
            lc=gammaln(d+1)-gammaln(count+1)-gammaln(d-count+1)
            logq=selected-lc[:,None]
            rb1=-logq[:,0];rb10=-logsumexp(logq,axis=1)+np.log(10)
            with np.load(HERE/'scores'/f'NEG_d{d}_stream{stream}.npz') as archive:
                assert np.array_equal(count,archive['count'][:64]),'existing teacher replay differs'
                error=float(max(np.max(abs(rb1-archive['losses'][:64,3])),np.max(abs(rb10-archive['losses'][:64,4]))))
            detail.append(dict(d=d,stream=stream,reused_observations=64,independent_log_DP_loss_max_error=error,
                               log_DP_normalization_max_error=float(np.max(abs(logsumexp(logmass,axis=1))))))
    maximum=max(r['independent_log_DP_loss_max_error'] for r in detail)
    assert max(stat_error,stored_mean_error,maximum)<1e-10
    receipt=dict(time_utc=datetime.now(timezone.utc).isoformat(),status='pass_existing_stream_replay_log_DP_and_public_statistics',
                 new_observations=0,new_streams=0,existing_observations_replayed=16*64,
                 existing_RB_particles_replayed=16*64*10,statistics_max_error=float(stat_error),stored_mean_max_error=float(stored_mean_error),
                 independent_log_DP_loss_max_error=maximum,detail=detail,
                 original_COMPLETE_sha256=hashlib.sha256((HERE/'COMPLETE.json').read_bytes()).hexdigest(),
                 audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),endpoint_or_rule_changes=False)
    output=HERE/'POSTCHECK.json';assert not output.exists()
    output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print({k:v for k,v in receipt.items() if k!='detail'})


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()

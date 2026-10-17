"""Planned descriptive summaries from saved complete coefficients; no inference selection."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ALPHA=.3201223434446316
V=np.array([1.,1.,0.])/math.sqrt(2)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def stat(a):
    a=np.asarray(a,float);assert a.shape==(16,) and np.isfinite(a).all()
    return {'replicates':a.tolist(),'mean':float(a.mean()),'SE':float(a.std(ddof=1)/4)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert not args.out.exists()
    sources={};rows=[]
    for phase in ('coefficients','validation'):
        for rep in range(3):
            for n in (16384,65536):
                p=args.root/f'{phase}_rep{rep}_001/rep_{rep}_N{n}_sym/COEFFICIENT_STATISTICS.json'
                sources[p.resolve().relative_to(ROOT).as_posix()]=sha(p)
                a=json.loads(p.read_text(encoding='utf-8'));s=a['statistics'];items={}
                for tag in ('cv','q2'):
                    h=np.array(s[tag+'_H']['replicates']);b=np.array(s[tag+'_B']['replicates'])
                    assert h.shape==b.shape==(16,3,3)
                    hp=.5*(ALPHA*ALPHA-1)*np.einsum('i,rij,j->r',V,h,V)
                    bp=-(ALPHA-1)*np.einsum('i,rij,j->r',V,b,V)
                    ht=np.trace(h,axis1=1,axis2=2)/3;bt=np.trace(b,axis1=1,axis2=2)/3
                    covariance=float(np.cov(ht,bt,ddof=1)[0,1]/16)
                    rr=float(bt.mean()/ht.mean()) if ht.mean()>0 else None
                    delta=bt/ht.mean()-bt.mean()*ht/ht.mean()**2 if rr is not None else None
                    items[tag]={'Q':stat(hp+bp),'H_quadratic':stat(hp),'B_quadratic':stat(bp),
                                'H_B_sample_covariance':float(np.cov(hp,bp,ddof=1)[0,1]),
                                'trace_H_over3':stat(ht),'trace_B_over3':stat(bt),
                                'trace_ratio':rr,'trace_ratio_delta_SE':float(delta.std(ddof=1)/4) if rr is not None else None,
                                'mean_h_b_covariance':covariance}
                ratio=(items['q2']['Q']['SE']/items['cv']['Q']['SE'])**2 if items['cv']['Q']['SE']>0 else None
                rows.append({'phase':phase,'rep':rep,'N':n,'kind':'sym','controls':items,
                             'q2_over_base_Q_variance':ratio,
                             'variance_ratio_status':'defined' if ratio is not None else 'undefined_zero_baseline_variance'})
    record={'schema':'learned_precision_descriptive_v1','source_sha256':sha(Path(__file__)),
            'input_sha256':sources,'rows':rows,'all12_stage_case_rows_retained':True,
            'new_tests_or_family_claims':False,'new_scientific_calls':0,
            'scope':'Planned original/new-control, complete covariance and local trace summaries; conditional numerical SE, no new training-generalization test or global-optimum claim; full matrices, not trace summaries, are used in primary predictions.'}
    with args.out.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps({'rows':len(rows),'output':str(args.out),'sha256':sha(args.out)}))


if __name__=='__main__':main()

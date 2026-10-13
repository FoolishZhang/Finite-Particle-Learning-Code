"""Summarize the frozen population panel with teacher-independent corrections."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, log_ndtr
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/regularization_learning/GROWTH'
REF=ROOT/'artifacts/regularization_learning/NUMERIC/GROWTH_REFERENCE.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    assert not (OUT/'SUMMARY.json').exists()
    result,ref=read(OUT/'RESULTS.json'),read(REF)
    rho=np.array(result['rho']);ns=np.array(result['ns']);signs=np.array(result['signs'])
    assert np.allclose(rho,ref['rho_grid21'],atol=1e-15,rtol=0)
    protocol=dict(source_sha256=sha(__file__),production_sha256=sha(OUT/'RESULTS.json'),reference_sha256=sha(REF),
        precision_SE_threshold=5e-4,qualification='all 360 adjacent contrasts, raw and marg, 3n x 3theta x 20pairs; no posthoc budget',
        J_method='Emax = integral_0^12 [1-Phi(x)^K-(1-Phi(x))^K] dx',
        J_epsabs=[1e-11,1e-12],J_epsrel=[1e-11,1e-12],J_agreement=1e-10,
        pointwise_intervals='Student t across 16 independent streams, development approximation, no simultaneous coverage or selected-minimum guarantee',
        selections='36 fixed-candidate mean-score selections, not continuous optima or observed-data fits')
    save(OUT/'SUMMARY_PROTOCOL.json',protocol)
    modes={key:np.array([rec[key+'_mean'] for rec in result['records']]) for key in ['raw','marg']}
    J=[]
    for n in ns:
        k=int(n+2);rows=[]
        def integrand(x):return -np.expm1(k*log_ndtr(x))-np.exp(k*log_ndtr(-x))
        for tol in [1e-11,1e-12]:
            val,err,info=quad(integrand,0.,12.,epsabs=tol,epsrel=tol,full_output=True)
            rows.append(dict(tolerance=tol,value=val,quad_error_estimate=err,evaluations=info['neval']))
        # Tail upper bound integral_12^infty K*P(G>x)dx <= K*phi(12).
        tail=k*np.exp(-72)/np.sqrt(2*np.pi)
        J.append(dict(n=int(n),K=k,main=rows[0]['value'],variants=rows,tail_upper_bound=tail,
            pass_=abs(rows[0]['value']-rows[1]['value'])<=1e-10))
    save(OUT/'J_REFERENCE.json',dict(rows=J,new_scalar_integrals=6,
        scope='Analytic tail bound plus numerical quadrature qualification; reported quad error is not a rigorous total certificate'))
    curves=[];gates=[];summaries=[];stat_reference=[]
    tcrit=float(student_t.ppf(.975,len(result['records'])-1))
    H=np.array(ref['H_grid21']);A=ref['A'];gamma=ref['gamma0']
    for teacher in ref['teachers']:
        theta=teacher['theta'];moment=teacher['r_theta'];weights=(1+moment*signs[:,0]*signs[:,1])/4
        risk=np.array(teacher['risk_grid21']);true_index=int(np.argmin(abs(rho-theta)))
        assert abs(rho[true_index]-theta)<1e-12
        for ni,n in enumerate(ns):
            w=J[ni]['main']/np.sqrt(n*ref['var_logsigmoid'])
            raw=np.einsum('brs,s->br',modes['raw'][:,ni,:,:],weights)
            marg=np.einsum('brs,s->br',modes['marg'][:,ni,:,:],weights)
            by={'raw':raw,'marg':marg,'quadratic':raw+w*gamma*rho**2,'informed':raw+2*A*w*H}
            for method,x in by.items():
                mean=x.mean(axis=0);se=x.std(axis=0,ddof=1)/np.sqrt(len(x))
                idx=int(np.argmin(mean));each=np.argmin(x,axis=1)
                selected=dict(theta=theta,n=int(n),K=int(n+2),method=method,rho=float(rho[idx]),
                    reference_joint_KL=float(risk[idx]),mean_score_contrast=float(mean[idx]),
                    mean_score_vs_truth=float(mean[idx]-mean[true_index]),
                    stream_selected_rhos=rho[each].tolist(),mean_stream_selected_reference_KL=float(risk[each].mean()),
                    stream_selection_unit='one prior-stream population score average, not a data-training repetition',w_n=float(w))
                summaries.append(selected)
                curves.append(dict(theta=theta,n=int(n),K=int(n+2),method=method,rho=rho.tolist(),
                    mean=mean.tolist(),SE=se.tolist(),normalized_mean=(mean/w).tolist(),
                    lower=(mean-tcrit*se).tolist(),upper=(mean+tcrit*se).tolist(),
                    reference_joint_KL=risk.tolist(),w_n=float(w)))
                if method in ['raw','marg']:
                    adjacent=np.diff(x,axis=1)
                    adjse=adjacent.std(axis=0,ddof=1)/np.sqrt(len(x))
                    for ri,v in enumerate(adjse):
                        gates.append(dict(theta=theta,n=int(n),method=method,left=float(rho[ri]),right=float(rho[ri+1]),SE=float(v),pass_=bool(v<=5e-4)))
            # Predefined fixed comparisons: raw endpoint-minus-truth and corrections at those exact candidates.
            for method,x in by.items():
                delta=x[:,-1]-x[:,true_index]
                stat_reference.append(dict(theta=theta,n=int(n),method=method,comparison='R_minus_truth',
                    mean=float(delta.mean()),SE=float(delta.std(ddof=1)/4),
                    lower=float(delta.mean()-tcrit*delta.std(ddof=1)/4),
                    upper=float(delta.mean()+tcrit*delta.std(ddof=1)/4),
                    normalized_mean=float(delta.mean()/w)))
    save(OUT/'SUMMARY.json',dict(source_sha256=sha(__file__),production_sha256=sha(OUT/'RESULTS.json'),
        reference_sha256=sha(REF),J_reference_sha256=sha(OUT/'J_REFERENCE.json'),protocol_sha256=sha(OUT/'SUMMARY_PROTOCOL.json'),
        selections=summaries,curves=curves,fixed_comparisons=stat_reference,
        qualification=dict(checks=len(gates),passed=sum(g['pass_'] for g in gates),failed=[g for g in gates if not g['pass_']],all=gates),
        independent_streams=16,confidence_scope=protocol['pointwise_intervals'],new_observed_data=0,new_data_fits=0))
    print(json.dumps({'checks':len(gates),'passed':sum(g['pass_'] for g in gates),'selections':len(summaries),
        'selected':[{k:r[k] for k in ['theta','n','method','rho','reference_joint_KL']} for r in summaries]}))


if __name__=='__main__':main()

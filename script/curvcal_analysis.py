"""Prespecified cross-case primary family and curvature diagnostics; no RNG."""
from pathlib import Path
import os,json,hashlib,math
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.stats import t
from scipy.interpolate import PchipInterpolator
import fincal_train as old

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/calibration_precision'
DOC=ROOT/'docs/08_research_研究工作/校准精度与跨尺度学习_20261002'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def stats(v,crit=None):
    a=np.asarray(v,float);s=dict(mean=float(a.mean()),SE=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,count=len(a))
    if crit is not None:s.update(lower=s['mean']-crit*s['SE'],upper=s['mean']+crit*s['SE'])
    return s
def run():
    tr=read(OUT/'TRAIN/RESULTS.json');full=read(OUT/'CAL/FULL/MODEL.json');loc=read(OUT/'CAL/LOCAL/MODEL.json')
    rows=[v for flow in tr['records'] for case in flow['cases'] for v in case['outputs']]
    lookup={(v['stream'],v['case'],v['theta'],v['B'],v['method']):v for v in rows}
    cases=[f"n{c['n']}_K{c['K']}" for c in tr['config']['cases']];ths=tr['config']['teachers'];Bs=tr['config']['bank_prefixes'];methods=tr['methods']
    def arr(c,th,B,m,field):return np.array([lookup[s,c,th,B,m][field] for s in range(16)])
    def mse(c,th,B,m):return (arr(c,th,B,m,'rho')-arr(c,th,B,'exact','rho'))**2
    tables=[]
    for c in cases:
        for th in ths:
            for B in Bs:
                for m in methods:
                    tables.append(dict(case=c,theta=th,B=B,method=m,rho=stats(arr(c,th,B,m,'rho')),reference_joint_KL=stats(arr(c,th,B,m,'reference_joint_KL')),squared_discrepancy_to_data_MLE=stats(mse(c,th,B,m)),endpoint_count=int(np.count_nonzero(abs(arr(c,th,B,m,'rho'))==.75))))
    critical=float(t.ppf(.995,15));gates=[]
    for c in cases:
        ds=np.mean([arr(c,th,512,'localquad','reference_joint_KL')-arr(c,th,512,'fullfinite','reference_joint_KL') for th in ths],axis=0)
        ss=stats(ds,critical);gates.append(dict(name=c+'_local_minus_full_KL_B512',case=c,stream_differences=ds.tolist(),**ss,direction='upper_below',threshold=.0001,passed=bool(ss['upper']<.0001)))
    ds=np.mean([mse(c,th,128,'fullfinite')-mse(c,th,512,'fullfinite') for c in cases for th in ths],axis=0)
    ss=stats(ds,critical);gates.append(dict(name='all_cases_full_B128_minus_B512_MSE',stream_differences=ds.tolist(),**ss,direction='lower_above',threshold=0.,passed=bool(ss['lower']>0)))
    reference=full['reference'];rp=PchipInterpolator(reference['rho801'],reference['rp801'],extrapolate=False)
    models={}
    for case in full['cases']:
        key=f"n{case['n']}_K{case['K']}";cfg=dict(tr['config'],n=case['n'],K=case['K'])
        _,rfun,_,g=old.load_reference(OUT/'CAL/FULL/MODEL.json',cfg)
        models[key]=(PchipInterpolator(case['rho101'],case['b_slope101'],extrapolate=False),g.derivative())
    gradients=[];diagnostics=[]
    for flow in tr['records']:
        s=flow['stream']
        for case in flow['cases']:
            c=case['case'];bf,gp=models[c]
            with np.load(ROOT/case['file']) as z:
                for ti,th in enumerate(ths):
                    q=float(z['gradient_rho'][ti]);boundary=bool(z['gradient_MLE_boundary'][ti]);kappa=float(bf(q)*rp(q));valid=bool(np.isfinite(kappa) and kappa>0)
                    for B in Bs:
                        for m in methods:
                            if m=='exact':continue
                            v=z['marg_path_gradient' if m=='marg' else 'raw_path_gradient'][:B,ti]
                            penalty=0. if m in ('raw','marg') else (float(gp(q)) if m=='fullfinite' else 2*tr['coefficients'][c][m]*q)
                            mu=float(v.mean()+penalty);var=float(v.var(ddof=1));ub=mu*mu-var/B
                            base=dict(stream=s,case=c,theta=th,B=B,method=m,q=q,boundary=boundary,gradient_mean=mu,sample_variance=var,variance_over_B=var/B,squared_mean=mu*mu,unbiased_squared_mean=ub)
                            gradients.append(base)
                            if m in ('fullfinite','localquad'):
                                actual=lookup[s,c,th,B,m]['rho']-q;use=valid and not boundary
                                pred=-mu/kappa if use else None
                                diagnostics.append(dict(**base,kappa_reference=kappa if np.isfinite(kappa) else None,kappa_positive_finite=valid,interior_diagnostic_usable=use,actual_shift=actual,actual_squared_shift=actual*actual,linearized_shift=pred,linearization_residual=actual-pred if use else None,normalized_noise=var/(B*kappa*kappa) if use else None,normalized_squared_mean=mu*mu/(kappa*kappa) if use else None,unbiased_normalized_mean_square=ub/(kappa*kappa) if use else None,local_minus_full_penalty_gradient=2*tr['coefficients'][c]['localquad']*q-float(gp(q))))
    summaries=[]
    for c in cases:
        for th in ths:
            for B in Bs:
                for m in ('fullfinite','localquad'):
                    rr=[v for v in diagnostics if v['case']==c and v['theta']==th and v['B']==B and v['method']==m and v['interior_diagnostic_usable']]
                    if not rr:continue
                    keys=['kappa_reference','actual_squared_shift','normalized_noise','normalized_squared_mean','unbiased_normalized_mean_square','local_minus_full_penalty_gradient']
                    entry=dict(case=c,theta=th,B=B,method=m,interior_count=len(rr),**{key:stats([v[key] for v in rr]) for key in keys})
                    entry['mean_squared_linearization_residual']=float(np.mean([v['linearization_residual']**2 for v in rr]))
                    entry['actual_MSE_over_normalized_noise']=entry['actual_squared_shift']['mean']/entry['normalized_noise']['mean']
                    summaries.append(entry)
    bounds=[v['continuous_objective_regret_bound_real_arithmetic'] for v in rows if v['method']!='exact']
    result=dict(source_sha256=sha(__file__),training_sha256=sha(OUT/'TRAIN/RESULTS.json'),full_model_sha256=sha(OUT/'CAL/FULL/MODEL.json'),local_model_sha256=sha(OUT/'CAL/LOCAL/MODEL.json'),plan_sha256=sha(DOC/'PLAN.md'),family_size=5,alpha=.05,df=15,critical_t=critical,primary_gates=gates,tables=tables,gradient_rows=gradients,curvature_diagnostic_rows=diagnostics,curvature_summaries=summaries,known_error_qualification=dict(count=len(bounds),maximum=max(bounds),threshold=3e-5,pass_count=sum(b<=3e-5 for b in bounds)),output_count=len(rows),boundary_unique_MLE=sum(sum(f['gradient_MLE_boundary']) for f in tr['records']),negative_unbiased_squared_gradient_count=sum(v['unbiased_squared_mean']<0 for v in gradients),new_random_samples=0,scope='Fresh data and prior, conditional on frozen shared LOCAL/FULL calibration. Curvature is a numerical CAL proxy; same-bank local linearization is diagnostic, not independent risk prediction or causal attribution.')
    write(OUT/'ANALYSIS.json',result)
    lines=['# 全部预定读数','', '4case×2B×3teacher×6方法共144格；16基础流独立，case/teacher/B共享输入。共同95%开发近似区间条件于冻结校准，非严格覆盖保证。','', '|主门|mean|SE|共同95%近似区间|阈值/方向|通过|','|---|---:|---:|---|---|---|']
    for g in gates:lines.append(f"|{g['name']}|{g['mean']:.10g}|{g['SE']:.8g}|[{g['lower']:.9g},{g['upper']:.9g}]|{g['direction']} {g['threshold']}|{g['passed']}|")
    lines+=['','|case|theta|B|方法|mean rho|mean KL|KL SE|MSE to data MLE|端点/16|','|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for v in tables:lines.append(f"|{v['case']}|{v['theta']}|{v['B']}|{v['method']}|{v['rho']['mean']:.8g}|{v['reference_joint_KL']['mean']:.10g}|{v['reference_joint_KL']['SE']:.7g}|{v['squared_discrepancy_to_data_MLE']['mean']:.10g}|{v['endpoint_count']}|")
    lines+=['',f"1920网格/插值已知实算术项最大{max(bounds):.12g}；{sum(b<=3e-5 for b in bounds)}/1920通过3e−5资格。不含ζ/MC/校准误差。",'',f"48唯一MLE边界{result['boundary_unique_MLE']}；固定点梯度负无偏均值平方{result['negative_unbiased_squared_gradient_count']}条原样保留。",'', '## 内点曲率诊断','', 'κ来自新FULL估计的b与已知r′表，含未严格量化的校准误差。线性化使用同bank，非独立风险预测；原逐点数值与所有负估计在ANALYSIS.json。','', '|case|theta|B|方法|内点数|mean κ|actual MSE|mean s²/(Bκ²)|ratio|线性化残差MSE|','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for v in summaries:lines.append(f"|{v['case']}|{v['theta']}|{v['B']}|{v['method']}|{v['interior_count']}|{v['kappa_reference']['mean']:.7g}|{v['actual_squared_shift']['mean']:.8g}|{v['normalized_noise']['mean']:.8g}|{v['actual_MSE_over_normalized_noise']:.7g}|{v['mean_squared_linearization_residual']:.8g}|")
    (DOC/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'gates':gates,'known_error':result['known_error_qualification'],'boundary':result['boundary_unique_MLE']}))
if __name__=='__main__':run()

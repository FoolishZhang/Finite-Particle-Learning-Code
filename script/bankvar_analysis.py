"""Predefined paired analyses of the bank-precision experiment; no simulation."""
from pathlib import Path
import hashlib
import json
import math
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.stats import t
import fincal_train as old

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/bank_precision'
DOC=ROOT/'docs/08_research_研究工作/粒子精度与局部校准_20261002'

def read(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def stats(x,critical=None):
    x=np.asarray(x,dtype=float)
    ans=dict(mean=float(x.mean()),SE=float(x.std(ddof=1)/math.sqrt(len(x))),n=len(x))
    if critical is not None:
        ans.update(lower=ans['mean']-critical*ans['SE'],upper=ans['mean']+critical*ans['SE'])
    return ans

def run():
    inp=OUT/'TRAIN/RESULTS.json'; data=read(inp)
    rows=[r for rec in data['records'] for r in rec['outputs']]
    lookup={(r['stream'],r['theta'],r['B'],r['method']):r for r in rows}
    ns=range(16);teachers=data['config']['teachers'];bs=data['config']['bank_prefixes']
    def values(th,B,m,field):return np.array([lookup[s,th,B,m][field] for s in ns])
    def mse(th,B,m):return (values(th,B,m,'rho')-values(th,B,'exact','rho'))**2
    tables=[]
    for th in teachers:
        for B in bs:
            for m in data['methods']:
                rh=values(th,B,m,'rho')
                tables.append(dict(theta=th,B=B,method=m,rho=stats(rh),
                    reference_joint_KL=stats(values(th,B,m,'reference_joint_KL')),
                    squared_discrepancy_to_data_MLE=stats(mse(th,B,m)),
                    endpoint_count=int(np.count_nonzero(np.abs(rh)==.75))))
    critical=float(t.ppf(.9875,15))
    differences=[np.mean([mse(th,128,'fullfinite')-mse(th,512,'fullfinite') for th in teachers],axis=0),
        np.mean([values(th,512,'localquad','reference_joint_KL')-values(th,512,'fullfinite','reference_joint_KL') for th in teachers],axis=0)]
    gates=[]
    for name,ds,threshold,direction in zip(('fullfinite_B128_minus_B512_MSE','local_minus_full_B512_KL'),differences,(0.,.0001),('lower_above','upper_below')):
        ss=stats(ds,critical)
        passed=ss['lower']>threshold if direction=='lower_above' else ss['upper']<threshold
        gates.append(dict(name=name,stream_differences=ds.tolist(),**ss,threshold=threshold,direction=direction,passed=bool(passed)))
    ref,rfun,H,g=old.load_reference(ROOT/'artifacts/finite_calibration/CAL/MODEL.json',data['config'])
    gradient_rows=[]
    for record in data['records']:
        with np.load(ROOT/record['file']) as z:
            for ti,th in enumerate(teachers):
                q=float(z['gradient_rho'][ti]);boundary=bool(z['gradient_MLE_boundary'][ti])
                for B in bs:
                    for m in data['methods']:
                        if m=='exact':continue
                        sample=z['marg_path_gradient' if m=='marg' else 'raw_path_gradient'][:B,ti]
                        penalty=0. if m in ('raw','marg') else (float(g.derivative()(q)) if m=='fullfinite' else 2*data['coefficients'][m]*q)
                        mu=float(sample.mean()+penalty);var=float(sample.var(ddof=1))
                        gradient_rows.append(dict(stream=record['stream'],theta=th,B=B,method=m,rho=q,boundary=boundary,
                            gradient_mean=mu,sample_variance=var,variance_over_B=var/B,
                            squared_mean=mu*mu,unbiased_squared_population_mean=mu*mu-var/B))
    gradient_summary=[]
    for th in teachers:
        for B in bs:
            for m in data['methods']:
                if m=='exact':continue
                for subset in ('all','interior'):
                    rr=[r for r in gradient_rows if r['theta']==th and r['B']==B and r['method']==m and (subset=='all' or not r['boundary'])]
                    if not rr:continue
                    gradient_summary.append(dict(theta=th,B=B,method=m,subset=subset,count=len(rr),**{key:float(np.mean([r[key] for r in rr])) for key in ('gradient_mean','sample_variance','variance_over_B','squared_mean','unbiased_squared_population_mean')}))
    bounds=[r['continuous_objective_regret_bound_real_arithmetic'] for r in rows if r['method']!='exact']
    result=dict(source_sha256=sha(__file__),input_sha256=sha(inp),plan_sha256=sha(DOC/'PLAN.md'),
        family_size=2,alpha=.05,df=15,critical_t=critical,primary_gates=gates,tables=tables,
        gradient_rows=gradient_rows,gradient_summary=gradient_summary,
        gradient_scope='Fixed old data and frozen shared calibration; direct path gradients at data MLE. No clipping negative unbiased squared-mean estimates, no optimized-risk causal attribution.',
        mesh_interpolation_qualification=dict(count=len(bounds),threshold=3e-5,maximum=max(bounds),pass_count=sum(v<=3e-5 for v in bounds)),
        output_count=len(rows),unique_data_teacher_MLE=48,
        boundary_MLE_count=sum(sum(r['gradient_MLE_boundary']) for r in data['records']),
        cheb_stream_count=sum(r['evaluation_mode']=='chebyshev193' for r in data['records']),
        calibration_uncertainty_in_CI=False,new_observations=0)
    write(OUT/'ANALYSIS.json',result)
    print(json.dumps(dict(gates=gates,mesh=result['mesh_interpolation_qualification'],boundaries=result['boundary_MLE_count'])))
    lines=['# 全部预定读数','', '同16个旧完整数据流、N4096，使用全新嵌套prior；区间条件于数据与冻结校准。所有KL为完整joint参考KL，非严格积分证书。','',
        '|新主门|mean|SE|共同95%近似区间|阈值/方向|通过|','|---|---:|---:|---|---|---|']
    for r in gates:lines.append(f"|{r['name']}|{r['mean']:.10g}|{r['SE']:.8g}|[{r['lower']:.9g}, {r['upper']:.9g}]|{r['direction']} {r['threshold']}|{r['passed']}|")
    lines+=['','每项先对三个teacher在流内平均，再以16流作paired t；临界t_.9875,15。不能由合并teacher门声称逐teacher非劣。旧原四门2/4保持。','',
        '|theta|B|方法|mean rho|mean KL|KL SE|MSE to data MLE|端点数/16|','|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in tables:lines.append(f"|{r['theta']}|{r['B']}|{r['method']}|{r['rho']['mean']:.8g}|{r['reference_joint_KL']['mean']:.10g}|{r['reference_joint_KL']['SE']:.6g}|{r['squared_discrepancy_to_data_MLE']['mean']:.9g}|{r['endpoint_count']}|")
    lines+=['',f"864条真实算术网格+插值已知误差项：最大 {max(bounds):.12g}，{sum(v<=3e-5 for v in bounds)}/864通过预定3e−5门。未包括浮点ζ、MC、校准误差，不是完整严格连续globopt证书。",'',
        f"同数据48个数值MLE中边界 {result['boundary_MLE_count']}；16流中Cheb193 {result['cheb_stream_count']}，其余保留原输入direct801。",'',
        '## 固定点梯度诊断','', '下表仅内点。逐流原值（含负的无偏均值平方估计）、边界与all统计均保存在ANALYSIS.json。加性惩罚同点方差完全相同；不能将表中比例解释为最终KL因果百分比。','',
        '|theta|B|方法|内点流数|平均梯度|平均s²/B|平均梯度²|无偏均值平方|','|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in gradient_summary:
        if r['subset']=='interior':lines.append(f"|{r['theta']}|{r['B']}|{r['method']}|{r['count']}|{r['gradient_mean']:.7g}|{r['variance_over_B']:.7g}|{r['squared_mean']:.7g}|{r['unbiased_squared_population_mean']:.7g}|")
    (DOC/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':run()

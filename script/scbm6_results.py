"""Descriptive aggregation of completed, frozen six-correlation experiments."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/scbm_multicorrelation'
DOC=ROOT/'docs/08_research_研究工作/六相关自由学习与方向机制_20261003'
METHODS=['FIXED','FREE','FACT']
PAIRS=['01','02','03','12','13','23']
TRUTH=np.array([.25,-.15,.10,.10,-.15,.25])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()

def run():
    assert not (OUT/'VERSION_LOCK.json').exists()
    paths=[OUT/'TRAIN/RESULTS.json',OUT/'REFERENCE/RESULTS.json',OUT/'DIRECTION/RESULTS.json']
    train,ref,direction=map(read,paths)
    assert len(train['records'])==16 and len(ref['endpoints'])==96
    assert ref['qualification_passed']
    risks={(r['stream'],r['method'],r['kind']):r['reference_joint_KL'] for r in ref['endpoints']}
    initial=ref['identity_reference']['reference_joint_KL']
    rows=[];parameter={}
    for record in train['records']:
        for method in METHODS:
            for kind in ['TAIL','LAST']:
                o=record['outputs'][method][kind]
                rows.append(dict(stream=record['stream'],method=method,kind=kind,
                    risk=risks[record['stream'],method,kind],rho=o['rho'],
                    eigenvalues=np.linalg.eigvalsh(o['covariance']).tolist(),
                    scale_mean=float(np.mean(o['scales']))))
    for method in METHODS:
        selected=[r for r in rows if r['method']==method and r['kind']=='TAIL']
        rho=np.array([r['rho'] for r in selected])
        coefficient=rho@TRUTH/(TRUTH@TRUTH)
        residual=rho-coefficient[:,None]*TRUTH
        spectrum=np.array([r['eigenvalues'] for r in selected])
        v=np.array([r['risk'] for r in selected])
        parameter[method]=dict(mean_rho=rho.mean(0).tolist(),SD_rho=rho.std(0,ddof=1).tolist(),
            sign_mismatch_counts=np.sum(rho*TRUTH<0,axis=0).tolist(),
            mean_ray_coefficient=float(coefficient.mean()),
            per_stream_ray_coefficients=coefficient.tolist(),
            mean_orthogonal_rho_norm=float(np.linalg.norm(residual,axis=1).mean()),
            per_stream_orthogonal_rho_norm=np.linalg.norm(residual,axis=1).tolist(),
            mean_eigenvalues=spectrum.mean(0).tolist(),
            mean_scale=float(np.mean([r['scale_mean'] for r in selected])),
            per_stream_repair_count=[r['diagnostics'][method]['feasibility_repair_count'] for r in train['records']],
            risk_below_identity_count=int(np.sum(v<initial)),
            risk_equal_identity_count=int(np.sum(v==initial)),
            risk_above_identity_count=int(np.sum(v>initial)))
    free_better=sum(risks[s,'FREE','TAIL']<risks[s,'FIXED','TAIL'] for s in range(16))
    fact_better=sum(risks[s,'FACT','TAIL']<risks[s,'FIXED','TAIL'] for s in range(16))
    pairs=[]
    for j,pair in enumerate(PAIRS):
        item=dict(pair=pair,truth_rho=float(TRUTH[j]),profiles=[])
        for k,profile in enumerate(['scale1','scale02','FACT']):
            item['profiles'].append(dict(profile=profile,
                **{metric:direction['descriptive'][metric]['mean'][1][k][j]
                   for metric in ['gradient','mean_mark_term','coupling_term','markmc']}))
        pairs.append(item)
    result=dict(source_sha256=sha(__file__),inputs={rel(p):sha(p) for p in paths},
        identity_reference_KL=initial,training_descriptive=ref['descriptive'],
        training_primary=ref['primary_family'],parameter_descriptive=parameter,
        complete_output_rows=rows,free_better_than_fixed_count=free_better,
        fact_better_than_fixed_count=fact_better,direction_pair_decomposition=pairs,
        direction_primary=direction['primary_family'],direction_descriptive=direction['descriptive'],
        counts=dict(training_effects=ref['primary_effects_passed'],
            training_precision=ref['primary_precision_passed'],training_total=3,
            training_combined=sum(r['effect_passed'] and r['precision_passed'] for r in ref['primary_family']),
            direction_effects=direction['primary_effects_passed'],
            direction_precision=direction['primary_precision_passed'],direction_total=18,
            direction_combined=sum(r['effect_passed'] and r['precision_passed'] for r in direction['primary_family'])),
        additional_diagnostic_status='Prespecified descriptive output types; this post-run script adds no tests, confidence intervals, selections, sampling, fitting or model integration.',
        interpretation='Parameter averages are descriptive across independent streams, not a deployed ensemble or KL evaluated at a mean matrix. Orthogonal/radial errors do not partition observation risk causally.')
    with (OUT/'RESULTS.json').open('x',encoding='utf8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    lines=['# 六相关实验的全部主要读数','',
        '本表由冻结结果确定性汇总；TRAIN与DIRECTION为不同统计家族。区间为预定Bonferroni开发t近似。效应、SE目标与数值资格分开；没有新增检验或补样本。','',
        '## 1. 完整四变量积分模型风险','',
        f'单位矩阵初始模型参考KL={initial:.12f}。风险先逐模型求值，再按16条流平均。','',
        '|输出|方法|mean joint KL|SE（描述）|','|---|---|---:|---:|']
    for r in ref['descriptive']:
        lines.append(f"|{r['kind']}|{r['method']}|{r['mean_joint_KL']:.12f}|{r['SE_joint_KL']:.12f}|")
    lines+=['','|TAIL预定比较|差均值|SE|共同区间|效应|精度|','|---|---:|---:|---|---|---|']
    for r in ref['primary_family']:
        lines.append(f"|{r['name']}|{r['mean']:.12f}|{r['SE']:.12f}|[{r['lower']:.12f}, {r['upper']:.12f}]|{'通过' if r['effect_passed'] else '未确认'}|{'通过' if r['precision_passed'] else '未达'}|")
    lines+=['',f'效应2/3、精度0/3，合并要求0/3。FREE和FACT各有{free_better}/16与{fact_better}/16条流风险低于FIXED（描述性计数，不另加检验）。',
        'FREE−FACT的原共同区间全在0以上，按该开发近似支持正差；但其上端未低于非劣容限.0002且SE未达.00005。不能追认旧实验，也不能据此说差额已确定超过.0002。','',
        '## 2. 六相关估计及可行性（描述）','',
        '|方法|mean rho (01,02,03,12,13,23)|mean tau|平均径向正交残差范数|修复总次数|低于/高于初始风险的流数|',
        '|---|---|---:|---:|---:|---|']
    for method,p in parameter.items():
        lines.append(f"|{method}|{', '.join(f'{v:.6f}' for v in p['mean_rho'])}|{p['mean_scale']:.9f}|{p['mean_orthogonal_rho_norm']:.9f}|{sum(p['per_stream_repair_count'])}|{p['risk_below_identity_count']}/{p['risk_above_identity_count']}|")
    lines+=['','真实rho=(.25,−.15,.10,.10,−.15,.25)。均值接近真值不表示各流风险低；上表没有对跨流平均矩阵另作风险积分。FACT的tau=0为解析因子分解约定。','',
        '## 3. 真矩阵处的18项方向比较','',
        '32组×512bank；区间条件于旧数值参考PMF。符号按对应真相关对齐，负梯度对应局部增加其幅度。绝对方向、相对FACT和尺度收缩比较使用同一个q=18家族；全部SE门均为5e−5。','',
        '|比较|pair|均值|SE|共同区间|效应|精度|','|---|---|---:|---:|---|---|---|']
    for r in direction['primary_family']:
        pair=''.join(map(str,r['pair']))
        lines.append(f"|{r['name']}|{pair}|{r['mean']:.10f}|{r['SE']:.10f}|[{r['lower']:.10f}, {r['upper']:.10f}]|{'通过' if r['effect_passed'] else '未确认'}|{'通过' if r['precision_passed'] else '未达'}|")
    lines+=['','效应18/18、精度0/18，合并要求0/18；原失败保留。统一.2尺度是预定探针，不是FREE实际学得向量。','',
        '## 4. 各方向分解（描述均值）','',
        '|pair|profile|Price gradient|−m_ref·overlap|参考中心化耦合|',
        '|---|---|---:|---:|---:|']
    for r in pairs:
        for p in r['profiles']:
            lines.append(f"|{r['pair']}|{p['profile']}|{p['gradient']:.10f}|{p['mean_mark_term']:.10f}|{p['coupling_term']:.10f}|")
    for metric in ['overlap','mean_ess','mean_inverse_ess']:
        lines+=['',metric+'（truth，依次scale1/.2/FACT）='+', '.join(f'{v:.10f}' for v in direction['descriptive'][metric]['mean'][1])+'。']
    lines+=['','重叠为完整似然权重的E[1−Σπ²]=1−E[1/ESS]，不是noise-only ESS，也不是1−1/E[ESS]。分解为精确代数身份；分项受参考中心选择影响，不是因果百分比。','',
        '## 5. 输入和机器可读来源','',
        '[TRAIN](../../../artifacts/scbm_multicorrelation/TRAIN/RESULTS.json) · [完整C参考](../../../artifacts/scbm_multicorrelation/REFERENCE/RESULTS.json) · [DIRECTION](../../../artifacts/scbm_multicorrelation/DIRECTION/RESULTS.json) · [完整汇总](../../../artifacts/scbm_multicorrelation/RESULTS.json)。']
    with (DOC/'RESULTS.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'completed':True,'counts':result['counts'],'new_science':0}))

if __name__=='__main__':run()

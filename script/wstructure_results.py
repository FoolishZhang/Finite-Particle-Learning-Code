"""Deterministic readable synthesis of completed, independently audited results."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nuisance_structure'
DOC=ROOT/'docs/08_research_研究工作/无关权重结构与方差匹配_20261002'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(x):return f'{x:.9g}'
def main():
    assert not (OUT/'VERSION_LOCK.json').exists()
    ref,mc,tr,audit=[read(OUT/p) for p in ['REFERENCE/MODEL.json','MC/RESULTS.json','TRAIN/RESULTS.json','AUDIT/REPORT.json']]
    assert ref['qualification_passed'] and audit['passed'] and read(OUT/'TRAIN/STATE.json')['status']=='complete'
    geom=[]
    for i,c in enumerate(mc['cases']):
        ess=np.array([r['means']['nuisance_ESS'][i] for r in mc['records']]);mx=np.array([r['means']['nuisance_max_weight'][i] for r in mc['records']])
        geom.append(dict(**c,mean_ESS=float(ess.mean()),mean_ESS_fraction=float(ess.mean()/c['K']),mean_maximum_weight=float(mx.mean()),descriptive=True))
    train_index={(r['theta'],r['method']):r for r in tr['tables']}
    ratios={str(t):train_index[t,'diffuse']['mean_KL']/train_index[t,'concentrated']['mean_KL'] for t in [0.,.6]}
    summary=dict(inputs={p:sha(ROOT/p) for p in [str((OUT/x).relative_to(ROOT)).replace('\\','/') for x in ['REFERENCE/MODEL.json','MC/RESULTS.json','TRAIN/RESULTS.json','AUDIT/REPORT.json']]},source_sha256=sha(__file__),
        variance_profiles=[{k:p[k] for k in ['n','family','min_scale','max_scale','total_variance','relative_variance_residual','maximum_variance_share']} for p in ref['profiles']],
        weight_geometry=geom,MC_primary=mc['primary_family'],TRAIN_primary=tr['comparisons'],TRAIN_tables=tr['tables'],
        descriptive_KL_ratios=ratios,statistical_families_separate=True,
        new_independent_prior_banks=mc['accounting']['independent_banks']+tr['ledger']['prior_banks'],
        new_prior_vectors=mc['accounting']['prior_vectors']+tr['ledger']['prior_vectors'],
        new_prior_Gaussian_scalars=mc['accounting']['Gaussian_scalars']+tr['ledger']['prior_Gaussian'],
        new_base_observations=tr['ledger']['new_base_observations'],shared_input_theta_vectors=tr['ledger']['shared_input_teacher_vectors'],
        data_Gaussian=tr['ledger']['data_Gaussian'],data_Uniform=tr['ledger']['data_Uniform'],MC_Uniform=mc['accounting']['uniform_scalars'],
        reference_work=ref['accounting'],additional_audit_model_component_integrals=audit['new_reference_audit_component_integrals'],
        all_registered_failures_retained=True,new_analysis_RNG=0,new_analysis_model_integrals=0)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    s='''# 全部主要读数与数值范围

全部科学执行和独立审计完成。下列数值是冻结输出的确定性汇总，未增加随机数、模型积分或检验门。

## 1. 三种尺度的方差匹配

|n|family|最小尺度|最大尺度|总log方差|最大单项方差份额|
|---:|---|---:|---:|---:|---:|
'''
    for p in ref['profiles']:s+=f"|{p['n']}|{p['family']}|{f(p['min_scale'])}|{f(p['max_scale'])}|{f(p['total_variance'])}|{f(p['maximum_variance_share'])}|\n"
    s+='''
三族匹配的是无关单粒子log权重的总体方差；理论根精确匹配，实际双精度根通过数值资格。七尺度不同表达及六profile总方差核对通过，独立50位审计的最大相对匹配残差约1.30e−16。不是严格全积分余项证书。

## 2. MC：原12项主家族完整报告

固定n254。gap=Fθ(.75)−Fθ(θ)；负值偏好端点，正值偏好真实参数这个候选。表中是共同95%的经验Bernstein抽样区间，基于65536个独立bank。θ.6使用冻结数值r权重；区间不包括浮点评价、PRNG或Gaussian参考求积误差。正负gap不等于有限n连续全局极小点证书。

|K|family|θ|gap均值|区间下端|区间上端|预定方向|原门|
|---:|---|---:|---:|---:|---:|---|---|
'''
    for p in mc['primary_family']:s+=f"|{p['K']}|{p['family']}|{p['theta']}|{f(p['mean'])}|{f(p['lower'])}|{f(p['upper'])}|{p['expected_sign']}|{'pass' if p['passed'] else 'fail'}|\n"
    s+=f"\n原主门{sum(p['passed'] for p in mc['primary_family'])}/12。固定点Price精度{mc['precision']['price']['pass_count']}/{mc['precision']['price']['assessed_count']}、非锚点分数差精度{mc['precision']['score_difference']['pass_count']}/{mc['precision']['score_difference']['assessed_count']}；它们是16组均值的SE资格，不是附加显著性门。MC总耗时{f(mc['elapsed_seconds'])}秒。\n"
    s+='''
![同方差下的无关权重与目标方向。只显示原主家族中n254/K256的6项区间，沿用12项共同水平。](../../../artifacts/nuisance_structure/figures/variance_weights_gap.png)

## 3. 全部noise-only几何读数

下表仅描述均值，独立单位来自与上表同一批prior，不能当作新增确认；ESS指无关因子权重，不是完整active+nuisance权重。

|n|K|family|mean ESS|mean ESS/K|mean最大归一权重|
|---:|---:|---|---:|---:|---:|
'''
    for p in geom:s+=f"|{p['n']}|{p['K']}|{p['family']}|{f(p['mean_ESS'])}|{f(p['mean_ESS_fraction'])}|{f(p['mean_maximum_weight'])}|\n"
    s+='''
## 4. 新完整观测训练：所有方法与边界输出

固定n254/K256/B128/N4096，16个新基础流，两个θ共享输入但不是32个独立单位。三族使用相同数据、基础prior和801候选；没有估计校准池。marg与MLE在三族间相同，故各只记一次。数值MLE的θ.6分支有1个边界数据实例，该实例和全部边界拟合均保留。

|θ|方法|mean rho|mean joint KL|mean与同数据MLE参数平方差|端点数/16|mean一致事件绝对误差|
|---:|---|---:|---:|---:|---:|---:|
'''
    for p in tr['tables']:s+=f"|{p['theta']}|{p['method']}|{f(p['mean_rho'])}|{f(p['mean_KL'])}|{f(p['mean_MLE_MSE'])}|{p['endpoint_count']}|{f(p['mean_absolute_agreement_error'])}|\n"
    s+='''
零相关下对称端点的均值可能抵消，不能用mean rho代替风险或端点频率。concentrated接近MLE是这里的描述性读数，没有另行设定对MLE的非劣门。

|原主比较：diffuse−concentrated|均值|SE|区间下端|区间上端|原门|
|---|---:|---:|---:|---:|---|
'''
    for p in tr['comparisons']:
        label=p['name']+(f"，θ={p['theta']}" if 'theta'in p else '')
        s+=f"|{label}|{f(p['mean'])}|{f(p['SE'])}|{f(p['low'])}|{f(p['high'])}|{'pass' if p['passed'] else 'fail'}|\n"
    s+=f"\n训练主门{tr['primary_passed']}/3；两侧Bonferroni t开发近似，临界值{f(tr['comparisons'][0]['critical'])}，不作精确有限样本覆盖声明。这个3项家族与MC的12项家族分开，不能合成全报告共同95%。\n"
    s+=f"\n128网格输出的实算术网格/插值分项{tr['grid_precision_passed']}/128通过，最大{f(tr['known_regret_max'])}，阈值3e−5。全部16流采用原定Cheb193路径，无回退。浮点ζ、MC、参考误差另列；不是总体连续globopt认证。\n"
    s+='''
![同完整数据与prior的学习结果。所有16流与边界点保留，风险为描述性均值，统计区间见上表。](../../../artifacts/nuisance_structure/figures/training_structure.png)

## 5. 成本、审计与公开数据范围

'''
    s+=f"新增prior共{summary['new_independent_prior_banks']}个bank、{summary['new_prior_vectors']}条向量、{summary['new_prior_Gaussian_scalars']}个Gaussian标量。另有65536个新基础观测、131072个共享输入θ世界向量，数据Gaussian131072、Uniform16777216；MC另用65536个Uniform。三族/n/K/候选复用不重计；同bank预算不等于已测得同运行时间。\n\n"
    s+='''确定性参考包括30个主v求值（60个分量积分）、7个tight v求值（14个分量积分）、14个Legendre方差检查及3个尺度根。独立审计另加14个高精度分量积分；未新积分active r/H。

输入全部重放，但底层MC评分只独立重建每流前4bank（64个唯一bank）；训练raw只重建每流前2bank、全部4096观测及固定9点（32个唯一bank、三族96个family-bank）。保存的MC/训练统计、marg节点、DCT插值、160输出与原门全部另核。全输入重放不等于全部底层评分重算。审计不增加独立样本数。

全部固定点/模式/组均值与SE见[MC原记录](../../../artifacts/nuisance_structure/MC/RESULTS.json)，全部160学习输出见[TRAIN原记录](../../../artifacts/nuisance_structure/TRAIN/RESULTS.json)，[独立审计](../../../artifacts/nuisance_structure/AUDIT/REPORT.json)明确核验范围。大NPZ保留本地而不提交；公开JSON和哈希不是完整二进制备份。旧科学包及失败原样保留，不由本轮成功追认。
'''
    (DOC/'RESULTS.md').write_text(s,encoding='utf8',newline='\n')
    print(json.dumps(dict(MC_primary=sum(p['passed'] for p in mc['primary_family']),TRAIN_primary=tr['primary_passed'],KL_ratios=ratios,summary_sha256=sha(OUT/'SUMMARY.json'))))
if __name__=='__main__':main()

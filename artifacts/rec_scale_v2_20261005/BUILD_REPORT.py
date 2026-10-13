"""Render public tables from frozen results only; no new fitting or inference."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/rec_scale_v2_20261005'
DOC=ROOT/'docs/08_research_研究工作/REC_v2_20261005'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
r=read(OUT/'RESULTS.json');gh=read(OUT/'GH_AUDIT.json');audit=read(OUT/'ARITHMETIC_AUDIT.json')
arms=('NEAR_FREE','UNIT_FREE')
risks={(x['arm'],x['state']):x for x in r['risk']}
cells={(x['arm'],x['candidate'],x['mode'],x['target']):x for x in r['family']}
lines=['# REC-v2：尺度能力不足与有限目标冲突可以同时存在','',
'2026-10-05。新周期参考更新后的HANDOFF，独立完成解析推导、冻结和计算。**全部32个实际输出在固定有效尺度下，都有至少一对教师依赖超出其表达上界；保持当前观察边际的预定尺度放大候选平均降低风险，却提高总体有限粒子目标，消去nuisance后仍成立。** 旧RHO候选愿被目标接受，不代表释放尺度也会被接受。','',
'本结论条件于原16份数据的32个保存输出。SCALE2并非逐输出成功、没有达到完整恢复，也不识别训练历史。均值的有下降控制新候选数值上改善全部32个经验观察NLL，但没有通过预定面板风险幅度门。旧MU固定0.05失败继续保留。','',
'## 1. 能力界回答了什么','',
'对自旋S=2Y−1，固定有效尺度s、允许任意均值和C≽0.1I时，|Cov(S_i,S_j)|≤0.9s_i s_j/4。结合协方差的TV Lipschitz界、Pinsker和数据处理，可得KL(P||Q)≥max_ij[(|Cov_P|−上界)_+²/8]，不要求P/Q边际相同。推导见[THEORY](THEORY.md)，不将经典Price或Pinsker本身列为新贡献。','',
'|初值|教师依赖超出能力界的对数|至少一对超界的输出|每输出保守KL下界范围|',
'|---|---:|---:|---:|']
for row in r['capacity']:
    lo,hi=row['KL_lower_bound_range']
    lines.append(f'|{row["arm"]}|{row["excluded_pairs"]}/96|{row["models_with_any_gap"]}/16|[{lo:.10f}, {hi:.10f}]|')
lines+=['',
'共189/192个“输出×变量对”的能力上界低于教师依赖，32/32输出不能在当前尺度下精确表达教师联合律，改变均值也不能突破该上界。这里没有192个独立数据单位。剩余3对的界不够紧，并非证明其可实现。','',
f'所有KL下界都低于真实产品风险{r["true_product_risk"]:.10f}；因此**不能据此判定固定尺度模型无法超过产品基线**，也不能把下界视为全部风险的因果份额。六对下界取最大值而非相加，门槛条件于旧数值教师，没有连续教师严格积分余项。','',
'## 2. 预定尺度放大：风险改善但总体有限目标恶化','',
'SCALE2逐维取s′=min(2,2s)，保留当前C和tau，并解均值补偿保持基点观察边际。32/32候选在原均值盒内可行，没有更换倍率、选方向或利用教师/evaluation bank。四维PMF边际最大变化约1e−15，故本次风险变化来自依赖分布变化，而非顺带修复边际。','',
'|初值|原平均KL|SCALE2平均KL|平均ΔR|逐流ΔR范围|',
'|---|---:|---:|---:|---:|']
for arm in arms:
    x=risks[(arm,'SCALE2')];lo,hi=x['delta_range']
    lines.append(f'|{arm}|{risks[(arm,"BASE")]["mean_risk"]:.10f}|{x["mean_risk"]:.10f}|{x["mean_delta"]:+.10f}|[{lo:+.10f}, {hi:+.10f}]|')
lines+=['',
'两组平均风险下降都超过预定4e−6分辨阈，但两组扰动后平均风险仍高于真实产品。以下有限目标变化均为候选减基点，单位为10⁻⁶；区间是固定16项家族的Bonferroni MC近似区间，只覆盖条件评价prior误差。','',
'|初值|保留 F_D [CI]|消去 F_D [CI]|保留 F_P [CI]|消去 F_P [CI]|',
'|---|---:|---:|---:|---:|']
def cell(arm,state,mode,target):
    x=cells[(arm,state,mode,target)];lo,hi=x['interval']
    return f'{x["delta"]*1e6:+.3f} [{lo*1e6:+.3f}, {hi*1e6:+.3f}]'
for arm in arms:
    entries=[cell(arm,'SCALE2',mode,target) for target in ('F_D','F_P') for mode in ('retained','eliminated')]
    lines.append('|'+arm+'|'+'|'.join(entries)+'|')
lines+=['',
'两个F_P在两初值都上升，与平均风险下降相反。这是实际非零尺度输出上的有限候选冲突，不是由产品基点Hessian外推。消去nuisance仍保留active有限K偏差，不能把ZERO评分当作精确似然。NEAR的F_D在保留时上升、消去时下降；UNIT保留F_D未决、消去F_D下降。消去后的F_D和F_P排序相反，说明训练标签权重确实影响这条候选排序，但不是全部数据噪声或完整历史原因。','',
'每种初值均有14个风险差读数为负、2个为正（描述性计数，不是逐输出同时显著结论）。伤害个例全部保留：','',
'|初值/流|ΔR（正数为恶化）|',
'|---|---:|']
for arm in arms:
    for stream,delta in enumerate(risks[(arm,'SCALE2')]['paired_delta']):
        if delta>0:lines.append(f'|{arm}/{stream:02}|{delta:+.10f}|')
lines+=['',
'因此放大有效尺度并非通用恢复方法。尤其UNIT的单个恶化幅度可明显大于面板平均收益；没有新训练总体区间，不得将平均结论推广至新数据。','',
'## 3. 经典Jensen账解释评分差，不增添独立证据','',
'对固定模型，总体有限评分F_P等于精确交叉熵加非负Jensen偏差（共同常数不影响差额），所以Δ偏差=ΔF_P−ΔR。下面是已有读数的事后代数派生，无新采样、优化或统计家族。','',
'|初值|保留模式的偏差增量|消去模式的active偏差增量|',
'|---|---:|---:|']
for arm in arms:
    d=risks[(arm,'SCALE2')]['mean_delta']
    lines.append(f'|{arm}|{cells[(arm,"SCALE2","retained","F_P")]["delta"]-d:.10f}|{cells[(arm,"SCALE2","eliminated","F_P")]["delta"]-d:.10f}|')
lines+=['',
'本候选增加的active有限K偏差大于精确风险下降，足以解释消去nuisance后F_P仍不支持它。此账不将偏差解释成全部训练风险的因果份额，也不证明提高K一定让原程序恢复。','',
'## 4. MU_SAFE：经验下降与风险收益仍是不同问题','',
'固定协方差的观察经验NLL满足H_μμ≼I/4，因此精确梯度的投影步μ′=clip(μ−4g)有下降界。本次复用旧GH24梯度并以新高阶PMF核验实际有限步；不把数值梯度当作精确保证。','',
'|初值|经验NLL平均变化|经验下降/上升/未决|平均ΔR|风险幅度门|',
'|---|---:|---|---:|---|']
for arm in arms:
    x=risks[(arm,'MU_SAFE')];c=x['empirical_descent_counts']
    lines.append(f'|{arm}|{x["empirical_NLL_mean_change"]:+.11f}|{c["decrease"]}/{c["increase"]}/{c["unresolved"]}|{x["mean_delta"]:+.11f}|未通过|')
lines+=['',
'新步在全部32个经验观察NLL上数值下降，说明旧固定大步失败不能排除均值更新。但两组平均风险变化都未跨过4e−6负向分辨阈；四个MU_SAFE条件有限F_D区间均跨零。总体F_P的NEAR微增、UNIT微降可在MC层面区分，不自动升级为经过风险幅度门的收益。两个候选步幅没有匹配，不能据此给均值/尺度因素排重要性名次。','',
'完整16项有限目标结果（单位10⁻⁶）：','',
'|初值/候选/模式/目标|ΔF及CI|方向|',
'|---|---:|---|']
for x in r['family']:
    lines.append(f'|{x["arm"]}/{x["candidate"]}/{x["mode"]}/{x["target"]}|{cell(x["arm"],x["candidate"],x["mode"],x["target"])}|{x["direction"]}|')
lines+=['',
'## 5. 本地核验、资源和未解决范围','',
f'- 64候选的192个GH32/40/48 PMF、32尺度候选的一维/四维边际资格全部通过。固定8个GH44对称根通过，最大PMF/KL差为{max(x["max_PMF_error"] for x in gh["records"]):.3g}/{max(x["KL_error"] for x in gh["records"]):.3g}。',
f'- 全部96模型风险、384次能力界、16项MC统计独立表达式复算通过；风险最大差{audit["max_risk_error"]:.3g}，768项直接概率评分最大差{audit["max_direct_score_error"]:.3g}。独立一维GH96边际差{audit["max_marginal_error"]:.3g}。这是本地数值核验，不是外部专家复现。',
f'- MC绝对精度16/16通过，最大SE={max(x["SE"] for x in r["family"]):.6g}；仍有5项方向未决，绝对精度门不保证微小效应可辨。df2047开发t仅处理固定面板MC误差，不能代替新训练总体推断。',
f'- 0新观察/训练轨迹/独立基线拟合；64预定候选，570,687,488主4D节点+29,984,768审计节点。128个基点GH64边际，{r["marginal_root_calls"]}次求根GH64函数调用，生产与独审各256次GH96边际；3,276,800新prior向量/851,968,000 Gaussian/6,291,456个pattern评分，独审另再生3,328,000原Gaussian。',
f'- 主计算{r["seconds"]:.2f}秒，GH独审{gh["seconds"]:.2f}秒；不是端到端成本匹配。制造资格两次及无历史数据的隔离demo另记录，不算新科学独立重复。',
'- 冻结前复核补记了SCALE2能力界算术次数，并显式添加边际/经验下降资格；早期冻结与资格收据原样保留。真实面板只有一次运行，无重试、换步长或扩预算。',
'- GH一致性不构成严格余项证书。能力下界条件于已冻结数值教师；风险幅度门只是操作性分辨标准。','',
'## 6. 下一轮的具体问题','',
'当前证据支持把“能力不足”和“有限目标对尺度恢复的惩罚”同时保留，不能仅由RHO可改进将恢复不足归为优化历史。下一项更有区分力的问题是：在这些已保存的SCALE2端点，active有限K偏差随K变化是否按足以改判候选的幅度减小？应另冻一个小型K/精度对照，保留相同端点与数据，区分剩余目标偏差和程序难以利用好方向；不能把它当作重新训练已获授权的自动队列。','',
'顺序/坐标训练需要能区分相反解释的合同，本轮没有启动。USE方面接受“简单修复可用也可有用途价值”，当前独立子类仍没有增加新的使用判断，不因接口更完整而追加训练。旧REC/USE、v9.1和所有原失败保持；本轮作为独立技术包交付，不自动形成v10。']
with (DOC/'RESULTS.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'report_written':True,'source':'frozen RESULTS and audit receipts','new_scientific_computation':False}))

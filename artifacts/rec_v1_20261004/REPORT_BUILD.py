"""Render REC-v1 tables from saved results; no scoring, fitting or RNG."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/rec_v1_20261004'
DOC = ROOT / 'docs/08_research_研究工作/REC_v1_20261004'
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
r = read(OUT / 'REC1/RESULTS.json')
gh = read(OUT / 'REC1/INDEPENDENT_GH.json')
audit = read(OUT / 'REC1/INDEPENDENT_ARITHMETIC.json')['result']
risks = {(x['arm'], x['state']): x for x in r['descriptive_risk']}
cells = {(x['arm'], x['candidate'], x['mode'], x['target']): x for x in r['primary_MC_family']}
arms = ('NEAR_FREE', 'UNIT_FREE')

lines = ['# REC-v1 结果：可实现基线改变恢复解读，实际输出仍有有益相关改进', '',
'2026-10-04。REC0–REC1按各自预先冻结合同完成。新结果只属于本REC包；v9.1、原风险区间、原数值失败及封存源码均不改写。', '',
'**这批FREE输出的平均风险低于同数据拟合的独立基线；预定、保持边际不变的相关扰动还能同时改善有限目标和风险。** 这使“没有依赖收益”以及“最终点附近全部有益候选仍被有限目标惩罚”的解释不适用于本次面板平均诊断。它不证明全局恢复、逐输出显著改善或训练历史原因。', '',
'## 1. 冻结对象与独立基线', '',
'原16份N4096数据各有NEAR/UNIT两个FREE TAIL，共32个单Gaussian输出。新增16次闭式独立模型拟合，边际固定为(count+.5)/(4096+1)。不使用真实边际选平滑、不训练新模型。下表风险是旧mixed教师数值PMF下的KL；各FREE行为16流等权均值，没有新的训练总体置信区间。', '',
'|模型|面板平均KL|FREE减同数据独立基线|FREE减真实产品|',
'|---|---:|---:|---:|',
f'|真实产品（oracle）|{r["true_product_risk"]:.10f}|—|—|',
f'|同数据独立拟合|{r["baseline_mean_risk"]:.10f}|—|—|']
for arm in arms:
    row = risks[(arm, 'BASE')]
    lines.append(f'|{arm}|{row["mean_risk"]:.10f}|{row["mean_versus_same_data_independent"]:+.10f}|{row["mean_risk"]-r["true_product_risk"]:+.10f}|')
lines += ['',
'两种初值在本面板平均上都优于可实现独立拟合，同时平均上仍劣于真实产品边际。真实产品使用oracle边际，不能将其优势与同数据可部署比较混成一个结论。原“尚未确认超过真实产品”的主区间结论保持；新基线比较是条件面板描述，未新增跨训练数据的显著性声明。逐流差额与全部PMF保存在[REC1原始结果](../../../artifacts/rec_v1_20261004/REC1/RESULTS.json)。', '',
'## 2. REC0排除了什么', '',
'独立证明正iid权重的逆矩上界及Gaussian-logistic特化成立。32个最终tau均低于预定0.053；共同碰撞上界0.020953278787240716，低于mixed GH40门槛0.04309002908822146。GH24/32/40最小门槛与包络间隔为0.02213675029871356。', '',
'因此在虚拟真实产品基点存在某个合法相关结构与非负尺度激活使原有限目标和风险同降，不能再以该点“全相关类统一稳定”解释最终尺度。此结论不涉及实际学得C、偏移均值、非零尺度或Adam路径。证明与数值误差范围见[REC0](REC0.md)及[阶段结果](../../../artifacts/rec_v1_20261004/REC0/RESULTS.md)。', '',
'## 3. REC1实际状态的两个有限扰动', '',
'每个实际输出用原训练标签构造GH24观察似然梯度；MU只变均值，RHO只变相关系数，分别按maxabs归一化走固定0.05并作预定可行修复。s与tau保持。方向不使用教师或评价bank，但依赖正确active块这一受控结构信息。RHO精确保留一维观察边际，因此其风险下降确实是依赖参数变化带来的同状态收益；MU的坐标标签不是唯一因果分解。', '',
'下面均为候选减基点。风险没有新增训练总体区间；四个评分列是面板平均delta及Bonferroni q=16、alpha=.05的MC近似区间。为易读，**所有数值乘以10⁶**；负数较好。保留/消去nuisance时active观察模型相同。', '',
'|初值/候选|ΔR ×10⁶|保留 F_D [CI] ×10⁶|保留 F_P [CI] ×10⁶|消去 F_D [CI] ×10⁶|消去 F_P [CI] ×10⁶|',
'|---|---:|---:|---:|---:|---:|']
for arm in arms:
    for state in ('MU', 'RHO'):
        row = risks[(arm, state)]
        values = []
        for mode in ('retained', 'eliminated'):
            for target in ('F_D', 'F_P'):
                cell = cells[(arm, state, mode, target)]
                lo, hi = cell['interval']
                values.append(f'{cell["delta"]*1e6:+.3f} [{lo*1e6:+.3f}, {hi*1e6:+.3f}]')
        lines.append(f'|{arm}/{state}|{row["mean_delta"]*1e6:+.3f}|'+'|'.join(values)+'|')
lines += ['',
'RHO两初值的平均风险分别下降约6.47e−6、2.66e−5，超过预定2e−6负向分辨阈；保留nuisance时F_D、F_P均支持这个候选，消去后排序也同向。由此确认**本次实际状态附近仍有有限幅度、保持边际的有益依赖改进余量**，而不是仅靠虚拟产品点的二阶式外推。平均改进较小，RHO后的平均风险仍高于真实产品；没有得到完整恢复或优化器可达性结论。', '',
'MU固定步在两初值的平均风险和四种有限目标上都变差。该候选失败原样保留，不缩小步幅重试；它不能排除其他均值幅度或方向，也不能证明边际误差无关。本报告不把负的无穷小经验梯度当作有限步必降的保证。', '',
'F_D/F_P在所测面板平均候选上没有出现相反排序，这只约束所测候选，不能排除其他方向的数据权重影响。没有比较两种nuisance模式的方向差额是否显著，也不能由两者同号断言残余nuisance的作用为零。', '',
'## 4. 原状态目标读数', '',
'以下评分都已减去共同常数256 log2；F_D只用原数据的16标签经验权重，F_P只用固定教师权重。通过nuisance Gaussian对称性精确折叠原标签，再对prior做独立MC。该折叠针对期望目标，不声称每个原生训练bank不受标签影响。下表仅描述性报告均值，不新增检验家族。', '',
'|初值|保留 F_D|消去 F_D|保留 F_P|消去 F_P|',
'|---|---:|---:|---:|---:|']
for arm in arms:
    means = []
    for target, mode in (('F_D','retained'), ('F_D','eliminated'), ('F_P','retained'), ('F_P','eliminated')):
        values = [next(x['reduced_mean'] for x in record['score_rows'] if x['arm']==arm and x['state']=='BASE' and x['mode']==mode and x['target']==target) for record in r['records']]
        means.append(sum(values)/16)
    lines.append('|'+arm+'|'+'|'.join(f'{x:.10f}' for x in means)+'|')
lines += ['',
'上述差额只隔离同状态计算准则，既不是完整训练历史的因果份额，也不能将F_D−F_P称为全部数据噪声。这里评价的是稳定float64数学目标；制造输入与原API32的一致性资格不认证整条旧native32训练轨迹。', '',
'## 5. 数值、推断与资源', '',
f'- 新64候选的192个GH32/40/48 PMF全部通过；32基点复用旧96个合格PMF。固定8个RHO候选的独立GH44对称根检查{ "通过" if gh["passed"] else "未通过"}，最大PMF差{max(x["max_PMF_error"] for x in gh["records"]):.3g}、最大KL差{max(x["KL_error"] for x in gh["records"]):.3g}。',
f'- 主MC精度16/16通过，最大SE={max(x["MC_SE"] for x in r["primary_MC_family"]):.6g}，低于2e−5。8个RHO方向单元下降，8个MU单元上升；统计精度通过不等于每种候选成功。',
'- 每项SE由16流各2048个paired-bank差的方差合成，df2047开发t近似仅处理新MC误差。没有把32输出或64候选当成独立训练重复，没有给跨训练总体置信区间。GH阶数与独立根吻合不构成严格积分余项证书，2e−6风险分辨阈也不是严格误差界。',
f'- 独立算术核查覆盖全部数据counts、16基线、96模型PMF风险、候选规则、全部保存bank的16项家族；风险复算最大差{audit["max_saved_risk_difference"]:.3g}。固定4流首bank的768个pattern score直接概率乘积复核最大差{audit["max_direct_probability_score_difference"]:.3g}，首chunk Gaussian哈希一致；RHO边际数值变化最大{audit["max_RHO_preserved_marginal_difference"]:.3g}。',
f'- 0新观察、0新优化轨迹；16闭式拟合、64候选、3,276,800新prior向量（851,968,000 Gaussian标量）、6,291,456个pattern-bank-state-mode评分。方向构造10,616,832 GH节点，候选570,687,488节点，独审29,984,768节点。首bank审计另再生3,328,000个原Gaussian，不增加独立评价bank。',
f'- 主阶段本机墙钟{r["seconds"]:.2f}秒，GH独审{gh["seconds"]:.2f}秒；只记录本次时间，不作为方法成本相等或速度比较。制造资格及其相同确定性重放另见执行账目。', '',
'## 6. 本周期决策', '',
'REC0排除了虚拟产品基点的统一碰撞稳定解释；REC1表明真实产品基线掩盖了本面板相对同数据独立拟合的收益，并确认一个保持边际的实际状态候选同时改善目标与风险。最低完成标准已满足。', '',
'**本周期停在REC1，不启动REC2。** 终点改进余量没有区分早期尺度压缩、有限步数、梯度噪声、更新坐标或其他优化因素；当前证据不足以在原计划的顺序/坐标二选一中提出有鉴别力的单次控制。不能仅因还有改进空间就延长至2048步。所有未测方向和历史因果份额继续未决。', '',
'USE0独立报告确认当前单候选未通过完整用途资格，依合同停在USE0；这不否定其他使用场景。见[USE结果](../USE_v1_20261004/RESULTS.md)。本次不生成v10或改写v9.1。']
target = DOC / 'RESULTS.md'
with target.open('x', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines)+'\n')
print(str(target))

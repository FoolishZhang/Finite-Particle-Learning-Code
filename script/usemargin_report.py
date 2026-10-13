"""Render frozen margin diagnostics; never changes source studies or statistical tests."""
from pathlib import Path
import json
import math
import os
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/use_margin_v4_20261005'
DOC = ROOT / 'docs/08_research_研究工作/USE_v4_margin_20261005'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/usemargin_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r = json.loads((OUT / 'RESULTS.json').read_text(encoding='utf-8'))
rows, groups = r['records'], r['groups']
assert not (DOC / 'RESULTS.md').exists()
natural = [x for x in rows if x['case'] == 'Optdigits' and x['method'] == 'RAW100']
synthetic = [x for x in rows if x['case'] != 'Optdigits' and x['method'] == 'RAW100']
tail = next(x for x in synthetic if x['case'] == 'WEAK' and x['rep'] == 27)
natural0 = next(x for x in natural if x['rep'] == 0)
weak_total = math.fsum(x['selected_risk_difference'] for x in synthetic if x['case'] == 'WEAK')
derived = dict(result_sha256=hashlib.sha256((OUT / 'RESULTS.json').read_bytes()).hexdigest(),
               synthetic_RAW_changed=sum(x['changed'] for x in synthetic),
               synthetic_RAW_harmed=sum(x['selected_risk_difference'] > 1e-12 for x in synthetic),
               synthetic_RAW_helped=sum(x['selected_risk_difference'] < -1e-12 for x in synthetic),
               weak27_contribution_to_mean=tail['selected_risk_difference'] / 32,
               weak27_share_of_signed_total=tail['selected_risk_difference'] / weak_total,
               scope='Descriptive derived quantities, never leave-one-out reanalysis or new hypothesis tests')
with (OUT / 'DESCRIPTIVE.json').open('x', encoding='utf-8', newline='\n') as f:
    json.dump(derived, f, ensure_ascii=False, indent=2); f.write('\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
ax = axes[0]
ax.plot([.065, .15], [.065, .15], '--', color='#333333', linewidth=1.2, label='Switching boundary')
ax.scatter([x['runner_reference_gap'] for x in natural],
           [-x['runner_relative_error'] for x in natural], s=31, alpha=.8, color='#176899')
ax.axhline(0, color='#999999', linewidth=.6)
ax.set(xlabel='AGH loss gap: L010 minus L001', ylabel='Relative error favoring L010',
       title='A. Optdigits: RAW100 keeps the winner', xlim=(.065, .15), ylim=(-.115, .165))
ax.text(.069, .14, 'Above line: L010 would win', fontsize=9)
ax.text(.069, -.097, 'All 32 realized points stay below', fontsize=9, color='#176899')
ax.grid(alpha=.15)
ax = axes[1]
for i, case in enumerate(('NULL', 'WEAK', 'STRONG')):
    subset = [x for x in synthetic if x['case'] == case]
    ax.scatter([i + .12 * math.sin(x['rep'] * 2.399) for x in subset],
               [x['selected_risk_difference'] for x in subset], s=29, alpha=.75,
               color=('#5d7790', '#b45d25', '#278079')[i])
ax.set_yscale('symlog', linthresh=1e-5)
ax.axhline(0, color='#666666', linewidth=.8)
ax.set_xticks([0, 1, 2], ['NULL', 'WEAK', 'STRONG'])
ax.set(ylabel='Teacher KL: RAW100 choice minus GH choice',
       title='B. Changed choices can help or harm')
ax.annotate('WEAK rep27: +0.00586', xy=(1 + .12 * math.sin(27 * 2.399), tail['selected_risk_difference']),
            xytext=(.0, .0075), fontsize=9, arrowprops=dict(arrowstyle='-', color='#777777'))
ax.grid(axis='y', alpha=.15)
fig.subplots_adjust(left=.09, right=.98, bottom=.2, top=.88, wspace=.37)
fig.text(.5, .035, 'Saved outputs only. Panel B uses a symmetric log scale; dots are individual flows, not confidence intervals.', ha='center', fontsize=9)
(DOC / 'figures').mkdir(exist_ok=True)
fig.savefig(DOC / 'figures/margin_and_risk.png', dpi=180, facecolor='white')
fig.savefig(DOC / 'figures/margin_and_risk.svg', facecolor='white')
plt.close(fig)

lines = ['# USE-v4：评分误差、候选改选与风险损失是三个层次', '',
'**全部128个已有流程的512个方法决定已完成事后解释。Optdigits的RAW100存在不可忽略的实现误差，但全部候选仍未越过赢家；合成RAW100发生28次改选，19次增加教师风险、9次降低教师风险。** 本轮没有新增训练、数据、prior或积分，原置信区间和失败记录保持。', '',
'![候选间隔与选择风险](figures/margin_and_risk.png)', '',
'左图是Optdigits的RAW100：L010在全部32流程中既是AGH次优者，也是最强近似竞争者；点位于切换线下。右图是合成RAW100全部96流相对GH选择的教师KL差，使用对称对数轴，零值也保留。两面板单位与条件不同，不能合并均值。', '',
'## 1. 原数据和本轮口径', '',
'每个流程四候选，固定近似RAW100、OUT16、INNER1600、JVI100，对照原GH/AGH验证分数。高精度赢家为w，g_j=ℓ_j−ℓ_w，d_j=(h_j−ℓ_j)−(h_w−ℓ_w)。逐候选复核h_j−h_w=g_j+d_j；它是排序恒等式，不是新理论。共1536个竞争比较，原1e-12规则下没有近似或参考tie。', '',
'e=h−ℓ是单个保存prior池的实现误差，不是期望偏差。GH/AGH是已通过原数值资格的参考，不声称严格解析误差界。所有下文充分条件只核对这些已保存分数，不给未来新池稳定概率或无害保证。', '',
'## 2. Optdigits：误差存在，间隔未被跨越', '',
'高精度赢家全部为L001，次优全部为L010；间隔范围[0.0768870,0.1432202]，中位0.1124287 nats/图像。', '',
'|方法|最大绝对评分误差A的中位数|joint候选误差范围中位数|全部流程/候选的最小剩余间隔|绝对界通过|平移不变界通过|改选|',
'|---|---:|---:|---:|---:|---:|---:|']
for g in groups:
    if g['case'] == 'Optdigits':
        lines.append(f"|{g['method']}|{g['max_absolute_error']['median']:.7f}|{g['joint_error_range']['median']:.7f}|{g['approximate_min_gap']['min']:.7f}|{g['absolute_bound_certificate']}/32|{g['range_bound_certificate']}/32|0/32|")
lines += ['', '充分条件：令Δ=min g_j、A=max|e_j|、E=max e−min e。Δ>2A+1e-12或更紧的Δ>E+1e-12都足以保持赢家；不通过不等于会改选。公共偏移不影响排序。表中的两个条件均使用包括IND的完整四候选池；joint三候选误差范围仅作抵消诊断，不冒充完整候选池证书。', '',
'RAW100仅4/32满足绝对界，平移不变界增加到15/32；OUT16从0/32增加到21/32。其余流程仍因逐候选的有符号相对误差没有越过各自间隔而稳定，不能将充分界失败解释为实际失稳。INNER1600的绝对界已覆盖32/32；这使“误差较小”和“误差有抵消/间隔保护”能具体区分。', '',
f"反馈所举rep0得到复核：AGH间隔{natural0['runner_reference_gap']:.8f}，L001/L010的RAW误差分别{natural0['realized_errors'][1]:+.8f}/{natural0['realized_errors'][2]:+.8f}，相对误差{natural0['runner_relative_error']:+.8f}，最终间隔{natural0['runner_approximate_gap']:.8f}。但这个方向不是所有流程共有：RAW相对误差在14/32流程缩小原间隔，18/32扩大间隔，均未改选。", '',
'高精度测试赢家恰也都是L001，故本候选池五方法测试差为零。32次仍条件于同一官方数据；原零事件同时概率上界约12.8%和85/192优化收敛限制保持。没有据结果扩展正则网格。', '',
'## 3. 合成流程：改选不等于风险一定变坏', '',
'下表全部相对GH验证赢家。均值是教师KL差；OUT16−GH等新增汇总仅为描述性，不是旧15项家族中的新显著性结论。', '',
'|条件|方法|改选|风险增加/降低|个体差>0.0002|平均教师KL差|最大教师KL差|',
'|---|---|---:|---:|---:|---:|---:|']
for g in groups:
    if g['case'] != 'Optdigits':
        lines.append(f"|{g['case']}|{g['method']}|{g['changed']}/32|{g['harmed']}/{g['helped']}|{g['exceed_old_individual_epsilon']}|{g['selected_risk_difference']['mean']:+.9f}|{g['selected_risk_difference']['max']:+.9f}|")
lines += ['', 'RAW100在NULL/WEAK/STRONG分别改选3/10/15次，总28次；其中19次增加、9次降低教师KL。高精度验证赢家不等于总体oracle，所以近似改选可以偶然改善风险。逐例越过0.0002只是沿用旧量级描述，不能替代旧均值及区间门。全部旧RAW100−GH区间跨零的结论保持。', '',
'合成高精度最小间隔中位数为NULL 6.16e-7、WEAK 1.64e-4、STRONG 2.53e-4，远小于自然数据此候选池间隔，但维数、教师和损失单位不同，不把这个跨研究比较当成因果效应。', '',
'WEAK的9/10次RAW改选发生在L0/L002之间；另1次是rep27从L0变为L020。STRONG的15次全部在L0/L002之间。频繁改变接近候选，与造成大风险损失不是一回事。', '',
'## 4. 保留尾部：只看参考次优候选会漏掉真正代价', '',
'WEAK rep27：L002相对GH赢家L0的验证间隔仅0.00015483，RAW相对误差−0.00352031，已反转排序；但L002的教师风险比L0低0.00003665。真正RAW赢家是L020，它不是参考次优：原间隔0.00719116、相对误差−0.01202236，得到近似间隔−0.00483120，教师KL代价+0.00585984，独立测试NLL代价+0.00536786。', '',
f"该流对WEAK平均RAW−GH差的贡献为{derived['weak27_contribution_to_mean']:.9f}，约占有符号总差{100*derived['weak27_share_of_signed_total']:.2f}%。这是对原均值的算术解释，不删除该流、不重算删除后检验，也不把单个尾部当总体危害证据。完整三竞争者而非只有runner-up，才足以解释实际选择及代价。", '',
'## 5. 这项分析新增了什么', '',
'USE阴性结果不意味着评分准确：Optdigits在较大误差下仍由候选间隔和模型间相对误差保护当前选择。合成流程有真实改选，但候选风险接近、风险变化方向混合，且WEAK的均值受保留的尾部个例明显影响。这为原结果提供了决策层解释，没有增加独立统计证据。', '',
'两项USE的随机积分均在二维latent内进行；观察维数64并不等于原理论中不断增加独立非退化nuisance log-weight。用途阴性约束普遍实践叙事，不是原高维方向定理的直接反例。REC-v4的正反学习控制与USE决策诊断应各自承担证据职责，见[作者文本](AUTHOR_TEXT.md)。', '',
'制造数组6例通过；独立Decimal审计全部512决定、1536恒等式/竞争比较、16分组及17536标量，最大差4.74e-15。主派生约0.11秒，无新增训练/prior/积分。完整结果与原输入路径见 `artifacts/use_margin_v4_20261005/RESULTS.json`；原研究保持封存。']
with (DOC / 'RESULTS.md').open('x', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines) + '\n')
print(json.dumps(dict(report=True, figures=2, new_experiments=0)))

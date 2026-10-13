"""Readable tables from frozen learning outputs; no new scientific computation."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/joint_scale_learning'
DOC=ROOT/'docs/08_research_研究工作/两组尺度共同学习与恢复边界_20261004'


def main():
    r=json.loads((OUT/'REFERENCE/RESULTS.json').read_text('utf8'))
    tr=json.loads((OUT/'TRAIN/RESULTS.json').read_text('utf8'))
    arms=r['config']['arms'];p=np.array(r['teacher_marginals'])
    patterns=np.array(list(__import__('itertools').product((0,1),repeat=4)))
    f=lambda v:f'{v:.10f}'
    lines=['# 全部预定学习读数','','主读出为最后64次更新后的μ与完整Σ的均值构成的单一Gaussian模型。所有风险单位为nats。没有LAST或中间checkpoint风险积分。', '',
           '## 主风险家族','','六项共享Bonferroni开发t家族，alpha=.05、df15、16独立数据/prior流，SE门均为.0002；积分误差不在区间内。真实产品边际风险为 '+f(r['truth_product_risk'])+'。','',
           '|初值/对比|风险差|原共同区间|SE|方向门|精度门|','|---|---:|---|---:|---|---|']
    for x in r['primary']:
        lines.append(f"|{x['name']}|{f(x['mean'])}|[{f(x['lower'])}, {f(x['upper'])}]|{f(x['SE'])}|{x['effect_passed']}|{x['precision_passed']}|")
    lines += ['',f"方向{r['primary_effects_passed']}/6、精度{r['primary_precision_passed']}/6；同项合并"+str(sum(x['effect_passed'] and x['precision_passed'] for x in r['primary']))+'/6。所有失败保持，不挑初值或增加预算。', '',
              '## 六程序描述','','均值是16个流的均值，下面不新增置信区间。边际KL为四个观察边际KL之和，不能当作因果份额。','','|程序|mean joint KL|mean 边际KL和|mean τ|mean active方差|最小/最大输出τ|','|---|---:|---:|---:|---:|---|']
    for arm in arms:
        models=[x['outputs'][arm]['TAIL'] for x in tr['records']]
        taus=np.array([x['tau'] for x in models]);var=np.array([np.trace(x['covariance'])/4 for x in models])
        margins=[]
        for x in r['records']:
            if x['arm']==arm:
                q=np.array(x['main_pmf'])@patterns
                margins.append(float(np.sum(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q)))))
        lines.append(f"|{arm}|{f(r['descriptive'][arm]['mean_joint_KL'])}|{f(np.mean(margins))}|{f(taus.mean())}|{f(var.mean())}|[{f(taus.min())}, {f(taus.max())}]|")
    lines += ['', '## 全16流风险','','|流|'+'|'.join(arms)+'|','|---:|'+'---:|'*6]
    for s in range(16):lines.append('|'+str(s)+'|'+'|'.join(f(r['descriptive'][a]['risks'][s]) for a in arms)+'|')
    lines += ['', '## 同家族派生FREE−ZERO','','以下是原共享区间的事后矩形相减，不新增更窄检验、SE门或非劣门。','','|初值|FREE−ZERO均值|派生共同范围|','|---|---:|---|']
    for start in ('NEAR','UNIT'):
        a=next(x for x in r['primary'] if x['name']==start+'_FREE_minus_PRODUCT')
        b=next(x for x in r['primary'] if x['name']==start+'_ZERO_minus_PRODUCT')
        lines.append(f"|{start}|{f(a['mean']-b['mean'])}|[{f(a['lower']-b['upper'])}, {f(a['upper']-b['lower'])}]|")
    lines += ['', '## 数值资格与工作量','','制造数组kernel结构/微分资格通过；compute数值5/6，NEAR_FREE共享τ梯度的4.5489e−5超过原4e−5，原失败保留。', '',
              '主训练原API32/math64资格：'+str(sum(x['numerical_qualification']['passed'] for x in tr['records']))+'/384。最大loss差 '+f(max(x['numerical_qualification']['maximum_loss_difference'] for x in tr['records']))+'；最大梯度差 '+f(max(x['numerical_qualification']['maximum_gradient_difference'] for x in tr['records']))+'。此数值家族不与风险主门合并，也不追认制造数组失败。', '',
              '主TAIL积分资格：'+str(sum(x['qualification_passed'] for x in r['records']))+'/96模型；288个PMF、856031232四维节点。每模型固定32/40/48阶且40为主，数值吻合不是严格积分余项证书。', '',
              '65536新完整观察；96私有优化、49152主API及384额外数值API。52428800训练prior向量/13631488000Gaussian；数据262144Gaussian/17039360Uniform。配对的六程序共享输入只计一次，0神经网络。', '',
              '独审另列完整输入/保存更新重算、384底层状态及32GH44参考；不能称全底层重新训练。所有原失败与源/收据公开保存。']
    with (DOC/'RESULTS.md').open('x',encoding='utf8',newline='\n') as h:h.write('\n'.join(lines)+'\n')


if __name__=='__main__':main()

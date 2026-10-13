"""Descriptive KL identity from already integrated outputs; no new inference."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/joint_scale_learning'
DOC=ROOT/'docs/08_research_研究工作/两组尺度共同学习与恢复边界_20261004'


def main():
    source=OUT/'REFERENCE/RESULTS.json';r=json.loads(source.read_text('utf8'))
    p=np.array(r['teacher_marginals']);tc=r['truth_product_risk']
    patterns=np.array(list(itertools.product((0,1),repeat=4)))
    rows=[]
    for x in r['records']:
        q=np.array(x['main_pmf'])@patterns
        marginal=float(np.sum(p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))))
        own_product=tc+marginal
        rows.append(dict(stream=x['stream'],arm=x['arm'],marginal_KL_sum=marginal,
                         own_product_KL=own_product,joint_KL=x['reference_joint_KL'],
                         joint_logscore_gain_over_own_product=own_product-x['reference_joint_KL']))
    means={arm:{key:float(np.mean([x[key] for x in rows if x['arm']==arm]))
                for key in ['marginal_KL_sum','own_product_KL','joint_KL','joint_logscore_gain_over_own_product']}
           for arm in r['config']['arms']}
    result=dict(source=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                formula='KL(P||prod Q_i)=TC(P)+sum_i KL(P_i||Q_i); gain=KL(P||prod Q_i)-KL(P||Q)',
                records=rows,means=means,new_integrals=0,new_RNG=0,new_tests=0,new_confidence_intervals=0,
                scope='Posthoc descriptive identity, not mutual information, a causal share or confirmation of any primary gate.')
    with (OUT/'POSTHOC_PRODUCT_IDENTITY.json').open('x',encoding='utf8',newline='\n') as h:
        json.dump(result,h,ensure_ascii=False,indent=2);h.write('\n')
    text='\n## 事后描述：与各输出自身产品边际比较\n\n'
    text+='经典恒等式 KL(P∥∏Q_i)=TC(P)+ΣKL(P_i∥Q_i)。下表收益为该输出自身产品模型的风险减去联合模型风险；正数表示联合模型优于它自身的产品化。它不是互信息或因果份额，没有新增检验/区间，不替代原真实产品基线门。\n\n'
    text+='|程序|mean 自身产品风险|mean joint相对自身产品的logscore收益|\n|---|---:|---:|\n'
    for arm,x in means.items():text+=f"|{arm}|{x['own_product_KL']:.10f}|{x['joint_logscore_gain_over_own_product']:.10f}|\n"
    pth=DOC/'RESULTS.md';old=pth.read_text('utf8');assert '事后描述：与各输出自身产品边际比较' not in old
    pth.write_text(old+text,encoding='utf8',newline='\n')
    print(json.dumps(means))


if __name__=='__main__':main()

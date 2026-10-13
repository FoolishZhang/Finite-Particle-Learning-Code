"""Render fixed active-scale JSON readings to Markdown; no science or RNG."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/active_scale_boundary'
DOC=ROOT/'docs/08_research_研究工作/有效尺度自由度与有限曲率_20261004'


def main():
    r=json.loads((OUT/'SUMMARY/RESULTS.json').read_text('utf8'))
    ref=json.loads((OUT/'REFERENCE/RESULTS.json').read_text('utf8'))
    f=lambda v:f'{v:.10f}'
    lines=['# 全部固定读数', '', '来源：`artifacts/active_scale_boundary/SUMMARY/RESULTS.json` 与 `REFERENCE/RESULTS.json`。所有差额均为激活 t 减去产品边际 t=0；单位 nats，越小越好。参考 P 为归一化 GH40 数值权重。', '',
           '## 主有限差额 t=.2', '', 'q8、alpha=.025、df15，16独立prior流的Bonferroni开发t区间，条件于四个固定数值教师。不覆盖训练/数据总体随机性或积分误差。三个依赖教师的方向门6/6、八项SE门8/8；null不设置方向门。', '',
           '|教师|n|raw差|共同区间|SE|参考KL差|方向门|SE门|','|---|---:|---:|---|---:|---:|---|---|']
    for x in r['raw_primary_family']['results']:
        direction='控制' if x['direction_passed'] is None else str(x['direction_passed'])
        lines.append(f"|{x['teacher']}|{x['nuisance']}|{f(x['raw_score_contrast'])}|[{f(x['lower'])}, {f(x['upper'])}]|{f(x['SE_across_16_stream_means'])}|{f(x['reference_KL_contrast'])}|{direction}|{x['precision_passed']}|")
    lines += ['', '## nuisance碰撞与二阶门槛', '', 'χ家族q4、alpha=.025、B=32768，Hoeffding共同半宽 '+f(r['collision_family']['halfwidth_before_clipping'])+'；n0为解析1/K。下列曲率由同一个χ共同区间单调映射，条件于数值教师矩，不是新的独立检验。χ=E[1/ESS]，不能写成1/E[ESS]。', '',
              '|n|χ|共同区间|','|---:|---:|---|']
    for x in r['collision_family']['results']:
        lines.append(f"|{x['nuisance']}|{f(x['mean'])}|[{f(x['lower'])}, {f(x['upper'])}]|")
    lines += ['', '|教师|λmax(RY)|χ临界值|exact二阶曲率|产品模型KL|','|---|---:|---:|---:|---:|']
    for x in ref['records']:
        d=x['derived'];lines.append(f"|{x['name']}|{f(d['lambda_max_R_Y'])}|{f(d['collision_threshold'])}|{f(d['exact_scale_curvature'])}|{f(d['product_KL'])}|")
    lines += ['', '|教师|n|raw曲率|映射区间|全C横向稳定|构造方向下降|','|---|---:|---:|---|---|---|']
    for x in r['collision_family']['results']:
        for v in x['curvatures']:
            lines.append(f"|{v['teacher']}|{x['nuisance']}|{f(v['mean'])}|[{f(v['lower'])}, {f(v['upper'])}]|{v['raw_uniform_C_strict_min_qualified']}|{v['raw_constructed_descent_qualified']}|")
    lines += ['', '## 全部预定正激活描述', '', '此表展示所有n、t；只有上方明确列入q8的读数带主区间，其余不事后升级。', '',
              '|教师|t|参考KL|参考KL差|raw n0|n4|n16|n64|n256|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lookup={(x['teacher'],x['t'],x['nuisance']):x for x in r['descriptive_curves']+r['raw_primary_family']['results']}
    for x in ref['records']:
        for c in x['candidates']:
            vals='|'.join(f(lookup[(x['name'],c['t'],n)]['raw_score_contrast']) for n in [0,4,16,64,256])
            lines.append(f"|{x['name']}|{c['t']}|{f(c['joint_KL'])}|{f(c['KL_minus_product'])}|{vals}|")
    lines += ['', '## 计算与参考资格', '',
              '48个主PMF积分、63045632个四维Gaussian节点；4个零尺度产品PMF为解析式。40−32 PMF、KL与质量资格全部通过，24阶差额另作描述。数值交叉检查不是严格积分余项证书。', '',
              '32768新bank，3276800 prior向量；838860800 nuisance Gaussian + 13107200 active Gaussian = 851968000。全部教师/n/t共享输入，不重复计数。主math64评分，无新观察、拟合、神经训练或主native调用；独审另列。', '',
              '旧新数据 OUT4−FACT 原家族派生：+.0010272047，区间[+.0002362414,+.0018181681]。事后矩形相减，无新检验或SE门，不追认旧门。']
    with (DOC/'RESULTS.md').open('x',encoding='utf8',newline='\n') as h:h.write('\n'.join(lines)+'\n')


if __name__=='__main__': main()

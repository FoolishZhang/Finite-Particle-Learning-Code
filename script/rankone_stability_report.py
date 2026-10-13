"""Produce readable tables and a scientific figure from the frozen matrices."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/rankone_stability_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/RankOneStability_v1_20261007"
assert not (OUT/"DELIVERY_MANIFEST.json").exists(), "Sealed study: do not regenerate reports"


def read(p):return json.loads(Path(p).read_text("utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


v=read(OUT/"DIRECTIONAL_ANALYSIS.json")["validation"]
spectra=v["projected_spectra"]
sources=dict(v["sources"])
sources["artifacts/rankone_stability_v1_20261007/DIRECTIONAL_ANALYSIS.json"]=sha(OUT/"DIRECTIONAL_ANALYSIS.json")
sources["artifacts/rankone_stability_v1_20261007/AUDIT.json"]=sha(OUT/"AUDIT.json")
get=lambda r,a,t,k:next(x for x in spectra if (x["rep"],x["anchor"],x["target"],x["K"])==(r,a,t,str(k)))
corners=[]
for rep in range(4):
    for a in ["b8_s8","b8_sREF","bREF_s8","bREF_sREF"]:
        corners.append(dict(rep=rep,anchor=a,empirical_F8=get(rep,a,"empirical",8)["minimum"],
            population_F8=get(rep,a,"population",8)["minimum"],
            population_F64=get(rep,a,"population",64)["minimum"],
            population_integrated=get(rep,a,"population","inf")["minimum"]))
qual=[];primary_rows=[];resources=[]
for rep in range(4):
    folder=OUT/f"rep_{rep:02d}"
    resources.append(read(folder/"RESULTS.json")["resources"])
    for p in sorted(folder.glob("*_QUALIFICATION.json")):qual.extend(read(p)["targets"])
    for p in sorted(folder.glob("*_C.json")):primary_rows.extend(read(p)["matrices"])
assert len(qual)==len(primary_rows)==168
max_lap=max(x["C_D"]["laplacian_max_difference"] for x in qual)
max_eig=max(x["C_D"]["projected_eigenvalue_max_difference"] for x in qual)
max_ward=max(x["independent_Ward_max"] for x in primary_rows)
work={k:sum(r["high_level_factor_totals"][k] for r in resources)
      for k in resources[0]["high_level_factor_totals"]}
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
    sources=sources,corner_rows=corners,primary_matrices=168,all_rule_matrices=672,
    primary_qualified=sum(x["qualified"] for x in qual),
    maximum_C_D_laplacian_difference=max_lap,maximum_C_D_eigenvalue_difference=max_eig,
    maximum_primary_Ward_residual=max_ward,
    resources=dict(new_observations=0,new_random_draws=0,optimizer_updates=0,
                   statistical_alpha_spent=0,anchors=28,kernel_calls=112,
                   per_label_K_cells=sum(r["per_label_K_cells"] for r in resources),
                   high_level_factor_totals=work),
    full_parameter_stationarity_claimed=False,rigorous_quadrature_certificate=False,
    new_handoff_prompt_generated=False)
(OUT/"SUMMARY.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")

lines=["# 完整数值表：固定边际的角度曲率","",
       "数值为去掉共同旋转后的7维角度Hessian最小特征值，单位nats/radian²。"
       "C为主值，D为预定核对；168项主资格全部通过。积分目标的总体Hessian也就是观察KL的角度Hessian。","",
       "|原流|规范锚点|F8，经验|F8，总体|F64，总体|积分目标，总体|",
       "|---|---|---:|---:|---:|---:|"]
for r in corners:
    lines.append(f"|{r['rep']}|{r['anchor']}|"+ "|".join(f"{r[k]:.9f}" for k in
       ["empirical_F8","population_F8","population_F64","population_integrated"])+"|")
lines += ["","b8/bREF只指定偏置来源，s8/sREF指定行范数来源；所有锚点是全正共线构造，"
          "并非原训练终态。两目标都条件于既有冻结频率/teacher权重，没有新训练流。","",
          "## 同一风险改善方向","",
          "每个b8_s8锚点取预先规定的总体积分目标最小特征方向。"
          "这是teacher辅助的局部风险诊断，不是可部署修复程序。","",
          "|原流|同方向F8，经验|同方向F8，总体|同方向F64，总体|观察风险|",
          "|---|---:|---:|---:|---:|"]
for r in v["directional_rows"]:
    if r["anchor"]!="b8_s8":continue
    values={(z["target"],z["K"]):z["C"] for z in r["curvatures"]}
    lines.append(f"|{r['rep']}|"+ "|".join(f"{values[key]:.9f}" for key in
        [("empirical","8"),("population","8"),("population","64"),("population","inf")])+"|")
lines += ["","四流在这个相同方向上都出现风险负曲率、F8正曲率、F64负曲率。"
          "因此是局部二阶意义的偏好冲突；没有给有限步长或全训练过程保证。","",
          "## 固定b8、插值行长的全五点","",
          "|原流|目标|λ=0|λ=.25|λ=.5|λ=.75|λ=1|数值异号夹区|",
          "|---|---|---:|---:|---:|---:|---:|---|"]
for r in v["K8_length_interpolation"]:
    interval="; ".join(f"({z['left']}, {z['right']})" for z in r["sign_change_intervals"]) or "无"
    lines.append(f"|{r['rep']}|{r['target']}|"+
        "|".join(f"{z:.9f}" for z in r["minimum_eigenvalues"])+f"|{interval}|")
lines += ["","仅记录既定五点。连续性最多支持至少一个过零位置的夹区；没有证明唯一或单调，"
          "也没有追加二分点。","",
          "## 数值与驻定限制","",
          f"- C/D矩阵最大元素差：{max_lap:.6g}；投影谱最大差：{max_eig:.6g}。",
          f"- 主规则独立Ward残差最大值：{max_ward:.6g}，不能用强制行和为零替代这个检查。",
          "- 共同旋转零模已明确排除；非旋转正定只定义固定b/a的角度局部极小。",
          "- b8_s8经验F8的偏置梯度∞范数约.000287–.00253，行长梯度约.000700–.00138，"
          "没有确认完整参数驻定。总体梯度更不能据经验拟合写成零。",
          "- 跨规则吻合及独立执行审计不是严格Gaussian求积余项认证。"
          "A/B结果与所有规则配对差均保存；本轮实际未出现资格失败，不虚构阴性。",
          "- 原JointDependence的强载荷K2门失败、profile混合排序及路径摘要勘误保持原貌。"]
(DOC/"NUMERICAL_TABLES.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
fig,ax=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
colors=["#146b78","#c26226","#65539c","#538450"]
for r in v["K8_length_interpolation"]:
    if r["target"]=="population":
        ax[0].plot(r["lambdas"],r["minimum_eigenvalues"],"o-",lw=1.6,
                   color=colors[r["rep"]],label=f"Saved case {r['rep']}")
ax[0].axhline(0,color="#475569",lw=1)
ax[0].set(xlabel=r"Loading norms: $(1-\lambda)s_8+\lambda s_{\mathrm{REF}}$",
          ylabel="Minimum non-rotation angular curvature",
          title="(a) Population F8 versus loading norms")
ax[0].set_xticks([0.,.25,.5,.75,1.])
ax[0].legend(fontsize=8)
x=np.arange(4);width=.23
for i,(K,name,color) in enumerate([("8","F8","#146b78"),("64","F64","#c26226"),("inf","Observation risk","#65539c")]):
    values=[]
    for rep in range(4):
        r=next(z for z in v["directional_rows"] if z["rep"]==rep and z["anchor"]=="b8_s8")
        values.append(next(z["C"] for z in r["curvatures"] if z["target"]=="population" and z["K"]==K))
    ax[1].bar(x+(i-1)*width,values,width=width,label=name+" (population)" if K!="inf" else name,color=color)
ax[1].axhline(0,color="#475569",lw=1)
ax[1].set_xticks(x,[f"Case {r}" for r in range(4)])
ax[1].set(ylabel="Curvature along the same risk-descent direction",
          title="(b) At the saved K8 loading norms")
ax[1].legend(fontsize=8)
for a in ax:a.grid(axis="y",alpha=.18)
fig.suptitle("Fixed bias and row norms within each angular comparison",fontsize=13)
outputs={}
for ext in ["png","svg"]:
    p=OUT/f"stability_and_norms.{ext}";fig.savefig(p);outputs[p.relative_to(ROOT).as_posix()]=sha(p)
plt.close(fig)
(OUT/"FIGURE.json").write_text(json.dumps(dict(source_sha256=sha(__file__),
    directional_analysis_sha256=sha(OUT/"DIRECTIONAL_ANALYSIS.json"),outputs=outputs,
    connecting_segments_do_not_prove_monotonicity=True,new_scientific_computation=False),
    indent=2)+"\n",encoding="utf-8",newline="\n")
print(json.dumps(dict(summary_written=True,qualified=summary["primary_qualified"],matrices=168,
                     maximum_matrix_difference=max_lap,maximum_Ward_residual=max_ward)))

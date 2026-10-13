"""Deterministic v2 figures/tables; no additional observations or fitting."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import argparse
import csv
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/constraint_boundary_v2"
FIG = BASE / "figures"
DOC = ROOT / "docs/08_research_研究工作/全局约束与采样门槛_20260929"
SOURCES = [
    BASE/"M/FREEZE.json", BASE/"M/PARAMETERS.json", BASE/"M/NUMERICAL_FAILURE.json",
    BASE/"M2/CONFIG.json", BASE/"M2/PARAMETERS.json", BASE/"M2/COEFFICIENTS.json",
    BASE/"M2/EVENT_INTERVALS.csv", BASE/"M2/BOUNDS.csv", BASE/"M2/MOMENTS.csv",
    BASE/"M2/STREAM_COUNTS.csv", BASE/"M2/EMPIRICAL_MOMENTS.csv",
    BASE/"M2/DECISION.json", BASE/"M2/AUDIT.json", BASE/"M2/MANIFEST.json",
    BASE/"T/PRIMARY.csv", BASE/"T/STREAM_LOSSES.csv", BASE/"T/TIMING_AND_PRIMITIVES.json",
    BASE/"T/DECISION.json", BASE/"T/COMPLETE.json", DOC/"MIXTURE_BOUND.md",
    Path(__file__),
]
COLORS = dict(NEG="#AB4B45", POS="#3A6D9C", TC="#287F70", BLOCK="#D79A36", RB="#287F70")

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def table(path):
    with Path(path).open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def textwrite(path,text):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(text.rstrip()+"\n")

def csvwrite(path,rows):
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def verify_manifest(path):
    for name,h in read(path)["files"].items():
        assert sha(ROOT/name)==h,name

def display(value):
    if value in [None,""]:
        return ""
    if isinstance(value,(bool,int)):
        return str(value)
    try:
        return f"{float(value):.8g}"
    except (ValueError,TypeError):
        return str(value).replace("|","\\|")

def markdown_table(rows,columns=None,labels=None):
    columns=list(rows[0]) if columns is None else columns
    labels=columns if labels is None else labels
    lines=["|"+"|".join(labels)+"|","|"+"|".join("---" for _ in columns)+"|"]
    for row in rows:
        lines.append("|"+"|".join(display(row.get(k,"")) for k in columns)+"|")
    return "\n".join(lines)

def derive_tables(events,trows):
    c=Decimal(str(read(BASE/"M2/CONFIG.json")["c"]))
    central={int(r["d"]):r for r in events if r["family"]=="NEG" and r["event"]=="central"}
    mixtures,comparisons=[],[]
    for d in [8,24,128,512]:
        source=central[d]
        for K in [4,10,100,512]:
            with localcontext() as ctx:
                ctx.prec=70
                tc=Decimal(source["simultaneous_information_lower"])
                penalty=c*Decimal(K).ln()/Decimal(d-1)
                kl=max(Decimal(0),tc-penalty)
                floor=-penalty
                absolute=Decimal(d)*Decimal(2).ln()-penalty
            assert abs(float(kl)-max(0.,float(tc)-float(c)*math.log(K)/(d-1)))<1e-14
            mixtures.append(dict(d=d,product_components=K,c=str(c),
                M2_TC_lower_decimal=str(tc),component_allowance_decimal=str(penalty),
                observation_joint_KL_lower_decimal=str(kl),
                observation_joint_KL_lower=float(kl),
                product_F_minus_F0_lower_decimal=str(floor),
                product_F_minus_F0_lower=float(floor),
                product_absolute_F_lower_decimal=str(absolute),
                confidence="M2 original 24-event simultaneous 95% CP coverage; no new statistical interval",
                scope="Any K-product mixture; not integrated Qtheta, latent posterior, or task risk"))
    for d in [24,128]:
        rb=next(r for r in trows if r["family"]=="NEG" and int(r["d"])==d and r["method"]=="count_RB_IID10")
        assert rb["precision_pass"]=="True"
        for K in [10,100]:
            mix=next(r for r in mixtures if r["d"]==d and r["product_components"]==K)
            with localcontext() as ctx:
                ctx.prec=70
                floor=Decimal(mix["product_F_minus_F0_lower_decimal"])
                upper=Decimal(rb["upper"])
                gap=floor-upper
            comparisons.append(dict(d=d,product_components=K,RB_particles=10,
                product_F_minus_F0_lower_decimal=str(floor),
                RB_F_minus_F0_mean=rb["mean"],RB_original_T_lower=rb["lower"],RB_original_T_upper=rb["upper"],
                uniform_product_minus_RB_gap_lower_decimal=str(gap),
                uniform_product_minus_RB_gap_lower=float(gap),
                positive_uniform_gap=bool(gap>0),
                confidence="Only original T 20-contrast simultaneous t integration interval; analytic product floor; TC cancels",
                interpretation="Derived inequality, not a new success-selection gate or same-wall-clock claim"))
    csvwrite(FIG/"MIXTURE_BOUNDS.csv",mixtures)
    csvwrite(FIG/"RB_vs_product_bound.csv",comparisons)
    return mixtures,comparisons

def save(fig,name):
    fig.savefig(FIG/f"{name}.png",dpi=190)
    fig.savefig(FIG/f"{name}.svg")
    plt.close(fig)
    path=FIG/f"{name}.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",
                    encoding="utf-8",newline="\n")

def plot_structure(params,bounds,events):
    d=np.asarray([8,24,128,512])
    selected=[next(r for r in bounds if int(r["d"])==n and int(r["K"])==100) for n in d]
    neg=np.array([float(r["NEG_Kcrit"]) for r in selected])
    pos=np.array([float(r["POS_Kcrit"]) for r in selected])
    tc=np.array([float(next(r for r in events if r["family"]=="NEG" and int(r["d"])==n and r["event"]=="central")["simultaneous_information_lower"]) for n in d])
    block=np.array([float(next(r for r in events if r["family"]=="NEG" and int(r["d"])==n and r["event"]=="equal_block_sign")["simultaneous_information_lower"]) for n in d])
    pair=np.array([float(r["pair_MI"]) for r in selected])
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.6))
    ax=axes[0]
    ax.plot(d,neg,"o-",lw=2,color=COLORS["NEG"],label="NEG: distributed constraint")
    ax.plot(d,pos,"s-",lw=2,color=COLORS["POS"],label="POS: both matched families")
    ax.axhline(100,color="#555555",ls="--",lw=1.2,label="K = 100")
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xticks(d,[str(n) for n in d]);ax.set_ylim(2,1700)
    ax.annotate("266.18",(128,neg[2]),xytext=(7,8),textcoords="offset points",color=COLORS["NEG"])
    ax.set_xlabel("Number of concepts d");ax.set_ylabel("Critical iid particle budget")
    ax.set_title("A. Same pair information, different local thresholds")
    ax.legend(loc="upper left",fontsize=9)
    ax=axes[1]
    ax.plot(d,pair,"o-",lw=2,color=COLORS["POS"],label="Per-pair MI (matched + / -)")
    ax.plot(d,tc,"s-",lw=2,color=COLORS["TC"],label="NEG full-joint TC: lower bound")
    ax.plot(d,block,"^-",lw=2,color=COLORS["BLOCK"],label="NEG two-block MI: lower bound")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xticks(d,[str(n) for n in d])
    ax.set_ylim(1e-7,.16);ax.set_xlabel("Number of concepts d");ax.set_ylabel("Information (nats)")
    ax.set_title("B. Weak pairs retain collective information")
    ax.legend(loc="lower left",fontsize=9)
    for ax in axes:ax.grid(alpha=.18,which="both")
    fig.suptitle("Dependence structure changes finite-MC geometry",fontsize=16,y=.98)
    fig.text(.5,.035,"Strict iid descent requires K > Kcrit. TC / block curves are M2 simultaneous 95% lower bounds, not exact information.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=[0,.09,1,.92])
    save(fig,"structure_and_information")

def bars(ax,rows,labels,colors,ylim):
    for j,(r,label,color) in enumerate(zip(rows,labels,colors)):
        mean,lo,hi=map(float,[r["mean"],r["lower"],r["upper"]])
        ok=r["precision_pass"]=="True"
        ax.bar(j,mean,width=.65,color=color if ok else "#E2E2E2",
               edgecolor=color if ok else "#777777",hatch=None if ok else "///",zorder=3)
        ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt="none",ecolor="#333333",capsize=3,lw=1,zorder=4)
        shift=(ylim[1]-ylim[0])*.025
        ax.text(j,hi+shift if mean>=0 else lo-shift,
                f"{mean:+.3f}" if abs(mean)>1 else f"{mean:+.4f}",
                ha="center",va="bottom" if mean>=0 else "top",fontsize=9)
    ax.axhline(0,color="#444444",lw=1);ax.set_xticks(range(len(rows)),labels)
    ax.set_ylim(*ylim);ax.grid(axis="y",alpha=.18,zorder=0)
    ax.set_ylabel("F(true theta) - F(independence), nats")

def plot_objectives(trows):
    get=lambda family,method:next(r for r in trows if r["family"]==family and int(r["d"])==128 and r["method"]==method)
    fig,axes=plt.subplots(1,3,figsize=(15,5.7),gridspec_kw={"width_ratios":[1.05,.9,1.45]})
    rows=[get("NEG",m) for m in ["original_IID10","original_IID100","original_ANTI100"]]
    bars(axes[0],rows,["IID\nK=10","IID\nK=100","ANTI\nK=100"],[COLORS["NEG"]]*3,(0,111))
    axes[0].set_title("A. NEG: pattern likelihood")
    rows=[get("NEG",m) for m in ["count_RB_IID1","count_RB_IID10"]]
    bars(axes[1],rows,["Count RB\nK=1","Count RB\nK=10"],[COLORS["RB"]]*2,(-.10,.24))
    axes[1].set_title("B. Same NEG Q: count RB\n(smaller vertical scale)")
    methods=["original_IID10","original_IID100","original_ANTI100","count_RB_IID1","count_RB_IID10"]
    bars(axes[2],[get("POS_RANK1",m) for m in methods],
         ["IID\n10","IID\n100","ANTI\n100","RB\n1","RB\n10"],
         [COLORS["POS"]]*3+[COLORS["RB"]]*2,(-.082,.315))
    axes[2].set_title("C. Separate POS_RANK1 control\n(pair-matched, not the same joint Q)")
    fig.suptitle("Same-K10 estimator change reverses the NEG ranking (d = 128)",fontsize=15.5,y=.99)
    fig.text(.5,.045,"Error bars: original T 20-contrast simultaneous t integration intervals. Hatched bars failed the predefined precision gate.",ha="center",fontsize=9,color="#555555")
    fig.text(.5,.015,"Different panel scales are explicit. Count RB uses more work per particle; this is not a same-wall-clock comparison. No model fitting.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=[0,.1,1,.91])
    save(fig,"objective_ordering")

def plot_mixtures(mixtures,comparisons,trows):
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.6),gridspec_kw={"width_ratios":[1,1.15]})
    colors=["#999999","#BC8B32",COLORS["NEG"],COLORS["TC"]]
    for d,color in zip([8,24,128,512],colors):
        rows=[r for r in mixtures if r["d"]==d]
        axes[0].plot([r["product_components"] for r in rows],
                     [r["observation_joint_KL_lower"] for r in rows],"o-",color=color,lw=2,label=f"d = {d}")
    axes[0].set_xscale("log");axes[0].set_xticks([4,10,100,512],["4","10","100","512"])
    axes[0].set_ylim(-.002,.064);axes[0].axvline(100,ls="--",lw=1,color="#555555")
    axes[0].set_xlabel("Number of product components K")
    axes[0].set_ylabel("Observed joint KL lower bound (nats)")
    axes[0].set_title("A. Uniform bound for any K-product mixture")
    axes[0].legend(fontsize=9,loc="upper right");axes[0].grid(alpha=.2)
    value=next(r["observation_joint_KL_lower"] for r in mixtures if r["d"]==128 and r["product_components"]==100)
    axes[0].annotate(f"{value:.8f}",(100,value),xytext=(-4,-20),textcoords="offset points",ha="right",color=COLORS["NEG"])
    ax=axes[1]
    rb=next(r for r in trows if r["family"]=="NEG" and int(r["d"])==128 and r["method"]=="count_RB_IID10")
    mean,lo,hi=map(float,[rb["mean"],rb["lower"],rb["upper"]])
    floors={}
    for K,y,color in [(10,2,COLORS["POS"]),(100,1,COLORS["NEG"])]:
        row=next(r for r in comparisons if r["d"]==128 and r["product_components"]==K)
        floor=float(row["product_F_minus_F0_lower_decimal"]);floors[K]=floor
        ax.scatter(floor,y,marker="|",s=200,color=color,zorder=4)
        ax.annotate("",xy=(.004,y),xytext=(floor,y),arrowprops=dict(arrowstyle="->",color=color,lw=2))
        ax.text(floor,y+.14,f"all objectives >= {floor:.6f}",ha="left",fontsize=9,color=color)
    ax.errorbar(mean,0,xerr=[[mean-lo],[hi-mean]],fmt="o",capsize=4,color=COLORS["RB"],lw=2)
    ax.text(hi,.15,f"RB upper = {hi:.6f}",ha="right",fontsize=9,color=COLORS["RB"])
    ax.axvspan(hi,floors[100],color=COLORS["RB"],alpha=.10)
    gap=next(r["uniform_product_minus_RB_gap_lower"] for r in comparisons if r["d"]==128 and r["product_components"]==100)
    ax.text((hi+floors[100])/2,.50,f"uniform gap\n>= {gap:.5f}",ha="center",fontsize=10,color=COLORS["RB"])
    ax.set_yticks([0,1,2],["Count RB, 10 particles","Any 100-product mixture","Any 10-product mixture"])
    ax.set_xlim(-.075,.008);ax.set_ylim(-.35,2.45);ax.axvline(0,color="#777777",lw=.8)
    ax.set_xlabel("Population objective minus independence (nats)")
    ax.set_title("B. d = 128: RB beats the uniform product bound")
    ax.grid(axis="x",alpha=.17)
    fig.suptitle("An observed-distribution bound, separate from latent posterior KL",fontsize=15.5,y=.98)
    fig.text(.5,.045,"A uses the original M2 24-event CP lower bounds. B uses an analytic product floor and the original T 20-contrast interval; TC cancels.",ha="center",fontsize=9,color="#555555")
    fig.text(.5,.015,"RB components are not products. Product-component count and RB particle count are different resources; integrated Qtheta can still be correct.",ha="center",fontsize=9,color="#555555")
    fig.tight_layout(rect=[0,.11,1,.91])
    save(fig,"observed_mixture_bounds")

def full_tables():
    names=["M2/BOUNDS.csv","M2/MOMENTS.csv","M2/STREAM_COUNTS.csv","M2/EMPIRICAL_MOMENTS.csv",
           "M2/EVENT_INTERVALS.csv","T/PRIMARY.csv","T/STREAM_LOSSES.csv"]
    out=["# M2/T全部CSV可读表","","所有原CSV列及行均保留；浮点显示8有效位，原CSV保留冻结精度。无筛格。"]
    for name in names:
        rows=table(BASE/name)
        out += ["",f"## {name}",f"",f"共{len(rows)}行；[原CSV](../{name})。","",markdown_table(rows)]
    out += ["","## M2 Hermite积分系数",""]
    coefficients=read(BASE/"M2/COEFFICIENTS.json")["rows"]
    out.append(markdown_table(coefficients,["n","coefficient","split_value","partition_difference",
        "coefficient_truncation_bound","propagated_coefficient_error_estimate","passed"]))
    out += ["","## T时间与计算primitive","",
            "按32个family/d/stream完整保留全部标量列。时间是共享bank程序计时，不是各方法独立benchmark；计数直方图保存在[原JSON](../T/TIMING_AND_PRIMITIVES.json)。","",
            markdown_table([{k:v for k,v in row.items() if not isinstance(v,(list,dict))}
                            for row in read(BASE/"T/TIMING_AND_PRIMITIVES.json")])]
    textwrite(FIG/"ALL_M2_T_TABLES.md","\n".join(out))

def results_doc(params,bounds,events,trows,mixtures,comparisons):
    old=read(BASE/"M/PARAMETERS.json")["rows"]
    parameter_table=[]
    for p in params:
        b=next(r for r in bounds if int(r["d"])==p["d"] and int(r["K"])==100)
        parameter_table.append(dict(d=p["d"],r=p["spin_r"],rank1_v=p["rank1_variance"],
            pair_MI=p["pair_MI"],NEG_Kcrit=b["NEG_Kcrit"],POS_Kcrit=b["POS_Kcrit"],
            r_estimated_error_plus_tail=p["r_total_estimated_error_plus_analytic_tail"]))
    primary=[]
    for r in trows:
        primary.append(dict(family=r["family"],d=r["d"],method=r["method"],
            mean=r["mean"],lower=r["lower"],upper=r["upper"],SE=r["SE"],
            numerical_pass=r["precision_pass"],
            qualified_true_better=r["true_theta_better_than_independent"],
            qualified_true_worse=r["true_theta_worse_than_independent"]))
    text=f"""# 完整结果与确定性解析后处理

全部M2/T数值保留。图表不增加观测、模型、拟合、MC格点或成功选择门。M原数值失败保持；M2是先另行冻结的新积分算法，随后才执行原定模拟。以下区分观察联合误差、有限MC人口目标和latent后验，不把它们互相替代。

## 1. M原积分门失败与M2参数

M原d24的GH160/240差超过1e−9，其余三维度通过；原包0观测，没有追认原通过。原四行如下：

{markdown_table(old,["d","spin_r_160","spin_r_240","quadrature_difference","numerical_pass"])}

M2使用固定奇Hermite阶1..13和quad；误差列是quad估计误差加解析尾界，不是严格认证的全部浮点误差。rank1方差只用来匹配观察两两相关，POS_RANK1与POS_FULL不保证完整联合相同。

{markdown_table(parameter_table)}

![谱门槛与集体信息](../../../artifacts/constraint_boundary_v2/figures/structure_and_information.png)

Kcrit是实数临界值，严格K>Kcrit才有负曲率。这里的正负门槛比较包含同latent对角的POS_FULL；T的正对照则为另外的POS_RANK1。不能把T称为仅改变相关符号。

## 2. M2全部24项粗事件区间

每cell262144观测，12cells共3145728；八个独立流，全部保留。区间是原24项Bonferroni同时95% Clopper–Pearson区间。central的DPI下界针对完整联合TC，equal-block-sign的下界针对两个完整半块的互信息；不是每对概念或单查询收益。

{markdown_table(events,["family","d","event","estimate","lower","upper","independent_baseline","simultaneous_information_lower"])}

NEG/d128的TC与半块MI下界分别为.05798455790与.01896756484。所有正族、维数和事件也列出，不以这些下界替代完整KL的精确值。

## 3. T全部20项有限MC目标比较

每项为F(theta_true)−F0，F0=d log2。负值表示该估计器的期望有限MC目标更偏好指定真参数；不是模型训练结果或全局最优证明。区间沿用原20项Bonferroni同时t积分口径、df7；它依赖流均值的近似t推断，不是自然数据确认或精确有限样本概率定理。

{markdown_table(primary)}

**四个RB1格都没有通过原精度门，完整保留。** K1存在局部下降方向不能推出该真参数点已胜过独立点。其正均值也不能代替未通过的资格判定。

![同Q估计器的目标排序](../../../artifacts/constraint_boundary_v2/figures/objective_ordering.png)

d128、同K10，NEG original的差为+99.032829，count-RB10为−.062556；IID100仍为+78.878107，ANTI100仍为+78.767003。改变的是似然估计器，integrated观察Q不变。RB每粒子动态规划更贵，不声称同wall-clock效率；同粒子POS_RANK1的RB和original表达在数值核验中相同。

## 4. 任意有限product mixture的观察KL下界

这里使用新解析结论 KL(P_NEG||Q_product,K) >= max(0,TC(P_NEG)−c logK/(d−1))，安全系数为latent Gaussian的c/(d−1)，没有用更小的观察相关r替代它。下表仅把M2原完整精度TC下界代入；仍继承原M2的同时覆盖事件，没有重新定义置信门。

{markdown_table(mixtures,["d","product_components","M2_TC_lower_decimal","observation_joint_KL_lower","product_F_minus_F0_lower"])}

d128/K100的观察联合KL下界约.02534949359 nats，**不是**latent SIR的25.34325 nats或d512的117.29766 nats。该全局观察界对任意K个product components及权重成立；integrated Qtheta可为无限混合并正确，独立theta也未被证明是全局最优。

![观察有限混合界与RB目标差](../../../artifacts/constraint_boundary_v2/figures/observed_mixture_bounds.png)

## 5. RB相对任意有限product目标的统一差

对于当前观察独立的prior-bank目标，任意theta的F_K−F0 >= −c logK/(d−1)。这里TC恰好消去，右侧是解析量。因此下表只使用原T的20项t积分区间，不需要将M2和T两套95%区间错误拼成新的95%。

{markdown_table(comparisons,["d","product_components","RB_particles","product_F_minus_F0_lower_decimal","RB_original_T_upper","uniform_product_minus_RB_gap_lower","positive_uniform_gap"])}

在d128/K100，解析product下界约−.03263506431，RB10原上界−.05685684424，相减约.02422177993；同K10差更大。d24两行没有正的统一差，也完整保留。positive_uniform_gap只是解析差是否为正的描述，不是新增实验成功门。

这个统一比较补充了T的pointwise见证，但没有证明integrated Qtheta错误，也没有认证独立globopt。RB每粒子是非product component，内部隐含置换平均；10个RB粒子不是10个product components，不能把该结果说成违反了有限混合界。

## 6. 读取入口与边界

[全部M2/T CSV可读表](../../../artifacts/constraint_boundary_v2/figures/ALL_M2_T_TABLES.md)包含192条解析模态、96条模拟流、12条样本矩诊断、16条latent/表示界、24事件、20主差、160积分流行及32计时行。原CSV和全部失败状态保持。

[观察mixture后处理](../../../artifacts/constraint_boundary_v2/figures/MIXTURE_BOUNDS.csv) · [RB统一目标差后处理](../../../artifacts/constraint_boundary_v2/figures/RB_vs_product_bound.csv) · [图表来源](../../../artifacts/constraint_boundary_v2/figures/SOURCE_HASHES.json)。

解析前提与证明见[MIXTURE_BOUND](MIXTURE_BOUND.md)、[估计器控制](ESTIMATOR_CONTROL.md)、[理论候选](THEORY_CANDIDATE.md)。这些结果不认证自然SOTA、通用非交换CBM方法、所有推断的指数困难、优化器收敛或原创性。
"""
    textwrite(DOC/"RESULTS.md",text)

def run():
    assert not FIG.exists(),"Refuse to overwrite existing figure package"
    assert not (DOC/"RESULTS.md").exists(),"Refuse to overwrite RESULTS"
    verify_manifest(BASE/"M2/MANIFEST.json")
    verify_manifest(BASE/"T/COMPLETE.json")
    FIG.mkdir(parents=True)
    write(FIG/"SOURCE_HASHES.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in SOURCES},
        operation="deterministic plotting and analytic postprocessing only",new_samples=0,new_fits=0,
        M2_confidence="original 24-event simultaneous CP",T_confidence="original 20-contrast simultaneous t integration",
        no_new_success_selection_gates=True))
    params=read(BASE/"M2/PARAMETERS.json")["rows"]
    bounds=table(BASE/"M2/BOUNDS.csv")
    events=table(BASE/"M2/EVENT_INTERVALS.csv")
    trows=table(BASE/"T/PRIMARY.csv")
    assert len(params)==4 and len(events)==24 and len(trows)==20
    assert sum(r["precision_pass"]=="False" for r in trows)==4
    mixtures,comparisons=derive_tables(events,trows)
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none",
        "axes.spines.top":False,"axes.spines.right":False})
    plot_structure(params,bounds,events)
    plot_objectives(trows)
    plot_mixtures(mixtures,comparisons,trows)
    full_tables()
    results_doc(params,bounds,events,trows,mixtures,comparisons)
    paths=[*sorted(p for p in FIG.rglob("*") if p.is_file()),DOC/"RESULTS.md"]
    write(FIG/"MANIFEST.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),
        files={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        SVG_trailing_whitespace_removed_before_seal=True))
    check()

def check():
    verify_manifest(FIG/"SOURCE_HASHES.json")
    verify_manifest(FIG/"MANIFEST.json")
    assert len(table(FIG/"MIXTURE_BOUNDS.csv"))==16
    assert len(table(FIG/"RB_vs_product_bound.csv"))==4
    for path in FIG.glob("*.svg"):
        assert all(s.rstrip()==s for s in path.read_text(encoding="utf-8").splitlines())
    row=next(r for r in table(FIG/"RB_vs_product_bound.csv") if r["d"]=="128" and r["product_components"]=="100")
    assert abs(float(row["uniform_product_minus_RB_gap_lower"])-.0242217799267675)<1e-12
    print(json.dumps(dict(pass_v2_figures=True,figures=3,mixture_rows=16,uniform_comparison_rows=4,
        failed_T_RB1_rows_preserved=4,new_samples=0,new_fits=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    check() if args.check else run()

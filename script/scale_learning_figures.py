"""Post-execution scientific figures/tables only; no fitting or new random draws."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/scale_learning_v2"
FIG=BASE/"figures"
DOC=ROOT/"docs/08_research_研究工作/有限粒子尺度学习_v2_20260930"
CSV_NAMES=["REFERENCE_CURVES","POPULATION_BANDS","POPULATION_FITS","FINITE_FITS","FINITE_RISK_INTERVALS"]
PROFILES=["null","contrast"]
OBJECTIVES=["J","P","G10","G100","G1000","S10","S100","S1000"]
LABELS=["Joint","Pair","G10","G100","G1000","S10","S100","S1000"]
COLORS={"G":"#2878A5","S":"#D18B28","J":"#42464A","P":"#258B77"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,"axes.labelsize":11,
                     "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none",
                     "svg.hashsalt":"scale-learning-v2-20260930","savefig.facecolor":"white"})
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def table(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def flag(value):return value is True or value=="True"
def verify(ok,msg):
    if not ok:raise AssertionError(msg)
def write_json(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def write_text(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as f:f.write(value.rstrip()+"\n")
def fmt(x):
    if x is None or x=="":return ""
    if isinstance(x,bool):return "通过" if x else "未通过"
    if isinstance(x,list):return ", ".join(fmt(v) for v in x)
    if isinstance(x,dict):return json.dumps(x,ensure_ascii=False)
    try:return f"{float(x):.9g}"
    except (ValueError,TypeError):return str(x).replace("|","\\|")
def markdown(rows,columns=None):
    if not rows:return "无记录。"
    if columns is None:
        columns=[]
        for row in rows:
            for key in row:
                if key not in columns:columns.append(key)
    lines=["|"+"|".join(columns)+"|","|"+"|".join("---" for _ in columns)+"|"]
    for row in rows:lines.append("|"+"|".join(fmt(row.get(k,"")) for k in columns)+"|")
    return "\n".join(lines)
def save(fig,name):
    fig.savefig(FIG/(name+".png"),dpi=180,bbox_inches="tight")
    fig.savefig(FIG/(name+".svg"),bbox_inches="tight",metadata={"Date":None})
    svg=FIG/(name+".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
    plt.close(fig)
def color(objective):
    return COLORS[objective[0]] if objective[0] in COLORS else "#444444"

def validate_inputs(data):
    expected=dict(REFERENCE_CURVES=62,POPULATION_BANDS=360,POPULATION_FITS=208,FINITE_FITS=512,FINITE_RISK_INTERVALS=20)
    for name,count in expected.items():verify(len(data[name])==count,f"{name}: row count")
    ref={(r["profile"],float(r["t"])):r for r in data["REFERENCE_CURVES"]}
    ff=data["FINITE_FITS"]
    keys={(r["profile"],int(r["stream"]),int(r["N"]),r["objective"]) for r in ff}
    verify(len(keys)==512,"duplicate finite fits")
    maxrisk=0.
    for row in ff:
        r=ref[(row["profile"],float(row["t"]))]
        wanted=float(r["deployed_S_KL"] if row["objective"].startswith("S") else r["true_Q_KL"])
        maxrisk=max(maxrisk,abs(wanted-float(row["deployed_KL"])),abs(float(r["true_Q_KL"])-float(row["true_Q_KL"])))
        verify(float(row["grid_minimum_gap"])==0,"grid minimum gap")
    for row in data["POPULATION_FITS"]:
        r=ref[(row["profile"],float(row["t"]))]
        wanted=float(r["deployed_S_KL"] if row["objective"].startswith("S") else r["true_Q_KL"])
        maxrisk=max(maxrisk,abs(wanted-float(row["deployed_KL"])))
    maxstat=0.;crit=float(student_t.ppf(1-.05/(2*20),15))
    lookup={(r["profile"],int(r["N"]),r["objective"],int(r["stream"])):float(r["deployed_KL"]) for r in ff}
    for row in data["FINITE_RISK_INTERVALS"]:
        left,right=row["comparison"].split("-")
        vals=[lookup[(row["profile"],int(row["N"]),left,s)]-lookup[(row["profile"],int(row["N"]),right,s)] for s in range(16)]
        mean=math.fsum(vals)/16;se=math.sqrt(math.fsum((v-mean)**2 for v in vals)/15/16)
        for key,value in zip(["mean","SE","lower","upper"],[mean,se,mean-crit*se,mean+crit*se]):
            maxstat=max(maxstat,abs(float(row[key])-value))
        verify(flag(row["precision_pass"])==(se<=.001),"precision flag")
    verify(maxrisk<2e-12 and maxstat<2e-12,"table cross-check")
    return dict(source_row_counts=expected,deployment_lookup_max_error=maxrisk,independent_interval_replay_max_error=maxstat)

def population_curves(data):
    fig,axes=plt.subplots(2,3,figsize=(14.2,7.8),sharex=True)
    aggregate={(r["profile"],r["objective"]):float(r["t"]) for r in data["POPULATION_FITS"] if r["unit"]=="aggregate"}
    for pi,profile in enumerate(PROFILES):
        for ki,k in enumerate([10,100,1000]):
            ax=axes[pi,ki]
            for method in ["G","S"]:
                rows=[r for r in data["POPULATION_BANDS"] if r["profile"]==profile and r["method"]==method and int(r["K"])==k]
                rows=rows+[dict(t="1",mean="0",lower="0",upper="0",precision_pass=True)]
                rows=sorted(rows,key=lambda r:float(r["t"]))
                x=np.array([float(r["t"]) for r in rows]);y=np.array([float(r["mean"]) for r in rows])
                lo=np.array([float(r["lower"]) for r in rows]);hi=np.array([float(r["upper"]) for r in rows])
                co=COLORS[method]
                ax.fill_between(x,lo,hi,color=co,alpha=.17,lw=0)
                ax.plot(x,y,color=co,lw=1.7,label=method)
                failed=np.array([not flag(r["precision_pass"]) for r in rows])
                ax.scatter(x[failed],y[failed],marker="x",s=25,color="#B13D48",zorder=5)
                fit=aggregate[(profile,method+str(k))];ix=int(np.argmin(abs(x-fit)))
                ax.scatter([fit],[y[ix]],s=45,marker="D",color=co,edgecolors="white",linewidths=.8,zorder=6)
            ax.axhline(0,color="#666666",lw=.8)
            ax.axvline(1,color="#333333",ls="--",lw=.9)
            ax.axvline(.5,color="#888888",ls=":",lw=.8)
            ax.set_title(f"{profile}  |  K = {k:,}")
            ax.grid(alpha=.16);ax.set_xlim(0,1.5)
            if pi==1:ax.set_xlabel("Candidate amplitude t")
            if ki==0:ax.set_ylabel("Population log-score difference\nrelative to true t = 1 (nats)")
    handles=[Line2D([0],[0],color=COLORS["G"],label="G: correct group model"),
             Line2D([0],[0],color=COLORS["S"],label="S: full-count symmetrization"),
             Line2D([0],[0],marker="D",color="none",markerfacecolor="#666666",label="Aggregate noisy grid argmin"),
             Line2D([0],[0],marker="x",color="#B13D48",ls="none",label="SE gate failed")]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle("Finite-particle population objectives across the complete frozen grid",y=.99,fontsize=15)
    fig.text(.5,.085,"Shading: 360-contrast simultaneous t-approximation bands; 16 streams. No continuous or exact population optimum is claimed.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.95]);save(fig,"population_objectives")

def fitted_scales(data):
    fig,axes=plt.subplots(2,2,figsize=(13.1,8),sharex=True,sharey=True)
    agg={(r["profile"],r["objective"]):float(r["t"]) for r in data["POPULATION_FITS"] if r["unit"]=="aggregate"}
    for pi,profile in enumerate(PROFILES):
        for ni,n in enumerate([4096,65536]):
            ax=axes[pi,ni]
            for j,obj in enumerate(OBJECTIVES):
                records=sorted([r for r in data["FINITE_FITS"] if r["profile"]==profile and int(r["N"])==n and r["objective"]==obj],
                               key=lambda r:int(r["stream"]))
                values=np.array([float(r["t"]) for r in records])
                ax.scatter(j+np.linspace(-.17,.17,16),values,s=22,color=color(obj),alpha=.7,edgecolors="white",linewidths=.35)
                ax.scatter(j,float(values.mean()),marker="_",s=260,color=color(obj),linewidths=2.2,zorder=5)
                ax.scatter(j,agg[(profile,obj)],marker="D",s=52,facecolors="none",edgecolors="#222222",linewidths=1.1,zorder=6)
            ax.axhline(1,color="#333333",ls="--",lw=1)
            ax.set_ylim(-.055,1.555);ax.set_yticks(np.arange(0,1.51,.25))
            ax.set_title(f"{profile}  |  N = {n:,}")
            ax.set_xticks(range(8),LABELS,rotation=35,ha="right")
            ax.grid(axis="y",alpha=.18)
            if ni==0:ax.set_ylabel("Fitted amplitude t")
    handles=[Line2D([0],[0],marker="o",ls="none",color="#555555",label="All 16 finite-data fits"),
             Line2D([0],[0],marker="_",markersize=14,ls="none",color="#555555",label="Finite-fit mean (descriptive)"),
             Line2D([0],[0],marker="D",markerfacecolor="none",ls="none",color="#222222",label="Aggregate grid argmin (J/Pair deterministic)"),
             Line2D([0],[0],color="#333333",ls="--",label="True amplitude t = 1")]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle("Learned scales: all finite-data replicates and aggregate objective curves",y=.99,fontsize=15)
    fig.tight_layout(rect=[0,.09,1,.955]);save(fig,"fitted_scales")

def deployment_risks(data):
    fig,axes=plt.subplots(2,2,figsize=(13.1,8),sharex=True,sharey=True)
    for pi,profile in enumerate(PROFILES):
        for ni,n in enumerate([4096,65536]):
            ax=axes[pi,ni]
            for j,obj in enumerate(OBJECTIVES):
                records=sorted([r for r in data["FINITE_FITS"] if r["profile"]==profile and int(r["N"])==n and r["objective"]==obj],
                               key=lambda r:int(r["stream"]))
                dep=np.array([float(r["deployed_KL"]) for r in records]);true=np.array([float(r["true_Q_KL"]) for r in records])
                jitter=np.linspace(-.17,.17,16)
                ax.scatter(j+jitter,dep,s=22,color=color(obj),alpha=.72,edgecolors="white",linewidths=.35)
                ax.scatter(j,float(dep.mean()),marker="_",s=260,color=color(obj),linewidths=2.1,zorder=6)
                if obj.startswith("S"):
                    ax.scatter(j+jitter,true,s=20,facecolors="none",edgecolors="#34566F",linewidths=.65,alpha=.8,zorder=4)
            ax.axhline(.001,color="#777777",ls=":",lw=.9)
            ax.axhline(0,color="#666666",lw=.7)
            ax.set_yscale("symlog",linthresh=1e-5,linscale=.75,base=10)
            ax.set_title(f"{profile}  |  N = {n:,}")
            ax.set_xticks(range(8),LABELS,rotation=35,ha="right")
            ax.grid(axis="y",alpha=.18)
            if ni==0:ax.set_ylabel("High-precision joint KL (nats)\nsymlog axis; linear near zero")
    handles=[Line2D([0],[0],marker="o",ls="none",color="#555555",label="Actual deployment KL (all 16 fits)"),
             Line2D([0],[0],marker="o",markerfacecolor="none",ls="none",color="#34566F",label="S fits decoded with correct Q: diagnostic only"),
             Line2D([0],[0],marker="_",markersize=14,ls="none",color="#555555",label="Deployment mean (descriptive)"),
             Line2D([0],[0],color="#777777",ls=":",label="0.001 nats reference scale")]
    fig.legend(handles=handles,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(.5,.005))
    fig.suptitle("Training and deployment are separate contracts",y=.99,fontsize=15)
    fig.text(.5,.085,"J, Pair and G deploy Q(t). S deploys S(Q(t)); open S markers must not replace the deployment risks.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.955]);save(fig,"deployment_risks")

def finite_intervals(data):
    fig,axes=plt.subplots(2,2,figsize=(13,7.5),sharex=True,sharey=True)
    order=["G10-J","G100-J","G1000-J","S10-G10","P-J"]
    for pi,profile in enumerate(PROFILES):
        for ni,n in enumerate([4096,65536]):
            ax=axes[pi,ni]
            ax.axvspan(-.001,.001,color="#DDDDDD",alpha=.55)
            ax.axvline(0,color="#666666",lw=.8)
            for j,name in enumerate(order):
                r=next(r for r in data["FINITE_RISK_INTERVALS"] if r["profile"]==profile and int(r["N"])==n and r["comparison"]==name)
                value=float(r["mean"]);lo=float(r["lower"]);hi=float(r["upper"]);ok=flag(r["precision_pass"])
                co=color(name.split("-")[0])
                ax.errorbar(value,j,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt="o",color=co,
                            markerfacecolor=co if ok else "white",capsize=3,markersize=5,lw=1.5)
                if not ok:ax.text(hi,j,"  SE fail",color="#B13D48",va="center",fontsize=8)
            ax.set_title(f"{profile}  |  N = {n:,}")
            ax.set_yticks(range(5),order);ax.set_ylim(4.6,-.6)
            ax.set_xscale("symlog",linthresh=1e-4,linscale=.8,base=10)
            ax.grid(axis="x",alpha=.18)
            if pi==1:ax.set_xlabel("Paired deployment-KL difference (nats; symlog)")
    fig.suptitle("All 20 finite-data risk comparisons",y=.99,fontsize=15)
    fig.text(.5,.04,"20-contrast simultaneous t-approximation intervals, 16 paired streams. Shaded region: [-0.001, +0.001]. Open marker = precision failure.",
             ha="center",fontsize=9)
    fig.text(.5,.012,"Zero-width Pair-J intervals arise from identical grid choices; they do not certify rigorous population equivalence.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.075,1,.95]);save(fig,"finite_risk_intervals")

def summaries(data):
    rows=[]
    for profile in PROFILES:
        for n in [4096,65536]:
            for obj in OBJECTIVES:
                rr=[r for r in data["FINITE_FITS"] if r["profile"]==profile and int(r["N"])==n and r["objective"]==obj]
                ts=np.array([float(r["t"]) for r in rr]);dep=np.array([float(r["deployed_KL"]) for r in rr]);q=np.array([float(r["true_Q_KL"]) for r in rr])
                rows.append(dict(profile=profile,N=n,objective=obj,fits=16,t_min=float(ts.min()),t_mean=float(ts.mean()),t_max=float(ts.max()),
                                 mean_true_Q_KL=float(q.mean()),mean_deployed_KL=float(dep.mean())))
    return rows

def readable(data,decision,sets,summary,audit):
    tables=FIG/"tables";tables.mkdir()
    for name in CSV_NAMES:
        rows=data[name]
        intro=f"# {name}：全部{len(rows)}行\n\n来源：../../{name}.csv。无删除或重拟合；浮点以9位有效数字显示，精确值保留在原CSV。\n\n"
        if name=="FINITE_RISK_INTERVALS":
            intro+="大N的Pair−Joint区间[0,0]来自相同离散网格选择及SE=0的退化，不是严格总体±.001等效证书。原t主门保持；新增保守敏感性见[PAIR_SENSITIVITY.json](../../PAIR_SENSITIVITY.json)。\n\n"
        write_text(tables/(name+".md"),intro+markdown(rows))
    write_text(tables/"SELECTION_SETS.md","# 可能最优集合：全部12臂\n\n集合是360项t近似带的派生，不是连续参数最优集合。true_Q下界对S只属正确Q参数诊断。\n\n"+markdown(sets))
    write_text(tables/"FINITE_SUMMARY.md","# 有限数据拟合的描述汇总\n\n所有16拟合均保留。均值和范围不是新置信区间。\n\n"+markdown(summary))
    failed_pop=[r for r in data["POPULATION_BANDS"] if not flag(r["precision_pass"])]
    failed_finite=[r for r in data["FINITE_RISK_INTERVALS"] if not flag(r["precision_pass"])]
    failed_primary=[r for r in decision["primary"] if not flag(r["pass_primary"])]
    large_choices={(r["profile"],int(r["stream"]),r["objective"]):int(r["selected_index"])
                   for r in data["FINITE_FITS"] if int(r["N"])==65536}
    large_pair_matches=sum(large_choices[(p,s,"J")]==large_choices[(p,s,"P")] for p in PROFILES for s in range(16))
    pair_sensitivity=read(BASE/"PAIR_SENSITIVITY.json")
    conservative_bounds="、".join(f"{r['profile']} {r['mean_risk_difference_absolute_bound']:.7g}" for r in pair_sensitivity["rows"])
    main=["# 有限粒子尺度学习：完整结果与图表","",
          "本文件只读取已封存结果作图和整理，不增加样本、拟合、参考积分或统计家族。完整31点网格保留；聚合bank曲线的argmin只是含积分噪声的描述端点，不能称精确总体或连续空间全局最优。","",
          "固定d=16、c=.9，null基础尺度为(4,4)，contrast为(6,2)；唯一自由参数t同时乘两个尺度，真值为1，候选网格0,.05,...,1.5。G/S使用K10/100/1000，有限数据N4096/65536，各16个配对流。Joint和Pair是该指定教师族下的高精度计算对照。","",
          f"512个有限数据网格拟合、192个bank-only网格端点及16个聚合/确定目标端点。总体360项精度通过{decision['population_precision_pass']}/360，有限数据20项通过{decision['finite_precision_pass']}/20；预定主门通过{len(decision['primary'])-len(failed_primary)}/{len(decision['primary'])}。","",
          "## 全部主门","",markdown(decision["primary"]),"",
          "主总体见证是360项家族中预先指定的两个G10、t=.5相对1比较；没有另立12项置信家族。有限数据主比较使用各方法实际部署joint KL：J/Pair/G为Q(t_hat)，S为S(Q(t_hat))。","",
          "## 总体目标曲线与积分噪声","",
          "![总体目标曲线](../../../artifacts/scale_learning_v2/figures/population_objectives.png)","",
          "阴影为360项同时t近似带；红叉保留未过SE门的点；菱形为聚合噪声曲线网格argmin。1为真参数，.5为固定比较点。带的统计近似和高精度参考资格分开解释。","",
          "## 学得尺度与部署风险","",
          "![有限数据拟合尺度](../../../artifacts/scale_learning_v2/figures/fitted_scales.png)","",
          "每格16个有限数据拟合全部显示；短横线为描述均值，空心菱形为聚合目标端点（J/Pair为确定高精度目标，G/S含bank噪声），均不是置信区间。","",
          "![真实模型与部署KL](../../../artifacts/scale_learning_v2/figures/deployment_risks.png)","",
          "实心点是实际部署模型的高精度joint KL；S的空心点只是将其学得尺度代回正确Q的诊断。二者不能互换。风险图使用symlog轴以同时展示近零和较大损失；没有对已保存风险值剪零。","",
          "## 全部有限数据配对区间","",
          "![20项配对区间](../../../artifacts/scale_learning_v2/figures/finite_risk_intervals.png)","",
          markdown(data["FINITE_RISK_INTERVALS"]),"",
          "20项区间基于16个配对流、df15及原Bonferroni家族；灰色范围为±.001。Pair等效只指原门下的风险差，不是参数一致、效率相同或普遍支配。","",
          f"N65536下Joint与Pair在{large_pair_matches}/32个训练集选中相同的离散网格点。因此本次大N的Pair−Joint风险差为0，不能泛化成连续参数估计或统计效率完全相同。","",
          f"原[0,0]区间由离散同选点及SE=0造成退化，不能作为严格总体±.001等效证书。事后保守敏感性给出的绝对均值风险差上界为{conservative_bounds} nats，均未认证±.001；它不改变原t主门，详见[PAIR_SENSITIVITY](PAIR_SENSITIVITY.md)。","",
          "## 失败完整账本","",
          f"主门未通过{len(failed_primary)}项；总体精度未通过{len(failed_pop)}项；有限数据精度未通过{len(failed_finite)}项。","",
          "### 未通过主门","",markdown(failed_primary),"",
          "### 未通过总体精度","",markdown(failed_pop),"",
          "### 未通过有限数据精度","",markdown(failed_finite),"",
          "## 可能最优集合与全部可读表","",
          "集合由固定truth anchor带构造，覆盖逻辑条件于这些区间同时覆盖目标；当前是t近似，不是严格有限样本覆盖。集合内最小true_Q_KL对S仅为正确Q的参数诊断，不能冒充S部署风险。","",
          markdown(sets,["profile","method","K","grid","all_precision_pass","truth_excluded","min_true_Q_KL","min_true_Q_KL_minus_qualification_allowance"]),"",
          "以下表不删除任何行；全部原CSV保留精确浮点值。",""]
    for name in CSV_NAMES+["SELECTION_SETS","FINITE_SUMMARY"]:
        main.append(f"- [{name}完整可读表](../../../artifacts/scale_learning_v2/figures/tables/{name}.md)")
    main+=["","## 只读交付校验","",
           f"图表生成器核对512条拟合的风险映射、全部聚合端点，并从逐流已保存部署风险重算20项区间；最大区间重算差为{audit['independent_interval_replay_max_error']:.3g}。该计算不产生新科学样本。图源文件和输出哈希记录于figures/FIGURE_MANIFEST.json；SVG去除行尾空格后再登记哈希。","",
           "本页不代替总体REPORT，不把已知教师、固定分组/比例和高精度参考下的结果扩展为一般CBM或自然任务保证。"]
    write_text(DOC/"RESULTS.md","\n".join(main))

def build():
    verify(read(BASE/"STATE.json")["status"]=="complete","Science STATE is not complete; no figures written")
    verify(not FIG.exists(),"Figure output already exists; no silent overwrite")
    verify(not (DOC/"RESULTS.md").exists(),"RESULTS already exists; no silent overwrite")
    for name in ["FREEZE.json","RESULTS_FREEZE.json"]:
        for source,expected in read(BASE/name)["files"].items():verify(sha(ROOT/source)==expected,"source hash: "+source)
    data={n:table(BASE/(n+".csv")) for n in CSV_NAMES}
    decision=read(BASE/"DECISION.json");sets=read(BASE/"SELECTION_SETS.json")
    audit=validate_inputs(data);verify(len(sets)==12,"selection sets count")
    FIG.mkdir(parents=True)
    population_curves(data);fitted_scales(data);deployment_risks(data);finite_intervals(data)
    summary=summaries(data);readable(data,decision,sets,summary,audit)
    write_json(FIG/"FIGURE_AUDIT.json",dict(passed=True,**audit,new_samples=0,new_fits=0,new_reference_integrals=0,
               no_new_statistical_intervals=True,all_primary_failures_shown=True,all_precision_failures_shown=True))
    sources=[Path(__file__),BASE/"CONFIG.json",BASE/"FREEZE.json",BASE/"RESULTS_FREEZE.json",BASE/"STATE.json",
             BASE/"DECISION.json",BASE/"SELECTION_SETS.json",BASE/"PAIR_SENSITIVITY.json"]+[BASE/(n+".csv") for n in CSV_NAMES]
    outputs=[p for p in FIG.rglob("*") if p.is_file()]+[DOC/"RESULTS.md"]
    write_json(FIG/"FIGURE_MANIFEST.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),
        kind="post_execution_figures_and_readable_tables",source_files={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        output_files={p.relative_to(ROOT).as_posix():sha(p) for p in outputs},no_new_scientific_computation=True))
    print(json.dumps(dict(passed=True,figures=4,complete_tables=7,manifest_sha256=sha(FIG/"FIGURE_MANIFEST.json")),ensure_ascii=False))
def check():
    receipt=read(FIG/"FIGURE_MANIFEST.json")
    for category in ["source_files","output_files"]:
        for name,expected in receipt[category].items():verify(sha(ROOT/name)==expected,name)
    print(json.dumps(dict(passed=True,sources=len(receipt["source_files"]),outputs=len(receipt["output_files"]))))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["build","check"]);args=parser.parse_args()
    build() if args.phase=="build" else check()


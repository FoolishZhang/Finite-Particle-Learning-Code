"""Post-execution A/B/C figures and complete readable tables. No scientific fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/structure_learning_v2"
FIG=BASE/"figures"
DOC=ROOT/"docs/08_research_研究工作/结构先验释放与分组学习_v2_20260930"
FILES={"A":["FINITE_FITS","FINITE_RISK_INTERVALS","POPULATION_FITS","POPULATION_CURVES"],
       "B":["FINITE_FITS","FINITE_RISK_INTERVALS","POPULATION_FITS","POPULATION_FIXED_COMPARISONS","REFERENCE_CURVES","GRID"],
       "C":["FITS","RISK_INTERVALS","SUMMARY"]}
COLORS={"floor":"#44484D","A":"#CD8A2D","B":"#287AA7","pair":"#258875","G":"#B44F57","oracle":"#CF4892"}
CMETHODS=["ORACLE_J","ORACLE_P","GLOBAL_P","ROW_P","GLOBAL_G10","GLOBAL_G100"]
CLABELS=["Oracle J","Oracle P","Global P","Row P","Global G10","Global G100"]
CCOLORS=["#555555","#258875","#287AA7","#9A70A5","#CF7A32","#B9A054"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,"axes.labelsize":11,
 "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","svg.hashsalt":"structure-learning-v2-20260930"})
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def yes(x):return x is True or x=="True"
def verify(x,msg):
    if not x:raise AssertionError(msg)
def textwrite(p,s):
    with p.open("x",encoding="utf-8",newline="\n") as f:f.write(s.rstrip()+"\n")
def jsonwrite(p,x):
    with p.open("x",encoding="utf-8",newline="\n") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def fmt(x):
    if x is None or x=="":return ""
    if isinstance(x,bool):return "通过" if x else "未通过"
    if isinstance(x,(dict,list)):return json.dumps(x,ensure_ascii=False,separators=(",",":")).replace("|","\\|")
    try:return f"{float(x):.10g}"
    except (TypeError,ValueError):return str(x).replace("|","\\|")
def md(rows,columns=None):
    if not rows:return "无记录。"
    if columns is None:
        columns=[]
        for r in rows:
            for k in r:
                if k not in columns:columns.append(k)
    return "\n".join(["|"+"|".join(columns)+"|","|"+"|".join("---" for _ in columns)+"|"]+
                    ["|"+"|".join(fmt(r.get(k,"")) for k in columns)+"|" for r in rows])
def save(fig,name):
    fig.savefig(FIG/(name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(FIG/(name+".svg"),bbox_inches="tight",facecolor="white",metadata={"Date":None})
    p=FIG/(name+".svg")
    p.write_text("\n".join(s.rstrip() for s in p.read_text(encoding="utf-8").splitlines()).rstrip()+"\n",encoding="utf-8",newline="\n")
    plt.close(fig)
def load():
    for package in ["A","B","C"]:
        verify(read(BASE/package/"STATE.json")["status"]=="complete",package+" incomplete")
        for receipt in ["FREEZE.json","RESULTS_FREEZE.json"]:
            for path,expected in read(BASE/package/receipt)["files"].items():
                verify(sha(ROOT/path)==expected,path)
    data={p:{f:table(BASE/p/(f+".csv")) for f in fs} for p,fs in FILES.items()}
    expected={("A","FINITE_FITS"):1024,("A","FINITE_RISK_INTERVALS"):40,("A","POPULATION_FITS"):32,("A","POPULATION_CURVES"):992,
              ("B","FINITE_FITS"):640,("B","FINITE_RISK_INTERVALS"):32,("B","POPULATION_FITS"):148,
              ("B","POPULATION_FIXED_COMPARISONS"):8,("B","REFERENCE_CURVES"):1444,("B","GRID"):361,
              ("C","FITS"):576,("C","RISK_INTERVALS"):36,("C","SUMMARY"):36}
    for (p,f),n in expected.items():verify(len(data[p][f])==n,p+"/"+f+" row count")
    verify({r["profile"] for r in data["C"]["FITS"]}=={"null","contrast"},"null profile missing")
    verify({int(r["N"]) for r in data["C"]["FITS"]}=={256,4096,65536},"C N values")
    return data,expected

def floor_figure(data):
    teacher=read(BASE/"A/TEACHER.json");r=np.array(teacher["r_values"])
    floor=np.array(teacher["projection_floor_by_r"])
    a=np.array(read(BASE/"A/CANDIDATE_REFERENCE.json")["J_grid_best_KL"])
    b=np.array([min(float(x["joint_KL"]) for x in data["B"]["REFERENCE_CURVES"] if int(x["r"])==v) for v in r])
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    for values,label,co,mark in [(floor,"Wrong-group projection floor",COLORS["floor"],"o"),
                                 (a,"A: best J on fixed-ratio grid",COLORS["A"],"s"),
                                 (b,"B: best J on two-scale grid",COLORS["B"],"D")]:
        axes[0].plot(r,values,marker=mark,color=co,lw=1.7,label=label)
    axes[1].plot(r,a-floor,"s-",color=COLORS["A"],label="A grid best minus projection floor")
    axes[1].plot(r,b-floor,"D-",color=COLORS["B"],label="B grid best minus projection floor")
    for ax in axes:
        ax.set_xticks(r);ax.set_xlabel("Exchanged coordinates per group r")
        ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0));ax.grid(alpha=.2)
        ax.legend(fontsize=8.6,frameon=False)
    axes[0].set_ylabel("Full-vector joint KL (nats)");axes[0].set_title("Structural floor and best candidates")
    axes[1].set_ylabel("Remaining model/grid gap (nats)");axes[1].set_title("Freeing the scale ratio reduces extra loss")
    fig.suptitle("Incorrect group information leaves a projection floor",fontsize=14,y=1.01)
    fig.text(.5,-.015,"Same teacher and candidate-group contracts; each best J is only the declared finite-grid minimum.",ha="center",fontsize=9)
    fig.tight_layout();save(fig,"projection_floors")
    return [dict(r=int(rr),projection_floor=float(ff),A_best_J_KL=float(aa),B_best_J_KL=float(bb))
            for rr,ff,aa,bb in zip(r,floor,a,b)]

def b_landscapes(data):
    records=[read(BASE/"B/streams"/f"s{s:02}.json") for s in range(16)]
    panel=np.array([r["population_objectives"] for r in records]).mean(0)
    rr=[0,4];methods=["J","P","T","G10"];units=["full-vector nats","nats per pair","nats per count","full-vector nats"]
    mats={}
    for r in rr:
        rows=sorted([x for x in data["B"]["REFERENCE_CURVES"] if int(x["r"])==r],key=lambda x:int(x["index"]))
        for method in methods:
            v=panel[[0,1,2,4].index(r),:,0] if method=="G10" else np.array([float(x[method]) for x in rows])
            mats[r,method]=(v-v.min()).reshape(19,19).T
    oracle=read(BASE/"B/ORACLE_DIAGNOSTIC.json");w=oracle["continuous_scales"][0]
    fig,axes=plt.subplots(2,4,figsize=(14.8,8.0),sharex=True,sharey=True,layout="constrained")
    for j,method in enumerate(methods):
        vmax=max(float(mats[r,method].max()) for r in rr)
        norm=PowerNorm(gamma=.5,vmin=0,vmax=vmax)
        for i,r in enumerate(rr):
            ax=axes[i,j]
            im=ax.imshow(mats[r,method],origin="lower",extent=[-.25,9.25,-.25,9.25],cmap="viridis",norm=norm,aspect="equal")
            selected=next(x for x in data["B"]["POPULATION_FITS"] if int(x["r"])==r and x["unit"]=="aggregate_or_deterministic" and x["objective"]==method)
            ax.scatter(float(selected["u"]),float(selected["v"]),marker="s",facecolors="none",edgecolors="#FFDF5D",s=72,linewidths=1.5)
            ax.scatter(6,2,marker="*",s=80,color="white",edgecolors="black",linewidths=.55)
            ax.scatter(2,6,marker="o",s=45,facecolors="none",edgecolors="white",linewidths=1.1)
            if r==4 and method=="P":
                ax.scatter(w,w,marker="x",s=85,color=COLORS["oracle"],linewidths=1.8)
                ax.annotate(f"continuous P:\n({w:.3f}, {w:.3f})",xy=(w,w),xytext=(4.7,5.1),fontsize=7.7,color="white",
                            arrowprops=dict(arrowstyle="-",color="white",lw=.7))
            ax.set_title(f"r = {r}  |  "+("G10 bank estimate" if method=="G10" else method))
            ax.set_xticks([0,3,6,9]);ax.set_yticks([0,3,6,9])
            if i==1:ax.set_xlabel("Assumed-group scale u")
            if j==0:ax.set_ylabel("Assumed-group scale v")
        cb=fig.colorbar(im,ax=axes[:,j],shrink=.64,pad=.018,location="bottom")
        ticks=[0.,vmax/9,4*vmax/9,vmax]
        cb.set_ticks(ticks,labels=["0"]+[f"{v:.2g}" for v in ticks[1:]])
        cb.set_label("Excess above this panel minimum\n("+units[j]+"; square-root color scale)",fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Two-scale objective geometry: high-precision targets and a finite-bank estimate",fontsize=14)
    fig.legend(handles=[Line2D([0],[0],marker="*",ls="none",color="#555555",label="True (6,2)"),
                        Line2D([0],[0],marker="o",ls="none",markerfacecolor="none",color="#555555",label="Mirror (2,6)"),
                        Line2D([0],[0],marker="s",ls="none",markerfacecolor="none",color="#B89A16",label="Reported grid point; G10 is a noisy argmin"),
                        Line2D([0],[0],marker="x",ls="none",color=COLORS["oracle"],label="Continuous r4 pair oracle; outside fitting grid")],
               loc="lower center",bbox_to_anchor=(.5,-.06),ncol=2,frameon=False,fontsize=9)
    save(fig,"two_scale_landscapes")

def c_risks(data):
    lib=read(BASE/"C/RISK_LIBRARY.json")
    fig,axes=plt.subplots(2,3,figsize=(15,8.2),sharey=True)
    for pi,profile in enumerate(["null","contrast"]):
        for ni,N in enumerate([256,4096,65536]):
            ax=axes[pi,ni]
            for j,method in enumerate(CMETHODS):
                rows=sorted([x for x in data["C"]["FITS"] if x["profile"]==profile and int(x["N"])==N and x["method"]==method],
                            key=lambda x:int(x["stream"]))
                values=np.array([float(x["deployed_KL"]) for x in rows])
                ax.scatter(j+np.linspace(-.18,.18,16),values,s=20,color=CCOLORS[j],alpha=.75,edgecolors="white",linewidths=.3)
                ax.scatter(j,float(values.mean()),marker="_",s=210,color=CCOLORS[j],linewidths=2.2,zorder=6)
            if profile=="contrast":ax.axhline(lib[profile]["count_only_bound"],color="#BB4856",ls="--",lw=1.2)
            ax.axhline(0,color="#888888",lw=.65)
            ax.set_yscale("symlog",linthresh=1e-6,linscale=.7,base=10)
            ax.set_xticks(range(6),CLABELS,rotation=43,ha="right",fontsize=8)
            ax.set_title(f"{profile}  |  N = {N:,}");ax.grid(axis="y",alpha=.18)
            if ni==0:ax.set_ylabel("Deployment joint KL (nats)\nsymlog; linear near zero")
    fig.suptitle("Unknown-group learning: all 576 finite-data fitted risks",fontsize=15,y=.995)
    handles=[Line2D([0],[0],marker="o",ls="none",color="#555555",label="All 16 fits at each N"),
             Line2D([0],[0],marker="_",ls="none",markersize=15,color="#555555",label="Descriptive mean"),
             Line2D([0],[0],ls="--",color="#BB4856",label="Prior-average count-only lower bound (contrast)")]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.0),fontsize=9)
    fig.text(.5,.06,"N values are paired prefixes. Oracle J/P know the true group. Global G refits scale after Global P selects a group and reuses B panels.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.95]);save(fig,"unknown_group_risks")

def c_recovery(data):
    bound=read(BASE/"C/ROW_RECOVERY_BOUND.json")
    fig,axes=plt.subplots(1,2,figsize=(12.6,5.2),gridspec_kw={"width_ratios":[1.35,1]})
    ax=axes[0];x=[];labels=[]
    for ni,N in enumerate([256,4096,65536]):
        for mi,method in enumerate(["GLOBAL_P","ROW_P"]):
            xx=ni*2.6+mi;x.append(xx);labels.append(("Global P" if mi==0 else "Row P")+f"\nN={N:,}")
            rr=[r for r in data["C"]["FITS"] if r["profile"]=="contrast" and int(r["N"])==N and r["method"]==method]
            exact=sum(yes(r["high_group_exact_recovery"]) for r in rr)
            identified=sum(yes(r["group_recovery_identifiable"]) for r in rr)
            wrong=identified-exact;unidentified=16-identified
            ax.bar(xx,exact,width=.75,color="#2B8D77");ax.bar(xx,wrong,bottom=exact,width=.75,color="#D4A15B")
            ax.bar(xx,unidentified,bottom=identified,width=.75,color="#C8CCD0")
            ax.text(xx,16.35,f"{exact}/16",ha="center",fontsize=9)
    ax.set_xticks(x,labels,fontsize=8.2);ax.set_ylim(0,18.1);ax.set_yticks([0,4,8,12,16])
    ax.set_ylabel("Observed replicate counts");ax.set_title("Exact high-group recovery: contrast")
    ax.legend(handles=[Patch(color="#2B8D77",label="Exact"),Patch(color="#D4A15B",label="Identified but wrong"),Patch(color="#C8CCD0",label="Equal scales: group not identified")],
              fontsize=8,loc="upper left",frameon=False,bbox_to_anchor=(0,-.18),ncol=1)
    ax.grid(axis="y",alpha=.15)
    ax=axes[1];ns=[r["N"] for r in bound["bounds"]];bs=[r["numerical_allowance_union_failure_upper"] for r in bound["bounds"]]
    ax.plot(ns,bs,"o-",color=COLORS["B"],lw=1.7)
    for n,b in zip(ns,bs):
        ax.annotate("vacuous = 1" if b>=1 else f"{b:.3g}",(n,b),xytext=(0,9),textcoords="offset points",ha="center",fontsize=9)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_ylim(5e-5,2.7)
    ax.set_xticks(ns,[f"{n:,}" for n in ns]);ax.set_xlabel("N");ax.set_ylabel("ROW failure-probability upper bound")
    ax.set_title("A separate Chernoff + 64-pair union bound");ax.grid(alpha=.2,which="both")
    fig.suptitle("Empirical recovery and a conditional probabilistic bound are different evidence",fontsize=13.5,y=1.015)
    fig.text(.5,-.075,"16/16 is only this paired experiment. The ROW bound is conditional on the qualified numerical teacher; the allowance is not a strict quadrature certificate.",
             ha="center",fontsize=8.8)
    fig.tight_layout(rect=[0,.04,1,.96]);save(fig,"group_recovery")

def readable(data,floors):
    tables=FIG/"tables";tables.mkdir()
    links=[]
    for package,files in FILES.items():
        for name in files:
            label=package+"_"+name
            intro=f"# {label}：全部{len(data[package][name])}行\n\n来源：../../{package}/{name}.csv。无删除；显示10位有效数字，原CSV保留完整浮点。\n\n"
            if package=="A":intro+="A的S部署S(Q)，其余目标部署假定组下的Q；复用旧bank是开发性重加权。\n\n"
            if package=="B":intro+="B的T以total-count估计参数后部署Q，完整镜像类另列；G聚合argmin含积分噪声。\n\n"
            if package=="C":intro+="C的各N为同流前缀。未知组GLOBAL_P与已知组ORACLE_J/P不同；GLOBAL_G复用B粒子曲面。\n\n"
            textwrite(tables/(label+".md"),intro+md(data[package][name]));links.append(label)
    ties=read(BASE/"B/T_EQUIVALENCE_CLASSES.json")
    textwrite(tables/"B_T_EQUIVALENCE_CLASSES.md","# B T：全部132个精确/近似最小等价类\n\n近似类不改变原lexicographic tie-break；镜像方向不能计为识别成功。\n\n"+md(ties));links.append("B_T_EQUIVALENCE_CLASSES")
    decisions={p:read(BASE/p/"DECISION.json") for p in ["A","B","C"]}
    primaries=[];failures=[]
    for p,dec in decisions.items():
        for r in dec["primary"]:
            primaries.append(dict(package=p,**r))
    for p,file in [("A","FINITE_RISK_INTERVALS"),("B","FINITE_RISK_INTERVALS"),("B","POPULATION_FIXED_COMPARISONS"),("C","RISK_INTERVALS")]:
        for r in data[p][file]:
            if not yes(r["precision_pass"]):failures.append(dict(package=p,source=file,**r))
    verify(len(failures)==4,"expected complete four precision failures")
    textwrite(tables/"ALL_PRIMARY.md","# 全部原定主判定\n\n各包的家族、SE门与问题分别解释；C通过不追认B未通过。\n\n"+md(primaries));links.append("ALL_PRIMARY")
    textwrite(tables/"PRECISION_FAILURES.md","# 全部4项精度失败\n\n原门保持，未追加种子、bank或改阈值。\n\n"+md(failures));links.append("PRECISION_FAILURES")
    js=read(BASE/"B/INFO_OBSTRUCTION.json");oracle=read(BASE/"B/ORACLE_DIAGNOSTIC.json");lib=read(BASE/"C/RISK_LIBRARY.json")
    bounds=read(BASE/"C/ROW_RECOVERY_BOUND.json")
    csummary=[r for r in data["C"]["SUMMARY"] if r["profile"]=="contrast"]
    lines=["# 结构先验释放与分组学习：A/B/C完整结果","",
           "本页只整理已经完成并封存的A/B/C，0新样本、拟合、参考积分或统计家族。A有1024次、B有640次、C有576次有限数据拟合，共2240次；完整表全部保留。","",
           "A在固定尺度比例下给错分组，S实际部署完整对称化S(Q)；B释放两个尺度，T只用总count估参后部署假定组下的Q，保留所有镜像；C再释放组信息，由带坐标pair学习分组。A/B共用A观察，C另生数据；A复用旧bank、C复用B bank，不能叫全部独立新确认。","",
           "## 原判定与完整失败账本","",
           "A：40/40精度、4/4主门通过。B：32项有限数据精度29通过、8/8固定总体精度通过；人口固定比较主门通过，有限数据主门未通过。C：36项精度35通过、3/3主门通过。","",
           "B的r0/N65536 G10−J虽然区间为正，SE=.00101459565略高于原.001门，因此原主门false保持。C有不同问题、不同家族及SE门.002，其通过不能追认B。","",md(primaries),"",
           "全部4项精度失败如下，均保持原规则：","",md(failures),"",
           "## 错误分组造成的结构下界","",
           "![分组projection floor](../../../artifacts/structure_learning_v2/figures/projection_floors.png)","",md(floors),"",
           "所有risk均减原真实教师H(P)。floor是错误组内轨道投影损失；A/B的J最佳值分别只在各自预定网格内最优，不是连续全局最优。释放比例降低了额外模型/网格损失，但不消除错误分组floor。","",
           "## 二维目标、镜像歧义与连续诊断","",
           "![二维目标](../../../artifacts/structure_learning_v2/figures/two_scale_landscapes.png)","",
           "每个panel减自己的最小值，P、T、J/G分别使用pair、count、完整向量单位；各列色标独立，不比较它们原始训练目标的大小。G10仅为16流聚合bank目标估计，不能称精确总体最优。T的mirror具有严格共享的total-count概率。","",
           f"已知两组但尺度朝向未知时，任意数量count-only训练数据的两点minimax joint风险下界为JS(P6,2,P2,6)，本数值为{js['primary_JS']:.10g} nats。T按小u选择镜像只是预定tie-break实例；全部132个最小等价类和各方向风险另表完整保留。","",
           f"r4的连续精确pair oracle为({oracle['continuous_scales'][0]:.10g},{oracle['continuous_scales'][1]:.10g})，joint KL={oracle['continuous_oracle_joint_KL']:.10g}；网格选择{oracle['grid_selected_scales']}，pair网格gap={oracle['grid_minus_continuous_pair_gap']:.10g}。连续解不加入361拟合候选，也不把离散对角选择当成一般定理。","",
           "B四个r的人口J/P当前网格点相同，不能据此宣称发现了P相对J的错设优势或劣势。","",
           "## 未知组学习的全部有限风险","",
           "![未知组风险](../../../artifacts/structure_learning_v2/figures/unknown_group_risks.png)","",
           f"C的contrast count-only prior-average joint风险下界为{lib['contrast']['count_only_bound']:.10g} nats，与B的两方向JS下界是不同信息合同。图中每个点为一次拟合的部署KL，短横线仅为描述均值；水平界约束count-only程序的prior-average风险，不是每个数据集的逐点界。","",
           "ORACLE_J/P已知真组；GLOBAL_P穷举全mask/pair候选，ROW_P使用行分数恢复组。GLOBAL_G10/G100先沿用GLOBAL_P选组再按B有限粒子曲面拟合尺度；并未穷举未知组的完整joint likelihood，也不是新的独立bank确认。","",
           "contrast全部N与六程序汇总如下。N256的边界尺度和较高损失保留，不只展示大N成功：","",md(csummary),"",
           "## 组恢复与概率界","",
           "![组恢复](../../../artifacts/structure_learning_v2/figures/group_recovery.png)","",
           "contrast的GLOBAL_P与ROW_P在N65536各16/16恢复正确高尺度组，只是本次16个配对数据流的经验结果；不同N是前缀，不能当48个独立重复，也不能说普遍100%。null的组本来不可识别，不列为恢复成功。","",
           "右图为预定lambda网格上的ROW Chernoff与64对union上界；小N界等于1属于无信息，不是预测一定失败。该界条件于qualified数值教师，数值余量并非严格Gaussian积分证书。","",md(bounds["bounds"]),"",
           "## 完整可读表","",
           "以下包含全部2240次拟合、116项原统计（A40/B32+8/C36）、所有人口/参考曲线以及T等价类。未增加置信家族；各原家族分开解释。",""]
    for label in links:lines.append(f"- [{label}](../../../artifacts/structure_learning_v2/figures/tables/{label}.md)")
    lines+=["","## 来源与生成","",
            "图表来源使用csv.DictReader，字符串null保留为真实profile。每个科学包的原FREEZE与RESULTS_FREEZE逐hash核对后才作图。SVG行尾空格及末尾换行规范化后再登记figures/FIGURE_MANIFEST.json。原科学freeze、失败与统计门不改。"]
    textwrite(DOC/"RESULTS.md","\n".join(lines))
    return primaries,failures,links

def build():
    verify(not FIG.exists(),"figure output already exists");verify(not (DOC/"RESULTS.md").exists(),"RESULTS exists")
    data,expected=load();FIG.mkdir(parents=True)
    floors=floor_figure(data);b_landscapes(data);c_risks(data);c_recovery(data)
    primaries,failures,links=readable(data,floors)
    audit=dict(passed=True,source_rows={p+"/"+f:n for (p,f),n in expected.items()},finite_fits=2240,
               original_statistics=116,precision_failures=len(failures),primary_records=len(primaries),
               null_profile_retained=True,new_scientific_samples=0,new_fits=0,new_reference_integrals=0,
               new_statistical_intervals=0,complete_readable_tables=len(links))
    jsonwrite(FIG/"FIGURE_AUDIT.json",audit)
    sources=[Path(__file__)]+[BASE/p/(name+".csv") for p,files in FILES.items() for name in files]
    for p in ["A","B","C"]:sources += [BASE/p/"FREEZE.json",BASE/p/"RESULTS_FREEZE.json",BASE/p/"STATE.json",BASE/p/"DECISION.json"]
    sources += [BASE/"A/TEACHER.json",BASE/"A/CANDIDATE_REFERENCE.json",BASE/"B/INFO_OBSTRUCTION.json",
                BASE/"B/ORACLE_DIAGNOSTIC.json",BASE/"B/T_EQUIVALENCE_CLASSES.json",
                BASE/"C/RISK_LIBRARY.json",BASE/"C/ROW_RECOVERY_BOUND.json"]
    sources += [BASE/"B/streams"/f"s{s:02}.json" for s in range(16)]
    outputs=[p for p in FIG.rglob("*") if p.is_file()]+[DOC/"RESULTS.md"]
    jsonwrite(FIG/"FIGURE_MANIFEST.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),
        source_files={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
        output_files={p.relative_to(ROOT).as_posix():sha(p) for p in outputs},
        kind="post_execution_figures_and_readable_tables"))
    print(json.dumps(dict(passed=True,figures=4,tables=len(links),manifest=sha(FIG/"FIGURE_MANIFEST.json"))))
def check():
    r=read(FIG/"FIGURE_MANIFEST.json")
    for kind in ["source_files","output_files"]:
        for p,h in r[kind].items():verify(sha(ROOT/p)==h,p)
    print(json.dumps(dict(passed=True,sources=len(r["source_files"]),outputs=len(r["output_files"]))))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["build","check"]);args=p.parse_args()
    build() if args.phase=="build" else check()

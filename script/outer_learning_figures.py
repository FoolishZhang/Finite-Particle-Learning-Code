"""Two scientific figures from sealed gradient and training results only.

build is blocked until training is complete/result-sealed. Inspect rendered PNGs,
obtain root review, then seal --visual-qa --root-reviewed. No science is run here.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from fractions import Fraction
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/nuisance_learning"
GRAD=BASE/"GRADIENT"
OUT=BASE/"figures"
DOC=ROOT/"docs/08_research_研究工作/外层精度与有限样本伪相关_20261001/FIGURES.md"
NAMES=("gradient_budget","finite_learning")
METHODS=("Price","raw_path","noise_CV")
LABEL={"Price":"Price (label RB)","raw_path":"Raw path (no CV)","noise_CV":"Noise-softmax CV"}
COLOR={"Price":"#0072B2","raw_path":"#D55E00","noise_CV":"#009E73"}
MARK={"Price":"o","raw_path":"s","noise_CV":"^"}

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def need(ok,msg):
    if not ok:raise AssertionError(msg)
def verify(m):
    for p,h in m.items():need(sha(ROOT/p)==h,"changed "+p)
def fraction(r):return Fraction(int(r["numerator"]),int(r["denominator"]))
def outward_decimal(value,digits=6,upper=False):
    """Exact rational floor for a lower bound, ceiling for an upper bound."""
    scale=10**digits
    units,remainder=divmod(value.numerator*scale,value.denominator)
    if upper and remainder:units+=1
    sign="-" if units<0 else "";magnitude=abs(units)
    return f"{sign}{magnitude//scale}.{magnitude%scale:0{digits}d}"

def load_inputs():
    # No gradient or training risk data is opened before both training guards.
    need((BASE/"STATE.json").exists(),"training not started; no results opened")
    need(read(BASE/"STATE.json")["state"]=="complete","training incomplete; no risk results opened")
    need((BASE/"RESULTS_FREEZE.json").exists(),"training result seal missing")
    trainseal=read(BASE/"RESULTS_FREEZE.json");trainfreeze=read(BASE/"FREEZE.json")
    public={rel(BASE/n):h for n,h in trainseal["files"].items()}
    public.update({i["snapshot"]:i["sha256"] for i in trainfreeze["inputs"]})
    for n,k in (("CONFIG.json","config_sha256"),("REFERENCE.json","reference_sha256"),("UNIT_CHECK.json","unit_sha256")):
        public[rel(BASE/n)]=trainfreeze[k]
    public[rel(BASE/"RESULTS_FREEZE.json")]=sha(BASE/"RESULTS_FREEZE.json")
    need(read(GRAD/"STATE.json")["status"]=="complete","gradient component incomplete")
    gs=read(GRAD/"RESULTS_FREEZE.json");gf=read(GRAD/"FREEZE.json");move=read(GRAD/"SOURCE_RELOCATION.json")
    need(move["old_path"]=="script/nuisance_learning_gradient.py" and move["new_path"]=="script/outer_learning_gradient_source.py","unexpected source relocation")
    need(gf["files"][move["old_path"]]==move["unchanged_source_sha256"],"gradient source bytes")
    need(gs["source_freeze_sha256"]==sha(GRAD/"FREEZE.json")==move["source_freeze_sha256"],"gradient source seal")
    need(sha(GRAD/"RESULTS_FREEZE.json")==move["result_freeze_sha256"],"gradient results seal")
    public.update({(move["new_path"] if p==move["old_path"] else p):h for p,h in gf["files"].items()})
    public.update(gs["files"])
    for p in (GRAD/"RESULTS_FREEZE.json",GRAD/"SOURCE_RELOCATION.json"):
        public[rel(p)]=sha(p)
    verify(public)
    gradient=read(GRAD/"RESULTS.json");gh=read(GRAD/"GH_REFERENCE.json")
    evaluation=read(BASE/"EVALUATION.json")["paired_cells"];training=read(BASE/"RESULTS.json")
    need(len(evaluation)==576,"paired training rows (1,152 raw/RB assignments)")
    need(len(gradient["estimators"])==18 and len(gradient["outer_B"])==90,"gradient result count")
    for r in evaluation:
        if r["n"]==0:need(r["raw_index"]==r["RB_index"] and r["raw_reference_KL"]==r["RB_reference_KL"],"n0 raw must equal nuisance-RB reuse")
    need(sorted(x["K"] for x in training["endpoint_CP_family"])==[2,4],"CP family")
    for r in training["endpoint_CP_family"]:
        need(fraction(r["true_raw_expected_KL_lower"])==fraction(r["CP_lower"])/128,"CP-derived expected-risk bound")
    return gradient,gh,evaluation,training,public

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
        "axes.labelsize":12,"axes.titlesize":13,"xtick.labelsize":10,
        "ytick.labelsize":10,"legend.fontsize":10,"svg.fonttype":"none",
        "savefig.dpi":220,"axes.spines.top":False,"axes.spines.right":False})

def export(fig,name):
    fig.savefig(OUT/(name+".png"),facecolor="white")
    fig.savefig(OUT/(name+".svg"),facecolor="white",metadata={"Creator":"outer_learning_figures.py"})
    plt.close(fig)

def gradient_figure(result,gh):
    fig,ax=plt.subplots(1,2,figsize=(12.3,6.0))
    variance_rows=[];mse_rows=[]
    for method in METHODS:
        values=sorted([r for r in result["estimators"] if r["method"]==method and r["n"]>0],key=lambda r:r["n"])
        ns=[r["n"] for r in values];ys=[r["variance_ddof1"] for r in values]
        need(all(y>0 for y in ys),"nonpositive sample variance cannot use log axis")
        ax[0].plot(ns,ys,color=COLOR[method],marker=MARK[method],lw=1.9,ms=6,label=LABEL[method])
        variance_rows.extend({"method":method,"n":n,"sample_variance":y} for n,y in zip(ns,ys))
        bdata=sorted([r for r in result["outer_B"] if r["method"]==method and r["n"]==4094],key=lambda r:r["outer_B"])
        bs=[r["outer_B"] for r in bdata];ys=[r["MSE_to_fixed_reference"]/r["reference_gradient"]**2 for r in bdata]
        need(all(y>0 for y in ys),"nonpositive relative MSE cannot use log axis")
        ax[1].plot(bs,ys,color=COLOR[method],marker=MARK[method],lw=1.9,ms=6,label=LABEL[method])
        mse_rows.extend({"method":method,"n":4094,"B":r["outer_B"],"groups":r["groups"],"fixed_gradient_reference":r["reference_gradient"],"relative_MSE":y} for r,y in zip(bdata,ys))
    theory=next(r["raw_path_RB_variance_limit"] for r in gh["rows"] if r["order"]==96)
    ax[0].axhline(theory,color="#333333",ls="--",lw=1.5,label="Raw-path RB asymptotic variance")
    ax[0].text(.03,.04,f"Analytic-limit reference: {theory:.5f}",transform=ax[0].transAxes,fontsize=9,color="#444444")
    for a in ax:
        a.set_xscale("log");a.set_yscale("log");a.grid(True,which="major",color="#dddddd",lw=.6)
    ax[0].set_title("Single-bank variability",loc="left",pad=12)
    ax[0].set_xlabel("Independent nuisance coordinates n")
    ax[0].set_ylabel("Sample variance (all 65,536 banks)")
    ax[0].set_xticks([14,62,254,1022,4094],["14","62","254","1,022","4,094"])
    ax[1].set_title("Outer-bank error at n = 4,094",loc="left",pad=12)
    ax[1].set_xlabel("Banks per non-overlapping group B")
    ax[1].set_ylabel(r"Group MSE / frozen $\mu_n^2$")
    ax[1].set_xticks([16,64,256,1024,4096],["16","64","256","1,024","4,096"])
    ax[1].axhline(1,color="#bbbbbb",ls=":",lw=1,zorder=0)
    handles=[Line2D([],[],color=COLOR[m],marker=MARK[m],lw=1.8,label=LABEL[m]) for m in METHODS]
    handles.append(Line2D([],[],color="#333333",ls="--",label="Raw-path RB limit (left)"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.53,.887),ncol=4,frameon=False,columnspacing=1.6)
    fig.suptitle("Outer-bank precision for the same K = 2 gradient",x=.07,ha="left",y=.979,fontsize=16)
    fig.text(.07,.910,"Null teacher; a = 1, rho = 0.75. All methods average the same four active labels.",fontsize=11)
    foot="Old 65,536 banks reused after K = 4 development; no new independent confirmation or confidence intervals.\nFor B = 16 / 64 / 256 / 1,024 / 4,096, there are 4,096 / 1,024 / 256 / 64 / 16 groups. B and n are nested.\nMSE uses the frozen numerical population gradient; equal B is not equal running time."
    if not gh["qualified"]:foot+=" The dashed variance reference did not pass its numerical gate."
    fig.text(.07,.026,foot,fontsize=9.1,linespacing=1.4,color="#333333")
    fig.subplots_adjust(left=.09,right=.975,bottom=.23,top=.78,wspace=.29)
    export(fig,NAMES[0])
    return {"variance":variance_rows,"MSE":mse_rows,"raw_path_RB_limit":theory,"GH_reference_qualified":gh["qualified"]}

def training_figure(evaluation,result):
    fig,ax=plt.subplots(1,2,figsize=(12.3,6.2),gridspec_kw={"width_ratios":[1.08,1]})
    risk_rows=[]
    for method,color in (("raw","#D55E00"),("RB","#0072B2")):
        for K,linestyle,marker in ((2,"-","o"),(4,"--","s")):
            bs=[];means=[]
            for B in (32,512,8192):
                records=sorted([r for r in evaluation if (r["n"],r["N"],r["B"],r["K"])==(254,1024,B,K)],key=lambda r:r["stream"])
                need([r["stream"] for r in records]==list(range(16)),"all training streams required")
                values=[r[method+"_reference_KL"] for r in records]
                bs.append(B);means.append(float(np.mean(values)))
                risk_rows.append({"method":method,"K":K,"N":1024,"n":254,"B":B,"reference_KL_values":values,"reference_KL_mean":means[-1],"streams":16})
            ax[0].plot(bs,means,color=color,ls=linestyle,marker=marker,ms=6,lw=1.9,label=("Raw" if method=="raw" else "Nuisance-RB")+f", K = {K}")
    ax[0].set_xscale("log");ax[0].set_xticks([32,512,8192],["32","512","8,192"])
    ax[0].set_ylim(bottom=0);ax[0].grid(axis="y",color="#dddddd",lw=.6)
    ax[0].set_xlabel("Shared global prior banks B")
    ax[0].set_ylabel("Mean reference joint KL (nats)")
    ax[0].set_title("N = 1,024 labels, n = 254 nuisance bits",loc="left",pad=12)
    ax[0].legend(loc="best",frameon=True,framealpha=.9,edgecolor="#dddddd",fontsize=9.3)
    bound_rows=[];maxbound=0.
    for i,K in enumerate((2,4)):
        r=next(r for r in result["endpoint_CP_family"] if r["K"]==K)
        lower_fraction=fraction(r["true_raw_expected_KL_lower"]);upper_fraction=fraction(r["true_table_MLE_expected_KL_upper"])
        lower=float(lower_fraction);upper=float(upper_fraction)
        lower_label=outward_decimal(lower_fraction);upper_label=outward_decimal(upper_fraction,upper=True)
        maxbound=max(maxbound,lower,upper)
        ax[1].scatter(i-.07,lower,color="#D55E00",marker="^",s=98,edgecolors="black",linewidth=.6,zorder=5)
        ax[1].scatter(i+.07,upper,color="#009E73",marker="v",s=98,edgecolors="black",linewidth=.6,zorder=5)
        ax[1].plot([i-.07,i+.07],[lower,upper],color="#cccccc",lw=.9,zorder=1)
        ax[1].annotate(f">= {lower_label}",(i-.07,lower),xytext=(0,10),textcoords="offset points",ha="center",fontsize=9.4,color="#9a3b00")
        ax[1].annotate(f"<= {upper_label}",(i+.07,upper),xytext=(0,-19),textcoords="offset points",ha="center",fontsize=9.4,color="#007451")
        bound_rows.append({"K":K,"endpoint_selections":r["successes"],"trials":r["trials"],"CP_lower":r["CP_lower"],"raw_true_expected_KL_lower":r["true_raw_expected_KL_lower"],"table_MLE_true_expected_KL_upper":r["true_table_MLE_expected_KL_upper"],"display_lower_outward":lower_label,"display_upper_outward":upper_label,"display_rounding":"exact Fraction floor/ceiling to 6 decimals","derived_lower_positive":r["derived_lower_positive"]})
    ax[1].set_xlim(-.5,1.5);ax[1].set_xticks([0,1],["K = 2","K = 4"])
    ax[1].set_ylim(-.1*maxbound,1.26*maxbound if maxbound else .001)
    ax[1].set_title("N = 1,024, B = 8,192: analytic bounds",loc="left",pad=12)
    ax[1].set_ylabel("Expected joint KL bound (nats)")
    ax[1].grid(axis="y",color="#dddddd",lw=.6)
    bound_legend=[Line2D([],[],marker="^",ls="none",color="#D55E00",ms=8,label=r"Raw lower bound: $p_L / 128$"),Line2D([],[],marker="v",ls="none",color="#009E73",ms=8,label="Table-MLE analytic upper bound")]
    ax[1].legend(handles=bound_legend,loc="upper center",bbox_to_anchor=(.5,-.125),frameon=False,fontsize=9.1)
    primary=result["primary_sign_test"];pval=float(fraction(primary["exact_one_sided_pvalue"]))
    primary_text=f"Separate paired K = 4 test: {'passed' if primary['passed'] else 'failed'} ({primary['successes']}/{primary['trials']} improvement events; p = {pval:.4g})."
    fig.suptitle("Finite-label training with and without nuisance marginalization",x=.07,ha="left",y=.979,fontsize=16)
    fig.text(.07,.910,"Left: all 16 training streams; means of numerical reference risks, not confidence intervals.",fontsize=11)
    endpoint=lambda method,K:next(r["reference_KL_mean"] for r in risk_rows if r["method"]==method and r["K"]==K and r["B"]==8192)
    fig.text(.07,.866,f"At B = 8,192: K = 2 raw / RB = {endpoint('raw',2):.5f} / {endpoint('RB',2):.5f}; "
             f"K = 4 = {endpoint('raw',4):.5f} / {endpoint('RB',4):.5f}.",fontsize=10,color="#333333")
    foot="Nuisance-RB exactly reuses n = 0 raw selections. Left curves share data and nested N/B prefixes.\nRight markers are bounds, not risk estimates. The two K-specific CP lower bounds have joint 95% coverage.\n"+primary_text+"\nThe paired test has its own alpha = .05 family and is not a mean-gain test; families are not combined into overall 95% coverage."
    fig.text(.07,.022,foot,fontsize=8.9,linespacing=1.4,color="#333333")
    fig.subplots_adjust(left=.09,right=.975,bottom=.32,top=.79,wspace=.32)
    export(fig,NAMES[1])
    return {"risk_curves":risk_rows,"bounds":bound_rows,"original_primary_sign_test":primary,"families_separate":True}

def write_doc(points):
    primary=points["training"]["original_primary_sign_test"]
    text="# 外层精度与有限标签学习：两张结果图\n\n"
    text+="图仅从已冻结GRADIENT与训练结果确定性派生。0新随机数、标签、拟合、定位或积分；未替换任何原门。图片均提供PNG与SVG，点值见artifacts/nuisance_learning/figures/PLOT_DATA.json。\n\n"
    text+="## 梯度预算\n\n![梯度方差与外层相对MSE](../../../artifacts/nuisance_learning/figures/gradient_budget.png)\n\n"
    text+="左图为n>0的单bank样本方差；虚线为raw-path四标签RB的理论极限数值参考。右图为n4094的group MSE除以冻结梯度平方，未使用样本均值作真值。65536旧bank已有K4开发观察；不同B的组数和嵌套关系已标注，无新CI或独立确认，同B不等于同运行时间。\n\n"
    text+="## 真实有限标签训练\n\n![有限标签参考风险与解析期望风险界](../../../artifacts/nuisance_learning/figures/finite_learning.png)\n\n"
    text+="左图包含高N、n254下所有B和两个K的raw/RB平均参考风险，共4条线；nuisance-RB与n0 raw为精确选点复用。右图是raw期望真KL的CP推导下界p_L/128与table-MLE解析期望上界，均不是风险点估计。两个K的CP下界属于联合95%家族；配对改善事件主检验是另一alpha=.05家族，不合称整个研究95%。\n\n"
    risks=points["training"]["risk_curves"]
    get=lambda m,k:next(r["reference_KL_mean"] for r in risks if r["method"]==m and r["K"]==k and r["B"]==8192)
    text+=f"固定高N/B格中，K2 raw/RB参考均值为{get('raw',2):.9f}/{get('RB',2):.9f}，K4为{get('raw',4):.9f}/{get('RB',4):.9f}。两个K的效果不同，不能包装成普遍修复；原B曲线所有点与非单调变化均保留。\n\n"
    text+=f"原主检验：{'通过' if primary['passed'] else '未通过'}，{primary['successes']}/{primary['trials']}个充分改善事件；精确单侧p值{float(fraction(primary['exact_one_sided_pvalue'])):.10g}。失败不会被图形过滤。它检验改善事件概率是否超过1/2，不是平均风险增量。\n\n"
    text+="图形完成实际视觉QA并经root复核后才可seal；最终收据绑定图源、全部输入、点值及PNG/SVG。本页不认证连续最优、当前预算的理论渐近阈值或新通用方法。\n"
    DOC.write_text(text,encoding="utf-8")

def build():
    need(not (OUT/"FIGURE_RECEIPT.json").exists(),"figures sealed; no overwrite")
    gradient,gh,ev,tr,inputs=load_inputs();OUT.mkdir(parents=True,exist_ok=True);style()
    points={"gradient":gradient_figure(gradient,gh),"training":training_figure(ev,tr)}
    save(OUT/"PLOT_DATA.json",points);write_doc(points)
    outputs=[OUT/(name+"."+ext) for name in NAMES for ext in ("png","svg")]+[OUT/"PLOT_DATA.json",DOC]
    save(OUT/"BUILD.json",{"utc":utc(),"draft_source_sha256":sha(__file__),"inputs":inputs,"outputs":{rel(p):sha(p) for p in outputs},"new_samples":0,"new_RNG":0,"new_fits":0,"new_integrals":0,"new_statistics_families":0,"visual_QA_completed":False,"root_reviewed":False,"matplotlib":matplotlib.__version__,"pre_render_software_correction":"First build stopped before rendering because its count guard confused 1,152 raw/RB assignments with 576 paired rows; guard corrected before any figure seal, no science rerun."})
    print("Draft figures built. Actual visual QA and root review are required before seal.")

def seal(visual,root_reviewed):
    need(visual and root_reviewed,"both actual visual QA and root review required")
    need(not (OUT/"FIGURE_RECEIPT.json").exists(),"already sealed")
    b=read(OUT/"BUILD.json");need(b["draft_source_sha256"]==sha(__file__),"source changed after build")
    verify(b["inputs"]);verify(b["outputs"])
    save(OUT/"FIGURE_RECEIPT.json",{**b,"utc":utc(),"final_source_sha256":sha(__file__),"build_sha256":sha(OUT/"BUILD.json"),"visual_QA_completed":True,"root_reviewed":True,"actual_viewed_PNGs":[n+".png" for n in NAMES]})
    print("Figures sealed after visual QA and root review.")

def check():
    r=read(OUT/"FIGURE_RECEIPT.json")
    need(r["final_source_sha256"]==sha(__file__) and r["build_sha256"]==sha(OUT/"BUILD.json"),"figure source/build changed")
    need(r["visual_QA_completed"] and r["root_reviewed"],"review incomplete")
    verify(r["inputs"]);verify(r["outputs"])
    print(json.dumps({"passed":True,"read_only":True,"render_or_science_reexecuted":False,"receipt_sha256":sha(OUT/"FIGURE_RECEIPT.json")}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("build","seal","check"));p.add_argument("--visual-qa",action="store_true");p.add_argument("--root-reviewed",action="store_true");a=p.parse_args()
    if a.action=="build":build()
    elif a.action=="seal":seal(a.visual_qa,a.root_reviewed)
    else:check()

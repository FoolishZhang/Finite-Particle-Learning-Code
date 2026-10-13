"""Saved-result scientific figures; no fitting, sampling, or quadrature.

Risk-bearing inputs are opened only after RUN is complete and result-sealed.
Use build, visually inspect PNGs, then seal --visual-qa; check is read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"artifacts/compensation_learning/RUN"
OUT=ROOT/"artifacts/compensation_learning/figures"
METHODS=("MOM","quasiMLE","raw","col")
LABELS={"MOM":"MOM","quasiMLE":"Quasi-MLE","raw":"Raw","col":"Collapsed"}
COLORS={"MOM":"#0072B2","quasiMLE":"#009E73","raw":"#D55E00","col":"#CC79A7"}
MARKERS={"MOM":"o","quasiMLE":"s","raw":"^","col":"v"}
FIGURES=("joint_risk","learned_loadings_high_N")

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def verify(mapping):
    for path,digest in mapping.items():require(sha(ROOT/path)==digest,"hash mismatch: "+path)

def inputs():
    # Do not even read RESULTS.json until both required completion guards pass.
    require((RUN/"STATE.json").exists(),"RUN has not started")
    require(read(RUN/"STATE.json")["status"]=="complete","RUN incomplete: no risk inputs opened")
    require((RUN/"RESULTS_FREEZE.json").exists(),"RUN results not sealed: no risk inputs opened")
    seal=read(RUN/"RESULTS_FREEZE.json")
    require(seal["source_freeze_sha256"]==sha(RUN/"FREEZE.json"),"RUN source seal link")
    freeze=read(RUN/"FREEZE.json")
    verify(freeze["files"])
    paths=[RUN/"STATE.json",RUN/"RESULTS.json",RUN/"FREEZE.json",RUN/"SELECTION_FREEZE.json"]
    for p in paths:require(seal["files"][rel(p)]==sha(p),"RUN result input seal: "+rel(p))
    result=read(RUN/"RESULTS.json")
    rows=result["risks"]
    require(len(rows)==128,"expected 128 outputs")
    require(len({(r["d"],r["N"],r["stream"],r["method"]) for r in rows})==128,"duplicate output")
    for d in (16,64):
        for n in (256,4096):
            for method in METHODS:
                group=[r for r in rows if (r["d"],r["N"],r["method"])==(d,n,method)]
                require(sorted(r["stream"] for r in group)==list(range(8)),"missing stream")
                require(all(np.isfinite(r["joint_KL"]) and r["joint_KL"]>=0 for r in group),"invalid risk")
                require(all(len(r["w"])==2 and all(.5-1e-12<=x<=2.5+1e-12 for x in r["w"]) for r in group),"invalid loading")
    require(result["continuous_procedures"]==96 and result["MOM_outputs"]==32,"output accounting")
    files={rel(p):sha(p) for p in paths+[RUN/"RESULTS_FREEZE.json"]}
    files.update(freeze["files"])
    return result,files

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.titlesize":13,
      "axes.labelsize":12,"xtick.labelsize":11,"ytick.labelsize":10,
      "legend.fontsize":10,"svg.fonttype":"none","savefig.dpi":220,
      "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.8})

def export(fig,name):
    for extension in ("png","svg"):
        fig.savefig(OUT/f"{name}.{extension}",facecolor="white",metadata={"Creator":"compensation_learning_figures.py"} if extension=="svg" else None)
    plt.close(fig)

def risk_plot(result):
    rows=result["risks"];fig,axs=plt.subplots(2,2,figsize=(10.8,8.0),sharey=True)
    positive=[r["joint_KL"] for r in rows if r["joint_KL"]>0]
    require(bool(positive),"all risks zero; revise display transparently")
    ymin=10**np.floor(np.log10(min(positive))-.22);ymax=10**np.ceil(np.log10(max(positive))+.08)
    zeros=sum(r["joint_KL"]==0 for r in rows)
    for row,d in enumerate((16,64)):
        for col,n in enumerate((256,4096)):
            ax=axs[row,col]
            for x,method in enumerate(METHODS):
                a=sorted([r for r in rows if (r["d"],r["N"],r["method"])==(d,n,method)],key=lambda r:r["stream"])
                vals=np.array([r["joint_KL"] for r in a]);jitter=np.linspace(-.16,.16,8)
                shown=np.where(vals>0,vals,ymin*1.1)
                ax.scatter(x+jitter,shown,color=COLORS[method],marker=MARKERS[method],s=30,alpha=.63,zorder=3,linewidths=.4)
                for j,r in enumerate(a):
                    if not r["optimization_qualified"]:
                        ax.scatter(x+jitter[j],shown[j],marker="x",color="black",s=52,zorder=6,linewidths=1.1)
                mean=float(vals.mean());ax.scatter(x,max(mean,ymin*1.1),s=100,marker="D",facecolor=COLORS[method],edgecolor="black",linewidth=.8,zorder=5)
                ax.plot([x-.25,x+.25],[max(mean,ymin*1.1)]*2,color=COLORS[method],linewidth=1.7,zorder=4)
            ax.set_yscale("log");ax.set_ylim(ymin,ymax);ax.set_xlim(-.48,3.48)
            ax.set_xticks(range(4),[LABELS[m] for m in METHODS]);ax.set_title(f"d = {d}, N = {n:,}",loc="left",pad=9)
            ax.grid(axis="y",which="major",color="#d8d8d8",linewidth=.65)
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=7));ax.yaxis.set_minor_formatter(NullFormatter())
            if col==0:ax.set_ylabel("Full joint KL (nats)")
            if n==4096:
                gates=[r for r in result["risk_contrasts"] if r["d"]==d and r["N"]==n and "main_gate" in r]
                ax.text(.02,.025,f"Original main risk gates: {sum(r['main_gate'] for r in gates)}/{len(gates)} met",transform=ax.transAxes,fontsize=9,color="#444444",bbox={"facecolor":"white","edgecolor":"none","alpha":.85,"pad":2})
    fig.suptitle("Learning within the same fixed-floor candidate class",fontsize=16,x=.08,ha="left",y=.974)
    fig.text(.08,.933,"Eight streams per cell; dots are outputs and diamonds are arithmetic means (not CIs).",fontsize=11)
    qualifier=f"Numerical reference PMFs; full teacher-to-candidate KL, not a strict integration certificate.\nRaw and collapsed use K = d and shared banks; equal K does not imply equal computational cost."
    if any(not r["optimization_qualified"] for r in rows):qualifier+=" Black crosses retain optimization-unqualified outputs."
    if zeros:qualifier+=f" {zeros} zero estimates shown at the lower plotting boundary; no risk floor applied."
    fig.text(.08,.034,qualifier,fontsize=9.2,linespacing=1.45,color="#333333")
    fig.subplots_adjust(left=.09,right=.98,bottom=.145,top=.865,hspace=.34,wspace=.15)
    export(fig,FIGURES[0])

def loading_plot(result):
    fig,axs=plt.subplots(1,2,figsize=(10.8,5.9),sharex=True,sharey=True)
    for ax,d in zip(axs,(16,64)):
        for method in METHODS:
            rows=sorted([r for r in result["risks"] if (r["d"],r["N"],r["method"])==(d,4096,method)],key=lambda r:r["stream"])
            w=np.array([r["w"] for r in rows])
            ax.scatter(w[:,0],w[:,1],s=42,marker=MARKERS[method],color=COLORS[method],alpha=.62,linewidths=.4,zorder=3)
            ax.scatter(*w.mean(axis=0),s=135,marker=MARKERS[method],facecolor=COLORS[method],edgecolor="black",linewidth=1.3,zorder=6)
            for r in rows:
                if not r["optimization_qualified"]:ax.scatter(*r["w"],marker="x",color="black",s=64,zorder=7,linewidths=1.2)
        ax.set_xlim(.42,2.58);ax.set_ylim(.42,2.58);ax.set_aspect("equal")
        ax.set_xticks([.5,1,1.5,2,2.5]);ax.set_yticks([.5,1,1.5,2,2.5])
        ax.set_xlabel(r"Group A normalized loading $w_A$");ax.set_title(f"d = {d}, N = 4,096",loc="left",pad=9)
        ax.grid(color="#dfdfdf",linewidth=.6,zorder=0)
        ax.axvline(.5,color="#888888",linestyle=":",linewidth=.8);ax.axvline(2.5,color="#888888",linestyle=":",linewidth=.8)
        ax.axhline(.5,color="#888888",linestyle=":",linewidth=.8);ax.axhline(2.5,color="#888888",linestyle=":",linewidth=.8)
    axs[0].set_ylabel(r"Group B normalized loading $w_B$")
    handles=[Line2D([],[],marker=MARKERS[m],color="none",markerfacecolor=COLORS[m],markeredgecolor=COLORS[m],markersize=8,label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.854),ncol=4,frameon=False,columnspacing=2.2)
    fig.suptitle("Learned loadings after observing the same training labels",fontsize=16,x=.08,ha="left",y=.978)
    fig.text(.08,.928,"Small markers: eight streams. Large outlined markers: means; no confidence regions.",fontsize=11)
    fig.text(.08,.035,"Known two groups, zero means, floor a = 1; all methods use the box [0.5, 2.5] squared.\nRaw and collapsed share K = d and B = 64; equal particle counts are not an equal-cost claim.\nParameter scatter is descriptive; full observed-distribution KL is the primary comparison.",fontsize=9.2,linespacing=1.42,color="#333333")
    fig.subplots_adjust(left=.09,right=.98,bottom=.205,top=.766,wspace=.18)
    export(fig,FIGURES[1])

def build():
    require(not (OUT/"FIGURE_RECEIPT.json").exists(),"figures already sealed")
    result,files=inputs();OUT.mkdir(parents=True,exist_ok=True);style()
    risk_plot(result);loading_plot(result)
    summaries=[]
    for d in (16,64):
        for n in (256,4096):
            for method in METHODS:
                a=sorted([r for r in result["risks"] if (r["d"],r["N"],r["method"])==(d,n,method)],key=lambda r:r["stream"])
                summaries.append({"d":d,"N":n,"method":method,"streams":[r["stream"] for r in a],"joint_KL":[r["joint_KL"] for r in a],"joint_KL_mean":float(np.mean([r["joint_KL"] for r in a])),"w":[r["w"] for r in a],"w_mean":np.mean([r["w"] for r in a],axis=0).tolist(),"optimization_qualified":[r["optimization_qualified"] for r in a]})
    save(OUT/"PLOT_DATA.json",{"source_result_sha256":sha(RUN/"RESULTS.json"),"groups":summaries,"original_risk_main_gates":result["risk_main_gates"],"original_score_main_gates":result["score_main_gates"],"original_main_contrasts":[r for r in result["risk_contrasts"]+result["score_contrasts"] if "main_gate" in r]})
    outputs=[OUT/(name+"."+extension) for name in FIGURES for extension in ("png","svg")]+[OUT/"PLOT_DATA.json"]
    save(OUT/"BUILD.json",{"utc":now(),"source_sha256":sha(__file__),"inputs":files,"outputs":{rel(p):sha(p) for p in outputs},"visual_QA_completed":False,"figures":list(FIGURES),"new_samples":0,"new_fits":0,"new_integrals":0,"numpy":np.__version__,"matplotlib":matplotlib.__version__,"python":sys.version,"styles_are_descriptive_not_inferential":True})
    print("Built two figures. Inspect both PNGs before seal --visual-qa.")

def seal(visual_qa):
    require(visual_qa,"explicit completed visual QA required")
    require(not (OUT/"FIGURE_RECEIPT.json").exists(),"figures already sealed")
    _,files=inputs();b=read(OUT/"BUILD.json")
    require(b["source_sha256"]==sha(__file__),"figure source changed after build")
    require(b["inputs"]==files,"inputs changed after build");verify(b["outputs"])
    save(OUT/"FIGURE_RECEIPT.json",{**b,"utc":now(),"visual_QA_completed":True,"visual_QA_files":[name+".png" for name in FIGURES],"build_sha256":sha(OUT/"BUILD.json"),"no_scientific_results_replaced":True})
    print("Figures sealed after visual QA.")

def check():
    r=read(OUT/"FIGURE_RECEIPT.json")
    require(r["source_sha256"]==sha(__file__),"figure source changed")
    require(r["build_sha256"]==sha(OUT/"BUILD.json"),"build receipt changed")
    require(r["visual_QA_completed"],"visual QA missing")
    verify(r["inputs"]);verify(r["outputs"])
    require(read(RUN/"STATE.json")["status"]=="complete","RUN completion changed")
    print(json.dumps({"passed":True,"read_only":True,"rendering_or_science_reexecuted":False,"receipt_sha256":sha(OUT/"FIGURE_RECEIPT.json")}))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("build","seal","check"));p.add_argument("--visual-qa",action="store_true");args=p.parse_args()
    if args.action=="build":build()
    elif args.action=="seal":seal(args.visual_qa)
    else:check()

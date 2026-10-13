"""Prespecified structure figures from completed MC/TRAIN packages only.

No partial records are inspected to choose panels. MC uses its existing 12-test
sampling intervals; training plots are descriptive and have no recycled MC CI.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/nuisance_structure"
DEST=OUT/"figures"
FAMILIES=["diffuse","boundedhetero","concentrated"]
COLORS=dict(diffuse="#B4414A",boundedhetero="#346FA5",concentrated="#D68E26",marg="#178A77",exact="#30383F")
LABELS=dict(diffuse="Diffuse",boundedhetero="Bounded heterogeneity",concentrated="Concentrated",marg="Marginalized",exact="MLE reference")
SHORT=dict(diffuse="Diffuse",boundedhetero="Bounded\nheterogeneity",concentrated="Concentrated",marg="Marg.",exact="MLE")
METHODS=FAMILIES+["marg","exact"]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def style():
    assert not (OUT/"VERSION_LOCK.json").exists(),"Research package already sealed"
    DEST.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.labelsize":9,
        "axes.titlesize":9.5,"xtick.labelsize":8,"ytick.labelsize":8,
        "legend.fontsize":8,"axes.spines.top":False,"axes.spines.right":False,
        "axes.linewidth":.7,"grid.alpha":.18,"svg.hashsalt":"wstructure-fixed-panels",
        "savefig.facecolor":"white"})

def export(fig,name):
    result={}
    for suffix in ("png","svg"):
        path=DEST/f"{name}.{suffix}"
        assert not path.exists(),"Do not overwrite a completed plot without an explicit revision record"
        fig.savefig(path,dpi=230,metadata={"Date":None} if suffix=="svg" else None)
        result[rel(path)]=sha(path)
    plt.close(fig)
    return result

def mc():
    inputs=[OUT/"MC/RESULTS.json",OUT/"MC/FREEZE.json",OUT/"REFERENCE/MODEL.json",
            ROOT/"docs/08_research_研究工作/无关权重结构与方差匹配_20261002/MC_DESIGN.md"]
    bindings={rel(p):sha(p) for p in inputs}
    data=read(inputs[0]);reference=read(inputs[2])
    assert len(data["records"])==16 and data["accounting"]["independent_banks"]==65536
    assert len(data["primary_family"])==12 and data["config"]["primary_family_size"]==12
    assert data["freeze_sha256"]==sha(inputs[1]) and data["reference_sha256"]==sha(inputs[2])
    assert reference["qualification_passed"]
    style()
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.72),gridspec_kw={"width_ratios":[.9,1.05,1.15]})
    n,K=254,256
    case_indices={f:next(i for i,c in enumerate(data["cases"]) if c==dict(n=n,K=K,family=f)) for f in FAMILIES}
    variances=[];weight_points=[]
    for f in FAMILIES:
        profile=next(p for p in reference["profiles"] if p["n"]==n and p["family"]==f)
        variances.append(profile["total_variance"])
        ci=case_indices[f]
        ess=[r["means"]["nuisance_ESS"][ci]/K for r in data["records"]]
        maximum=[r["means"]["nuisance_max_weight"][ci] for r in data["records"]]
        weight_points.append(dict(family=f,mean_ESS_over_K=float(np.mean(ess)),mean_max_weight=float(np.mean(maximum)),
            group_ESS_over_K=ess,group_max_weight=maximum))
    xs=np.arange(3)
    axes[0].bar(xs,variances,color=[COLORS[f] for f in FAMILIES],width=.55,alpha=.86)
    axes[0].set(title="(a) Matched total log variance",ylabel=r"Reference $\sum_i v(a_i)$",
                xticks=xs,xticklabels=[SHORT[f] for f in FAMILIES],ylim=(0,max(variances)*1.16))
    axes[0].grid(axis="y")
    for x,v in zip(xs,variances):axes[0].text(x,v*1.018,f"{v:.3f}",ha="center",va="bottom",fontsize=8)
    for i,row in enumerate(weight_points):
        color=COLORS[row["family"]]
        axes[1].scatter(i-.095,row["mean_ESS_over_K"],color=color,marker="o",s=35)
        axes[1].scatter(i+.095,row["mean_max_weight"],color=color,marker="x",s=40,linewidths=1.5)
    axes[1].set(title="(b) Nuisance-weight concentration",ylabel="Mean fraction / weight",
                xticks=xs,xticklabels=[SHORT[f] for f in FAMILIES],ylim=(-.03,1.06))
    axes[1].legend(handles=[Line2D([0],[0],marker="o",ls="",color="#555555",label="ESS / K"),
        Line2D([0],[0],marker="x",ls="",color="#555555",label="Maximum weight")],loc="upper center",fontsize=7.5,frameon=False)
    axes[1].grid(axis="y")
    gap_points=[]
    for fi,f in enumerate(FAMILIES):
        for ti,theta in enumerate((0.,.6)):
            row=next(p for p in data["primary_family"] if p["n"]==n and p["K"]==K and p["family"]==f and p["theta"]==theta)
            axes[2].errorbar(ti+(fi-1)*.11,row["mean"],yerr=[[row["mean"]-row["lower"]],[row["upper"]-row["mean"]]],
                color=COLORS[f],marker="o",ls="",ms=4,capsize=3,lw=1.2)
            gap_points.append(dict(family=f,theta=theta,mean=row["mean"],lower=row["lower"],upper=row["upper"],
                M=row["M"],interval="Original simultaneous sampling interval from the full twelve-comparison MC family"))
    axes[2].axhline(0,color="#555555",ls=":",lw=.8)
    axes[2].set(title="(c) Raw-score preference",ylabel="Price-integrated endpoint gap (nats)",
                xticks=[0,1],xticklabels=["0","0.6"],xlabel=r"Teacher correlation $\theta$",xlim=(-.3,1.3))
    axes[2].grid(axis="y")
    handles=[Line2D([0],[0],marker="o",color=COLORS[f],label=LABELS[f],lw=1.4) for f in FAMILIES]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.065,right=.985,top=.88,bottom=.25,wspace=.43)
    outputs=export(fig,"variance_weights_gap")
    points=dict(n=n,K=K,families=FAMILIES,reference_total_variance=variances,
        nuisance_weight_means=weight_points,primary_gap_points=gap_points,
        scope="Panels fixed before results. Weight means descriptive; MC gaps use per-bank M=65536 bounds from the original twelve-test family. Theta=.6 uses the frozen numerical teacher weights.")
    save(OUT/"FIGURES_MC_POINTS.json",points)
    for p,h in bindings.items():assert sha(ROOT/p)==h
    save(OUT/"FIGURES_MC.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        inputs=bindings,outputs=outputs,points_sha256=sha(OUT/"FIGURES_MC_POINTS.json"),
        independent_MC_banks=65536,original_primary_family_size=12,displayed_primary_intervals=6,
        new_significance_gates=0,new_RNG=0,new_model_integrals=0,new_training_fits=0,
        visual_review_pending=True,scientific_unit="MC banks; not training datasets"))
    print(json.dumps(dict(completed=True,phase="mc",outputs=outputs)),flush=True)

def train():
    inputs=[OUT/"TRAIN/RESULTS.json",OUT/"TRAIN/FREEZE.json",OUT/"TRAIN/STATE.json",
            ROOT/"docs/08_research_研究工作/无关权重结构与方差匹配_20261002/TRAIN_DESIGN.md"]
    bindings={rel(p):sha(p) for p in inputs}
    data=read(inputs[0]);freeze=read(inputs[1]);state=read(inputs[2])
    assert state["status"]=="complete" and freeze["trigger"]["passed"]
    assert len(data["records"])==16 and data["outputs"]==160 and data["independent_streams"]==16
    assert freeze["config"]["teachers"]==[0.,.6]
    index={(r["stream"],r["theta"],r["method"]):r for rec in data["records"] for r in rec["outputs"]}
    assert len(index)==160
    style();fig,axes=plt.subplots(1,3,figsize=(10.5,3.72),gridspec_kw={"width_ratios":[1,1,1.12]})
    selected=[];risks=[]
    training_short=dict(diffuse="Diffuse",boundedhetero="Bounded",concentrated="Concentr.",marg="Marg.",exact="MLE")
    for ti,theta in enumerate((0.,.6)):
        ax=axes[ti]
        for s in range(16):
            yy=[index[s,theta,m]["rho"] for m in METHODS]
            xx=np.arange(5)+(s-7.5)*.006
            ax.plot(xx,yy,color="#A6B0B8",alpha=.3,lw=.6,zorder=1)
            for x,y,m in zip(xx,yy,METHODS):ax.scatter(x,y,color=COLORS[m],s=13,zorder=3,edgecolor="white",linewidth=.25)
            selected.append(dict(stream=s,theta=theta,methods=METHODS,rho=yy))
        ax.axhline(theta,color="#555555",lw=.85,ls=":")
        ax.set(title=f"({'ab'[ti]}) Paired estimates, theta = {theta:g}",ylabel=r"Estimated correlation $\hat\rho$",
            xticks=np.arange(5),xticklabels=[training_short[m] for m in METHODS],ylim=(-.82,.83),yticks=[-.75,0,.75])
        ax.grid(axis="y")
    for mi,m in enumerate(METHODS):
        for ti,theta in enumerate((0.,.6)):
            yy=[index[s,theta,m]["reference_joint_KL"] for s in range(16)]
            average=float(np.mean(yy));assert average>0,"If a zero mean occurs, preserve it and request a layout revision; never drop the cell"
            axes[2].scatter(ti+(mi-2)*.075,average,color=COLORS[m],marker="x" if m=="exact" else "o",s=30,linewidth=1.2)
            risks.append(dict(method=m,theta=theta,mean_reference_joint_KL=average,individual_risks=yy))
    axes[2].set(title="(c) Training risk",yscale="log",ylabel="Mean reference joint KL (nats)",
        xticks=[0,1],xticklabels=["0","0.6"],xlabel=r"Teacher correlation $\theta$",xlim=(-.35,1.35))
    axes[2].grid(axis="y",which="both")
    handles=[Line2D([0],[0],marker="x" if m=="exact" else "o",ls="",color=COLORS[m],label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
    fig.subplots_adjust(left=.065,right=.985,top=.88,bottom=.27,wspace=.43)
    outputs=export(fig,"training_structure")
    save(OUT/"FIGURES_TRAIN_POINTS.json",dict(n=254,K=256,B=128,N=4096,paired_estimates=selected,risks=risks,
        horizontal_offsets="Fixed stream/method index offsets for overlap only; no random jitter",
        scope="All sixteen independent fresh training flows displayed; means are descriptive and no MC interval is transferred to training."))
    for p,h in bindings.items():assert sha(ROOT/p)==h
    save(OUT/"FIGURES_TRAIN.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        inputs=bindings,outputs=outputs,points_sha256=sha(OUT/"FIGURES_TRAIN_POINTS.json"),
        independent_training_flows=16,displayed_methods=5,displayed_teachers=2,
        new_significance_gates=0,new_RNG=0,new_model_integrals=0,new_training_fits=0,
        visual_review_pending=True,scientific_unit="Paired training flows, distinct from MC bank confidence intervals"))
    print(json.dumps(dict(completed=True,phase="train",outputs=outputs)),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["mc","train"])
    globals()[parser.parse_args().phase]()

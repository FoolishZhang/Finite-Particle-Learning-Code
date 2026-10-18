# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Figure-source projection: same saved points and drawing; PNG only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot(data,folder):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,
        "axes.titlesize":8.8,"axes.labelsize":8.5,"xtick.labelsize":7.5,"ytick.labelsize":8,"pdf.fonttype":42,"savefig.facecolor":"white"})
    colors={"F":"#5268a5","L":"#cf7c2a","A":"#007c75"};markers={"F":"o","L":"+","A":"^"}
    fig,axes=plt.subplots(1,3,figsize=(7.25,2.75),sharey=True)
    for ax,panel in zip(axes,data["panels"]):
        amplitudes=sorted(set(r["t"] for r in panel["points"]));end=max(amplitudes);start=min(amplitudes)
        for theory in panel["theory"]:
            if theory["family"]=="L":continue
            ax.hlines(theory["response"],start*.6,end*1.05,color="#747b83",lw=1.05,linestyle=(0,(4,2)),zorder=1)
        for family in ("F","L","A"):
            rows=sorted((r for r in panel["points"] if r["family"]==family),key=lambda r:r["t"])
            for r in rows:
                ax.plot(r["t"],r["response"],markers[family],color=colors[family],ms=6 if family=="F" else 5,
                    markerfacecolor="white" if family=="F" or not r["ordinary_numerical_qualification"] else colors[family],
                    markeredgewidth=1.1,zorder=3 if family=="F" else 4)
        ax.set_title(panel["title"],pad=7);ax.set(xlim=(0,end*1.12),ylim=(.21,.65),xlabel="Teacher amplitude t",xticks=amplitudes)
        ax.set_xticklabels([format(t,"g").lstrip("0") for t in amplitudes]);ax.grid(axis="y",color="#e7e9ec",lw=.5)
    axes[0].set_ylabel(r"Response $\theta_{ij}/\eta_{ij}$")
    handles=[Line2D([],[],color="#747b83",ls="--",lw=1.05,label="Leading theory"),
        Line2D([],[],color=colors["F"],marker="o",mfc="white",ls="none",ms=6,label="F: no free tilt"),
        Line2D([],[],color=colors["L"],marker="+",ls="none",ms=6,label="L: local input"),
        Line2D([],[],color=colors["A"],marker="^",ls="none",ms=5,label="A: cross-group input")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,1.005),ncol=4,frameon=False,fontsize=7.8,columnspacing=1.2)
    fig.subplots_adjust(left=.085,right=.99,bottom=.17,top=.79,wspace=.19)
    for suffix in ("png",):
        target=folder/("adaptation_response."+suffix)
        if target.exists():raise FileExistsError("figure output already exists: "+str(target))
        fig.savefig(target,dpi=240,bbox_inches="tight")
    plt.close(fig)

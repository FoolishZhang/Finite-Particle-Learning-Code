"""Plot every prespecified symmetric panel from saved analysis, without science."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(analysis, out):
    path=analysis/"ANALYSIS.json"
    data=json.loads(path.read_text(encoding="utf-8"))
    cells=data["primary_cells"]
    expected=[(rep,n,t) for rep in range(3) for n in (16384,65536) for t in (.1,.2)]
    if [(c["rep"],c["N"],c["t"]) for c in cells]!=expected:
        raise ValueError("All12 prespecified cells in fixed order required")
    out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                        "axes.spines.top":False,"axes.spines.right":False,
                        "svg.fonttype":"none","svg.hashsalt":"learned-local-v1"})
    fig,axes=plt.subplots(2,1,figsize=(12,7.2),sharex=True,layout="constrained")
    labels=[f"r{c['rep']} / {c['N']}\nt={c['t']:.1f}" for c in cells]
    axes[1].axhspan(-.01,.01,color="#ddd7bc",alpha=.5,label="Predeclared prediction tolerance")
    axes[1].axhline(0,color="#6d7883",ls="--",lw=1)
    complete=0
    for i,c in enumerate(cells):
        if c["status"]!="available":
            axes[0].text(i,.5,"Unresolved",rotation=90,transform=axes[0].get_xaxis_transform(),ha="center")
            axes[1].text(i,.5,"Unresolved",rotation=90,transform=axes[1].get_xaxis_transform(),ha="center")
            continue
        complete+=1;t2=c["t"]**2
        axes[0].errorbar(i-.1,c["prediction"]["mean"]/t2,yerr=c["prediction"]["SE"]/t2,
                         fmt="D",ms=5,mfc="white",color="#c06b20",capsize=2,
                         label="Frozen local prediction ±1 SE" if complete==1 else None)
        axes[0].errorbar(i+.1,c["D"]["mean"]/t2,yerr=c["D"]["SE"]/t2,
                         fmt="o",ms=5,color="#16679a",capsize=2,
                         label="Independent candidate contrast ±1 SE" if complete==1 else None)
        r=c["normalized_residual"];ci=c["normalized_residual_simultaneous_interval"]
        axes[1].errorbar(i,r["mean"],yerr=[[r["mean"]-ci["lower"]],[ci["upper"]-r["mean"]]],
                         fmt="o",ms=5,color="#16679a",capsize=3,
                         label="Residual: approximate simultaneous 95% interval" if complete==1 else None)
    axes[0].set_title("A. Finite-objective contrasts: negative values favor the shrink candidate",loc="left")
    axes[0].set_ylabel(r"Contrast / $t^2$")
    axes[1].set_title("B. Independent contrast minus frozen local prediction",loc="left")
    axes[1].set_ylabel(r"Prediction residual / $t^2$")
    axes[1].set_xticks(np.arange(12),labels,fontsize=8)
    axes[1].set_xlabel("Training repeat / training sample count; teacher amplitude t")
    for ax in axes:
        ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True);ax.set_xlim(-.6,11.6)
        for boundary in (3.5,7.5):ax.axvline(boundary,color="#c3c8cd",lw=.8)
        ax.legend(frameon=False,fontsize=9,loc="best")
    fig.suptitle("Symmetric learned local proposals | K=8 | all 12 prespecified panels",fontsize=13)
    fig.savefig(out/"independent_prediction.png",dpi=220)
    fig.savefig(out/"independent_prediction.svg",metadata={"Date":None})
    plt.close(fig)
    receipt={"source_sha256":sha(Path(__file__)),"analysis_sha256":sha(path),
             "prespecified_panels":12,"available_panels":complete,"selected_panels_removed":False,
             "uncertainty":"Conditional numerical randomization after density selection; not training-population uncertainty",
             "top_error_bars":"1 numerical SE for prediction and independent candidate difference",
             "bottom_error_bars":"Approximate family24 Bonferroni95% t intervals,df15",
             "shaded_region":"Predeclared finite-objective prediction tolerance, not risk-equivalence claim",
             "no_model_calls":True,
             "files":{name:sha(out/name) for name in ("independent_prediction.png","independent_prediction.svg")}}
    (out/"FIGURE_RECEIPT.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();main(args.analysis,args.out)

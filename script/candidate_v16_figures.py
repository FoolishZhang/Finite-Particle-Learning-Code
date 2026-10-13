"""Publication figure from already sealed public scalars; no scientific replay."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/"docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_6_20261007"
OLD=DOC.parent/"submission_candidate_v1_5_1_20261006"
ART=ROOT/"artifacts/submission_candidate_v1_6_20261007"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text("utf-8"))

def main():
    assert not (ART/"DELIVERY_MANIFEST.json").exists() and not (ART/"VERSION_LOCK.json").exists()
    files=[ROOT/f"artifacts/joint_dependence_v1_20261007/rep_{i:02d}/ENDPOINT_QUALIFICATION.json" for i in range(4)]
    files+=[ROOT/"artifacts/rankone_continuation_v1_20261007/SUMMARY.json",
            ROOT/"artifacts/noncollinear_probe_v1_20261007/SUMMARY.json"]
    methods=("K8","K64","REF","IND")
    risks=np.array([[next(v["risk_KL"] for v in read(p)["endpoints"] if v["name"]==m)
                     for m in methods] for p in files[:4]])
    continuation=read(files[4]);probe=read(files[5])
    selected=[next(r for r in continuation["rows"] if r["K"]==K) for K in (8,11,12,64,"inf")]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.1,
        "axes.titlesize":8.4,"axes.labelsize":8.1,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
        "axes.spines.top":False,"axes.spines.right":False,"svg.hashsalt":"candidate-v16-main"})
    fig,axs=plt.subplots(2,2,figsize=(7.0,4.15),layout="constrained")
    ax=axs[0,0]
    for i,marker in enumerate(("o","s","^","D")):
        ax.plot(np.arange(4),risks[i],color="#a6a6a6",lw=.55,alpha=.55,zorder=1)
        ax.scatter(np.arange(4)+(i-1.5)*.045,risks[i],marker=marker,s=17,
                   color=("#2465a1","#e09535","#4f8c60","#9467a5")[i],zorder=3)
    ax.scatter(np.arange(4),risks.mean(axis=0),marker="_",s=160,color="black",lw=1.4,zorder=4)
    ax.set_yscale("log");ax.set_xticks(range(4),methods);ax.set_ylabel("Observation KL (log scale)")
    ax.set_title("(a) Four exploratory training streams",loc="left")
    ax.grid(axis="y",alpha=.16)
    ax=axs[0,1];x=np.arange(len(selected))
    ax.plot(x,[r["D_Hperp_transverse_min"] for r in selected],"o-",color="#355f82",lw=1,ms=3.5)
    ax.axhline(0,color="#777777",lw=.7)
    ax.set_xticks(x,["8","11","12","64",r"$\infty$"])
    ax.set_xlabel("Refitted K (categorical positions)")
    ax.set_ylabel(r"$\lambda_{\min}(U_a^\top H_\perp U_a)$")
    ax.set_title("(b) Transverse local type",loc="left");ax.grid(alpha=.16)
    for K,color in ((11,"#2863a0"),(12,"#b84c36")):
        rows=sorted((r for r in probe["point_rows"] if r["K"]==K),key=lambda r:r["t"])
        for ax,key,unit in [(axs[1,0],"paired_F",1e-6),(axs[1,1],"paired_R",1e-5)]:
            ax.plot([r["t"] for r in rows],[r["rules"]["B"][key]/unit for r in rows],
                    "o-",color=color,lw=1,ms=3,label=f"K{K}")
            ax.axhline(0,color="#777777",lw=.7)
            ax.set_xticks([-.1,-.05,0,.05,.1]);ax.set_xlabel("Norm-preserving angle amplitude t")
            ax.grid(alpha=.16)
    axs[1,0].set_title("(c) Finite objective change",loc="left")
    axs[1,0].set_ylabel(r"$\Delta F_K\ (10^{-6})$")
    axs[1,1].set_title("(d) Observation risk change",loc="left")
    axs[1,1].set_ylabel(r"$\Delta\mathcal{R}\ (10^{-5})$")
    axs[1,0].legend(frameon=False,fontsize=7,loc="upper center",ncol=2)
    for suffix in ("png","svg"):
        fig.savefig(DOC/"figures"/f"joint_mechanism.{suffix}",dpi=300,
                    metadata={"Date":None} if suffix=="svg" else None)
    plt.close(fig)
    old=[p for p in (OLD/"figures").glob("*") if p.is_file()]
    for p in old:assert sha(p)==sha(DOC/"figures"/p.name),p
    outputs={rel(p):sha(p) for p in sorted((DOC/"figures").glob("*")) if p.is_file()}
    record=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(Path(__file__)),
        kind="New main synthesis from sealed public data; all inherited figures copied unchanged",
        sources={rel(p):sha(p) for p in [*files,*old]},
        outputs=outputs,old_figure_bytes_preserved=True,new_scientific_computation=False,
        main_figures=["finite_instance","joint_mechanism"],
        supplementary_figures=["representation_structure","overlap_coupling","preference_geometry",
                               "learning_risk","readout_and_selection"],
        plotted_data=dict(methods=list(methods),risks=risks.tolist(),mean_risks=risks.mean(axis=0).tolist(),
                         selected_refit_K=[r["K"] for r in selected],
                         transverse_minima=[r["D_Hperp_transverse_min"] for r in selected]),
        no_new_draws_fits_integrals=True,no_confidence_intervals_created=True)
    (ART/"FIGURES.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(dict(created=True,mean_risks=risks.mean(axis=0).tolist(),
                          new_scientific_computation=False,outputs=len(outputs))))

if __name__=="__main__":main()

"""Public JSON-only finite-path reductions and figures; no new scientific calls."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/noncollinear_probe_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/NoncollinearProbe_v1_20261007"

def read(p):return json.loads(Path(p).read_text("utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def save(p,x):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def reduce():
    paths=[OUT/"RESULTS.json",OUT/"AUDIT.json",OUT/"SOURCE.json",OUT/"PANEL_LOCK.json"]
    r=read(paths[0]);audit=read(paths[1]);rows=[];baseline_rows=[]
    maxima={k:0. for k in ("F_BC","R_BC","F_AC","R_AC","even_F","even_R","even_PMF",
        "marginal_change","row_norm_error","paired_F_interface","paired_R_interface",
        "baseline_old_1D_loss","baseline_old_1D_PMF")}
    for anchor in r["anchors"]:
        paths.append(ROOT/anchor["result_artifact"]["path"])
        for rule,b in anchor["baselines"].items():
            comp=b["qualification"]["comparisons"]
            maxima["baseline_old_1D_loss"]=max(maxima["baseline_old_1D_loss"],comp["old_1D_D_loss_max_difference"])
            maxima["baseline_old_1D_PMF"]=max(maxima["baseline_old_1D_PMF"],comp["old_1D_D_PMF_max_difference"])
            baseline_rows.append(dict(anchor=anchor["name"],rule=rule,
                qualified=b["qualification"]["qualified"],comparisons=comp))
        for point in anchor["points"]:
            t=point["t"];rules=point["rules"];B,C=rules["B"],rules["C"]
            row=dict(anchor=anchor["name"],K=anchor["K"],point_id=point["point_id"],t=t,
                baseline=t==0,old_curvature=anchor["curvature"]["D"],
                old_risk_curvature=anchor["risk_curvature"]["D"],
                rules={rule:{k:record[k] for k in ("paired_F","paired_R","quadratic_F","quadratic_R","raw_KL","absolute_loss")}
                       for rule,record in rules.items()},
                singular_values=B["geometry"]["singular_values"])
            if t==0:
                row.update(qualified=point["qualified"],F_sign="zero_by_definition",
                           R_sign="zero_by_definition",F_resolved=False,R_resolved=False)
            else:
                q=point["qualification"]
                row.update(qualified=q["qualified"],qualified_F=q["qualified_F"],qualified_R=q["qualified_R"],
                    F_sign=q["effects"]["F"]["sign"]["sign"],R_sign=q["effects"]["R"]["sign"]["sign"],
                    F_resolved=q["effects"]["F"]["sign"]["resolved"],R_resolved=q["effects"]["R"]["sign"]["resolved"],
                    shared_failed_gates=q["shared_failed_gates"],
                    effect_failed_gates={key:value["failed_gates"] for key,value in q["effects"].items()},
                    F_sign_floor=q["effects"]["F"]["sign"]["required_minimum_magnitude"],
                    R_sign_floor=q["effects"]["R"]["sign"]["required_minimum_magnitude"],
                    B_quadratic_F_minus_old=B["quadratic_F"]-anchor["curvature"]["D"],
                    B_quadratic_R_minus_old=B["quadratic_R"]-anchor["risk_curvature"]["D"])
                for effect in ("F","R"):
                    for pair in ("BC","AC"):
                        maxima[effect+"_"+pair]=max(maxima[effect+"_"+pair],q["effects"][effect]["precision"][pair]["difference"])
                for rule,record in rules.items():
                    ev=q["evenness"][rule]["comparisons"]
                    for field,key in (("F_difference","even_F"),("R_difference","even_R"),("PMF_max_difference","even_PMF")):
                        maxima[key]=max(maxima[key],ev[field])
                    maxima["marginal_change"]=max(maxima["marginal_change"],record["marginal_max_difference"])
                    maxima["row_norm_error"]=max(maxima["row_norm_error"],record["geometry"]["row_norm_max_error"])
                    maxima["paired_F_interface"]=max(maxima["paired_F_interface"],max(abs(x) for x in record["delta_consistency_error"]))
                    maxima["paired_R_interface"]=max(maxima["paired_R_interface"],abs(record["risk_consistency_error"]))
            rows.append(row)
    return dict(source_sha256=sha(Path(__file__)),sources={rel(p):sha(p) for p in paths},
        baseline="94fec1e1a2166ed7342752ab8c3bd43e39951714",
        primary_outcome=r["outcome"]["primary"]["status"],outcome=r["outcome"],
        point_rows=rows,baseline_rows=baseline_rows,nonzero_points=sum(not p["baseline"] for p in rows),
        qualified_nonzero_points=sum(p["qualified"] for p in rows if not p["baseline"]),
        qualified_baseline_rules=sum(p["qualified"] for p in baseline_rows),precision_maxima=maxima,
        resources=r["resources"],scientific_seconds=r["seconds"],audit_passed=audit["passed"],
        route_closed=r["scope"]["current_local_mechanism_diagnosis_closed_after_fixed_panel"],
        no_new_report_quadrature=True,manuscript_modified=False,third_tier_started=False,
        two_dimensional_optimization_started=False,new_transfer_prompt_created=False)

def draw(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.hashsalt":"noncollinear-probe-v1"})
    fig,axs=plt.subplots(1,3,figsize=(13,3.9),layout="constrained")
    for K,color in ((11,"#2863a0"),(12,"#b84c36")):
        rows=sorted((r for r in s["point_rows"] if r["K"]==K),key=lambda r:r["t"])
        for ax,key,title in zip(axs[:2],("paired_F","paired_R"),
                                ("Direct finite-objective change","Direct observation-risk change")):
            ax.plot([r["t"] for r in rows],[r["rules"]["B"][key] for r in rows],"o-",
                    color=color,linewidth=1.3,markersize=4,label=f"K={K}, rule B")
            ax.plot([r["t"] for r in rows],[r["rules"]["C"][key] for r in rows],"x",
                    color=color,markersize=6,alpha=.65)
            ax.set_ylabel(title);ax.set_xlabel("Fixed angle amplitude t")
            ax.axhline(0,color="#888888",lw=.7);ax.grid(alpha=.17)
            ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        positives=[r for r in rows if r["t"]>0]
        axs[2].plot([r["t"] for r in positives],[r["rules"]["B"]["quadratic_F"] for r in positives],
                    "o-",color=color,markersize=4,label=f"K={K}: 2 delta F / t^2")
        axs[2].axhline(rows[0]["old_curvature"],color=color,linestyle="--",linewidth=1)
    axs[0].legend(frameon=False,fontsize=8)
    axs[2].set_xlabel("Positive amplitudes (negative side in tables)")
    axs[2].set_ylabel("Finite quadratic readout vs saved curvature")
    axs[2].axhline(0,color="#888888",lw=.7);axs[2].grid(alpha=.17)
    axs[2].legend(frameon=False,fontsize=8)
    fig.suptitle("Matched norm-preserving paths: fixed panel, direct paired integrals",fontsize=12)
    outputs={}
    for suffix in ("png","svg"):
        p=DOC/f"finite_paths.{suffix}";assert not p.exists()
        fig.savefig(p,dpi=180,metadata={"Date":None} if suffix=="svg" else None)
        outputs[rel(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    plt.close(fig);return outputs

def main(build):
    current=reduce()
    if build:
        assert not (OUT/"SUMMARY.json").exists()
        fig=draw(current)
        save(OUT/"SUMMARY.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),**current,figure_outputs=fig))
    else:
        saved=read(OUT/"SUMMARY.json");assert all(saved[k]==v for k,v in current.items())
        for name,entry in saved["figure_outputs"].items():assert sha(ROOT/name)==entry["sha256"]
    print(json.dumps(dict(passed=True,mode="build" if build else "check",
        primary_outcome=current["primary_outcome"],nonzero_points=current["nonzero_points"],
        qualified_nonzero_points=current["qualified_nonzero_points"],route_closed=current["route_closed"],
        new_quadrature_calls=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("build","check"))
    main(parser.parse_args().mode=="build")

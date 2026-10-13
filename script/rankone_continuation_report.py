"""JSON-only continuation summary and sampled-point figure; no new integration."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/rankone_continuation_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/RankOneContinuation_v1_20261007"

def read(p):return json.loads(Path(p).read_text("utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def save(p,value):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def reduce():
    result_path=OUT/"RESULTS.json";audit_path=OUT/"AUDIT.json"
    r=read(result_path);audit=read(audit_path)
    paths=[result_path,audit_path,OUT/"INITIAL_K8.json",OUT/"ENDPOINTS_LOCK.json",
           OUT/"BRACKET_FINAL.json",OUT/"SOURCE.json"]
    points=[r["initial"],*r["points"]]
    points.sort(key=lambda p:float("inf") if p["K"]=="inf" else p["K"])
    assert len({p["K"] for p in points})==len(points)
    rows=[];max_diff={};end_blocks=qualified=0
    for p in points:
        theta=np.array(p["theta"]);own=p["own_blocks"]["D"];a=theta[8:]
        row=dict(point_id=p["point_id"],kind=p["kind"],K=p["K"],
            radial_qualified=p["radial_qualification"]["qualified"],transverse_type=p["transverse_label"],
            radial_failed_gates=p["radial_qualification"]["failed_gates"],
            own_numerical_qualified=p["own_qualification"]["qualified"],
            D_gradient_Linf=p["D_gradient_Linf"],D_H_ba_min=p["D_H_ba_minimum"],
            D_Hperp_transverse_min=p["D_transverse_minimum"],
            D_Hangle_min=own["Hangle_projected_eigenvalues"][0],
            D_Hperp_negative_modes=int(np.sum(np.array(own["Hperp_projected_eigenvalues"]) < -1e-5)),
            loading_RMS=float(np.sqrt(np.mean(a*a))),loading_min=float(a.min()),loading_max=float(a.max()),
            bias_Linf=float(np.max(abs(theta[:8]))),D_observation_KL=p["probability"]["D"]["raw_KL"],
            C_observation_KL=p["probability"]["C"]["raw_KL"],own_loss_D=own["loss"],
            theta=p["theta"],new_optimization=p["kind"]!="reused_initial")
        if row["new_optimization"]:
            paths.append(ROOT/p["result_artifact"]["path"])
            paths.append(ROOT/p["solver_artifact"]["path"])
            solver=read(paths[-1])
            row.update(accepted_steps=solver["accepted_step_count"],optimizer_calls=solver["total_calls"],
                       stage_stop_reasons=p["stage_stop_reasons"])
            for q in p["evaluation"]["qualifications"]:
                end_blocks+=1;qualified+=q["qualified"]
                for key,value in q["comparisons"].items():max_diff[key]=max(max_diff.get(key,0.),value)
        else:row.update(accepted_steps=0,optimizer_calls=0,stage_stop_reasons=["reused_prior_qualified_P"])
        rows.append(row)
    pair=r["refinement"]["final_pair"]
    bracket=None if pair is None else dict(lower_K=pair["lower"]["K"],upper_K=pair["upper"]["K"],
        lower_point_id=pair["lower"]["point_id"],upper_point_id=pair["upper"]["point_id"],
        width=pair["upper"]["K"]-pair["lower"]["K"])
    cross=r["cross"];cross_summary=dict(status=cross["status"])
    if cross["status"]=="completed":
        paths.append(ROOT/cross["result_artifact"]["path"])
        cross_summary.update(lower_K=cross["lower_K"],upper_K=cross["upper_K"],
            all_four_qualified=cross["all_four_cells_numerically_qualified"],
            direction=cross["direction"],cells={k:{f:v[f] for f in
                ("K","theta_source","C_curvature","D_curvature","curvature_rule_difference")}
                for k,v in cross["cells"].items()},algebra=cross["algebra"])
    return dict(source_sha256=sha(Path(__file__)),sources={rel(p):sha(p) for p in paths},
        baseline="f8354584fd321f909c8319000485c89cb303188d",
        coarse_new_solves=r["coarse_solve_count"],refinement_new_solves=r["refinement_solve_count"],
        new_endpoints=r["new_endpoint_count"],reused_anchor_count=1,rows=rows,
        new_endpoint_blocks=end_blocks,qualified_new_endpoint_blocks=qualified,
        own_radial_qualified_new=sum(p["radial_qualification"]["qualified"] for p in r["points"]),
        new_transverse_counts={label:sum(p["transverse_label"]==label for p in r["points"])
                               for label in ("positive","negative","unresolved")},
        refinement_status=r["refinement"]["status"],final_bracket=bracket,
        refinement_decisions=[{k:d[k] for k in ("index","midpoint_K","action","before","after")}
                              for d in r["refinement"]["decisions"]],
        cross=cross_summary,maximum_C_D_new_endpoint_differences=max_diff,
        scientific_resources=r["resources"],scientific_seconds=r["seconds"],
        execution_audit_passed=audit["passed"],new_report_quadrature_calls=0,new_report_random_draws=0,
        exact_existence_certified=False,unique_or_monotone_branch_claimed=False,
        manuscript_modified=False,new_transfer_prompt_created=False)

def draw(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.hashsalt":"rankone-continuation-v1"})
    rows=s["rows"];x=np.arange(len(rows));labels=[str(p["K"]) for p in rows]
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout="constrained")
    fields=("D_Hperp_transverse_min","loading_RMS","D_observation_KL")
    names=("Minimum transverse curvature","Loading RMS after refitting","Observation KL")
    for ax,field,name in zip(axs,fields,names):
        values=[p[field] for p in rows]
        ax.plot(x,values,color="#355f82",linewidth=1,alpha=.7)
        for kind,marker,title in (("reused_initial","*","Saved K8"),("coarse","o","Fixed sequence"),("refinement","D","Integer refinement")):
            indices=[i for i,p in enumerate(rows) if p["kind"]==kind]
            if indices:ax.scatter(x[indices],[values[i] for i in indices],marker=marker,
                s=35 if kind!="reused_initial" else 85,color="#b34a32" if kind=="refinement" else "#355f82",
                label=title,zorder=3)
        ax.set_xticks(x,labels,rotation=45);ax.set_ylabel(name)
        ax.set_xlabel("K (categorical spacing)");ax.grid(alpha=.15)
    axs[0].axhline(0,color="#444444",lw=.8);axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Refitted population rank-one sequence; lines only connect evaluated points",fontsize=12)
    output={}
    for suffix in ("png","svg"):
        path=DOC/f"continuation.{suffix}";assert not path.exists()
        fig.savefig(path,dpi=180,metadata={"Date":None} if suffix=="svg" else None)
        output[rel(path)]=dict(sha256=sha(path),bytes=path.stat().st_size)
    plt.close(fig);return output

def main(build):
    current=reduce()
    if build:
        assert not (OUT/"SUMMARY.json").exists()
        figures=draw(current)
        save(OUT/"SUMMARY.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                     **current,figure_outputs=figures))
    else:
        saved=read(OUT/"SUMMARY.json")
        assert all(saved[k]==v for k,v in current.items())
        for name,binding in saved["figure_outputs"].items():assert sha(ROOT/name)==binding["sha256"]
    print(json.dumps(dict(passed=True,mode="build" if build else "check",
        new_endpoints=current["new_endpoints"],final_bracket=current["final_bracket"],
        endpoint_qualified=current["qualified_new_endpoint_blocks"],
        cross_status=current["cross"]["status"],new_quadrature_calls=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("build","check"))
    main(parser.parse_args().mode=="build")

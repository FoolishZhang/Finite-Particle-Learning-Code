"""Public JSON-only reductions and a fixed-probe figure, without new quadrature."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/rankone_stationary_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/RankOneStationary_v1_20261007"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text("utf-8"))
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def write(p,value):
    with Path(p).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")

def reduce():
    paths=[OUT/"RESULTS.json",OUT/"ENDPOINTS_LOCK.json",OUT/"SOURCE.json",OUT/"AUDIT.json"]
    result=read(paths[0]);lock=read(paths[1])
    rows=[];probe_rows=[];comparison_max={}
    def block(data,target,K):
        return next(r for r in data["blocks"] if r["target"]==target and r["K"]==K)
    for endpoint,locked in zip(result["endpoints"],lock["endpoints"]):
        name=endpoint["name"]
        files=[OUT/"evaluation"/f"{name}_{rule}.json" for rule in ("C","D")]
        paths.extend(files);D=read(files[1])
        solver_path=ROOT/locked["solver_artifact"]["path"];paths.append(solver_path)
        solver=read(solver_path);fit=block(D,endpoint["fit_target_name"],8)
        pop={str(k):block(D,"population",k) for k in (8,64,"inf")}
        initial=np.array(solver["initial_theta"]);theta=np.array(endpoint["theta"])
        rows.append(dict(name=name,fit_target=endpoint["fit_target_name"],
            signature=endpoint["fitted_signature"],theta=theta.tolist(),
            parameter_movement_Linf=float(np.max(abs(theta-initial))),
            accepted_steps=solver["accepted_step_count"],kernel_calls=solver["total_calls"],
            stages=[{k:s[k] for k in ("phase","call_count","accepted_step_count","stop_reason")} for s in solver["stages"]],
            F8_fit_loss=fit["loss"],F8_fit_gradient_Linf=fit["gradient_Linf"],
            F8_fit_Hba_min=fit["H_ba_eigenvalues"][0],
            F8_fit_Hperp_transverse_min=fit["Hperp_projected_eigenvalues"][0],
            F8_fit_Hangle_min=fit["Hangle_projected_eigenvalues"][0],
            population_angular_min={k:v["Hangle_projected_eigenvalues"][0] for k,v in pop.items()},
            population_losses={k:v["loss"] for k,v in pop.items()},
            integrated_gradient_L2=float(np.linalg.norm(pop["inf"]["gradient"])),
            observation_KL_D=D["probability"]["raw_KL"],
            observation_KL_C=read(files[0])["probability"]["raw_KL"],
            teacher_mass=D["probability"]["teacher_mass"],model_mass_D=D["probability"]["pmf_mass"]))
        for item in endpoint["qualifications"]:
            for key,value in item["comparisons"].items():
                comparison_max[key]=max(comparison_max.get(key,0.),float(value))
    planpath=ROOT/result["probe"]["plan_artifact"]["path"];paths.append(planpath)
    plan=read(planpath)
    for point in result["probe"]["points"]:
        row=dict(index=point["index"],t=point["t"],status=point["status"])
        if point["status"]=="evaluated":
            row["qualified"]=point["qualified"]
            for rule in ("C","D"):
                path=ROOT/point["rule_artifacts"][rule]["summary_artifact"]["path"]
                paths.append(path);data=read(path)
                row[rule]=dict(losses={str(k):block(data,"population",k)["loss"] for k in (8,64,"inf")},
                    KL=data["probability"]["raw_KL"])
        probe_rows.append(row)
    zero=next((p for p in probe_rows if p["t"]==0 and p["status"]=="evaluated"),None)
    if zero:
        for row in probe_rows:
            if row["status"]=="evaluated":
                for rule in ("C","D"):
                    row[rule]["delta_F8"]=row[rule]["losses"]["8"]-zero[rule]["losses"]["8"]
                    row[rule]["delta_KL"]=row[rule]["KL"]-zero[rule]["KL"]
    resources=dict(optimization=result["optimization_resources"],evaluation=result["evaluation_resources"])
    return dict(source_sha256=sha(Path(__file__)),sources={rel(p):sha(p) for p in paths},
        endpoint_blocks=result["endpoint_blocks"],endpoint_rule_blocks=result["endpoint_rule_blocks"],
        qualified_endpoint_blocks=result["qualified_endpoint_blocks"],
        fit_signatures_passed=result["fit_signatures_passed"],endpoint_rows=rows,
        maximum_C_D_endpoint_differences=comparison_max,
        probe_status=result["probe"]["status"],probe_direction=plan.get("direction"),
        P_integrated_gradient_L2=plan.get("gradient_L2"),probe_rows=probe_rows,
        qualified_probe_points=sum(p.get("qualified",False) for p in probe_rows),
        resources=resources,new_quadrature_calls=0,new_random_draws=0,
        numerical_stationary_signature_not_interval_existence=True,
        full_independent_optimizer_reexecution=False,manuscript_modified=False,
        new_transfer_prompt_created=False)

def figure(summary):
    rows=[p for p in summary["probe_rows"] if p["status"]=="evaluated" and "delta_F8" in p["D"]]
    if not rows:return {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.hashsalt":"rankone-stationary-v1"})
    fig,axs=plt.subplots(1,2,figsize=(10,3.9),layout="constrained")
    for ax,field,label,color in zip(axs,("delta_F8","delta_KL"),
                                  ("Finite objective change","Observation KL change"),("#2863a0","#b64c37")):
        for rule,style in (("D","o-"),("C","--")):
            ax.plot([p["t"] for p in rows],[p[rule][field] for p in rows],style,
                    color=color if rule=="D" else "#333333",label=f"Rule {rule}",linewidth=1.5,
                    markersize=5,alpha=.85)
        ax.axhline(0,color="#888888",lw=.7);ax.axvline(0,color="#cccccc",lw=.7)
        ax.set_xlabel("Step t along fixed negative risk gradient")
        ax.set_ylabel(label);ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ax.grid(alpha=.16);ax.legend(frameon=False)
    fig.suptitle("Locked population F8 endpoint: fixed bias / row-length probe",fontsize=12)
    output={}
    for suffix in ("png","svg"):
        path=DOC/f"population_probe.{suffix}"
        assert not path.exists()
        fig.savefig(path,dpi=180,metadata={"Date":None} if suffix=="svg" else None)
        output[rel(path)]=dict(sha256=sha(path),bytes=path.stat().st_size)
    plt.close(fig)
    return output

def main(build):
    current=reduce()
    if build:
        assert not (OUT/"SUMMARY.json").exists()
        pictures=figure(current)
        write(OUT/"SUMMARY.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),**current,
                                      figure_outputs=pictures))
    else:
        saved=read(OUT/"SUMMARY.json")
        assert all(saved[key]==value for key,value in current.items())
        for name,binding in saved["figure_outputs"].items():assert sha(ROOT/name)==binding["sha256"]
    print(json.dumps(dict(passed=True,mode="build" if build else "check",
        fit_signatures_passed=current["fit_signatures_passed"],
        qualified_endpoint_blocks=current["qualified_endpoint_blocks"],
        qualified_probe_points=current["qualified_probe_points"],
        new_quadrature_calls=0,new_random_draws=0)))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("build","check"))
    main(parser.parse_args().mode=="build")

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Check compact saved identities and arithmetic, without scientific producers."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from risk_factorized import Audit, audit_case

HERE=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_saved(folder,a):
    folder=Path(folder);risk=read(folder/"data/risk.json");adapt=read(folder/"data/adaptation.json")
    channel=read(folder/"data/learnable_channel.json");execution=read(folder/"data/execution.json")
    expected=[(rule,family) for rule in ("main","high") for family in ("F","L","A")]
    a.same([(r["rule"],r["family"]) for r in risk["cases"]],expected,"complete six-case risk census")
    for row in risk["cases"]:
        statistics=read(folder/"data/moments"/(row["rule"]+".json"))["statistics"]
        a.section(row["rule"]+"/"+row["family"],lambda:audit_case(row,statistics,8,a))
    a.require(risk["completed"] is True and risk["passed"] is True,"saved risk qualification")
    for key in ("new_quadratures","RNG_draws","training_datasets","optimization_calls"):
        a.same(risk["science_counts"][key],0,"historical risk count/"+key)
    a.same(risk["science_counts"]["fixed_rule_state_enumerations"],2,"historical two-rule arithmetic scope")
    for comparison in risk["rule_comparison"]:
        a.require(comparison["passed"] is True,"saved main/high sensitivity")
        a.close(comparison["maximum_absolute_difference"],max(comparison["differences"].values()),"saved sensitivity maximum",1e-15)
        a.require(comparison["maximum_absolute_difference"]<=1e-9,"sensitivity tolerance")
    a.same([len(p["points"]) for p in adapt["panels"]],[12,9,9],"all thirty displayed branch points")
    for panel in adapt["panels"]:
        theory={r["family"]:r["response"] for r in panel["theory"]}
        a.close(theory["F"],theory["L"],"same leading local response",1e-14)
        a.require(0<theory["L"]<theory["A"]<1,"leading response order")
        for point in panel["points"]:
            edge=panel["edge"]
            a.close(point["response"],point["theta"][edge]/(point["t"]/math.sqrt(2)),"saved response coordinate",1e-12)
            a.require(point["ordinary_numerical_qualification"] is True,"saved branch qualification")
    a.same([r["name"] for r in channel["cases"]],[f"{f}_{t:.1f}" for t in (.1,.2) for f in ("F","L","A")],"all six channel cases")
    for row in channel["cases"]:
        r=row["readout"];p=np.asarray(row["coordinates"]["joint"]);known=np.asarray(row["coordinates"]["known"]);initial=np.asarray(row["coordinates"]["initial"]);t=row["t"]
        a.close(r["initial_parameter"],initial,"saved predicted initialization",0.)
        a.close(r["final_minus_initial_parameter"],p-initial,"actual correction separate from initialization",1e-14)
        a.close(r["lambda_shift_over_t2"],(p[3]-known[3])/t**2,"channel shift/t2",1e-13)
        a.close(r["theta_joint_minus_known_over_t3"],(p[:3]-known[:3])/t**3,"full theta difference/t3",1e-12)
        a.close(r["response_joint_minus_known_over_t2"],(p[:2]-known[:2])/(t/math.sqrt(2))/t**2,"direction normalization",1e-12)
        values=row["objective_values"];d24=values["joint24"]-values["known24"];d32=values["joint32"]-values["known32"]
        sensitivity=max(abs(d32-d24),abs(values["joint32"]-values["joint24"])+abs(values["known32"]-values["known24"]))
        a.close(r["joint_minus_known_finite_objective_over_t4"],d32/t**4,"objective difference/t4",1e-13)
        a.close(r["finite_objective_actual_order_difference"],sensitivity,"conservative objective sensitivity",1e-15)
        a.same(r["finite_objective_difference_sign_resolved"],bool(abs(d32)>10*sensitivity),"objective sign rule")
        known_rules=row["known_KL"]["rules"];joint_rules=row["joint_KL"]["rules"]
        differences=[j["joint_KL"]-k["joint_KL"] for j,k in zip(joint_rules,known_rules)]
        a.close(r["joint_KL_joint_minus_known_over_t4"],differences[1]/t**4,"joint KL difference/t4",1e-12)
        a.close(r["joint_KL_difference_paired_order_error"],abs(differences[1]-differences[0]),"paired KL sensitivity",1e-15)
        both=row["known_KL"]["passed"] and row["joint_KL"]["passed"]
        a.same(r["joint_KL_difference_sign_resolved"],bool(both and abs(differences[1])>10*abs(differences[1]-differences[0])),"KL sign rule")
        a.close(r["three_marginal_KL_sum_over_t4"],joint_rules[1]["marginal_KL_sum"]/t**4,"three-group marginal sum/t4",1e-12)
        a.require(row["qualified"] is True,"saved full channel case qualification")
        for field in ("main","high"):a.require(row["qualification"][field]["passed"] is True,"saved "+field+" stationarity")
    a.same(len(execution["selected_old_endpoints"]),9,"all selected old endpoints")
    for row in execution["selected_old_endpoints"]:
        q=row["decomposition24"]
        a.close(q["total_difference"],q["proposal_difference"]+q["profiled_generation_difference"],"two-term objective decomposition",1e-14)
        a.require(q["global_regret"] is False,"local reference scope")
        a.require(row["qualified"] is True,"saved profile qualification")
    a.same(len(execution["A_minus_L"]),12,"all matched-control/output comparisons")
    a.same(len(execution["joint_A_noise"]),4,"all four fixed-point control arms")
    for row in execution["A_minus_L"]:
        a.require(len(row["mean"])==len(row["SD"])==2 and min(row["SD"])>=0,"two-edge R4 descriptive statistics")
    a.counts["displayed_adaptation_points"]=30;a.counts["channel_cases"]=6;a.counts["selected_old_endpoints"]=9
    return {"risk_cases":len(risk["cases"]),"adaptation_points":30,"channel_cases":6,"selected_old_endpoints":9,
        "R4_matched_control_rows":12,"R3_joint_A_noise_rows":4}


def run(folder=HERE,identity=True):
    folder=Path(folder);a=Audit()
    if identity:
        manifest=read(folder/"MANIFEST.json");repo=folder.parents[2]
        declared=manifest["files"]
        actual={p.relative_to(repo).as_posix() for p in folder.rglob("*") if p.is_file() and p.name!="MANIFEST.json"}
        a.same(set(declared),actual,"complete extension manifest")
        for name,record in declared.items():
            p=repo/name;a.same(sha(p),record["sha256"],"file identity/"+name);a.same(p.stat().st_size,record["bytes"],"file size/"+name)
    payload=a.section("saved arithmetic",lambda:check_saved(folder,a)) or {}
    return {"schema":"h34_public_saved_check_v1","passed":not a.failures,"checks":a.count,"failures":a.failures,
        "maximum_absolute_check_error":max(a.maximum_errors.values(),default=0.),"counts":dict(a.counts),**payload,
        "new_integrals":0,"new_RNG_draws":0,"new_training_runs":0,"new_optimization_calls":0,
        "scope":"projected saved identities and algebra; no original model/PMF integration or training replay"}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    target=args.output.resolve();repo=HERE.parents[2]
    if target.is_relative_to(repo):raise ValueError("choose a fresh output outside the source checkout")
    result=run();target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf8") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(result,indent=2));raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":main()

"""Predeclared deterministic readings of the saved collinear Hessian matrices."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/rankone_stability_v1_20261007"
DOC=ROOT/"docs/08_research_研究工作/RankOneStability_v1_20261007"
TAU=1e-5


def read(p):
    return json.loads(Path(p).read_text("utf-8"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def helmert(m):
    Q=np.zeros((m,m-1))
    for j in range(m-1):
        Q[:j+1,j]=1/np.sqrt((j+1)*(j+2))
        Q[j+1,j]=-(j+1)/np.sqrt((j+1)*(j+2))
    return Q


def label(v, qualified):
    if not qualified:return "unqualified"
    if v>TAU:return "positive"
    if v<-TAU:return "negative"
    return "unresolved"


def collect():
    Q=helmert(8)
    rows=[];spectra=[];sources={}
    for rep in range(4):
        folder=OUT/f"rep_{rep:02d}"
        result_path=folder/"RESULTS.json"
        assert read(result_path)["status"]=="completed"
        sources[result_path.relative_to(ROOT).as_posix()]=sha(result_path)
        for qualpath in sorted(folder.glob("*_QUALIFICATION.json")):
            qual=read(qualpath);anchor=qual["anchor"]["name"]
            sourcepaths=[qualpath,folder/f"{anchor}_C.json",folder/f"{anchor}_D.json"]
            for p in sourcepaths:sources[p.relative_to(ROOT).as_posix()]=sha(p)
            C,D=[read(p) for p in sourcepaths[1:]]
            c={(r["target"],str(r["K"])):r for r in C["matrices"]}
            d={(r["target"],str(r["K"])):r for r in D["matrices"]}
            q={(r["target"],str(r["K"])):r["qualified"] for r in qual["targets"]}
            risk=np.array(c[("population","inf")]["laplacian"])
            vals,vecs=np.linalg.eigh(Q.T@risk@Q)
            psi=Q@vecs[:,0]
            if psi[np.argmax(abs(psi))]<0:psi=-psi
            assert abs(psi.sum())<1e-12 and abs(psi@psi-1)<1e-12
            directional=[]
            for key in c:
                x=c[key];H=np.array(x["laplacian"]);H2=np.array(d[key]["laplacian"])
                value=float(psi@H@psi);check=float(psi@H2@psi)
                qualified=q[key] and q[("population","inf")]
                directional.append(dict(target=key[0],K=key[1],C=value,D=check,
                   C_D_difference=abs(value-check),qualified=qualified,
                   sign=label(value,qualified)))
                eig=np.linalg.eigvalsh(Q.T@H@Q)
                spectra.append(dict(rep=rep,anchor=anchor,bias_source=qual["anchor"]["bias_source"],
                    lambda_value=qual["anchor"]["lambda_value"],target=key[0],K=key[1],
                    minimum=float(eig[0]),maximum=float(eig[-1]),qualified=q[key],
                    sign=label(float(eig[0]),q[key]),bias_gradient_Linf=x["bias_gradient_Linf"],
                    radial_gradient_Linf=x["radial_gradient_Linf"]))
            values={(r["target"],r["K"]):r for r in directional}
            riskrow=values[("population","inf")]
            rows.append(dict(rep=rep,anchor=anchor,psi=psi.tolist(),
                selection="C population integrated minimum projected eigenvector",
                selection_eigenvalue=float(vals[0]),selection_eigengap=float(vals[1]-vals[0]),
                normalization=float(psi@psi),common_rotation_component=float(psi.sum()),
                risk_direction_is_oracle_diagnostic_not_training_input=True,
                curvatures=directional,risk_decreasing=riskrow["sign"]=="negative",
                empirical_F8_rejects_risk_direction=bool(riskrow["sign"]=="negative"
                    and values[("empirical","8")]["sign"]=="positive"),
                population_F8_rejects_risk_direction=bool(riskrow["sign"]=="negative"
                    and values[("population","8")]["sign"]=="positive"),
                F64_population_accepts_risk_direction=bool(riskrow["sign"]=="negative"
                    and values[("population","64")]["sign"]=="negative")))
    brackets=[]
    names=["b8_s8","b8_lambda025","b8_lambda050","b8_lambda075","b8_sREF"]
    lambdas=[0.,.25,.5,.75,1.]
    for rep in range(4):
        for target in ["empirical","population"]:
            seq=[next(r for r in spectra if r["rep"]==rep and r["anchor"]==name
                      and r["target"]==target and r["K"]=="8") for name in names]
            changes=[dict(left=lambdas[i],right=lambdas[i+1])
                     for i in range(4) if {seq[i]["sign"],seq[i+1]["sign"]}=={"positive","negative"}]
            brackets.append(dict(rep=rep,target=target,lambdas=lambdas,
                minimum_eigenvalues=[r["minimum"] for r in seq],signs=[r["sign"] for r in seq],
                sign_change_intervals=changes,uniqueness_or_monotonicity_proved=False))
    return dict(rule_sha256=sha(DOC/"ANALYSIS_RULES.md"),source_sha256=sha(__file__),
        sources=sources,classification_threshold=TAU,
        directional_rows=rows,projected_spectra=spectra,
        K8_length_interpolation=brackets,new_quadrature_evaluations=0,new_random_draws=0,
        full_parameter_stationarity_claimed=False,finite_step_risk_change_guaranteed=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["write","check"])
    args=parser.parse_args();value=collect();path=OUT/"DIRECTIONAL_ANALYSIS.json"
    if args.mode=="write":
        record=dict(created_utc=datetime.now(timezone.utc).isoformat(),validation=value)
        with path.open("x",encoding="utf-8",newline="\n") as f:
            json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    else:
        assert read(path)["validation"]==value
    selected=[r for r in value["directional_rows"] if r["anchor"]=="b8_s8"]
    print(json.dumps(dict(passed=True,mode=args.mode,anchors=len(value["directional_rows"]),
        matrices=len(value["projected_spectra"]),canonical_K8_anchors=len(selected),
        population_F8_rejects_risk_direction=sum(r["population_F8_rejects_risk_direction"] for r in selected),
        empirical_F8_rejects_risk_direction=sum(r["empirical_F8_rejects_risk_direction"] for r in selected),
        population_F64_accepts=sum(r["F64_population_accepts_risk_direction"] for r in selected))))

"""Seven fixed block-whitening/entropy-bound arithmetic checks; zero RNG."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import traceback
for _name in ["OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"]:
    os.environ[_name]="2"
import numpy as np
import scipy
from scipy.linalg import block_diag, eigh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/structure_tradeoff_v2/B"
DOC=ROOT/"docs/08_research_研究工作/结构预算与对称化代价_v2_20260930"
M2=ROOT/"artifacts/constraint_boundary_v2/M2"
REVIEW_SHA="bb07fde29f24236e8d28699f9b28d70cdd5deb2b8199e9de05b1c82b1bcb653c"
CFG=dict(d=128,c=.9,b=[1,2,4,8,16,32,64],K=100,dtype="float64",decimal_precision=70,
    matrix_atol=1e-12,decimal_inequality_atol="1e-60",
    inherited_teacher="homogeneous NEG, sigma4",random_observations=0,fits=0,
    confidence="one existing M2 lower bound, inherited original 24-event CP family; no new interval",
    candidate_scope="all components share the same fixed nonoverlapping equal-block partition")

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="\n") as f:
        f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def csvwrite(p,data):
    with Path(p).open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)

def freeze():
    assert not OUT.exists(),"Refuse to overwrite arithmetic package"
    assert sha(ROOT/"HANDOFF.md")==REVIEW_SHA
    config=read(M2/"CONFIG.json")
    assert config["c"]==.9 and config["sigma"]==4
    for name,h in read(M2/"MANIFEST.json")["files"].items():
        assert sha(ROOT/name)==h,name
    old=[r for r in rows(M2/"EVENT_INTERVALS.csv")
         if (r["family"],r["d"],r["event"])==("NEG","128","central")]
    assert len(old)==1 and old[0]["family_size"]=="24"
    assert old[0]["simultaneous_information_lower"]=="0.057984557899112235"
    write(OUT/"CONFIG.json",CFG)
    write(OUT/"ENVIRONMENT.json",dict(time_utc=now(),python=sys.version,
        numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        threads=2,RNG_used=False))
    snapshot=OUT/"source_inputs/HANDOFF.md";snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes((ROOT/"HANDOFF.md").read_bytes())
    write(OUT/"INPUT.json",dict(time_utc=now(),TC_lower_row=old[0],
        reviewed_commit="ce1aa8340028dbdfafc42187876bfc44999e2813",
        review_mode="reference, malformed formulas independently reconstructed",
        review_sha256=REVIEW_SHA,
        vector_BL_source="https://arxiv.org/pdf/1306.2410",
        vector_BL_scope="Chen-Dafnis-Paouris Theorem1(i), vector dimensions arbitrary"))
    files=[Path(__file__),DOC/"BLOCK_PLAN.md",OUT/"CONFIG.json",OUT/"ENVIRONMENT.json",
           OUT/"INPUT.json",snapshot,M2/"CONFIG.json",M2/"EVENT_INTERVALS.csv",M2/"MANIFEST.json"]
    write(OUT/"FREEZE.json",dict(time_utc=now(),before_arithmetic=True,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
    print(json.dumps(dict(freeze_sha256=sha(OUT/"FREEZE.json"),cases=7)),flush=True)

def check_freeze():
    assert read(OUT/"CONFIG.json")==CFG
    for name,h in read(OUT/"FREEZE.json")["files"].items():assert sha(ROOT/name)==h,name

def calculate():
    d=CFG["d"];c=CFG["c"];rho=c/(d-1)
    R=(1+rho)*np.eye(d)-rho*np.ones((d,d))
    inherited=read(OUT/"INPUT.json")["TC_lower_row"]["simultaneous_information_lower"]
    table=[];details=[];spectra=[]
    for b in CFG["b"]:
        assert d%b==0 and d//b>=2
        m=d//b;A=R[:b,:b];ev,V=np.linalg.eigh(A)
        Ainv=(V*(1/np.sqrt(ev)))@V.T
        W=block_diag(*[Ainv for _ in range(m)])
        marginal=block_diag(*[A for _ in range(m)])
        T=W@R@W.T
        eigen=np.linalg.eigvalsh(T)
        generalized=eigh(R,marginal,eigvals_only=True)
        with localcontext() as ctx:
            ctx.prec=CFG["decimal_precision"]
            cd=Decimal(str(c));dd=Decimal(d);bd=Decimal(b)
            rd=cd/(dd-1);q=1-rd*(bd-1)
            gamma=rd*bd/q;L=1+gamma
            low=(1-cd)/q
            U=-((bd-1)*(1+rd).ln()+q.ln())/2
            total=Decimal(m)*U
            simple=dd*(bd-1)*rd*rd/(4*q)
            tc=Decimal(inherited)
            tcblock=max(Decimal(0),tc-total)
            allowance=gamma*Decimal(CFG["K"]).ln()
            kl=max(Decimal(0),tc-total-allowance)
            score_floor=-total-allowance
            assert total>=-Decimal(CFG["decimal_inequality_atol"])
            assert total<=simple+Decimal(CFG["decimal_inequality_atol"])
        expected=np.sort(np.r_[float(low),np.ones(d-m),np.full(m-1,float(L))])
        expected_matrix=np.empty_like(T)
        for i in range(m):
            for j in range(m):
                expected_matrix[i*b:(i+1)*b,j*b:(j+1)*b]=np.eye(b) if i==j else -rho/float(q)*np.ones((b,b))
        sign,logdet=np.linalg.slogdet(A)
        signall,logdetall=np.linalg.slogdet(marginal)
        assert sign==signall==1
        slack=float(np.linalg.eigvalsh(float(L)*np.eye(d)-T)[0])
        detail=dict(b=b,blocks=m,
            covariance_R_min_eigenvalue=float(np.linalg.eigvalsh(R)[0]),
            A_min_eigenvalue=float(ev[0]),
            whitened_min_eigenvalue=float(eigen[0]),
            whitening_formula_max_error=float(np.max(abs(T-expected_matrix))),
            full_spectrum_max_error=float(np.max(abs(eigen-expected))),
            generalized_spectrum_max_error=float(np.max(abs(generalized-expected))),
            maximum_eigenvalue_error=abs(float(eigen[-1])-float(L)),
            direct_logdet_error=abs(-.5*float(logdet)-float(U)),
            total_logdet_error=abs(-.5*float(logdetall)-float(total)),
            PSD_slack_min_eigenvalue=slack,
            symmetry_max_error=float(np.max(abs(T-T.T))))
        checks=[value<CFG["matrix_atol"] for key,value in detail.items() if key.endswith("error")]
        detail["passed"]=bool(all(checks) and slack>=-CFG["matrix_atol"] and detail["whitened_min_eigenvalue"]>0)
        details.append(detail)
        spectra.append(dict(b=b,blocks=m,observed=eigen.tolist(),analytic=expected.tolist(),
                            generalized=generalized.tolist()))
        table.append(dict(d=d,b=b,blocks=m,c=c,K=CFG["K"],q_b=str(q),
            L_b=str(L),L_b_minus_1=str(gamma),
            direct_lambda_max=float(eigen[-1]),
            one_block_Gaussian_TC_upper=str(U),
            sum_block_Gaussian_TC_upper=str(total),
            finite_d_simple_internal_TC_upper=str(simple),
            inherited_coordinate_TC_lower=inherited,
            derived_block_TC_lower=str(tcblock),
            mixture_allowance_gamma_logK=str(allowance),
            mixture_observed_KL_lower=str(kl),
            coordinate_baseline_F_minus_F0_lower=str(score_floor),
            numerical_pass=detail["passed"],
            confidence="inherited M2 original family; deterministic postprocessing"))
    assert len(table)==len(details)==7
    return table,details,spectra

def run():
    check_freeze();assert not (OUT/"TABLE.csv").exists()
    table,details,spectra=calculate()
    csvwrite(OUT/"TABLE.csv",table);csvwrite(OUT/"AUDIT_DETAILS.csv",details)
    write(OUT/"SPECTRA.json",dict(cases=spectra))
    audit=dict(time_utc=now(),passed=all(r["passed"] for r in details),cases=7,
        max_errors={key:max(r[key] for r in details) for key in details[0] if key.endswith("error")},
        minimum_PSD_slack=min(r["PSD_slack_min_eigenvalue"] for r in details),
        no_random_observations=True,no_fits=True,new_confidence_intervals=0,
        theorem_certified_by_arithmetic=False,originality_certified=False)
    write(OUT/"AUDIT.json",audit)
    assert audit["passed"],"Fixed arithmetic gate failed; retain all outputs"
    text=["# 固定d128分块算术表","",
          "c=.9、K100；仅既有homogeneous NEG/sigma4的TC下界用于确定性代入。不是新观察、拟合或CI。",
          "", "|b|块数|L_b−1|块内TC和上界|块间TC下界|观察KL下界|数值核验|",
          "|---:|---:|---:|---:|---:|---:|---|"]
    for r in table:
        text.append(f'|{r["b"]}|{r["blocks"]}|{float(r["L_b_minus_1"]):.9g}|{float(r["sum_block_Gaussian_TC_upper"]):.9g}|{float(r["derived_block_TC_lower"]):.9g}|{float(r["mixture_observed_KL_lower"]):.9g}|{r["numerical_pass"]}|')
    text += ["","下界为0表示这一固定代入没有给出正认证，不表示真实块间依赖或误差为0。全部七格保留。"]
    (OUT/"READABLE.md").write_text("\n".join(text)+"\n",encoding="utf-8",newline="\n")
    files=sorted(p for p in OUT.rglob("*") if p.is_file())
    write(OUT/"MANIFEST.json",dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        scope="fixed arithmetic package; theorem/review prose separately managed",
        random_observations=0,fits=0))
    check()

def check():
    check_freeze()
    for name,h in read(OUT/"MANIFEST.json")["files"].items():assert sha(ROOT/name)==h,name
    expected,details,_=calculate()
    actual=rows(OUT/"TABLE.csv")
    assert len(actual)==7
    for a,b in zip(actual,expected):
        for key,value in b.items():
            if isinstance(value,(str,bool,int)):
                assert a[key]==str(value),(key,a[key],value)
            else:assert abs(float(a[key])-value)<1e-12,(key,a[key],value)
    assert all(d["passed"] for d in details)
    print(json.dumps(dict(pass_block_arithmetic=True,cases=7,
        max_lambda_error=max(r["maximum_eigenvalue_error"] for r in details),
        random_observations=0,fits=0)),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["freeze","run","check"]);a=p.parse_args()
    try:globals()[a.phase]()
    except Exception:
        if a.phase=="run" and OUT.exists():
            write(OUT/f"FAILURE_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json",
                  dict(time_utc=now(),traceback=traceback.format_exc(),no_extra_parameters=True))
        raise

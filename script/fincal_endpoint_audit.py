"""Independent fixed-input audit of the posthoc endpoint-sign diagnostic.

Recompute actual observed pair sums and saved endpoint contrasts. The expected
sign is an analytic identity, not an estimated validation derivative sign.
"""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import math
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts/finite_calibration"
OUT=BASE/"ENDPOINT_DIAGNOSTIC"
PROTOCOL=OUT/"AUDIT_PROTOCOL.json"
DEST=OUT/"AUDIT.json"
TOL=1e-11


def read(path):return json.loads(Path(path).read_text("utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ff(value):return Fraction.from_float(float(value))
def fraction_record(value):return dict(numerator=str(value.numerator),denominator=str(value.denominator),float_value=float(value))
def save_new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")


def freeze():
    assert not PROTOCOL.exists() and not DEST.exists()
    paths=[OUT/"PROTOCOL.json",OUT/"RESULTS.json",BASE/"TRAIN/RESULTS.json",
           BASE/"CAL/MODEL.json",ROOT/"script/fincal_endpoint_diagnostic.py"]
    train=read(BASE/"TRAIN/RESULTS.json")
    save_new(PROTOCOL,dict(
        schema="fincal-endpoint-independent-audit-v1",source_sha256=sha(__file__),
        inputs={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
        npz_expected_sha256={r["file"]:r["file_sha256"] for r in train["records"]},
        comparison_absolute_tolerance=TOL,
        sign_tie_absolute_threshold=read(OUT/"PROTOCOL.json")["arithmetic_near_tie"],
        data_scope="Integer y1*y2 sums from every saved teacher data vector, preserving N prefixes",
        score_scope="96 saved R-minus-minusR raw endpoint contrasts, independently fsum and exact Fraction arithmetic; no raw likelihood re-evaluation",
        expected_sign="2*t_N*V(R) has sign minus sign(t_N), from exact finite-model parity and Vprime<0; no numerical population integral",
        finite_penalty="Exact Fraction integral of stored cubic gprime over [-R,R], then compare producer floating endpoint difference and classification",
        interpretation="Posthoc fixed-input sign diagnostic only; no population frequency, new significance test or causal percentage",
        preserve_failures=True,automatic_rerun=False,new_random_draws=0,new_data_fits=0,new_Gaussian_integrals=0))
    print(json.dumps({"frozen":True,"protocol_sha256":sha(PROTOCOL)}),flush=True)


def run():
    assert not DEST.exists(),"Preserve previous audit result"
    protocol=read(PROTOCOL);assert protocol["source_sha256"]==sha(__file__)
    for path,digest in protocol["inputs"].items():assert sha(ROOT/path)==digest,path
    started=time.perf_counter();comparisons=[];exacts=[];rows=[];summaries=[];npz_inputs=[]
    failure=None;pair_products=0;endpoint_values=0;parity=Fraction(0)

    def compare(name,a,e):
        difference=abs(float(a)-float(e))
        comparisons.append(dict(name=name,count=1,tolerance=TOL,max_absolute_error=difference,failures=int(difference>TOL)))
    def exact(name,a,e):exacts.append(dict(name=name,passed=bool(a==e)))
    def sign_with_threshold(value,tolerance):
        return 1 if value>tolerance else (-1 if value < -tolerance else 0)

    try:
        p,target=read(OUT/"PROTOCOL.json"),read(OUT/"RESULTS.json")
        train,model=read(BASE/"TRAIN/RESULTS.json"),read(BASE/"CAL/MODEL.json")
        source=sha(ROOT/"script/fincal_endpoint_diagnostic.py")
        exact("producer source",p["source_sha256"],source);exact("target source",target["source_sha256"],source)
        exact("target protocol",target["protocol_sha256"],sha(OUT/"PROTOCOL.json"))
        for path,digest in p["inputs"].items():exact("input "+path,sha(ROOT/path),digest)
        exact("posthoc",p["posthoc"],True)
        exact("no new test",p["new_tests"],False)
        exact("no causal percentage",p["no_causal_percentage_claim"],True)
        near=float(p["arithmetic_near_tie"]);near_fraction=ff(near)
        case=next(c for c in model["cases"] if c["n"]==254)
        c=case["gprime_spline_c"];knots=case["gprime_spline_x"]
        assert knots[0]==-.75 and knots[-1]==.75
        for j in range(len(knots)-1):
            width=ff(knots[j+1])-ff(knots[j])
            parity+=(ff(c[0][j])*width**4/4+ff(c[1][j])*width**3/3+ff(c[2][j])*width**2/2+ff(c[3][j])*width)
        compare("finite endpoint penalty difference",parity,target["finite_spline_endpoint_penalty_difference"])
        exact("finite parity below original near-tie threshold",abs(parity)<near_fraction,True)
        expected={(r["stream"],r["theta"],r["N"]):r for r in target["rows"]}
        assert len(expected)==96 and len(train["records"])==16
        assert train["config"]["teachers"]==[0.,.3,.6] and train["config"]["sample_sizes"]==[1024,4096]
        for stream in train["records"]:
            path=ROOT/stream["file"];digest=sha(path)
            exact("npz "+str(stream["stream"]),digest,protocol["npz_expected_sha256"][stream["file"]])
            npz_inputs.append(dict(file=stream["file"],sha256=digest))
            with np.load(path,allow_pickle=False) as data:
                raw=data["raw"];y1=data["y1"];y2=data["y2"]
            assert raw.shape[0]==128 and raw.shape[1:3]==(3,2) and raw.shape[-1]==801
            assert len(y1)==4096 and y2.shape==(3,4096)
            assert set(np.unique(y1))<={-1,1} and set(np.unique(y2))<={-1,1}
            outputs={(r["theta"],r["N"],r["method"]):r for r in stream["outputs"]}
            for ti,theta in enumerate(train["config"]["teachers"]):
                products=[int(a)*int(b) for a,b in zip(y1,y2[ti])]
                pair_products+=len(products)
                for ni,N in enumerate(train["config"]["sample_sizes"]):
                    ident=(stream["stream"],theta,N);saved=expected[ident]
                    pair=sum(products[:N]);t=Fraction(pair,N)
                    differences=[float(raw[b,ti,ni,-1])-float(raw[b,ti,ni,0]) for b in range(128)]
                    gap=math.fsum(differences)/128
                    rational_gap=sum((ff(raw[b,ti,ni,-1])-ff(raw[b,ti,ni,0]) for b in range(128)),Fraction(0))/128
                    endpoint_values+=256
                    expected_sign=-((pair>0)-(pair<0))
                    actual_sign=sign_with_threshold(rational_gap,near_fraction)
                    finite_sign=sign_with_threshold(rational_gap+parity,near_fraction)
                    actual=outputs[(theta,N,"raw")]
                    exact("pair sum "+str(ident),pair,saved["pair_sum"])
                    exact("pair ratio "+str(ident),float(t),saved["t_N"])
                    exact("TRAIN pair "+str(ident),float(t),actual["empirical_pair"])
                    compare("endpoint gap "+str(ident),gap,saved["actual_raw_endpoint_gap"])
                    compare("rational endpoint gap "+str(ident),rational_gap,saved["actual_raw_endpoint_gap"])
                    exact("expected sign "+str(ident),expected_sign,saved["expected_conditional_gap_sign"])
                    exact("actual sign "+str(ident),actual_sign,saved["actual_gap_sign"])
                    opposite=expected_sign*actual_sign<0
                    exact("opposite "+str(ident),opposite,saved["opposite"])
                    exact("expectation tie "+str(ident),pair==0,saved["expectation_tie"])
                    exact("numeric tie "+str(ident),abs(rational_gap)<=near_fraction,saved["numeric_gap_tie"])
                    exact("raw rho "+str(ident),actual["rho"],saved["raw_rho"])
                    rows.append(dict(stream=stream["stream"],theta=theta,N=N,pair_sum=pair,t_N=float(t),
                        independent_raw_gap=gap,exact_stored_score_gap=fraction_record(rational_gap),
                        expected_conditional_gap_sign=expected_sign,actual_gap_sign=actual_sign,
                        finite_penalty_gap_sign=finite_sign,finite_preserves_actual_endpoint_sign=finite_sign==actual_sign,
                        opposite=opposite,expectation_tie=pair==0,numeric_gap_tie=abs(rational_gap)<=near_fraction,
                        raw_rho=actual["rho"]))
        target_summary={(r["theta"],r["N"]):r for r in target["summary"]}
        for theta in [0.,.3,.6]:
            for N in [1024,4096]:
                subset=[r for r in rows if r["theta"]==theta and r["N"]==N]
                summary=dict(theta=theta,N=N,opposite_count=sum(r["opposite"] for r in subset),
                             expectation_ties=sum(r["expectation_tie"] for r in subset),
                             numeric_ties=sum(r["numeric_gap_tie"] for r in subset))
                exact("summary "+str((theta,N)),summary,target_summary[(theta,N)])
                summaries.append(summary)
        exact("96 rows",len(rows),96);exact("six summaries",len(summaries),6)
    except Exception:
        failure=traceback.format_exc()
    nf=sum(r["failures"] for r in comparisons);ef=sum(not r["passed"] for r in exacts)
    result=dict(status="passed" if failure is None and nf==0 and ef==0 else "failed",
        code_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),input_sha256=protocol["inputs"],
        elapsed_seconds=time.perf_counter()-started,numerical_comparisons=comparisons,exact_checks=exacts,
        numerical_failures=nf,exact_failures=ef,npz_inputs=npz_inputs,rows=rows,summary=summaries,
        finite_penalty_endpoint_integral_exact=fraction_record(parity),
        positive_theta_highN_opposites=[r for r in rows if r["theta"]>0 and r["N"]==4096 and r["opposite"]],
        positive_theta_highN_negative_raw=[r for r in rows if r["theta"]>0 and r["N"]==4096 and r["raw_rho"]<0],
        ledger=dict(new_random_draws=0,new_fits=0,new_Gaussian_integrals=0,
            observed_pair_products_recomputed=pair_products,saved_endpoint_score_values_reaggregated=endpoint_values,
            endpoint_assignments=len(rows),summary_cells=len(summaries),independent_training_streams_reused=16),
        software_error=failure,
        scope="Fixed-input endpoint ranking signs from saved observed data and scores; analytic conditional expectation sign; no population frequency, new tests or causal percentage")
    save_new(DEST,result)
    print(json.dumps({"status":result["status"],"numeric_failures":nf,"exact_failures":ef,
        "ledger":result["ledger"],"sha256":sha(DEST)}),flush=True)
    if result["status"]!="passed":raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["freeze","run"])
    a=p.parse_args();freeze() if a.mode=="freeze" else run()

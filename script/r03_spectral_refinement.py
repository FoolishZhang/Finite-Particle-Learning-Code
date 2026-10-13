"""Saved-matrix R03 spectral refinement, with operator error before projection.

No quadrature, optimizer, RNG or scientific producer imports. ``--self-test``
uses manufactured matrices. ``--write`` creates a new receipt from explicitly
named existing certificates; ``--check`` replays only their saved arithmetic.
Neither mode modifies any original certificate.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import flint
from flint import arb, ctx


M = 8
TARGETS = {"Hbb": "1/10", "Haa": "1/50", "Hperp": "1/200"}
ROOT = Path(__file__).resolve().parents[1]
METHOD = "docs/08_research_研究工作/R03TeacherFamily_v1_20261011/SPECTRAL_REFINEMENT.md"


def rational(value):
    if isinstance(value, (bool,float)):
        raise ValueError("explicit rational required")
    return Fraction(value)


def aq(value):
    q = rational(value)
    return arb(q.numerator)/q.denominator


def dyadic(value):
    if type(value["exponent_of_two"]) is not int:
        raise ValueError("integer dyadic exponent required")
    return Fraction(int(value["mantissa"]))*Fraction(2)**value["exponent_of_two"]


def endpoints(value):
    lo,hi = [dyadic(value[key]) for key in ("lower_exact_dyadic","upper_exact_dyadic")]
    if lo>hi:
        raise ValueError("reversed saved interval")
    return lo,hi


def endpoint(value):
    m,e=value.man_exp()
    return dict(mantissa=str(m),exponent_of_two=int(e))


def record(value):
    if not value.is_finite():
        raise ArithmeticError("nonfinite output enclosure")
    return dict(arb_display_only=str(value),lower_exact_dyadic=endpoint(value.lower()),
                upper_exact_dyadic=endpoint(value.upper()))


def ball(value):
    lo,hi=endpoints(value)
    return aq(lo).union(aq(hi))


def exact_dyadic(value):
    value=rational(value)
    denominator=value.denominator
    if denominator & (denominator-1):
        raise ValueError("expected an exact dyadic rational")
    return dict(mantissa=str(value.numerator),exponent_of_two=-(denominator.bit_length()-1))


def matmul(A,B):
    return [[sum((A[i][k]*B[k][j] for k in range(len(B))),arb(0))
             for j in range(len(B[0]))] for i in range(len(A))]


def transpose(A):
    return [list(row) for row in zip(*A)]


def matrix_records(A):
    return [[record(x) for x in row] for row in A]


def householder(a0):
    norm=sum((x*x for x in a0),arb(0)).sqrt()
    v=[x/norm for x in a0]
    v[0]+=1
    den=sum((x*x for x in v),arb(0))
    return [[arb(int(i==j))-2*v[i]*v[j]/den for j in range(1,M)] for i in range(M)]


def ldl_positive(A,shift):
    """All interval pivots strictly positive; failures retained as unresolved."""
    n=len(A)
    S=[list(row) for row in A]
    for i in range(n):
        for j in range(i):
            if not S[i][j].overlaps(S[j][i]):
                raise ValueError("symmetric point-matrix arithmetic inconsistent")
            S[i][j]=S[j][i]=S[i][j].intersection(S[j][i])
    L=[[arb(0) for _ in range(n)] for _ in range(n)]
    D=[]
    for i in range(n):
        L[i][i]=arb(1)
        d=S[i][i]-shift-sum((L[i][k]*L[i][k]*D[k] for k in range(i)),arb(0))
        D.append(d)
        if not d>0:
            return dict(passed=False,status="positive_lower_bound_unresolved",failed_pivot=i,
                        pivots=[record(x) for x in D],shift=record(shift))
        for j in range(i+1,n):
            L[j][i]=(S[j][i]-sum((L[j][k]*L[i][k]*D[k] for k in range(i)),arb(0)))/d
    return dict(passed=True,status="strictly_positive",failed_pivot=None,
                pivots=[record(x) for x in D],L=matrix_records(L),shift=record(shift))


def midpoint_radius(saved):
    """Exact Fraction midpoint and Frobenius radius from original 8x8 entries."""
    if len(saved)!=M or any(len(row)!=M for row in saved):
        raise ValueError("original eight-dimensional block required")
    boxes=[[endpoints(x) for x in row] for row in saved]
    if any(boxes[i][j]!=boxes[j][i] for i in range(M) for j in range(M)):
        raise ValueError("original matrix must have matching symmetric saved enclosures")
    mid=[[(lo+hi)/2 for lo,hi in row] for row in boxes]
    radius=[[(hi-lo)/2 for lo,hi in row] for row in boxes]
    squared=sum((x*x for row in radius for x in row),Fraction(0))
    rho=aq(squared).sqrt()
    upper=dyadic(endpoint(rho.upper()))
    # Independent standard-library rational comparison, not an Arb comparison.
    if upper<0 or squared>upper*upper:
        raise ArithmeticError("Fraction square audit of Frobenius cap failed")
    audit=dict(arithmetic="fractions.Fraction exact dyadic endpoints",
        sum_halfwidth_squares_exact=str(squared),rho_upper_exact=str(upper),
        rho_upper_squared_exact=str(upper*upper),square_margin_exact=str(upper*upper-squared),
        square_inequality_passed=True)
    return mid,upper,dict(midpoint_exact_dyadic=[[exact_dyadic(x) for x in row] for row in mid],
        halfwidth_exact_dyadic=[[exact_dyadic(x) for x in row] for row in radius],
        rho_F_enclosure=record(rho),rho_F_upper_exact_dyadic=exact_dyadic(upper),
        Fraction_radius_audit=audit)


def refine_certificate(certificate):
    """Pure refinement of one completed saved certificate; does not read files."""
    if certificate.get("schema")!="r03_continuous_box_certificate_v1":
        raise ValueError("completed R03 continuous-box certificate required")
    bits=certificate["precision_bits"]
    if type(bits) is not int or bits<128:
        raise ValueError("saved precision_bits>=128 required")
    h=rational(certificate["h_exact"])
    if h not in (Fraction(1,100),Fraction(1,1000),Fraction(1,10000)):
        raise ValueError("only the three original h candidates are eligible")
    oldroot=certificate["contraction"]["passed"]
    if type(oldroot) is not bool or certificate["root_exists_unique_in_box_for_every_lambda"] != oldroot:
        raise ValueError("inconsistent original root status")
    oldrisk=certificate["risk"]["passed"]
    if type(oldrisk) is not bool:
        raise ValueError("original risk status must be boolean")
    a0q=[rational(x) for x in certificate["a0_exact"]]
    if len(a0q)!=M or any(x<=0 for x in a0q):
        raise ValueError("fixed exact positive eight-coordinate a0 required")
    radii=[rational(x) for x in certificate["radii_exact"]]
    if radii!=[Fraction(3,4)*h]+[Fraction(3,20)*h]*7:
        raise ValueError("original prescribed radii required")
    slice_lower=sum((a*(a-r) for a,r in zip(a0q,radii)),Fraction(0))
    if slice_lower<=0:
        raise ValueError("fixed Householder slice is not uniformly transverse")
    previous=ctx.prec;ctx.prec=bits
    try:
        U=householder([aq(x) for x in a0q])
        blocks={}
        for name,target in TARGETS.items():
            mid,radius,details=midpoint_radius(certificate["signed_combinations"]["Hlambda_finite"][name])
            point=[[aq(x) for x in row] for row in mid]
            error_key="Hperp_total_operator" if name=="Hperp" else "Hba_total_operator"
            error_lo,error_hi=endpoints(certificate["analytic_errors"][error_key])
            if error_lo<0:
                raise ValueError("nonnegative saved analytic operator-error enclosure required")
            shift_exact=rational(target)+error_hi+radius
            matrix=matmul(matmul(transpose(U),point),U) if name=="Hperp" else point
            proof=ldl_positive(matrix,aq(shift_exact))
            blocks[name]=dict(target_mu_exact=target,original_dimension=M,
                checked_dimension=7 if name=="Hperp" else M,
                analytic_operator_error_upper_exact=str(error_hi),
                total_shift_exact=str(shift_exact),shift=record(aq(shift_exact)),
                checked_midpoint_matrix=matrix_records(matrix),ldl=proof,
                perturbation_bound_before_projection=True,
                projected_interval_entry_radii_used=False,**details)
        allstable=all(row["ldl"]["passed"] for row in blocks.values())
        core=oldroot and allstable and oldrisk
        return dict(schema="r03_one_box_spectral_refinement_v1",h_exact=str(h),
            a0_exact=[str(x) for x in a0q],radii_exact=[str(x) for x in radii],precision_bits=bits,
            original_status=certificate["status"],original_passed=certificate["passed"],
            original_root_passed=oldroot,original_risk_passed=oldrisk,
            original_stability=copy.deepcopy(certificate["stability"]),
            original_reversal_derivative=copy.deepcopy(certificate.get("reversal_derivative")),
            original_true_risk_curvature=copy.deepcopy(certificate["risk"].get("true_curvature")),
            original_contraction=copy.deepcopy(certificate["contraction"]),
            source_certificate_object_sha256=object_sha(certificate),
            blocks=blocks,fixed_Householder_basis=matrix_records(U),
            exact_slice_dot_lower=str(slice_lower),
            all_three_stability_targets_passed=allstable,core_passed=core,
            claimed_lambda_interval_exact=[str(-h),str(h)] if core else None,
            status="certified_local_teacher_family" if core else "unresolved",
            original_failed_enclosures_preserved=True,targets_lowered=False,
            original_integrals_replayed=False,new_integral_nodes=0,new_optimizer_updates=0,
            proof_scope="same fixed local box and original branch; no global robustness claim",
            operator_projection_argument="exact U0^T U0=I, so ||U0^T E U0||op<=||E||op<=rho_F")
    finally:ctx.prec=previous


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def compose(certificates):
    if not certificates:
        raise ValueError("at least one completed input certificate required")
    hs=[rational(x["h_exact"]) for x in certificates]
    if len(hs)!=len(set(hs)):
        raise ValueError("duplicate original h certificate")
    rows=[refine_certificate(x) for x in certificates]
    successful=[rational(x["h_exact"]) for x in rows if x["core_passed"]]
    largest=max(successful) if successful else None
    return dict(schema="r03_spectral_refinement_v1",python_flint_version=flint.__version__,
        rows=rows,claimed_lambda_interval_exact=None if largest is None else [str(-largest),str(largest)],
        selected_largest_among_supplied_certified_boxes=True,
        every_supplied_completed_box_processed=True,automatic_new_box_execution=False,
        new_integral_nodes=0,new_target_integrals=0,new_teacher_integrals=0,new_random_draws=0,
        new_optimizer_updates=0,old_failed_certificates_modified=False,
        method="midpoint plus exact original-matrix Frobenius radius; operator before projection")


def manufactured_checks():
    previous=ctx.prec;ctx.prec=192
    try:
        def fake(h="1/1000"):
            rec=lambda x:record(aq(x))
            matrices={name:[[rec(Fraction(1,5) if i==j else 0) for j in range(8)] for i in range(8)]
                      for name in TARGETS}
            return dict(schema="r03_continuous_box_certificate_v1",precision_bits=192,h_exact=h,
                a0_exact=["1/2"]*8,radii_exact=[str(Fraction(3,4)*rational(h))]+[str(Fraction(3,20)*rational(h))]*7,
                contraction=dict(passed=True),root_exists_unique_in_box_for_every_lambda=True,
                risk=dict(passed=True,true_curvature=rec("-1/100")),
                analytic_errors=dict(Hba_total_operator=rec("1/1000"),Hperp_total_operator=rec("1/1000")),
                signed_combinations=dict(Hlambda_finite=matrices),
                stability={name:dict(passed=False,status="manufactured_original_failure",failed_pivot=6)
                           for name in ("Hbb","Haa","transverse_slice")},
                status="unresolved",passed=False)
        checks={};source=fake();before=object_sha(source);result=refine_certificate(source)
        checks["manufactured_three_stability_targets_pass"]=result["core_passed"]
        checks["original_failure_retained_and_input_unchanged"]=(object_sha(source)==before and
            result["original_stability"]==source["stability"] and not result["original_passed"])
        checks["mu_targets_unchanged"]={k:v["target_mu_exact"] for k,v in result["blocks"].items()}==TARGETS
        narrow=copy.deepcopy(source)
        for name in TARGETS:
            for i in range(8):
                for j in range(8):
                    center=Fraction(1,4) if i==j else Fraction(0)
                    narrow["signed_combinations"]["Hlambda_finite"][name][i][j]=record(
                        aq(center-Fraction(1,1024)).union(aq(center+Fraction(1,1024))))
        detailed=refine_certificate(narrow)
        audit=detailed["blocks"]["Hperp"]["Fraction_radius_audit"]
        checks["independent_Fraction_square_bound"]=(rational(audit["sum_halfwidth_squares_exact"])<=
            rational(audit["rho_upper_exact"])**2 and audit["square_inequality_passed"])
        original_boxes=narrow["signed_combinations"]["Hlambda_finite"]["Hperp"]
        direct_radius=sum(((hi-lo)/2)**2 for row in original_boxes for lo,hi in map(endpoints,row))
        checks["preprojection_radius_uses_original_64_entries"]=(
            rational(audit["sum_halfwidth_squares_exact"])==direct_radius and
            not detailed["blocks"]["Hperp"]["projected_interval_entry_radii_used"])
        U=householder([aq("1/2")]*8);gram=matmul(transpose(U),U)
        checks["Householder_interval_identity_fixture"]=all(gram[i][j].contains(int(i==j)) for i in range(7) for j in range(7))
        checks["Householder_normal_orthogonality_fixture"]=all(
            sum((U[i][j]*aq("1/2") for i in range(8)),arb(0)).contains(0) for j in range(7))
        failed=fake();failed["contraction"]["passed"]=False;failed["root_exists_unique_in_box_for_every_lambda"]=False
        checks["failed_original_root_cannot_be_promoted"]=not refine_certificate(failed)["core_passed"]
        failed=fake();failed["risk"]["passed"]=False
        checks["failed_original_risk_cannot_be_promoted"]=not refine_certificate(failed)["core_passed"]
        failed=fake()
        failed["signed_combinations"]["Hlambda_finite"]["Hperp"]=[[record(arb(0)) for j in range(8)] for i in range(8)]
        failed_result=refine_certificate(failed)
        checks["insufficient_midpoint_is_unresolved"]=not failed_result["core_passed"] and not failed_result["blocks"]["Hperp"]["ldl"]["passed"]
        many=compose([fake("1/10000"),fake("1/1000")])
        checks["largest_only_among_supplied_boxes"]=many["claimed_lambda_interval_exact"]==["-1/1000","1/1000"]
        checks["strict_json_roundtrip"]=json.loads(json.dumps(many,allow_nan=False))["rows"][0]["core_passed"]
        if not all(checks.values()):raise AssertionError(checks)
        return dict(schema="r03_spectral_refinement_manufactured_checks_v1",passed=True,checks=checks,
                    new_integrals=0,real_science_input_reads=0)
    finally:ctx.prec=previous


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(value,existing=True):
    path=Path(value)
    path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT) or path.suffix.lower()!=".json":
        raise ValueError("JSON path must remain inside repository")
    if existing and not path.is_file():raise ValueError("missing input JSON: "+str(path))
    return path


def produce(input_paths):
    paths=[safe_path(x) for x in input_paths]
    if len(paths)!=len(set(paths)):raise ValueError("duplicate input path")
    original_bytes=[x.read_bytes() for x in paths]
    hashes=[hashlib.sha256(x).hexdigest() for x in original_bytes]
    certificates=[json.loads(x.decode("utf-8")) for x in original_bytes]
    result=compose(certificates)
    if any(file_sha(path)!=digest for path,digest in zip(paths,hashes)):
        raise ValueError("input certificate changed during saved-arithmetic refinement")
    result["input_files"]=[dict(path=x.relative_to(ROOT).as_posix(),sha256=digest) for x,digest in zip(paths,hashes)]
    result["source_bindings"]={Path(__file__).resolve().relative_to(ROOT).as_posix():file_sha(Path(__file__).resolve()),
                               METHOD:file_sha(ROOT/METHOD)}
    result["manufactured_implementation_checks"]=manufactured_checks()
    return result


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write",action="store_true");mode.add_argument("--check",action="store_true")
    mode.add_argument("--self-test",action="store_true")
    parser.add_argument("--input",action="append",default=[])
    parser.add_argument("--inputs",nargs="+",default=[])
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.self_test:
        if args.input or args.inputs or args.output:parser.error("self-test accepts no scientific paths")
        print(json.dumps(manufactured_checks(),indent=2,allow_nan=False));return
    if args.output is None:parser.error("--output is required")
    output=safe_path(args.output,existing=args.check)
    if args.write:
        if output.exists():raise ValueError("refusing to overwrite an existing refinement receipt")
        inputs=args.input+args.inputs
        if not inputs:parser.error("--write requires explicit --input or --inputs certificates")
        result=produce(inputs)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    else:
        if args.input or args.inputs:parser.error("--check uses the receipt's bound input list")
        saved=json.loads(output.read_text(encoding="utf-8"))
        for path,digest in saved["source_bindings"].items():
            if file_sha(ROOT/path)!=digest:raise ValueError("refinement source drift: "+path)
        for item in saved["input_files"]:
            if file_sha(safe_path(item["path"]))!=item["sha256"]:raise ValueError("certificate input drift: "+item["path"])
        result=produce([x["path"] for x in saved["input_files"]])
        if result!=saved:raise ValueError("saved spectral arithmetic replay mismatch")
    print(json.dumps(dict(passed=True,action="check" if args.check else "write",
        boxes=len(result["rows"]),claimed_lambda_interval_exact=result["claimed_lambda_interval_exact"],
        new_integral_nodes=0),indent=2))


if __name__=="__main__":main()

"""Frozen saved-endpoint KL/chi-square synthesis; no grids or integrations.

The exact model distributions have unit mass; their marginal probability
balls need not sum to a point and are never renormalized.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import operator
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".cache/finite_selection_tools"))
import flint
from flint import arb,ctx
OUT=ROOT/"artifacts/submission_candidate_v1_8_20261008"
DOC=ROOT/"docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_8_20261008"
FREEZE=OUT/"RISK_RANGE_FREEZE.json";RESULT=OUT/"RISK_RANGE_RESULT.json"
MANUFACTURED=OUT/"RISK_RANGE_MANUFACTURED_CHECK.json"
SOURCE_PATHS=(
    "artifacts/certified_teacher_v1_20261008/TEACHER_CERTIFICATE.json",
    "artifacts/certified_teacher_v1_20261008/TEACHER_RISK_CERTIFICATE.json",
    "artifacts/certified_f8_hessian_v1_20261008/HESSIAN_AND_WITNESS_CERTIFICATE.json",
    "artifacts/certified_teacher_v1_20261008/POSTHOC_RISK_BALL.json")
CODE_DOC_PATHS=("script/candidate_v18_risk_range.py","script/candidate_v18_risk_range_test.py",
    "docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_8_20261008/RISK_RANGE_CONTRACT.md",
    "docs/07_papers_论文工作/ParticleJointSupervision/submission_candidate_v1_8_20261008/RISK_RANGE_PROOF.md")
HANDOFF_SHA="b3626d6e32b302328bd719b507c5af8378b57c79699c495d321fb294e57aa36f"
REVIEWED_COMMIT="56bb45f0b137dc7e94d5b9cff348a076967d961e"
BITS=192


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(data,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")


def exact(value):
    if isinstance(value,(bool,float)):raise ValueError("exact rational str/int/Fraction required")
    return Fraction(value)


def integer(value):
    if isinstance(value,bool):raise ValueError("actual integer precision required")
    try:value=operator.index(value)
    except TypeError as error:raise ValueError("actual integer precision required") from error
    if value<64:raise ValueError("precision>=64 required")
    return value


def aq(value):
    value=exact(value)
    return arb(value.numerator)/value.denominator


def endpoint(value):
    m,e=value.man_exp()
    return dict(mantissa=str(m),exponent_of_two=int(e))


def record(value):
    if not value.is_finite():raise ArithmeticError("nonfinite Arb result")
    return dict(arb_display_only=str(value),lower_exact_dyadic=endpoint(value.lower()),
                upper_exact_dyadic=endpoint(value.upper()))


def from_endpoint(value):
    m=int(value["mantissa"]);e=operator.index(value["exponent_of_two"])
    return arb(m)*(arb(2)**e)


def from_record(value):
    lo,hi=from_endpoint(value["lower_exact_dyadic"]),from_endpoint(value["upper_exact_dyadic"])
    if lo>hi:raise ValueError("inverted exact dyadic endpoints")
    return lo.union(hi)


def ball(value):
    if isinstance(value,dict):return from_record(value)
    if isinstance(value,arb):
        if not value.is_finite():raise ValueError("finite Arb value required")
        return value
    return aq(value)


def risk_range(q_true,Q_true,root_distance_upper,risk_Hessian_upper,*,
               precision_bits=192,exact_unit_mass=True):
    """Pure saved-box arithmetic, full-physical-gradient Fisher bound 1/4.

    exact_unit_mass is a mathematical model premise, not inferred from a
    numeric sum enclosing 1. Generic fixtures must satisfy that premise too.
    """
    precision_bits=integer(precision_bits)
    if exact_unit_mass is not True:raise ValueError("exact mathematical unit mass required")
    q_true=list(q_true);Q_true=list(Q_true)
    if not q_true or len(q_true)!=len(Q_true):raise ValueError("equally sized complete alphabets required")
    previous=ctx.prec;ctx.prec=precision_bits
    try:
        q=[ball(value) for value in q_true];Q=[ball(value) for value in Q_true]
        if any(not x.is_finite() or not x.lower()>0 for x in q+Q):
            raise ValueError("all true probability balls must have strictly positive lower endpoints")
        qmass=sum(q,arb(0));Qmass=sum(Q,arb(0))
        if not qmass.contains(1) or not Qmass.contains(1):
            raise ValueError("saved probability boxes inconsistent with exact unit mass")
        rr,mm=ball(root_distance_upper),ball(risk_Hessian_upper)
        if not rr.lower()>=0 or not mm.lower()>=0:
            raise ValueError("nonnegative distance/Hessian enclosures required")
        r,M=rr.upper(),mm.upper()
        KL=arb(0);chi=arb(0);terms=[]
        for qi,Qi in zip(q,Q):
            kt=qi*(qi/Qi).log()
            ct=(qi-Qi)**2/Qi
            KL+=kt;chi+=ct
            terms.append(dict(KL=record(kt),chi_square=record(ct)))
        chiupper=chi.upper()
        if not chiupper>=0:raise ArithmeticError("chi-square upper endpoint unexpectedly negative")
        # Use an exact positive endpoint, not sqrt of a ball whose outward
        # lower endpoint could cross zero. This never clips probability balls.
        G=chiupper.sqrt()/2
        first=G.upper()*r
        second=M*r*r/2
        allowance=first+second
        rootKL=KL+arb(0,allowance.upper())
        return dict(schema="candidate_v18_saved_probability_risk_range_v1",
            precision_bits=precision_bits,python_flint_version=flint.__version__,label_count=len(q),
            scope="saved_true_probability_endpoint_arithmetic_only",
            exact_unit_mass_model_premise=True,marginal_mass_enclosures_consistent=True,
            teacher_probability_mass_enclosure=record(qmass),candidate_probability_mass_enclosure=record(Qmass),
            all_true_probability_lower_endpoints_positive=True,probability_balls_renormalized=False,
            KL_definition="sum_y q_y log(q_y/Q_y)",KL_units="nats",center_KL_enclosure=record(KL),
            chi_square_definition="sum_y (q_y-Q_y)^2/Q_y",
            chi_square_enclosure=record(chi),chi_square_upper_exact_dyadic=endpoint(chiupper),
            observed_Fisher_operator_upper_exact="1/4",
            full_physical_gradient_norm_upper_enclosure=record(G),
            root_distance_upper_exact_dyadic=endpoint(r),
            risk_Hessian_operator_upper_exact_dyadic=endpoint(M),
            first_order_root_transfer_upper=record(first),second_order_root_transfer_upper=record(second),
            total_root_transfer_upper=record(allowance),root_KL_enclosure=record(rootKL),
            center_KL_lower_positive=bool(KL.lower()>0),root_KL_lower_positive=bool(rootKL.lower()>0),
            full_gradient_bound_not_directional_derivative=True,legacy_float_gradient_used=False,
            label_terms=terms,new_grid_evaluations=0,new_integrals=0,new_PMF_evaluations=0,
            new_random_draws=0,new_optimizer_updates=0,scientific_NPZ_reads=0,
            true_root_explicitly_solved_or_optimized=False)
    finally:ctx.prec=previous


def frozen_sources():
    names=(*SOURCE_PATHS,*CODE_DOC_PATHS,MANUFACTURED.relative_to(ROOT).as_posix(),"HANDOFF.md")
    return {name:sha(ROOT/name) for name in names}


def freeze():
    """Root may call only after reviewing contract/manufacture; no actual KL."""
    manufactured=read(MANUFACTURED)
    if manufactured["status"]!="passed" or manufactured["real_probability_vectors_used"]:
        raise ValueError("passing manufactured-only qualification required")
    for name,digest in manufactured["source_sha256"].items():
        if sha(ROOT/name)!=digest:raise ValueError("manufacture source drift: "+name)
    if sha(ROOT/"HANDOFF.md")!=HANDOFF_SHA:raise ValueError("HANDOFF binding changed")
    result=dict(schema="candidate_v18_risk_range_freeze_v1",created_utc=now(),sources=frozen_sources(),
        precision_bits=BITS,label_count=256,reviewed_commit=REVIEWED_COMMIT,
        handoff_sha256=HANDOFF_SHA,handoff_use="reference",
        scope="deterministic_saved_true_probability_KL_chi_square_and_root_range",
        no_new_grid_or_integral=True,probability_mass_renormalization=False)
    write_new(FREEZE,result)
    return dict(status="frozen",path=FREEZE.relative_to(ROOT).as_posix(),new_probability_syntheses=0)


def verify_freeze():
    result=read(FREEZE)
    if result["precision_bits"]!=BITS or result["label_count"]!=256 or result["sources"]!=frozen_sources():
        raise ValueError("frozen contract/input/source drift")
    if result["handoff_sha256"]!=HANDOFF_SHA or result["reviewed_commit"]!=REVIEWED_COMMIT:
        raise ValueError("frozen review provenance drift")
    return result


def source_values():
    teacher,candidate,witness,risk=[read(ROOT/name) for name in SOURCE_PATHS]
    if (not teacher["all_true_probabilities_positive"] or teacher["true_PMF_enclosures"] is None
            or candidate["teacher_or_candidate_mass_renormalized"]):
        raise ValueError("certified nonrenormalized probability sources required")
    if (not witness["same_exact_center_across_gradient_Hessian_risk"]
            or not witness["witness"]["true_F8_stationary_existence_certified"]):
        raise ValueError("same-center true root certificate required")
    if risk["parameter_dimension"]!=24 or risk["physical_radius_exact"]!=witness["witness"]["physical_radius_exact"]:
        raise ValueError("full-physical risk-ball metric/radius drift")
    q,Q=teacher["true_PMF_enclosures"],candidate["candidate_Q_true_enclosures"]
    if len(q)!=256 or len(Q)!=256:raise ValueError("complete 256 probability vectors required")
    return q,Q,witness["witness"]["root_distance_upper"],risk["risk_Hessian_operator_uniform_upper_expression"],risk["physical_radius_exact"]


def synthesis():
    q,Q,r,M,radius=source_values()
    previous=ctx.prec;ctx.prec=BITS
    try:
        if not from_record(r).upper()<aq(radius):
            raise ValueError("root segment not inside certified risk-Hessian ball")
        return risk_range(q,Q,r,M,precision_bits=BITS)
    finally:ctx.prec=previous


def run():
    frozen=verify_freeze()
    # Exclusive start prevents repeated synthesis replacing either a success
    # or a failed attempt. A failure receipt retains the concrete exception.
    write_new(OUT/"RISK_RANGE_RUN_START.json",dict(created_utc=now(),freeze_sha256=sha(FREEZE),
        source_sha256=frozen["sources"],scope="saved_endpoint_scalar_synthesis"))
    try:result=synthesis()
    except BaseException as error:
        write_new(OUT/"RISK_RANGE_FAILURE.json",dict(created_utc=now(),error_type=type(error).__name__,
            error=str(error),freeze_sha256=sha(FREEZE),new_grid_evaluations=0,new_integrals=0))
        raise
    result.update(created_utc=now(),source_sha256=frozen["sources"],freeze_sha256=sha(FREEZE),
        run_start_sha256=sha(OUT/"RISK_RANGE_RUN_START.json"),reviewed_commit=REVIEWED_COMMIT,
        handoff_sha256=HANDOFF_SHA)
    write_new(RESULT,result)
    return dict(status="completed",path=RESULT.relative_to(ROOT).as_posix(),
        center_KL_enclosure=result["center_KL_enclosure"],root_KL_enclosure=result["root_KL_enclosure"],
        new_grid_evaluations=0,new_integrals=0,new_random_draws=0)


def check():
    frozen=verify_freeze();saved=read(RESULT)
    if (saved["source_sha256"]!=frozen["sources"] or saved["freeze_sha256"]!=sha(FREEZE)
            or saved["run_start_sha256"]!=sha(OUT/"RISK_RANGE_RUN_START.json")):
        raise ValueError("saved synthesis bindings drift")
    expected=synthesis()
    for name,value in expected.items():
        if saved.get(name)!=value:raise ValueError("saved scalar synthesis drift: "+name)
    return dict(passed=True,replayed_saved_probability_syntheses=1,
        new_grid_evaluations=0,new_integrals=0,new_PMF_evaluations=0,new_random_draws=0)


if __name__=="__main__":
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true");mode.add_argument("--run",action="store_true")
    mode.add_argument("--check",action="store_true");args=parser.parse_args()
    print(json.dumps(freeze() if args.freeze else run() if args.run else check(),ensure_ascii=False,indent=2))

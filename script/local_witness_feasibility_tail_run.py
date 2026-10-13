"""Frozen exact-rational parameter box and analytic Arb truncation readout.

No kernel, finite-domain quadrature, optimizer, RNG, or scientific array reads.
The analytic tail bounds leave finite-domain quadrature and existence open.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import local_witness_feasibility_bounds as bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/local_witness_feasibility_v1_20261008"
INPUTS = OUT / "INPUTS.json"
FREEZE = OUT / "FREEZE.json"
OUTPUT = OUT / "TAIL_TRUNCATION_BOUNDS.json"
SOURCE_NAMES = ("script/local_witness_feasibility_tail_run.py",
                "script/local_witness_feasibility_tail_run_test.py",
                "script/local_witness_feasibility_bounds.py",
                "artifacts/local_witness_feasibility_v1_20261008/INPUTS.json")
FIXED_RULES = {
    "C1": dict(normal_rule="legendre", normal_order=256, normal_limit=10.,
               logu_min=-28., logu_max=56., logu_step=.25, outer_rule="trapezoid"),
    "D1": dict(normal_rule="legendre", normal_order=384, normal_limit=11.,
               logu_min=-32., logu_max=64., logu_step=.125, outer_rule="trapezoid"),
}
TARGETS = {name: "1/1000000000" for name in ("gradient", "radial_hessian", "transverse_hessian")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def exact_binary64(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("a JSON float64-roundtrip number is required")
    value = float(value)
    if not math.isfinite(value): raise ValueError("nonfinite binary64 value")
    numerator, denominator = value.as_integer_ratio()
    return Fraction(numerator, denominator)


def argument_plan(inputs):
    """Derive exact box rationals from saved float64 inputs; no bound evaluation."""
    if inputs.get("schema") != "local_witness_feasibility_inputs_v1": raise ValueError("unexpected INPUTS schema")
    if inputs.get("dimension") != 8 or inputs.get("K_values") != [8,"inf"]: raise ValueError("fixed dimension/K drift")
    if inputs.get("rules") != FIXED_RULES or inputs.get("likelihood_scale") != 1.: raise ValueError("fixed rules/scale drift")
    if inputs.get("physical_radius") != "1/2000000" or inputs.get("physical_radius_float") != 5e-7:
        raise ValueError("fixed radius drift")
    if inputs.get("tangent_T") != 4 or inputs.get("teacher_truncation_L") != 10: raise ValueError("fixed truncation parameters drift")
    theta=inputs["center_theta"]
    if len(theta)!=16 or any(float(x)!=0. for x in theta[:8]):raise ValueError("fixed zero-bias center missing")
    a=[exact_binary64(x) for x in theta[8:]]
    if a != a[::-1] or any(x<=0 for x in a):raise ValueError("positive exact reversal center missing")
    if len(inputs["center_theta_dyadic"])!=16:raise ValueError("dyadic center record incomplete")
    for value,record in zip(theta,inputs["center_theta_dyadic"]):
        if float(value).hex()!=record["hex"] or exact_binary64(value)!=Fraction(int(record["numerator"]),int(record["denominator"])):
            raise ValueError("center dyadic record drift")
    r=Fraction(1,2000000);margin=3*r
    a_sum=sum(map(abs,a),Fraction(0))
    box=dict(radius=str(r), center_a_l1=str(a_sum), b_l1=str(margin), a_l1=str(a_sum+margin),
        l1_radius_multiplier="3", multiplier_reason="sqrt(8)<3 for each 8-coordinate physical-radius projection",
        m=8,K=8,T="4",teacherL="10",likelihood_scale="1",precision_bits=192)
    arguments={}
    for name,rule in FIXED_RULES.items():
        arguments[name]=dict(m=8,K=8,b_l1=box["b_l1"],a_l1=box["a_l1"],
            normalL=str(exact_binary64(rule["normal_limit"])),teacherL="10",
            vmin=str(exact_binary64(rule["logu_min"])),vmax=str(exact_binary64(rule["logu_max"])),T="4",
            precision_bits=192,targets=TARGETS.copy())
    return dict(parameter_box_exact=box,arguments_by_rule=arguments,targets_exact=TARGETS.copy())


def evaluate_plan(plan):
    """Two explicit scalar-bound API calls. No integrator is used."""
    rows={}
    for name,args in plan["arguments_by_rule"].items():
        record=bounds.bound_box(**args)
        rows[name]=dict(arguments_exact=args,bounds_readout=record)
    comparisons=[dict(rule=name,**comparison) for name,row in rows.items() for comparison in row["bounds_readout"]["comparisons"]]
    return dict(parameter_box_exact=plan["parameter_box_exact"],targets_exact=plan["targets_exact"],
        rules=rows,total_truncation_comparisons=comparisons,
        all_six_total_truncation_targets_passed=len(comparisons)==6 and all(x["strictly_below"] for x in comparisons),
        scope="analytic total truncation only; no finite-domain quadrature error enclosure",
        resources=dict(analytic_bound_box_calls=len(rows),new_kernel_calls=0,new_integrals=0,acb_integral_calls=0,
            new_random_draws=0,new_optimizer_updates=0,scientific_array_reads=0),
        finite_interval_quadrature_certified=False,strict_stationary_existence_certified=False)


def verify_freeze():
    freeze=read(FREEZE);sources=freeze.get("sources")
    if not isinstance(sources,dict) or any(name not in sources for name in SOURCE_NAMES):
        raise ValueError("FREEZE must bind wrapper, test, bounds and INPUTS")
    for name,digest in sources.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or not isinstance(digest,str) or sha(path)!=digest:
            raise ValueError("frozen source drift: "+name)
    inputs=read(INPUTS)
    return argument_plan(inputs),{name:sha(ROOT/name) for name in SOURCE_NAMES}


def run():
    if OUTPUT.exists():raise FileExistsError("TAIL_TRUNCATION_BOUNDS already exists; no overwrite")
    plan,sources=verify_freeze()
    payload=evaluate_plan(plan)
    payload.update(schema="local_witness_tail_truncation_readout_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE),
        source_sha256=sources,public_input_JSON_reads=1,exact_box_derived_from_saved_binary64=True)
    write_new(OUTPUT,payload)
    return dict(status="completed",all_six_total_truncation_targets_passed=payload["all_six_total_truncation_targets_passed"],
        **payload["resources"])


def check():
    plan,sources=verify_freeze();saved=read(OUTPUT)
    if saved.get("schema")!="local_witness_tail_truncation_readout_v1" or saved.get("inputs_sha256")!=sha(INPUTS) or saved.get("freeze_sha256")!=sha(FREEZE) or saved.get("source_sha256")!=sources:
        raise ValueError("tail readout source/freeze locks drift")
    rebuilt=evaluate_plan(plan)
    for key,value in rebuilt.items():
        if saved.get(key)!=value:raise ValueError("saved analytic tail scalar readout drift: "+key)
    if saved.get("exact_box_derived_from_saved_binary64") is not True or saved.get("public_input_JSON_reads")!=1:
        raise ValueError("input semantics drift")
    return dict(status="passed",new_kernel_calls=0,new_integrals=0,new_random_draws=0,scientific_array_reads=0,
        analytic_bound_box_replays=2,all_six_total_truncation_targets_passed=rebuilt["all_six_total_truncation_targets_passed"],
        validation_scope="source locks, public JSON and deterministic 192-bit Arb scalar-bound replay")


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run",action="store_true");mode.add_argument("--check",action="store_true");args=parser.parse_args()
    print(json.dumps(run() if args.run else check(),ensure_ascii=False,indent=2))


if __name__=="__main__":main()

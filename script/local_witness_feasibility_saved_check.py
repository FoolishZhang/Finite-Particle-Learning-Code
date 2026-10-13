"""Supplemental saved-array check preserving the frozen checker failure.

Hperp/Hangle follow the kernel's aggregate-then-assemble operation order;
aggregate-after-label-assembly is recorded separately, never tolerance-masked.
No kernel, integral, RNG, or old scientific array is read or executed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import local_witness_feasibility_compute as frozen

OUT=frozen.OUT
DIAGNOSIS=OUT/"SAVED_CHECK_DIAGNOSIS.json"
RECEIPT=OUT/"SAVED_CHECK_MANUFACTURED_CHECK.json"
SOURCE_NAMES=("script/local_witness_feasibility_saved_check.py","script/local_witness_feasibility_saved_check_test.py")


def require_equal(name,actual,expected):
    if not np.array_equal(actual,expected):raise ValueError("saved arithmetic drift: "+name)


def aggregate_diagnosis(result):
    q=result["targets"];a=result["a"];m=len(a);ii=np.arange(m)
    combine=lambda array:np.einsum("tp,kp...->tk...",q,array,optimize=True)
    output={}
    for aggregate,label in (("loss","label_loss"),("gradient","label_gradient"),("H_ba","label_H_ba")):
        require_equal(aggregate,result[aggregate],combine(result[label]))
        output[aggregate]=dict(direct_label_aggregation_Linf=0.,direct_label_aggregation_unequal_entries=0,kernel_operation_order_bitwise_equal=True)
    # The kernel assembles each label Hba before target aggregation.
    require_equal("label_H_ba",result["label_H_ba"],result["label_B"]-result["label_same_particle_term"]+result["label_cross_particle_term"])
    M=combine(result["label_M"]);sigma=combine(result["label_sigma_prime_average"])
    perpendicular=-M.copy();perpendicular[...,ii,ii]+=sigma
    edges=M*a[None,None,:,None]*a[None,None,None,:];edges[...,ii,ii]=0
    angular=-edges.copy();angular[...,ii,ii]=edges.sum(axis=-1)
    label_perpendicular=-result["label_M"].copy();label_perpendicular[...,ii,ii]+=result["label_sigma_prime_average"]
    label_edges=result["label_M"]*a[None,:,None]*a[None,None,:];label_edges[...,ii,ii]=0
    label_angular=-label_edges.copy();label_angular[...,ii,ii]=label_edges.sum(axis=-1)
    require_equal("label_Hperp",result["label_Hperp"],label_perpendicular)
    require_equal("label_Hangle",result["label_Hangle"],label_angular)
    for name,label,rebuilt in (("Hperp","label_Hperp",perpendicular),("Hangle","label_Hangle",angular)):
        require_equal(name,result[name],rebuilt)
        difference=np.abs(result[name]-combine(result[label]))
        output[name]=dict(direct_label_aggregation_Linf=float(difference.max()),
            direct_label_aggregation_unequal_entries=int(np.count_nonzero(difference)),
            kernel_operation_order_Linf=0.,kernel_operation_order_bitwise_equal=True)
    require_equal("edge_weights",result["edge_weights"],edges)
    ward=perpendicular*a[None,None,:,None]*a[None,None,None,:]
    ward[...,ii,ii]-=a*result["gradient"][...,m:]
    require_equal("directWard",result["directWard"],ward)
    require_equal("ward_residual",result["ward_residual"],ward-angular)
    require_equal("bias_gradient",result["bias_gradient"],result["gradient"][...,:m])
    require_equal("radial_gradient",result["radial_gradient"],result["gradient"][...,m:])
    return output


def current_locks(summary):
    files={str(frozen.INPUTS.relative_to(frozen.ROOT)).replace("\\","/"):frozen.sha(frozen.INPUTS),
           str(frozen.FREEZE.relative_to(frozen.ROOT)).replace("\\","/"):frozen.sha(frozen.FREEZE),
           "artifacts/local_witness_feasibility_v1_20261008/NOMINAL_SUMMARY.json":frozen.sha(OUT/"NOMINAL_SUMMARY.json"),
           "script/local_witness_feasibility_compute.py":frozen.sha(Path(frozen.__file__)),
           "script/rankone_stationary_kernel.py":frozen.sha(Path(frozen.kernel.__file__))}
    for row in summary["rules"].values():
        entry=row["array_artifact"];files[entry["path"]]=frozen.sha(frozen.ROOT/entry["path"])
    return files


def validate_saved():
    """All frozen check gates, with exact kernel-order matrix reconstruction."""
    inputs,_=frozen.verify_freeze();summary=frozen.read(OUT/"NOMINAL_SUMMARY.json");start=frozen.read(OUT/"RUN_START.json")
    if summary["inputs_sha256"]!=frozen.sha(frozen.INPUTS) or summary["freeze_sha256"]!=frozen.sha(frozen.FREEZE) or summary["run_start_sha256"]!=frozen.sha(OUT/"RUN_START.json"):
        raise ValueError("saved run locks drift")
    if start["inputs_sha256"]!=frozen.sha(frozen.INPUTS) or start["freeze_sha256"]!=frozen.sha(frozen.FREEZE):raise ValueError("start locks drift")
    results,diagnostics={},{}
    for name in frozen.RULES:
        row=summary["rules"][name];entry=row["array_artifact"]
        if entry!=frozen.artifact(frozen.ROOT/entry["path"]):raise ValueError("new array artifact drift")
        result=frozen.rebuild_saved(frozen.ROOT/entry["path"],row)
        if result["K_values"]!=frozen.KS or result["target_names"]!=("population",):raise ValueError("target/K drift")
        require_equal("center",result["theta"],inputs["center_theta"])
        require_equal("target",result["targets"][0],inputs["teacher_pmf"])
        require_equal("b",result["b"],result["theta"][:8]);require_equal("a",result["a"],result["theta"][8:])
        diagnostics[name]=aggregate_diagnosis(result)
        expected=frozen.readout(result,result["theta"])
        for key,value in expected.items():
            if row[key]!=value:raise ValueError("derived readout drift: "+name+"/"+key)
        if frozen.read(OUT/(name+".json"))!=row:raise ValueError("per-rule JSON drift")
        conf=result["config"]
        if conf["compute_hessian"] is not True or conf["coordinate_order"]!="b_then_a":raise ValueError("kernel config drift")
        for key,value in frozen.RULES[name].items():
            if conf[key]!=value:raise ValueError("saved rule drift")
        if conf["likelihood_scale_source"]!="supplied_positive_scale":raise ValueError("unit scale drift")
        require_equal("unit log scale",result["log_likelihood_scale"],np.zeros(256))
        n=frozen.RULES[name]["normal_order"];outer=round((frozen.RULES[name]["logu_max"]-frozen.RULES[name]["logu_min"])/frozen.RULES[name]["logu_step"])+1
        counts=dict(targets=1,labels=256,normal_nodes=n,outer_nodes=outer,finite_K_greater_than_one=1,
            bernoulli_logit_evaluations=n*8,conditional_loglikelihood_values=256*n,
            laplace_exp_minus_ux_factors=256*outer*n,finite_d_coefficients=256*n,
            finite_c_coefficients=256*n,finite_cross_particle_outer_products=256*outer,
            interpretation="high_level_scalar_factor_counts_not_CPU_instruction_counts")
        if result["work_counts"]!=counts:raise ValueError("fixed work counts drift")
        results[name]=result
    if summary["qualifications"]!=frozen.summarize_pair(results) or summary["risk_direction"]!=frozen.risk_direction(results["D1"]):raise ValueError("pair/risk arithmetic drift")
    if summary["old_to_center_nominal"]!=inputs["old_to_center_nominal"]:raise ValueError("projection drift")
    if summary["center"]!=dict(theta=inputs["center_theta"],original_theta=inputs["old_theta"],full24_theta=inputs["center_full24_theta"],exact_center_dyadic=inputs["center_theta_dyadic"]):raise ValueError("summary center drift")
    resources=summary["resources"]
    if resources["kernel_calls"]!=2 or resources["target_K_readouts"]!=4:raise ValueError("call budget drift")
    totals={}
    for result in results.values():
        for key,value in result["work_counts"].items():
            if isinstance(value,int):totals[key]=totals.get(key,0)+value
    if resources["high_level_factor_totals"]!=totals:raise ValueError("factor totals drift")
    return summary,diagnostics


def diagnose():
    if DIAGNOSIS.exists():raise FileExistsError("diagnosis already exists; no overwrite")
    try:
        frozen.check()
    except ValueError as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            frozen_checker_modified=False,failure_preserved=True)
        if str(error)!="saved label aggregation drift: Hperp":raise
    else:raise ValueError("expected frozen checker failure was not reproduced")
    summary,diagnostics=validate_saved()
    sources={name:frozen.sha(frozen.ROOT/name) for name in SOURCE_NAMES}
    output=dict(schema="local_witness_saved_check_diagnosis_v1",created_utc=datetime.now(timezone.utc).isoformat(),
        original_failure=failure,source_sha256=sources,evidence_sha256=current_locks(summary),rules=diagnostics,
        source_locations=dict(frozen_failure="script/local_witness_feasibility_compute.py:336-338",
            kernel_target_aggregation="script/rankone_stationary_kernel.py:303-305",
            kernel_Hperp_assembly="script/rankone_stationary_kernel.py:310-312",
            kernel_Hangle_assembly="script/rankone_stationary_kernel.py:313-316",
            alternate_per_label_assembly="script/rankone_stationary_kernel.py:320-325"),
        Hangle_same_class_issue_found_before_it_was_reached_by_frozen_checker=True,
        diagnosis="frozen checker combines already assembled per-label Hperp/Hangle; kernel combines label_M/sigma first then assembles matrices; float64 operations are not distributive bitwise",
        resolution="supplemental checker reconstructs original kernel arithmetic exactly; no tolerance relaxation, science-source or raw change",
        saved_kernel_order_matrices_bitwise_equal=True,scientific_artifacts_modified=False,
        frozen_scientific_sources_and_raw_artifact_hashes_unchanged=True,
        independent_integration_performed=False,new_kernel_calls=0,new_integrals=0,new_random_draws=0,
        new_saved_array_reads_for_diagnosis=3,old_scientific_array_reads=0,
        strict_stationary_existence_certified=False)
    frozen.write_new(DIAGNOSIS,output)
    return dict(status="diagnosed",new_kernel_calls=0,new_integrals=0,rules=diagnostics)


def check():
    summary,diagnostics=validate_saved();diagnosis=frozen.read(DIAGNOSIS);receipt=frozen.read(RECEIPT)
    sources={name:frozen.sha(frozen.ROOT/name) for name in SOURCE_NAMES}
    if diagnosis["source_sha256"]!=sources or receipt["source_sha256"]!=sources:raise ValueError("supplemental checker source drift")
    if diagnosis["evidence_sha256"]!=current_locks(summary) or diagnosis["rules"]!=diagnostics:raise ValueError("diagnosis evidence drift")
    if diagnosis["original_failure"]["message"]!="saved label aggregation drift: Hperp" or not diagnosis["original_failure"]["failure_preserved"]:raise ValueError("old failure record missing")
    return dict(status="passed",saved_array_reads=2,new_kernel_calls=0,new_integrals=0,new_random_draws=0,
        original_frozen_check_failure_preserved=True,kernel_order_matrices_bitwise_equal=True,
        validation_scope="source/raw/hash, saved label components, kernel-order aggregation, config/counts, readouts/qualification/risk; no independent integration")


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--diagnose",action="store_true");modes.add_argument("--check",action="store_true");args=parser.parse_args()
    print(json.dumps(diagnose() if args.diagnose else check(),ensure_ascii=False,indent=2))


if __name__=="__main__":main()

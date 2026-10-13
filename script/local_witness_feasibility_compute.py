"""Fixed nominal center readout; no optimization or rigorous integral certificate.

--prepare-inputs reads public JSON only. --run requires the author's FREEZE,
makes exactly two rank-one kernel calls, and refuses existing RUN_START.
--check reconstructs new saved readouts without calling the kernel.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import rankone_stationary_kernel as kernel
import rankone_stationary_compute as previous

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/local_witness_feasibility_v1_20261008"
INPUTS = OUT / "INPUTS.json"
FREEZE = OUT / "FREEZE.json"
BASELINE = "e924b13643e09a9ad190b71c03cabeac501b3971"
OLD_SOURCE = "artifacts/rankone_stationary_v1_20261007/SOURCE.json"
OLD_SOLVER = "artifacts/rankone_stationary_v1_20261007/optimization/P/SOLVER.json"
SOURCE_PATHS = (OLD_SOURCE, OLD_SOLVER, "script/rankone_stationary_kernel.py",
                "script/rankone_stationary_compute.py")
RULES = {
    "C1": dict(normal_rule="legendre", normal_order=256, normal_limit=10.,
               logu_min=-28., logu_max=56., logu_step=.25, outer_rule="trapezoid"),
    "D1": dict(normal_rule="legendre", normal_order=384, normal_limit=11.,
               logu_min=-32., logu_max=64., logu_step=.125, outer_rule="trapezoid"),
}
KS = (8, "inf")


def native(value):
    if isinstance(value, np.ndarray): return native(value.tolist())
    if isinstance(value, np.generic): return native(value.item())
    if isinstance(value, dict): return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [native(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("nonfinite JSON")
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(native(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def artifact(path):
    return dict(path=rel(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def now():
    return datetime.now(timezone.utc).isoformat()


def dyadic_records(values):
    result = []
    for value in np.asarray(values, dtype=np.float64).ravel():
        x = float(value)
        numerator, denominator = x.as_integer_ratio()
        result.append(dict(value=x, hex=x.hex(), numerator=str(numerator), denominator=str(denominator)))
    return result


def symmetric_center(old):
    old = np.asarray(old, dtype=np.float64)
    if old.shape != (16,) or not np.isfinite(old).all(): raise ValueError("old theta must have 16 finite values")
    center = np.zeros(16, dtype=np.float64)
    for j in range(4):
        value = np.float64(np.float64(old[8+j] + old[15-j]) / np.float64(2.))
        center[8+j] = center[15-j] = value
    return center


def prepared_inputs():
    source, solver = read(ROOT / OLD_SOURCE), read(ROOT / OLD_SOLVER)
    old = np.asarray(solver["final_theta"], dtype=np.float64)
    q = np.asarray(source["teacher_pmf"], dtype=np.float64)
    ps = next(s for s in source["solves"] if s["name"] == "P")
    if solver["name"] != "P" or solver["status"] != "completed": raise ValueError("saved P not completed")
    if q.shape != (256,) or not np.isfinite(q).all() or np.any(q <= 0): raise ValueError("invalid saved q")
    if not np.array_equal(q, solver["fit_target"]) or not np.array_equal(q, ps["target"]): raise ValueError("saved target mismatch")
    center = symmetric_center(old)
    delta = center - old
    return native(dict(schema="local_witness_feasibility_inputs_v1", baseline_commit=BASELINE,
        source_files={p: sha(ROOT / p) for p in SOURCE_PATHS},
        parameter_center_source_sha256=sha(ROOT / OLD_SOLVER), parameter_center_source_definition="saved population P SOLVER JSON; no random seed",
        original_source_links=source["sources"],
        input_semantics="JSON numbers parsed as float64; hex/ratios define the exact dyadic saved inputs, not decimal literals",
        old_theta=old, old_theta_dyadic=dyadic_records(old), old_full24_theta=np.r_[old[:8], old[8:], np.zeros(8)],
        center_theta=center, center_theta_dyadic=dyadic_records(center), center_full24_theta=np.r_[center[:8], center[8:], np.zeros(8)],
        center_rule="b=0; each reversal pair receives the same float64((old_a_i+old_a_7minus_i)/2)",
        old_to_center_nominal=dict(delta=delta, Linf=float(np.max(abs(delta))), L2=float(np.linalg.norm(delta)),
            old_bias_Linf=float(np.max(abs(old[:8]))), old_reversal_a_Linf=float(np.max(abs(old[8:]-old[8:][::-1]))),
            exact_dyadic_distance_certificate_in_separate_output=True),
        teacher_pmf=q, teacher_pmf_dyadic=dyadic_records(q), teacher_mass_nominal=float(q.sum()),
        teacher_renormalized=False, teacher_symmetrized=False,
        physical_radius="1/2000000", physical_radius_float=5e-7, tangent_T=4, teacher_truncation_L=10,
        target_names=["population"], K_values=KS, dimension=8, rules=RULES,
        likelihood_scale=1., physical_u="exp(v); unlike old default evidence scaling",
        compute_hessian=True, return_coefficients=False,
        qualification_thresholds={k: previous.GATES[k] for k in ("loss", "gradient", "matrix", "eigenvalue", "normal_mass", "d_relative_mass", "c_raw_mass", "Ward", "PMF_difference", "PMF_mass", "KL_difference")},
        symmetry_parity_residual_threshold=2e-7,
        fixed_slice_definition="V_j=e_j-(a0_j/a0_7)e7, j=0..6; saved float ratios nominal only; exact ratios recoverable from center dyadics",
        fixed_risk_direction_rule="each h reversal pair receives identical float64((D1 inf ga_i+ga_reverse_i)/2); saved h dyadics define mathematical d=-h/sqrt(h.T h), with b/transverse zero",
        call_budget=2, target_K_readout_budget=4, new_optimizer_updates=0, new_random_draws=0,
        new_teacher_PMF_evaluations=0, adaptive_rule_or_radius_changes=False,
        rigorous_integral_error_or_existence_certificate=False))


def validate_inputs():
    saved = read(INPUTS)
    if saved != prepared_inputs(): raise ValueError("INPUTS drift from the fixed public source/algorithm/configuration")
    return saved


def verify_freeze():
    inputs = validate_inputs()
    freeze = read(FREEZE)
    sources = freeze.get("sources")
    if not isinstance(sources, dict) or not sources: raise ValueError("FREEZE.sources must be a nonempty path-to-SHA256 map")
    required = (rel(INPUTS), "script/local_witness_feasibility_compute.py", "script/local_witness_feasibility_compute_test.py")
    if any(p not in sources for p in required): raise ValueError("FREEZE omits compute/input/test source")
    for name, digest in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file(): raise ValueError("invalid FREEZE source path")
        if not isinstance(digest, str) or sha(path) != digest: raise ValueError("frozen source drift: " + name)
    return inputs, freeze


def reversal_basis(m):
    if m % 2: raise ValueError("even dimension required")
    plus, minus = np.zeros((m, m//2)), np.zeros((m, m//2))
    for j in range(m//2):
        plus[j,j] = plus[m-1-j,j] = 1 / math.sqrt(2)
        minus[j,j] = 1 / math.sqrt(2); minus[m-1-j,j] = -1 / math.sqrt(2)
    return plus, minus


def fixed_slice(a):
    a = np.asarray(a, dtype=float)
    if a[-1] == 0: raise ValueError("last center amplitude must be nonzero")
    V = np.zeros((len(a), len(a)-1))
    V[:-1] = np.eye(len(a)-1)
    V[-1] = -a[:-1] / a[-1]
    Q, _ = np.linalg.qr(V, mode="reduced")
    return V, Q


def sym(A):
    return .5 * (A + A.T)


def eig(A):
    return np.linalg.eigvalsh(sym(A))


def readout(result, theta):
    m = len(result["a"])
    plus, minus = reversal_basis(m)
    V, Q = fixed_slice(result["a"])
    rows = []
    for ki, K in enumerate(result["K_values"]):
        H, P = result["H_ba"][0,ki], result["Hperp"][0,ki]
        g = result["gradient"][0,ki]
        Hbb, Haa, Hba = H[:m,:m], H[m:,m:], H[:m,m:]
        rows.append(dict(K=K, loss=result["loss"][0,ki], gradient=g,
            H_ba=H, Hperp=P, H_ba_symmetric=sym(H), Hperp_symmetric=sym(P), Hangle=result["Hangle"][0,ki],
            full16_eigenvalues=eig(H), bias8_eigenvalues=eig(Hbb),
            a_symmetric4_matrix=plus.T@Haa@plus, a_antisymmetric4_matrix=minus.T@Haa@minus,
            a_symmetric4_eigenvalues=eig(plus.T@Haa@plus), a_antisymmetric4_eigenvalues=eig(minus.T@Haa@minus),
            fixed_slice_raw7_matrix=V.T@P@V, fixed_slice_raw7_eigenvalues=eig(V.T@P@V),
            fixed_slice_QR7_matrix=Q.T@P@Q, fixed_slice_QR7_eigenvalues=eig(Q.T@P@Q),
            fixed_slice_raw_spectrum_is_not_orthonormal=True,
            nominal_spectra_not_certified=True, gradient_Linf=float(np.max(abs(g))), gradient_L2=float(np.linalg.norm(g)),
            parity_reversal_residuals=dict(bias_gradient_Linf=float(np.max(abs(g[:m]))),
                bias_radial_mixed_Linf=float(np.max(abs(Hba))),
                radial_gradient_reversal_Linf=float(np.max(abs(g[m:]-g[m:][::-1]))),
                Hbb_reversal_Linf=float(np.max(abs(Hbb-Hbb[::-1,::-1]))),
                Haa_reversal_Linf=float(np.max(abs(Haa-Haa[::-1,::-1]))),
                Hperp_reversal_Linf=float(np.max(abs(P-P[::-1,::-1]))),
                radial_plus_minus_mixed_Linf=float(np.max(abs(plus.T@Haa@minus)))),
            Cartesian_Ward_Linf=float(np.max(abs(P@result["a"]-g[m:]))),
            scalar_diagnostics=previous.scalar_diagnostics(result,ki)))
    q = result["targets"][0]
    reverse_index = kernel.patterns(m)[:,::-1] @ (2**np.arange(m-1,-1,-1))
    return native(dict(theta=theta, blocks=rows,
        fixed_slice_basis=V, fixed_slice_QR_basis=Q, fixed_slice_Gram=V.T@V,
        fixed_slice_basis_exact_certification=False,
        reversal_plus_basis=plus, reversal_minus_basis=minus,
        probability=previous.probability_summary(result,q),
        teacher_quality=dict(raw_mass=float(q.sum()), normalization_applied=False, symmetrization_applied=False,
            complement_Linf=float(np.max(abs(q-q[::-1]))), reversal_Linf=float(np.max(abs(q-q[reverse_index])))),
        config=result["config"], log_likelihood_scale=result["log_likelihood_scale"],
        normal_quadrature_mass=result["normal_quadrature_mass"], work_counts=result["work_counts"],
        kernel_qualifications=result.get("qualifications",dict(rigorous_Gaussian_remainder_certificate=False,
            rigorous_outer_truncation_certificate=False,signs_or_eigenvalues_clipped=False,coefficient_masses_renormalized=False)),
        numerical_qualification_only=True, rigorous_integral_error_or_existence_certificate=False))


def risk_direction(D):
    m = len(D["a"]); ki = list(D["K_values"]).index("inf")
    g = D["gradient"][0,ki]
    h = np.zeros(m)
    for j in range(m//2): h[j] = h[m-1-j] = np.float64((g[m+j]+g[2*m-1-j])/2)
    norm = float(np.linalg.norm(h))
    if not norm > 0: raise ValueError("predefined risk direction is undefined")
    d = np.r_[np.zeros(m), -h/norm, np.zeros(m)]
    return native(dict(h=h, h_dyadic=dyadic_records(h), h_L2_nominal=norm,
        direction_full24_nominal=d, mathematical_direction="(0,-h/sqrt(h.T h),0), h defined by exact saved dyadics after float64 pair-average rounding",
        direction_is_unit_only_nominal_in_JSON=True,
        integrated_gradient_ba=g, integrated_gradient_L2=float(np.linalg.norm(g)),
        dot_nominal=float(g[m:]@d[m:2*m]), negative_h_norm_nominal=-norm,
        reversal_residual_Linf=float(np.max(abs(h-h[::-1]))),
        original_radial_projection_residual_Linf=float(np.max(abs(g[m:]-h))),
        nominal_unit_residual=abs(float(d@d)-1),
        rigorous_risk_direction_certificate=False))


def snapshot_arrays(result, theta):
    arrays = {k:v.copy() for k,v in result.items() if isinstance(v,np.ndarray)}
    arrays.update(theta=np.asarray(theta).copy(), normal_quadrature_mass=np.asarray(result["normal_quadrature_mass"]),
        target_names=np.asarray(result["target_names"]), K_names=np.asarray([str(k) for k in result["K_values"]]))
    V,Q = fixed_slice(result["a"])
    arrays.update(fixed_slice_basis=V, fixed_slice_QR_basis=Q)
    return arrays


def rebuild_saved(path, summary):
    with np.load(path, allow_pickle=False) as raw:
        result = {k:raw[k].copy() for k in raw.files}
    result["target_names"] = tuple(result.pop("target_names").tolist())
    result["K_values"] = tuple(int(k) if k != "inf" else "inf" for k in result.pop("K_names").tolist())
    result["normal_quadrature_mass"] = float(result["normal_quadrature_mass"])
    result["config"] = summary["config"]
    result["work_counts"] = summary["work_counts"]
    result["qualifications"] = summary["kernel_qualifications"]
    previous.project_result(result)
    return result


def summarize_pair(results):
    C,D = results["C1"],results["D1"]
    quals = [dict(K=K, **previous.qualification(C,D,0,ki)) for ki,K in enumerate(KS)]
    pc,pd = previous.probability_summary(C,C["targets"][0]),previous.probability_summary(D,D["targets"][0])
    symmetrics = [readout(result,result["theta"])["blocks"] for result in (C,D)]
    parity_max = max(v for rows in symmetrics for row in rows for v in row["parity_reversal_residuals"].values())
    prob = dict(PMF_Linf=float(np.max(abs(pc["pmf"]-pd["pmf"]))), KL_difference=abs(pc["raw_KL"]-pd["raw_KL"]))
    prob_gates = dict(PMF_difference=prob["PMF_Linf"]<=previous.GATES["PMF_difference"],
        KL_difference=prob["KL_difference"]<=previous.GATES["KL_difference"],
        C1_PMF_mass=pc["pmf_mass_error"]<=previous.GATES["PMF_mass"], D1_PMF_mass=pd["pmf_mass_error"]<=previous.GATES["PMF_mass"])
    return native(dict(blocks=quals, probability_comparisons=prob, probability_gates=prob_gates,
        maximum_parity_reversal_residual=parity_max, parity_reversal_gate=parity_max<=2e-7,
        all_numerical_gates_passed=all(x["qualified"] for x in quals) and all(prob_gates.values()) and parity_max<=2e-7,
        rigorous_remainder_or_existence_certificate=False))


def run():
    inputs,_ = verify_freeze()
    start = OUT / "RUN_START.json"
    if start.exists(): raise FileExistsError("RUN_START already exists; no overwrite or automatic retry")
    write_new(start,dict(created_utc=now(), inputs_sha256=sha(INPUTS), freeze_sha256=sha(FREEZE),
        prescribed_kernel_calls=2, prescribed_target_K_readouts=4, likelihood_scale=1.,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            threads={k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")})))
    theta=np.asarray(inputs["center_theta"],dtype=float);q=np.asarray(inputs["teacher_pmf"],dtype=float)
    results, rows = {}, {}
    tick_all=time.perf_counter()
    for name,rule in RULES.items():
        tick=time.perf_counter()
        result=kernel.rankone_stationary(theta[:8].copy(),theta[8:].copy(),{"population":q.copy()},
            K_values=KS,**rule,likelihood_scale=1.,compute_hessian=True,return_coefficients=False)
        seconds=time.perf_counter()-tick
        previous.project_result(result)
        result["theta"]=theta.copy()
        path=OUT/(name+".npz")
        with path.open("xb") as stream:np.savez_compressed(stream,**snapshot_arrays(result,theta))
        row=readout(result,theta);row.update(array_artifact=artifact(path),kernel_wall_seconds=seconds)
        write_new(OUT/(name+".json"),row)
        results[name]=result;rows[name]=row
    totals={}
    for row in rows.values():
        for key,value in row["work_counts"].items():
            if isinstance(value,int):totals[key]=totals.get(key,0)+value
    summary=dict(schema="local_witness_feasibility_compute_summary_v1",created_utc=now(),
        inputs_sha256=sha(INPUTS),freeze_sha256=sha(FREEZE),run_start_sha256=sha(start),
        center=dict(theta=inputs["center_theta"],original_theta=inputs["old_theta"],full24_theta=inputs["center_full24_theta"],
            exact_center_dyadic=inputs["center_theta_dyadic"]),
        old_to_center_nominal=inputs["old_to_center_nominal"],rules=rows,
        qualifications=summarize_pair(results),risk_direction=risk_direction(results["D1"]),
        resources=dict(kernel_calls=2,target_K_readouts=4,kernel_wall_seconds=sum(r["kernel_wall_seconds"] for r in rows.values()),
            total_execution_wall_seconds=time.perf_counter()-tick_all,high_level_factor_totals=totals,
            new_random_draws=0,new_optimizer_updates=0,new_teacher_PMF_evaluations=0),
        zero_new_exact_integral_certificate=True,strict_stationary_existence_certified=False)
    write_new(OUT/"NOMINAL_SUMMARY.json",summary)
    return native(summary["resources"])


def check():
    inputs,_=verify_freeze();summary=read(OUT/"NOMINAL_SUMMARY.json");start=read(OUT/"RUN_START.json")
    if summary["inputs_sha256"]!=sha(INPUTS) or summary["freeze_sha256"]!=sha(FREEZE) or summary["run_start_sha256"]!=sha(OUT/"RUN_START.json"):
        raise ValueError("saved run locks drift")
    if start["inputs_sha256"]!=sha(INPUTS) or start["freeze_sha256"]!=sha(FREEZE):raise ValueError("start locks drift")
    results={}
    for name in RULES:
        row=summary["rules"][name];entry=row["array_artifact"]
        if entry!=artifact(ROOT/entry["path"]):raise ValueError("new array artifact drift")
        result=rebuild_saved(ROOT/entry["path"],row)
        if result["K_values"]!=KS or result["target_names"]!=("population",):raise ValueError("target/K drift")
        if not np.array_equal(result["theta"],inputs["center_theta"]) or not np.array_equal(result["targets"][0],inputs["teacher_pmf"]):raise ValueError("saved center/target drift")
        if not np.array_equal(result["b"],result["theta"][:8]) or not np.array_equal(result["a"],result["theta"][8:]):raise ValueError("kernel coordinate drift")
        for aggregate,labels in (("loss","label_loss"),("gradient","label_gradient"),("H_ba","label_H_ba"),("Hperp","label_Hperp"),("Hangle","label_Hangle")):
            rebuilt_aggregate=np.einsum("tp,kp...->tk...",result["targets"],result[labels],optimize=True)
            if not np.array_equal(result[aggregate],rebuilt_aggregate):raise ValueError("saved label aggregation drift: "+aggregate)
        expected=readout(result,result["theta"])
        for key,value in expected.items():
            if row[key]!=value:raise ValueError("derived readout drift: "+name+"/"+key)
        if read(OUT/(name+".json"))!=row:raise ValueError("per-rule JSON drift")
        conf=result["config"]
        if conf["compute_hessian"] is not True or conf["coordinate_order"]!="b_then_a":raise ValueError("kernel configuration drift")
        for key,value in RULES[name].items():
            if conf[key]!=value:raise ValueError("saved rule drift")
        if conf["likelihood_scale_source"]!="supplied_positive_scale" or not np.array_equal(result["log_likelihood_scale"],np.zeros(256)):
            raise ValueError("fixed unit scale drift")
        n=RULES[name]["normal_order"];outer=round((RULES[name]["logu_max"]-RULES[name]["logu_min"])/RULES[name]["logu_step"])+1
        expected_counts=dict(targets=1,labels=256,normal_nodes=n,outer_nodes=outer,finite_K_greater_than_one=1,
            bernoulli_logit_evaluations=n*8,conditional_loglikelihood_values=256*n,
            laplace_exp_minus_ux_factors=256*outer*n,finite_d_coefficients=256*n,
            finite_c_coefficients=256*n,finite_cross_particle_outer_products=256*outer,
            interpretation="high_level_scalar_factor_counts_not_CPU_instruction_counts")
        if result["work_counts"]!=expected_counts:raise ValueError("work counts do not match fixed shape and rule")
        results[name]=result
    if summary["qualifications"]!=summarize_pair(results) or summary["risk_direction"]!=risk_direction(results["D1"]):raise ValueError("pair/risk arithmetic drift")
    if summary["old_to_center_nominal"]!=inputs["old_to_center_nominal"]:raise ValueError("projection drift")
    if summary["center"]!=dict(theta=inputs["center_theta"],original_theta=inputs["old_theta"],full24_theta=inputs["center_full24_theta"],exact_center_dyadic=inputs["center_theta_dyadic"]):raise ValueError("summary center drift")
    counts=summary["resources"]
    if counts["kernel_calls"]!=2 or counts["target_K_readouts"]!=4:raise ValueError("call budget drift")
    totals={}
    for result in results.values():
        for key,value in result["work_counts"].items():
            if isinstance(value,int):totals[key]=totals.get(key,0)+value
    if counts["high_level_factor_totals"]!=totals:raise ValueError("factor totals drift")
    return dict(status="passed",saved_array_reads=2,new_kernel_calls=0,new_integrals=0,
        validation_scope="new saved arrays/hash and deterministic readout arithmetic, not independent integration")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-inputs",action="store_true");modes.add_argument("--run",action="store_true");modes.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.prepare_inputs:
        OUT.mkdir(parents=True,exist_ok=True);write_new(INPUTS,prepared_inputs());output=dict(status="inputs_prepared",real_kernel_calls=0)
    elif args.run:output=run()
    else:output=check()
    print(json.dumps(native(output),ensure_ascii=False,indent=2))


if __name__=="__main__":main()

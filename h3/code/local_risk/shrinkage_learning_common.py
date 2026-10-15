"""LocalShrinkage execution helpers; derived from the frozen GroupPosterior helpers; imports never sample or fit."""
from __future__ import annotations

from pathlib import Path
import shutil
import time
import numpy as np
from scipy.stats import qmc

import active_reference_run as old
from group_posterior_marginal import GroupPosteriorMarginal, RejectionExhausted
from group_posterior_objective import PreparedGroupBanks

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "local_shrinkage_execution_v1"
BASELINE = "3430d60cb067332135ca341242d153c62df333c5"
HANDOFF_SHA = "31a7468ea339e5ef160a4fb90bc266aa12312cd866f1860ab4906b87d3644637"
read, write_json, write_npz, sha, serial = old.read, old.write_json, old.write_npz, old.sha, old.serial
SAMPLE_COUNTERS = ("qmc_uniform_scalars", "qmc_trial_nets", "rejection_candidate_coordinates",
                   "posterior_sample_coordinates", "rejection_private_node_terms")


class ExecutionBudget(old.Budget):
    def snapshot(self):
        result = super().snapshot()
        result["QMC_uncertainty_unit"] = "independent_replicate_including_all_6K_rejection_trial_nets"
        result["rejection_private_node_terms_scope"] = "requested_candidate_coordinates_times_inner_order; not_measured_CPU_operations"
        return result


def validate(contract, phase):
    old.require(contract["schema"] == SCHEMA and contract["phase"] == phase, "explicit phase schema required")
    old.require(not contract.get("draft_not_authorized_to_run", False), "draft cannot execute")
    old.require(contract["baseline_commit"] == BASELINE and contract["handoff_sha256"] == HANDOFF_SHA, "feedback identity differs")
    old.require(contract["model"]["M"] == 8 and contract["model"]["tau"] == 3. and contract["model"]["K"] == 8,
                "fixed m3 M8 K8 model required")
    old.require(np.allclose(contract["model"]["epsilon"], np.full(3, 1.4 / np.sqrt(8)), rtol=0, atol=1e-15), "fixed local couplings required")
    old.require(contract["initial_coordinates"] == [[0., 0., 0.], [.2, -.15, .1]], "two original starts retained")
    old.require(contract["sampling"]["maximum_trials"] == 8 and contract["sampling"]["bits"] == 52,
                "bounded rejection and finite-bit net must be declared")
    old.require(contract["group_control"] == "uniform_logratio", "frozen scalar control required")
    old.require(contract["risk_orders"] == [[20,128],[28,192],[36,256]], "bounded ordinary risk qualification required")
    for name, value in contract["caps"].items():
        old.require(isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) and value > 0,
                    "positive global cap required: " + name)
    required = set(old.REQUIRED_SOURCES) | {"script/shrinkage_learning_common.py",
        "script/group_posterior_marginal.py","script/group_posterior_objective.py",contract["producer_source"],
        "docs/08_research_研究工作/LocalShrinkage_v1_20261015/PLAN.md",
        "artifacts/local_shrinkage_v1_20261015/input_reference/IDENTITY.json",
        "artifacts/local_shrinkage_v1_20261015/input_reference/HANDOFF.md"}
    old.require(required.issubset(contract["source_sha256"]), "complete production/reference source closure required")
    for name, expected in contract["source_sha256"].items():
        path = Path(name)
        old.require(not path.is_absolute() and ".." not in path.parts, "safe repository-relative source required")
        old.require(sha(REPO / path) == expected, "source changed before execution: " + name)
    old.require(sha(REPO / "HANDOFF.md") == HANDOFF_SHA, "original feedback changed before execution")
    return contract


def start(contract_path, out, phase):
    contract_path, out = Path(contract_path).resolve(), Path(out).resolve()
    contract = validate(read(contract_path), phase)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(contract_path, out / "CONTRACT.json")
    sources = {}
    for name, expected in contract["source_sha256"].items():
        source = REPO / name; target = out / "SEALED_SOURCE" / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        old.require(sha(source) == sha(target) == expected, "source changed during freeze")
        sources[name] = {"path":target.relative_to(out).as_posix(),"sha256":expected}
    write_json(out / "FREEZE.json", {"schema":SCHEMA, "phase":phase,
        "contract_sha256":sha(contract_path), "source_snapshots":sources,
        "before_scientific_RNG_and_fitting":True,
        "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    return contract, ExecutionBudget(contract)


def new_budget(contract, started, caps=None):
    budget = ExecutionBudget({**contract,"caps":contract["caps"] if caps is None else caps})
    budget.started = started
    return budget


def combine_budgets(contract, root, workers):
    budgets = [root, *workers]
    counts = {}
    for budget in budgets:
        for key,value in budget.counts.items(): counts[key] = counts.get(key,0) + value
    for key,cap in contract["caps"].items():
        if key != "seconds": old.require(counts.get(key,0) <= cap, "combined global cap exceeded: " + key)
    elapsed = time.perf_counter() - root.started
    old.require(elapsed < contract["caps"]["seconds"], "shared wall cap exceeded")
    return {"counts":counts,"caps":contract["caps"],"seconds":elapsed,
            "budget_scope":"root_plus_disjoint_worker_counts; common_wall_not_sum",
            "qmc_unit":"one_replicate_includes_all_independent_trial_nets",
            "particle_pattern_terms_are_work_not_independent_samples":True}


def make_marginal(contract, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    m = contract["model"]
    sampler = GroupPosteriorMarginal(M=m["M"],epsilon=m["epsilon"],tau=m["tau"],**contract["marginal"])
    moments = sampler.moments()
    write_json(destination / "MOMENTS.json",moments)
    write_json(destination / "ENVELOPE.json",sampler.envelope_diagnostics())
    low = sampler.cdf(outer_order=256)
    high = sampler.cdf(outer_order=384)
    difference = float(np.max(np.abs(np.asarray(high["cdf"])-np.asarray(low["cdf"]))))
    write_json(destination / "CDF_REFERENCE.json",{"low":low,"high":high,"maximum_order_difference":difference})
    old.require(difference <= contract["qualification"]["cdf_tol"], "posterior CDF quadrature qualification failed before RNG")
    return sampler, moments


def make_banks(contract, sampler, settings, role, teacher_id, destination, budget):
    """Generate independent complete 6K nets per replicate and rejection trial.

    All eight nets are allocated even when fewer are used by rejection; both
    allocated and actually used costs remain explicit. No retry or seed change.
    """
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    R,B,K = int(settings["R"]),int(settings["B"]),int(contract["model"]["K"])
    A,bits = contract["sampling"]["maximum_trials"],contract["sampling"]["bits"]
    old.require(B > 0 and B & (B-1) == 0 and R > 0, "power-of-two whole-bank B required")
    role_id = {"fit":10,"evaluation":20,"sensitivity":30}[role]
    size = R*A*B*K*6
    budget.check("qmc_uniform_scalars",size); budget.check("qmc_trial_nets",R*A)
    u = np.full((R,A,B,K,3,2),np.nan); states = []
    try:
        for rep in range(R):
            for trial in range(A):
                entropy = [contract["seed_family"],role_id,int(teacher_id),K,rep,trial]
                rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))
                net = qmc.Sobol(d=6*K,scramble=True,bits=bits,rng=rng)
                values = net.random_base2(int(np.log2(B)))
                # Prechecked allocation cap; record completed nets before a
                # noninterior or elapsed-wall failure can hide their work.
                budget.counts["qmc_uniform_scalars"] += values.size
                budget.counts["qmc_trial_nets"] += 1
                u[rep,trial] = values.reshape(B,K,3,2)
                states.append({"replicate":rep,"trial":trial,"entropy":entropy,
                               "rng_state_after_net_generation":rng.bit_generator.state})
                old.require(np.all(values > 0) and np.all(values < 1), "noninterior finite-bit uniform; no repair")
                budget.check()
    except Exception as error:
        write_npz(destination / "FAILED_PARTIAL_UNIFORMS.npz",{"uniforms":u})
        write_json(destination / "FAILED_NET_GENERATION.json",{"error":repr(error),"states":states,
            "budget":budget.snapshot(),"NaN_entries_were_never_generated":True,"automatic_retry":False})
        raise
    write_npz(destination / "UNIFORMS.npz",{"uniforms":u})
    write_json(destination / "RNG.json",{"role":role,"teacher_id":teacher_id,"R":R,"A":A,"B":B,"K":K,
        "dimension_per_net":6*K,"bits":bits,"states":states,
        "all_trial_nets_independently_randomized":True,
        "state_coupling_not_independent_observations":True})
    worst = R*A*B*K*3*sampler.S
    budget.check("rejection_candidate_coordinates",worst)
    budget.check("rejection_private_node_terms",worst*sampler.inner_order)
    try:
        result = sampler.sample_rejection(u)
    except Exception as error:
        result = getattr(error,"result",None)
        if result is not None:
            budget.counts["rejection_candidate_coordinates"] += result["candidate_coordinates"]
            budget.counts["rejection_private_node_terms"] += result["candidate_coordinates"]*sampler.inner_order
            accepted = result.get("accepted_trial")
            budget.counts["posterior_sample_coordinates"] += 0 if accepted is None else int(np.sum(accepted > 0))
            write_npz(destination / "FAILED_PARTIAL_SAMPLES.npz", {k:v for k,v in result.items() if isinstance(v,np.ndarray)})
            write_json(destination / "FAILED_REJECTION.json", {**{k:v for k,v in result.items() if not isinstance(v,np.ndarray)},
                "budget":budget.snapshot(),"automatic_retry":False})
        else:
            write_json(destination / "FAILED_REJECTION.json",{"error":repr(error),
                "partial_sampler_cost_unavailable":True,"budget":budget.snapshot(),"automatic_retry":False})
        raise
    budget.counts["rejection_candidate_coordinates"] += result["candidate_coordinates"]
    budget.counts["rejection_private_node_terms"] += result["candidate_coordinates"]*sampler.inner_order
    budget.counts["posterior_sample_coordinates"] += result["x_group"].size
    arrays = {k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    write_npz(destination / "SAMPLES.npz", arrays)
    metadata = {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    metadata.update(driver_generated_uniform_scalars=size,driver_generated_trial_nets=R*A,
                    sampler_internal_zero_RNG_does_not_mean_driver_zero_RNG=True,
                    maximum_accepted_trial=int(result["accepted_trial"].max()),
                    used_trial_nets=int(np.sum(result["candidates_by_trial"] > 0)))
    write_json(destination / "RECEIPT.json",metadata)
    expected = sampler.cdf(outer_order=384)
    x = result["x_group"]
    sample_mean = x.mean(axis=(1,2)); sample_second = (x*x).mean(axis=(1,2))
    sample_cdf = np.stack([(x <= anchor).mean(axis=(1,2)) for anchor in expected["anchors"]],axis=-1)
    def recovery(values,target):
        return {"replicate_estimates":values,"mean":values.mean(axis=0),
            "reference":target,"difference":values.mean(axis=0)-target,
            "replicate_SE":values.std(axis=0,ddof=1)/np.sqrt(R) if R>1 else None}
    write_json(destination / "POSTERIOR_RECOVERY.json",{
        "mean":recovery(sample_mean,np.asarray(result["proposal_mean"])),
        "second_moment":recovery(sample_second,np.asarray(result["proposal_second_moment"])),
        "cdf":recovery(sample_cdf,np.asarray(expected["cdf"])),"anchors":expected["anchors"],
        "scope":"descriptive_sampling_diagnostic; no_new_tests_or_deletion; states_and_banks_not_independent_replicates"})
    for v in arrays.values(): v.setflags(write=False)
    budget.check()
    return result


def prepare(samples, patterns):
    return PreparedGroupBanks(samples["x_group"],samples["log_marginal_weight"],
        patterns["state_indices"],patterns["weights"],proposal_kind="rejection_continuous_finite_private_GH_group_posterior",
        proposal_mean=samples["proposal_mean"],proposal_second_moment=samples["proposal_second_moment"])


def group_value(prepared, coordinates, budget, contract):
    terms = prepared.R*prepared.B*prepared.K*len(prepared.weights)
    budget.check("particle_pattern_terms",terms)
    budget.add("group_objective_calls");budget.add("particle_pattern_terms",terms)
    result = prepared.evaluate(old.legacy.correlation_matrix(np.asarray(coordinates)),
                              control=contract["group_control"],pattern_chunk=contract["pattern_chunk"])
    budget.check()
    return result


def save_group_result(path, result):
    path = Path(path)
    write_npz(path.with_suffix(".npz"),{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    write_json(path.with_suffix(".json"),{k:v for k,v in result.items() if not isinstance(v,np.ndarray)})


def manifest(out):
    out = Path(out)
    write_json(out / "MANIFEST.json",{"schema":SCHEMA,"files":[
        {"path":p.relative_to(out).as_posix(),"sha256":sha(p),"bytes":p.stat().st_size}
        for p in sorted(out.rglob("*")) if p.is_file()],"manifest_self_hash_excluded":True})

"""Independent numerical replay of the locked marginal-preserving profiles.

No profile/kernel scoring routine is imported or called. Source training
artifacts are hash-bound; optimizer trajectories are the separate training
audit's responsibility. No RNG or new quadrature order is used here.
"""
from pathlib import Path
from datetime import datetime, timezone
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
import math
import time
import numpy as np
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/joint_dependence_v1_20261007"
TARGET = OUT / "PROFILE_AUDIT.json"
INDICES = (0, 17, 511, 2048, 4095)
NAMES = ("REF", "H8", "H64")
TARGETS = ("empirical", "population")
KS = (8, 64)
REDUCTION_TOL = 2e-11
DIRECT_MC_TOL = 2e-10
DIRECT_GH_TOL = 2e-10
LABELS = ((np.arange(256)[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.float64)


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def close(actual, expected, tolerance=REDUCTION_TOL):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(("shape or finiteness", a.shape, b.shape))
    error = float(np.max(abs(a-b))) if a.size else 0.
    if error > tolerance:
        raise AssertionError(("numerical discrepancy", error, tolerance))
    return error


def parameters(row):
    b = np.asarray(row["b"], dtype=float)
    W = np.asarray(row["W"], dtype=float)
    theta = np.asarray(row["theta"], dtype=float)
    assert b.shape == (8,) and W.shape == (8, 2) and theta.shape == (24,)
    assert np.isfinite(theta).all() and np.array_equal(theta, np.r_[b, W.ravel()])
    return b, W


def conditional_probabilities(b, W, nodes):
    """Direct Bernoulli products over coordinates, unlike the log-matrix kernel."""
    eta = np.asarray(nodes) @ W.T + b
    yes, no = expit(eta), expit(-eta)
    result = np.ones((len(nodes), 256), dtype=float)
    for j in range(8):
        result *= np.where(LABELS[None, :, j] == 1, yes[:, j, None], no[:, j, None])
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise FloatingPointError("direct conditional-product reference became invalid")
    return result


def direct_bank_logloss(b, W, bank):
    conditional = conditional_probabilities(b, W, bank)
    return math.log(len(bank))-logsumexp(np.log(conditional), axis=0)


def direct_rotated_pmf(b, W):
    """Independent physicists' Hermite rule, exactly the already used 72/pi7 rule."""
    x, w = np.polynomial.hermite.hermgauss(72)
    x = math.sqrt(2)*x
    w = w/math.sqrt(math.pi)
    grid = np.array([(u, v) for u in x for v in x], dtype=float)
    angle = math.pi/7
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    grid = grid @ R.T
    weights = np.outer(w, w).ravel()
    conditional = conditional_probabilities(b, W, grid)
    # math.fsum is independent of the producer's logsumexp reduction.
    q = np.array([math.fsum((weights*conditional[:, j]).tolist()) for j in range(256)])
    assert np.isfinite(q).all() and np.all(q > 0)
    return q


def nll(frequency, q):
    return -math.fsum(float(p)*math.log(float(v)) for p, v in zip(frequency, q))


def kl(P, q):
    return math.fsum(float(p)*(math.log(float(p))-math.log(float(v))) for p, v in zip(P, q))


def marginals(q):
    return np.array([math.fsum(q[LABELS[:, j] == 1].tolist()) for j in range(8)])


def independent_stats(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (4096,) and np.isfinite(values).all()
    mean = math.fsum(values.tolist())/len(values)
    variance = math.fsum((float(x)-mean)**2 for x in values)/(len(values)-1)
    sd = math.sqrt(variance)
    return dict(mean=mean, sample_sd=sd, MCSE=sd/math.sqrt(len(values)), banks=len(values))


def audit():
    freeze_path = OUT/"PROFILE_FREEZE.json"
    frozen = read(freeze_path)
    assert frozen["profile_parameters_locked_before_profile_risk"] is True
    assert frozen["staged_exploratory"] is True and frozen["confirmatory"] is False
    assert frozen["config"]["banks"] == 4096 and frozen["config"]["particles"] == [8, 64]
    assert frozen["config"]["seed"] == 202610070300 and frozen["config"]["row_norm_threshold"] == 1e-10
    assert frozen["reference_config"]["rotated_order"] == 72
    assert frozen["reference_config"]["rotation"] == math.pi/7
    config = frozen["reference_config"]
    verified = {}
    def bind(path, digest, size=None):
        path = Path(path).resolve()
        assert path.is_relative_to(ROOT)
        name = rel(path)
        if name in verified:
            assert verified[name] == digest
        else:
            assert sha(path) == digest, name
            verified[name] = digest
        if size is not None:assert path.stat().st_size == size
    for path, digest in frozen["sources"].items():
        bind(ROOT/path, digest)
    pre = read(OUT/"PROFILE_PREDECLARATION.json")
    assert pre["training_result_files_present"] == []
    assert pre["design_sha256"] == sha(ROOT/"docs/08_research_研究工作/JointDependence_v1_20261007/PROFILE_DESIGN.md")
    assert [s["rep"] for s in frozen["streams"]] == list(range(4))
    all_rows = []
    max_mc = max_gh = max_statistics = max_exact = max_marginal = 0.
    signs = {"positive": 0, "negative": 0, "zero": 0}
    for rep, stream in enumerate(frozen["streams"]):
        source = ROOT/stream["folder"]
        for path, digest in stream["sources"].items():
            bind(ROOT/path, digest)
        train_params = read(source/"PARAMS.json")
        train_lock = read(source/"CANDIDATES_LOCK.json")
        train_result = read(source/"RESULTS.json")
        train_qualification = read(source/"ENDPOINT_QUALIFICATION.json")
        assert train_result["status"] == "completed" and train_result["rep"] == rep
        assert train_result["params_sha256"] == train_lock["params_sha256"] == sha(source/"PARAMS.json")
        assert train_result["candidates_lock_sha256"] == train_qualification["candidates_lock_sha256"] == sha(source/"CANDIDATES_LOCK.json")
        assert train_result["candidates"] == train_lock["candidates"]
        assert train_qualification["all_qualified"] and all(r["qualified"] for r in train_qualification["endpoints"])
        assert train_lock["locked_before_teacher_risk_and_independent_cross_evaluation"]
        assert not train_params["teacher_used_by_training_function"]
        assert datetime.fromisoformat(pre["created_utc"]) < datetime.fromisoformat(train_lock["created_utc"])
        for entry in train_result["artifacts"].values():
            bind(ROOT/entry["path"], entry["sha256"], entry["bytes"])
        with np.load(source/"INPUTS.npz", allow_pickle=False) as data:
            labels = data["labels"].astype(np.int64)
            f_saved = data["frequency"]
        powers = 2**np.arange(7, -1, -1)
        frequency = np.bincount(labels@powers, minlength=256)/len(labels)
        assert np.array_equal(frequency, f_saved)
        assert np.array_equal(frequency, np.asarray(train_params["frequency"]))
        assert np.array_equal(frequency, np.asarray(stream["frequency"]))

        folder = OUT/"profile"/f"rep_{rep:02}"
        result = read(folder/"RESULTS.json")
        lock = read(folder/"CANDIDATES_LOCK.json")
        qualification = read(folder/"ENDPOINT_QUALIFICATION.json")
        assert result["status"] == "completed" and result["rep"] == rep
        assert not (folder/"FAILURE.json").exists()
        assert result["profile_freeze_sha256"] == lock["profile_freeze_sha256"] == sha(freeze_path)
        assert result["source_stream"] == stream["folder"]
        assert result["candidates_lock_sha256"] == qualification["candidates_lock_sha256"] == sha(folder/"CANDIDATES_LOCK.json")
        assert lock["source_candidate_lock_sha256"] == sha(source/"CANDIDATES_LOCK.json")
        assert lock["locked_before_profile_teacher_risk_and_prior"] and not lock["teacher_alignment_used"]
        assert lock["candidates"] == result["profiles"] == stream["profiles"]
        assert tuple(c["name"] for c in lock["candidates"]) == NAMES
        assert result["endpoints"] == qualification["endpoints"] and qualification["all_qualified"]
        assert qualification["source_teacher_qualification_sha256"] == sha(source/"ENDPOINT_QUALIFICATION.json")
        for entry in result["artifacts"].values():
            bind(ROOT/entry["path"], entry["sha256"], entry["bytes"])
        bind(folder/"RESULTS.json", sha(folder/"RESULTS.json"))
        with np.load(folder/"EVALUATION.npz", allow_pickle=False) as saved:
            losses=saved["loss"];paired=saved["paired_loss"];prior=saved["prior"]
            empirical=saved["frequency"];P=saved["teacher_pmf"]
            names=tuple(saved["candidate_names"].tolist())
            left=tuple(saved["comparison_left"].tolist());right=tuple(saved["comparison_right"].tolist())
            particles=saved["particles"];targets=tuple(saved["target_names"].tolist())
        with np.load(folder/"EVALUATION_PRIOR.npz", allow_pickle=False) as saved:
            assert np.array_equal(prior, saved["prior"])
        assert names == NAMES and left == ("H8", "H64") and right == ("REF", "REF")
        assert np.array_equal(particles, KS) and targets == TARGETS
        assert losses.shape == (4096,3,2,2) and paired.shape == (4096,2,2,2)
        assert prior.shape == (4096,64,2)
        assert np.isfinite(losses).all() and np.isfinite(prior).all()
        assert ah(prior) == result["evaluation_prior_sha256"]
        assert result["seed"] == [202610070300, rep, 0]
        for value in (empirical, np.asarray(lock["frequency"]), np.asarray(result["frequency"])):
            assert np.array_equal(value, frequency)
        assert np.array_equal(P, np.asarray(result["teacher_pmf"]))
        assert np.array_equal(P, np.asarray(qualification["teacher_pmf"]))
        assert np.array_equal(P, np.asarray(train_qualification["teacher_pmf"]))
        assert np.array_equal(paired[:,0], losses[:,1]-losses[:,0])
        assert np.array_equal(paired[:,1], losses[:,2]-losses[:,0])

        source_candidates={c["name"]:c for c in train_lock["candidates"]}
        reference_b, reference_W=parameters(source_candidates["REF"])
        reference_norm=np.sqrt(np.sum(reference_W**2, axis=1))
        parameter_rows=[]
        for candidate in lock["candidates"]:
            b,W=parameters(candidate);name=candidate["name"]
            assert candidate["defined"] and np.array_equal(b,reference_b)
            if name=="REF":
                assert np.array_equal(W, reference_W)
                expected_W=reference_W
            else:
                donor=source_candidates["K8" if name=="H8" else "K64"]
                _,donor_W=parameters(donor)
                donor_norm=np.sqrt(np.sum(donor_W**2,axis=1))
                assert np.all(reference_norm>1e-10) and np.all(donor_norm>1e-10)
                expected_W=reference_norm[:,None]*(donor_W/donor_norm[:,None])
                assert candidate["source_direction"] == donor["name"]
                assert not candidate["teacher_alignment_used"] and not candidate["common_rotation_alignment_used"]
            reconstruction_error=close(W,expected_W,2e-15)
            norm_error=close(np.sqrt(np.sum(W**2,axis=1)),reference_norm,2e-15)
            assert not candidate["box"]["projected_or_repaired"]
            feasible=bool(np.all(abs(b)<=2) and np.all(abs(W)<=2))
            assert candidate["box"]["feasible"] == feasible
            assert candidate["box"]["maximum_excess"] == max(0.,float(np.max(abs(b)))-2,float(np.max(abs(W)))-2)
            parameter_rows.append(dict(name=name,bias_equal_exactly=True,
                maximum_rule_reconstruction_error=reconstruction_error,
                maximum_REF_norm_difference=norm_error,box_feasible=feasible,clipped=False))

        stored_means={(r["candidate"],r["K"],r["target"]):r for r in result["cross_evaluation"]["means"]}
        stored_pairs={(r["left"],r["K"],r["target"]):r for r in result["cross_evaluation"]["paired_differences"]}
        assert len(stored_means)==12 and len(stored_pairs)==8
        pairs_out=[]
        for ki,K in enumerate(KS):
            for ti,target in enumerate(TARGETS):
                for ci,name in enumerate(names):
                    stats=independent_stats(losses[:,ci,ki,ti]);old=stored_means[name,K,target]
                    for key in ("mean","sample_sd","MCSE"):
                        max_statistics=max(max_statistics,close(stats[key],old[key]))
                    assert old["banks"] == stats["banks"]
                for pi,name in enumerate(left):
                    stats=independent_stats(paired[:,pi,ki,ti]);old=stored_pairs[name,K,target]
                    for key in ("mean","sample_sd","MCSE"):
                        max_statistics=max(max_statistics,close(stats[key],old[key]))
                    assert old["banks"] == stats["banks"]
                    signs["positive" if old["mean"]>0 else "negative" if old["mean"]<0 else "zero"]+=1
                    pairs_out.append(dict(left=name,right="REF",K=K,target=target,
                                          stored_mean=old["mean"],stored_MCSE=old["MCSE"],**stats))

        endpoints={r["name"]:r for r in qualification["endpoints"]}
        exact_values={};independent_q={};gh_rows=[]
        for candidate in lock["candidates"]:
            name=candidate["name"];b,W=parameters(candidate);row=endpoints[name]
            q=np.asarray(row["pmf"]);assert np.all(q>0) and np.isfinite(q).all()
            assert row["qualified"] and row["marginals_qualified"]
            exact_values[name]=dict(empirical_NLL=nll(frequency,q),risk_KL=kl(P,q))
            max_exact=max(max_exact,close(exact_values[name]["empirical_NLL"],row["empirical_NLL"]),
                          close(exact_values[name]["risk_KL"],row["risk_KL"]))
            close(marginals(q),row["marginals"])
            reference_marginals=np.asarray(endpoints["REF"]["marginals"])
            largest_marginal=max(float(np.max(abs(marginals(np.asarray(v["pmf"]))-reference_marginals))) for v in row["variants"])
            close(largest_marginal,row["maximum_marginal_difference_from_REF"])
            assert largest_marginal<=2e-7
            assert bool(row["original_box_claim_allowed"]) == bool(candidate["box"]["feasible"])
            rq=direct_rotated_pmf(b,W);independent_q[name]=rq
            rotation_row=next(v for v in row["variants"] if v["order"]==72 and abs(v["rotation"]-math.pi/7)<1e-15)
            pe=close(rq,rotation_row["pmf"],DIRECT_GH_TOL)
            le=close(nll(frequency,rq),rotation_row["empirical_NLL"],DIRECT_GH_TOL)
            ke=close(kl(P,rq),rotation_row["risk_KL"],DIRECT_GH_TOL)
            mass=abs(float(rq.sum())-1);assert mass<=config["mass_tolerance"]
            main_pmf_discrepancy=float(np.max(abs(rq-q)))
            main_risk_discrepancy=abs(kl(P,rq)-row["risk_KL"])
            assert main_pmf_discrepancy<=config["pmf_tolerance"]
            assert main_risk_discrepancy<=config["risk_tolerance"]
            max_gh=max(max_gh,pe,le,ke)
            gh_rows.append(dict(candidate=name,order=72,rotation=math.pi/7,raw_mass_error=mass,
                 PMF_error_vs_saved_rotated72=pe,NLL_error_vs_saved_rotated72=le,
                 KL_error_vs_saved_rotated72=ke,PMF_difference_vs_main60=main_pmf_discrepancy,
                 KL_difference_vs_main60=main_risk_discrepancy,
                 independent_empirical_NLL=nll(frequency,rq),independent_KL=kl(P,rq),pmf_sha256=ah(rq)))
        independent_marginal=max(float(np.max(abs(marginals(independent_q[name])-marginals(independent_q["REF"])))) for name in NAMES)
        assert independent_marginal<=2e-7;max_marginal=max(max_marginal,independent_marginal)
        exact_rows=[]
        for row in result["exact_differences"]:
            name=row["left"];assert row["right"]=="REF"
            dN=exact_values[name]["empirical_NLL"]-exact_values["REF"]["empirical_NLL"]
            dK=exact_values[name]["risk_KL"]-exact_values["REF"]["risk_KL"]
            max_exact=max(max_exact,close(dN,row["empirical_NLL_difference"]),close(dK,row["risk_KL_difference"]))
            exact_rows.append(dict(left=name,right="REF",empirical_NLL_difference=dN,risk_KL_difference=dK,
                stored_empirical_NLL_difference=row["empirical_NLL_difference"],
                stored_risk_KL_difference=row["risk_KL_difference"]))
        mc_rows=[]
        for bi in INDICES:
            for ci,candidate in enumerate(lock["candidates"]):
                b,W=parameters(candidate)
                for ki,K in enumerate(KS):
                    logloss=direct_bank_logloss(b,W,prior[bi,:K,:])
                    for ti,target in enumerate((frequency,P)):
                        value=math.fsum(float(a)*float(v) for a,v in zip(target,logloss))
                        error=close(value,losses[bi,ci,ki,ti],DIRECT_MC_TOL);max_mc=max(max_mc,error)
                        mc_rows.append(dict(bank=bi,candidate=candidate["name"],K=K,target=TARGETS[ti],
                                            independent_loss=value,saved_loss=float(losses[bi,ci,ki,ti]),absolute_error=error))
        assert not result["cross_evaluation"]["statistical_family_claimed"]
        assert not result["claims"]["all_training_risk_difference_explained"]
        all_rows.append(dict(rep=rep,result_sha256=sha(folder/"RESULTS.json"),
            profile_lock_sha256=sha(folder/"CANDIDATES_LOCK.json"),parameter_checks=parameter_rows,
            paired_statistics=pairs_out,exact_reference_differences=exact_rows,
            fixed_bank_checks=mc_rows,independent_GH72=gh_rows,
            independent_maximum_marginal_difference_from_REF=independent_marginal))
    return dict(passed=True,scope="Numerical/source/profile-construction audit only; no scientific interpretation.",
        profile_freeze_sha256=sha(freeze_path),verified_input_files=verified,
        source_training_trajectory_audit_in_scope=False,uses_source_candidate_locks_not_redundant_path_theta=True,
        reps=4,profile_candidates=12,all_biases_preserved=True,all_row_norms_preserved_numerically=True,
        profile_construction_matches_locked_donor_directions=True,teacher_alignment_used=False,box_clipping_used=False,
        saved_bank_loss_cells_checked=4*4096*3*2*2,saved_paired_cells_checked=4*4096*2*2*2,
        fixed_bank_indices=list(INDICES),independent_bank_loss_checks=4*5*3*2*2,
        independent_rotated_GH_order=72,independent_GH_PMFs=12,independent_GH_nodes=12*72**2,
        maximum_independent_MC_error=max_mc,maximum_independent_GH_error=max_gh,
        maximum_saved_statistics_error=max_statistics,maximum_reference_NLL_KL_error=max_exact,
        maximum_independent_marginal_difference=max_marginal,observed_paired_mean_sign_counts=signs,
        no_confidence_or_resolution_classification_added=True,all_paired_means_and_MCSEs_preserved=True,
        new_random_draws=0,new_banks=0,new_orders_tried=0,profile_or_kernel_score_functions_called=False,
        tolerances=dict(saved_reductions=REDUCTION_TOL,direct_MC=DIRECT_MC_TOL,direct_rotated_GH=DIRECT_GH_TOL,
                        marginal=2e-7,PMF=config["pmf_tolerance"],risk=config["risk_tolerance"]),
        streams=all_rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if not args.check and TARGET.exists():raise FileExistsError(TARGET)
    started=time.perf_counter();validation=audit()
    if args.check:
        saved=read(TARGET)
        assert saved["source_sha256"]==sha(__file__)
        assert saved["validation"]==validation
    else:
        record=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
                    wall_seconds=time.perf_counter()-started,validation=validation)
        with TARGET.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(passed=True,check=args.check,reps=4,
          max_MC_error=validation["maximum_independent_MC_error"],
          max_GH_error=validation["maximum_independent_GH_error"],
          saved_mean_signs=validation["observed_paired_mean_sign_counts"],
          seconds=time.perf_counter()-started)),flush=True)

"""Frozen per-stream dense joint training and locked-endpoint evaluation.

The learner consumes only empirical frequencies, initialization and numerical
prior inputs. Teacher probabilities enter only after CANDIDATES_LOCK.json.
Four streams are exploratory; bank MCSEs condition on the locked endpoints.
"""
from __future__ import annotations
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
import traceback
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
import joint_dependence_kernel as kernel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/joint_dependence_v1_20261007"
DESIGN = ROOT / "docs/08_research_研究工作/JointDependence_v1_20261007/DESIGN.md"
ARMS = ("K8", "K64", "GH")
NAMES = (*ARMS, "REF", "IND")
CONFIG = dict(seed=202610070100, reps=4, observations=4096, dimension=8, latent=2,
              steps=1200, step_prior=128, particles=[8, 64], bias_bounds=[-2., 2.],
              loading_bounds=[-2., 2.], initial_loading_sd=.25,
              learning_rates=[.02, .005], switch_step=800, beta1=.9, beta2=.999,
              adam_epsilon=1e-8, gh_order=20, checkpoint_every=300,
              path_reference_order=40, path_loss_tolerance=1e-5,
              path_gradient_tolerance=1e-4,
              ref_order=40, ref_optimizer=dict(maxiter=300, maxfun=450, maxls=25,
                                              gtol=1e-7, ftol=1e-12),
              endpoint_orders=[40, 60, 80], main_reference_order=60,
              rotated_order=72, rotation=math.pi/7, pmf_tolerance=2e-7,
              risk_tolerance=2e-6, mass_tolerance=2e-12,
              evaluation_banks=4096, evaluation_particles=64,
              evaluation_block=128, pattern_chunk_size=32)
PHI = np.array([-.7, -.5, -.3, -.1, .1, .3, .5, .7], dtype=np.float64)


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(item) for item in value]
    return value


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(native(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_arrays(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def read(path):
    return json.loads(Path(path).read_text("utf-8"))


def unpack(theta):
    theta = np.asarray(theta, dtype=np.float64)
    m = len(theta)//3
    if theta.ndim != 1 or len(theta) != 3*m:
        raise ValueError("theta must contain b then row-major two-column W")
    return theta[:m], theta[m:].reshape(m, 2)


def pack(b, W):
    return np.concatenate((np.asarray(b), np.asarray(W).ravel())).astype(np.float64)


def frequency(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.isin(labels, (0, 1)).all():
        raise ValueError("binary observation matrix required")
    powers = 2**np.arange(labels.shape[1]-1, -1, -1)
    return np.bincount(labels.astype(np.int64)@powers, minlength=2**labels.shape[1])/len(labels)


def initial_parameters(labels, loading):
    p = (np.asarray(labels).sum(axis=0)+.5)/(len(labels)+1)
    raw = pack(np.log(p)-np.log1p(-p), loading)
    initial = np.clip(raw, -2., 2.)
    return p, raw, initial


def objective(theta, probs, *, order=None, banks=None, angle=0.):
    b, W = unpack(theta)
    if (order is None) == (banks is None):
        raise ValueError("provide exactly one of GH order and MC banks")
    if order is not None:
        value, gb, gW = kernel.exact_loss_grad(b, W, probs, order=order, angle=angle)
    else:
        value, gb, gW = kernel.mc_loss_grad(b, W, probs, banks)
    gradient = pack(gb, gW)
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite training objective/gradient")
    return value, gradient


def adam_step(theta, gradient, first, second, step, lr, config):
    first = config["beta1"]*first+(1-config["beta1"])*gradient
    second = config["beta2"]*second+(1-config["beta2"])*gradient**2
    corrected_first = first/(1-config["beta1"]**step)
    corrected_second = second/(1-config["beta2"]**step)
    proposal = theta-lr*corrected_first/(np.sqrt(corrected_second)+config["adam_epsilon"])
    updated = np.clip(proposal, -2., 2.)
    if not np.isfinite(updated).all():
        raise FloatingPointError("nonfinite projected Adam state")
    return updated, first, second, proposal


def path_check(theta, probs, config, step):
    loss, gradient = objective(theta, probs, order=config["gh_order"])
    loss_ref, gradient_ref = objective(theta, probs, order=config["path_reference_order"])
    le = abs(loss-loss_ref);ge = float(np.max(abs(gradient-gradient_ref)))
    return dict(step=step,theta=theta,loss=loss,gradient=gradient,
                reference_loss=loss_ref,reference_gradient=gradient_ref,
                loss_difference=le,gradient_max_difference=ge,
                qualified=le<=config["path_loss_tolerance"] and ge<=config["path_gradient_tolerance"])


def train_joint(probs, initial, training_prior, *, config=CONFIG, folder=None):
    """No teacher argument or teacher access: full b/W joint training only."""
    probs=np.asarray(probs,dtype=np.float64);initial=np.asarray(initial,dtype=np.float64)
    prior=np.asarray(training_prior,dtype=np.float64);steps=config["steps"]
    if prior.shape != (steps,128,2) or initial.shape != (3*config["dimension"],):
        raise ValueError("training shape does not match the frozen config")
    theta=np.repeat(initial[None,:],3,axis=0);first=np.zeros_like(theta);second=np.zeros_like(theta)
    parameters=np.full((steps+1,3,len(initial)),np.nan);parameters[0]=theta
    losses=np.full((steps,3),np.nan);gradients=np.full((steps,3,len(initial)),np.nan)
    proposals=np.full_like(gradients,np.nan);projection=np.zeros_like(gradients,dtype=bool)
    checkpoints=[];completed=0;calls=np.zeros(3,dtype=np.int64);timing=np.zeros(3)
    try:
        check=path_check(theta[2],probs,config,0);checkpoints.append(check)
        if folder:save(folder/"PATH_0000.json",check)
        if not check["qualified"]:raise ArithmeticError("GH path qualification failed at step 0")
        for step in range(1,steps+1):
            lr=config["learning_rates"][0 if step<=config["switch_step"] else 1]
            for arm,K in enumerate((8,64,None)):
                tick=time.perf_counter();calls[arm]+=1
                if K is None:value,g=objective(theta[arm],probs,order=config["gh_order"])
                else:value,g=objective(theta[arm],probs,banks=prior[step-1].reshape(128//K,K,2))
                losses[step-1,arm]=value;gradients[step-1,arm]=g
                updated,first[arm],second[arm],proposal=adam_step(theta[arm],g,first[arm],second[arm],step,lr,config)
                proposals[step-1,arm]=proposal;projection[step-1,arm]=proposal!=updated
                theta[arm]=updated;timing[arm]+=time.perf_counter()-tick
            parameters[step]=theta;completed=step
            if step%config["checkpoint_every"]==0 or step==steps:
                check=path_check(theta[2],probs,config,step);checkpoints.append(check)
                if folder:
                    save_arrays(folder/f"CHECKPOINT_{step:04}.npz",theta=theta,loss_before_update=losses[step-1],
                                gradient_before_update=gradients[step-1],proposal=proposals[step-1],
                                projection=projection[step-1],adam_first=first,adam_second=second)
                    save(folder/f"PATH_{step:04}.json",check)
                if not check["qualified"]:raise ArithmeticError(f"GH path qualification failed at step {step}")
                if folder:print(json.dumps(dict(rep_folder=folder.name,step=step,loss=losses[step-1].tolist())),flush=True)
        traces=dict(theta=parameters,loss_before_update=losses,gradient_before_update=gradients,
                    proposal=proposals,projection_mask=projection,training_prior=prior)
        meta=dict(completed_steps=completed,arms=ARMS,objective_calls=calls,arm_seconds=timing,
                  path_checks=checkpoints,projection_counts=projection.sum(axis=(0,2)),
                  readout="LAST only; loss/gradient tape is at the pre-update theta",
                  teacher_passed_to_learner=False)
        return theta,traces,meta
    except BaseException:
        if folder:
            save_arrays(folder/"TRAINING_PARTIAL.npz",theta=parameters[:completed+1],
                        loss_before_update=losses[:completed],gradient_before_update=gradients[:completed],
                        proposal=proposals[:completed],projection_mask=projection[:completed],
                        current_partial_step_theta=theta,adam_first=first,adam_second=second)
            save(folder/"TRAINING_PARTIAL.json",dict(completed_steps=completed,objective_calls=calls,
                 arm_seconds=timing,path_checks=checkpoints,traceback=traceback.format_exc()))
        raise


def fit_reference(probs, initial, *, config=CONFIG):
    xs=[];values=[];gradients=[];tick=time.perf_counter()
    def fun(theta):
        value,g=objective(theta,probs,order=config["ref_order"])
        xs.append(np.asarray(theta).copy());values.append(value);gradients.append(g.copy())
        return value,g
    result=minimize(fun,np.asarray(initial).copy(),method="L-BFGS-B",jac=True,
                    bounds=[(-2.,2.)]*len(initial),options=config["ref_optimizer"])
    if not np.isfinite(result.x).all() or not math.isfinite(result.fun) or np.any(abs(result.x)>2.+1e-12):
        raise FloatingPointError("REF did not return a finite feasible endpoint")
    metadata=dict(theta=result.x,loss=float(result.fun),gradient=result.jac,success=bool(result.success),
                  status=int(result.status),message=str(result.message),iterations=int(result.nit),
                  objective_calls=int(result.nfev),seconds=time.perf_counter()-tick,
                  convergence_unconfirmed=not bool(result.success),restarted=False,global_MLE_claimed=False)
    assert result.nfev==len(values)
    return result.x.copy(),dict(theta=np.asarray(xs),loss=np.asarray(values),gradient=np.asarray(gradients)),metadata


def geometry(theta, pmf, teacher_pmf):
    b,W=unpack(theta);length=np.linalg.norm(W,axis=1);gram=W@W.T
    denominator=np.outer(length,length);normalized=np.full_like(gram,np.nan)
    np.divide(gram,denominator,out=normalized,where=denominator>0)
    spin=2*kernel.patterns(len(b)).astype(float)-1
    moments=spin.T@(pmf[:,None]*spin);truth=spin.T@(teacher_pmf[:,None]*spin)
    upper=np.triu_indices(len(b),1);error=moments[upper]-truth[upper]
    return dict(bias=b,row_norms=length,loading_gram=gram,normalized_gram=normalized,
                normalized_gram_undefined_rows=np.flatnonzero(length==0),pair_spin_moments=moments,
                pair_moment_error=error,pair_moment_max_error=float(np.max(abs(error))),
                pair_moment_rmse=float(np.sqrt(np.mean(error**2))),
                coordinate_W_error_used_as_risk=False)


def pmf_variants(theta, config):
    b,W=unpack(theta)
    specs=[(n,0.) for n in config["endpoint_orders"]]+[(config["rotated_order"],config["rotation"])]
    return [(n,a,kernel.model_pmf(b,W,n,a)) for n,a in specs]


def endpoint_reference(candidate, probs, teacher_variants, config=CONFIG):
    main_index=config["endpoint_orders"].index(config["main_reference_order"])
    P=teacher_variants[main_index][2]
    if candidate["name"]=="IND":
        labels=kernel.patterns(config["dimension"]).astype(float);p=np.asarray(candidate["p"])
        logq=labels@np.log(p)+(1-labels)@np.log1p(-p);q=np.exp(logq)
        specs=[(0,0.,q)];grads=[pack(p-probs@labels,np.zeros((config["dimension"],2)))]
        nlls=[float(-probs@logq)];risk=[float(P@(np.log(P)-logq))]
    else:
        specs=pmf_variants(candidate["theta"],config);grads=[];nlls=[];risk=[]
        for n,a,q in specs:
            value,g=objective(candidate["theta"],probs,order=n,angle=a)
            nlls.append(value);grads.append(g);risk.append(float(P@(np.log(P)-np.log(q))))
        q=specs[main_index][2]
    if any(not np.isfinite(x[2]).all() or np.any(x[2]<=0) for x in specs):
        raise FloatingPointError("nonpositive/nonfinite endpoint probability")
    main=0 if candidate["name"]=="IND" else main_index
    masses=[abs(float(x[2].sum())-1) for x in specs]
    pe=max(float(np.max(abs(x[2]-q))) for x in specs)
    re=max(abs(value-risk[main]) for value in risk)
    # Include the separate teacher-reference order variation in the KL qualification.
    te=max(abs(float(tq@(np.log(tq)-np.log(q)))-risk[main]) for _,_,tq in teacher_variants)
    le=max(abs(v-nlls[main]) for v in nlls)
    ge=max(float(np.max(abs(g-grads[main]))) for g in grads)
    qualified=(max(masses)<=config["mass_tolerance"] and pe<=config["pmf_tolerance"] and
               max(re,te)<=config["risk_tolerance"] and le<=config["path_loss_tolerance"] and
               ge<=config["path_gradient_tolerance"])
    record=dict(name=candidate["name"],qualified=qualified,main_order=specs[main][0],
                empirical_NLL=nlls[main],risk_KL=risk[main],gradient=grads[main],pmf=q,
                geometry=geometry(candidate["theta"],q,P),pmf_max_discrepancy=pe,
                risk_max_discrepancy=max(re,te),NLL_max_discrepancy=le,
                gradient_max_discrepancy=ge,mass_max_error=max(masses),
                variants=[dict(order=n,rotation=a,pmf=v,empirical_NLL=nlls[i],risk_KL=risk[i],
                               gradient=grads[i]) for i,(n,a,v) in enumerate(specs)])
    return record


def paired_summary(values):
    values=np.asarray(values,dtype=float)
    return dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),
                MCSE=float(values.std(ddof=1)/math.sqrt(len(values))),banks=len(values),
                scope="conditional on locked endpoints; not independent training repeats")


def cross_evaluate(candidates, probs, teacher_pmf, prior, *, config=CONFIG):
    # Axes: bank, candidate, K (8/64), target (empirical/population).
    losses=np.empty((len(prior),len(candidates),2,2),dtype=np.float64)
    for ci,candidate in enumerate(candidates):
        if candidate["name"]=="IND":
            labels=kernel.patterns(config["dimension"]).astype(float);p=np.asarray(candidate["p"])
            logq=labels@np.log(p)+(1-labels)@np.log1p(-p)
            losses[:,ci,:,:]=np.array([-probs@logq,-teacher_pmf@logq])[None,None,:]
        else:
            b,W=unpack(candidate["theta"])
            for ki,K in enumerate((8,64)):
                for ti,target in enumerate((probs,teacher_pmf)):
                    losses[:,ci,ki,ti]=kernel.mc_bank_losses(b,W,target,prior[:,:K,:],
                        bank_chunk_size=config["evaluation_block"])
    if not np.isfinite(losses).all():raise FloatingPointError("nonfinite cross-evaluation losses")
    rows=[];differences=[];rankings={}
    for ki,K in enumerate((8,64)):
        for ti,target in enumerate(("empirical","population")):
            key=f"K{K}_{target}"
            rankings[key]=[candidates[j]["name"] for j in np.argsort(losses[:,:,ki,ti].mean(axis=0),kind="stable")]
            for ci,candidate in enumerate(candidates):
                rows.append(dict(candidate=candidate["name"],K=K,target=target,**paired_summary(losses[:,ci,ki,ti])))
            for left in range(len(candidates)):
                for right in range(left+1,len(candidates)):
                    differences.append(dict(left=candidates[left]["name"],right=candidates[right]["name"],
                        K=K,target=target,**paired_summary(losses[:,left,ki,ti]-losses[:,right,ki,ti])))
    return losses,dict(means=rows,paired_differences=differences,rankings=rankings,
                      statistical_family_claimed=False,training_streams_in_this_record=1)


def freeze_path(attempt):
    return OUT/("TRAIN_FREEZE.json" if attempt==1 else f"TRAIN_FREEZE_ATTEMPT_{attempt:03}.json")


def freeze(attempt):
    OUT.mkdir(parents=True,exist_ok=True);path=freeze_path(attempt)
    if path.exists():raise FileExistsError(path)
    tests=sorted(OUT.glob("TRAIN_TEST_ATTEMPT_*.json"))
    if not tests:raise RuntimeError("run --self-test before freezing")
    test=read(tests[-1])
    assert test["passed"] and test["source_sha256"]==sha(__file__) and test["kernel_sha256"]==sha(kernel.__file__)
    angular=read(OUT/"ANGULAR_RESULTS.json")
    teacher=next(row for row in angular["rows"] if row["epsilon"]==.8)
    assert teacher["teacher_mass_error"]<=CONFIG["mass_tolerance"]
    assert teacher["teacher_pmf_discrepancy"]<=CONFIG["pmf_tolerance"]
    paths=[Path(__file__),Path(kernel.__file__),DESIGN,tests[-1],OUT/"ANGULAR_RESULTS.json",
           ROOT/"script/joint_dependence_kernel_test.py",OUT/"KERNEL_TEST.log",OUT/"HANDOFF_INPUT.md"]
    snapshot=OUT/f"TRAIN_SOURCE_ATTEMPT_{attempt:03}"
    snapshot.mkdir(exist_ok=False)
    for source in [Path(__file__),Path(kernel.__file__),DESIGN]:
        (snapshot/source.name).write_bytes(source.read_bytes())
    save(path,dict(created_utc=now(),attempt=attempt,config=CONFIG,sources={rel(x):sha(x) for x in paths},
                   source_snapshot=rel(snapshot),teacher_2D_reference_qualified=True,
                   K2_4D_direction_gate_not_used_as_training_gate=True,
                   scientific_run_requires_separate_primary_authorization=True,
                   environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)))
    print(json.dumps(dict(frozen=True,path=rel(path),sha256=sha(path))),flush=True)


def verify_freeze(attempt):
    path=freeze_path(attempt);frozen=read(path)
    assert frozen["attempt"]==attempt and frozen["config"]==CONFIG
    for name,digest in frozen["sources"].items():
        assert sha(ROOT/name)==digest,("frozen input changed; use a new explicit attempt",name)
    return frozen


def run(rep, attempt):
    if rep not in range(CONFIG["reps"]):raise ValueError("rep must be 0,1,2,3")
    frozen=verify_freeze(attempt);tick=time.perf_counter()
    base=OUT/f"rep_{rep:02}"
    folder=base if attempt==1 else base/f"attempt_{attempt:03}"
    folder.mkdir(parents=True,exist_ok=False)
    try:
        rng={role:np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([CONFIG["seed"],rep,role]))) for role in range(5)}
        latent=rng[0].standard_normal((4096,2));uniform=rng[1].random((4096,8))
        teacher_b=np.zeros(8);teacher_W=.8*np.stack((np.cos(PHI),np.sin(PHI)),axis=1)
        labels=(uniform<expit(latent@teacher_W.T+teacher_b)).astype(np.uint8)
        loading=.25*rng[2].standard_normal((8,2));p,raw_initial,initial=initial_parameters(labels,loading)
        probs=frequency(labels);training_prior=rng[3].standard_normal((1200,128,2))
        inputs=folder/"INPUTS.npz"
        save_arrays(inputs,labels=labels,latent=latent,uniform=uniform,initial_loading=loading,
                    raw_initial=raw_initial,initial=initial,training_prior=training_prior,frequency=probs)
        params=dict(rep=rep,attempt=attempt,config=CONFIG,seed=[CONFIG["seed"],rep],
                    roles={"latent":0,"uniform":1,"initialization":2,"training_prior":3,"evaluation_prior":4},
                    frequency=probs,initial=initial,raw_initial=raw_initial,initial_projection_count=int(np.sum(raw_initial!=initial)),
                    inputs=rel(inputs),inputs_sha256=sha(inputs),source_sha256=sha(__file__),
                    kernel_sha256=sha(kernel.__file__),design_sha256=sha(DESIGN),freeze_sha256=sha(freeze_path(attempt)),
                    data_sha256=ah(labels),training_prior_sha256=ah(training_prior),
                    training_interface_inputs=["empirical frequency","initialization","numerical training prior"],
                    teacher_used_by_training_function=False,training_readout="LAST")
        save(folder/"PARAMS.json",params)
        ends,traces,training=train_joint(probs,initial,training_prior,folder=folder)
        save_arrays(folder/"TRAINING.npz",**traces)
        reference,ref_trace,ref_meta=fit_reference(probs,ends[2])
        save_arrays(folder/"REFERENCE_TRAINING.npz",**ref_trace)
        candidates=[]
        for name,theta in zip((*ARMS,"REF"),[*ends,reference]):
            b,W=unpack(theta);candidates.append(dict(name=name,theta=theta,b=b,W=W))
        independent=pack(np.log(p)-np.log1p(-p),np.zeros((8,2)))
        candidates.append(dict(name="IND",theta=independent,b=independent[:8],W=np.zeros((8,2)),p=p))
        lock=folder/"CANDIDATES_LOCK.json"
        save(lock,dict(created_utc=now(),rep=rep,candidates=candidates,frequency=probs,
                       params_sha256=sha(folder/"PARAMS.json"),training_sha256=sha(folder/"TRAINING.npz"),
                       reference_training_sha256=sha(folder/"REFERENCE_TRAINING.npz"),reference_optimizer=ref_meta,
                       locked_before_teacher_risk_and_independent_cross_evaluation=True))
        # No teacher probability or risk was passed to train_joint/fit_reference.
        teacher_variants=pmf_variants(pack(teacher_b,teacher_W),CONFIG)
        main=CONFIG["endpoint_orders"].index(CONFIG["main_reference_order"]);P=teacher_variants[main][2]
        teacher_error=max(float(np.max(abs(q-P))) for _,_,q in teacher_variants)
        teacher_mass=max(abs(float(q.sum())-1) for _,_,q in teacher_variants)
        if teacher_error>CONFIG["pmf_tolerance"] or teacher_mass>CONFIG["mass_tolerance"] or np.any(P<=0):
            raise ArithmeticError("locked-stage teacher PMF qualification failed")
        endpoint_rows=[endpoint_reference(c,probs,teacher_variants) for c in candidates]
        save(folder/"ENDPOINT_QUALIFICATION.json",dict(candidates_lock_sha256=sha(lock),
             teacher_pmf=P,teacher_variants=[dict(order=n,rotation=a,pmf=q) for n,a,q in teacher_variants],
             teacher_pmf_error=teacher_error,teacher_mass_error=teacher_mass,endpoints=endpoint_rows,
             all_qualified=all(x["qualified"] for x in endpoint_rows)))
        if not all(x["qualified"] for x in endpoint_rows):
            raise ArithmeticError("fixed endpoint reference qualification failed; endpoints retained")
        evaluation_prior=rng[4].standard_normal((4096,64,2))
        # Persist the independent evaluation input before evaluating it.
        save_arrays(folder/"EVALUATION_PRIOR.npz",prior=evaluation_prior)
        bank_losses,cross=cross_evaluate(candidates,probs,P,evaluation_prior)
        save_arrays(folder/"EVALUATION.npz",loss=bank_losses,prior=evaluation_prior,frequency=probs,
                    teacher_pmf=P,candidate_names=np.asarray(NAMES),particles=np.array([8,64]),
                    target_names=np.array(["empirical","population"]))
        path_calls=len(training["path_checks"])
        result=dict(status="completed",rep=rep,attempt=attempt,params_sha256=sha(folder/"PARAMS.json"),
             candidates_lock_sha256=sha(lock),freeze_sha256=sha(freeze_path(attempt)),candidates=candidates,
             frequency=probs,training=training,reference_optimizer=ref_meta,endpoints=endpoint_rows,
             cross_evaluation=cross,seconds=time.perf_counter()-tick,
             resources=dict(new_observations=4096,data_Gaussian_scalars=8192,data_uniform_scalars=32768,
                initialization_Gaussian_scalars=16,training_prior_vectors=1200*128,
                training_Gaussian_scalars=1200*128*2,independent_evaluation_prior_vectors=4096*64,
                evaluation_Gaussian_scalars=4096*64*2,Adam_updates=3*1200,
                Adam_objective_gradient_calls={name:1200 for name in ARMS},
                GH_Adam_nodes=1200*20**2,GH_path_nodes=path_calls*(20**2+40**2),
                REF_objective_gradient_calls=ref_meta["objective_calls"],REF_nodes=ref_meta["objective_calls"]*40**2,
                terminal_GH_PMF_nodes=5*(40**2+60**2+80**2+72**2),
                terminal_GH_loss_gradient_nodes=4*(40**2+60**2+80**2+72**2),
                MC_evaluation_bank_candidate_K_target_cells=4096*5*2*2,
                MC_computed_nonindependent_cells=4096*4*2*2,
                MC_evaluation_particle_pattern_products=4096*4*(8+64)*(int(np.count_nonzero(probs))+256),
                training_MC_particle_pattern_products=2*1200*128*int(np.count_nonzero(probs)),
                training_GH_node_pattern_products=1200*20**2*int(np.count_nonzero(probs)),
                active_empirical_patterns=int(np.count_nonzero(probs)),
                observed_patterns=256,independent_training_streams=1,reference_extra_cost_separate=True),
             artifacts={path.name:dict(path=rel(path),sha256=sha(path),bytes=path.stat().st_size)
                        for path in folder.iterdir() if path.is_file()},
             claims=dict(exploratory_training_panel=True,confirmatory_training_t_family=False,
                MCSE_conditions_on_fixed_endpoints=True,extra_risk_alone_proves_angle_mechanism=False,
                reference_is_global_MLE=False,quadrature_agreement_is_rigorous_error_bound=False))
        verify_freeze(attempt)
        save(folder/"RESULTS.json",result)
        print(json.dumps(dict(completed=True,rep=rep,seconds=result["seconds"],
             risks={x["name"]:x["risk_KL"] for x in endpoint_rows},folder=rel(folder))),flush=True)
    except BaseException:
        save(folder/"FAILURE.json",dict(rep=rep,attempt=attempt,seconds=time.perf_counter()-tick,
             source_sha256=sha(__file__),freeze_sha256=sha(freeze_path(attempt)),
             traceback=traceback.format_exc(),automatic_retry=False))
        raise


def self_test():
    started=time.perf_counter();config=copy.deepcopy(CONFIG)
    config.update(steps=4,switch_step=2,checkpoint_every=2,gh_order=6,path_reference_order=8,
                  ref_order=8,ref_optimizer=dict(maxiter=2,maxfun=5,maxls=5,gtol=1e-7,ftol=1e-12),
                  endpoint_orders=[6,8,10],main_reference_order=8,rotated_order=10,
                  evaluation_banks=4,evaluation_block=2)
    labels=kernel.patterns(8);probs=np.arange(1.,257.);probs/=probs.sum()
    initial=pack(np.linspace(-.08,.08,8),.03*np.cos(np.arange(16)).reshape(8,2))
    prior=np.sin(np.arange(4*128*2).reshape(4,128,2)*.113)
    before=(probs.copy(),initial.copy(),prior.copy())
    ends,tape,meta=train_joint(probs,initial,prior,config=config)
    assert tape["theta"].shape==(5,3,24) and np.isfinite(tape["theta"]).all()
    assert all(np.array_equal(a,b) for a,b in zip(before,(probs,initial,prior)))
    assert [x["step"] for x in meta["path_checks"]]==[0,2,4]
    for arm,K in enumerate((8,64)):
        loss,g=objective(initial,probs,banks=prior[0].reshape(128//K,K,2))
        assert loss==tape["loss_before_update"][0,arm]
        assert np.array_equal(g,tape["gradient_before_update"][0,arm])
    for step in range(1,5):
        for arm in range(3):
            m=np.zeros(24);v=np.zeros(24);x=initial.copy()
            for j in range(1,step+1):
                lr=config["learning_rates"][0 if j<=2 else 1]
                x,m,v,_=adam_step(x,tape["gradient_before_update"][j-1,arm],m,v,j,lr,config)
            assert np.array_equal(x,tape["theta"][step,arm])
    x,m,v,proposal=adam_step(np.array([1.999,-1.999]),np.array([-1.,1.]),np.zeros(2),np.zeros(2),1,.02,config)
    assert np.array_equal(x,np.array([2.,-2.])) and np.any(x!=proposal)
    ref,ref_tape,ref_meta=fit_reference(probs,ends[2],config=config)
    assert np.isfinite(ref).all() and len(ref_tape["loss"])==ref_meta["objective_calls"]
    p,_,indinit=initial_parameters(labels,.03*np.ones((8,2)))
    candidates=[dict(name=name,theta=theta) for name,theta in zip((*ARMS,"REF"),[*ends,ref])]
    candidates.append(dict(name="IND",theta=pack(np.log(p)-np.log1p(-p),np.zeros((8,2))),p=p))
    evalprior=np.cos(np.arange(4*64*2).reshape(4,64,2)*.091)
    losses,cross=cross_evaluate(candidates,probs,np.ones(256)/256,evalprior,config=config)
    assert losses.shape==(4,5,2,2) and len(cross["means"])==20 and len(cross["paired_differences"])==40
    assert np.array_equal(losses[:,4,0],losses[:,4,1])
    # Fixed manufactured-model references exercise endpoint PMF, risk and gradient qualification.
    teacher_variants=pmf_variants(initial,config)
    terminal_rows=[endpoint_reference(candidate,probs,teacher_variants,config) for candidate in candidates]
    assert all(row["qualified"] for row in terminal_rows)
    assert np.array_equal(terminal_rows[-1]["gradient"],pack(p-probs@labels,np.zeros((8,2))))
    # Geometry records undefined IND row angles honestly as null, never fabricated cosines.
    geom=geometry(candidates[-1]["theta"],np.ones(256)/256,np.ones(256)/256)
    assert len(geom["normalized_gram_undefined_rows"])==8
    json.dumps(native(geom),allow_nan=False)
    qualification=dict(passed=True,source_sha256=sha(__file__),kernel_sha256=sha(kernel.__file__),
         manufactured_only=True,study_teacher_used=False,new_scientific_random_draws=0,
         projected_Adam_replay=True,shared_128_point_K8_K64_grouping=True,
         learner_inputs_unchanged=True,finite_REF_unconverged_endpoint_retained=True,
         fixed_endpoint_MC_axes_and_pairs=True,endpoint_reference_qualification_exercised=True,
         independent_baseline_analytic_gradient_checked=True,undefined_Gram_rows_not_fabricated=True,
         fixture_steps=4,path_checks=meta["path_checks"],seconds=time.perf_counter()-started)
    OUT.mkdir(parents=True,exist_ok=True)
    attempt=len(list(OUT.glob("TRAIN_TEST_ATTEMPT_*.json")))+1
    save(OUT/f"TRAIN_TEST_ATTEMPT_{attempt:03}.json",qualification)
    print(json.dumps(dict(passed=True,fixture_steps=4,seconds=qualification["seconds"],
                          source_sha256=qualification["source_sha256"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test",action="store_true")
    group.add_argument("--freeze",action="store_true")
    group.add_argument("--run",action="store_true")
    parser.add_argument("--rep",type=int)
    parser.add_argument("--attempt",type=int,default=1)
    args=parser.parse_args()
    if args.attempt<1:parser.error("--attempt must be positive")
    try:
        if args.self_test:self_test()
        elif args.freeze:freeze(args.attempt)
        else:
            if args.rep is None:parser.error("--run requires --rep")
            run(args.rep,args.attempt)
    except BaseException:
        if args.self_test:
            OUT.mkdir(parents=True,exist_ok=True)
            save(OUT/f"TRAIN_TEST_FAILURE_{time.time_ns()}.json",dict(source_sha256=sha(__file__),
                 kernel_sha256=sha(kernel.__file__),traceback=traceback.format_exc(),scientific_run=False))
        raise

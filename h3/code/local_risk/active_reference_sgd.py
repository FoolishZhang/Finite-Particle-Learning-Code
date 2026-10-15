"""Replay-only collapsed arm of the original teacher endpoint SGD program.

Inputs contain actual observations and caller-frozen indices/prior banks. There
is no teacher parameter, latent data or RNG interface in the learning function.
"""
from __future__ import annotations

import numpy as np
import signal_proposal_core as legacy
import teacher_endpoint_core as core
import teacher_endpoint_training as original

SCHEMA = "active_reference_original_collapsed_sgd_v1"


def replay(counts, observed_labels, initial_coordinates, indices, z, *, config,
           before_score=None):
    counts, y = np.asarray(counts), np.asarray(observed_labels)
    c = np.asarray(initial_coordinates, float).copy()
    indices, z = np.asarray(indices), np.asarray(z, float)
    required = {"M", "epsilon", "tau", "inner_order", "K", "steps", "banks",
                "learning_rate0", "decay_scale", "spectral_floor", "output_average_last"}
    if set(config) != required:
        raise ValueError("Explicit original collapsed-arm configuration required")
    S, B, K = config["steps"], config["banks"], config["K"]
    if indices.shape != (S, B) or z.shape != (S, B, K, 3) or not np.isfinite(z).all():
        raise ValueError("frozen replay arrays differ from the declared configuration")
    if c.ndim != 2 or c.shape[1] != 3 or not np.isfinite(c).all():
        raise ValueError("fixed initial coordinates must be (starts,3)")
    for v in c:
        if np.linalg.eigvalsh(legacy.correlation_matrix(v))[0] < config["spectral_floor"]-1e-12:
            raise ValueError("infeasible initialization")
    L, I = config["output_average_last"], len(c)
    if not 1 <= L <= S:
        raise ValueError("invalid last-state averaging length")
    core.observed_training_batch(counts,y,np.arange(len(y)),M=config["M"])
    trace = {"coordinates": np.empty((S+1,I,3)), "gradient": np.empty((S,I,3)),
             "proposed_coordinates": np.empty((S,I,3)), "mean_bank_loss": np.empty((S,I)),
             "learning_rate": np.empty(S), "projection_lambda_min": np.empty((S,I)),
             "projection_iterations": np.empty((S,I),dtype=np.int64),
             "projection_residual": np.empty((S,I))}
    trace["coordinates"][0] = c
    for step in range(S):
        batch = core.observed_training_batch(counts,y,indices[step],M=config["M"])
        rate = original.learning_rate(config,step)
        trace["learning_rate"][step] = rate
        for ii in range(I):
            if before_score is not None:
                before_score(B)
            scores = core.collapsed_banks(z[step],batch["counts"],legacy.correlation_matrix(c[ii]),
                M=config["M"],epsilon=config["epsilon"],tau=config["tau"],inner_order=config["inner_order"],
                observed_labels=batch["observed_labels"])
            gradient = scores["price"].mean(axis=0)
            updated, projection = original.sgd_update(c[ii],gradient,rate=rate,spectral_floor=config["spectral_floor"])
            trace["gradient"][step,ii] = gradient
            trace["mean_bank_loss"][step,ii] = scores["loss"].mean()
            trace["proposed_coordinates"][step,ii] = projection["proposed_coordinates"]
            trace["projection_lambda_min"][step,ii] = projection["lambda_min"]
            trace["projection_iterations"][step,ii] = projection["iterations"]
            trace["projection_residual"][step,ii] = projection["residual"]
            c[ii] = updated
        trace["coordinates"][step+1] = c
    return {"schema":SCHEMA,"coordinates":trace["coordinates"][-L:].mean(axis=0),"trace":trace,
            "optimizer_updates":S*I,"bank_calls":S*B*I,
            "output_rule":"mean_last_L_post_update_coordinates_per_start; not_best_start_selection",
            "trace_loss_convention":"legacy_ordered_U_loss_omits_count_multiplicity; postfit_joint_count_objectives_include_it; gradients_unchanged",
            "teacher_information_used_for_training":False,"new_RNG_inside_replay":0}


def selftest():
    import tempfile
    from pathlib import Path
    S,B,K,M=4,4,2,2
    counts=np.array([[0,1,2],[2,1,0],[1,0,1],[1,2,1]])
    labels=np.array([[-1,1,1],[1,-1,-1],[-1,-1,1],[1,1,-1]])
    initials=np.array([[0.,0.,0.],[.2,-.15,.1]])
    indices=np.tile(np.arange(B),(S,1))
    z=np.sin(np.arange(S*B*K*3).reshape(S,B,K,3)+.1)
    cfg=dict(M=M,epsilon=[.3]*3,tau=3.,inner_order=32,K=K,steps=S,banks=B,
             learning_rate0=1.,decay_scale=64.,spectral_floor=.15,output_average_last=2)
    result=replay(counts,labels,initials,indices,z,config=cfg)
    class Active:
        def __init__(self):self.step=0
        def standard_normal(self,size):
            value=z[self.step].copy();self.step+=1;assert value.shape==size;return value
    class Index:
        def integers(self,*args,**kwargs):return np.arange(B)
    class Private:
        def standard_normal(self,size):return np.zeros(size)
        def random(self,size):return np.full(size,.25)
    with tempfile.TemporaryDirectory() as tmp:
        oldcfg={**cfg,"save_block_steps":2,"proposal_batch_size":128,"max_private_proposals_per_step":None}
        original.run_context(counts,labels,initials,config=oldcfg,rng_indices=Index(),rng_active=Active(),
            rng_private=Private(),output_dir=Path(tmp)/"legacy",context_id="manufactured",rng_provenance={})
        with np.load(Path(tmp)/"legacy/TRACE.npz") as f:
            assert np.array_equal(result["trace"]["coordinates"],f["coordinates"][:,:,1])
            assert np.array_equal(result["trace"]["gradient"],f["gradient"][:,:,1])
    print("collapsed replay matches original full-context collapsed trace bitwise; manufactured only")


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--self-test",action="store_true",required=True)
    p.parse_args();selftest()

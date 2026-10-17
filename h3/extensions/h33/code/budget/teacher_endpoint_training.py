"""Explicit-config observed-data training for paired teacher endpoints.

A context is one finite-N observed dataset: fixed initializations and both
endpoints share each step's indices, prior z and full-endpoint private T. This
module generates no teacher, evaluates no true risk, and chooses no output by
teacher information. Scientific runs require the owner's frozen contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import signal_proposal_core as legacy
import teacher_endpoint_core as core

SCHEMA = "teacher_endpoint_training_v1"
CONFIG_FIELDS = {"M", "epsilon", "tau", "inner_order", "K", "steps", "banks",
                 "learning_rate0", "decay_scale", "spectral_floor", "output_average_last",
                 "save_block_steps", "proposal_batch_size", "max_private_proposals_per_step"}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_config(config):
    config = json.loads(json.dumps(config, allow_nan=False))
    if set(config) != CONFIG_FIELDS:
        raise ValueError(f"Config fields: missing={sorted(CONFIG_FIELDS-set(config))}; unknown={sorted(set(config)-CONFIG_FIELDS)}")
    for name in ("M", "inner_order", "K", "steps", "banks", "output_average_last", "save_block_steps", "proposal_batch_size"):
        config[name] = _integer(config[name], name)
    if config["inner_order"] < 2 or config["output_average_last"] > config["steps"]:
        raise ValueError("inner_order >=2 and output_average_last <=steps required")
    if config["max_private_proposals_per_step"] is not None:
        config["max_private_proposals_per_step"] = _integer(config["max_private_proposals_per_step"], "max_private_proposals_per_step")
        if config["max_private_proposals_per_step"] < config["banks"]*config["K"]*3*config["M"]:
            raise ValueError("Private proposal budget must cover requested draws")
    epsilon = np.asarray(config["epsilon"], dtype=float)
    if epsilon.shape != (3,) or not np.isfinite(epsilon).all():
        raise ValueError("epsilon must be three finite group couplings")
    for name in ("tau", "learning_rate0", "decay_scale", "spectral_floor"):
        v = config[name]
        if isinstance(v, bool) or not np.isscalar(v) or not np.isfinite(v):
            raise ValueError(f"{name} must be a finite scalar")
        config[name] = float(v)
    if config["tau"] < 0 or config["learning_rate0"] <= 0 or config["decay_scale"] <= 0 or not 0 < config["spectral_floor"] < 1:
        raise ValueError("Require tau>=0, learning_rate0/decay_scale>0, spectral_floor in (0,1)")
    return config


def learning_rate(config, step_index):
    """Zero-based update t=0..T-1: lr0/sqrt(1+t/decay_scale)."""
    step_index = _integer(step_index, "step_index", 0)
    return config["learning_rate0"] / np.sqrt(1+step_index/config["decay_scale"])


def sgd_update(coordinates, gradient, *, rate, spectral_floor):
    """Physical-coordinate SGD followed by the existing Dykstra projection."""
    c, g = np.asarray(coordinates, dtype=float), np.asarray(gradient, dtype=float)
    if c.shape != (3,) or g.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(g).all():
        raise ValueError("Finite three-coordinate state and gradient required")
    if not np.isscalar(rate) or not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    proposed = c-float(rate)*g
    projection = core.project_coordinates(proposed, delta=spectral_floor)
    return projection["coordinates"], {"proposed_coordinates": proposed, **projection}


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, default=_serial, allow_nan=False)
        stream.write("\n")


def _npz(path, arrays):
    with path.open("xb") as stream:
        np.savez(stream, **arrays)


def _file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return {"path": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def run_context(counts, observed_labels, initial_coordinates, *, config,
                rng_indices, rng_active, rng_private, output_dir, context_id,
                rng_provenance=None, progress=None):
    """Train paired endpoints from observed counts/Y and save complete replay inputs.

    Caller supplies all RNGs and frozen configuration. initial_coordinates[I,3]
    are fixed before training, not chosen using true teacher evaluation. Return
    outputs follow a fixed average of the last L post-update coordinates.
    output_dir must not exist; a failed attempt is preserved and never resumed
    or overwritten here. No RNG draw occurs during import or validation.
    """
    config = validate_config(config)
    y = np.asarray(observed_labels)
    core.observed_label_probabilities(y)
    counts = np.asarray(counts)
    # Validate the full dataset through the observation-only interface.
    core.observed_training_batch(counts, y, np.arange(len(y)), M=config["M"])
    initial = np.asarray(initial_coordinates, dtype=float)
    if initial.ndim != 2 or len(initial) < 1 or initial.shape[1] != 3 or not np.isfinite(initial).all():
        raise ValueError("initial_coordinates must be finite nonempty (I,3)")
    for c in initial:
        C = legacy.correlation_matrix(c)
        if np.linalg.eigvalsh(C)[0] < config["spectral_floor"]:
            raise ValueError("Initial coordinates must already satisfy the frozen spectral floor")
    for rng, method, role in ((rng_indices, "integers", "indices"), (rng_active, "standard_normal", "active"), (rng_private, "random", "private")):
        if rng is None or not hasattr(rng, method):
            raise ValueError(f"Explicit caller-owned {role} RNG required")
    if not hasattr(rng_private, "standard_normal"):
        raise ValueError("Private RNG must provide standard_normal")
    if not isinstance(context_id, str) or not context_id:
        raise ValueError("context_id must be a nonempty string")
    # Freeze all serializable metadata before creating an attempt or drawing.
    json.dumps(rng_provenance, allow_nan=False)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    input_record = {"schema": SCHEMA, "context_id": context_id, "config": config,
                    "initial_coordinates": initial, "endpoints": core.ENDPOINTS,
                    "rng_provenance": rng_provenance, "rng_ownership": "caller",
                    "teacher_latent_is_learning_visible": False,
                    "learning_rate_indexing": "zero_based_t_0_to_steps_minus_1",
                    "output_rule": "arithmetic_mean_last_L_post_update_coordinates",
                    "context_unit": "one_fixed_observed_dataset_shared_across_all_initializations_and_endpoints"}
    _json(destination/"CONTEXT.json", input_record)
    _npz(destination/"DATA.npz", {"counts": counts, "observed_labels": y})
    S, B, K, M, I = config["steps"], config["banks"], config["K"], config["M"], len(initial)
    coordinates = np.broadcast_to(initial[:, None, :], (I, 2, 3)).copy()
    trace = {"coordinates": np.empty((S+1, I, 2, 3)),
             "gradient": np.empty((S, I, 2, 3)), "proposed_coordinates": np.empty((S, I, 2, 3)),
             "mean_bank_loss": np.empty((S, I, 2)), "learning_rate": np.empty(S),
             "projection_lambda_min": np.empty((S, I, 2)),
             "projection_iterations": np.empty((S, I, 2), dtype=np.int64),
             "projection_residual": np.empty((S, I, 2)),
             "projection_roundoff_retraction": np.empty((S, I, 2))}
    trace["coordinates"][0] = coordinates
    ledger = {"indices_drawn": 0, "active_Gaussian_scalars": 0, "private_proposed": 0,
              "private_accepted": 0, "private_used": 0, "private_discarded_accepted": 0,
              "private_uniforms_generated": 0, "core_bank_calls": 0,
              "optimizer_updates": 0, "completed_steps": 0}
    block_records, block_indices, block_z, block_T = [], [], [], []
    block_start = 0
    active_step = active_init = active_endpoint = None

    def flush_block():
        nonlocal block_start
        if not block_indices:
            return
        path = destination/f"BANKS_{block_start:04d}_{block_start+len(block_indices)-1:04d}.npz"
        _npz(path, {"indices": np.stack(block_indices), "z": np.stack(block_z), "T": np.stack(block_T),
                    "step_indices": np.arange(block_start, block_start+len(block_indices))})
        block_records.append(_file_record(path))
        block_start += len(block_indices)
        block_indices.clear(); block_z.clear(); block_T.clear()

    try:
        for step in range(S):
            active_step = step
            active_init = active_endpoint = None
            indices = rng_indices.integers(len(y), size=B)
            z = rng_active.standard_normal((B, K, 3))
            ledger["indices_drawn"] += B
            ledger["active_Gaussian_scalars"] += B*K*3
            try:
                T, private = core.sample_oriented_qbeta(rng_private, (B, K, 3, M), tau=config["tau"], beta=1.,
                        proposal_batch_size=config["proposal_batch_size"], max_proposals=config["max_private_proposals_per_step"])
            except Exception as sampling_error:
                _npz(destination/"FAILED_SAMPLING_INPUTS.npz", {"indices": indices, "z": z,
                        "step_index": np.array(step)})
                # The existing exact sampler reports its exhausted-budget
                # counters in this specific exception. Preserve those costs
                # without pretending to have its unavailable partial T array.
                counters = re.fullmatch(r"Rejection budget exhausted: proposed=(\d+), accepted=(\d+), used=(\d+)", str(sampling_error))
                if counters is not None:
                    proposed, accepted, used = map(int, counters.groups())
                    ledger["private_proposed"] += proposed
                    ledger["private_accepted"] += accepted
                    ledger["private_used"] += used
                    ledger["private_discarded_accepted"] += accepted-used
                    ledger["private_uniforms_generated"] += proposed
                raise
            block_indices.append(indices.copy()); block_z.append(z); block_T.append(T)
            for field in ("proposed", "accepted", "used", "discarded_accepted", "uniforms_generated"):
                ledger["private_"+field] += private[field]
            batch = core.observed_training_batch(counts, y, indices, M=M)
            rate = float(learning_rate(config, step)); trace["learning_rate"][step] = rate
            for init in range(I):
                for arm, endpoint in enumerate(core.ENDPOINTS):
                    active_init, active_endpoint = init, endpoint
                    scores = core.endpoint_banks(endpoint, z, batch["counts"], legacy.correlation_matrix(coordinates[init, arm]),
                            T=T, M=M, epsilon=config["epsilon"], tau=config["tau"], inner_order=config["inner_order"],
                            observed_labels=batch["observed_labels"])
                    ledger["core_bank_calls"] += B
                    gradient = scores["price"].mean(axis=0)
                    new, projection = sgd_update(coordinates[init, arm], gradient, rate=rate, spectral_floor=config["spectral_floor"])
                    trace["gradient"][step, init, arm] = gradient
                    trace["mean_bank_loss"][step, init, arm] = scores["loss"].mean()
                    trace["proposed_coordinates"][step, init, arm] = projection["proposed_coordinates"]
                    for field in ("lambda_min", "iterations", "residual", "roundoff_retraction"):
                        trace["projection_"+field][step, init, arm] = projection[field]
                    coordinates[init, arm] = new
                    ledger["optimizer_updates"] += 1
            trace["coordinates"][step+1] = coordinates
            ledger["completed_steps"] = step+1
            if len(block_indices) >= config["save_block_steps"] or step+1 == S:
                flush_block()
                if progress is not None:
                    progress({"context_id": context_id, "completed_steps": step+1, "total_steps": S,
                              "seconds": time.perf_counter()-start, "ledger": ledger.copy()})
        outputs = []
        L = config["output_average_last"]
        averaged = trace["coordinates"][-L:].mean(axis=0)
        for init in range(I):
            for arm, endpoint in enumerate(core.ENDPOINTS):
                C = legacy.correlation_matrix(averaged[init, arm])
                lam = float(np.linalg.eigvalsh(C)[0])
                if lam < config["spectral_floor"]-2e-12:
                    raise FloatingPointError("Average of projected matrices violates spectral floor")
                outputs.append({"initialization_index": init, "endpoint": endpoint,
                                "coordinates": averaged[init, arm], "C": C, "lambda_min": lam,
                                "post_update_steps_averaged": [S-L+1, S], "average_count": L})
        _npz(destination/"TRACE.npz", trace)
        files = [_file_record(destination/name) for name in ("CONTEXT.json", "DATA.npz", "TRACE.npz")]+block_records
        summary = {"schema": SCHEMA, "context_id": context_id, "status": "complete", "N": len(y),
                   "outputs": outputs, "ledger": ledger, "bank_blocks": block_records, "files": files,
                   "seconds": time.perf_counter()-start,
                   "trace_axes": {"coordinates": "step_including_initial,initialization,endpoint,physical_pair",
                                  "gradient": "zero_based_update,initialization,endpoint,physical_pair",
                                  "endpoints": core.ENDPOINTS, "pair_order": legacy.pair_order(3)},
                   "teacher_information_used_for_output_or_hyperparameters": False,
                   "qualification": "observed_data_finite_target_Price_SGD; numerical_Dykstra_projection; no_true_risk_evaluation"}
        _json(destination/"SUMMARY.json", summary)
        return summary
    except Exception as exc:
        flush_block()
        completed = ledger["completed_steps"]
        partial = {key: value[:completed+1] if key == "coordinates" else value[:completed] for key, value in trace.items()}
        _npz(destination/"PARTIAL_TRACE.npz", partial)
        _npz(destination/"FAILED_CURRENT_STATE.npz", {"coordinates": coordinates})
        _json(destination/"FAILED.json", {"schema": SCHEMA, "context_id": context_id,
                "status": "failed_preserved", "exception_type": type(exc).__name__, "message": str(exc),
                "ledger": ledger, "bank_blocks": block_records, "seconds": time.perf_counter()-start,
                "active_zero_based_step": active_step, "active_initialization": active_init,
                "active_endpoint": active_endpoint})
        raise

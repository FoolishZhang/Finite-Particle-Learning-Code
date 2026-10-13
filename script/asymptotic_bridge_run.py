"""Frozen, nonadaptive bridge bank scheduler; analysis lives in a separate module.

Production run requires an existing source-bound FREEZE.json. --check only
reads existing outputs. --dummy exercises scheduling in a temporary directory
with a distinct manufactured seed and never writes production raw banks.
"""
from __future__ import annotations
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
import asymptotic_bridge_core as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/asymptotic_bridge_v1_20261008"
REFERENCE = OUT / "REFERENCE.json"
ARRAY_FIELDS = ("z", "s_prefix", "u", "scores", "clipmask", "metadata")


def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text("utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")


def verify_freeze(out: Path = OUT) -> tuple[dict,str,dict,str]:
    freeze_path=out/"FREEZE.json"
    frozen=read(freeze_path)
    sources=frozen.get("sources")
    if not isinstance(sources,dict) or not sources:
        raise ValueError("FREEZE.json must contain nonempty sources path-to-SHA256 mapping")
    for name,digest in sources.items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("freeze source escapes repository")
        if sha(path)!=digest:
            raise ValueError(f"changed frozen source: {name}")
    required=(Path(__file__).resolve(),ROOT/"script/asymptotic_bridge_core.py",out/"REFERENCE.json")
    for path in required:
        if path.relative_to(ROOT).as_posix() not in sources:
            raise ValueError(f"required source missing from freeze: {path.name}")
    config=frozen.get("config")
    if not isinstance(config,dict):
        raise ValueError("FREEZE.json must contain config=core.production_config()")
    for name,value in core.production_config().items():
        if config.get(name)!=value:
            raise ValueError(f"frozen config mismatch: {name}")
    reference=read(out/"REFERENCE.json")
    m=float(reference["m_H45"])
    if not np.isfinite(m):
        raise ValueError("nonfinite reference m_H45")
    return frozen,sha(freeze_path),reference,sha(out/"REFERENCE.json")


def expected_shapes(banks: int, particles: int) -> dict:
    return dict(z=(banks,particles,3),s_prefix=(banks,4,particles),u=(banks,),
                scores=(banks,4,5),clipmask=(banks,))


def validate_saved_block(path: Path, metadata_expected: dict, *, banks: int, particles: int) -> dict:
    """Only check saved shapes/finite values/metadata, never RNG or kernel recomputation."""
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!=set(ARRAY_FIELDS):
            raise ValueError(f"unexpected raw fields: {path.name}")
        for name,shape in expected_shapes(banks,particles).items():
            values=saved[name]
            if values.shape!=shape:
                raise ValueError(f"shape error {path.name}:{name}")
            if values.dtype != (np.dtype(bool) if name=="clipmask" else np.dtype("float64")):
                raise ValueError(f"dtype error {path.name}:{name}")
            if name!="clipmask" and not np.isfinite(values).all():
                raise ValueError(f"nonfinite {path.name}:{name}")
        meta=json.loads(str(saved["metadata"].item()))
        for name,value in metadata_expected.items():
            if meta.get(name)!=value:
                raise ValueError(f"metadata mismatch {path.name}:{name}")
    return meta


def block_metadata(block_id: int, freeze_sha: str, reference_sha: str, *, banks: int=core.BLOCK,
                   particles: int=core.K, seed_family: int=core.SEED_FAMILY, scientific: bool=True) -> dict:
    return dict(schema_version=core.SCHEMA_VERSION,block_id=block_id,bank_first=block_id*banks,
                banks=banks,particles=particles,n_values=list(core.N_VALUES),metric_names=core.METRIC_NAMES,
                label_encoding=core.LABEL_ENCODING,freeze_sha256=freeze_sha,reference_sha256=reference_sha,
                freeze_reference="FREEZE.json",coefficient_reference="REFERENCE.json",
                scientific=scientific,role_seed_sequences={str(role):[seed_family,block_id,role] for role in (0,1,2)},
                nuisance_method="inplace_log_expit_segment_sum_then_add",
                gamma_method="dominant_complement_from_other_weights")


def worker_block(job: dict) -> dict:
    """One worker owns one deterministic block path; it never overwrites an output."""
    out=Path(job["out"]);block_id=job["block_id"]
    path=out/"raw"/f"block_{block_id:05d}.npz"
    receipt_path=path.with_suffix(".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(f"existing block cannot be rerun: {block_id}")
    start=time.monotonic()
    try:
        z,s,u=core.generate_block(block_id,seed_family=job["seed_family"],banks=job["banks"],particles=job["particles"])
        generated=time.monotonic()
        scores,clipmask=core.score_block(z,s,u,job["m_H"])
        scored=time.monotonic()
        metadata=block_metadata(block_id,job["freeze_sha"],job["reference_sha"],banks=job["banks"],
                                particles=job["particles"],seed_family=job["seed_family"],scientific=job["scientific"])
        metadata["attempt"]=job["attempt"]
        metadata["generated_utc"]=now()
        temporary=path.with_suffix(".npz.partial")
        with temporary.open("xb") as handle:
            np.savez(handle,z=z,s_prefix=s,u=u,scores=scores,clipmask=clipmask,
                     metadata=np.asarray(json.dumps(metadata,ensure_ascii=False,sort_keys=True)))
            handle.flush()
            os.fsync(handle.fileno())
        # Windows rename refuses an existing target; no completed output is replaced.
        temporary.rename(path)
        receipt=dict(path=path.relative_to(out).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
                     block_id=block_id,banks=job["banks"],metadata=metadata,
                     elapsed_seconds=time.monotonic()-start,generation_seconds=generated-start,
                     scoring_seconds=scored-generated)
        write_new(receipt_path,receipt)
        return receipt
    except BaseException as error:
        error_path=out/"raw"/f"ERROR_block_{block_id:05d}_attempt_{job['attempt']:03d}.json"
        if not error_path.exists():
            write_new(error_path,dict(block_id=block_id,attempt=job["attempt"],utc=now(),
                                     error=repr(error),traceback=traceback.format_exc()))
        raise


def existing_receipts(out: Path, expected: dict, *, block_count: int,banks: int,particles: int,
                      seed_family: int,scientific: bool) -> dict[int,dict]:
    records={}
    if list((out/"raw").glob("*.partial")):
        raise ValueError("unfinished raw files remain; use explicit resume to retain and recover them")
    for path in sorted((out/"raw").glob("block_*.npz")):
        block_id=int(path.stem.split("_")[1])
        if not 0<=block_id<block_count or block_id in records:
            raise ValueError("unexpected block id")
        meta=block_metadata(block_id,expected["freeze_sha"],expected["reference_sha"],banks=banks,
                            particles=particles,seed_family=seed_family,scientific=scientific)
        validate_saved_block(path,meta,banks=banks,particles=particles)
        receipt=read(path.with_suffix(".json"))
        if receipt["sha256"]!=sha(path) or receipt["bytes"]!=path.stat().st_size:
            raise ValueError(f"raw receipt mismatch: {path.name}")
        if receipt["block_id"]!=block_id or receipt["metadata"].get("freeze_sha256")!=expected["freeze_sha"]:
            raise ValueError("receipt metadata mismatch")
        records[block_id]=receipt
    return records


def schedule(out: Path, *, m_H: float, freeze_sha: str, reference_sha: str, attempt: int,
             block_count: int=core.BLOCK_COUNT,banks: int=core.BLOCK,particles: int=core.K,
             workers: int=core.WORKERS,seed_family: int=core.SEED_FAMILY,scientific: bool=True,
             resume: bool=False) -> dict:
    """Bounded panel scheduler. No CI, selection, or additional bank allocation."""
    (out/"raw").mkdir(parents=True,exist_ok=True)
    expected=dict(freeze_sha=freeze_sha,reference_sha=reference_sha)
    # Only explicit resume quarantines unfinished files; no completed result is
    # changed, and every replacement is the same prescribed block, not extra B.
    if resume:
        partials=list((out/"raw").glob("block_*.npz.partial"))
        if (out/"BANK_SCORES.npy.partial").exists():
            partials.append(out/"BANK_SCORES.npy.partial")
        for path in partials:
            archive=out/"failed"/f"attempt_{attempt:03d}"
            archive.mkdir(parents=True,exist_ok=True)
            path.rename(archive/path.name)
        if partials:
            write_new(out/f"RESUME_PARTIALS_{attempt:03d}.json",
                      dict(utc=now(),attempt=attempt,retained_paths=[str((out/"failed"/f"attempt_{attempt:03d}"/p.name).relative_to(out)) for p in partials],
                           scope="unfinished files only; same frozen bank indices regenerated"))
    records=existing_receipts(out,expected,block_count=block_count,banks=banks,particles=particles,
                             seed_family=seed_family,scientific=scientific) if resume else {}
    if not resume and any((out/"raw").iterdir()):
        raise FileExistsError("fresh run requires empty raw directory")
    pending=[index for index in range(block_count) if index not in records]
    start=time.monotonic();last_progress=start
    if pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures={pool.submit(worker_block,dict(out=str(out),block_id=index,m_H=m_H,freeze_sha=freeze_sha,
                         reference_sha=reference_sha,attempt=attempt,banks=banks,particles=particles,
                         seed_family=seed_family,scientific=scientific)):index for index in pending}
            for future in as_completed(futures):
                record=future.result()
                records[record["block_id"]]=record
                current=time.monotonic()
                if current-last_progress>=30 or len(records)==block_count:
                    print(json.dumps(dict(stage="banks",completed_blocks=len(records),total_blocks=block_count,
                                          elapsed_seconds=current-start)),flush=True)
                    last_progress=current
    if len(records)!=block_count:
        raise ValueError("incomplete panel")
    ordered=[records[index] for index in range(block_count)]
    score_path=out/"BANK_SCORES.npy"
    if not score_path.exists():
        temporary=score_path.with_suffix(".npy.partial")
        if temporary.exists():
            raise FileExistsError("partial score assembly retained; explicit repair required")
        joined=np.lib.format.open_memmap(temporary,mode="w+",dtype="float64",shape=(block_count*banks,4,5))
        for index in range(block_count):
            with np.load(out/records[index]["path"],allow_pickle=False) as raw:
                joined[index*banks:(index+1)*banks]=raw["scores"]
        joined.flush();del joined
        temporary.rename(score_path)
    else:
        saved=np.load(score_path,mmap_mode="r",allow_pickle=False)
        if saved.shape!=(block_count*banks,4,5):
            raise ValueError("existing BANK_SCORES.npy shape mismatch")
        for index in range(block_count):
            with np.load(out/records[index]["path"],allow_pickle=False) as raw:
                if not np.array_equal(saved[index*banks:(index+1)*banks],raw["scores"]):
                    raise ValueError("existing BANK_SCORES.npy differs from raw")
        del saved
    counts=dict(independent_banks=block_count*banks,block_count=block_count,particles_per_bank=particles,
                active_Gaussian_scalars=block_count*banks*particles*3,
                nuisance_Gaussian_scalars=block_count*banks*particles*4096,
                uniform_rho_scalars=block_count*banks,prior_vectors=block_count*banks*particles,
                new_observations=0,training_updates=0,scientific_banks=block_count*banks if scientific else 0,
                score_entries=block_count*banks*4*5,label_loop_iterations_per_bank=4*8*3,
                active_likelihood_evaluations_per_bank=4*8*4)
    index=dict(schema_version=core.SCHEMA_VERSION,freeze_sha256=freeze_sha,reference_sha256=reference_sha,
               attempt=attempt,scientific=scientific,metric_names=core.METRIC_NAMES,n_values=list(core.N_VALUES),
               records=ordered,resources=counts,scores=dict(path=score_path.name,sha256=sha(score_path),bytes=score_path.stat().st_size),
               elapsed_seconds=time.monotonic()-start)
    if (out/"INDEX.json").exists():
        old=read(out/"INDEX.json")
        if old["records"]!=ordered or old["scores"]!=index["scores"]:
            raise ValueError("existing completed index differs")
        return old
    write_new(out/"INDEX.json",index)
    return index


def run(*,resume: bool=False) -> dict:
    frozen,freeze_sha,reference,reference_sha=verify_freeze()
    start=OUT/"RUN_START.json"
    if (OUT/"RUN_COMPLETE.json").exists():
        raise FileExistsError("completed panel cannot be rerun")
    if resume:
        initial=read(start)
        if initial["freeze_sha256"]!=freeze_sha or initial["reference_sha256"]!=reference_sha:
            raise ValueError("resume requires identical freeze and reference")
    else:
        if start.exists():
            raise FileExistsError("existing RUN_START; explicit --resume required")
        write_new(start,dict(utc=now(),freeze_sha256=freeze_sha,reference_sha256=reference_sha,config=core.production_config()))
    previous=list(OUT.glob("RUN_ATTEMPT_*.json"))
    attempt=max((int(path.stem.split("_")[-1]) for path in previous),default=0)+1
    write_new(OUT/f"RUN_ATTEMPT_{attempt:03d}.json",dict(utc=now(),attempt=attempt,resume=resume,
                                                       freeze_sha256=freeze_sha,reference_sha256=reference_sha))
    try:
        index=schedule(OUT,m_H=float(reference["m_H45"]),freeze_sha=freeze_sha,reference_sha=reference_sha,
                       attempt=attempt,resume=resume)
        complete=dict(utc=now(),attempt=attempt,freeze_sha256=freeze_sha,reference_sha256=reference_sha,
                      index_sha256=sha(OUT/"INDEX.json"),resources=index["resources"],no_analysis_performed=True)
        write_new(OUT/"RUN_COMPLETE.json",complete)
        print(json.dumps(dict(stage="complete",blocks=core.BLOCK_COUNT,banks=core.TOTAL_BANKS)),flush=True)
        return complete
    except BaseException as error:
        write_new(OUT/f"ERROR_attempt_{attempt:03d}.json",dict(utc=now(),attempt=attempt,error=repr(error),
                                                           traceback=traceback.format_exc(),freeze_sha256=freeze_sha))
        raise


def check() -> dict:
    _,freeze_sha,_,reference_sha=verify_freeze()
    complete=read(OUT/"RUN_COMPLETE.json");index=read(OUT/"INDEX.json")
    for value in (complete,index):
        if value["freeze_sha256"]!=freeze_sha or value["reference_sha256"]!=reference_sha:
            raise ValueError("completion/freeze mismatch")
    if complete["index_sha256"]!=sha(OUT/"INDEX.json"):
        raise ValueError("index SHA mismatch")
    records=existing_receipts(OUT,dict(freeze_sha=freeze_sha,reference_sha=reference_sha),
                             block_count=core.BLOCK_COUNT,banks=core.BLOCK,particles=core.K,
                             seed_family=core.SEED_FAMILY,scientific=True)
    if len(records)!=core.BLOCK_COUNT or [records[i] for i in range(core.BLOCK_COUNT)]!=index["records"]:
        raise ValueError("index incomplete or differs from block receipts")
    score_path=OUT/index["scores"]["path"]
    if sha(score_path)!=index["scores"]["sha256"]:
        raise ValueError("score aggregate hash mismatch")
    saved=np.load(score_path,mmap_mode="r",allow_pickle=False)
    if saved.shape!=(core.TOTAL_BANKS,4,5) or saved.dtype!=np.dtype("float64") or not np.isfinite(saved).all():
        raise ValueError("score aggregate schema failure")
    for block_id in range(core.BLOCK_COUNT):
        with np.load(OUT/records[block_id]["path"],allow_pickle=False) as raw:
            if not np.array_equal(saved[block_id*core.BLOCK:(block_id+1)*core.BLOCK],raw["scores"]):
                raise ValueError("aggregate differs from block scores")
    result=dict(passed=True,scope="saved source hashes, raw schema/hashes and aggregate equality only",
                blocks=core.BLOCK_COUNT,banks=core.TOTAL_BANKS,new_RNG_calls=0,new_score_calls=0)
    print(json.dumps(result),flush=True)
    return result


def dummy() -> dict:
    start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bridge_manufactured_schedule_") as directory:
        out=Path(directory)
        index=schedule(out,m_H=.0123,freeze_sha="manufactured-only",reference_sha="manufactured-only",attempt=1,
                       block_count=3,banks=2,particles=3,workers=2,seed_family=202001010001,scientific=False)
        before={p.name:sha(p) for p in (out/"raw").glob("*.npz")}
        again=schedule(out,m_H=.0123,freeze_sha="manufactured-only",reference_sha="manufactured-only",attempt=2,
                       block_count=3,banks=2,particles=3,workers=2,seed_family=202001010001,scientific=False,resume=True)
        assert before=={p.name:sha(p) for p in (out/"raw").glob("*.npz")}
        assert index==again
        recovery_out=out/"incomplete_recovery"
        (recovery_out/"raw").mkdir(parents=True)
        (recovery_out/"raw"/"block_00000.npz.partial").write_bytes(b"manufactured incomplete output")
        recovered=schedule(recovery_out,m_H=.0123,freeze_sha="manufactured-only",reference_sha="manufactured-only",attempt=2,
                           block_count=1,banks=2,particles=3,workers=2,seed_family=202001010002,scientific=False,resume=True)
        assert (recovery_out/"failed"/"attempt_002"/"block_00000.npz.partial").read_bytes()==b"manufactured incomplete output"
        assert recovered["resources"]["independent_banks"]==2
        result=dict(passed=True,manufactured_only=True,scientific_banks=0,scientific_seed_calls=0,
                    scheduler_blocks=3,scheduler_banks=6,workers=2,resume_reused_blocks=3,
                    resumed_generation_calls=0,resumed_score_calls=0,temporary_outputs_removed=True,
                    incomplete_recovery_blocks=1,incomplete_recovery_banks=2,unfinished_file_preserved=True,
                    manufactured_rng_banks=8,manufactured_normal_scalars=8*3*(4096+3),
                    resources=index["resources"],elapsed_seconds=time.monotonic()-start)
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=OUT/"THROUGHPUT_DUMMY_SCHEDULE.json"
    if receipt.exists():
        result["previous_manufactured_receipt"]=read(receipt)
    result["sources"]={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),ROOT/"script/asymptotic_bridge_core.py")}
    with receipt.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    multiprocessing.freeze_support()
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run",action="store_true")
    modes.add_argument("--resume",action="store_true")
    modes.add_argument("--check",action="store_true")
    modes.add_argument("--dummy",action="store_true")
    args=parser.parse_args()
    if args.dummy:dummy()
    elif args.check:check()
    else:run(resume=args.resume)

"""RNG-free Windows spawn/environment check; never invokes production run."""
from pathlib import Path
import sys
import os
import json
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'script'))
_BARRIER=None

def initialize(barrier):
    global _BARRIER
    _BARRIER=barrier

def worker(index):
    import scbmeval_run as runner
    value=runner.environment()
    _BARRIER.wait(timeout=30)
    return dict(index=index,pid=os.getpid(),environment=value,source_sha256=runner.sha(runner.SOURCE))

if __name__=='__main__':
    import scbmeval_run as runner
    context=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=initialize,initargs=(context.Barrier(2),)) as pool:
        futures=[pool.submit(worker,i) for i in range(2)]
        results=[future.result() for future in futures]
    assert len({r['pid'] for r in results})==2
    assert all(r['environment']==runner.environment() for r in results)
    assert all(r['source_sha256']==runner.sha(runner.SOURCE) for r in results)
    result=dict(passed=True,source_sha256=runner.sha(runner.SOURCE),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        workers=results,new_scientific_RNG_draws=0,new_model_integrals=0,new_fits=0)
    runner.save(runner.OUT/'SCHEDULING_SELFTEST.json',result)
    runner.save(runner.OUT/'SCHEMA.json',runner.schema())
    print(json.dumps(dict(passed=True,distinct_workers=2,environment=runner.environment(),source_sha256=result['source_sha256']),indent=2))

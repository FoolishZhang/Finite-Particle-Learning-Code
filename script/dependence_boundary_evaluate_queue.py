"""Bounded two-process runner; scientific choices live in the frozen E evaluator."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, subprocess, sys, json

ROOT=Path(__file__).resolve().parents[1]
IDS=[f'{r}_K{k}_{s}_sd1' for r in ['I','O'] for k in [4,100] for s in ['iid','antithetic']]
def run(identifier):
    out=ROOT/'artifacts/dependence_boundary_v1/E';log=out/'logs'/f'{identifier}.log';log.parent.mkdir(parents=True,exist_ok=True)
    if (out/'scores'/identifier/'COMPLETE.json').exists():return dict(id=identifier,status='existing_complete_not_restarted')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    with log.open('x',encoding='utf-8') as stream:
        r=subprocess.run([sys.executable,'-m','script.dependence_boundary_evaluate','one','--id',identifier],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    return dict(id=identifier,exit_code=r.returncode,log=log.relative_to(ROOT).as_posix())
if __name__=='__main__':
    assert (ROOT/'artifacts/dependence_boundary_v1/E/FREEZE.json').exists()
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,i) for i in IDS]):
            r=f.result();results.append(r);print(json.dumps(r),flush=True)
    assert all(r.get('exit_code',0)==0 for r in results),results

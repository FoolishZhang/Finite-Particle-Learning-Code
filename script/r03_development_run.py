"""Frozen R03 numerical development; new production only under --run."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import r03_development_compute as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/r03_teacher_family_v1_20261011'
DOC='docs/08_research_研究工作/R03TeacherFamily_v1_20261011/'
BASE='f3e7a89b11bab82c9f93be6f24e628161e8b15ad'
HANDOFF_SHA='79be895b92cda2aac605797c758756e7932a7ce80ccb6d180735195fbfc0b17e'
A='artifacts/certified_teacher_v1_20261008/CANDIDATE_RISK_INTEGRALS.json'
D='artifacts/angular_witness_v1_20261010/DIRECTION.json'
PAPER='artifacts/submission_candidate_v1_9_1_20261010/DELIVERY_MANIFEST.json'
PAPER_SHA='6162e70659728bf956cf0be7ea56766624c55924d306cb8fe026b0751963fa8c'
SOURCES=[A,D,PAPER,'script/r03_development_run.py','script/r03_development_compute.py',
 'script/rankone_stationary_kernel.py','script/joint_dependence_kernel.py',
 DOC+'DEVELOPMENT_CONTRACT.md',DOC+'DEVELOPMENT_METHOD.md',
 'artifacts/r03_teacher_family_v1_20261011/DEV_IMPLEMENTATION_CHECK.json']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def now():return datetime.now(timezone.utc).isoformat()
def write(p,data,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf8',newline='\n') as f:
        json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify():
    lock=read(OUT/'DEV_FREEZE.json')
    for name,digest in lock['bindings'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    return read(OUT/'DEV_INPUTS.json')
def freeze():
    assert sha(ROOT/'HANDOFF.md')==HANDOFF_SHA and sha(ROOT/PAPER)==PAPER_SHA
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'HANDOFF_INPUT.md').open('xb') as f:f.write((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'DEV_INPUTS.json',dict(schema='r03_development_inputs_v1',baseline=BASE,
      handoff_sha256=HANDOFF_SHA,feedback_use='reference; user authorizes R03 through reasonable closure',
      initial_exact=read(ROOT/A)['a_exact'],direction=read(ROOT/D),
      lambdas=list(core.LAMBDAS),rules=core.RULES,gates=core.GATES,wall_seconds=600,
      manuscript_changed=False,new_version_created=False,exact_continuous_certificate=False))
    names=SOURCES+['artifacts/r03_teacher_family_v1_20261011/'+x for x in ('DEV_INPUTS.json','HANDOFF_INPUT.md')]
    write(OUT/'DEV_FREEZE.json',dict(schema='r03_development_freeze_v1',frozen_utc=now(),
      bindings={p:sha(ROOT/p) for p in names},target_calls_before_freeze=0))
def run():
    inp=verify();write(OUT/'DEV_START.json',dict(started_utc=now(),executable=sys.executable,
      freeze_sha256=sha(OUT/'DEV_FREEZE.json'),automatic_retry=False))
    saved=set()
    def checkpoint(state):
        compact={k:v for k,v in state.items() if k!='points'}
        compact['points']=[{k:v for k,v in p.items() if k!='evaluations'} for p in state['points']]
        write(OUT/'DEV_CHECKPOINT.json',compact,False)
        for p in state['points']:
            if p['status'] not in ('pending','running') and p['index'] not in saved:
                write(OUT/f"DEV_POINT_{p['index']:03}.json",p);saved.add(p['index'])
    result=core.compute(inp['initial_exact'],inp['direction'],wall_seconds=inp['wall_seconds'],
                        gates=inp['gates'],checkpoint=checkpoint)
    result['run_provenance']=dict(freeze_sha256=sha(OUT/'DEV_FREEZE.json'),
      started_record_sha256=sha(OUT/'DEV_START.json'),finished_utc=now())
    write(OUT/'DEVELOPMENT.json',result)
    print(json.dumps(dict(status=result['status'],all_points_qualified=result['all_points_qualified'],
       seconds=result['elapsed_compute_seconds'],points=[dict(lambda_exact=p['lambda_exact'],status=p['status']) for p in result['points']]),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true');g.add_argument('--check-freeze',action='store_true')
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.run:run()
    else:verify();print('Frozen development inputs unchanged.')

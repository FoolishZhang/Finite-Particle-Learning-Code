"""One-time pre-bank source lock for the fixed AsymptoticBridge-v1 panel."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess,sys
import asymptotic_bridge_core as core
import asymptotic_bridge_statistics as stats

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/asymptotic_bridge_v1_20261008'
DOC=ROOT/'docs/08_research_研究工作/AsymptoticBridge_v1_20261008'
BASE='1cfd58ddaeba096e39974f783a80f8a09153c24f'
HANDOFF='3e2703ead9948ba86d115335b0c31aeac2a903517c0064e846151e77678e6e2f'
PRIOR='67282c57b9962688f25ccdf629b007ab3c2bffd35a604e4888f656fe8f92959f'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8'))

def freeze():
    assert not (OUT/'FREEZE.json').exists(), 'Scientific source lock already exists'
    assert not (OUT/'RUN_START.json').exists(), 'A scientific run was already started'
    assert not list(OUT.rglob('*.npz')) and not list(OUT.rglob('*.npy')), 'Scientific raw output predates source freeze'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert sha(ROOT/'HANDOFF.md')==sha(OUT/'HANDOFF_INPUT.md')==HANDOFF
    prior=ROOT/'artifacts/submission_candidate_v1_7_1_20261008/DELIVERY_MANIFEST.json'
    assert sha(prior)==PRIOR
    stats.check_reference()
    receipts=['CORE_MANUFACTURED_CHECK.json','STATISTICS_MANUFACTURED_CHECK.json','AUDIT_MANUFACTURED_CHECK.json']
    for name in receipts:
        r=read(OUT/name)
        assert r.get('passed') is True, (name, 'Manufactured qualification must pass')
    paths=[Path(__file__), *[ROOT/('script/'+name) for name in (
        'asymptotic_bridge_core.py','asymptotic_bridge_core_test.py','asymptotic_bridge_run.py',
        'asymptotic_bridge_statistics.py','asymptotic_bridge_statistics_test.py',
        'asymptotic_bridge_audit.py','asymptotic_bridge_audit_test.py')],
        *[DOC/name for name in ('CONTRACT.md','MATH.md','HANDOFF_DISPOSITION.md')],
        *[OUT/name for name in ('HANDOFF_INPUT.md','REFERENCE_CONTRACT.json','REFERENCE.json',
            'PREFREEZE_RNG_NAMESPACE.json','MANUFACTURED_RESOURCE_LEDGER.json',*receipts)],prior]
    paths+= [ROOT/name for name in read(OUT/'REFERENCE.json')['source_hashes']]
    sources={p.resolve().relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    record={'schema':'asymptotic_bridge_scientific_freeze_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'baseline':BASE,'handoff_sha256':HANDOFF,'prior_v171_manifest_sha256':PRIOR,
        'config':core.production_config(),'sources':sources,'statistical_family_targets':26,'alpha':.01,
        'scientific_bank_count_before_freeze':0,'reference_calculations_already_completed':True,
        'reference_integrals':4,'reference_pair_nodes':31296,'manuscript_modified':False,
        'route2_executed':False,'no_scientific_values_used_for_budget_choice':True,
        'manufactured_and_reference_work_separately_recorded':True}
    with (OUT/'FREEZE.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'passed':True,'mode':'freeze','sources':len(sources),'config':record['config'],
        'freeze_sha256':sha(OUT/'FREEZE.json'),'new_scientific_banks':0}))

if __name__=='__main__':
    if sys.argv[1:]!=['--freeze']:raise SystemExit('Use --freeze exactly once before the formal run')
    freeze()

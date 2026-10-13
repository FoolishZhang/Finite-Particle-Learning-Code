"""Read-only frozen gradient verification with one explicit source relocation.

The original producer bytes and all scientific freezes/results are unchanged.
Its former filename collided with an older package's source-name enumeration.
"""
from pathlib import Path
import argparse, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nuisance_learning/GRADIENT'
OLD='script/nuisance_learning_gradient.py'
NEW='script/outer_learning_gradient_source.py'
HASH='9fc1e271678512e1b43f8386b298199264b0e1b2aa68926766c0b1c232036626'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(mapping):
    for p,h in mapping.items():
        actual=NEW if p==OLD else p
        assert sha(ROOT/actual)==h,(p,actual)
def check(workspace=False):
    relocation=read(OUT/'SOURCE_RELOCATION.json')
    assert relocation['old_path']==OLD and relocation['new_path']==NEW
    assert relocation['unchanged_source_sha256']==HASH==sha(ROOT/NEW)
    assert relocation['checker_sha256']==sha(__file__)
    f=read(OUT/'FREEZE.json');r=read(OUT/'RESULTS_FREEZE.json')
    assert f['files'][OLD]==HASH and not (ROOT/OLD).exists()
    assert r['source_freeze_sha256']==sha(OUT/'FREEZE.json')==relocation['source_freeze_sha256']
    assert sha(OUT/'RESULTS_FREEZE.json')==relocation['result_freeze_sha256']
    verify(f['files']);verify(r['files'])
    if workspace:verify(f['local_inputs']);verify(r['local_outputs'])
    assert read(OUT/'STATE.json')['status']=='complete'
    result=read(OUT/'RESULTS.json')
    assert len(result['estimators'])==18 and len(result['outer_B'])==90
    print(json.dumps(dict(integrity_passed=True,workspace=workspace,
        mapped_source_paths=1,original_source_bytes_unchanged=True,
        original_scientific_freezes_unchanged=True,
        GH_reference_qualified=result['GH_reference_qualified'],
        statistics_or_integrals_reexecuted=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',action='store_true');a=p.parse_args();check(a.workspace)

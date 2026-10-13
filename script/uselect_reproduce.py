"""Reproduce one USE-v2 stream in a fresh output directory with corrected JSON IO.

The original frozen numerical/training functions remain unchanged. This wrapper
only replaces their JSON writer and output location; all seeds/rules are reused.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import uselect_compute as core

def writer(path,value):
    def scalar(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=core.CASES,default='NULL');parser.add_argument('--rep',type=int,default=0)
    parser.add_argument('--out',required=True,help='Fresh directory below this repository .cache/')
    args=parser.parse_args();assert 0<=args.rep<32
    root=core.ROOT;original=core.OUT;core.validate()
    target=(root/args.out).resolve();allowed=(root/'.cache').resolve()
    assert target.is_relative_to(allowed) and target!=allowed and not target.exists()
    target.mkdir(parents=True)
    (target/'FREEZE.json').write_bytes((original/'FREEZE.json').read_bytes())
    core.OUT=target;core.save=writer
    core.teacher_reference();result=core.stream_run(core.CASES.index(args.case),args.rep)
    expected=core.read(original/args.case/f'rep_{args.rep:02}'/'RESULT.json')
    assert result['choices']==expected['choices']
    error=float(np.max(abs(np.array(result['candidate_risks'])-expected['candidate_risks'])))
    assert error<2e-10
    receipt=dict(passed=True,case=args.case,rep=args.rep,maximum_risk_difference=error,choices_identical=True,
        new_statistical_units=0,replayed_optimizer_runs=6,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Same-data/seed reproduction with original frozen math and corrected NumPy-scalar JSON serialization; no source file patched')
    writer(target/'REPRODUCTION.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()

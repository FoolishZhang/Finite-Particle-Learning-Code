"""One-shot orchestration of the already frozen TAIL reference computation.

This helper neither samples nor changes the scientific contract.  It waits for
all training records, stops on any training error, and invokes the frozen
reference entry point exactly once.  It does not retry either stage.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/joint_scale_learning'
TRAIN = OUT / 'TRAIN'


def run():
    log_path = OUT / 'REFERENCE_CONSOLE.log'
    assert not log_path.exists(), 'One-shot reference console already exists'
    print(json.dumps(dict(waiting_for='TRAIN/RESULTS.json', poll_seconds=45)), flush=True)
    while True:
        errors = sorted(TRAIN.glob('ERROR_stream_*.json'))
        if errors:
            print(json.dumps(dict(stopped_before_reference=True,
                training_errors=[p.name for p in errors], automatic_retry=False)), flush=True)
            return 2
        result_path = TRAIN / 'RESULTS.json'
        if result_path.exists():
            # The producer opens this file only after every worker has ended;
            # tolerate only a brief concurrent final-write window, not a retry.
            try:
                training = json.loads(result_path.read_text('utf8'))
            except json.JSONDecodeError:
                time.sleep(1)
                training = json.loads(result_path.read_text('utf8'))
            assert [r['stream'] for r in training['records']] == list(range(16))
            assert all(r['completed_steps'] == 512 for r in training['records'])
            break
        time.sleep(45)
    assert not (OUT / 'REFERENCE/EXECUTION.json').exists(), 'Reference already started'
    command = [sys.executable, '-u', str(ROOT / 'script/jointscale_compute.py'), 'reference']
    print(json.dumps(dict(reference_starting=True, training_numerical_qualification_passed=
        training['numerical_qualification_passed'], console=str(log_path))), flush=True)
    with log_path.open('x', encoding='utf8', newline='\n') as log:
        log.write(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(), command=command,
            automatic_retry=False)) + '\n')
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(json.dumps(dict(finished_utc=datetime.now(timezone.utc).isoformat(), exit_code=result.returncode)) + '\n')
    print(json.dumps(dict(reference_finished=True, exit_code=result.returncode,
        console=str(log_path))), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(run())

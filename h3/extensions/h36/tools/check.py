# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Check file identities and saved R12/R64 arithmetic; no training or integration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from saved import Audit, neural, read, risk64

HERE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(folder=HERE, identity=True):
    folder = Path(folder)
    audit = Audit()
    if identity:
        repo = folder.parents[2]
        for name, row in read(repo / 'release/FILE_MANIFEST.json')['files'].items():
            audit.require((repo / name).is_file(), 'declared file exists/' + name)
            if (repo / name).is_file():
                audit.require(sha(repo / name) == row['sha256'], 'file SHA256/' + name)
                audit.require((repo / name).stat().st_size == row['bytes'], 'file bytes/' + name)
        manifest = read(folder / 'MANIFEST.json')
        actual = {p.relative_to(repo).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'MANIFEST.json'}
        audit.require(set(manifest['files']) == actual, 'exact extension file manifest')
        for name, row in manifest['files'].items():
            audit.require(sha(repo / name) == row['sha256'], 'extension SHA256/' + name)
    panels, risk = neural(folder, audit), risk64(folder, audit)
    return {'schema': 'h36_public_saved_check_v1', 'passed': not audit.failures,
            'checks': audit.count, 'failures': audit.failures,
            'maximum_absolute_check_error': audit.maximum_absolute_error,
            'fresh_neural': panels, 'old_R64_risk': risk,
            'new_integrals': 0, 'new_training_runs': 0, 'new_optimization_calls': 0, 'new_RNG_draws': 0,
            'scope': 'Saved file identity, 96-C to 12 paired neural statistics, old equal-update scalar summaries, and 256-dataset R64 risk/control/bias-variance arithmetic. Reference summaries are saved descriptions; no original data or checkpoint revalidation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(HERE.parents[2]):
        raise ValueError('choose a fresh output outside the source checkout')
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'passed': result['passed'], 'checks': result['checks'], 'failures': result['failures']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

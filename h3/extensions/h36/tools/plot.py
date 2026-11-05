# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Redraw the h3.6 neural figure from saved matrices and paired statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from saved import Audit, neural
import drawing
from drawing import figure

HERE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(HERE.parents[2]):
        raise ValueError('choose a fresh output outside the source checkout')
    out.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    result = neural(HERE, audit)
    if audit.failures:
        raise ValueError('saved arithmetic failed: ' + '; '.join(audit.failures))
    # The manuscript function saves its optional SVG through a second global
    # output directory. Both outputs belong to this fresh external directory.
    drawing.OUT = out
    figure(result, out)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    expected = HERE / 'notes/figures/neural_structure.png'
    receipt = {'schema': 'h36_public_redraw_v1', 'saved_C_matrices': 96, 'independent_dataset_count': 12,
               'png_sha256': sha(out / 'neural_structure.png'),
               'approved_png_sha256': sha(expected),
               'png_bytes_identical': (out / 'neural_structure.png').read_bytes() == expected.read_bytes(),
               'new_integrals': 0, 'new_training_runs': 0, 'new_RNG_draws': 0,
               'scope': 'Deterministic redraw of the neural structure figure from saved matrices; no fitted coefficient or scientific producer.'}
    (out / 'REDRAW.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if receipt['png_bytes_identical'] else 1)


if __name__ == '__main__':
    main()

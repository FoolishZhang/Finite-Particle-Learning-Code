"""Verify the published file inventory. This does not verify mathematical proofs."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    record=json.loads((ROOT/'release/FILE_MANIFEST.json').read_text('utf-8'))
    for name,row in record['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or path.is_symlink():raise ValueError('Unsafe manifest path')
        data=path.read_bytes()
        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Release file differs: '+name)
    print(json.dumps({'passed':True,'files_checked':len(record['files']),
                      'scope':'published file identity only; not scientific recomputation'}))


if __name__=='__main__':main()

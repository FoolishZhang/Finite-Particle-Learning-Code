"""Fetch the original official template separately; do not relicense or vendor it."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True,help='Fresh templates/icml2026 directory in your paper working copy')
    args=parser.parse_args();target=args.destination.resolve()
    if target.exists():raise ValueError('Use a new directory; no reference files are overwritten')
    config=json.loads((ROOT/'paper/STYLE_DEPENDENCY.json').read_text('utf-8'))
    with urllib.request.urlopen(config['url']) as r:data=r.read()
    if hashlib.sha256(data).hexdigest()!=config['sha256']:raise ValueError('Official template archive changed')
    found={}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            name=PurePosixPath(info.filename).name
            if name in config['files']:
                if name in found:raise ValueError('Duplicate template member')
                content=z.read(info)
                if hashlib.sha256(content).hexdigest()!=config['files'][name]:raise ValueError('Template member differs')
                found[name]=content
    if set(found)!=set(config['files']):raise ValueError('Incomplete official archive')
    target.mkdir(parents=True)
    for name,data in found.items():
        with (target/name).open('xb') as stream:stream.write(data)
    print(json.dumps({'downloaded_from_official_source':True,'files':len(found),'relicensed':False}))


if __name__=='__main__':main()

"""Download one explicitly selected release asset; verify before safe extraction."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--asset')
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--extract-to',type=Path)
    args=parser.parse_args()
    assets=json.loads((ROOT/'assets/ASSETS.json').read_text('utf-8'))['assets']
    if args.list:
        print(json.dumps([{k:r[k] for k in ('asset_id','status','bytes','filename')} for r in assets],indent=2));return
    require(args.asset and args.cache,'Select --asset and a --cache directory; nothing downloads by default')
    matches=[r for r in assets if r['asset_id']==args.asset];require(len(matches)==1,'Unknown asset')
    item=matches[0];require(item['status']=='available' and item['download_url'],'Asset is not published')
    print(f"Selected {item['asset_id']}: {item['bytes']:,} bytes",flush=True)
    require(item['download_url'].startswith('https://github.com/FoolishZhang/Finite-Particle-Learning-Code/releases/download/'),'Unexpected asset host')
    name=item['filename'];require(PurePosixPath(name).name==name and '\\' not in name,'Unsafe filename')
    cache=args.cache.resolve();cache.mkdir(parents=True,exist_ok=True);path=cache/name
    if not path.exists():
        partial=cache/(name+'.partial');require(not partial.exists(),'Partial file already exists; inspect it before retrying')
        with urllib.request.urlopen(item['download_url']) as response,partial.open('xb') as output:
            shutil.copyfileobj(response,output,4*1024*1024)
        require(partial.stat().st_size==item['bytes'] and digest(partial)==item['sha256'],'Downloaded asset checksum differs')
        os.rename(partial,path)
    require(path.stat().st_size==item['bytes'] and digest(path)==item['sha256'],'Cached asset checksum differs')
    if args.extract_to:
        target=args.extract_to.resolve();require(not target.exists(),'Extraction target must be a fresh directory')
        expected={r['path']:r for r in item['members']}
        with zipfile.ZipFile(path) as z:
            require(set(z.namelist())==set(expected) and len(z.namelist())==len(expected),'Archive inventory differs')
            for info in z.infolist():
                pure=PurePosixPath(info.filename)
                require(not pure.is_absolute() and all(p not in ('','..','.') for p in pure.parts)
                        and '\\' not in info.filename and ':' not in info.filename,'Unsafe archive path')
                require(not stat.S_ISLNK(info.external_attr>>16),'Archive links are forbidden')
                require(info.file_size==expected[info.filename]['bytes'],'Member size differs')
            target.mkdir(parents=True)
            for info in z.infolist():
                out=(target/info.filename).resolve();require(out.is_relative_to(target),'Path escapes extraction root')
                out.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,out.open('xb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
                require(digest(out)==expected[info.filename]['sha256'],'Extracted member hash differs')
    print(json.dumps({'verified':True,'asset':item['asset_id'],'extracted':bool(args.extract_to)}))


if __name__=='__main__':main()

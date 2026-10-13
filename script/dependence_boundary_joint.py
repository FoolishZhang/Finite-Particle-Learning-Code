"""Registered 60-contrast cross-domain risk ledger; no new model selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, json, math
import numpy as np

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/dependence_boundary_v1';OUT=BASE/'JOINT'
METHODS=['NP','EQ','NPp','EQp']
NAMES=['NP_gain','EQ_gain','feedback_increment','prior_joint_increment','conditional_joint_increment']
W=np.array([[1,0,0,0],[0,1,0,0],[-1,1,0,0],[1,0,-1,0],[0,1,0,-1]],dtype=float)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def csvwrite(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seal(p,files,**extra):write(p,dict(time=datetime.now(timezone.utc).isoformat(),files={f.relative_to(ROOT).as_posix():sha(f) for f in files},**extra))
def freeze():
    assert not (OUT/'IMPLEMENTATION_FREEZE.json').exists();reg=read(OUT/'PLAN_FREEZE.json')
    assert sha(ROOT/reg['plan'])==reg['sha256']
    seal(OUT/'IMPLEMENTATION_FREEZE.json',[Path(__file__),ROOT/reg['plan'],OUT/'PLAN_FREEZE.json'],family=60)
def check(p):
    for name,h in read(p)['files'].items():assert sha(ROOT/name)==h,name
def load_npz(path):
    with np.load(path) as f:return {k:f[k] for k in ['pooled_scores','half_pooled_scores','stream_scores','influence']}
def load_cells(domain):
    cells=[]
    if domain=='CUB':
        for role in ['I','O']:
            for K in [4,100]:
                groups=[];passes=[]
                for sampler in ['iid','antithetic']:
                    ident=f'{role}_K{K}_{sampler}_sd1';folder=BASE/'E/scores'/ident;rec=read(folder/'COMPLETE.json');arr=[]
                    for rep in range(6):
                        p=folder/f'{rep}.npz';assert sha(p)==rec['models'][rep]['sha'];arr.append(load_npz(p))
                        passes.append([rec['models'][rep]['numerical_method_pass'][m] for m in METHODS])
                    groups.append({k:np.stack([a[k] for a in arr]) for k in arr[0]})
                cells.append((f'CUB/{role}/K{K}',groups,np.all(passes,axis=0)))
        with (BASE/'E/IMAGE_METRICS.csv').open(encoding='utf-8',newline='') as f:meta=list(csv.DictReader(f))
        ids=[x['image_id'] for x in meta];gids=[x['group_id'] for x in meta]
    else:
        for head in ['word','word_char']:
            for role in ['train','association_fit']:
                for K in [10,100]:
                    ident=f'{head}_{role}_K{K}';folder=BASE/'GE/scores'/ident;rec=read(folder/'COMPLETE.json');groups=[];passes=[]
                    for sampler_i in range(2):
                        arr=[]
                        for rep in range(3):
                            j=3*sampler_i+rep;p=folder/f'{j}.npz';assert sha(p)==rec['models'][j]['sha256'];arr.append(load_npz(p));passes.append(rec['models'][j]['method_pass'])
                        groups.append({k:np.stack([a[k] for a in arr]) for k in arr[0]})
                    cells.append((f'GoEmotions/{head}/{role}/K{K}',groups,np.all(passes,axis=0)))
        with np.load(BASE/'GE/cache/test.npz') as f:ids=f['ids'].astype(str).tolist();gids=f['groups'].astype(str).tolist()
    return cells,ids,gids
def bootstrap(d,gids,seed):
    _,gi=np.unique(gids,return_inverse=True);counts=np.bincount(gi);sums=np.stack([np.bincount(gi,weights=d[:,j]) for j in range(d.shape[1])],axis=1)
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(100):
        ix=rng.integers(0,len(counts),size=(100,len(counts)));vals.append(sums[ix].sum(1)/counts[ix].sum(1)[:,None])
    v=np.concatenate(vals);lo,hi=np.quantile(v,[.05/120,1-.05/120],axis=0);return v,lo,hi
def run():
    check(OUT/'IMPLEMENTATION_FREEZE.json');assert not (OUT/'COMPLETE.json').exists()
    assert (BASE/'E/COMPLETE.json').exists() and (BASE/'GE/COMPLETE.json').exists()
    seal(OUT/'INPUT_FREEZE.json',[BASE/'E/RESULTS_FREEZE.json',BASE/'GE/RESULTS_FREEZE.json',BASE/'E/PRIMARY.csv',BASE/'GE/PRIMARY.csv'])
    all_rows=[];details={};receipts=[]
    for domain,seed in [('CUB',20269601),('GoEmotions',20269602)]:
        cells,ids,gids=load_cells(domain);diffs=[];names=[];diags=[]
        for cell,(iid,anti),method_ok in cells:
            base=(iid['pooled_scores'][:,:,:4,0]-anti['pooled_scores'][:,:,:4,0]).mean(0)
            half=(iid['half_pooled_scores'][:,:,:4,0]-anti['half_pooled_scores'][:,:,:4,0]).mean(0)
            inf=(iid['influence'][:,:,:,:4]-anti['influence'][:,:,:,:4]).mean((0,2))
            stream=(iid['stream_scores'][:,:,:,:4,0]-anti['stream_scores'][:,:,:,:4,0]).mean((0,2))
            d=base@W.T;dh=half@W.T;infs=inf@W.T;st=stream@W.T
            for j,name in enumerate(NAMES):
                n=f'{cell}/{name}';names.append(n);diffs.append(d[:,j]);se=float(infs[:,j].std(ddof=1)/2);sens=abs(float((d[:,j]-dh[:,j]).mean()))
                ok=bool(np.all(method_ok[W[j]!=0]) and se<.0005 and sens<.0005)
                diags.append(dict(MC_SE=se,node_sensitivity=sens,cdf_floor_pass=bool(np.all(method_ok[W[j]!=0])),precision_pass=ok,
                    paired_pool_vs_mean_NLL_gap=float(st[:,j].mean()-d[:,j].mean())))
                details[n]=dict(weights=W[j].tolist(),influence=infs[:,j].tolist(),interpretation='risk identity; not causal share or information quantity')
        d=np.stack(diffs,axis=1);assert len(ids)==d.shape[0];v,lo,hi=bootstrap(d,gids,seed)
        p=OUT/f'{domain}_bootstrap.npz';np.savez_compressed(p,values=v);receipts.append(dict(domain=domain,seed=seed,images=len(ids),groups=len(set(gids)),contrasts=len(names),family=60,bootstrap_sha=sha(p)))
        image_rows=[]
        for i in range(len(ids)):image_rows.append(dict(id=ids[i],group=gids[i],**{n:float(d[i,j]) for j,n in enumerate(names)}))
        csvwrite(OUT/f'{domain}_IMAGE_CONTRASTS.csv',image_rows)
        for j,n in enumerate(names):
            all_rows.append(dict(contrast=n,mean=float(d[:,j].mean()),lower=float(lo[j]),upper=float(hi[j]),**diags[j],
                                 material_positive=bool(diags[j]['precision_pass'] and lo[j]>.005),material_negative=bool(diags[j]['precision_pass'] and hi[j]<-.005)))
    assert len(all_rows)==60
    csvwrite(OUT/'SIMULTANEOUS_60.csv',all_rows);write(OUT/'NUMERICAL_DETAILS.json',details);write(OUT/'BOOTSTRAP_RECEIPT.json',receipts)
    seal(OUT/'RESULTS_FREEZE.json',[OUT/'SIMULTANEOUS_60.csv',OUT/'NUMERICAL_DETAILS.json',OUT/'BOOTSTRAP_RECEIPT.json',OUT/'CUB_IMAGE_CONTRASTS.csv',OUT/'GoEmotions_IMAGE_CONTRASTS.csv'])
    write(OUT/'COMPLETE.json',dict(time=datetime.now(timezone.utc).isoformat(),contrasts=60,positive=sum(x['material_positive'] for x in all_rows),negative=sum(x['material_negative'] for x in all_rows),numerical_failed=sum(not x['precision_pass'] for x in all_rows)))
    print(read(OUT/'COMPLETE.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);a=p.parse_args();globals()[a.phase]()

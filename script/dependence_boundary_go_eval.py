"""GoEmotions historical test evaluation after all GN covariance fits are sealed."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math,os,time,traceback,warnings
import numpy as np
from scipy.special import expit,ndtri
from scipy.stats import qmc
from script.dependence_boundary_evaluate import predict as qmc_predict
from script.shared_uncertainty_holdout_inference import blocks,take,batch_grid

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/dependence_boundary_v1'
OUT=BASE/'GE';GN=BASE/'GN';GDATA=BASE/'GDATA';LEGACY=ROOT/'artifacts/cbm_public/reference_source_v5_20260908'
PLAN=ROOT/'docs/08_research_研究工作/边际依赖接口_20260929/PLAN_GE.md'
MAN=ROOT/'data/GoEmotions_reader_release_v5/processed/RECORD_MANIFEST.csv'
METHODS=['NP','EQ','NPp','EQp']
CFG=dict(heads=['word','word_char'],roles=['train','association_fit'],K=[10,100],samplers=['iid','antithetic'],reps=3,
         streams=4,nodes=1024,half_nodes=512,sobol_seed=20269801,cdf_grid=2049,cdf_check=4097,cdf_gate=1e-4,
         MC_SE_gate=.0005,node_gate=.0005,floor=1e-12,contrasts=16,bootstrap=10000,bootstrap_seed=20269811,material=.005)
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj,exclusive=True):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check_seal(p):
    for name,h in read(p)['files'].items():assert sha(ROOT/name)==h,name
def check_models():
    f=read(GN/'MODELS_FREEZE.json');assert f['models']==48 and f['test_opened'] is False
    assert read(GN/'AUDIT.json')['passed'];check_seal(GN/'FREEZE.json');check_seal(GN/'MODELS_FREEZE.json')
    assert len(f['files'])==8
    for name in f['files']:
        m=read(ROOT/name);assert np.asarray(m['factors']).shape==(6,27,27) and m['steps']==2000
    return [ROOT/p for p in f['files']]
def freeze():
    assert not OUT.exists();models=check_models()
    paths=[Path(__file__),PLAN,ROOT/'script/dependence_boundary_evaluate.py',ROOT/'script/shared_uncertainty_holdout_inference.py',
           ROOT/'script/shared_uncertainty_core.py',ROOT/'script/dependence_v4_core.py',ROOT/'script/dependence_boundary_goemotions.py',
           GN/'FREEZE.json',GN/'MODELS_FREEZE.json',GN/'AUDIT.json',GDATA/'FREEZE.json',GDATA/'DATA_SEAL.json',
           GDATA/'PREDICTION_SEAL.json',GDATA/'cache/train.npz',MAN]
    paths += sorted((GDATA/'raw').glob('*'))
    paths += [LEGACY/'models/GoEmotions'/h/'predictor.joblib' for h in CFG['heads']]
    write(OUT/'FREEZE.json',dict(time_utc=now(),config=CFG,files={p.relative_to(ROOT).as_posix():sha(p) for p in paths+models},
          new_data_role_authorization='Historical GoEmotions test rater0, only after all GN48 models sealed; independent of GDATA no-test package',
          no_new_fitting=True))
    (OUT/'.gitignore').write_text('/cache/\n/scores/\n*.npz\n',encoding='utf-8',newline='\n')
def check_freeze():
    assert read(OUT/'FREEZE.json')['config']==CFG;check_seal(OUT/'FREEZE.json');check_models()
def prepare():
    check_freeze();assert not (OUT/'DATA_FREEZE.json').exists()
    from script.dependence_boundary_goemotions import digest,norm,manual_tfidf
    import joblib
    from scipy.sparse import hstack
    from threadpoolctl import threadpool_limits
    with np.load(GDATA/'cache/train.npz') as z:C=z['labels'][:,0]
    table=[]
    for i in range(27):
        ranking=[]
        for j in range(27):
            if j==i:continue
            joint=np.bincount((2*C[:,i]+C[:,j]).astype(int),minlength=4).reshape(2,2)/len(C)
            independent=joint.sum(1)[:,None]*joint.sum(0)[None,:];mask=joint>0
            mi=float(np.sum(joint[mask]*np.log(joint[mask]/independent[mask])));ranking.append((mi,j))
        top=sorted(ranking,key=lambda t:(-t[0],t[1]))[:2]
        table.append(dict(query=i,partners=[t[1] for t in top],MI=[t[0] for t in top]))
    write(OUT/'PARTNERS.json',dict(time_utc=now(),train_only=True,rater=0,items=table))
    with MAN.open(encoding='utf-8',newline='') as f:manifest=list(csv.DictReader(f))
    selected=[r for r in manifest if r['role']=='test_locked'];lookup={r['id']:r for r in selected};assert len(lookup)==5622
    names=read(GDATA/'DATA_AUDIT.json')['concepts'];records={}
    for path in sorted((GDATA/'raw').glob('goemotions_*.csv')):
        with path.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                ident=digest('go|'+r['id'])[:24]
                if ident not in lookup:continue
                assert digest(norm(r['text']))==lookup[ident]['text_hash']
                if ident not in records:records[ident]=dict(text=r['text'],clear=[])
                assert records[ident]['text']==r['text']
                if r['example_very_unclear'].lower()=='false':
                    assert all(r[c] in ['0','1'] for c in names)
                    records[ident]['clear'].append((digest('20260908501|'+r['id']+'|'+r['rater_id']),r['rater_id'],[int(r[c]) for c in names]))
    assert set(records)==set(lookup)
    ids=np.array([r['id'] for r in selected]);groups=np.array([r['group_id'] for r in selected]);texts=[records[i]['text'] for i in ids]
    assert len(set(groups))==3595 and all(len(records[i]['clear'])>=3 for i in ids)
    Y=np.array([sorted(records[i]['clear'])[0][2] for i in ids],dtype=float)
    (OUT/'cache').mkdir(exist_ok=True);files=[];audits=[]
    data=OUT/'cache/test.npz';np.savez_compressed(data,ids=ids,groups=groups,labels=Y);files.append(data)
    with threadpool_limits(limits=2):
        for head in CFG['heads']:
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter('always');model=joblib.load(LEGACY/'models/GoEmotions'/head/'predictor.joblib')
            word=model['word'].transform(texts);X=word if head=='word' else hstack([word,model['char'].transform(texts)],format='csr')/np.sqrt(2)
            mu=np.stack([c.decision_function(X) for c in model['classifiers']],axis=1)
            coef=np.stack([c.coef_[0] for c in model['classifiers']]);bias=np.array([c.intercept_[0] for c in model['classifiers']])
            manual=manual_tfidf(model['word'],texts[:16])
            if head=='word_char':manual=hstack([manual,manual_tfidf(model['char'],texts[:16])],format='csr')/np.sqrt(2)
            error=max(float(np.max(abs(np.asarray(X@coef.T)+bias-mu))),float(np.max(abs(np.asarray(manual@coef.T)+bias-mu[:16]))))
            assert error<1e-10 and not notices
            query=np.argmin(abs(mu),axis=1);partners=np.array([table[q]['partners'] for q in query])
            path=OUT/'cache'/f'{head}.npz';np.savez_compressed(path,mu=mu,query=query,partners=partners,ids=ids);files.append(path)
            audits.append(dict(head=head,n=5622,feature_dimensions=X.shape[1],independent_prediction_error=error,
                               query_counts=np.bincount(query,minlength=27).tolist(),query_uses_target=False,partner_selection='train label MI',new_training=0))
    write(OUT/'DATA_AUDIT.json',dict(time_utc=now(),status='pass_historical_test_restoration_after_GN_seal',records=5622,groups=3595,
          rater=0,heads=audits,GN_models_sealed_before_test=True,no_clinical_data=True))
    files += [OUT/'DATA_AUDIT.json',OUT/'PARTNERS.json']
    write(OUT/'DATA_FREEZE.json',dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
def product(q):
    a=q[:,2]+q[:,3];b=q[:,1]+q[:,3]
    return np.column_stack([(1-a)*(1-b),(1-a)*b,a*(1-b),a*b])
def all_methods(base):return np.stack([base[:,0],base[:,1],product(base[:,0]),product(base[:,1])],axis=1)
def targets(q,Y):
    state=(2*Y[:,0]+Y[:,1]).astype(int);return q[np.arange(len(Y))[:,None],np.arange(4)[None],state[:,None]]
def scored(q,Y):
    floor=(q<CFG['floor']).sum((0,2));targetfloor=(targets(q,Y)<=CFG['floor']).sum(0)
    q=np.maximum(q,CFG['floor']);q/=q.sum(2,keepdims=True);nll=-np.log(targets(q,Y))
    margins=np.stack([q[:,:,2]+q[:,:,3],q[:,:,1]+q[:,:,3]],axis=2)
    brier=((margins-Y[:,None])**2).mean(2);parity=((q[:,:,0]+q[:,:,3])-(Y[:,0]==Y[:,1])[:,None])**2
    return np.stack([nll,brier,parity],axis=2),q,floor,targetfloor
def cdf_audit(p,query,Y,u):
    chosen=set()
    for q in np.unique(query):
        for y in [0,1]:
            ix=np.flatnonzero((query==q)&(p['y']==y))
            if len(ix):
                chosen.add(int(ix[np.argmin(p['query_mu'][ix])]))
                chosen.add(int(ix[np.argmax(p['query_mu'][ix])]))
    chosen.update(np.argsort(-np.max(abs(p['mu']),axis=1),kind='stable')[:32].tolist());ix=np.array(sorted(chosen),dtype=int)
    low,_=qmc_predict(take(p,ix),u,2049,512);high,_=qmc_predict(take(p,ix),u,4097,512)
    l,_,lf,lt=scored(all_methods(low),Y[ix]);h,_,hf,ht=scored(all_methods(high),Y[ix])
    error=np.max(abs(l[:,:,0]-h[:,:,0]),axis=0)
    return dict(indices=ix.tolist(),cases=len(ix),selection_used_partner_labels=False,NLL_max_error=error.tolist(),
                target_floors=(lt+ht).tolist(),floor_entries=(lf+hf).tolist(),method_pass=((error<1e-4)&(lt+ht==0)).tolist())
def identifiers():
    for h in CFG['heads']:
        for r in CFG['roles']:
            for k in CFG['K']:yield f'{h}_{r}_K{k}'
def evaluate_one(name):
    check_freeze();check_seal(OUT/'DATA_FREEZE.json');model=read(GN/'models'/f'{name}.json');c=model['config']
    path=OUT/'scores'/name;path.mkdir(parents=True,exist_ok=True);assert not (path/'COMPLETE.json').exists()
    with np.load(OUT/'cache/test.npz') as z:Yall=z['labels']
    with np.load(OUT/'cache'/f"{c['head']}.npz") as z:mu=z['mu'];query=z['query'];partners=z['partners']
    feedback=Yall[np.arange(len(mu)),query];Y=np.take_along_axis(Yall,partners,axis=1)
    uniforms=[qmc.Sobol(3,scramble=True,seed=CFG['sobol_seed']+i).random_base2(10) for i in range(4)]
    normals=[ndtri(u) for u in uniforms]
    receipts=[];start=time.perf_counter()
    for j in range(6):
        output=path/f'{j}.npz';assert not output.exists();L=np.array(model['factors'][j]);p=blocks(mu,L@L.T,query,partners,feedback)
        cdf=cdf_audit(p,query,Y,uniforms[0]);pool=np.zeros((5622,4,3));half=np.zeros_like(pool)
        streams=np.zeros((4,5622,4,3));influence=np.zeros((4,5622,4))
        floors={k:np.zeros(4,dtype=int) for k in ['pool','half','streams']};tfloors={k:np.zeros(4,dtype=int) for k in floors}
        for begin in range(0,5622,128):
            sl=slice(begin,begin+128);pp=take(p,sl);fullbases=[];halfbases=[];qs=[];grid=batch_grid(pp,2049)
            for st,u in enumerate(uniforms):
                full,small=qmc_predict(pp,u,2049,512,grid=grid,normal=normals[st]);fullbases.append(full);halfbases.append(small)
                value,q,f,tf=scored(all_methods(full),Y[sl]);streams[st,sl]=value;qs.append(q);floors['streams']+=f;tfloors['streams']+=tf
            value,q,f,tf=scored(all_methods(np.mean(fullbases,axis=0)),Y[sl]);pool[sl]=value;floors['pool']+=f;tfloors['pool']+=tf
            value,_,f,tf=scored(all_methods(np.mean(halfbases,axis=0)),Y[sl]);half[sl]=value;floors['half']+=f;tfloors['half']+=tf
            target=targets(q,Y[sl]);influence[:,sl]=np.stack([-(targets(qq,Y[sl])-target)/target for qq in qs])
        np.savez_compressed(output,pooled_scores=pool,half_pooled_scores=half,stream_scores=streams,influence=influence)
        passed=np.array(cdf['method_pass']) & (sum(tfloors.values())==0)
        receipt=dict(index=j,rep=j%3,sampler=model['samplers'][j],cdf=cdf,method_pass=passed.tolist(),
                     floors={k:v.tolist() for k,v in floors.items()},target_floors={k:v.tolist() for k,v in tfloors.items()},sha256=sha(output))
        receipts.append(receipt)
        print(name,j,dict(zip(METHODS,np.round(pool[:,:,0].mean(0),6))),round(time.perf_counter()-start,1),flush=True)
    write(path/'COMPLETE.json',dict(time_utc=now(),id=name,models=receipts,seconds=time.perf_counter()-start))
def bootstrap(d,groups):
    unique=sorted(set(groups));mapping={g:i for i,g in enumerate(unique)};gi=np.array([mapping[g] for g in groups]);counts=np.bincount(gi)
    sums=np.stack([np.bincount(gi,weights=d[:,j]) for j in range(d.shape[1])],axis=1);rng=np.random.default_rng(CFG['bootstrap_seed']);out=[]
    for _ in range(50):
        ix=rng.integers(0,len(unique),size=(200,len(unique)));out.append(sums[ix].sum(1)/counts[ix].sum(1)[:,None])
    values=np.concatenate(out);lo,hi=np.quantile(values,[.05/(2*16),1-.05/(2*16)],axis=0);return values,lo,hi
def collect():
    check_freeze();check_seal(OUT/'DATA_FREEZE.json');assert not (OUT/'COMPLETE.json').exists()
    summaries=[];primary=[];differences=[];names=[];streamrows=[];numerical=[];all_receipts={}
    with np.load(OUT/'cache/test.npz') as z:ids=z['ids'];groups=z['groups']
    for name in identifiers():
        receipt=read(OUT/'scores'/name/'COMPLETE.json');all_receipts[name]=receipt;m=read(GN/'models'/f'{name}.json');c=m['config'];arrays=[]
        for j in range(6):
            p=OUT/'scores'/name/f'{j}.npz';assert sha(p)==receipt['models'][j]['sha256']
            with np.load(p) as z:a={k:z[k] for k in z.files}
            assert a['pooled_scores'].shape==(5622,4,3);arrays.append(a)
            for method in range(4):summaries.append(dict(**c,sampler=m['samplers'][j],rep=j%3,method=METHODS[method],
                NLL=float(a['pooled_scores'][:,method,0].mean()),Brier=float(a['pooled_scores'][:,method,1].mean()),
                parity_Brier=float(a['pooled_scores'][:,method,2].mean()),numerical_method_pass=receipt['models'][j]['method_pass'][method]))
        for method in [0,1]:
            diff=np.mean([arrays[j]['pooled_scores'][:,method,0]-arrays[j+3]['pooled_scores'][:,method,0] for j in range(3)],axis=0)
            half=np.mean([arrays[j]['half_pooled_scores'][:,method,0]-arrays[j+3]['half_pooled_scores'][:,method,0] for j in range(3)],axis=0)
            influence=np.mean([arrays[j]['influence'][:,:,method]-arrays[j+3]['influence'][:,:,method] for j in range(3)],axis=(0,2))
            streamdiff=np.mean([arrays[j]['stream_scores'][:,:,method,0]-arrays[j+3]['stream_scores'][:,:,method,0] for j in range(3)],axis=(0,2))
            se=float(influence.std(ddof=1)/2);node=abs(float(diff.mean()-half.mean()));passed=all(r['method_pass'][method] for r in receipt['models'])
            contrast=f'{name}_{METHODS[method]}_iid_minus_antithetic';names.append(contrast);differences.append(diff)
            numerical.append(dict(contrast=contrast,MC_SE=se,node_sensitivity=node,cdf_and_target_floors_pass=bool(passed),
                                  precision_pass=bool(passed and se<.0005 and node<.0005),half_difference=float(half.mean()),
                                  mean_stream_NLL_difference=float(streamdiff.mean()),pool_vs_mean_stream_gap=float(streamdiff.mean()-diff.mean())))
            for st in range(4):streamrows.append(dict(contrast=contrast,stream=st,influence=float(influence[st]),stream_NLL_difference=float(streamdiff[st])))
    d=np.stack(differences,axis=1);assert d.shape==(5622,16);values,lo,hi=bootstrap(d,groups)
    for j,name in enumerate(names):primary.append(dict(**numerical[j],mean=float(d[:,j].mean()),lower=float(lo[j]),upper=float(hi[j]),
                                                    material_pass=bool(numerical[j]['precision_pass'] and lo[j]>.005)))
    csvwrite(OUT/'SUMMARY.csv',summaries);csvwrite(OUT/'PRIMARY.csv',primary);csvwrite(OUT/'STREAM_DIAGNOSTICS.csv',streamrows)
    csvwrite(OUT/'IMAGE_CONTRASTS.csv',[dict(id=ids[i],group_id=groups[i],**{name:float(d[i,j]) for j,name in enumerate(names)}) for i in range(len(ids))])
    write(OUT/'BOOTSTRAP.json',dict(seed=CFG['bootstrap_seed'],contrasts=names,values=values.tolist(),draws=10000,groups=3595))
    write(OUT/'NUMERICAL_RECEIPTS.json',all_receipts)
    write(OUT/'AUDIT.json',dict(time_utc=now(),status='complete_all_fixed_GE_comparisons',models=48,contrasts=16,
          precision_pass=sum(r['precision_pass'] for r in primary),material_pass=sum(r['material_pass'] for r in primary),
          failed_precision_contrasts=[r['contrast'] for r in primary if not r['precision_pass']],negative_results_retained=True,
          no_new_fits=True,historical_test=True,claim_scope='conditional fixed-model development; not best feedback strategy or original-method confirmation'))
    write(OUT/'RUN_STATE.json',dict(status='complete',active_jobs=[],time_utc=now()),exclusive=False)
    public=sorted(p for p in OUT.rglob('*') if p.is_file() and p.relative_to(OUT).parts[0] not in ['cache','scores'])
    write(OUT/'COMPLETE.json',dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in public},all_scientific_comparisons_complete=True))
    print(read(OUT/'AUDIT.json'),flush=True)
def run():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        if not (OUT/'DATA_FREEZE.json').exists():prepare()
        for name in identifiers():
            if (OUT/'scores'/name/'COMPLETE.json').exists():continue
            write(OUT/'RUN_STATE.json',dict(status='running',pid=os.getpid(),configuration=name,time_utc=now()),exclusive=False)
            evaluate_one(name)
        collect()
def check():
    check_freeze();check_seal(OUT/'DATA_FREEZE.json');check_seal(OUT/'COMPLETE.json');print('pass_GE_freezes_and_public_outputs')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','prepare','run','collect','check']);args=parser.parse_args()
    try:globals()[args.phase]()
    except Exception:
        if OUT.exists():write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=now(),traceback=traceback.format_exc()))
        raise

"""Independent stored-bank statistics and polynomial probability spot replay."""
from pathlib import Path
import argparse,csv,hashlib,json,math,time
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/objective_strengthening/B'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def verify(m):
    for n,h in m.items():assert sha(ROOT/n)==h,n
def stats(v,family,gate):
    v=np.asarray(v,dtype=np.longdouble);avg=float(np.sum(v)/len(v));se=float(np.sqrt(np.sum((v-avg)**2)/(len(v)*(len(v)-1))))
    radius=float(t.isf(.025/(2*family),15))*se
    return dict(mean=avg,SE=se,lower=avg-radius,upper=avg+radius,precision_pass=bool(se<=gate))
def audit():
    start=time.perf_counter();seal=read(OUT/'RESULTS_SEAL.json');verify(seal['files']);verify(seal['local_arrays'])
    teacher=read(ROOT/'artifacts/structure_learning_v2/A/TEACHER.json');p=np.array(teacher['variants'][2])
    qs=np.array([read(ROOT/f'artifacts/structure_learning_v2/B/reference/g{i:03}.json')['variants'][2] for i in range(361)])
    risk=np.sum(p*(np.log(p)-np.log(qs)),axis=1)
    storedrisk=table(OUT/'REFERENCE_RISK.csv');riskerr=max(abs(risk[i]-float(r['true_joint_KL'])) for i,r in enumerate(storedrisk));assert riskerr<1e-11
    pop=np.empty((16,3,361));fin=np.empty_like(pop);cmaps=[];fmap=[];raws=[];labels=[]
    endpoints=table(OUT/'ENDPOINTS.csv');selected={};select_count=0;curveerr=0.;meanerr=0.
    for s in range(16):
        old=read(ROOT/f'artifacts/structure_learning_v2/A/streams/s{s:02}.json')
        ni=old['N'].index(65536);ri=old['r_values'].index(0);labels.append(np.asarray(old['training_counts_by_r'][ri][ni])/65536)
        with np.load(OUT/f'local/train_s{s:02}.npz') as z:
            raws.append((z['population'][:4].copy(),z['finite'][:4].copy()))
            for bi,B in enumerate([64,256,1024]):
                pop[s,bi]=np.asarray(z['population'][:B],np.longdouble).mean(0)
                fin[s,bi]=np.asarray(z['finite'][:B],np.longdouble).mean(0)
        rec=read(OUT/f'confirmation/s{s:02}.json')
        with np.load(ROOT/rec['array_path']) as z:
            ids=z['indices'].tolist();a=np.asarray(z['population'],np.longdouble).mean(0);b=np.asarray(z['finite'],np.longdouble).mean(0)
            meanerr=max(meanerr,float(np.max(abs(a-np.asarray(rec['population_means'])))),float(np.max(abs(b-np.asarray(rec['finite_means'])))))
            cmaps.append(dict(zip(ids,map(float,a))));fmap.append((ids,z['population'][:4].copy(),z['finite'][:4].copy()))
        sel=read(OUT/f'selections/s{s:02}.json')
        for row in sel['selections']:
            bi=[64,256,1024].index(row['B']);v=pop[s,bi] if row['mode']=='population' else fin[s,bi]
            ids=np.flatnonzero(v<=v.min()+1e-12).tolist();near=np.flatnonzero(v<=v.min()+.005).tolist()
            assert row['index']==ids[0] and row['near_indices']==near
            curveerr=max(curveerr,abs(float(v[ids[0]])-row['objective']))
            selected[s,row['B'],row['mode']]=row['index'];select_count+=1
    assert meanerr<1e-10 and curveerr<1e-10 and select_count==96
    # Reconstruct old exact J choices directly from fixed labels and references.
    oldj=[int(np.argmin(-np.log(qs)@v)) for v in labels]
    serr=0.;fail=[]
    for row in table(OUT/'CONTRASTS_35.csv'):
        kind=row['kind'];B=int(row['B'])
        if kind=='fixed_confirmation_F':values=[v[int(row['index'])]-v[232] for v in cmaps];gate=.005
        elif kind=='selected_procedure_confirmation_F':values=[cmaps[s][selected[s,B,'population']]-cmaps[s][232] for s in range(16)];gate=.005
        elif kind=='selected_procedure_true_KL':values=[risk[selected[s,B,'population']] for s in range(16)];gate=.001
        else:assert kind=='old_labels_selected_procedure_risk';values=[risk[selected[s,B,'finite']]-risk[oldj[s]] for s in range(16)];gate=.001
        actual=stats(values,35,gate)
        for k in ['mean','SE','lower','upper']:serr=max(serr,abs(actual[k]-float(row[k])))
        assert actual['precision_pass']==(row['precision_pass']=='True')
        if not actual['precision_pass']:fail.append(row['comparison'])
    bands=table(OUT/'POPULATION_BANDS_1080.csv');bounds={B:(np.zeros(361),np.zeros(361)) for B in [64,256,1024]}
    for row in bands:
        B=int(row['B']);i=int(row['index']);bi=[64,256,1024].index(B)
        actual=stats(pop[:,bi,i]-pop[:,bi,232],1080,.01)
        for k in ['mean','SE','lower','upper']:serr=max(serr,abs(actual[k]-float(row[k])))
        assert actual['precision_pass']==(row['precision_pass']=='True')
        bounds[B][0][i]=actual['lower'];bounds[B][1][i]=actual['upper']
    for row in read(OUT/'POSSIBLE_GRID_OPTIMAL_SETS.json'):
        lo,hi=bounds[row['B']];assert np.flatnonzero(lo<=hi.min()).tolist()==row['possible_grid_optimal_indices']
        assert np.flatnonzero(lo<=hi.min()+.005).tolist()==row['possible_near_optimal_indices']
    assert serr<1e-10
    # Independent probability path: dense symmetric matrix square root + convolution.
    d=16;c=.9;R=(1+c/15)*np.eye(d)-c/15*np.ones((d,d));w,V=np.linalg.eigh(R);A=(V*np.sqrt(w))@V.T
    lg=np.array([math.log(math.comb(8,i)*math.comb(8,j)) for i in range(9) for j in range(9)])
    anchors=[40,60,116,232];perr=0.;replayed=0
    def poly(x):
        a=np.array([1.])
        for v in x:
            ev=math.exp(-abs(float(v)));pr=1/(1+ev) if v>=0 else ev/(1+ev);qr=ev/(1+ev) if v>=0 else 1/(1+ev)
            a=np.convolve(a,[qr,pr])
        return a
    for s in range(16):
        for phase,seed in [('train',20601001+s),('confirm',30601001+s)]:
            g=np.random.Generator(np.random.PCG64(seed));z=g.standard_normal((160,16))[:40]@A
            for i in anchors:
                u,v=divmod(i,19);uv=(u/2,v/2);eta=z*np.r_[np.full(8,uv[0]),np.full(8,uv[1])]
                masses=np.array([np.outer(poly(x[:8]),poly(x[8:])).ravel() for x in eta]).reshape(4,10,81).sum(1)/10
                loss=-np.log(masses)+lg
                if phase=='train':column=i;st=raws[s]
                else:column=fmap[s][0].index(i);st=fmap[s][1:]
                perr=max(perr,float(np.max(abs(loss@p-st[0][:,column]))),float(np.max(abs(loss@labels[s]-st[1][:,column]))));replayed+=4
    assert perr<1e-9
    return dict(passed=True,source_sha256=sha(__file__),results_seal_sha256=sha(OUT/'RESULTS_SEAL.json'),
      public_inputs=seal['files'],local_arrays=seal['local_arrays'],all_selections=96,contrasts_recomputed=35,bands_recomputed=1080,
      fixed_candidate_bank_scores_replayed=replayed,probability_replay_anchors=anchors,first_banks_per_stream_phase=4,
      full_probability_run_reexecuted=False,producer_functions_imported=False,old_exact_J_recomputed=16,
      reference_risk_error=riskerr,confirmation_mean_error=meanerr,selection_curve_error=curveerr,
      statistics_max_error=serr,polynomial_score_replay_max_error=perr,precision_failures_preserved=fail,
      new_scientific_samples=0,new_scientific_fits=0,elapsed_seconds=time.perf_counter()-start)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['write','check']);a.add_argument('--workspace',action='store_true');args=a.parse_args()
    path=OUT/'INDEPENDENT_AUDIT.json'
    if args.phase=='write':
        assert not path.exists();r=audit()
        with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in r.items() if k not in ['public_inputs','local_arrays']}))
    else:
        r=read(path);assert r['passed'] and r['source_sha256']==sha(__file__) and r['results_seal_sha256']==sha(OUT/'RESULTS_SEAL.json')
        verify(r['public_inputs'])
        if args.workspace:verify(r['local_arrays'])
        print(json.dumps(dict(passed=True,mode='receipt_and_hash_check',arithmetic_reexecuted=False)))

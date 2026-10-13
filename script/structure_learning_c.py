"""Unknown-group scale learning, with exhaustive pair search and fixed panels."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, csv, hashlib, itertools, json, math, os, sys, time, traceback
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
from scipy.stats import t as student_t
import scale_learning_core as core
from structure_learning_mapping import swap_kernel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/structure_learning_v2/C'
B = ROOT / 'artifacts/structure_learning_v2/B'
DOC = ROOT / 'docs/08_research_研究工作/结构先验释放与分组学习_v2_20260930'
CFG = dict(d=16, c=.9, profiles={'null': [4., 4.], 'contrast': [6., 2.]},
    scales=[j/2 for j in range(19)], streams=16, N=[256, 4096, 65536], seed=20350101,
    methods=['ORACLE_J', 'ORACLE_P', 'GLOBAL_P', 'ROW_P', 'GLOBAL_G10', 'GLOBAL_G100'],
    tie_tolerance=1e-12, gap_roundoff_tolerance=1e-13, numerical_gate=1e-10,
    reference_sensitivity_gate=1e-7, mass_gate=1e-12, family=36, alpha=.05, SE_gate=.002,
    primary_N=65536, material=.001, lambdas=[.0001, .0003, .001, .003, .01, .03, .1])

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def native(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return x.relative_to(ROOT).as_posix()
    raise TypeError(type(x).__name__)
def write(p, value, replace=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w' if replace else 'x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=native, allow_nan=False); f.write('\n')
def csvwrite(p, values):
    with p.open('x', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
def verify(condition, message):
    if not condition: raise RuntimeError(message)
def hash_map(paths): return {Path(p).relative_to(ROOT).as_posix(): sha(p) for p in paths}

def freeze():
    verify(not OUT.exists(), 'C directory exists')
    verify(read(B/'STATE.json')['status']=='complete', 'B must finish before C freeze')
    OUT.mkdir(parents=True); (OUT/'local').mkdir(); (OUT/'source_inputs').mkdir()
    (OUT/'.gitignore').write_text('/local/\n', encoding='utf-8', newline='\n')
    (OUT/'source_inputs/HANDOFF.md').write_bytes((ROOT/'HANDOFF.md').read_bytes())
    write(OUT/'CONFIG.json', CFG)
    write(OUT/'ENVIRONMENT.json', dict(python=sys.version, numpy=np.__version__))
    inputs=[Path(__file__), ROOT/'script/structure_learning_mapping.py', ROOT/'script/scale_learning_core.py',
            DOC/'PLAN_C.md', DOC/'UNKNOWN_GROUP_THEORY.md', OUT/'CONFIG.json', OUT/'ENVIRONMENT.json', OUT/'source_inputs/HANDOFF.md',
            B/'FREEZE.json', B/'RESULTS_FREEZE.json']
    inputs+=sorted((B/'reference').glob('g*.json'))+sorted((B/'streams').glob('*.json'))
    arrays={str((B/'local'/f's{s:02}.npz').relative_to(ROOT)).replace('\\','/'):sha(B/'local'/f's{s:02}.npz') for s in range(16)}
    write(OUT/'FREEZE.json', dict(time_utc=now(), baseline_commit='5cfbd418a48732ce33c5cde04b370fa5e098cd5f',
          files=hash_map(inputs), B_local_panel_hashes=arrays, reused_B_panels_not_fresh_confirmation=True))
    write(OUT/'STATE.json', dict(status='frozen', time_utc=now(), active_jobs=[]))
    print(json.dumps(dict(phase='freeze_C', sha256=sha(OUT/'FREEZE.json'))), flush=True)

def checkfreeze():
    seal=read(OUT/'FREEZE.json'); verify(read(OUT/'CONFIG.json')==CFG, 'C config changed')
    for group in ('files','B_local_panel_hashes'):
        for name, expected in seal[group].items(): verify(sha(ROOT/name)==expected, 'C frozen input changed '+name)

def geometry():
    patterns=np.arange(65536, dtype=np.uint16)
    popcount=np.array([int(x).bit_count() for x in range(65536)], dtype=np.uint8)
    masks=np.array([x for x in range(65536) if int(x).bit_count()==8], dtype=np.uint16)
    membership=((masks[:, None].astype(np.uint32)>>np.arange(16))&1).astype(float)
    spins=2*((patterns[:, None].astype(np.uint32)>>np.arange(16))&1).astype(float)-1
    return patterns,popcount,masks,membership,spins

def code_for_mask(patterns,popcount,mask):
    return 9*popcount[np.bitwise_and(patterns, np.uint16(mask))]+popcount[np.bitwise_and(patterns, np.uint16(65535^int(mask)))]

def load_library():
    refs=[read(B/'reference'/f'g{i:03}.json') for i in range(361)]
    for i,r in enumerate(refs):
        verify(r.get('qualified',r.get('numerical_pass',False)), 'unqualified B reference')
        verify(np.allclose(r['scales'],[i//19/2,i%19/2],rtol=0,atol=0),'B reference ordering')
    variants=np.array([r['variants'] for r in refs])
    _,_,n,lg,lt,_,ind=core.geometry(16)
    q=variants[:,2]; joint=-np.log(q)+lg
    pair=np.array([core.pair_loss_vectors(r) for r in refs])
    mag=np.array([r['independent_Hermite_moments']['pair_magnitudes_AA_BB_AB'] for r in refs])
    coefficient=np.arctanh(mag)/120
    constant=math.log(4)-.5*(np.log1p(-mag*mag)@np.array([28,28,64]))/120
    ordered=np.array([i for i in range(361) if i//19>=i%19],dtype=int)
    verify(len(ordered)==190,'ordered scale count')
    return dict(refs=refs,variants=variants,q=q,joint=joint,pair=pair,coefficient=coefficient,
                constant=constant,ordered=ordered,lg=lg,n=n,ind=ind)

def pair_features(M,membership):
    aa=.5*((membership@M)*membership).sum(1)-4
    complement=1-membership
    bb=.5*((complement@M)*complement).sum(1)-4
    total=.5*(M.sum()-16)
    return np.column_stack([aa,bb,total-aa-bb]),total

def choose(values,ids=None):
    values=np.asarray(values); minimum=float(values.min())
    index=int(np.flatnonzero(values <= minimum+CFG['tie_tolerance'])[0])
    return (index if ids is None else int(ids[index])),minimum,float(values[index]-minimum)

def global_pair(M,membership,masks,lib):
    features,total=pair_features(M,membership)
    ordered=lib['ordered']
    values=features@lib['coefficient'][ordered].T+lib['constant'][ordered]
    for oi,ti in enumerate(ordered):
        if ti//19==ti%19:
            values[:,oi]=lib['constant'][ti]+total*lib['coefficient'][ti,0]
    minimum=float(values.min()); flat=int(np.flatnonzero(values.ravel()<=minimum+CFG['tie_tolerance'])[0])
    mi,si=divmod(flat,len(ordered)); selected=float(values[mi,si])
    return int(masks[mi]),int(ordered[si]),minimum,selected-minimum,features[mi]

def smoke():
    labels=np.array(list(itertools.product([-1.,1.],repeat=4)))
    membership=np.array([[1.,1.,0.,0.],[1.,0.,1.,0.]])
    errors=[];r=np.array([.1,.15,.2]);coef=np.arctanh(r)/6
    const=math.log(4)-.5*np.log1p(-r*r)@np.array([1,1,4])/6
    for y in labels:
        M=np.outer(y,y)
        for membership_row in membership:
            aa=((membership_row@M)*membership_row).sum()/2-1
            b=1-membership_row;bb=((b@M)*b).sum()/2-1
            ab=(M.sum()-4)/2-aa-bb
            fast=const+np.array([aa,bb,ab])@coef
            direct=0.
            for i in range(4):
                for j in range(i+1,4):
                    kind=0 if membership_row[i] and membership_row[j] else (1 if not membership_row[i] and not membership_row[j] else 2)
                    direct-=math.log((1-r[kind]*y[i]*y[j])/4)/6
            errors.append(abs(fast-direct))
    verify(max(errors)<1e-12,'C pair affine smoke')
    write(OUT/'SMOKE.json',dict(passed=True,error=max(errors),enumerated_labels=16,new_scientific_samples=0))

def row_bound(lib):
    teacher=lib['variants'][12*19+4]
    a,b,_,_,_,_,_=core.geometry(16)
    W=[];conditional=[]
    for ya in [-1,1]:
        for yb in [-1,1]:
            W.append(-(ya-yb)*(2*(a+b)-16-ya-yb))
            conditional.append((a/8 if ya==1 else 1-a/8)*(b/8 if yb==1 else 1-b/8))
    W=np.array(W);conditional=np.array(conditional)
    probabilities=teacher[:,None,:]*conditional[None,:,:]
    verify(np.max(abs(probabilities.sum((1,2))-1))<1e-12,'ROW support distribution mass')
    mgf=np.array([np.sum(probabilities*np.exp(-lam*W)[None,:,:],axis=(1,2)) for lam in CFG['lambdas']])
    chosen=int(np.argmin(mgf[:,2])); primary=float(mgf[chosen,2]); conservative=float(mgf[chosen].max()+1e-10)
    bounds=[]
    for N in CFG['N']:
        bounds.append(dict(N=N,primary_union_failure_upper=math.exp(min(0.,math.log(64)+N*math.log(primary))),
            numerical_allowance_union_failure_upper=math.exp(min(0.,math.log(64)+N*math.log(conservative)))))
    write(OUT/'ROW_RECOVERY_BOUND.json',dict(profile='contrast',lambda_grid=CFG['lambdas'],mgf_by_lambda_and_variant=mgf,
        chosen_lambda=CFG['lambdas'][chosen],mean_W=(probabilities*W).sum((1,2)),
        second_moment_W=(probabilities*W*W).sum((1,2)),bounds=bounds,
        simultaneous_pairs=64,independence_between_pairs_not_assumed=True,
        conditional_on_fixed_numerical_teacher=True,allowance_not_rigorous_Gaussian_integration_certificate=True))

def risks(profile,lib):
    sA,sB=CFG['profiles'][profile];teacher_index=int(2*sA)*19+int(2*sB)
    variants=lib['variants'][teacher_index];p=variants[2]
    H=-float(p@(np.log(p)-lib['lg']))
    mapped=np.array([p@np.array(swap_kernel(r)['kernel']) for r in range(9)])
    risk=mapped@lib['joint'].T-H
    pn=p@lib['ind'];count=-np.log(lib['q']@lib['ind'])@pn+pn@np.log(pn)
    sensitivity=0.
    for vi in range(3):
        pv=variants[vi];hv=-float(pv@(np.log(pv)-lib['lg']))
        pm=np.array([pv@np.array(swap_kernel(r)['kernel']) for r in range(9)])
        for vj in range(3):
            other=pm@(-np.log(lib['variants'][:,vj])+lib['lg']).T-hv
            sensitivity=max(sensitivity,float(np.max(abs(other-risk))))
    verify(sensitivity<=CFG['reference_sensitivity_gate'],'C full-risk reference sensitivity')
    bound_variants=[float(core.functionals(pv,16)[1]) for pv in variants]
    verify(max(bound_variants)-min(bound_variants)<=CFG['reference_sensitivity_gate'],'C count-bound reference sensitivity')
    count_bound=0. if profile=='null' else float(core.functionals(p,16)[1])
    return dict(p=p,H=H,mapped=mapped,risk=risk,count=count,sensitivity=sensitivity,variants=variants,
                count_only_bound=count_bound,count_bound_variants=bound_variants)

def one_stream(profile,pi,stream,lib,geo,ri):
    patterns,popcount,masks,membership,spins=geo
    seed=CFG['seed']+1000*pi+stream;rng=np.random.Generator(np.random.PCG64(seed));initial=rng.bit_generator.state
    true_mask=int(masks[int(rng.integers(len(masks)))])
    codes=code_for_mask(patterns,popcount,true_mask)
    probability=ri['p'][codes]*np.exp(-lib['lg'][codes]);mass_error=abs(float(probability.sum())-1)
    verify(mass_error<=CFG['mass_gate'],'C teacher pattern mass');probability/=probability.sum()
    hist=[];current=np.zeros(65536,dtype=np.int64);last=0
    for N in CFG['N']:
        current=current+rng.multinomial(N-last,probability);hist.append(current.copy());last=N
    hist=np.array(hist);start=time.perf_counter()
    panel=np.load(B/'local'/f's{stream:02}.npz')['mean_pattern_loss']
    verify(panel.shape==(361,2,81),'B particle panel shape')
    fitrows=[];diagnostics=[];matrices=[]
    for ni,N in enumerate(CFG['N']):
        empirical=hist[ni]/N;M=(spins.T*empirical)@spins;matrices.append(M)
        verify(np.max(abs(np.diag(M)-1))<1e-12,'C spin diagonals')
        gmask,gti,gminimum,ggap,_=global_pair(M,membership,masks,lib)
        row_scores=-(M.sum(1)-1)/15
        high=np.lexsort((np.arange(16),-row_scores))[:8]
        row_mask=sum(1<<int(i) for i in high)
        count_cache={}
        def counts_for(mask):
            if mask not in count_cache:
                count_cache[mask]=np.bincount(code_for_mask(patterns,popcount,mask),weights=hist[ni],minlength=81)
                verify(np.array_equal(count_cache[mask],count_cache[mask].astype(np.int64)) and count_cache[mask].sum()==N,'C integer grouped counts')
            return count_cache[mask]
        candidates=[]
        for method in CFG['methods']:
            mask=true_mask if method.startswith('ORACLE') else (row_mask if method=='ROW_P' else gmask)
            group_counts=counts_for(mask);w=group_counts/N
            if method=='GLOBAL_P':
                ti=gti;minimum=gminimum;gap=ggap;value=float(lib['pair'][ti]@w)
                verify(abs(value-(minimum+gap))<=CFG['numerical_gate'],'C global pair direct replay')
            else:
                losses=(lib['joint'] if method=='ORACLE_J' else lib['pair'])@w
                if method.startswith('GLOBAL_G'):
                    losses=panel[:,0 if method=='GLOBAL_G10' else 1]@w
                ids=lib['ordered'] if method=='ROW_P' else np.arange(361)
                ti,minimum,gap=choose(losses[ids],ids);value=float(losses[ti])
            verify(gap<=CFG['tie_tolerance']+CFG['gap_roundoff_tolerance'],'C search gap')
            u,v=ti//19/2,ti%19/2
            overlap=int(popcount[np.uint16(mask&true_mask)]);r=8-overlap
            identifiable=profile=='contrast' and u!=v
            highmask=mask if u>v else 65535^mask
            swapped_high=8-int(popcount[np.uint16(highmask&true_mask)]) if identifiable else None
            fitrows.append(dict(profile=profile,stream=stream,N=N,method=method,true_mask=true_mask,
                selected_mask=mask,candidate_index=ti,u=u,v=v,training_objective=value,minimum_numerical_objective=minimum,
                grid_gap=gap,overlap_swap_r=r,deployed_KL=float(ri['risk'][r,ti]),
                assumed_group_best_grid_KL=float(ri['risk'][r].min()),
                excess_above_assumed_group_grid_best=float(ri['risk'][r,ti]-ri['risk'][r].min()),
                total_count_KL=float(ri['count'][ti]),group_recovery_identifiable=identifiable,
                high_group_swaps=swapped_high,high_group_exact_recovery=bool(swapped_high==0) if identifiable else None))
            entry=dict(method=method,mask=mask,training_group_counts=group_counts.astype(np.int64))
            if method.startswith('GLOBAL_G'):
                entry['training_objectives']=losses
            candidates.append(entry)
        diagnostics.append(dict(N=N,row_scores=row_scores,selected_group_counts=candidates))
    target=OUT/'local'/f'{profile}_s{stream:02}.npz'
    np.savez_compressed(target,pattern_histograms=hist,empirical_spin_matrices=np.array(matrices))
    receipt=dict(profile=profile,stream=stream,seed=seed,true_mask=true_mask,initial_RNG=initial,final_RNG=rng.bit_generator.state,
        numerical_pass=True,teacher_mass_error=mass_error,reference_sensitivity=ri['sensitivity'],
        full_pattern_observations=65536,finite_grid_fits=18,exhaustive_global_pair_fits=3,
        global_candidate_representations=2445300,diagnostics=diagnostics,fits=fitrows,
        empirical_spin_matrices=np.array(matrices),
        array_path=target,array_sha256=sha(target),elapsed_seconds=time.perf_counter()-start)
    write(OUT/'streams'/f'{profile}_s{stream:02}.json',receipt)
    print(json.dumps(dict(phase='C_stream',profile=profile,stream=stream,seconds=receipt['elapsed_seconds'],passed=True)),flush=True)

def collect():
    fits=[]
    for profile in CFG['profiles']:
        for stream in range(16): fits+=read(OUT/'streams'/f'{profile}_s{stream:02}.json')['fits']
    verify(len(fits)==576,'C fit count')
    index={(r['profile'],r['stream'],r['N'],r['method']):r for r in fits}
    comparisons=[('GLOBAL_P','ORACLE_J'),('GLOBAL_P','ORACLE_P'),('ROW_P','GLOBAL_P'),('GLOBAL_G10','GLOBAL_P'),('GLOBAL_G100','GLOBAL_P'),('GLOBAL_P','COUNT_ONLY_BOUND')]
    bound_info=read(OUT/'RISK_LIBRARY.json')
    crit=float(student_t.ppf(1-.05/(2*CFG['family']),15));stats=[];primary=[];recovery=[]
    for profile in CFG['profiles']:
        for N in CFG['N']:
            for left,right in comparisons:
                values=np.array([index[profile,s,N,left]['deployed_KL']-(bound_info[profile]['count_only_bound'] if right=='COUNT_ONLY_BOUND' else index[profile,s,N,right]['deployed_KL']) for s in range(16)])
                mean=float(values.mean());se=float(values.std(ddof=1)/4)
                row=dict(profile=profile,N=N,comparison=left+'-'+right,mean=mean,SE=se,lower=mean-crit*se,upper=mean+crit*se,precision_pass=se<=.002)
                stats.append(row)
                if N==65536 and left=='GLOBAL_G10':
                    primary.append(dict(kind='finite_particle_bias_after_learned_group',profile=profile,passed=bool(row['precision_pass'] and row['lower']>.001),**{k:row[k] for k in ['mean','lower','upper']}))
                if N==65536 and profile=='contrast' and right=='COUNT_ONLY_BOUND':
                    primary.append(dict(kind='labelled_pair_below_count_only_bound',profile=profile,passed=bool(row['precision_pass'] and row['upper']<0),**{k:row[k] for k in ['mean','lower','upper']}))
            for method in CFG['methods']:
                chosen=[index[profile,s,N,method] for s in range(16)]
                recovery.append(dict(profile=profile,N=N,method=method,mean_KL=float(np.mean([r['deployed_KL'] for r in chosen])),
                    mean_u=float(np.mean([r['u'] for r in chosen])),mean_v=float(np.mean([r['v'] for r in chosen])),
                    identifiable_groups=sum(r['group_recovery_identifiable'] for r in chosen),
                    exact_high_groups=sum(r['high_group_exact_recovery'] is True for r in chosen),
                    mean_swaps_if_identifiable=(float(np.mean([r['high_group_swaps'] for r in chosen if r['group_recovery_identifiable']])) if any(r['group_recovery_identifiable'] for r in chosen) else None)))
    csvwrite(OUT/'FITS.csv',fits);csvwrite(OUT/'RISK_INTERVALS.csv',stats);csvwrite(OUT/'SUMMARY.csv',recovery)
    verify(len(stats)==36 and len(primary)==3,'C fixed statistical family counts')
    write(OUT/'DECISION.json',dict(primary=primary,precision_pass=sum(r['precision_pass'] for r in stats),family=CFG['family'],
        finite_fits=576,global_exhaustive_fits=96,new_pattern_observations=2097152,new_prior_particles=0,
        existing_B_panels_reused=True,unknown_group_joint_likelihood_not_exhaustively_optimized=True,
        no_strict_equivalence_claim_from_degenerate_intervals=True))
    paths=sorted((OUT/'streams').glob('*.json'))+[OUT/n for n in ['FITS.csv','RISK_INTERVALS.csv','SUMMARY.csv','DECISION.json','SMOKE.json','ROW_RECOVERY_BOUND.json','RISK_LIBRARY.json']]
    write(OUT/'RESULTS_FREEZE.json',dict(time_utc=now(),files=hash_map(paths)))
    write(OUT/'STATE.json',dict(status='complete',time_utc=now(),active_jobs=[]),replace=True)
    print(json.dumps(read(OUT/'DECISION.json')),flush=True)

def run():
    checkfreeze();verify(read(OUT/'STATE.json')['status']=='frozen','No silent C restart')
    smoke();lib=load_library();geo=geometry();row_bound(lib)
    libraries={profile:risks(profile,lib) for profile in CFG['profiles']}
    write(OUT/'RISK_LIBRARY.json',{p:dict(H=r['H'],teacher_count_p=r['p'],mapped_teacher_p=r['mapped'],
        candidate_joint_KL_by_swap=r['risk'],candidate_total_count_KL=r['count'],reference_sensitivity=r['sensitivity'],
        count_only_bound=r['count_only_bound'],count_bound_variants=r['count_bound_variants']) for p,r in libraries.items()})
    for pi,profile in enumerate(CFG['profiles']):
        for stream in range(16):
            write(OUT/'STATE.json',dict(status='running',profile=profile,stream=stream,pid=os.getpid(),time_utc=now()),replace=True)
            one_stream(profile,pi,stream,lib,geo,libraries[profile])
    collect()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['freeze','run']);args=parser.parse_args()
    try: freeze() if args.phase=='freeze' else run()
    except Exception as ex:
        if (OUT/'FREEZE.json').exists() and not (OUT/'RESULTS_FREEZE.json').exists():
            write(OUT/f'FAILURE_{time.time_ns()}.json',dict(time_utc=now(),error=repr(ex),traceback=traceback.format_exc()))
            write(OUT/'STATE.json',dict(status='stopped_failure',time_utc=now(),active_jobs=[]),replace=True)
        raise
